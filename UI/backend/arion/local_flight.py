"""Local worker bridge. The UI interpreter never imports PyTorch."""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse

WHOLE_MODELS = [
    (f"whole-run-{kind}-{size}", f"{label} · {size.replace('x', ' × ')}")
    for kind, label in (("lstm", "LSTM"), ("transformer", "Transformer"))
    for size in ("64x2", "128x4")
]
FLIGHT_MODELS = [
    (model.replace("whole-run-", "inflight-"), label)
    for model, label in WHOLE_MODELS
]
V02_ROLES = ("adaptation", "calibration", "evaluation")
V02_JOB_ACTIONS = {
    "inspect",
    "timeline",
    "prepare",
    "admit_live",
    "v02_analyze",
    "v02_adapt",
}
V02_WORKER_ACTIONS = V02_JOB_ACTIONS | {
    "profiles",
    "save_profile",
    "whole_live_validate",
    "buffered_live_validate",
    "buffered_live_replay",
}
V02_API_SECTIONS = {
    "catalog",
    "profiles",
    "logs",
    "timeline",
    "prepare",
    "live-prepared",
    "jobs",
    "v02",
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


class LocalFlight:
    """Run local CPU analysis jobs in the isolated AI environment, one at a
    time.
    """

    def __init__(self, root=None):
        candidates = (
            [Path(os.environ["ARION_AI_ROOT"])]
            if os.environ.get("ARION_AI_ROOT")
            else list(Path(__file__).resolve().parents)
        )
        self.project = next(
            (p for p in candidates if (p / "ai/localflight/cli.py").is_file()),
            None,
        )
        self.root = (
            Path(root)
            if root
            else (
                self.project / "ai/.local/desktop"
                if self.project
                else Path.home() / ".local/share/arion/local-ai"
            )
        )
        self.python = (
            Path(os.environ["ARION_AI_PYTHON"])
            if os.environ.get("ARION_AI_PYTHON")
            else self.project / "ai/.venv/bin/python"
            if self.project
            else None
        )
        self.active = None
        self.gpu_python = (
            Path(os.environ["ARION_GPU_PYTHON"])
            if os.environ.get("ARION_GPU_PYTHON")
            else self.project / "ai/.local/gpu-runtime/bin/python"
            if self.project
            else None
        )
        self.live_busy = False
        whole_root = os.getenv("ARION_V02_WHOLE_ROOT") or os.getenv(
            "ARION_NEW19_WHOLE_ROOT"
        )
        self.whole_root = Path(whole_root).resolve() if whole_root else None
        inflight_root = os.getenv("ARION_V02_INFLIGHT_ROOT") or os.getenv(
            "ARION_NEW19_INFLIGHT_ROOT"
        )
        self.inflight_root = (
            Path(inflight_root).resolve() if inflight_root else None
        )
        self.v02 = bool(self.whole_root or self.inflight_root)
        self.process = None
        self.task = None
        self.root.mkdir(parents=True, exist_ok=True)
        for p in (self.root / "jobs").glob("*.json"):
            row = json.loads(p.read_text(encoding="utf-8"))
            if row["status"] in {"queued", "running"}:
                row.update(
                    status="interrupted",
                    error="프로그램이 종료되었습니다. 자동 재실행하지 않습니다.",
                )
                write(p, row)

    def available(self):
        return self.python is not None and self.python.is_file()

    async def call(self, request, *, job=False):
        if self.v02 and request["action"] not in V02_WORKER_ACTIONS:
            raise ValueError("V0.2에서 제공하는 기능이 아닙니다.")
        if not self.available():
            raise ValueError(
                "로컬 AI 환경을 찾지 못했습니다. ARION_AI_ROOT와 ai/.venv를 확인하세요."
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            str(self.project / p) for p in ("ai", "ai/src", "Data/generation")
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        python = self.python
        if request.get("device") == "cuda":
            if request["action"] not in {"adapt", "tune", "device_info"}:
                raise ValueError("GPU is available for local training only")
            if self.gpu_python is None or not self.gpu_python.is_file():
                raise ValueError(
                    "GPU 학습 환경이 없습니다. ARION_GPU_PYTHON을 설정하세요."
                )
            python = self.gpu_python
            # A bundled CPU PYTHONHOME would hide the GPU venv's packages.
            env.pop("PYTHONHOME", None)
        if request["action"] == "tune":
            # The server owns this path; clients cannot choose an executable.
            request = {
                **request,
                "cpu_python": str(self.python),
                "cpu_python_home": os.environ.get("PYTHONHOME"),
            }
        process = await asyncio.create_subprocess_exec(
            str(python),
            "-B",
            "-m",
            "localflight.cli",
            "worker",
            cwd=self.project,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if job:
            self.process = process
        try:
            out, err = await asyncio.wait_for(
                process.communicate(json.dumps(request).encode()),
                timeout=3610
                if job and request["action"] in {"adapt", "tune"}
                else 1800
                if job or request["action"] == "buffered_live_replay"
                else 30,
            )
            if process.returncode:
                raise ValueError(
                    err.decode(errors="replace")[-4000:]
                    or f"로컬 작업 실패 (종료 코드 {process.returncode})"
                )
            return json.loads(out)
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if job:
                self.process = None

    def job_path(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job ID")
        return self.root / "jobs" / (job_id + ".json")

    def create_job(self, request):
        if self.v02 and request["action"] not in V02_JOB_ACTIONS:
            raise HTTPException(404, "V0.2에서 제공하는 기능이 아닙니다.")
        if self.active or self.live_busy:
            raise HTTPException(
                409, "다른 로컬 작업이 실행 중입니다. 완료하거나 중지해 주세요."
            )
        job_id = uuid.uuid4().hex
        row = dict(
            id=job_id,
            action=request["action"],
            status="queued",
            created_at=time.time(),
            request=request,
        )
        write(self.job_path(job_id), row)
        self.active = job_id
        self.task = asyncio.create_task(self.run_job(row, request))
        return row

    async def run_job(self, row, request):
        path = self.job_path(row["id"])
        row["status"] = "running"
        write(path, row)
        try:
            row["result"] = await self.call(request, job=True)
            row["status"] = "complete"
            if row["action"] == "admit_live":
                row["action"] = "prepare"
        except asyncio.CancelledError:
            row["status"] = "cancelled"
        except Exception as error:
            row.update(status="failed", error=str(error))
        finally:
            row["finished_at"] = time.time()
            write(path, row)
            self.active = None

    async def close(self):
        if self.task and not self.task.done():
            job_id = self.active
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                # Cancellation before the coroutine's first instruction bypasses
                # its finally.
                if job_id:
                    path = self.job_path(job_id)
                    row = json.loads(path.read_text(encoding="utf-8"))
                    row.update(status="cancelled", finished_at=time.time())
                    write(path, row)
                    self.active = None

    def v02_catalog(self):
        if (
            not self.available()
            or not self.project
            or not (
                self.project / "ai/learning/new19/real_adaptation.py"
            ).is_file()
        ):
            return []
        models = []
        for family, task, root in (
            (FLIGHT_MODELS, "inflight", self.inflight_root),
            (WHOLE_MODELS, "whole-run", self.whole_root or self.inflight_root),
        ):
            if not root:
                continue
            for model, label in family:
                folder = root / model
                if all(
                    (folder / file).is_file()
                    for file in (
                        "fit/bundle.json",
                        "fit/model.safetensors",
                        "calibration/calibration.json",
                    )
                ):
                    models.append(
                        dict(id=model, task=task, label=label, root=str(root))
                    )
        return models


def attach(app, root=None):
    service = LocalFlight(root)
    app.state.local_flight = service

    @app.get("/api/local-ai/catalog")
    async def catalog():
        if not service.available():
            return dict(
                available=False, models=[], reason="로컬 AI 실행환경 미설치"
            )
        if service.v02:
            return dict(
                available=True,
                edition="v0.2",
                mode="px4_whole_run" if service.whole_root else "px4_buffered",
                models=[
                    {
                        key: value
                        for key, value in model.items()
                        if key != "root"
                    }
                    for model in service.v02_catalog()
                ],
                device="cpu",
            )
        try:
            return dict(
                available=True, **await service.call(dict(action="catalog"))
            )
        except (ValueError, asyncio.TimeoutError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/local-ai/profiles")
    async def profiles():
        try:
            return await service.call(
                dict(action="profiles", data_root=str(service.root))
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/local-ai/gpu")
    async def gpu():
        try:
            return await service.call(dict(action="device_info", device="cuda"))
        except (ValueError, OSError, asyncio.TimeoutError) as error:
            return dict(device="cuda", available=False, reason=str(error))

    @app.post("/api/local-ai/profiles")
    async def save_profile(profile: dict):
        try:
            return await service.call(
                dict(
                    action="save_profile",
                    data_root=str(service.root),
                    profile=profile,
                )
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/logs")
    async def import_log(file: UploadFile = File(...)):
        if service.active:
            raise HTTPException(409, "현재 로컬 작업 완료 후 가져오세요.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".bin", ".ulg"}:
            raise HTTPException(
                422, "PX4 .ulg 또는 ArduPilot .BIN을 선택하세요."
            )
        folder = service.root / "inputs" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        name = Path(file.filename or "flight").name
        path = folder / name
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > 256 * 1024 * 1024:
                        raise HTTPException(
                            413, "로그 한 개는 256 MiB 이하여야 합니다."
                        )
                    stream.write(chunk)
            return service.create_job(dict(action="inspect", path=str(path)))
        except Exception:
            path.unlink(missing_ok=True)
            folder.rmdir()
            raise
        finally:
            await file.close()

    @app.post("/api/local-ai/benchmark")
    async def benchmark():
        return service.create_job(dict(action="benchmark"))

    def completed(job_id, action):
        try:
            row = json.loads(
                service.job_path(job_id).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise HTTPException(422, "작업 기록을 찾을 수 없습니다.") from error
        if row["action"] != action or row["status"] != "complete":
            raise HTTPException(422, "필요한 선행 작업이 완료되지 않았습니다.")
        return row

    @app.post("/api/local-ai/logs/{job_id}/reinspect")
    async def reinspect(job_id: str):
        previous = completed(job_id, "inspect")
        return service.create_job(
            dict(action="inspect", path=previous["request"]["path"])
        )

    @app.post("/api/local-ai/timeline")
    async def timeline(request: dict):
        try:
            log = completed(request["log_job"], "inspect")
            profile_id, version = (
                request["profile_id"],
                request["profile_version"],
            )
            if not re.fullmatch(
                r"[a-zA-Z0-9_-]{1,64}", profile_id
            ) or not re.fullmatch(r"[a-f0-9]{64}", version):
                raise ValueError("Invalid profile reference")
            profile = json.loads(
                (
                    service.root / "aircraft" / profile_id / (version + ".json")
                ).read_text(encoding="utf-8")
            )
            encoded = json.dumps(
                profile, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode()
            if hashlib.sha256(encoded).hexdigest() != version:
                raise ValueError("Saved profile version changed")
            if "flight_interval" in request:
                intervals = (
                    log["result"].get("flight_events", {}).get("intervals", [])
                )
                index = request["flight_interval"]
                if (
                    type(index) is not int
                    or not 0 <= index < len(intervals)
                    or not intervals[index]["usable"]
                ):
                    raise ValueError(
                        "사용 가능한 자동 비행 구간을 선택하세요. 과거 로그는 다시 검사하세요."
                    )
                interval = intervals[index]
                settings = {k: interval[k] for k in ("start_us", "end_us")}
                settings["expected_source_sha256"] = log["result"]["sha256"]
                if interval["imu_instance"] is not None:
                    profile["imu_instance"] = interval["imu_instance"]
                    provenance = profile.setdefault("provenance", {})
                    provenance.setdefault("fields", {})["imu_instance"] = dict(
                        value=interval["imu_instance"],
                        status="logged_configuration",
                        source=interval["evidence"],
                        source_sha256=log["result"]["sha256"],
                    )
                settings["interval_evidence"] = interval["evidence"]
            else:
                settings = {k: request[k] for k in ("start_us", "end_us")}
            for key in (
                "clock_corrections",
                "fast_max_age_ms",
                "battery_max_age_ms",
            ):
                if key in request:
                    settings[key] = request[key]
            return service.create_job(
                dict(
                    action="timeline",
                    path=log["request"]["path"],
                    profile=profile,
                    settings=settings,
                )
            )
        except (KeyError, ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/prepare")
    async def prepare(request: dict):
        try:
            preview = completed(request["timeline_job"], "timeline")
            previous = preview["request"]
            settings = {
                **previous["settings"],
                **{
                    k: request[k]
                    for k in (
                        "interval_evidence",
                        "role",
                        "health",
                        "health_evidence",
                        "selected_segments",
                    )
                },
            }
            settings["expected_source_sha256"] = preview["result"][
                "source_sha256"
            ]
            return service.create_job(
                dict(
                    action="prepare",
                    path=previous["path"],
                    profile=previous["profile"],
                    settings=settings,
                    output=str(service.root / "prepared" / uuid.uuid4().hex),
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    def v02_selection(request):
        available = {row["id"]: row["root"] for row in service.v02_catalog()}
        models = request.get("models")
        if not service.v02:
            raise HTTPException(404, "V0.2 앱에서 실행하세요.")
        if (
            not isinstance(models, list)
            or not models
            or any(
                not isinstance(model, str) or model not in available
                for model in models
            )
            or len(set(models)) != len(models)
        ):
            raise HTTPException(422, "사용 가능한 V0.2 모델을 선택하세요.")
        return dict(
            models=models,
            model_roots={model: available[model] for model in models},
            output=str(service.root / "v02-jobs" / uuid.uuid4().hex),
        )

    @app.post("/api/local-ai/v02/analyze")
    async def v02_analyze(request: dict):
        options = v02_selection(request)
        job_id = request.get("prepared_job")
        if not isinstance(job_id, str):
            raise HTTPException(422, "준비된 로그를 선택하세요.")
        source = completed(job_id, "prepare")
        return service.create_job(
            dict(
                action="v02_analyze",
                manifest=source["result"]["manifest"],
                path=source["request"].get("path"),
                **options,
            )
        )

    @app.post("/api/local-ai/v02/adapt")
    async def v02_adapt(request: dict):
        options = v02_selection(request)
        logs = request.get("logs")
        if not isinstance(logs, dict) or set(logs) != set(V02_ROLES):
            raise HTTPException(422, "학습·보정·평가 자료를 각각 선택하세요.")
        ids, sources, days = set(), {}, {}
        selected = {}
        for role in V02_ROLES:
            if not isinstance(logs[role], list) or not logs[role]:
                raise HTTPException(
                    422, "학습·보정·평가 자료를 각각 선택하세요."
                )
            selected[role] = []
            for entry in logs[role]:
                if not isinstance(entry, dict) or set(entry) != {
                    "job_id",
                    "date",
                }:
                    raise HTTPException(
                        422, "준비 구간과 수집 날짜를 지정하세요."
                    )
                job_id, day = entry["job_id"], entry["date"]
                if not isinstance(job_id, str) or job_id in ids:
                    raise HTTPException(
                        422, "같은 준비 구간을 중복 선택했습니다."
                    )
                ids.add(job_id)
                try:
                    if date.fromisoformat(day).isoformat() != day:
                        raise ValueError
                except (TypeError, ValueError) as error:
                    raise HTTPException(
                        422, "수집 날짜를 YYYY-MM-DD 형식으로 입력하세요."
                    ) from error
                result = completed(job_id, "prepare")["result"]
                health = result.get("health")
                if result.get("role") != role or not (
                    health == "confirmed_normal"
                    or (role == "evaluation" and health == "abnormal")
                ):
                    raise HTTPException(
                        422, "구간에 저장된 역할과 판정 근거를 확인하세요."
                    )
                source = result.get("source_sha256")
                if not source or sources.setdefault(source, (role, day)) != (
                    role,
                    day,
                ):
                    raise HTTPException(
                        422, "같은 원본에는 하나의 날짜와 역할만 지정하세요."
                    )
                if days.setdefault(day, role) != role:
                    raise HTTPException(
                        422, "같은 수집 날짜의 자료는 한 역할에 묶으세요."
                    )
                selected[role].append(
                    dict(manifest=result["manifest"], date=day)
                )
        return service.create_job(
            dict(action="v02_adapt", logs=selected, **options)
        )

    @app.get("/api/local-ai/v02/jobs/{job_id}/graphs/{model_id}/{index}")
    async def v02_graph(job_id: str, model_id: str, index: int):
        if (
            not service.v02
            or model_id not in dict(FLIGHT_MODELS + WHOLE_MODELS)
            or index < 0
        ):
            raise HTTPException(404, "그래프를 찾을 수 없습니다.")
        try:
            row = json.loads(
                service.job_path(job_id).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise HTTPException(404, "그래프를 찾을 수 없습니다.") from error
        if row["status"] != "complete" or row["action"] not in {
            "v02_analyze",
            "v02_adapt",
        }:
            raise HTTPException(404, "그래프를 찾을 수 없습니다.")
        key = "analyses" if row["action"] == "v02_analyze" else "candidates"
        if not any(
            model["model_id"] == model_id and index < len(model["evaluation"])
            for model in row["result"]["v02"][key]
        ):
            raise HTTPException(404, "그래프를 찾을 수 없습니다.")
        path = Path(row["request"]["output"]) / model_id / f"{index}.json"
        if not path.is_file():
            raise HTTPException(404, "그래프 파일을 찾을 수 없습니다.")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/local-ai/calibrations")
    async def calibrations():
        try:
            return await service.call(
                dict(action="calibration_list", data_root=str(service.root))
            )
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/calibrations")
    async def fit_calibration(request: dict):
        try:
            ids = request["prepared_jobs"]
            if not isinstance(ids, list) or not 2 <= len(ids) <= 20:
                raise ValueError("보정 전용 정상 비행 2~20개를 선택하세요.")
            manifests = [
                completed(i, "prepare")["result"]["manifest"] for i in ids
            ]
            references = request.get("reference_jobs", [])
            if not isinstance(references, list) or len(references) > 20:
                raise ValueError("정상 기준용 비행은 최대 20개입니다.")
            return service.create_job(
                dict(
                    action="calibration_fit",
                    data_root=str(service.root),
                    model_id=request["model_id"],
                    manifests=manifests,
                    normal_references=[
                        completed(i, "prepare")["result"]["manifest"]
                        for i in references
                    ],
                    quantile=request.get("quantile", 0.99),
                    use_active=request.get("use_active") is True,
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/calibrations/select")
    async def select_calibration(request: dict):
        return service.create_job(
            dict(
                action="calibration_select",
                data_root=str(service.root),
                id=request.get("id", ""),
                enabled=request.get("enabled") is True,
            )
        )

    @app.post("/api/local-ai/live-prepared")
    async def admit_live(request: dict):
        try:
            folder = app.state.local_live.folder(request["run_id"])
            result = json.loads(
                (folder / "result.json").read_text(encoding="utf-8")
            )
            if result["status"] != "complete":
                raise ValueError("보존 구간이 있는 종료된 비행이 필요합니다.")
            index = request["index"]
            if type(index) is not int or not 0 <= index < len(
                result["analyses"]
            ):
                raise ValueError("종료 후 구간을 선택하세요.")
            manifest = result["analyses"][index]["manifest"]
            # Paths are from the locally completed job, never supplied by the
            # client.
            if not Path(manifest).is_relative_to(folder / "completed"):
                raise ValueError("Live package path mismatch")
            job = service.create_job(
                dict(
                    action="admit_live",
                    manifest=manifest,
                    output=str(service.root / "prepared" / uuid.uuid4().hex),
                    role=request["role"],
                    health=request["health"],
                    evidence=request["evidence"],
                )
            )
            return job
        except (KeyError, ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/analyze")
    async def analyze(request: dict):
        try:
            ids = (
                request["prepared_jobs"]
                if "prepared_jobs" in request
                else [request["prepared_job"]]
            )
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 64
                or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("서로 다른 준비 구간 1~64개를 선택하세요.")
            prepared = [completed(i, "prepare") for i in ids]
            models = request["models"]
            if (
                not isinstance(models, list)
                or not 1 <= len(models) <= 2
                or any(not isinstance(m, str) for m in models)
            ):
                raise ValueError("Select one or two models")
            return service.create_job(
                dict(
                    action="analyze",
                    manifests=[p["result"]["manifest"] for p in prepared],
                    models=models,
                    use_active=request.get("use_active") is True,
                    data_root=str(service.root),
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/adapt")
    async def adapt(request: dict):
        try:
            train_ids, dev_ids = (
                request["adaptation_jobs"],
                request["development_jobs"],
            )
            if (
                not isinstance(train_ids, list)
                or not isinstance(dev_ids, list)
                or not 1 <= len(train_ids) <= 3
                or len(dev_ids) > 20
            ):
                raise ValueError(
                    (
                        "Use 1..3 adaptation flights and at most 20 "
                        "development flight"
                    )
                )
            train = [
                completed(i, "prepare")["result"]["manifest"] for i in train_ids
            ]
            dev = [
                completed(i, "prepare")["result"]["manifest"] for i in dev_ids
            ]
            return service.create_job(
                dict(
                    action="adapt",
                    model_id=request["model_id"],
                    adaptation=train,
                    development=dev,
                    epochs=request.get("epochs", 5),
                    device=request.get("device", "cpu"),
                    strategy=request.get("strategy", "head"),
                    time_limit_seconds=request.get("time_limit_seconds", 1800),
                    output=str(service.root / "adapted" / uuid.uuid4().hex),
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/normal-plan")
    async def normal_plan(request: dict):
        try:
            ids = request["prepared_jobs"]
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 50
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("정상 확인 비행 1~50개를 선택하세요.")
            return service.create_job(
                dict(
                    action="normal_plan",
                    manifests=[
                        completed(i, "prepare")["result"]["manifest"]
                        for i in ids
                    ],
                    output=str(
                        service.root / "normal-plans" / uuid.uuid4().hex
                    ),
                )
            )
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/tune")
    async def tune_pair(request: dict):
        try:
            manifests = {}
            for key in (
                "adaptation",
                "development",
                "calibrations",
                "evaluations",
            ):
                ids = request.get(key + "_jobs", [])
                if (
                    not isinstance(ids, list)
                    or len(ids) > 20
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError("서로 다른 준비 비행을 선택하세요.")
                manifests[key] = [
                    completed(i, "prepare")["result"]["manifest"] for i in ids
                ]
            if request.get("normal_plan_job"):
                plan = completed(request["normal_plan_job"], "normal_plan")[
                    "result"
                ]
                for key, role in [
                    ("adaptation", "adaptation"),
                    ("development", "development"),
                    ("calibrations", "calibration"),
                    ("evaluations", "evaluation"),
                ]:
                    selected = [
                        r["manifest"]
                        for r in plan["flights"]
                        if r["role"] == role
                    ]
                    manifests[key] = selected + (
                        manifests[key] if key == "evaluations" else []
                    )
            return service.create_job(
                dict(
                    action="tune",
                    models=request["models"],
                    **manifests,
                    device=request.get("device", "cuda"),
                    epochs=request.get("epochs", 30),
                    budget_seconds=request.get("budget_seconds", 3600),
                    events=request.get("events", {}),
                    data_root=str(service.root),
                    output=str(service.root / "tuning" / uuid.uuid4().hex),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/tune/{job_id}/select/{model_index}")
    async def select_tuned(job_id: str, model_index: int):
        try:
            pair = completed(job_id, "tune")["result"]
            if not 0 <= model_index < len(pair["tuned_models"]):
                raise ValueError("모델 결과를 선택하세요.")
            row = pair["tuned_models"][model_index]
            if (
                row.get("review_status") != "passed"
                or not pair["within_budget"]
            ):
                raise ValueError(
                    "시간 예산과 정상·이상 퇴행 검증을 통과해야 선택할 수 있습니다."
                )
            evidence = (
                dict(
                    comparison_file=row["review_file"],
                    comparison_sha256=row["review_sha256"],
                )
                if row.get("review_file")
                else dict(comparison=row["review"])
            )
            return service.create_job(
                dict(
                    action="tune_select",
                    data_root=str(service.root),
                    model_id=row["registered"]["id"],
                    **evidence,
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/compare")
    async def compare(request: dict):
        try:
            fitted = completed(request["adapted_job"], "adapt")
            prepared = completed(request["prepared_job"], "prepare")
            return service.create_job(
                dict(
                    action="compare",
                    adapted=fitted["result"]["output"],
                    manifest=prepared["result"]["manifest"],
                )
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/jobs/{job_id}/resume")
    async def resume(job_id: str):
        try:
            previous = json.loads(
                service.job_path(job_id).read_text(encoding="utf-8")
            )
            if previous["action"] not in {"adapt", "tune"} or previous[
                "status"
            ] not in {
                "cancelled",
                "interrupted",
                "failed",
            }:
                raise ValueError("Only interrupted adaptation jobs may resume")
            request = previous["request"]
            folder = Path(request["output"])
            checkpoint = (
                (folder / "checkpoint.json").is_file()
                if previous["action"] == "adapt"
                else any(folder.glob("*/checkpoint.json"))
            )
            if not checkpoint:
                raise ValueError("No completed epoch checkpoint to resume")
            return service.create_job(
                {
                    **request,
                    "resume": request["output"],
                    "output": str(
                        service.root
                        / (
                            "tuning"
                            if previous["action"] == "tune"
                            else "adapted"
                        )
                        / uuid.uuid4().hex
                    ),
                }
            )
        except (OSError, KeyError, ValueError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/local-ai/models")
    async def models():
        try:
            return await service.call(
                dict(action="model_list", data_root=str(service.root))
            )
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/local-ai/models/register")
    async def register_model(request: dict):
        fitted = completed(request.get("adapted_job", ""), "adapt")
        return service.create_job(
            dict(
                action="model_register",
                data_root=str(service.root),
                source=fitted["result"]["output"],
            )
        )

    @app.post("/api/local-ai/models/import")
    async def import_model(file: UploadFile = File(...)):
        if service.active:
            raise HTTPException(409, "현재 작업 완료 후 가져오세요.")
        folder = service.root / "model-imports"
        folder.mkdir(exist_ok=True)
        path = folder / (uuid.uuid4().hex + ".zip")
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise HTTPException(
                            413, "모델 파일은 64 MiB 이하여야 합니다."
                        )
                    stream.write(chunk)
            return service.create_job(
                dict(
                    action="model_import",
                    data_root=str(service.root),
                    source=str(path),
                )
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    def registered_path(model_id):
        if not re.fullmatch(r"[a-f0-9]{64}", model_id):
            raise HTTPException(422, "잘못된 모델 ID")
        return service.root / "models" / model_id

    @app.post("/api/local-ai/models/{model_id}/export")
    async def export_model(model_id: str):
        registered_path(model_id)
        folder = service.root / "model-exports"
        folder.mkdir(exist_ok=True)
        return service.create_job(
            dict(
                action="model_export",
                data_root=str(service.root),
                model_id=model_id,
                output=str(folder / (uuid.uuid4().hex + ".zip")),
            )
        )

    @app.get("/api/local-ai/model-exports/{job_id}")
    async def download_model(job_id: str):
        job = completed(job_id, "model_export")
        path = Path(job["result"]["path"])
        if path.parent != service.root / "model-exports":
            raise HTTPException(422, "잘못된 내보내기 경로")
        return FileResponse(
            path,
            filename="drone-model-" + job["result"]["model_id"][:12] + ".zip",
            media_type="application/zip",
        )

    @app.post("/api/local-ai/models/{model_id}/compare")
    async def compare_registered(model_id: str, request: dict):
        prepared = completed(request.get("prepared_job", ""), "prepare")
        return service.create_job(
            dict(
                action="compare",
                adapted=str(registered_path(model_id)),
                manifest=prepared["result"]["manifest"],
            )
        )

    @app.post("/api/local-ai/models/{model_id}/select")
    async def select_model(model_id: str, request: dict):
        registered_path(model_id)
        comparison = completed(request.get("comparison_job", ""), "compare")
        return service.create_job(
            dict(
                action="model_select",
                data_root=str(service.root),
                model_id=model_id,
                comparison_file=str(service.job_path(comparison["id"])),
                comparison_sha256=hashlib.sha256(
                    service.job_path(comparison["id"]).read_bytes()
                ).hexdigest(),
                comparison_is_job=True,
            )
        )

    @app.post("/api/local-ai/models/rollback")
    async def rollback_model(request: dict):
        return service.create_job(
            dict(
                action="model_rollback",
                data_root=str(service.root),
                event_id=request.get("event_id", ""),
            )
        )

    @app.get("/api/local-ai/jobs")
    async def jobs():
        rows = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (service.root / "jobs").glob("*.json")
        ]
        rows = [
            row
            for row in rows
            if (
                row["action"] in V02_JOB_ACTIONS
                if service.v02
                else row["action"] not in {"v02_analyze", "v02_adapt"}
            )
        ]
        for row in rows:
            if row["status"] == "running" and row["action"] in {
                "adapt",
                "tune",
                "v02_analyze",
                "v02_adapt",
            }:
                p = Path(row["request"]["output"]) / "progress.json"
                if p.is_file():
                    row["progress"] = json.loads(p.read_text(encoding="utf-8"))
        return dict(
            jobs=sorted(rows, key=lambda r: r["created_at"], reverse=True)
        )

    @app.post("/api/local-ai/jobs/{job_id}/cancel")
    async def cancel(job_id: str):
        if job_id != service.active:
            raise HTTPException(409, "실행 중인 작업이 아닙니다.")
        await service.close()
        return json.loads(service.job_path(job_id).read_text(encoding="utf-8"))

    # The two editions may share raw input/prepare contracts. Their model routes
    # and job results are exposed only by the edition that owns them.
    prefix = "/api/local-ai/"
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not route.path.startswith(prefix)
        or (
            route.path.removeprefix(prefix).split("/", 1)[0] in V02_API_SECTIONS
            if service.v02
            else not route.path.startswith(prefix + "v02/")
        )
    ]
    return service
