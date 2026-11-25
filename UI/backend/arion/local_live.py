"""Own one local CPU live worker; telemetry transport remains passive."""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .buffered_chart import chart
from .local_flight import WHOLE_MODELS, write
from .local_insights import insights
from .message_rates import input_policy
from .streaming import SessionRun


class LocalLive:
    """Own live AI workers, recovery state and serialized end-of-flight
    actions.
    """

    def __init__(self, service, store):
        self.service, self.store = service, store
        self.workers = {}
        self.finish_locks = {}
        self.root = service.root / "live"
        self.root.mkdir(exist_ok=True)
        self.delayed_bundle = os.getenv("ARION_NEW19_BUNDLE")
        self.delayed_calibration = os.getenv("ARION_NEW19_CALIBRATION")
        self.inflight_root = (
            str(service.inflight_root) if service.inflight_root else None
        )
        whole_models = (
            [
                model
                for model in service.v02_catalog()
                if model["task"] == "whole-run"
            ]
            if self.inflight_root
            else []
        )
        self.whole_run_root = (
            self.inflight_root
            if not service.whole_root and len(whole_models) == len(WHOLE_MODELS)
            else None
        )
        for path in self.root.glob("*/state.json"):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") not in ("finished", "error", "interrupted"):
                state.update(
                    status="interrupted",
                    prediction=None,
                    message="앱이 종료되었습니다. 저장된 기록을 확인하세요.",
                )
                write(path, state)

    def folder(self, run_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
            raise ValueError("Invalid session ID")
        if self.service.whole_root or self.inflight_root:
            if run_id not in self.store.catalog["sessions"]:
                raise KeyError(run_id)
        else:
            self.store.get(run_id)
        return self.root / run_id

    def state(self, run_id):
        path = self.folder(run_id) / "state.json"
        if not path.exists():
            run = self.store.get(run_id)
            if self.service.whole_root:
                return dict(
                    status="unconfigured",
                    mode="px4_whole_run",
                    message="PX4 UDP 세션에 전체 런 모델 4종을 연결하세요.",
                    whole_run_models=[model for model, _ in WHOLE_MODELS],
                )
            return dict(
                status="unconfigured",
                message="기체 프로파일과 로컬 모델을 연결하세요.",
                buffered_available=(
                    isinstance(run, SessionRun)
                    and run.config.kind == "udp"
                    and bool(
                        self.inflight_root
                        or (self.delayed_bundle and self.delayed_calibration)
                    )
                ),
                buffered_parallel=bool(
                    self.inflight_root and run.config.kind == "udp"
                ),
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            self.whole_run_root
            and state.get("mode") == "px4_buffered"
            and state["status"] == "finished"
        ):
            state["whole_available"] = (
                path.parent / "buffered-results.jsonl"
            ).is_file()
            summary = path.parent / "whole-summary.json"
            if "whole_summary" not in state and summary.is_file():
                state["whole_summary"] = json.loads(
                    summary.read_text(encoding="utf-8")
                )
        if (
            state.get("status") == "predicting"
            and time.time() - state.get("updated_at", 0) > 3
        ):
            return dict(
                state,
                status="stale",
                prediction=None,
                buffered_model_states=None,
                decision=dict(
                    status="unavailable", reason="CPU 작업 응답 지연"
                ),
                message="CPU 작업 응답 지연 · 이전 예측 숨김",
            )
        return state

    async def start(self, run_id, body):
        run = self.store.get(run_id)
        if (
            not isinstance(run, SessionRun)
            or not run.live
            or run.config.kind == "normalized"
        ):
            raise ValueError("활성 UDP/시리얼 MAVLink 세션에서 시작해 주세요.")
        if self.service.active or self.workers:
            raise HTTPException(409, "다른 로컬 AI 작업이 실행 중입니다.")
        folder = self.folder(run_id)
        if folder.exists():
            raise ValueError(
                "이 세션의 AI 설정은 고정됩니다. 새 수집 세션에서 다시 연결하세요."
            )
        profiles = await self.service.call(
            dict(action="profiles", data_root=str(self.service.root))
        )
        p = next(
            (
                p["profile"]
                for p in profiles["profiles"]
                if p["version"] == body.get("profile_version")
            ),
            None,
        )
        if p is None:
            raise ValueError("등록된 기체 프로파일 버전을 선택해 주세요.")
        request = {
            k: body.get(k)
            for k in (
                "forecast_model",
                "post_model",
                "source_system",
                "source_component",
                "scope",
                "imu_message",
                "imu_id",
                "use_active",
                "auto_rates",
            )
        }
        request["auto_finish_disarm"] = body.get("auto_finish_disarm", False)
        whole = bool(self.service.whole_root)
        parallel = bool(self.inflight_root and run.config.kind == "udp")
        if self.service.v02 and run.config.kind != "udp":
            raise ValueError(
                "V0.2 모델을 사용하려면 PX4 MAVLink UDP 세션을 연결하세요."
            )
        delayed = not whole and (parallel or body.get("mode") == "px4_buffered")
        if body.get("candidate_job"):
            if not (whole or parallel):
                raise ValueError("V0.2 UDP 세션에서 후보를 선택하세요.")
            choice = self.service.v02_candidate_selection(body["candidate_job"])
            if (whole or not self.whole_run_root) and not any(
                model.startswith("whole-run-" if whole else "inflight-")
                for model in choice["candidate_models"]
            ):
                raise ValueError("이 세션에서 실행할 모델의 후보를 선택하세요.")
            request.update(choice)
        if whole:
            if body.get("mode") != "px4_whole_run" or run.config.kind != "udp":
                raise ValueError(
                    "전체 런 모델 4종은 PX4 UDP 세션에서 시작하세요."
                )
            request.update(
                mode="px4_whole_run",
                auto_rates=False,
                auto_finish_disarm=False,
                whole_root=str(self.service.whole_root),
                wire=str(run.wire_path),
            )
        elif delayed:
            if run.config.kind != "udp" or not (
                self.inflight_root
                or (self.delayed_bundle and self.delayed_calibration)
            ):
                raise ValueError(
                    "PX4 10초 모드에는 UDP 세션과 서버의 학습·보정 번들 설정이 필요합니다."
                )
            if request["auto_finish_disarm"]:
                raise ValueError(
                    "PX4 10초 모드는 비행 종료 버튼으로 마감해 주세요."
                )
            request.update(
                mode="px4_buffered",
                auto_rates=False,
                wire=str(run.wire_path),
            )
            if parallel:
                request["inflight_root"] = self.inflight_root
                if self.whole_run_root:
                    request["whole_run_root"] = self.whole_run_root
            else:
                request.update(
                    new19_bundle=str(Path(self.delayed_bundle).resolve()),
                    new19_calibration=str(
                        Path(self.delayed_calibration).resolve()
                    ),
                )
        request["telemetry_policy"] = (
            input_policy(run, request)
            if request.get("auto_rates") is True
            else dict(
                schema="mavlink-acquisition/1",
                transport=run.config.kind,
                rates_hz={},
                automatic=False,
            )
        )
        request.update(
            profile=p,
            journal=str(run.path),
            output=str(folder),
            data_root=str(self.service.root),
        )
        admitted = await self.service.call(
            dict(
                request,
                action=(
                    "whole_live_validate"
                    if whole
                    else "buffered_live_validate"
                    if delayed
                    else "live_validate"
                ),
            )
        )
        if delayed or whole:
            request["binding"] = admitted
        # Recheck after subprocess awaits: only one worker/job may acquire the
        # CPU.
        if self.service.active or self.workers:
            raise HTTPException(409, "다른 로컬 AI 작업이 실행 중입니다.")
        folder.mkdir()
        write(folder / "request.json", request)
        initial = dict(status="starting", message="로컬 CPU 모델 준비 중")
        if delayed and parallel:
            initial.update(
                mode="px4_buffered", forecast_models=list(admitted["models"])
            )
        if request.get("candidate_job"):
            initial.update(
                candidate_job=request["candidate_job"],
                candidate_models=list(request["candidate_models"]),
            )
        write(folder / "state.json", initial)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            str(self.service.project / p)
            for p in ("ai", "ai/src", "Data/generation")
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.service.live_busy = True
        task = asyncio.create_task(self._worker(run_id, request, env))
        self.workers[run_id] = task
        if request.get("auto_rates") is True:
            try:
                self.store_sources.rates.start(run, request)
            except ValueError as error:
                write(folder / "rate-error.json", dict(error=str(error)))
        return self.state(run_id)

    async def _worker(self, run_id, request, env):
        folder = self.folder(run_id)
        process = None
        try:
            with (folder / "worker.log").open("wb") as log:
                module = {
                    "px4_buffered": "learning.new19.app_live",
                    "px4_whole_run": "learning.new19.app_whole",
                }.get(request.get("mode"), "localflight.live")
                process = await asyncio.create_subprocess_exec(
                    str(self.service.python),
                    "-B",
                    "-m",
                    module,
                    cwd=self.service.project,
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                )
                process.stdin.write(json.dumps(request).encode() + b"\n")
                await process.stdin.drain()
                process.stdin.close()
                while process.returncode is None:
                    if (folder / "auto-finish.json").exists() and not (
                        folder / "finish"
                    ).exists():
                        await self.finalize(run_id)
                    try:
                        await asyncio.wait_for(process.wait(), 0.25)
                    except asyncio.TimeoutError:
                        pass
                if (
                    process.returncode
                    and self.state(run_id).get("status") != "error"
                ):
                    write(
                        folder / "state.json",
                        dict(
                            status="error",
                            message="로컬 AI 작업 종료 오류. worker.log 확인",
                        ),
                    )
        except asyncio.CancelledError:
            if self.state(run_id).get("status") not in ("finished", "error"):
                write(
                    folder / "state.json",
                    dict(
                        status="interrupted",
                        prediction=None,
                        message="앱 종료 · 원본 기록 보존",
                    ),
                )
            raise
        except Exception as error:
            write(
                folder / "state.json",
                dict(status="error", prediction=None, message=str(error)),
            )
        finally:
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            self.workers.pop(run_id, None)
            self.service.live_busy = False

    def finish(self, run_id):
        if run_id in self.workers:
            (self.folder(run_id) / "finish").touch()
        return self.state(run_id)

    async def finalize(self, run_id):
        # Stop the writer before freezing/hash-checking the journal. Manual and
        # automatic finish share this lock, so concurrent requests cannot race.
        lock = self.finish_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            run = self.store.get(run_id)
            if run_id in self.store.active:
                await self.store_sources.disconnect(run)
                if self.service.whole_root and run.config.kind == "udp":
                    self.finish(run_id)
                run = await asyncio.to_thread(self.store.finish_session, run_id)
            self.finish(run_id)
            return run

    async def close(self):
        tasks = list(self.workers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.workers.clear()
        self.service.live_busy = False


def attach(app, service, store):
    live = LocalLive(service, store)
    app.state.local_live = live
    live.store_sources = app.state.sources

    @app.get("/api/sessions/{run_id}/local-ai")
    def state(run_id: str):
        result = live.state(run_id)
        if service.v02 or result.get("mode") in {
            "px4_whole_run",
            "px4_buffered",
        }:
            return JSONResponse(result)
        result["insights"] = insights(service.root, run_id, result)
        result["rates"] = app.state.sources.rates.status(run_id)
        saved = store.get(run_id).path.with_suffix(".rates.json")
        if result["rates"]["state"] == "not_requested" and saved.exists():
            result["rates"] = dict(
                json.loads(saved.read_text(encoding="utf-8")), recorded=True
            )
        error_file = live.folder(run_id) / "rate-error.json"
        if error_file.exists():
            result["rates"]["error"] = json.loads(
                error_file.read_text(encoding="utf-8")
            )["error"]
        return result

    @app.post("/api/sessions/{run_id}/local-ai")
    async def start(run_id: str, body: dict):
        try:
            return await live.start(run_id, body)
        except (ValueError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/sessions/{run_id}/local-ai/buffered-chart")
    def buffered_chart(
        run_id: str, model: str | None = None, comparison: str | None = None
    ):
        session = store.catalog["sessions"].get(run_id)
        folder = live.root / run_id
        if not session or not (folder / "request.json").is_file():
            raise HTTPException(404, "이 비행에 분석 그래프 기록이 없습니다.")
        workers = getattr(app.state, "buffered_chart_workers", {})
        try:
            if comparison:
                row = json.loads(
                    service.job_path(comparison).read_text(encoding="utf-8")
                )
                if (
                    row["action"] != "v02_candidate_replay"
                    or row["status"] != "complete"
                    or row["request"]["run_id"] != run_id
                    or row["result"]["buffered_summary"]["wire_sha256"]
                    != session.get("sha")
                ):
                    raise ValueError("이 비행의 완료된 후보 분석을 선택하세요.")
                folder = Path(row["request"]["output"])
            return JSONResponse(
                chart(
                    folder,
                    model,
                    workers.get(run_id) if not comparison else None,
                    wire_sha=session.get("sha"),
                )
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (OSError, KeyError, TypeError) as error:
            raise HTTPException(
                422, "저장된 그래프 자료를 읽지 못했습니다."
            ) from error

    @app.get("/api/sessions/{run_id}/local-ai/candidate-replays")
    def candidate_replays(run_id: str):
        try:
            folder = live.folder(run_id)
            saved = json.loads(
                (folder / "request.json").read_text(encoding="utf-8")
            )
        except (OSError, KeyError, ValueError) as error:
            raise HTTPException(
                404, "이 비행의 저장된 분석 설정이 없습니다."
            ) from error
        rows = []
        for path in (service.root / "jobs").glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            request = row["request"]
            if (
                row["action"] == "v02_candidate_replay"
                and request.get("run_id") == run_id
            ):
                rows.append(
                    dict(
                        id=row["id"],
                        candidate_job=request["candidate_job"],
                        status=row["status"],
                        error=row.get("error"),
                        result=row.get("result"),
                        created_at=row["created_at"],
                    )
                )
        return JSONResponse(
            dict(
                profile_id=saved["profile"]["id"],
                scope=saved["scope"],
                recorded_candidate=saved.get("candidate_job"),
                replays=sorted(
                    rows, key=lambda row: row["created_at"], reverse=True
                ),
            )
        )

    @app.post("/api/sessions/{run_id}/local-ai/candidate-replays")
    async def candidate_replay(run_id: str, request: dict):
        state = live.state(run_id)
        if (
            not live.inflight_root
            or state.get("mode") != "px4_buffered"
            or state["status"] != "finished"
        ):
            raise HTTPException(422, "종료된 PX4 UDP 비행 기록을 선택하세요.")
        session = store.catalog["sessions"].get(run_id, {})
        if not session.get("sha"):
            raise HTTPException(422, "비행의 확정 UDP 원문이 필요합니다.")
        try:
            folder = live.folder(run_id)
            saved = json.loads(
                (folder / "request.json").read_text(encoding="utf-8")
            )
            choice = service.v02_candidate_selection(
                request.get("candidate_job")
            )
            output = service.root / "v02-replays" / uuid4().hex
            return service.create_job(
                {
                    **saved,
                    **choice,
                    "action": "v02_candidate_replay",
                    "run_id": run_id,
                    "inflight_root": live.inflight_root,
                    "whole_run_root": live.whole_run_root,
                    "wire_sha256": session["sha"],
                    "output": str(output),
                }
            )
        except (KeyError, ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/sessions/{run_id}/local-ai/replay-buffered")
    async def replay_buffered(run_id: str):
        state = live.state(run_id)
        if state.get("mode") != "px4_buffered" or state["status"] != "finished":
            raise HTTPException(422, "종료된 PX4 10초 버퍼 세션을 선택하세요.")
        if service.active or service.live_busy or run_id in live.workers:
            raise HTTPException(409, "다른 로컬 AI 작업이 실행 중입니다.")
        service.live_busy = True
        try:
            return await service.call(
                dict(
                    action="buffered_live_replay",
                    output=str(live.folder(run_id)),
                )
            )
        except (ValueError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            service.live_busy = False

    @app.post("/api/sessions/{run_id}/local-ai/postflight")
    async def compute_postflight(run_id: str):
        state = live.state(run_id)
        if not state.get("whole_available"):
            raise HTTPException(422, "종료된 PX4 비행 기록을 선택하세요.")
        if state.get("whole_summary"):
            return state
        if service.active or service.live_busy:
            raise HTTPException(409, "다른 로컬 AI 작업이 실행 중입니다.")
        folder = live.folder(run_id)
        service.live_busy = True
        try:
            saved = json.loads(
                (folder / "request.json").read_text(encoding="utf-8")
            )
            await service.call(
                dict(
                    action="buffered_whole_score",
                    folder=str(folder),
                    whole_run_root=live.whole_run_root,
                    expected=saved["binding"].get("whole_models"),
                )
            )
            return live.state(run_id)
        except (ValueError, KeyError, OSError) as error:
            raise HTTPException(422, str(error)) from error
        finally:
            service.live_busy = False

    @app.post("/api/sessions/{run_id}/message-rates")
    async def configure_rates(run_id: str):
        try:
            request = json.loads(
                (live.folder(run_id) / "request.json").read_text(
                    encoding="utf-8"
                )
            )
            result = app.state.sources.rates.start(store.get(run_id), request)
            (live.folder(run_id) / "rate-error.json").unlink(missing_ok=True)
            return result
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error

    return live
