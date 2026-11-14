"""Own one local CPU live worker; telemetry transport remains passive."""

import asyncio
import json
import os
import re
import time

from fastapi import HTTPException

from .local_flight import write
from .local_insights import insights
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
        self.store.get(run_id)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", run_id):
            raise ValueError("Invalid session ID")
        return self.root / run_id

    def state(self, run_id):
        path = self.folder(run_id) / "state.json"
        if not path.exists():
            return dict(
                status="unconfigured",
                message="기체 프로파일과 로컬 모델을 연결하세요.",
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            state.get("status") == "predicting"
            and time.time() - state.get("updated_at", 0) > 3
        ):
            return dict(
                state,
                status="stale",
                prediction=None,
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
        request.update(
            profile=p,
            journal=str(run.path),
            output=str(folder),
            data_root=str(self.service.root),
        )
        await self.service.call(dict(request, action="live_validate"))
        # Recheck after subprocess awaits: only one worker/job may acquire the
        # CPU.
        if self.service.active or self.workers:
            raise HTTPException(409, "다른 로컬 AI 작업이 실행 중입니다.")
        folder.mkdir()
        write(folder / "request.json", request)
        write(
            folder / "state.json",
            dict(status="starting", message="로컬 CPU 모델 준비 중"),
        )
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
                process = await asyncio.create_subprocess_exec(
                    str(self.service.python),
                    "-B",
                    "-m",
                    "localflight.live",
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
