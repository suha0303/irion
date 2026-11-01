import asyncio
from bisect import bisect_right
from collections import deque
from contextlib import asynccontextmanager
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .ai import AIAdapter, fixture_events
from .schemas import AIConnection, DemoRequest
from .store_v2 import StoreV2 as Store
from .streaming import LiveSources, SessionRun
from .providers import ProviderRegistry, contract_for
from .contracts import SourceConfig, InputProfile, Provider


ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD = 64 * 1024 * 1024


def create_app(data_dir=None):
    store = Store(Path(data_dir or os.environ.get("ARION_DATA_DIR", ROOT / ".data")))
    adapter = AIAdapter(store)
    providers = ProviderRegistry(store)
    sources = LiveSources(store)
    demo = set()
    metrics = deque(maxlen=1000)
    result_metrics = deque(maxlen=1000)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await providers.close()
        await sources.close()
        await adapter.close()

    app = FastAPI(title="ARION Dashboard", version="0.2.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.state.store, app.state.adapter = store, adapter
    app.state.providers, app.state.sources = providers, sources

    @app.exception_handler(KeyError)
    async def missing_run(request, error):
        return JSONResponse(status_code=404, content={"detail": "해당 비행을 찾을 수 없습니다."})

    def snapshot(run, cursor):
        frame = run.frame(cursor)
        state = adapter.state(run.run_id)
        is_fixture = run.run_id in demo or state.get("fixture_only", False)
        events = store.events(run.run_id, "fixture" if is_fixture else "model")
        if run.run_id in demo:
            demo_keys = {adapter.key(e) for e in fixture_events(run)}
            events = [e for e in events if adapter.key(e) in demo_keys]
        elif state.get("fixture_only"):
            events = [e for e in events if adapter.key(e) in adapter.api_keys.get(run.run_id, set())]
        visible = [e for e in events if e["decided_at_ms"] <= frame["t_ms"]]
        active = [e for e in visible if e["epoch"] == frame["epoch"] and e["valid_until_ms"] >= frame["t_ms"]]
        ai_state = dict(state, state="fixture") if run.run_id in demo else state
        frame.update({"events": visible, "ai": ai_state, "fixture_mode": is_fixture,
                      "active_result": active[-1] if active and ai_state["state"] not in ("error", "connecting", "warming_up", "unavailable") else None})
        frame.setdefault('sequence',bisect_right(getattr(run,'sample_times',[]),cursor))
        frame.setdefault('profile_id',run.profile.profile_id if isinstance(run,SessionRun) else 'legacy-mavlink')
        frame.setdefault('preprocess_version',contract_for(run)['preprocess_version'])
        frame.setdefault('sample_period_ms',contract_for(run)['sample_period_ms'])
        frame['source']={'system':run.source[0] if run.source else None,'component':run.source[1] if run.source else None}
        provider_states, new_events, active_by_provider = providers.snapshot(run, cursor)
        for event in frame['events']:
            event.setdefault('provider_id','legacy'); event.setdefault('kind','detector')
        frame['events'] = sorted(frame['events'] + new_events, key=lambda e:(e['decided_at_ms'],e.get('provider_id','legacy')))
        frame.update(schema_version='arion-telemetry-v2', source_kind='live' if getattr(run,'live',False) else 'replay',
                     providers=provider_states, active_by_provider=active_by_provider,
                     fixture_mode=frame['fixture_mode'] or run.run_id in providers.demo or any(s.get('fixture_only') for s in provider_states),
                     input_contract=contract_for(run),session=run.session_status() if isinstance(run,SessionRun) else None)
        return frame

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.2.0", "application": "arion-dashboard", "instance_id": os.environ.get("ARION_SERVER_INSTANCE", "foreground"), "scope": "local-replay-and-live", "live_available": True}

    @app.get("/api/runs")
    def runs():
        return store.list()

    @app.post("/api/runs/import")
    async def import_run(file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "로그는 64 MiB 이하여야 합니다.")
        try:
            run = await asyncio.to_thread(store.import_bytes, data, file.filename or "flight.jsonl")
            return run.manifest()
        except (ValueError, UnicodeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/runs/{run_id}")
    def manifest(run_id: str):
        return store.get(run_id).manifest()

    @app.get("/api/runs/{run_id}/series")
    def series(run_id: str):
        return store.get(run_id).series()

    @app.get("/api/runs/{run_id}/frame")
    def frame(run_id: str, t_ms: float = 0):
        if not math.isfinite(t_ms):
            raise HTTPException(422, "시각은 유한한 숫자여야 합니다.")
        return snapshot(store.get(run_id), t_ms)

    @app.get("/api/runs/{run_id}/diagnostics")
    def diagnostics(run_id: str):
        return {"manifest": store.get(run_id).manifest(), "npz": store.npz_report(run_id)}

    @app.post("/api/runs/{run_id}/npz")
    async def npz(run_id: str, file: UploadFile = File(...)):
        store.get(run_id)
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "NPZ는 64 MiB 이하여야 합니다.")
        try:
            return await asyncio.to_thread(store.inspect_npz, run_id, io.BytesIO(data))
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
            raise HTTPException(422, f"NPZ 검사 실패: {error}") from error

    @app.get("/api/runs/{run_id}/inspection.json")
    def report(run_id: str):
        return JSONResponse({"manifest": store.get(run_id).manifest(), "npz": store.npz_report(run_id)},
                            headers={"Content-Disposition": f'attachment; filename="{run_id}-inspection.json"'})

    @app.post("/api/runs/{run_id}/demo")
    def toggle_demo(run_id: str, body: DemoRequest):
        run = store.get(run_id)
        if body.enabled:
            store.save_events(run_id, fixture_events(run))
            demo.add(run_id)
        else:
            demo.discard(run_id)
        return {"enabled": body.enabled, "notice": "시험 이벤트는 실제 고장이나 AI 탐지가 아닙니다."}

    @app.post("/api/runs/{run_id}/ai/connect")
    async def connect_ai(run_id: str, body: AIConnection):
        store.get(run_id)
        try:
            await adapter.connect(run_id, body.url)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        demo.discard(run_id)
        return adapter.state(run_id)

    @app.post("/api/runs/{run_id}/ai/disconnect")
    async def disconnect_ai(run_id: str):
        store.get(run_id)
        await adapter.disconnect(run_id)
        return adapter.state(run_id)

    @app.get('/api/profiles')
    def profiles_list(): return store.profiles()

    @app.post('/api/profiles')
    def profile_register(body: InputProfile):
        try: return store.register_profile(body)
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/api/sessions')
    async def session_create(body: SourceConfig):
        try:
            run=store.create_session(body)
            try: await sources.connect(run)
            except ValueError: pass  # Keep error state and retry controls visible.
            return run.manifest()
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/api/sessions/{run_id}/connect')
    async def session_connect(run_id: str):
        try:
            run=store.get(run_id)
            if not isinstance(run,SessionRun): raise ValueError('실시간 수집 세션이 아닙니다.')
            await sources.connect(run)
            return run.manifest()
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/api/sessions/{run_id}/disconnect')
    async def session_disconnect(run_id: str):
        run=store.get(run_id)
        if not isinstance(run,SessionRun): raise HTTPException(422,'실시간 수집 세션이 아닙니다.')
        await sources.disconnect(run)
        return run.manifest()

    @app.post('/api/sessions/{run_id}/finish')
    async def session_finish(run_id: str):
        if run_id not in store.active: raise HTTPException(422,'활성 세션이 아닙니다.')
        run=store.get(run_id); await sources.disconnect(run)
        return store.finish_session(run_id).manifest()

    @app.post('/api/sessions/{run_id}/frames')
    async def session_frame(run_id: str, body: dict):
        run=store.get(run_id)
        if not isinstance(run,SessionRun): raise HTTPException(422,'전처리 스트림 세션이 아닙니다.')
        try:
            run.append_normalized(body)
            return {'accepted':True,'sequence':run.accepted}
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.get('/api/runs/{run_id}/input-segment')
    def input_segment(run_id: str, start_sequence: int, end_sequence: int):
        run=store.get(run_id)
        if not isinstance(run,SessionRun): raise HTTPException(422,'세션 입력 참조는 수집 기록에만 사용합니다.')
        try:
            ref,items=run.segment(start_sequence,end_sequence)
            return {'input_ref':ref,'frames':[x['frame'] for x in items]}
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.get('/api/analysis/providers')
    def provider_list(): return store.providers()

    @app.post('/api/analysis/providers')
    async def provider_register(body: Provider):
        try:
            for rid,pid in list(providers.tasks):
                if pid==body.provider_id: await providers.disconnect(rid,pid)
            return store.register_provider(body)
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/api/runs/{run_id}/analysis/{provider_id}/connect')
    async def provider_connect(run_id: str, provider_id: str, body: AIConnection):
        try:
            await providers.connect(run_id,provider_id,body.url)
            return {'connected':True}
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.post('/api/runs/{run_id}/analysis/{provider_id}/disconnect')
    async def provider_disconnect(run_id: str,provider_id: str):
        await providers.disconnect(run_id,provider_id);return {'connected':False}

    @app.post('/api/runs/{run_id}/analysis-demo')
    async def provider_demo(run_id: str,body: DemoRequest):
        try:
            if body.enabled:
                providers.fixtures(store.get(run_id)); providers.demo.add(run_id)
            else:
                providers.demo.discard(run_id)
                for state in providers.states_for(store.get(run_id)):
                    if state.get('fixture_only'):await providers.disconnect(run_id,state['provider_id'])
            demo.discard(run_id)
            return {'enabled':body.enabled}
        except ValueError as e: raise HTTPException(422,str(e)) from e

    @app.get("/api/metrics")
    def latency_metrics():
        def stats(values):
            ordered = sorted(values)
            return {"samples": len(ordered), "p95_ms": round(ordered[max(0, math.ceil(len(ordered) * .95) - 1)], 2) if ordered else None}
        return {"render_ack": stats(metrics), "result_receipt_to_render_ack": stats(result_metrics),
                "target_ms": 500,
                "method": "UI의 React commit 뒤 requestAnimationFrame 확인을 서버가 수신한 왕복 상한값. 결과 수신 지표는 수신 시 이미 재생 시각에 도달한 신규 API 결과만 포함. AI 추론 시간·500ms API 폴링 대기는 제외."}

    @app.websocket("/api/ws/runs/{run_id}")
    async def replay(ws: WebSocket, run_id: str):
        origin = ws.headers.get("origin")
        if origin and origin not in {"http://localhost:8765", "http://127.0.0.1:8765", "http://localhost:5173", "http://127.0.0.1:5173"}:
            await ws.close(code=1008)
            return
        try:
            run = store.get(run_id)
        except KeyError:
            await ws.close(code=1008)
            return
        await ws.accept()
        state = {"cursor": 0.0, "playing": False, "speed": 1.0, "last": time.perf_counter(), "revision": 0, "following": getattr(run,"live",False)}
        sent = {}
        acknowledged = set()

        async def receive():
            while True:
                command = await ws.receive_json()
                if not isinstance(command, dict):
                    continue
                action = command.get("action")
                now = time.perf_counter()
                if action == "ack":
                    item = sent.pop(command.get("sequence"), None)
                    if item:
                        metrics.append((now - item[0]) * 1000)
                        for key, receipt in item[1]:
                            if key not in acknowledged:
                                result_metrics.append((now - receipt) * 1000)
                                acknowledged.add(key)
                    continue
                if state["playing"]:
                    state["cursor"] = min(run.duration_ms, state["cursor"] + (now - state["last"]) * 1000 * state["speed"])
                state["last"] = now
                if getattr(run,'live',False):
                    if action in ('go_live','play'): state['following']=True
                    elif action=='pause': state['following']=False
                    elif action=='seek' and type(command.get('t_ms')) in (int,float) and math.isfinite(command['t_ms']):
                        state['following']=False
                        state['cursor']=max(run.times[0] if run.times else 0,min(command['t_ms'],run.now_ms))
                    state['playing']=False;state['revision']+=1
                    continue
                if action == "play":
                    if state["cursor"] >= run.duration_ms:
                        state["cursor"] = 0.0
                    state["playing"] = True
                elif action == "pause":
                    state["playing"] = False
                elif action == "seek" and type(command.get("t_ms")) in (float, int) and math.isfinite(command["t_ms"]):
                    state["cursor"] = max(0.0, min(float(command["t_ms"]), run.duration_ms))
                elif action == "speed" and command.get("speed") in (.25, .5, 1, 2, 4):
                    state["speed"] = float(command["speed"])
                state["revision"] += 1

        receiver = asyncio.create_task(receive())
        sequence = 0
        seen_receipts = set()
        eligible_receipts = {}
        try:
            while not receiver.done():
                started = time.perf_counter()
                run=store.get(run_id)
                if getattr(run,'live',False) and state['following']: state['cursor']=run.now_ms
                if state["playing"]:
                    state["cursor"] = min(run.duration_ms, state["cursor"] + (started - state["last"]) * 1000 * state["speed"])
                    if state["cursor"] >= run.duration_ms:
                        state["playing"] = False
                state["last"] = started
                value = snapshot(run, state["cursor"])
                for event in store.events(run_id) + store.events(run_id, "fixture"):
                    key = adapter.key(event)
                    receipt = adapter.received.get(key)
                    receipt_id = (key, receipt)
                    if receipt is not None and receipt_id not in seen_receipts:
                        seen_receipts.add(receipt_id)
                        if event["decided_at_ms"] <= state["cursor"]:
                            eligible_receipts[key] = receipt
                measurable = [(adapter.key(e), eligible_receipts[adapter.key(e)]) for e in value["events"] if adapter.key(e) in eligible_receipts]
                for event in value['events']:
                    if not event.get('provider_id') or event['provider_id']=='legacy':continue
                    key=providers.key(event);entry=providers.received.get(key)
                    if entry and (key,entry[0]) not in seen_receipts:
                        seen_receipts.add((key,entry[0]))
                        if entry[1]<=state['cursor']:eligible_receipts[key]=entry[0]
                    if key in eligible_receipts:measurable.append((key,eligible_receipts[key]))
                if getattr(run,'live',False) and sequence%10==0:
                    value['live_series']=run.series()['points'];value['manifest']=run.manifest()
                sent[sequence] = (time.perf_counter(), measurable)
                if len(sent) > 100:
                    sent.pop(next(iter(sent)))
                await ws.send_json({"type": "frame", "sequence": sequence, "frame": value,
                                    "playback": {"playing": state["playing"], "speed": state["speed"], "revision": state["revision"], "following":state["following"], "live":getattr(run,"live",False)}})
                sequence += 1
                await asyncio.sleep(max(.001, .1 - (time.perf_counter() - started)))
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            receiver.cancel()
            try:
                await receiver
            except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                pass

    dist = ROOT / "frontend" / "dist"
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index():
        if (dist / "index.html").exists():
            return FileResponse(dist / "index.html", headers={"Cache-Control":"no-cache"})
        return JSONResponse({"detail": "frontend에서 npm run build를 실행하세요."}, status_code=503)

    return app


app = create_app()
