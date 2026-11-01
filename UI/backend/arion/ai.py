import asyncio
import time
from urllib.parse import urlparse

import httpx

from .schemas import AIResponse, DetectionResult


def validate_events(run, events):
    result = []
    for event in events:
        event = DetectionResult.model_validate(event)
        if event.run_id != run.run_id or event.input_sha256 != run.sha256:
            raise ValueError("비행 ID 또는 입력 SHA-256이 다릅니다. 결과를 결합하지 않았습니다.")
        if event.epoch >= len(run.epoch_starts):
            raise ValueError("존재하지 않는 epoch입니다.")
        start = run.epoch_starts[event.epoch]
        stop = run.epoch_starts[event.epoch + 1] if event.epoch + 1 < len(run.epoch_starts) else run.duration_ms
        if event.start_ms < start or event.decided_at_ms > run.duration_ms or run.epoch_at(event.decided_at_ms) != event.epoch or event.end_ms > stop:
            raise ValueError("근거 구간 또는 판정 시각이 epoch/비행 경계를 벗어납니다.")
        if event.valid_until_ms > stop:
            raise ValueError("판정 유효기간이 epoch/비행 경계를 벗어납니다.")
        result.append(event.model_dump())
    return result


def fixture_events(run):
    """Deterministic UI fixtures, never derived diagnoses or fault labels."""
    if run.duration_ms < 5000:
        return []
    result = []
    for n, (fraction, target, title) in enumerate([
        (.30, "battery", "전원 계통 이상 의심 · 시험 이벤트"),
        (.60, "powertrain", "구동계 이상 의심 · 시험 이벤트"),
        (.82, "battery", "판정 복귀 · 시험 이벤트"),
    ]):
        decided = round(run.duration_ms * fraction, 3)
        epoch = run.epoch_at(decided)
        end_bound = run.epoch_starts[epoch + 1] - .001 if epoch + 1 < len(run.epoch_starts) else run.duration_ms
        result.append(DetectionResult(event_id=f"fixture-{n + 1}", run_id=run.run_id, input_sha256=run.sha256,
            epoch=epoch, start_ms=max(run.epoch_starts[epoch], decided - 1800), end_ms=decided,
            decided_at_ms=decided, valid_until_ms=min(end_bound, decided + 4000), model_version="ui-fixture-v1",
            status="normal" if n == 2 else "suspect", target=target, score=.12 if n == 2 else .82,
            threshold=.65, title=title, origin="fixture",
            description="화면 연결을 검증하기 위한 인공 응답입니다. 이 비행에 실제 고장이나 AI 탐지가 있었다는 의미가 아닙니다.",
            evidence=[]).model_dump())
    return validate_events(run, result)


class AIAdapter:
    def __init__(self, store):
        self.store = store
        self.states = {}
        self.tasks = {}
        self.received = {}  # Accepted result revision -> monotonic receipt time.
        self.api_keys = {}

    def state(self, run_id):
        return self.states.get(run_id, {"state": "saved" if self.store.events(run_id) else "disconnected", "url": "", "message": ""})

    async def disconnect(self, run_id):
        task = self.tasks.pop(run_id, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.states[run_id] = {"state": "saved" if self.store.events(run_id) else "disconnected", "url": "", "message": ""}

    async def connect(self, run_id, url):
        parsed = urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("첫 버전은 Mac의 로컬 HTTP API 주소만 지원합니다.")
        await self.disconnect(run_id)
        self.states[run_id] = {"state": "connecting", "url": url, "message": "AI API 연결 중"}
        self.api_keys[run_id] = set()
        self.tasks[run_id] = asyncio.create_task(self._poll(run_id, url))

    async def _poll(self, run_id, url):
        run = self.store.get(run_id)
        async with httpx.AsyncClient(timeout=3, follow_redirects=False, trust_env=False) as client:
            while True:
                try:
                    response = await client.get(url, params={"run_id": run_id, "input_sha256": run.sha256})
                    response.raise_for_status()
                    payload = AIResponse.model_validate(response.json())
                    if payload.run_id != run_id or payload.input_sha256 != run.sha256:
                        raise ValueError("응답의 비행 ID / 입력 해시 불일치")
                    events = validate_events(run, payload.results)
                    prior = {self.key(e): e for e in self.store.events(run_id)}
                    prior.update({self.key(e): e for e in self.store.events(run_id, "fixture")})
                    for event in events:
                        if prior.get(self.key(event)) != event:
                            self.received[self.key(event)] = time.perf_counter()
                    self.store.save_events(run_id, events)
                    self.api_keys[run_id].update(self.key(event) for event in events)
                    fixture_only = bool(events) and all(e["origin"] == "fixture" for e in events)
                    self.states[run_id] = {"state": payload.state, "url": url, "message": "",
                                           "fixture_only": fixture_only, "last_sync_utc_ms": time.time() * 1000}
                except (httpx.HTTPError, ValueError, TypeError) as error:
                    self.states[run_id] = {"state": "error", "url": url, "message": str(error)[:300]}
                await asyncio.sleep(.5)

    @staticmethod
    def key(event):
        return f'{event["run_id"]}:{event["origin"]}:{event["model_version"]}:{event["event_id"]}'

    async def close(self):
        for run_id in list(self.tasks):
            await self.disconnect(run_id)
