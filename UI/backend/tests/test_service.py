import json
from pathlib import Path
import signal

import pytest
from fastapi.testclient import TestClient
from arion import service
from arion.main import create_app


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'DATA', tmp_path / 'state')
    monkeypatch.setattr(service, 'STATE', tmp_path / 'state' / 'server.json')
    monkeypatch.setattr(service.time, 'sleep', lambda _: None)
    return service.STATE


def save_owner(state):
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'pid': 123456, 'instance_id': 'owner'}))


def test_stop_does_not_signal_reused_pid(state, monkeypatch):
    save_owner(state)
    monkeypatch.setattr(service, 'health', lambda: {'instance_id': 'another-server'})
    calls = []
    monkeypatch.setattr(service.os, 'kill', lambda *args: calls.append(args))
    assert service.stop() == 1
    assert not calls and state.exists()


def test_stop_does_not_signal_without_api_identity(state, monkeypatch):
    save_owner(state)
    monkeypatch.setattr(service, 'health', lambda: None)
    calls = []
    monkeypatch.setattr(service.os, 'kill', lambda *args: calls.append(args))
    assert service.stop() == 0
    assert not calls and state.exists()


def test_stop_matching_instance_signals_only_its_pid(state, monkeypatch):
    save_owner(state)
    replies = iter([{'instance_id': 'owner'}, None])
    monkeypatch.setattr(service, 'health', lambda: next(replies))
    calls = []
    monkeypatch.setattr(service.os, 'kill', lambda *args: calls.append(args))
    assert service.stop() == 0
    assert calls == [(123456, signal.SIGTERM)] and not state.exists()


def test_start_does_not_replace_another_service(state, monkeypatch):
    monkeypatch.setattr(service, 'health', lambda: {'status': 'ok', 'application': 'another-service'})
    monkeypatch.setattr(service.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('다른 서버를 대체하면 안 됩니다.'))
    assert service.start() == 1
    assert not state.exists()


def test_start_preserves_existing_arion(state, monkeypatch):
    monkeypatch.setattr(service, 'health', lambda: {'status': 'ok', 'application': 'arion-dashboard'})
    monkeypatch.setattr(service.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('중복 서버를 시작하면 안 됩니다.'))
    assert service.start() == 0
    assert not state.exists()


def test_start_detaches_and_records_verified_instance(state, monkeypatch):
    index = service.ROOT / 'frontend' / 'dist' / 'index.html'
    index.parent.mkdir(parents=True)
    index.write_text('<html>시험 화면</html>')
    launched = {}

    class Process:
        pid = 123456
        def poll(self): return None
        def terminate(self): pytest.fail('검증된 서버를 종료하면 안 됩니다.')

    def spawn(command, **kwargs):
        launched.update(command=command, **kwargs)
        return Process()

    def health():
        if not launched: return None
        return {'status': 'ok', 'application': 'arion-dashboard', 'instance_id': launched['env']['ARION_SERVER_INSTANCE']}

    monkeypatch.setattr(service.subprocess, 'Popen', spawn)
    monkeypatch.setattr(service, 'health', health)
    assert service.start() == 0
    owner = json.loads(state.read_text())
    assert owner['pid'] == 123456 and owner['instance_id'] == launched['env']['ARION_SERVER_INSTANCE']
    assert launched['start_new_session'] is True
    assert launched['stdin'] == service.subprocess.DEVNULL
    assert launched['command'][launched['command'].index('--host') + 1] == '127.0.0.1'


def test_health_exposes_application_and_instance(monkeypatch, tmp_path):
    monkeypatch.setenv('ARION_SERVER_INSTANCE', 'test-owner')
    with TestClient(create_app(tmp_path)) as client:
        reply = client.get('/api/health').json()
        assert reply['status'] == 'ok'
        assert reply['application'] == 'arion-dashboard' and reply['instance_id'] == 'test-owner'


def test_index_requires_revalidation(monkeypatch, tmp_path):
    from arion import main
    index = tmp_path / 'frontend' / 'dist' / 'index.html'
    index.parent.mkdir(parents=True)
    index.write_text('<html>시험 화면</html>')
    monkeypatch.setattr(main, 'ROOT', tmp_path)
    with TestClient(create_app(tmp_path / 'data')) as client:
        reply = client.get('/')
        assert reply.status_code == 200 and reply.headers['cache-control'] == 'no-cache'
