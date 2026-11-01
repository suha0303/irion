import asyncio
import hashlib
import io
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from arion.ai import AIAdapter, fixture_events, validate_events
from arion.main import create_app
from arion.storage import Store
from arion.telemetry import TelemetryRun


def record(name, ms, fields, **extra):
    return dict(receive_monotonic_ns=10_000_000_000 + int(ms * 1e6), source_system=36, source_component=1,
                message_name=name, fields=fields, quality={"profile": "arion_cmu_v1"}, **extra)


def battery(ms, **fields):
    return record("BATTERY_STATUS", ms, dict(id=0, voltages=[3815] * 4 + [65535] * 6,
        current_battery=2369, battery_remaining=94, current_consumed=39, **fields))


def write_run(tmp_path, records):
    path = tmp_path / "flight.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return TelemetryRun(path)


def long_records():
    return [battery(i * 200) for i in range(101)]


def test_conversions_and_real_zero(tmp_path):
    run = write_run(tmp_path, [battery(0), record("ESC_STATUS", 0, {"index": 0, "rpm": [0, 3, 4, 5], "current": [0, 1, 2, 3]})])
    f = run.frame(0)["values"]
    assert f["battery_voltage_v"]["value"] == pytest.approx(15.260)
    assert f["battery_current_a"]["value"] == 23.69
    assert f["battery_soc_pct"]["value"] == 94
    assert f["motor_1_rpm"]["value"] == 0
    assert f["motor_1_current_a"]["status"] == "valid"
    assert f["north_m"]["value"] is None


@pytest.mark.parametrize("field,value", [("current_battery", -1), ("current_battery", -100), ("current_battery", float('nan')), ("current_battery", "22"), ("battery_remaining", -1), ("voltages", [65535] * 10)])
def test_sentinels_and_bad_values_not_normal_zero(tmp_path, field, value):
    r = battery(0); r['fields'][field] = value
    f = write_run(tmp_path, [r]).frame(0)["values"]
    key = {'current_battery': 'battery_current_a', 'battery_remaining': 'battery_soc_pct', 'voltages': 'battery_voltage_v'}[field]
    assert f[key]["value"] is None and f[key]["status"] == "invalid"
    json.dumps(f, allow_nan=False)


def test_negative_current_and_producer_contract(tmp_path):
    r = battery(0); r['fields']['current_battery'] = -250; r['quality']['profile'] = 'real-other'
    f = write_run(tmp_path, [r]).frame(0)["values"]
    assert f['battery_current_a']['value'] == -2.5
    assert f['battery_voltage_v']['status'] == 'invalid'


def test_future_not_held_and_age_expires(tmp_path):
    r = battery(200)
    run = write_run(tmp_path, [record('HEARTBEAT', 0, {}), r, record('HEARTBEAT', 1000, {})])
    assert run.frame(199)['values']['battery_voltage_v']['status'] == 'missing'
    assert run.frame(800)['values']['battery_voltage_v']['status'] == 'valid'
    assert run.frame(801)['values']['battery_voltage_v']['status'] == 'stale'


def test_bad_row_quarantined_not_mutated(tmp_path):
    damaged = battery(100); damaged['receive_broken_ns'] = damaged.pop('receive_monotonic_ns')
    run = write_run(tmp_path, [battery(0), damaged, battery(200)])
    original = run.path.read_bytes()
    assert run.manifest()['accepted_records'] == 2
    assert run.issues[0]['line'] == 2
    assert 'receive_monotonic_ns' in run.issues[0]['reason']
    run.series()
    assert run.path.read_bytes() == original


def test_gap_creates_boundary_and_drops_old_values(tmp_path):
    run = write_run(tmp_path, [battery(0), record('HEARTBEAT', 3000, {})])
    assert run.frame(3000)['epoch'] == 1
    assert run.frame(3000)['values']['battery_voltage_v']['status'] == 'missing'
    assert any(p['values']['battery_voltage_v'] is None for p in run.series()['points'] if 700 <= p['t_ms'] < 3000)


def test_cross_message_source_order_is_not_reset(tmp_path):
    run = write_run(tmp_path, [record('ATTITUDE', 0, {'time_boot_ms': 300}), record('LOCAL_POSITION_NED', 100, {'time_boot_ms': 100}), record('ATTITUDE', 200, {'time_boot_ms': 400})])
    assert run.manifest()['epochs'] == 1


def test_reset_invalidates_previous_epoch(tmp_path):
    run = write_run(tmp_path, [battery(0), record('ATTITUDE', 100, {'time_boot_ms': 500}), record('ATTITUDE', 200, {'time_boot_ms': 10})])
    assert run.frame(200)['epoch'] == 1
    assert run.frame(200)['values']['battery_current_a']['value'] is None


def test_source_and_instance_filter(tmp_path):
    foreign = battery(100); foreign['source_system'] = 9
    other_battery = battery(200); other_battery['fields']['id'] = 1
    run = write_run(tmp_path, [battery(0), foreign, other_battery])
    assert len(run.samples) == 1 and len(run.issues) == 2


def test_ned_height_and_speed(tmp_path):
    run = write_run(tmp_path, [record('LOCAL_POSITION_NED', 0, {'x': 0, 'y': 0, 'z': -10, 'vx': 0, 'vy': 0, 'vz': 0}), record('LOCAL_POSITION_NED', 100, {'x': 3, 'y': 4, 'z': -13, 'vx': 3, 'vy': 4, 'vz': 0})])
    f = run.frame(100)['values']
    assert f['height_m']['value'] == 3
    assert f['east_m']['value'] == 4
    assert f['speed_m_s']['value'] == 5


@pytest.fixture
def service(tmp_path):
    app = create_app(tmp_path / 'data')
    content = '\n'.join(json.dumps(r) for r in long_records()).encode()
    with TestClient(app) as client:
        r = client.post('/api/runs/import', files={'file': ('test.jsonl', content)})
        assert r.status_code == 200
        yield client, app, r.json(), content


def test_import_idempotent_and_manifest(service):
    client, app, manifest, content = service
    r = client.post('/api/runs/import', files={'file': ('duplicate.jsonl', content)})
    assert r.json()['run_id'] == manifest['run_id']
    assert len(client.get('/api/runs').json()) == 1
    assert hashlib.sha256(content).hexdigest() == manifest['input_sha256']
    assert client.get('/api/runs/not-found').status_code == 404


def test_bad_input_is_reported(service):
    client, *_ = service
    assert client.post('/api/runs/import', files={'file': ('bad.jsonl', b'not json')}).status_code == 422


def test_demo_is_explicit_and_rewind_hides_future(service):
    client, app, m, _ = service
    run_id = m['run_id']
    initial = client.get(f'/api/runs/{run_id}/frame').json()
    assert initial['ai']['state'] == 'disconnected' and initial['active_result'] is None
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': True})
    later = client.get(f'/api/runs/{run_id}/frame?t_ms=12500').json()
    assert later['fixture_mode'] and len(later['events']) == 2
    assert later['active_result']['origin'] == 'fixture'
    assert client.get(f'/api/runs/{run_id}/frame?t_ms=0').json()['events'] == []
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': False})
    assert client.get(f'/api/runs/{run_id}/frame?t_ms=18000').json()['events'] == []


def test_mismatch_and_cross_epoch_results_rejected(service):
    _, app, m, _ = service
    run = app.state.store.get(m['run_id'])
    event = fixture_events(run)[0]
    event['input_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA-256'):
        validate_events(run, [event])
    event = fixture_events(run)[0]; event['epoch'] = 2
    with pytest.raises(ValueError, match='epoch'):
        validate_events(run, [event])
    event = fixture_events(run)[0]; event['end_ms'] = event['decided_at_ms'] + 1
    with pytest.raises(ValueError):
        validate_events(run, [event])


def test_npz_report_does_not_merge_mismatched_source(service):
    client, app, m, _ = service
    archive = io.BytesIO()
    metadata = {'features': [{'name': 'voltage', 'unit': 'V'}], 'input_files': [{'sha256': '0' * 64}], 'normalization': 'identity_not_model_ready'}
    np.savez(archive, metadata=json.dumps(metadata), X=np.ones((2, 1)), mask=np.array([[True], [False]]), tick_ns=np.array([0, 100]), epoch=np.array([0, 0]), window_end=np.array([1]), valid_end=np.array([1]), invalid_end=np.array([]), model_ready_end=np.array([]), age_ns=np.zeros((2, 1)))
    r = client.post(f'/api/runs/{m["run_id"]}/npz', files={'file': ('test.npz', archive.getvalue())})
    assert r.status_code == 200
    assert r.json()['provenance_matches'] is False
    assert r.json()['model_ready_windows'] == 0
    assert r.json()['missing_cells'] == 1


def test_websocket_pause_seek_speed_and_causal_events(service):
    client, _, m, _ = service
    run_id = m['run_id']
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': True})
    with client.websocket_connect(f'/api/ws/runs/{run_id}') as ws:
        assert ws.receive_json()['frame']['events'] == []
        ws.send_json({'action': 'seek', 't_ms': 12500})
        later = ws.receive_json()
        assert later['frame']['t_ms'] == 12500 and len(later['frame']['events']) == 2
        ws.send_json({'action': 'speed', 'speed': 4})
        ws.send_json({'action': 'play'})
        advanced = ws.receive_json()
        assert advanced['playback']['speed'] == 4 and advanced['frame']['t_ms'] > 12500
        ws.send_json({'action': 'pause'})
        paused = ws.receive_json()['frame']['t_ms']
        assert ws.receive_json()['frame']['t_ms'] == paused
        ws.send_json({'action': 'seek', 't_ms': 0})
        assert ws.receive_json()['frame']['events'] == []
        ws.send_json({'action': 'seek', 't_ms': -100})
        assert ws.receive_json()['frame']['t_ms'] == 0


def test_sqlite_results_survive_restart(service):
    _, app, m, _ = service
    store = app.state.store
    events = fixture_events(store.get(m['run_id']))
    store.save_events(m['run_id'], events)
    reloaded = Store(store.root)
    assert len(reloaded.events(m['run_id'], 'fixture')) == 3


def test_ai_http_adapter_validates_atomically(service, monkeypatch):
    _, app, m, _ = service
    run_id = m['run_id']; run = app.state.store.get(run_id)
    response = {'schema_version': 'arion-ai-results-v1', 'run_id': run_id, 'input_sha256': run.sha256, 'state': 'ready', 'results': fixture_events(run)}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=response))
    original = httpx.AsyncClient
    monkeypatch.setattr('arion.ai.httpx.AsyncClient', lambda **kw: original(transport=transport, **kw))

    async def check():
        adapter = AIAdapter(app.state.store)
        await adapter.connect(run_id, 'http://127.0.0.1:9999/results')
        await asyncio.sleep(.05)
        assert adapter.state(run_id)['state'] == 'ready'
        assert adapter.state(run_id)['fixture_only']
        assert len(app.state.store.events(run_id, 'fixture')) == 3
        response['results'][0]['input_sha256'] = '0' * 64
        await asyncio.sleep(.55)
        assert adapter.state(run_id)['state'] == 'error'
        assert all(e['input_sha256'] == run.sha256 for e in app.state.store.events(run_id, 'fixture'))
        await adapter.close()
    asyncio.run(check())


def test_ai_url_requires_local_service(service):
    client, _, m, _ = service
    response = client.post(f'/api/runs/{m["run_id"]}/ai/connect', json={'url': 'https://example.com/results'})
    assert response.status_code == 422


def test_api_fixture_does_not_mix_with_local_demo(service):
    client, app, m, _ = service
    run_id = m['run_id']; run = app.state.store.get(run_id)
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': True})
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': False})
    remote = fixture_events(run)
    for event in remote:
        event['event_id'] = 'api-' + event['event_id']
    app.state.store.save_events(run_id, remote)
    app.state.adapter.states[run_id] = {'state': 'ready', 'url': '', 'message': '', 'fixture_only': True}
    app.state.adapter.api_keys[run_id] = {app.state.adapter.key(e) for e in remote}
    f = client.get(f'/api/runs/{run_id}/frame?t_ms=20000').json()
    assert len(f['events']) == 3
    assert all(e['event_id'].startswith('api-') for e in f['events'])
    client.post(f'/api/runs/{run_id}/demo', json={'enabled': True})
    f = client.get(f'/api/runs/{run_id}/frame?t_ms=20000').json()
    assert len(f['events']) == 3
    assert all(e['event_id'].startswith('fixture-') for e in f['events'])


def test_corrupt_npz_is_actionable_error(service):
    client, _, m, _ = service
    r = client.post(f'/api/runs/{m["run_id"]}/npz', files={'file': ('bad.npz', b'not a zip archive')})
    assert r.status_code == 422 and 'NPZ 검사 실패' in r.json()['detail']
