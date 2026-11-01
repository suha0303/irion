import asyncio
from copy import deepcopy
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pymavlink.dialects.v20 import common
from arion.contracts import InputProfile, SourceConfig, Provider, AnalysisResult, InputRequirement
from arion.main import create_app
from arion.providers import ProviderRegistry, validate_result, compatibility
from arion.store_v2 import StoreV2
from arion.streaming import SessionRun, LiveSources, encode
from arion.storage import Store
from arion.ai import fixture_events
from test_pipeline import battery, record, long_records

@pytest.fixture
def store(tmp_path): return StoreV2(tmp_path)

def normalized(run,sequence=1,ms=0,epoch=0,**extra):
    return dict(schema_version='arion-normalized-v2',session_id=run.run_id,sequence=sequence,epoch=epoch,
                source_time_ns=10_000_000_000+int(ms*1e6),preprocess_version='normalized-v1',sample_period_ms=100,
                values={'battery_current_a':{'value':0,'unit':'A','valid':True,'age_ms':0}},**extra)

def make_normalized(store): return store.create_session(SourceConfig(kind='normalized',profile_id='normalized_flight_v1'))

def test_normalized_zero_age_mask_and_missing(store):
    run=make_normalized(store);p=normalized(run);p['values']['battery_current_a']['age_ms']=500
    run.append_normalized(p)
    assert run.frame(0)['values']['battery_current_a']['value']==0
    assert run.frame(101)['values']['battery_current_a']['status']=='stale'
    assert run.frame(0)['values']['east_m']['status']=='missing'
    p=normalized(run,2,100);p['values']['battery_current_a']['valid']=False;run.append_normalized(p)
    assert run.frame(100)['values']['battery_current_a']['status']=='invalid'
    run.finish()

@pytest.mark.parametrize('change',[{'sample_period_ms':50},{'preprocess_version':'bad'},{'schema_version':'unknown'}, {'session_id':'other'}])
def test_normalized_contract_changes_are_reported(store,change):
    run=make_normalized(store);p=normalized(run);p.update(change)
    with pytest.raises(ValueError):run.append_normalized(p)
    assert run.excluded==1 and run.accepted==0
    run.finish()

def test_ordering_gap_epoch_and_replay_journal(store):
    run=make_normalized(store)
    run.append_normalized(normalized(run,1,0))
    run.append_normalized(normalized(run,1,0))
    assert run.accepted==1 and run.last_ingest['duplicate']
    run.append_normalized(normalized(run,3,200));assert run.dropped_sequences==1
    run.append_normalized(normalized(run,4,2500));assert run.frame(2500)['epoch']==1
    assert run.frame(2499)['values']['battery_current_a']['status']=='stale'
    before=run.frame(2500);id=run.run_id
    replay=store.finish_session(id)
    assert replay.frame(2500)==before and replay.sha256
    assert any(p['values']['battery_current_a'] is None for p in replay.series()['points'])
    assert len(StoreV2(store.root).list())==1


def test_units_and_large_ns_are_preserved(store):
    run=make_normalized(store);p=normalized(run);p['source_time_ns']=1_000_000_000_000_000_123
    p['values']['battery_current_a']['unit']='mA'
    with pytest.raises(ValueError):run.append_normalized(p)
    p['values']['battery_current_a']['unit']='A';run.append_normalized(p)
    run.finish()
    rows=[json.loads(x) for x in run.path.read_text().splitlines()]
    assert rows[-1]['raw']['source_time_ns']==1_000_000_000_000_000_123


def test_raw_shared_conversion_profiles_and_filters(store):
    run=store.create_session(SourceConfig(kind='udp',profile_id='arion_cmu_v1'))
    assert run.append_raw(battery(0))
    assert run.frame(0)['values']['battery_voltage_v']['value']==pytest.approx(15.26)
    assert run.append_raw(record('ESC_STATUS',50,{'index':0,'rpm':[0]*4,'current':[0]*4}))
    assert run.frame(50)['values']['motor_1_rpm']['value']==0
    p=battery(100);p['source_system']=77
    assert not run.append_raw(p)
    assert run.excluded==1
    run.finish()
    other=store.create_session(SourceConfig(kind='udp',profile_id='mavlink_common_v1'))
    p=battery(0);p['fields']['voltages']=[16000]+[65535]*9
    other.append_raw(p);assert other.frame(0)['values']['battery_voltage_v']['value']==16
    other.finish()


def test_buffer_is_bounded_while_disk_keeps_full_history(store):
    run=make_normalized(store)
    for i in range(3102):run.append_normalized(normalized(run,i+1,i*100))
    assert len(run.entries)<=3002 and run.entries[0]['sequence']>1
    assert len(run.core.samples)<=3002
    replay=store.finish_session(run.run_id)
    assert replay.accepted==3102 and replay.frame(0)['values']['battery_current_a']['value']==0


def test_profiles_immutable_and_provider_contract(store):
    p=store.profile('normalized_flight_v1')
    p.sample_period_ms=50
    with pytest.raises(ValueError):store.register_profile(p)
    p.profile_id='normalized_flight_20hz';store.register_profile(p)
    run=make_normalized(store)
    provider=Provider(provider_id='transformer',name='Transformer',requirements=InputRequirement(sample_period_ms=50))
    assert compatibility(run,provider)==['입력 주기 불일치']
    run.finish()


def stream_event(run,provider='lstm'):
    ref,items=run.segment(1,run.accepted)
    end=items[-1]['t_ms']
    return dict(event_id='a',provider_id=provider,run_id=run.run_id,input_ref=ref,epoch=0,
        start_ms=0,end_ms=end,decided_at_ms=end,valid_until_ms=end+100,model_version='test',kind='detector',
        status='suspect',target='battery',score=.8,threshold=.6,title='test',origin='fixture')


def test_live_result_identity_hash_epoch_and_provenance(store):
    run=make_normalized(store)
    for i in range(3):run.append_normalized(normalized(run,i+1,i*100))
    e=stream_event(run);p=store.provider('lstm')
    assert validate_result(run,p,e)['input_ref']['kind']=='stream'
    for change in ['hash','session','provider','epoch','future']:
        bad=deepcopy(e)
        if change=='hash':bad['input_ref']['segment_sha256']='0'*64
        elif change=='session':bad['input_ref']['session_id']='other'
        elif change=='provider':bad['provider_id']='transformer'
        elif change=='epoch':bad['epoch']=4
        else:bad.update(decided_at_ms=1e9,valid_until_ms=1e9)
        with pytest.raises(ValueError):validate_result(run,p,bad)
    run.finish()


def test_explanation_cannot_pose_as_detection():
    e=dict(event_id='a',provider_id='llm',run_id='r',input_ref={'kind':'file','sha256':'0'*64},epoch=0,
        start_ms=0,end_ms=1,decided_at_ms=1,valid_until_ms=2,model_version='test',kind='explanation',
        status='explanation',title='context',evidence=[{'signal':'battery_current_a','unit':'A','note':'observation'}])
    assert AnalysisResult.model_validate(e).score is None
    with pytest.raises(ValueError):AnalysisResult.model_validate(dict(e,score=.9))
    with pytest.raises(ValueError):AnalysisResult.model_validate(dict(e,evidence=[]))


def test_legacy_migration_keeps_original_db_rows_and_files(tmp_path):
    old=Store(tmp_path);raw='\n'.join(json.dumps(r) for r in long_records()).encode();run=old.import_bytes(raw,'legacy.jsonl')
    old.save_events(run.run_id,fixture_events(run))
    new=StoreV2(tmp_path)
    assert all(e['provider_id']=='legacy' for e in new.analysis(run.run_id))
    assert len(new.events(run.run_id,'fixture'))==3 and run.path.read_bytes()==raw
    assert not (tmp_path/'arion.sqlite3').exists()


def test_independent_providers_error_isolation_and_fixture_filter(store,monkeypatch):
    run=store.import_bytes('\n'.join(json.dumps(r) for r in long_records()).encode(),'demo')
    registry=ProviderRegistry(store);registry.fixtures(run);registry.demo.add(run.run_id)
    assert len(registry.snapshot(run,run.duration_ms)[1])==6
    states={}
    for pid in ('lstm','transformer'):
        events=[deepcopy(e) for e in store.analysis(run.run_id) if e['provider_id']==pid]
        for e in events:e['event_id']='api-'+e['event_id']
        states[pid]={'schema_version':'arion-provider-results-v2','provider_id':pid,'run_id':run.run_id,'state':'ready','results':events}
    transport=httpx.MockTransport(lambda req:httpx.Response(200,json=states[req.url.params['provider_id']]))
    original=httpx.AsyncClient
    monkeypatch.setattr('arion.providers.httpx.AsyncClient',lambda **kw:original(transport=transport,**kw))
    async def check():
        for pid in states:await registry.connect(run.run_id,pid,'http://127.0.0.1:9999/results')
        await asyncio.sleep(.06)
        assert len(registry.snapshot(run,run.duration_ms)[1])==6
        assert all(e['event_id'].startswith('api-') for e in registry.snapshot(run,run.duration_ms)[1])
        states['lstm']['results'][0]['input_ref']['sha256']='0'*64
        await asyncio.sleep(.55)
        statuses={s['provider_id']:s['state'] for s in registry.states_for(run)}
        assert statuses=={'lstm':'error','transformer':'ready'}
        assert registry.snapshot(run,0)[1]==[]
        await registry.close()
    asyncio.run(check())


def test_udp_uses_exclusive_port_and_records_passively(store):
    async def check():
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
            conflicting=store.create_session(SourceConfig(port=port));manager=LiveSources(store)
            with pytest.raises(ValueError):await manager.connect(conflicting)
            assert conflicting.state=='error'
            conflicting.finish();store.active.pop(conflicting.run_id)
        run=store.create_session(SourceConfig(port=port));await manager.connect(run)
        mav=common.MAVLink(None,srcSystem=42,srcComponent=1)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
            sender.settimeout(.05)
            sender.sendto(mav.attitude_encode(1,.1,.2,.3,0,0,0).pack(mav),('127.0.0.1',port))
            await asyncio.sleep(.06)
            assert run.accepted==1 and run.frame(0)['values']['yaw_deg']['value']==pytest.approx(17.1887345)
            with pytest.raises(socket.timeout):sender.recv(1) # Receiver sends no heartbeat/commands.
        await manager.disconnect(run);await manager.connect(run)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
            sender.sendto(mav.attitude_encode(2,0,0,0,0,0,0).pack(mav),('127.0.0.1',port))
        await asyncio.sleep(.05);assert run.frame(run.duration_ms)['epoch']==1
        await manager.close()
    import uvloop
    uvloop.run(check())


def test_serial_pseudo_terminal_passive_capture(store):
    import pty
    async def check():
        master,slave=pty.openpty();device=os.ttyname(slave)
        manager=LiveSources(store)
        try:
            run=store.create_session(SourceConfig(kind='serial',device=device));await manager.connect(run)
            mav=common.MAVLink(None,srcSystem=42,srcComponent=1)
            os.write(master,mav.attitude_encode(1,0,0,0,0,0,0).pack(mav))
            await asyncio.sleep(.07)
            assert run.accepted==1
            await manager.close()
            assert store.get(run.run_id).frame(0)['values']['yaw_deg']['value']==0
        finally:os.close(master);os.close(slave)
    asyncio.run(check())


def test_live_pause_keeps_recording_and_return_live(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        m=client.post('/api/sessions',json={'kind':'normalized','profile_id':'normalized_flight_v1'}).json();run=app.state.store.get(m['run_id'])
        client.post(f'/api/sessions/{run.run_id}/frames',json=normalized(run))
        with client.websocket_connect(f'/api/ws/runs/{run.run_id}') as ws:
            first=ws.receive_json();assert first['playback']['following']
            ws.send_json({'action':'pause'});paused=ws.receive_json()['frame']['t_ms']
            client.post(f'/api/sessions/{run.run_id}/frames',json=normalized(run,2,100))
            next=ws.receive_json();assert next['frame']['t_ms']==paused and run.accepted==2
            ws.send_json({'action':'go_live'});current=ws.receive_json()
            assert current['playback']['following'] and current['frame']['t_ms']>=100
        ended=client.post(f'/api/sessions/{run.run_id}/finish').json()
        assert ended['source_kind']=='replay' and ended['input_sha256']


def test_unknown_mavlink_binary_payload_is_preserved_without_stopping(store):
    run=store.create_session(SourceConfig())
    assert not run.append_raw(record('AUTOPILOT_VERSION',0,{'custom_version':bytearray([0,255,1])}))
    assert run.append_raw(record('ATTITUDE',100,{'roll':0,'pitch':0,'yaw':0}))
    run.finish()
    rows=[json.loads(line) for line in run.path.read_text().splitlines()]
    assert rows[1]['raw']['fields']['custom_version']=={'$bytes':'00ff01'}
    assert run.accepted==1


def test_profile_semantics_not_tied_to_hardcoded_profile_name(store):
    profile=store.profile('px4_sih_v1');profile.profile_id='custom_sitl_contract_v3';store.register_profile(profile)
    run=store.create_session(SourceConfig(profile_id=profile.profile_id))
    p=battery(0);p['fields']['voltages']=[16000]+[65535]*9;p['fields']['current_battery']=-100
    run.append_raw(p)
    assert run.frame(0)['values']['battery_voltage_v']['value']==16
    assert run.frame(0)['values']['battery_current_a']['status']=='invalid'
    run.finish()


def test_stale_live_input_cannot_show_active_normal(store):
    run=make_normalized(store)
    for i in range(3):run.append_normalized(normalized(run,i+1,i*100))
    e=stream_event(run);e.update(valid_until_ms=5000,status='normal')
    store.save_analysis([validate_result(run,store.provider('lstm'),e)])
    registry=ProviderRegistry(store);registry.demo.add(run.run_id);registry.demo_keys[run.run_id]={registry.key(e)}
    states,_,active=registry.snapshot(run,2000)
    assert active['lstm'] is None and states[0]['state']=='unavailable'
    run.finish()


def test_profile_cannot_mislabel_canonical_ampere_signal(store):
    p=store.profile('normalized_flight_v1').model_dump();p['signals']['battery_current_a']['unit']='mA'
    with pytest.raises(ValueError):InputProfile.model_validate(p)


def test_event_cursor_query_retains_early_results_after_many_future_events(store):
    run=store.import_bytes('\n'.join(json.dumps(r) for r in long_records()).encode(),'archive')
    reg=ProviderRegistry(store);reg.fixtures(run)
    e=store.analysis(run.run_id)[0];rows=[]
    for i in range(2100):
        item=deepcopy(e);item.update(event_id=f'e{i}',decided_at_ms=i*10);rows.append(item)
    store.save_analysis(rows)
    assert any(x['event_id']=='e0' for x in store.analysis(run.run_id,20))


def test_crashed_journal_partial_tail_is_reported_and_original_kept(store):
    run=make_normalized(store);run.append_normalized(normalized(run));run.writer.close()
    with run.path.open('ab') as f:f.write(b'{"type":"sample","partial":')
    original=run.path.read_bytes();replay=SessionRun(run.path)
    assert replay.accepted==1 and replay.excluded==1
    assert replay.frame(0)['values']['battery_current_a']['value']==0 and run.path.read_bytes()==original
    store.active.pop(run.run_id)


def test_disconnected_normalized_stream_rejects_ingest_until_reconnected(store):
    run=make_normalized(store);manager=LiveSources(store)
    async def check():
        await manager.connect(run);run.append_normalized(normalized(run))
        await manager.disconnect(run)
        with pytest.raises(ValueError):run.append_normalized(normalized(run,2,100))
        await manager.connect(run);run.append_normalized(normalized(run,2,100))
        assert run.frame(100)['epoch']==1
        await manager.close()
    asyncio.run(check())


def test_declared_but_masked_required_input_cannot_be_normal(store):
    run=make_normalized(store)
    for i in range(3):run.append_normalized(normalized(run,i+1,i*100))
    p=Provider(provider_id='lstm',name='LSTM',requirements=InputRequirement(signals={'motor_1_rpm':'rpm'}))
    assert not compatibility(run,p) # Schema is compatible; actual observations are not ready.
    with pytest.raises(ValueError,match='필수 입력'):validate_result(run,p,stream_event(run))
    run.finish()


def test_frame_v2_replay_exposes_source_sequence_and_preprocess(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as client:
        run=app.state.store.import_bytes('\n'.join(json.dumps(r) for r in long_records()).encode(),'file')
        frame=client.get(f'/api/runs/{run.run_id}/frame?t_ms=200').json()
        assert frame['sequence']==2 and frame['source']=={'system':36,'component':1}
        assert frame['preprocess_version']=='raw-v1' and frame['sample_period_ms'] is None
