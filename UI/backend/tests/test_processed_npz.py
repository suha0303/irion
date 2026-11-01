import hashlib
import io
import json
import os
import sqlite3
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from arion.contracts import InputProfile, SourceConfig, SignalDefinition
from arion.file_io import hash_file
from arion.main import create_app
from arion.npz_archive import load_npz, export_npz
from arion.npz_replay import NpzReplay
from arion.store_v2 import StoreV2
from arion.providers import validate_result
from test_pipeline import long_records
from test_streaming_v2 import stream_event


def create_processed(store,period=100):
    profile=InputProfile(profile_id=f'processed_{period}',input_format='normalized',preprocess_version='shared-test-v1',sample_period_ms=period,
        signals={'current':SignalDefinition(label='팩 전류',unit='A',ttl_ms=600,display_key='battery_current_a'),
                 'battery_voltage_v':SignalDefinition(label='전압',unit='V',ttl_ms=600),
                 'north_m':SignalDefinition(label='북',unit='m',ttl_ms=300),
                 'east_m':SignalDefinition(label='동',unit='m',ttl_ms=300),
                 'height_m':SignalDefinition(label='높이',unit='m',ttl_ms=300),
                 'roll_deg':SignalDefinition(label='Roll',unit='°',ttl_ms=300),
                 'pitch_deg':SignalDefinition(label='Pitch',unit='°',ttl_ms=300),
                 'yaw_deg':SignalDefinition(label='Yaw',unit='°',ttl_ms=300)},expected_rates_hz={'NORMALIZED':1000/period})
    store.register_profile(profile)
    return store.create_session(SourceConfig(name='시험 · 전처리 + NPZ',kind='normalized',profile_id=profile.profile_id))


def processed(run,sequence=1,ms=0,epoch=0):
    return {'schema_version':'arion-processed-v1','session_id':run.run_id,'profile_id':run.profile.profile_id,
            'sequence':sequence,'epoch':epoch,'source_time_ns':1_000_000_000_000_000_123+int(ms*1e6),
            'relative_time_ns':int(ms*1e6),'preprocess_version':run.profile.preprocess_version,'sample_period_ms':run.profile.sample_period_ms,
            'values':{'current':{'value':0,'unit':'A','mask':True,'status':'valid','age_ns':123},
                      'battery_voltage_v':{'value':16.25,'unit':'V','mask':True,'status':'valid','age_ns':0},
                      'yaw_deg':{'value':90,'unit':'°','mask':True,'status':'valid','age_ns':0}}}


def test_shared_canonical_values_nan_zero_age_and_replay(tmp_path):
    store=StoreV2(tmp_path);run=create_processed(store);p=processed(run)
    run.append_normalized(p);first=deepcopy(run.entries[0]);run.append_normalized(p)
    assert run.last_ingest['duplicate'] and run.accepted==1
    bad=deepcopy(p);bad['values']['current']['value']=1
    with pytest.raises(ValueError,match='식별자'):run.append_normalized(bad)
    p=processed(run,2,100);p['values']['current'].update(mask=False,status='invalid',value=7,age_ns=234)
    run.append_normalized(p);before=run.frame(100)
    replay=store.finish_session(run.run_id);assert isinstance(replay,NpzReplay)
    assert replay.frame(100)==before
    assert replay.frame(0)['values']['battery_current_a']['value']==0
    assert replay.frame(0)['values']['east_m']['value'] is None
    meta,arrays=load_npz(replay.archive.path/replay.archive.data['chunks'][0]['filename'])
    assert arrays['X'][0,0]==0 and arrays['X'][1,0]==7 and not arrays['mask'][1,0]
    assert np.isnan(arrays['X'][0,2]) and arrays['age_ns'][0,0]==123
    assert arrays['source_time_ns'][0]==1_000_000_000_000_000_123
    assert json.loads(arrays['rows'][0])==first and meta['normalization']=='physical_units'
    assert hash_file(run.path)==replay.sha256 and not (tmp_path/'arion.sqlite3').exists()
    store.close()


@pytest.mark.parametrize('period',[50,100,200])
def test_period_epoch_and_sixty_second_chunks(tmp_path,period):
    store=StoreV2(tmp_path);run=create_processed(store,period)
    for i in range(60_000//period+1):run.append_normalized(processed(run,i+1,i*period))
    run.archive.wait();assert run.archive.data['chunks'][0]['end_ms']==60000-period
    assert run.archive.data['chunks'][0]['rows']==60000//period
    run.append_normalized(processed(run,1,0,epoch=1));run.archive.wait()
    assert run.frame(run.duration_ms)['epoch']==1
    assert run.duration_ms==60000+period
    replay=store.finish_session(run.run_id)
    assert len(replay.archive.data['chunks'])==3 and replay.accepted==60000//period+2
    assert any(p['values']['current'] is None for p in replay.series()['points'])
    store.close()


def test_immutable_contract_missing_and_unit_mismatch(tmp_path):
    store=StoreV2(tmp_path);run=create_processed(store)
    for key,value in [('profile_id','other'),('schema_version','unknown'),('sample_period_ms',50),('preprocess_version','v2')]:
        p=processed(run);p[key]=value
        with pytest.raises(ValueError):run.append_normalized(p)
    p=processed(run);p['values']['current']['unit']='mA'
    with pytest.raises(ValueError,match='단위'):run.append_normalized(p)
    assert run.excluded==5 and run.accepted==0
    store.close()


def test_gap_invalidates_pose_and_segment_boundaries(tmp_path):
    store=StoreV2(tmp_path);run=create_processed(store)
    run.append_normalized(processed(run));run.append_normalized(processed(run,3,2500))
    assert run.dropped_sequences==1 and run.frame(2500)['epoch']==1
    assert run.frame(2499)['values']['yaw_deg']['status']=='stale'
    with pytest.raises(ValueError,match='epoch'):run.segment(1,2)
    store.close()


def test_failed_chunk_retries_from_journal_and_recovery_preserves_partial_tail(tmp_path,monkeypatch):
    import arion.npz_archive as archive_module
    store=StoreV2(tmp_path);run=create_processed(store)
    original=archive_module.atomic_npz
    monkeypatch.setattr(archive_module,'atomic_npz',lambda *args: (_ for _ in ()).throw(OSError('시험 디스크 오류')))
    run.append_normalized(processed(run));run.append_normalized(processed(run,2,60000));run.archive.wait()
    assert run.archive.status()['state']=='error' and '시험 디스크' in run.archive.status()['error']
    assert run.accepted==2 and run.frame(60000)['values']['battery_current_a']['value']==0
    monkeypatch.setattr(archive_module,'atomic_npz',original)
    run.archive.retry();run.archive.wait();assert run.archive.data['saved_rows']==1
    run.writer.close()
    with run.path.open('ab') as f:f.write(b'{"partial":')
    before=run.path.read_bytes()
    recovered=StoreV2(tmp_path);replay=recovered.get(run.run_id)
    assert replay.accepted==2 and replay.excluded==1 and run.path.read_bytes()==before
    assert replay.archive.status()['saved_rows']==2 and replay.frame(60000)['values']['current']['value']==0
    store.active.pop(run.run_id);run.archive.close();recovered.close()


def test_export_equal_to_chunks_and_independent_npz_import(tmp_path):
    store=StoreV2(tmp_path);run=create_processed(store)
    for i in range(4):run.append_normalized(processed(run,i+1,i*100))
    before=run.segment(1,4)[0];replay=store.finish_session(run.run_id)
    assert replay.segment(1,4)[0]==before
    path=export_npz(replay.archive,tmp_path/'full.npz');metadata,arrays=load_npz(path)
    assert len(arrays['X'])==4 and metadata['final_log_sha256']==replay.sha256
    imported=store.import_bytes(path.read_bytes(),'비행_내보내기.npz')
    assert imported.run_id!=run.run_id and imported.sha256==hash_file(path)
    assert imported.frame(100)['values']==replay.frame(100)['values']
    assert store.import_bytes(path.read_bytes(),'again.npz').run_id==imported.run_id
    assert not store.analysis(imported.run_id)
    store.close()


def test_physical_training_npz_import_no_fake_pose_or_model_readiness(tmp_path):
    store=StoreV2(tmp_path);buffer=io.BytesIO()
    meta={'normalization':'identity_not_model_ready','features':[{'name':'battery_current_a','unit':'A'},{'name':'roll_rad','unit':'rad'}],
          'config':{'tick_ns':100_000_000},'input_files':[{'sha256':'0'*64}]}
    np.savez(buffer,metadata=json.dumps(meta),X=np.array([[0.,.1],[2.,.2]]),mask=np.array([[True,False],[True,True]]),
             tick_ns=np.array([500000000,600000000],dtype=np.int64),epoch=np.zeros(2,dtype=np.int64),age_ns=np.zeros((2,2),dtype=np.int64))
    replay=store.import_bytes(buffer.getvalue(),'training.npz')
    assert replay.frame(0)['values']['battery_current_a']['value']==0
    assert 'yaw_deg' not in replay.frame(0)['values'] and 'east_m' not in replay.frame(0)['values']
    assert replay.run_id.startswith('npz-')
    store.close()


def test_sqlite_one_time_read_only_migration_and_provider_identity(tmp_path):
    database=tmp_path/'arion.sqlite3';raw='\n'.join(json.dumps(r) for r in long_records()).encode()
    raw_path=tmp_path/'original.jsonl';raw_path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest();id='run-'+sha[:16]
    event={'run_id':id,'event_id':'legacy-event','model_version':'old','origin':'model','decided_at_ms':100}
    with sqlite3.connect(database) as db:
        db.executescript('CREATE TABLE runs(id TEXT,sha TEXT,name TEXT,path TEXT,created_at TEXT); CREATE TABLE detections(payload TEXT); CREATE TABLE profiles(id TEXT,payload TEXT); CREATE TABLE providers(id TEXT,payload TEXT);')
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?)',(id,sha,'기존 비행',str(raw_path),'2026-10-10'))
        db.execute('INSERT INTO detections VALUES(?)',(json.dumps(event),))
    before=database.read_bytes();store=StoreV2(tmp_path)
    assert store.get(id).sha256==sha and raw_path.read_bytes()==raw and database.read_bytes()==before
    assert store.analysis(id)[0]['provider_id']=='legacy'
    assert (tmp_path/'backups'/'before-file-storage.sqlite3').exists()
    marker=(tmp_path/'backups'/'file-migration.json').read_bytes()
    StoreV2(tmp_path);assert (tmp_path/'backups'/'file-migration.json').read_bytes()==marker
    store.close()


def test_api_save_export_download_replay_and_future_results(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as c:
        run=create_processed(app.state.store)
        for i in range(3):
            response=c.post(f'/api/sessions/{run.run_id}/frames',json=processed(run,i+1,i*100));assert response.status_code==200
        assert c.post(f'/api/sessions/{run.run_id}/frames',json=processed(run)).json()['duplicate']
        event=stream_event(run);app.state.store.save_analysis([event])
        assert not c.get(f'/api/runs/{run.run_id}/frame?t_ms=0').json()['events']
        with c.websocket_connect(f'/api/ws/runs/{run.run_id}') as ws:
            ws.receive_json();ws.send_json({'action':'pause'});time.sleep(.12)
            c.post(f'/api/sessions/{run.run_id}/frames',json=processed(run,4,300))
            assert run.accepted==4 and run.path.stat().st_size>0
        assert c.post(f'/api/sessions/{run.run_id}/finish').status_code==200
        chunks=c.get(f'/api/runs/{run.run_id}/npz/chunks').json();assert chunks['npz']['saved_rows']==4
        job=c.post(f'/api/runs/{run.run_id}/npz/export').json()
        for _ in range(100):
            job=c.get('/api/exports/'+job['job_id']).json()
            if job['state'] in ('ready','error'):break
            time.sleep(.01)
        assert job['state']=='ready'
        download=c.get(job['download_url']);assert download.status_code==200
        assert c.post('/api/runs/import',files={'file':('재생.npz',download.content)}).status_code==200
        assert c.get('/api/health').json()['storage']=='files'


def test_older_duplicate_outside_display_buffer_is_idempotent(tmp_path):
    store=StoreV2(tmp_path);run=create_processed(store)
    for i in range(3102):run.append_normalized(processed(run,i+1,i*100))
    before=run.accepted;run.append_normalized(processed(run))
    assert run.last_ingest['duplicate'] and run.accepted==before and len(run.entries)<=3002
    store.close()


def test_restart_does_not_open_sqlite_and_retains_results_profiles(tmp_path,monkeypatch):
    store=StoreV2(tmp_path);run=create_processed(store)
    for i in range(3):run.append_normalized(processed(run,i+1,i*100))
    event=stream_event(run);store.save_analysis([event]);store.finish_session(run.run_id);store.close()
    monkeypatch.setattr(sqlite3,'connect',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('런타임 SQL 사용')))
    reopened=StoreV2(tmp_path)
    assert reopened.analysis(run.run_id)==[event]
    assert reopened.profile(run.profile.profile_id).model_dump()==run.profile.model_dump()
    assert reopened.get(run.run_id).accepted==3
    reopened.close()


def test_import_retries_after_invalid_row_and_original_is_preserved(tmp_path):
    store=StoreV2(tmp_path);buffer=io.BytesIO()
    meta={'normalization':'physical_units','features':[{'name':'current','unit':'A'}]}
    np.savez(buffer,metadata=json.dumps(meta),X=np.array([[1.],[2.]]),mask=np.array([[True],[True]]),
             tick_ns=np.array([0,100000000],dtype=np.int64),epoch=np.zeros(2,dtype=np.int64),age_ns=np.zeros((2,1),dtype=np.int64),
             rows=np.array([json.dumps({'frame':{'mode':'ok'}}),json.dumps({'frame':{'mode':123}})]))
    data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
    for _ in range(2):
        with pytest.raises(ValueError):store.import_bytes(data,'invalid.npz')
    assert (tmp_path/'logs'/(digest+'.npz')).read_bytes()==data
    assert not list((tmp_path/'imports').iterdir()) and not store.catalog['sessions']
    store.close()


def test_saved_result_and_future_seek_do_not_inflate_latency(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as c:
        run=app.state.store.import_bytes('\n'.join(json.dumps(r) for r in long_records()).encode(),'latency.jsonl')
        from arion.ai import fixture_events
        e=fixture_events(run)[0];e.pop('input_sha256')
        e.update(provider_id='lstm',kind='detector',schema_version='arion-analysis-v2',input_ref={'kind':'file','sha256':run.sha256},origin='model')
        app.state.store.save_analysis([e]);key=app.state.providers.key(e)
        app.state.providers.received[key]=(time.perf_counter()-600,e['decided_at_ms'])
        with c.websocket_connect(f'/api/ws/runs/{run.run_id}') as ws:
            packet=ws.receive_json();ws.send_json({'action':'seek','t_ms':e['decided_at_ms']})
            for _ in range(4):
                packet=ws.receive_json();ws.send_json({'action':'ack','sequence':packet['sequence']})
            assert c.get('/api/metrics').json()['result_receipt_to_render_ack']['samples']==0
        # A new result arriving while the replay cursor is before its decision is also excluded.
        e['event_id']='future';app.state.store.save_analysis([e]);key=app.state.providers.key(e)
        with c.websocket_connect(f'/api/ws/runs/{run.run_id}') as ws:
            ws.receive_json();app.state.providers.received[key]=(time.perf_counter(),e['decided_at_ms'])
            ws.receive_json();ws.receive_json();ws.send_json({'action':'seek','t_ms':e['decided_at_ms']})
            for _ in range(4):
                packet=ws.receive_json();ws.send_json({'action':'ack','sequence':packet['sequence']})
            assert c.get('/api/metrics').json()['result_receipt_to_render_ack']['samples']==0


def test_external_finish_updates_websocket_manifest_to_replay(tmp_path):
    app=create_app(tmp_path)
    with TestClient(app) as c:
        run=create_processed(app.state.store);run.append_normalized(processed(run))
        with c.websocket_connect(f'/api/ws/runs/{run.run_id}') as ws:
            assert ws.receive_json()['frame']['manifest']['source_kind']=='live'
            c.post(f'/api/sessions/{run.run_id}/finish').raise_for_status()
            changed=False
            for _ in range(12):
                packet=ws.receive_json()
                if packet['frame'].get('manifest',{}).get('source_kind')=='replay':
                    assert packet['frame']['manifest']['npz']['saved_rows']==1;changed=True;break
            assert changed
