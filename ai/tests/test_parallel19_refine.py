import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from learning.parallel19.refine import candidates,finish_round,key
from learning.parallel19.final_verify import events,summarize,predict
from learning.parallel19.handover import identity


def item():
    r={'lr':.1,'best_validation_loss':1.,'status':'FIT_COMPLETE_STOPPED'}
    return {'best':r,'factor':2.,'tested':{key(.1):r},'rounds':0,'no_improvement':0,'status':'SEARCHING'}


def test_fresh_rates_do_not_repeat_and_two_small_improvements_stop():
    x=item()
    for n in range(2):
        rates=candidates(x);assert len(rates)==2
        assert all(key(r) not in x['tested'] for r in rates)
        finish_round(x,[{'lr':r,'best_validation_loss':x['best']['best_validation_loss']*.999,'status':'FIT_COMPLETE_STOPPED'} for r in rates])
    assert x['status']=='LOCAL_NO_MEANINGFUL_IMPROVEMENT'
    assert x['rounds']==2


def test_significant_improvement_resets_stagnation_and_divergence_not_selected():
    x=item();x['no_improvement']=1
    finish_round(x,[{'lr':.05,'best_validation_loss':.9,'status':'FIT_COMPLETE_STOPPED'},{'lr':.2,'status':'LR_DIVERGED'}])
    assert x['best']['lr']==.05 and x['no_improvement']==0
    assert key(.2) in x['tested']


def test_boundary_excludes_current_rate():
    x=item();x['best']['lr']=1.;x['tested']={key(1):x['best']}
    assert candidates(x)==[.5]


def test_events_use_actual_onset_and_existing_has_none():
    r={'role':'anomaly_test','run_id':'a','mission_interval':{'start_us':1000000,'end_us_exclusive':9000000},'source_evidence':{'scenario_name':'battery_aging_step','transition_evidence':{'battery':{'actual_onset_us':5000000,'actual_endpoint_us':6000000}}}}
    assert events(r)[0]['physical_onset_time_us']==5000000
    r['source_evidence']['scenario_name']='battery_aging_existing';r['source_evidence']['transition_evidence']={}
    assert events(r)[0]['physical_onset_time_us'] is None
    r['source_evidence']['scenario_name']='battery_aging_step'
    with pytest.raises(ValueError,match='actual fault'):events(r)


def test_dispatcher_identity_is_process_specific():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'])
    try:
        cmd,start=identity(child.pid);assert b'time.sleep(10)' in cmd and start.isdigit()
    finally:child.terminate();child.wait()


@pytest.mark.parametrize('task',['forecast','run_health'])
def test_final_predict_uses_only_numeric_features_and_saves_predictions(tmp_path,monkeypatch,task):
    import torch
    from learning.parallel19.fit import model
    from learning.parallel19.control import sha
    torch.set_num_threads(1);monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    entry={'task':task,'name':'lstm','hidden':64,'layers':2,'batch':256 if task=='forecast' else 8}
    a=np.random.default_rng(0);source=tmp_path/'in.npz'
    np.savez(source,X=a.normal(size=(3,100,19)).astype('f4'),Y=a.normal(size=(3,10,4)).astype('f4'),values=a.normal(size=(201,19)),time_us=np.arange(201,dtype='i8')*100000,origin_us=np.arange(3)*1000000,segment_id=np.zeros(3,dtype='i8'))
    rows=[{'run_id':'one','path':source.name,'sha256':sha(source),'forbidden_truth':'NEVER_FORWARD'}]
    out=tmp_path/'predictions';out.mkdir()
    result=predict(model(entry).eval(),entry,tmp_path,rows,{'x_mean':np.zeros(19),'x_scale':np.ones(19)},out,'test')
    assert (out/'one.npz').is_file()
    if task=='forecast':assert result['one']['residual'].shape==(3,10,4)
    else:assert result['one']['status']=='available'


def test_scheduler_fills_five_idle_slots_before_other_four_baselines_finish(tmp_path,monkeypatch):
    from learning.parallel19 import refine as r
    import datetime
    from learning.parallel19.control import write,read,sha
    entries=[{'id':f'm{i}','node':'DIS05' if i%2 else 'DIS06','compute_seconds':16-i} for i in range(16)]
    for folder in ('inputs','final','old'):(tmp_path/folder).mkdir()
    write(tmp_path/'inputs/cache.json',{'files':{}});write(tmp_path/'final/cache.json',{'files':{}})
    plan={'models':entries,'inputs':str(tmp_path/'inputs'),'final_inputs':str(tmp_path/'final'),'baseline':str(tmp_path/'old'),
          'code':str(tmp_path),'output':str(tmp_path/'out'),'source_sha256':{},'cache_sha256':sha(tmp_path/'inputs/cache.json'),'final_cache_sha256':sha(tmp_path/'final/cache.json'),
          'allocation':{'leader':1,'components':{'DIS05':1,'DIS06':2},'raw':'EndTime='+datetime.datetime.fromtimestamp(r.time.time()+3600).isoformat()}}
    write(tmp_path/'old/plan.json',plan);write(tmp_path/'plan.json',plan)
    monkeypatch.setattr(r,'allocations',lambda p:p);monkeypatch.setattr(r.signal,'signal',lambda *a:None)
    def baseline(plan,e):
        if int(e['id'][1:])>=12:return None
        x=item();x['best']['initial']='initial';x.update(round_open=False,round_results=[]);return x
    monkeypatch.setattr(r,'load_baseline',baseline);monkeypatch.setattr(r,'baseline_occupancy',lambda p:{'DIS05':3,'DIS06':1})
    starts=[]
    class Process:
        def __init__(self,pid):self.pid=pid
        def poll(self):return None
        def terminate(self):pass
        def wait(self,timeout):return 0
    def start(plan,root,entry,path,node,phase="search"):
        starts.append(node);return Process(len(starts)),open(tmp_path/f'log{len(starts)}','w')
    monkeypatch.setattr(r,'start_trial',start)
    def timeout(_):raise TimeoutError('fixture allocation deadline')
    monkeypatch.setattr(r.time,'sleep',timeout)
    with pytest.raises(TimeoutError):r.campaign(tmp_path/'plan.json')
    assert starts.count('DIS05')==2 and starts.count('DIS06')==3
    state=read(tmp_path/'out/state.json')
    assert state['status']=='BUDGET_STOPPED_NOT_CONVERGED'
    assert len(state['pending_trials'])==24
    assert sum(m['status']=='WAITING_BASELINE' for m in state['models'].values())==4


def test_reallocation_only_retries_incomplete_artifacts(tmp_path):
    from learning.parallel19.refine import next_attempt
    from learning.parallel19.control import write,read
    folder=tmp_path/'candidate-1';folder.mkdir();(folder/'last.pt').write_bytes(b'evidence')
    path=tmp_path/'request.json';write(path,{'output':str(folder),'lr':.1})
    job={'request':str(path),'attempt':1};next_attempt(tmp_path,job)
    assert job['attempt']==2 and (folder/'last.pt').read_bytes()==b'evidence'
    assert read(job['request'])['output'].endswith('candidate-1-attempt02')
    write(folder/'result.json',{'status':'FIT_COMPLETE_STOPPED'})
    completed={'request':str(path),'attempt':1};next_attempt(tmp_path,completed)
    assert completed=={'request':str(path),'attempt':1}
