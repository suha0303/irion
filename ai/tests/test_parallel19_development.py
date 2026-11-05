from pathlib import Path
from learning.parallel19.refine import enqueue_development,development_done
from learning.parallel19.control import read,write,sha
from learning.parallel19.final_verify import summarize


def test_completed_fit_requires_development_and_deduplicates(tmp_path):
    folder=tmp_path/'fit'
    for name in ('model.safetensors','result.json','scaler.json'):write(folder/name,{'test':1})
    write(folder/'integrity.json',{p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    r={'folder':str(folder),'status':'FIT_COMPLETE_STOPPED','lr':.1,'best_validation_loss':.2}
    item={'status':'SEARCHING','tested':{'baseline':r},'round_results':[r]};dev={};pending=[]
    assert not development_done(item,dev)
    enqueue_development({},tmp_path/'output',{'m':item},pending,dev)
    enqueue_development({},tmp_path/'output',{'m':item},pending,dev)
    assert len(pending)==1 and pending[0]['kind']=='development'
    request=read(pending[0]['request']);assert request['cohort']=='development'
    assert 'selected-models.json' not in request['selection_manifest']
    dev[str(folder)]['status']='DEVELOPMENT_EVALUATION_COMPLETE'
    assert development_done(item,dev)
    assert not development_done({'status':'WAITING_BASELINE'},dev)


def test_development_summary_does_not_claim_729_runs():
    records=[{'role':'normal_test','alarm':False,'scenario':'hover','normal_kind':'standalone'} for _ in range(162)]
    records += [{'role':'anomaly_test','alarm':True,'scenario':'battery','normal_kind':'standalone'} for _ in range(81)]
    assert summarize(records,243)['all_required_runs_scored'] is True
    assert summarize(records,729)['all_required_runs_scored'] is False


def test_divergent_lr_is_not_sent_for_detection_evaluation(tmp_path):
    model={'status':'SEARCHING','tested':{'x':{'status':'LR_DIVERGED','folder':'unused'}},'round_results':[]}
    pending=[];development={};enqueue_development({},tmp_path,{'m':model},pending,development)
    assert pending==[] and development=={}


def test_development_worker_never_requires_final_inputs(tmp_path,monkeypatch):
    import argparse,contextlib,json
    import numpy as np
    import torch
    import safetensors.torch
    from learning.parallel19 import final_verify as f,fit
    root=tmp_path/'dev';root.mkdir();scaler={'x_mean':np.zeros(19),'x_scale':np.ones(19),'y_mean':np.zeros(4),'y_scale':np.ones(4)}
    np.savez(root/'scaler.npz',**scaler);normal=tmp_path/'inputs/main';normal.mkdir(parents=True);(normal/'scaler.npz').write_bytes((root/'scaler.npz').read_bytes())
    rows=[]
    for role,cohort,n in [('normal_validation','normal_validation',81),('normal_calibration','normal_calibration',81),('normal_test','development',162),('anomaly_test','development',81)]:
        for i in range(n):
            rid=f'{role}{i}';rows.append({'run_id':rid,'role':role,'cohort':cohort,'group_id':rid,'case_id':rid,'mission_interval':{'start_us':0,'end_us_exclusive':480000000,'clock_id':'px4'},'source_evidence':{'scenario_name':'battery_aging_existing' if role=='anomaly_test' else 'hover_normal','transition_evidence':{}}})
    write(root/'cache.json',{'runs':rows})
    campaign=tmp_path/'campaign';selection=campaign/'development-selections/a.json';write(selection,{'frozen':True})
    training=tmp_path/'training';training.mkdir();(training/'model.safetensors').write_bytes(b'fixture')
    request={'cohort':'development','node':'DIS05','output':str(campaign/'evaluation'),'selection_manifest':str(selection),'selected_manifest_sha256':sha(selection),'selected':{'folder':str(training),'model_sha256':sha(training/'model.safetensors'),'lr':.1}}
    write(selection,{'selected':request['selected'],'evaluation_cohort':'development','final_used':False})
    request['selected_manifest_sha256']=sha(selection)
    plan={'models':[{'id':'m','task':'forecast','node':'DIS05'}],'output':str(campaign),'inputs':str(tmp_path/'inputs'),'development_inputs':str(root),'development_cache_sha256':sha(root/'cache.json')}
    write(tmp_path/'plan.json',plan);write(tmp_path/'request.json',request)
    class Dummy:
        def load_state_dict(self,_):pass
        def cuda(self):return self
        def eval(self):return self
    monkeypatch.setattr(fit,'model',lambda _:Dummy());monkeypatch.setattr(fit,'runtime',lambda *a:None);monkeypatch.setattr(fit,'backend',lambda _:contextlib.nullcontext())
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda _:None);monkeypatch.setattr(safetensors.torch,'load_file',lambda _: {})
    phases=[]
    def predict(model,entry,root,rows,scaler,out,phase):
        phases.append(phase)
        return {r['run_id']:{'residual':np.full((4,10,4),1. if r['role']=='anomaly_test' else .01),'origin_us':np.arange(4)*1000000+10000000,'segment_id':np.zeros(4,dtype=int)} for r in rows}
    monkeypatch.setattr(f,'predict',predict)
    f.run(argparse.Namespace(plan=tmp_path/'plan.json',request=tmp_path/'request.json',entry='m'))
    result=read(campaign/'evaluation/result.json')
    assert result['status']=='DEVELOPMENT_EVALUATION_COMPLETE'
    assert result['summary']['normal']['runs']==162 and result['summary']['anomaly']['runs']==81
    assert phases==['normal_validation','normal_calibration','development_inference']
    assert result['final_cache_sha256'] is None
