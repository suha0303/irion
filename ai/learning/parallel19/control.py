"""One finite campaign under one user-owned heterogeneous Slurm allocation."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def completed_result(path, timeout=60):
    """NFS directory negative caches can outlive a successfully exited worker."""
    deadline=time.monotonic()+timeout
    while True:
        try:return read(path)
        except FileNotFoundError:
            if time.monotonic()>=deadline:raise
            time.sleep(.5)


def reusable_gates(plan):
    prior=plan.get('prior_gate_root')
    if not prior:return {}
    root=Path(prior);old=read(root/'plan.json')
    permitted='ai/learning/parallel19/control.py'
    if old['cache_sha256']!=plan['cache_sha256'] or old['models']!=plan['models']:
        raise ValueError('Prior gate data/models changed')
    before={k:v for k,v in old['source_sha256'].items() if k!=permitted}
    after={k:v for k,v in plan['source_sha256'].items() if k!=permitted}
    if before!=after:raise ValueError('Prior gate numerical code changed')
    passed={}
    for entry in plan['models']:
        folder=root/entry['id']/'gate';p=folder/'result.json'
        if not p.exists():continue
        result=read(p)
        if result.get('status')!='GATE_PASSED' or read(folder/'restored.json')!={'passed':True}:
            raise ValueError('Invalid saved gate result')
        passed[entry['id']]={**result,'reused_prior_gate':str(p),'prior_result_sha256':sha(p)}
    return passed


def allocations(plan):
    leader=os.environ['SLURM_JOB_ID']
    raw=subprocess.check_output(['scontrol','show','job','-o',leader],text=True)
    components={}
    for line in raw.splitlines():
        d=dict(x.split('=',1) for x in line.split() if '=' in x)
        node=d.get('NodeList');n={'DIS05':5,'DIS06':4}.get(node)
        if n is None or d.get('JobState')!='RUNNING' or d.get('Partition')!='p02' or d.get('UserId','').split('(')[0]!=plan['owner']:
            raise ValueError('Allocation owner/state/node/partition mismatch')
        if int(d['NumCPUs'])<n*4 or f'gres/gpu={n}' not in d['TRES'].split(','):
            raise ValueError('Allocation CPU/GPU count mismatch')
        components[node]=int(d['JobId'])
    if set(components)!={'DIS05','DIS06'}:raise ValueError('Both approved nodes required')
    plan['allocation']={'leader':int(leader),'components':components,'raw':raw}
    return plan


def campaign(template):
    plan=allocations(read(template));root=Path(plan['output']);root.mkdir(parents=True,exist_ok=False)
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(Path(plan['inputs'])/'cache.json')!=plan['cache_sha256']:raise ValueError('Cache manifest changed')
    for relative,digest in read(Path(plan['inputs'])/'cache.json')['files'].items():
        if sha(Path(plan['inputs'])/relative)!=digest:raise ValueError('Input changed: '+relative)
    for relative,digest in plan['source_sha256'].items():
        if sha(Path(plan['code'])/relative)!=digest:raise ValueError('Code changed: '+relative)
    frozen=root/'plan.json';write(frozen,plan)
    state=dict(status='RUNNING',stage='admission',allocation=plan['allocation'],completed={},active=[],updated_at=time.time())
    write(root/'state.json',state)
    active={};prior_gates=reusable_gates(plan)
    try:
        for stage in ('gate','lr','main'):
            pending=sorted(plan['models'],key=lambda e:-e['compute_seconds']);done=[]
            if stage=='gate':
                for entry in list(pending):
                    if entry['id'] in prior_gates:
                        write(root/entry['id']/'gate/result.json',prior_gates[entry['id']])
                        pending.remove(entry);done.append(entry['id'])
            state.update(stage=stage,completed_models=0,total_models=16)
            stage_start=time.monotonic()
            while pending or active:
                for node,capacity in [('DIS05',5),('DIS06',4)]:
                    used=sum(e['node']==node for _,e,_ in active.values())
                    while used<capacity:
                        entry=next((e for e in pending if e['node']==node),None)
                        if entry is None:break
                        pending.remove(entry)
                        argv=['srun','--jobid='+str(plan['allocation']['leader']),
                            '--het-group='+str(0 if node=='DIS05' else 1),'--partition=p02','--nodelist='+node,
                            '--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=16G','--exclusive','--exact',
                            '--kill-on-bad-exit=1',plan['python'],'-B','-m','learning.parallel19.fit',
                            '--plan',str(frozen),'--entry',entry['id'],'--stage',stage]
                        log=root/(entry['id']+'-'+stage+'.log');handle=log.open('xb')
                        process=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT)
                        active[process.pid]=(process,entry,handle);used+=1
                for pid,(process,entry,handle) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    handle.close();del active[pid]
                    if code:raise RuntimeError(f'{stage}:{entry["id"]}:exit{code}; preserve and stop')
                    result=completed_result(root/entry['id']/stage/'result.json')
                    expected={'gate':'GATE_PASSED','lr':'LR_COMPLETE','main':'FIT_COMPLETE_STOPPED'}[stage]
                    if result['status']!=expected:raise RuntimeError('Worker did not finish approved stage')
                    done.append(entry['id'])
                if stage=='lr' and time.monotonic()-stage_start>900:raise TimeoutError('Whole LR stage exceeded15minute budget')
                state.update(completed_models=len(done),active=[dict(model=e['id'],node=e['node'],pid=p.pid) for p,e,_ in active.values()],updated_at=time.time())
                write(root/'state.json',state)
                if active:time.sleep(2)
            if len(done)!=16:raise RuntimeError('Missing model result')
            state['completed'][stage]=done
        state.update(status='TRAINING_COMPLETE_STOPPED',active=[],training_complete=True,calibration_started=False,evaluation_started=False)
    except BaseException as error:
        state.update(status='FAILED_STOPPED',reason=f'{type(error).__name__}: {error}')
        # Only child steps launched by this exact controller are stopped.
        for process,_,_ in active.values():process.terminate()
        for process,_,handle in active.values():
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            handle.close()
        state['active']=[]
        raise
    finally:
        state['updated_at']=time.time();write(root/'state.json',state)
        lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('template');campaign(p.parse_args().template)
