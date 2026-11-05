"""Only scheduler-budget continuation; never retry a model/runtime failure."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from .control import read,write,sha


def queue(owner):
    return subprocess.check_output(['squeue','-u',owner,'-h','-o','%i|%P|%T|%N|%b'],text=True).strip()


def owns_job(lines,allocation):
    ids={str(allocation['leader']),*(str(x) for x in allocation['components'].values())}
    return any(row.split('|')[0].split('+')[0] in ids for row in lines.splitlines())


def run(template):
    plan=read(template);root=Path(plan['output']);control=Path(plan['handover'])/'allocation-control.json'
    while True:
        if not (root/'state.json').exists():return
        state=read(root/'state.json');allocation=state['allocation']
        while owns_job(queue(plan['owner']),allocation):
            write(control,{'status':'WAITING_OWN_ALLOCATION_RELEASE','allocation':allocation,'updated':time.time()});time.sleep(2)
        if state['status']=='FINAL_VERIFICATION_COMPLETE_PENDING_RELEASE_CHECK':
            state.update(status='REPORT_READY_STOPPED',allocation_released=True,released_at=time.time(),next_allocation_queued=False)
            write(root/'state.json',state);write(control,{'status':'COMPLETE_RELEASED','at':time.time()});return
        if state['status']!='BUDGET_STOPPED_NOT_CONVERGED' or (Path(plan['handover'])/'STOP').exists():
            write(control,{'status':'STOPPED_NO_RETRY','reason':state['status'],'at':time.time()});return
        if queue(plan['owner']):
            write(control,{'status':'BLOCKED_OTHER_OWNER_JOBS','jobs':queue(plan['owner']),'at':time.time()});return
        handles=[]
        try:
            for name in ('.drone-ai-submit.lock','.drone-ai-gpu.lock','.x500-ai-gpu.lock'):
                handle=(Path.home()/name).open('a');handles.append(handle)
                deadline=time.monotonic()+120
                while True:
                    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()>deadline:raise RuntimeError('Project lock busy; no new allocation')
                        time.sleep(2)
            if queue(plan['owner']):raise RuntimeError('New account job appeared; no duplicate submission')
            for name,digest in plan['source_sha256'].items():
                if sha(Path(plan['code'])/name)!=digest:raise ValueError('Successor code changed')
            argv=['salloc','--no-bell','--job-name=drone-ai-lr19-continue',
                  '--partition=p02','--nodelist=DIS05','--nodes=1','--ntasks=5','--cpus-per-task=4','--gres=gpu:5','--mem=96G','--time=06:00:00',':',
                  '--partition=p02','--nodelist=DIS06','--nodes=1','--ntasks=4','--cpus-per-task=4','--gres=gpu:4','--mem=96G','--time=06:00:00',
                  plan['python'],'-B','-m','learning.parallel19.refine','--plan',str(template)]
            write(control,{'status':'REQUESTING_NEXT_NINE_GPU_ALLOCATION','prior':allocation,'at':time.time()})
            child=subprocess.Popen(argv)
            while child.poll() is None:
                if (Path(plan['handover'])/'STOP').exists():
                    child.terminate();child.wait();return
                time.sleep(2)
            updated=read(root/'state.json')
            if updated.get('allocation')==allocation:
                write(control,{'status':'ALLOCATION_FAILED_NO_RETRY','exit':child.returncode,'at':time.time()});return
        finally:
            for handle in handles:handle.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();run(Path(a.plan))
