"""Hold an existing allocation by pausing only its idle-able dispatcher.

Workers continue normally. The original dispatcher is always resumed on exit,
so its original salloc owner performs the normal resource release. No GPU, model,
input or existing evidence is modified. A separate READY seals the successor.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from .control import allocations, read, write, sha


def identity(pid):
    root=Path('/proc')/str(pid)
    if root.stat().st_uid!=os.getuid():raise ValueError('Dispatcher owner mismatch')
    text=(root/'stat').read_text().rsplit(')',1)[1].split()
    return (root/'cmdline').read_bytes(),text[19]


def run(args):
    def interrupted(signum, frame):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    statepath=root/'handover-state.json'
    if statepath.exists():raise ValueError('Existing handover; no restart')
    baseline=Path(args.baseline);plan=read(baseline/'output/plan.json')
    os.environ['SLURM_JOB_ID']=str(args.job)
    allocation=allocations(plan)['allocation']
    before=identity(args.pid)
    if b'learning.parallel19.control\x00' not in before[0] or str(baseline/'plan-template.json').encode() not in before[0]:
        raise ValueError('Unexpected original dispatcher command')
    state=read(baseline/'output/state.json')
    if state['status']!='RUNNING' or state['stage']!='main' or state['completed_models']+len(state['active'])!=16:
        raise ValueError('Only a fully dispatched main stage can be handed over')
    status={'status':'HOLDING_ALLOCATION','baseline':str(baseline),'dispatcher_pid':args.pid,
            'dispatcher_start':before[1],'allocation':allocation,'workers_unchanged':True,'started':time.time()}
    child=None;paused=False
    try:
        os.kill(args.pid,signal.SIGSTOP);paused=True;write(statepath,status)
        while True:
            if (root/'STOP').exists():raise InterruptedError('User stopped successor')
            if time.time()-status['started']>1800:raise TimeoutError('Successor preparation exceeded30minutes')
            ready=root/'READY.json'
            if ready.exists():break
            time.sleep(2)
        ready=read(ready);template=Path(ready['plan'])
        if sha(template)!=ready['sha256']:raise ValueError('Successor plan changed')
        successor=read(template)
        if successor['baseline']!=str(baseline/'output') or successor['handover']!=str(root):raise ValueError('Successor binding mismatch')
        env=dict(os.environ,PYTHONPATH=successor['code']+'/ai:'+successor['code']+'/ai/src')
        status.update(status='RUNNING_SUCCESSOR',successor_plan=str(template));write(statepath,status)
        child=subprocess.Popen([successor['python'],'-B','-m','learning.parallel19.refine','--plan',str(template)],env=env)
        rc=child.wait()
        status.update(status='RELEASING_ALLOCATION',successor_exit=rc)
    except BaseException as e:
        status.update(status='FAILED_RELEASING',reason=f'{type(e).__name__}: {e}')
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=60)
        raise
    finally:
        if paused and identity(args.pid)==before:
            os.kill(args.pid,signal.SIGCONT);status['original_dispatcher_resumed']=True
        status['updated']=time.time();write(statepath,status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--job',required=True,type=int);p.add_argument('--pid',required=True,type=int)
    run(p.parse_args())
