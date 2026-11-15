"""Execute an immutable single-command task plan with durable lifecycle records.

SIGTERM/SIGINT forward to the cooperative child. Requeue with --resume after
stopping; command-specific checkpoints determine its documented resume grain.
"""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    plan=json.loads(args.plan.read_text());folder=args.plan.parent
    lock=(folder/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=folder/'state.json';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    state=json.loads(path.read_text()) if path.exists() else {'task_id':plan['task_id'],'created_utc':now(),'history':[]}
    if path.exists() and not args.resume:raise ValueError('Existing task; inspect then resume')
    state.update(status='preflight',pid=os.getpid(),updated_utc=now(),plan=str(args.plan),command=plan['command'],
        environment={'python':sys.version,'platform':platform.platform(),'torch':importlib.metadata.version('torch')},
        resources=plan['resources'],queue_record=str(folder/'queue_submission.json'))
    atomic_json(path,state)
    child=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for name,expected in {**plan['input_sha256'],**plan['source_sha256']}.items():
            if digest(Path(name))!=expected:raise ValueError(f'Frozen input/source changed: {name}')
        if state.get('output_sha256'):
            for name,expected in state['output_sha256'].items():
                if digest(Path(name))!=expected:raise ValueError(f'Completed output changed: {name}')
            state.update(status='completed',exit_code=0);return 0
        state.update(status='running');state['history'].append({'event':'start_or_resume','utc':now()});atomic_json(path,state)
        with (folder/'execution.log').open('a') as log:
            child=subprocess.Popen(plan['command'],cwd=plan['cwd'],stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;atomic_json(path,state)
            if stopping:child.send_signal(signal.SIGTERM)
            code=child.wait()
        state.update(exit_code=code,status='stopped' if stopping or code==130 else 'completed' if code==0 else 'failed')
        if code==0:state['output_sha256']={name:digest(Path(name)) for name in plan['outputs']}
        return 130 if stopping else code
    except BaseException as error:
        state.update(status='failed',error=repr(error),exit_code=1);raise
    finally:
        state.update(updated_utc=now());state['history'].append({'event':state['status'],'utc':now()});atomic_json(path,state)


if __name__=='__main__':sys.exit(main())
