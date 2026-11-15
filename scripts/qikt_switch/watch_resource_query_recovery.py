"""Bounded recovery of terminal monitor-query failures, never live training.

Does not alter frozen scientific sources or parameters. A resumed family keeps
completed training/evaluation and follows its existing checkpoint semantics.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import datetime
from prediction_progress import atomic_json


def active_family(folder,root,name):
    needles=[str(folder),str(folder.resolve()),str(root/name),str((root/name).resolve())]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if 'python' in cmd and any(n in cmd for n in needles):return int(proc.name)
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--socket',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    while not stop.is_set():
        env=os.environ.copy();env['TS_SOCKET']=a.socket
        try:listing=subprocess.check_output(['tsp','-l'],env=env,text=True,timeout=10)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
            atomic_json(a.output,{'status':'queue_observation_unavailable','action':'No restart; re-poll same queue'});stop.wait(10);continue
        live_lines=[line for line in listing.splitlines() if len(line.split())>1 and line.split()[1] in ['queued','running']];actions=[];all_terminal=True
        for root in a.roots:
            reg=json.loads((root/'registration.json').read_text())
            for name in reg['families']:
                if name=='B0':continue
                folder=root/'jobs'/name;statepath=folder/'state.json'
                if not statepath.exists():all_terminal=False;continue
                state=json.loads(statepath.read_text());status=state['status']
                if status not in ['completed','failed','stopped']:all_terminal=False;continue
                if status!='failed':continue
                queued=any(str(folder/'plan.json') in line for line in live_lines);pid=active_family(folder,root,name)
                if queued or pid:all_terminal=False;actions.append({'root':str(root),'family':name,'action':'Existing queued or live handle, no duplicate','pid':pid});continue
                log=(folder/'execution.log').read_text(errors='replace')[-12000:]
                expected="subprocess.TimeoutExpired: Command '['nvidia-smi'" in log and 'resource_admission.py' in log and log.rstrip().endswith('timed out after 10 seconds')
                count=sum(h.get('event')=='failed' for h in state.get('history',[]))
                if not expected or count>2:actions.append({'root':str(root),'family':name,'action':'Scientific/manual review required; no automatic restart','consecutive_recorded_failures':count});continue
                environment=os.environ.copy();environment.update(TS_SOCKET=a.socket,PYTHONPATH=str(root/'runtime'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1');environment.pop('CUDA_VISIBLE_DEVICES',None);py=state['command'][0]
                job=int(subprocess.check_output(['tsp','-L',f'MONITOR_RECOVERY_{root.name}_{name}',py,str(root/'runtime/scripts/qikt_switch/queued_task.py'),'--plan',str(folder/'plan.json'),'--resume'],env=environment,cwd=root/'runtime',text=True).strip())
                item={'job_id':job,'socket':a.socket,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Terminal verified nvidia-smi query timeout; no active/queued family; bounded same-plan resume, completed artifacts retained','recorded_failure_count':count};history=folder/'resource_recovery_history.json';records=json.loads(history.read_text()) if history.exists() else [];records.append(item);atomic_json(history,records);atomic_json(folder/'queue_submission.json',item);actions.append({'root':str(root),'family':name,'action':'Queued bounded same-plan recovery',**item});all_terminal=False
        atomic_json(a.output,{'status':'terminal' if all_terminal else 'watching','actions':actions,'policy':'At most two recorded query-failure recoveries; no observation-timeout restarts, no live training termination, no scientific-failure retries'})
        if all_terminal:return
        stop.wait(15)
    return 130


if __name__=='__main__':raise SystemExit(main())
