"""Wait for registered boundaries, share GPU slot0 and run a resumable pipeline."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
from cooperative_process import run_cooperative
from container_memory_budget import container_budget
from prediction_progress import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text());stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    progress=Path(spec['progress'])
    def wait(paths,successful):
        while not stop.is_set():
            states={p:json.loads(Path(p).read_text()).get('status') if Path(p).exists() else 'queued' for p in paths}
            if successful and any(s in ['failed','stopped'] for s in states.values()):
                raise RuntimeError('Required technical preflight failed or stopped: '+str(states))
            accepted={'completed'} if successful else {'completed','failed','stopped'}
            if all(s in accepted for s in states.values()):return True
            atomic_json(progress,{'stage':'waiting_required_success' if successful else 'waiting_prior_terminal_boundary','states':states})
            stop.wait(5)
        return False
    if not wait(spec.get('wait_terminal',[]),False):return 130
    if not wait(spec.get('require_success',[]),True):return 130
    lock=Path(spec['gpu_lock']).open('a')
    with lock:
        while not stop.is_set():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:stop.wait(3)
        if stop.is_set():return 130
        while not stop.is_set():
            budget=container_budget()
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True,timeout=10).strip())
            if budget['conservative_available_gib']>=6 and free>=6144:break
            atomic_json(progress,{'stage':'waiting_resource_budget','gpu_free_mib':free,'container':budget})
            stop.wait(5)
        if stop.is_set():return 130
        for stage in spec['stages']:
            marker=Path(stage['completed_output'])
            if marker.exists():
                atomic_json(progress,{'stage':stage['name'],'status':'retained_completed_output','output':str(marker)})
                continue
            atomic_json(progress,{'stage':stage['name'],'status':'running','command':stage['command']})
            result=run_cooperative(stage['command'],cwd=spec['cwd'],check=False)
            if result.returncode:return result.returncode
            if not marker.exists():raise RuntimeError('Stage returned without required output: '+str(marker))
            if stop.is_set():return 130
    atomic_json(progress,{'stage':'completed','status':'completed'})
    return 0


if __name__=='__main__':raise SystemExit(main())
