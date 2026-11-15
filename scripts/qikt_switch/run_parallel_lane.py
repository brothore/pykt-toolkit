"""One bounded second GPU lane; the existing exclusive lane is untouched."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from prediction_progress import atomic_json
from container_memory_budget import container_budget


def observe(spec):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    pids=[int(s.strip()) for s in raw.splitlines() if s.strip().isdigit()]
    commands={}
    for pid in pids:
        try:commands[str(pid)]=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:pass
    unknown=[pid for pid,cmd in commands.items() if not any(prefix in cmd for prefix in spec['primary_runtime_prefixes'])]
    raw_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    utilization,free,used=map(int,raw_gpu.strip().split(','))
    return {'time':time.time(),'primary_pids':commands,'unknown_gpu_pids':unknown,'gpu_utilization':utilization,
            'gpu_free_mib':free,'gpu_used_mib':used,**container_budget()}


def primary_status(spec):
    pending=False;handoff=[]
    for root in map(Path,spec.get('primary_roots',[])):
        for family in json.loads((root/'registration.json').read_text())['families']:
            path=root/'jobs'/family/'state.json'
            state=json.loads(path.read_text()) if path.exists() else {'status':'queued'}
            if state['status'] not in {'completed','failed','stopped'}:
                pending=True;p=root/family/'gpu_admission.json'
                if p.exists() and json.loads(p.read_text())['status']=='waiting_for_external_gpu_work':handoff.append(str(p))
    return pending,handoff


def checkpoint_ready(commands,own_prefix):
    for command in commands.values():
        if own_prefix not in command:continue
        words=command.split()
        if '--save_dir' in words:
            folder=Path(words[words.index('--save_dir')+1])
            if any(folder.rglob('training_resume.ckpt')):return True
        if '--checkpoint-dir' in words:
            folder=Path(words[words.index('--checkpoint-dir')+1])
            if (folder/'training_resume.ckpt').exists():return True
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());record=Path(spec['admission_record']);stop=threading.Event();child=None
    def stopping(*_):
        stop.set()
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    for dependency in spec.get('dependencies',[]):
        while not stop.is_set():
            state=Path(dependency['state']);artifact=Path(dependency['result'])
            if artifact.exists():
                if not json.loads(artifact.read_text()).get('passed'):raise ValueError('Native preflight did not pass')
                break
            if state.exists() and json.loads(state.read_text())['status'] in {'failed','stopped','completed'}:
                raise RuntimeError('Preflight terminated without success')
            stop.wait(3)
    if stop.is_set():return 130
    lock=Path(spec['lane_lock']).open('a');yields=0
    while not stop.is_set():
        while not stop.is_set():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:stop.wait(3)
        while not stop.is_set():
            obs=observe(spec)
            pending,_=primary_status(spec)
            admitted=(len(obs['primary_pids'])<=1 and not obs['unknown_gpu_pids'] and
                      (len(obs['primary_pids'])==1 or not pending) and
                      obs['gpu_free_mib']>=spec['min_gpu_free_mib'] and
                      obs['conservative_available_gib']>=spec['min_container_headroom_gib'])
            atomic_json(record,dict(obs,status='second_lane_admitted' if admitted else 'waiting_budget',
                                   max_physical_gpu_tasks=2,does_not_modify_primary_lock=True))
            if admitted:break
            stop.wait(5)
        if stop.is_set():return 130
        child=subprocess.Popen(spec['command'],cwd=spec['cwd']);yield_requested=False
        with record.with_suffix('.resources.jsonl').open('a',buffering=1) as log:
            while child.poll() is None:
                obs=observe(spec);_,handoff=primary_status(spec);obs['primary_handoff_waiters']=handoff
                log.write(json.dumps(obs)+'\n')
                own_present=any(spec['own_runtime_prefix'] in cmd for cmd in obs['primary_pids'].values())
                unregistered=any(not any(prefix in cmd for prefix in
                    [*spec['primary_runtime_prefixes'],spec['own_runtime_prefix']]) for cmd in obs['primary_pids'].values())
                pressure=(obs['conservative_available_gib']<2 or obs['gpu_free_mib']<2048 or
                          len(obs['primary_pids'])>2 or unregistered)
                if (handoff or pressure) and own_present and not yield_requested and checkpoint_ready(obs['primary_pids'],spec['own_runtime_prefix']):
                    yield_requested=True;child.send_signal(signal.SIGTERM)
                    atomic_json(record,dict(obs,status='yield_requested_at_safe_checkpoint',reason='primary_handoff' if handoff else 'resource_pressure'))
                time.sleep(2)
        code=child.wait();child=None;fcntl.flock(lock,fcntl.LOCK_UN)
        if stop.is_set():return 130
        if code==0:return 0
        if not yield_requested:return code
        yields+=1
        atomic_json(record,{'status':'checkpoint_yielded_for_primary','resume_count':yields,
                            'resume_granularity':'full training epoch, window chunk, or bounded fragment export partition'})
        if yields>=spec.get('max_handoffs',12):return 130
        # Let the primary legacy lane acquire its GPU context before re-entry.
        stop.wait(6)
    return 130


if __name__=='__main__':raise SystemExit(main())
