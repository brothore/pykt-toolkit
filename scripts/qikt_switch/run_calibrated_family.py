"""Use completed matched-workload evidence to bound a new experiment group."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
from cooperative_process import run_cooperative
from container_memory_budget import container_budget
from prediction_progress import atomic_json,digest



def effective_slots(recommendations,spec):
    slots=min(recommendations[k]['recommended_concurrency'] for k in ['training','inference'])
    cap=spec.get('shared_slot_limit')
    if cap is not None:
        if cap not in [1,2,3]:raise ValueError('Invalid shared slot limit')
        peers=spec.get('shared_peer_states',[])
        terminal=bool(peers) and all(Path(p).exists() and json.loads(Path(p).read_text()).get('status') in ['completed','failed','stopped'] for p in peers)
        if not terminal:slots=min(slots,cap)
    return slots


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    state=Path(spec['calibration_state']);result=Path(spec['calibration_result']);record=Path(spec['admission_record'])
    while not stop.is_set():
        value=json.loads(state.read_text()) if state.exists() else {'status':'queued'}
        if value['status'] in {'failed','stopped'}:raise RuntimeError('Calibration requires review, no fallback assumption')
        if value['status']=='completed' and result.exists():break
        atomic_json(record,{'status':'waiting_for_matched_calibration'});stop.wait(10)
    if stop.is_set():return 130
    signals=[]
    for root in map(Path,spec['signal_roots']):
        for p in root.glob('*/round*/balance.json'):
            if json.loads(p.read_text()).get('substantive_development_signal'):signals.append({'source':str(p),'sha256':digest(p)})
    if signals:
        atomic_json(Path(spec['result']),{'status':'skipped_existing_development_signal','executed':False,'signals':signals});return 0
    calibration=json.loads(result.read_text())
    if calibration.get('status')=='skipped_source_selection':
        atomic_json(Path(spec['result']),{'status':'skipped_source_selection','executed':False,'calibration':str(result)})
        return 0
    recommendations=calibration['cases'][spec['case']]['recommendations']
    slots=effective_slots(recommendations,spec)
    if slots not in {1,2,3}:raise ValueError('Unexpected calibrated slot count')
    atomic_json(record,{'status':'waiting_for_calibrated_slot','slots':slots,'shared_slot_limit':spec.get('shared_slot_limit')})
    directory=Path(spec['slot_directory']);directory.mkdir(parents=True,exist_ok=True);held=None
    while not stop.is_set():
        slots=effective_slots(recommendations,spec)
        for i in range(slots):
            handle=(directory/f'slot{i}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);held=handle;break
            except BlockingIOError:handle.close()
        if held:break
        stop.wait(3)
    if stop.is_set():return 130
    with held:
        while not stop.is_set():
            memory=container_budget();raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
            unknown=[]
            for pid in raw.splitlines():
                if not pid.strip().isdigit():continue
                try:cmd=Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode()
                except FileNotFoundError:unknown.append(pid);continue
                if spec['runtime_prefix'] not in cmd:unknown.append(pid)
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            ready=not unknown and free>=6144 and memory['conservative_available_gib']>=6
            atomic_json(record,{'status':'calibrated_slot_acquired' if ready else 'waiting_resource_budget','slots':slots,
                'calibration_sha256':digest(result),'recommendations':recommendations,'shared_slot_limit':spec.get('shared_slot_limit'),'shared_peer_states':spec.get('shared_peer_states',[]),'unknown_gpu_pids':unknown,**memory})
            if ready:break
            stop.wait(5)
        if stop.is_set():return 130
        return run_cooperative(spec['command'],cwd=spec['cwd'],check=False).returncode


if __name__=='__main__':raise SystemExit(main())
