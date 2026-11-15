"""Wait for explicit completed artifacts; reject failed prerequisites."""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path
import sys
import time
from cooperative_process import run_cooperative
from prediction_progress import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dependency', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--gpu-lock', type=Path)
    args = ap.parse_args(); spec = json.loads(args.dependency.read_text())
    while True:
        ready = True
        for dep in spec['artifacts']:
            artifact = Path(dep['result'])
            if artifact.exists():
                value = json.loads(artifact.read_text())
                if dep.get('require_passed') and not value.get('passed'):
                    raise RuntimeError('Completed prerequisite did not pass')
                if any(value.get(key)!=expected for key,expected in dep.get('required_values',{}).items()):
                    raise RuntimeError('Prerequisite result has unexpected status/identity')
                continue
            ready = False
            state = Path(dep['state'])
            if state.exists() and json.loads(state.read_text()).get('status') in ('failed','stopped','completed'):
                raise RuntimeError('Prerequisite terminated without required result')
        if ready: break
        time.sleep(5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir/'dependencies_verified.json', {dep['result']:digest(Path(dep['result'])) for dep in spec['artifacts']})
    def run():
        run_cooperative([sys.executable, str(Path(__file__).with_name('run_discovery_family.py')),
                         '--protocol', str(args.protocol), '--output-dir', str(args.output_dir)], check=True)
    if args.gpu_lock:
        args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(args.output_dir/'gpu_admission.json',{'status':'waiting_for_exclusive_slot','lock':str(args.gpu_lock)})
        with args.gpu_lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            while True:
                active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                if not active:break
                atomic_json(args.output_dir/'gpu_admission.json',{'status':'waiting_for_external_gpu_work','observed_pids':active,'lock':str(args.gpu_lock)})
                time.sleep(5)
            atomic_json(args.output_dir/'gpu_admission.json',{'status':'exclusive_slot_acquired','lock':str(args.gpu_lock)})
            try:run()
            finally:atomic_json(args.output_dir/'gpu_admission.json',{'status':'exclusive_slot_released','lock':str(args.gpu_lock)})
    else:run()


if __name__ == '__main__': main()
