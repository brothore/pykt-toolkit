"""Run a fixed command after successful dependencies, or record a signal skip."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import threading
from cooperative_process import run_cooperative
from prediction_progress import atomic_json, digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.spec.read_text())
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    while not stop.is_set():
        ready=True
        for value in spec['dependencies']:
            path=Path(value);state=json.loads(path.read_text()) if path.exists() else {'status':'queued'}
            if state['status'] in {'failed','stopped'}:raise RuntimeError(f'Unsuccessful prerequisite: {path}')
            ready=ready and state['status']=='completed'
        if ready:break
        stop.wait(10)
    if stop.is_set():return 130
    signals=[]
    for value in spec.get('signal_roots',[]):
        for path in Path(value).glob('*/round*/balance.json'):
            if json.loads(path.read_text()).get('substantive_development_signal'):
                signals.append({'source':str(path),'sha256':digest(path)})
    if signals:
        atomic_json(Path(spec['result']),{'status':'skipped_existing_development_signal','executed':False,
                    'signals':signals,'reason':'Registered next batch is unnecessary once upstream has a substantive signal.'})
        return 0
    def run():
        if stop.is_set():return 130
        return run_cooperative(spec['command'],cwd=spec['cwd'],check=False).returncode
    if spec.get('gpu_lock'):
        with Path(spec['gpu_lock']).open('a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if stop.wait(5):return 130
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip():
                if stop.wait(5):return 130
            return run()
    return run()


if __name__=='__main__':raise SystemExit(main())
