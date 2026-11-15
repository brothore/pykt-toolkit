"""Persist resource and epoch progress until a finite set of jobs terminates.

Read-only observer: never signals training or changes global queue capacity.
Restarting appends samples; queued_task handles its own lock and provenance.
"""
import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import threading

from prediction_progress import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    with (args.output_dir / 'samples.jsonl').open('a', buffering=1) as log:
        while not stopped.is_set():
            sample = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'jobs': {}}
            query = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            sample['gpu_csv'] = query.stdout.strip()
            sample['gpu_query_exit'] = query.returncode
            for job in protocol['jobs']:
                path = Path(job['state'])
                state = json.loads(path.read_text()) if path.exists() else {}
                progress = {}
                for item in Path(job['work_root']).rglob('training_progress.json'):
                    progress[str(item)] = json.loads(item.read_text())
                sample['jobs'][job['id']] = {'status': state.get('status', 'pending'),
                                            'progress': progress}
            with open('/proc/meminfo') as handle:
                sample['host_available_kib'] = next(int(line.split()[1]) for line in handle
                                                   if line.startswith('MemAvailable:'))
            log.write(json.dumps(sample) + '\n')
            atomic_json(args.output_dir / 'latest.json', sample)
            if all(item['status'] in {'completed', 'failed', 'stopped'}
                   for item in sample['jobs'].values()):
                atomic_json(args.output_dir / 'summary.json', {
                    'terminal_jobs': sample['jobs'], 'last_sample': sample,
                    'samples': str(args.output_dir / 'samples.jsonl'),
                    'note': 'Busy time and epoch counts recorded; no controlled serial speedup claim.'})
                return 0
            stopped.wait(protocol.get('interval_seconds', 10))
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
