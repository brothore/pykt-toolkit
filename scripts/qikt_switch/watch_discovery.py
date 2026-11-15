"""Maintain a live candidate ledger, failure records and resource samples."""
import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import threading

from prediction_progress import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    out = args.root / 'ledger'
    out.mkdir(exist_ok=True)
    registration=json.loads((args.root / 'registration.json').read_text())
    families = registration['families']
    controls=set(registration.get('control_families',[]))
    with (out / 'resources.jsonl').open('a', buffering=1) as log:
        while not stop.is_set():
            states, rounds, candidates = {}, [], []
            for family in families:
                path = args.root / 'jobs' / family / 'state.json'
                state = json.loads(path.read_text()) if path.exists() else {'status': 'queued'}
                states[family] = {key: state[key] for key in ['status', 'error', 'exit_code'] if key in state}
                for balance_path in sorted((args.root / family).glob('round*/balance.json')):
                    balance = json.loads(balance_path.read_text())
                    record = {'family': family, 'round': balance_path.parent.name,
                              'source': str(balance_path), 'sha256': digest(balance_path),
                              'deltas': balance['deltas'], 'worsened': balance['worsened'],
                              'N': balance['N'], 'candidate': balance['exploratory_candidate_pass'] and family not in controls,
                              'control':family in controls,'raw_point_conditions_pass':balance['exploratory_candidate_pass'],
                              'training': balance.get('training'), 'test_evaluated': False,
                              'stability_verified': False, 'formal_windowauclate_mean': None}
                    rounds.append(record)
                    if record['candidate']:
                        candidates.append(record)
            sample = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'states': states}
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            sample['gpu_csv'] = result.stdout.strip()
            log.write(json.dumps(sample) + '\n')
            ledger = {'updated_utc': sample['utc'], 'states': states, 'rounds': rounds,
                      'candidates': candidates, 'automatic_stability_validation': False}
            atomic_json(out / 'candidates.json', ledger)
            if all(state['status'] in {'completed', 'failed', 'stopped'} for state in states.values()):
                atomic_json(out / 'summary.json', ledger)
                return 0
            stop.wait(10)
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
