"""Review a finite, frozen replication batch after all dependencies terminate."""
import argparse
import json
from pathlib import Path
import signal
import threading

from prediction_progress import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        states = {}
        for name in protocol['dependencies']:
            path = Path(name)
            states[name] = json.loads(path.read_text()).get('status') if path.exists() else 'pending'
        atomic_json(args.output_dir / 'progress.json', {'dependencies': states})
        if any(value in {'failed', 'stopped'} for value in states.values()):
            raise RuntimeError('Replication dependency failed/stopped; review before continuing')
        if all(value == 'completed' for value in states.values()):
            break
        stop.wait(10)
    if stop.is_set():
        return 130
    records = []
    lines = ['# V2 fixed replication review', '',
             'Validation only. Formal windowauclate_mean is unavailable for these runs.', '',
             '| Run | Epochs completed/selected | Δstd | Δrange | Δpooled validation AUC | Worsened | Candidate pass |',
             '|---|---:|---:|---:|---:|---:|---|']
    for specification in protocol['results']:
        path = Path(specification['path'])
        data = json.loads(path.read_text())
        balance = data['balance'] if 'balance' in data else data
        training = balance.get('training', {})
        record = {'id': specification['id'], 'dataset': 'nips_task34', 'fold': 0,
                  'model': 'qikt_mamba', 'seed': specification['seed'],
                  'source': str(path), 'sha256': digest(path),
                  'completed_epochs': training.get('completed_epochs'),
                  'selected_epoch': training.get('selected_epoch'),
                  'metrics': balance, 'formal_windowauclate_mean': None}
        records.append(record)
        delta = balance['deltas']
        epochs = f"{record['completed_epochs']}/{record['selected_epoch']}" if training else 'fixed prediction mixture'
        lines.append(f"| {record['id']} | {epochs} | {delta['std_ddof0']:+.8f} | {delta['range']:+.8f} | "
                     f"{delta['pooled_auc']:+.8f} | {balance['worsened']}/{balance['N']} | {balance['exploratory_candidate_pass']} |")
    groups = {}
    for record in records:
        family = record['id'].split('/')[0]
        groups.setdefault(family, []).append(record['metrics']['exploratory_candidate_pass'])
    passes = {name: len(values) == 3 and all(values) for name, values in groups.items()}
    lines.extend(['', 'All three seeds must pass the exploratory point criteria to support a replication candidate.',
                  'Passing points alone does not prove uncertainty bounds or individual no-regression protection.',
                  'No automatic Peiyou transfer or test inference.'])
    atomic_json(args.output_dir / 'summary.json', {'runs': records,
                'three_seed_exploratory_pass': passes, 'test_evaluated': False,
                'automatic_transfer': False})
    (args.output_dir / 'review.md').write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
