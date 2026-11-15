"""Audit completed discovery caches without loading models or touching CUDA."""
import argparse
import json
import math
from pathlib import Path
import signal
import threading

from prediction_progress import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads((args.root / 'registration.json').read_text())
    families = [f for f in registration['families'] if f not in registration.get('control_families', [])]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    prerequisites = families + ['final_review']
    while not stop.is_set():
        ready = True
        for family in prerequisites:
            state = args.root / 'jobs' / family / 'state.json'
            value = json.loads(state.read_text()) if state.exists() else {'status': 'queued'}
            if value['status'] in {'failed', 'stopped'}:
                raise RuntimeError(f'Unsuccessful prerequisite: {family}')
            ready = ready and value['status'] == 'completed'
        if ready:
            break
        stop.wait(10)
    if stop.is_set():
        return 130
    records = []
    for family in families:
        for path in sorted((args.root / family).glob('round*/balance.json')):
            value = json.loads(path.read_text())
            protocol = value['protocol']
            if not protocol.get('epoch_rank_cache'):
                continue
            directory = Path(protocol['gradient_cache_directory'])
            entries = sorted(directory.glob('epoch_[0-9][0-9][0-9].json'))
            expected_epochs = value['training']['completed_epochs']
            if [p.stem for p in entries] != [f'epoch_{i:03d}' for i in range(1, expected_epochs + 1)]:
                raise ValueError(f'Completed epoch cache coverage differs: {path}')
            epochs = []
            for entry in entries:
                if stop.is_set():
                    return 130
                item = json.loads(entry.read_text())
                metadata = item['identity']['metadata']
                expected = dict(risk_focus_mode=protocol.get('cached_rank_focus_mode', 'off'),
                                risk_focus_strength=protocol.get('cached_rank_focus_strength', 0.))
                if any(metadata.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'Focus protocol differs: {entry}')
                weights = item['risk_focus_weights']
                risks = item['weighted_UID_risks']
                if set(weights) != set(risks) or len(weights) != metadata['eligible_UIDs']:
                    raise ValueError(f'Eligible UID coverage differs: {entry}')
                if any(not math.isfinite(w) or not .5 <= w <= 1.5 for w in weights.values()):
                    raise ValueError(f'Out-of-bounds focus weights: {entry}')
                if any(not math.isfinite(r) or r < 0 for r in risks.values()):
                    raise ValueError(f'Invalid weighted risks: {entry}')
                cache = entry.with_suffix('.pt')
                if digest(cache) != item['cache_sha256']:
                    raise ValueError(f'Cache checksum differs: {cache}')
                epochs.append(dict(epoch=item['identity']['epoch'], eligible_UIDs=len(weights),
                                   focus_min=min(weights.values(), default=None),
                                   focus_max=max(weights.values(), default=None),
                                   cache_sha256=item['cache_sha256'], metadata_sha256=digest(entry)))
            records.append(dict(family=family, round=path.parent.name, epochs=epochs,
                                balance_sha256=digest(path), protocol_sha256=digest(path.parent / 'protocol.json')))
    atomic_json(args.output, dict(passed=True, records=records, completed_cached_rounds=len(records),
                scope='Completed epoch coverage, UID coverage, registered focus settings, bounds, finite risks and file checksums; not a new gradient/effectiveness test',
                goal_achieved=False, source_sha256=digest(Path(__file__))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
