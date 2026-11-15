"""Count registered V2 discovery directions, not objective achievement.

A direction finishes screening at its first recorded candidate or its bounded
round limit. Engineering failures and unmet prerequisites are not completion.
Historical V1 work and seed replications do not inflate this denominator.
"""
import argparse
import datetime
import json
from pathlib import Path

from prediction_progress import atomic_json, digest


def completed_result(path, state_path):
    if not path.exists() or not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    expected = state.get('output_sha256', {}).get(str(path.resolve()))
    return state.get('status') == 'completed' and expected is not None and digest(path) == expected


def collect(root):
    entries = []
    for family in ['A1', 'A2', 'A3', 'B1', 'B2', 'C1']:
        path = root / family / 'family_summary.json'
        job = 'B2_retry_cfg1' if family == 'B2' else family
        complete = completed_result(path, root / 'jobs' / job / 'state.json')
        entries.append({'id': f'V2/{family}', 'status': 'screening_complete' if complete else 'incomplete',
                        'evidence': str(path), 'sha256': digest(path) if path.exists() else None})
    path = root / 'C2/summary.json'
    audit = json.loads(path.read_text()).get('learned_gate_audit', {}) if path.exists() else {}
    entries.append({'id': 'V2/C2', 'status': 'prerequisite_unmet',
                    'reason': audit.get('status', 'No verified learned-gate execution evidence'),
                    'fixed_mixtures_complete': path.exists(), 'evidence': str(path),
                    'sha256': digest(path) if path.exists() else None})
    for name in ['discovery07', 'discovery08']:
        batch = root / name
        registration = batch / 'registration.json'
        for family in json.loads(registration.read_text())['families']:
            result = batch / family / 'summary.json'
            state_path = batch / 'jobs' / family / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            summary = json.loads(result.read_text()) if result.exists() else None
            completed = summary is not None and completed_result(result, state_path)
            entries.append({'id': f'{name}/{family}',
                            'status': 'screening_complete' if completed else state.get('status', 'queued'),
                            'completed_rounds': len(list((batch / family).glob('round*/balance.json'))),
                            'candidate_recorded': bool(summary and summary['candidate_found']),
                            'evidence': str(result), 'sha256': digest(result) if result.exists() else None})
    completed = sum(item['status'] == 'screening_complete' for item in entries)
    return {'as_of_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'V2 seven directions plus registered discovery07/08 extensions; excludes historical V1 and seed replications',
            'completed_directions': completed, 'registered_directions': len(entries),
            'screening_percent': 100 * completed / len(entries), 'directions': entries,
            'warning': 'Attempt progress, not problem solved. Preliminary candidates do not prove individual no-regression or formal windowauclate_mean.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('logs/qikt_v2_20261105'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = collect(args.root)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
