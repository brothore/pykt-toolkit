"""Describe completed training rounds using previously saved diagnostic groups.

Grouping is explanatory only. It never defines training weights, expert routing
or a new candidate gate, and does not recompute predictions/AUC/bootstrap.
"""
import argparse
import datetime
import json
from pathlib import Path
import signal
import threading

import pandas as pd

from prediction_progress import atomic_json, digest


def summarize(paired, diagnostics):
    primary = paired[(paired.n >= 20) & (paired.minority >= 5)].copy()
    if primary.orirow.duplicated().any() or diagnostics.orirow.duplicated().any():
        raise ValueError('Duplicate orirow identity')
    if set(primary.orirow) != set(diagnostics.orirow):
        raise ValueError('Primary cohort differs from saved diagnostic cohort')
    table = primary.merge(diagnostics, on='orirow', validate='one_to_one')
    masks = {'stable_weak': table.stable_weak, 'stable_strong': table.stable_strong,
             'high_seed_variability': table.high_seed_variability,
             'reference_outperforms': table.reference_gap > 0,
             'stable_weak_reference_gap': table.stable_weak & (table.reference_gap > 0),
             'short': table.targets < 50, 'long': table.targets >= 400,
             'rare_kc': table.rare_kc_fraction >= .25,
             'little_relevant_history': table.mean_prefix_kc_support < 1,
             'low_correctness_posthoc': table.correctness_posthoc < .4}
    records = {}
    for name, mask in masks.items():
        frame = table[mask]
        records[name] = {'N': len(frame),
                         'baseline_macro_auc_seed42': float(frame.baseline_auc.mean()) if len(frame) else None,
                         'candidate_macro_auc_seed42': float(frame.auc.mean()) if len(frame) else None,
                         'mean_delta_auc': float(frame.delta_auc.mean()) if len(frame) else None,
                         'improved': int((frame.delta_auc > 1e-12).sum()),
                         'worsened': int((frame.delta_auc < -1e-12).sum()),
                         'worst_delta': float(frame.delta_auc.min()) if len(frame) else None,
                         'descriptive_small_group': len(frame) < 50}
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = pd.read_csv(args.diagnostics)
    families = [name for name in json.loads((args.root / 'registration.json').read_text())['families']
                if name.startswith('D')]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        completed, states = [], {}
        for family in families:
            state_path = args.root / 'jobs' / family / 'state.json'
            states[family] = json.loads(state_path.read_text()).get('status') if state_path.exists() else 'queued'
            for balance_path in sorted((args.root / family).glob('round*/balance.json')):
                paired_path = balance_path.parent / 'paired_orirow.csv'
                identity = {str(p): digest(p) for p in [balance_path, paired_path, args.diagnostics]}
                output = args.output_dir / family / balance_path.parent.name
                output.mkdir(parents=True, exist_ok=True)
                target = output / 'groups.json'
                if target.exists():
                    if json.loads(target.read_text())['input_sha256'] != identity:
                        raise ValueError('Previously analyzed inputs changed')
                else:
                    balance = json.loads(balance_path.read_text())
                    groups = summarize(pd.read_csv(paired_path), diagnostics)
                    atomic_json(target, {'family': family, 'protocol': balance['protocol'],
                                'training': balance.get('training'), 'input_sha256': identity,
                                'groups': groups, 'scope': 'Posthoc validation explanation, not causal evidence or training support',
                                'new_inference_or_training': False, 'test_evaluated': False})
                    lines = [f'# {family} {balance_path.parent.name} 分组复盘', '',
                             'nips_task34 / fold0 / seed42；固定 primary_20_5；使用既有诊断分组。',
                             'Δ是同一seed42基线的配对差；分组标签参考旧基线多seed诊断，不重新分组。', '',
                             '| 分组 | N | 平均ΔAUC | 退化数 | 最差ΔAUC |', '|---|---:|---:|---:|---:|']
                    for name, group in groups.items():
                        delta = f"{group['mean_delta_auc']:+.8f}" if group['N'] else 'NA'
                        worst = f"{group['worst_delta']:+.8f}" if group['N'] else 'NA'
                        lines.append(f"| {name} | {group['N']} | {delta} | {group['worsened']} | {worst} |")
                    lines.extend(['', 'N<50仅描述，不能当作训练群体支持或收益保证。',
                                  '稳定弱、正确率和参照差距等为事后解释，禁止据此给推理UID路由或训练加权。',
                                  '不代替原始标准差、极差、逐序列保护及正式 windowauclate_mean。'])
                    (output / 'review.md').write_text('\n'.join(lines) + '\n')
                completed.append(str(target))
        status = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'states': states, 'completed_reports': completed}
        atomic_json(args.output_dir / 'index.json', status)
        if args.once or all(value in {'completed', 'failed', 'stopped'} for value in states.values()):
            if not args.once:
                atomic_json(args.output_dir / 'summary.json', status)
            return 0
        stop.wait(10)
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
