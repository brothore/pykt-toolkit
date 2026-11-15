"""Write immutable per-round reviews from completed discovery evidence.

No fitting, inference or metric recomputation. The next action is the already
registered bounded rule, rather than a retrospective parameter recommendation.
"""
import argparse
import datetime
import json
from pathlib import Path
import signal
import threading

from prediction_progress import atomic_json, digest


def review(balance, design, number):
    candidate = balance['exploratory_candidate_pass']
    if candidate:
        action = '初步候选已记录，暂停本方向；不追加轮次、seed或数据集复验。'
    elif number == len(design['rounds']):
        action = '本方向达到预登记轮数上限，未通过初步筛选，停止本方向。'
    else:
        action = '未通过初步筛选，继续已登记下一轮：' + json.dumps(
            design['rounds'][number], ensure_ascii=False, sort_keys=True)
    labels = {'std_reduced': '标准差未下降', 'range_reduced': '极差未下降',
              'pooled_preserved': 'pooled验证AUC下降', 'macro_preserved': '序列宏AUC下降',
              'fixed_weak_preserved': '固定弱尾部均值下降', 'own_tail_preserved': '自身尾部均值下降',
              'strong_preserved': '固定强组均值下降'}
    failed = [labels.get(key, key) for key, passed in balance['checks'].items() if not passed]
    d = balance['deltas']
    training = balance.get('training')
    lines = [f"# {design['family']} 第{number}轮复盘", '',
             'nips_task34 / QIKT-Mamba / fold0 / seed42；仅验证集，primary_20_5。',
             '正式 windowauclate_mean 未计算，不以验证AUC替代。', '']
    if training:
        lines.append(f"实际训练 {training['completed_epochs']} epoch，原生选中第 {training['selected_epoch']} epoch。")
    else:
        lines.append('固定预测融合，无新增训练；组件训练元信息见各组件原始记录。')
    lines.extend(['', '| 指标 | 变化量 |', '|---|---:|'])
    for key in ['std_ddof0', 'range', 'pooled_auc', 'macro_auc',
                'fixed_baseline_bottom10_mean', 'own_bottom10_mean',
                'fixed_baseline_top25_mean', 'range_without_min_and_max']:
        lines.append(f"| {key} | {d[key]:+.10f} |")
    lines.extend(['', f"退化 {balance['worsened']}/{balance['N']} 条；最差变化 {balance['worst_delta']:+.10f}。",
                  '未通过项：' + ('；'.join(failed) if failed else '本轮点估计全部通过。'),
                  '下一步：' + action, '',
                  '解释边界：未通过项是观察结果，不等同于已证明其因果机制。',
                  '即使通过初步条件，仍可能有个体退化、区间跨0及重复验证选择偏差；不宣称稳定或严格目标完成。',
                  '配对区间：' + json.dumps(balance['paired_orirow_bootstrap_ci95'], ensure_ascii=False)])
    return '\n'.join(lines) + '\n', action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.root = [root.resolve() for root in args.root]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        states, records = {}, []
        for root in args.root:
            registration = json.loads((root / 'registration.json').read_text())
            for family in registration['families']:
                path = root / 'jobs' / family / 'state.json'
                state = json.loads(path.read_text()) if path.exists() else {'status': 'queued'}
                states[f'{root.name}/{family}'] = state.get('status')
                protocol = root / 'jobs' / family / 'protocol.json'
                design = json.loads(protocol.read_text())
                for source in sorted((root / family).glob('round*/balance.json')):
                    number = int(source.parent.name.removeprefix('round'))
                    identity = {str(p): digest(p) for p in [protocol, source]}
                    folder = args.output_dir / root.name / family / source.parent.name
                    folder.mkdir(parents=True, exist_ok=True)
                    manifest = folder / 'source.json'
                    if manifest.exists() and json.loads(manifest.read_text()) != identity:
                        raise ValueError(f'Previously reviewed evidence changed: {source}')
                    balance = json.loads(source.read_text())
                    text, action = review(balance, design, number)
                    if not manifest.exists():
                        (folder / 'review.md').write_text(text)
                        atomic_json(manifest, identity)
                    records.append({'family': family, 'round': number,
                                    'review': str(folder / 'review.md'), 'next': action,
                                    'candidate': balance['exploratory_candidate_pass']})
        status = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'states': states, 'reviews': records}
        atomic_json(args.output_dir / 'index.json', status)
        if args.once or all(s in {'completed', 'failed', 'stopped'} for s in states.values()):
            if not args.once:
                atomic_json(args.output_dir / 'summary.json', status)
            return 0
        stop.wait(10)
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
