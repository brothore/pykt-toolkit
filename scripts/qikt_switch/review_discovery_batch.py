"""Read-only final audit and self-contained report for a registered batch."""
import argparse
import datetime
import json
from pathlib import Path
import signal
import threading

import numpy as np
import pandas as pd

from prediction_progress import atomic_json, digest


def read(path):
    return json.loads(path.read_text())


def job_status(root, family):
    path = root / 'jobs' / family / 'state.json'
    # A valid tsp job can be queued before its worker creates the state file.
    return read(path)['status'] if path.exists() else 'queued'


def audit_round(folder):
    value = read(folder / 'balance.json')
    frame = pd.read_csv(folder / 'paired_orirow.csv')
    frame = frame[(frame.n >= 20) & (frame.minority >= 5)].sort_values('orirow')
    if len(frame) != value['N'] or frame.orirow.duplicated().any():
        raise ValueError(f'Primary cohort mismatch: {folder}')
    x = frame.baseline_auc.to_numpy()
    y = frame.auc.to_numpy()
    gain = frame.delta_auc.to_numpy()
    if not np.allclose(x + gain, y, rtol=0, atol=1e-12):
        raise ValueError(f'Paired gain mismatch: {folder}')
    for values, key in [(x, 'baseline'), (y, 'candidate')]:
        actual = dict(macro_auc=float(values.mean()), std_ddof0=float(values.std()),
                      range=float(np.ptp(values)))
        for metric, number in actual.items():
            if abs(number - value[key][metric]) > 1e-12:
                raise ValueError(f'Metric mismatch: {folder}: {key}/{metric}')
    if int((gain < -1e-12).sum()) != value['worsened']:
        raise ValueError(f'Regression count mismatch: {folder}')
    covariance = float(2 * np.mean((x - x.mean()) * (gain - gain.mean())))
    gain_variance = float(np.var(gain))
    variance_delta = float(np.var(y) - np.var(x))
    if abs(covariance + gain_variance - variance_delta) > 1e-15:
        raise ValueError(f'Variance identity mismatch: {folder}')
    order = np.argsort(x, kind='stable')
    weak10 = order[:int(np.ceil(len(x) * .1))]
    weak20 = order[:int(np.ceil(len(x) * .2))]
    strong25 = order[-int(np.ceil(len(x) * .25)):]
    fixed = {name: {'baseline': float(x[index].mean()), 'candidate': float(y[index].mean())}
             for name, index in [('weak10', weak10), ('weak20', weak20), ('strong25', strong25)]}
    return value, {'twice_covariance': covariance, 'gain_variance': gain_variance,
                   'variance_delta': variance_delta, 'fixed_groups': fixed,
                   'paired_sha256': digest(folder / 'paired_orirow.csv')}


def build(root, registration, states):
    identities = {}
    records = []
    controls = set(registration.get('control_families', []))
    for family in registration['families']:
        job = root / 'jobs' / family
        plan = read(job / 'plan.json')
        state = read(job / 'state.json')
        for category in ['input_sha256', 'source_sha256']:
            for path, checksum in plan[category].items():
                if path in identities and identities[path] != checksum:
                    raise ValueError(f'Conflicting pinned identity: {path}')
                identities[path] = checksum
        identities.update(state.get('output_sha256', {}))
        for path in sorted((root / family).glob('round*/balance.json')):
            value, audit = audit_round(path.parent)
            records.append({'family': family, 'round': path.parent.name, 'value': value,
                            'audit': audit, 'source': str(path), 'source_sha256': digest(path)})
            if value.get('training'):
                window=path.parent/'baseline/validation_window/validation_window_results.json'
                if not window.exists():raise ValueError(f'Completed training result lacks window identity: {path}')
                selected=Path(value['training']['checkpoint_dir'])/'attn_model.ckpt'
                identities[str(selected)]=read(window)['checkpoint_sha256']
                identities[str(window)]=digest(window)
    for path, checksum in identities.items():
        if digest(Path(path)) != checksum:
            raise ValueError(f'Frozen source/input/output changed: {path}')
    if not records:
        summaries = {family: read(root / family / 'summary.json') for family in registration['families']}
        if not all(value.get('status') == 'skipped_existing_development_signal' for value in summaries.values()):
            raise ValueError('No completed comparisons, and not a registered signal skip')
        out = root / 'completed_review'
        out.mkdir(exist_ok=True)
        atomic_json(out / 'audit.json', {'states': states, 'verified_unique_files': len(identities),
                    'completed_rounds': 0, 'status': 'skipped_existing_development_signal',
                    'skip_evidence': summaries, 'goal_achieved': False,
                    'note': 'Upstream development signal does not imply stable or formal-test success.'})
        (out / registration.get('completed_report_filename', 'V5_完整完成复盘.md')).write_text(
            '# ' + registration.get('completed_report_title', '有条件研究批次') + '\n\n'
            '上游已出现登记的实质开发信号，本批按规则全部跳过，没有新训练、推理或结果表。'
            '来源和校验见audit.json；开发信号仍不等于正式测试或稳定性验证完成。\n')
        return
    baseline = records[0]['value']['baseline']
    for record in records:
        if record['value']['baseline'] != baseline:
            raise ValueError('Baseline metrics differ between rounds')
    out = root / 'completed_review'
    out.mkdir(exist_ok=True)
    candidates = [r for r in records if r['family'] not in controls and r['value']['exploratory_candidate_pass']]
    signals = [r for r in records if r['family'] not in controls and r['value'].get('substantive_development_signal', False)]
    audit = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'states': states, 'verified_unique_files': len(identities), 'completed_rounds': len(records),
             'optimization_candidates': len(candidates), 'substantive_signals': len(signals),
             'records': records, 'test_evaluated': False, 'formal_windowauclate_mean': None,
             'goal_achieved': False, 'note': 'Candidate status is development evidence, not stable or strict no-regression proof.'}
    atomic_json(out / 'audit.json', audit)
    lines = ['# ' + registration.get('completed_report_title', 'V5：完整结果、双重对照与可核查复盘'), '',
             f"本批已终止等待：共取得{len(records)}条完成结果，优化点候选{len(candidates)}条，实质开发信号{len(signals)}条。任务失败/停止会明确保留，不能算完成实验。", '',
             registration.get('evaluation_description', 'nips_task34，QIKT-Mamba/public_mamba，fold0，seed42。主统计子集为至少20个计分位置、至少5个少数类位置的773条orirow；标准差ddof=0。均为开发验证结果，正式windowauclate_mean缺失，未进行新的seed/fold/Peiyou复验。'), '',
             registration.get('mechanism_description', '原基线训练14 epoch、选中4。G0仅复用原sampler/generator并添加片段索引，不加入排序；G1为完整训练序列logistic风险的epoch梯度缓存，G2换为平滑排序风险，G3再按训练类别支持度加权。各方向保留原生BCE、选模和早停。缓存是epoch-lag近似，补齐训练覆盖内跨片段配对，不给模型增加历史输入。'), '',
             registration.get('implementation_limitations', 'G0的微小点改善不算优化候选。初版G1–G3因原生UID为Python列表而在优化前失败，修复版接受列表且保留失败记录；从同seed重新初始化，不称断点续训。G0与原基线权重并非逐位相同，差异来源未独立归因。'), '']

    def table(headers, rows, directions=None):
        best = {j: (min if direction == 'min' else max)(row[j] for row in rows)
                for j, direction in (directions or {}).items()}
        lines.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        for row in rows:
            rendered = []
            for j, value in enumerate(row):
                text = f'{value:.12g}' if isinstance(value, float) else str(value)
                if j in best and value == best[j]:
                    text = '**' + text + '**'
                rendered.append(text)
            lines.append('| ' + ' | '.join(rendered) + ' |')
        lines.append('')

    names = registration.get('family_names', {})
    first = records[0]['value']
    baseline_epochs='14/4'
    for record in records:
        v=record['value']
        if record['family'] in controls and v['baseline']==v['candidate'] and v.get('training'):
            t=v['training'];baseline_epochs=f"{t['completed_epochs']}/{t['selected_epoch']}";break
    rows = [['Baseline', '原始模型', baseline_epochs, first['comparison']['baseline_validation_window_auc'],
             baseline['macro_auc'], baseline['std_ddof0'], baseline['range'], 0, '参照', '参照']]
    for record in records:
        v = record['value']; c = v['candidate']; p = v['protocol']; t = v.get('training', {})
        settings = p.get('model_settings', {})
        coefficients = ', '.join(f'{key}={settings[key]}' for key in
            ['learning_rate', 'causal_adapter_scale', 'causal_adapter_kl',
             'retrieval_adapter_scale', 'retrieval_adapter_kl', 'version',
             'ukt_transfer_mode','ukt_transfer_scale','ukt_transfer_fusion','ukt_transfer_freeze',
             'ukt_rasch_mode','ukt_rasch_scale','ukt_rasch_freeze','ukt_rasch_warmstart',
             'reference_fusion_mode','reference_fusion_scale','reference_fusion_bias','reference_cic_kernel','reference_cic_context','reference_history_weight'] if key in settings)
        for key in ['baseline_anchor_kl','baseline_pair_weight']:
            if key in p:coefficients+=f", {key}={p[key]}"
        if not coefficients:
            if 'consensus_mode' in p:
                coefficients=f"mode={p['consensus_mode']}, amount={p['consensus_amount']}"
            elif 'strength' in p:
                coefficients=f"strength={p['strength']}, mode={p['transfer_mode']}"
            elif 'prior' in p:
                coefficients=f"prior={p['prior']}, radius={p['radius']}, memory={p.get('memory', 200)}"
            else:
                coefficients = (f"candidate_weight={p['candidate_weight']}" if 'candidate_weight' in p
                                else f"rank={p.get('rank_weight', 0)}")
        if p.get('alpha'):
            coefficients += f", sequence_mix={p['alpha']}"
        description = names.get(record['family'], record['family']) + '；' + coefficients
        label = record['family'] + '/' + record['round']
        point = '控制，非优化候选' if record['family'] in controls else str(v['exploratory_candidate_pass'])
        epochs = f"{t.get('completed_epochs')}/{t.get('selected_epoch')}" if t else '未新增训练，组件见协议'
        rows.append([label, description, epochs,
                     v['comparison']['candidate_validation_window_auc'], c['macro_auc'], c['std_ddof0'],
                     c['range'], v['worsened'], point, v.get('substantive_development_signal', False)])
    lines.extend(['## 完整主指标', '', '每列最佳数值加粗；某列最优不等于联合目标通过。退化数相对原baseline计算。', ''])
    table(['方案', '改动', '实际/选中epoch', 'pooled AUC ↑', 'macro AUC ↑', 'std ↓', 'range ↓',
           '退化序列数 ↓', '点候选', '实质开发信号'], rows,
          {3: 'max', 4: 'max', 5: 'min', 6: 'min', 7: 'min'})
    lines.extend(['## 尾部、原有强项与分布宽度', '',
                  '固定弱/强组由原baseline主子集AUC稳定排序确定；自身尾部允许成员变化，两者不能混用。', ''])
    fixed_base = records[0]['audit']['fixed_groups']
    rows = [['Baseline', '原始模型', baseline['own_bottom10_mean'], baseline['own_bottom20_mean'],
             fixed_base['weak10']['baseline'], fixed_base['weak20']['baseline'], fixed_base['strong25']['baseline'],
             baseline['q95_minus_q05'], baseline['range_without_min_and_max']]]
    for record in records:
        c = record['value']['candidate']; fixed = record['audit']['fixed_groups']
        rows.append([record['family'] + '/' + record['round'], names.get(record['family'], ''),
                     c['own_bottom10_mean'], c['own_bottom20_mean'], fixed['weak10']['candidate'],
                     fixed['weak20']['candidate'], fixed['strong25']['candidate'],
                     c['q95_minus_q05'], c['range_without_min_and_max']])
    table(['方案', '改动', '自身尾10% ↑', '自身尾20% ↑', '固定弱10% ↑', '固定弱20% ↑',
           '固定强25% ↑', 'P95−P05 ↓', '去掉两端极差 ↓'], rows,
          {2: 'max', 3: 'max', 4: 'max', 5: 'max', 6: 'max', 7: 'min', 8: 'min'})
    lines.extend(['## 不确定性与退化', '',
                  '下列区间是相对原baseline的orirow配对重采样95%区间，不能代表独立学生或跨seed稳定性；极差受两个端点影响。', ''])
    rows = [['Baseline', '原始模型', 0., 0., '[0, 0]', 0, 0, 0, 0.]]
    for record in records:
        v = record['value']; d = v['deltas']; ci = v['paired_orirow_bootstrap_ci95']['std_delta']
        rows.append([record['family'] + '/' + record['round'], names.get(record['family'], ''),
                     d['std_ddof0'], d['range'], str(ci), v['decline_counts']['0.001'],
                     v['decline_counts']['0.005'], v['decline_counts']['0.01'], v['worst_delta']])
    table(['方案', '改动', 'Δstd ↓', 'Δrange ↓', 'Δstd区间', '下降>.001数 ↓', '下降>.005数 ↓',
           '下降>.01数 ↓', '最差序列增益 ↑'], rows,
          {2: 'min', 3: 'min', 5: 'min', 6: 'min', 7: 'min', 8: 'max'})
    control_report = root / 'V5_原基线与控制的双重对照.md'
    if control_report.exists():
        lines.extend(['## 同时相对零排序控制比较', '', control_report.read_text(), ''])
    lines.extend(['## 如何判断本批的证据', '',
                  '- 标准差下降需要弱序列增益与原AUC的负协方差足够大，抵消增益自身方差；逐轮两项分解保存在audit.json，恒等式已核对。极差还要求最差端点获益，不能用平均尾部改善替代。',
                  registration.get('attribution_note', '- G1–G3与G0保持采样包装一致，用于减少V4整序列分组带来的混杂。单次训练、epoch缓存、CUDA数值轨迹仍有限制，不能仅凭组合结果断言某模块具有普遍因果收益。'),
                  '- 实质门槛为std与range各相对下降至少1%、std区间上界<0，并保住pooled/macro、固定弱10%/20%、自身尾10%/20%及固定强25%均值。即使通过，逐序列无退化和正式测试仍需另外验证。',
                  registration.get('next_decision_note', '- 有初步候选的方向记录后暂停；没有信号时结束这批有界排序分支，再依据复盘设计新的信息机制，不继续无限调整同一排序系数。'), '',
                  '## 任务、参数和证据入口', ''])
    table(['方向', '改动', '任务状态'], [[f, names.get(f, ''), states[f]] for f in registration['families']])
    for record in records:
        v = record['value']; folder = Path(record['source']).parent
        lines.extend([f"- {record['family']}/{record['round']}：[完整协议]({folder / 'protocol.json'})；"
                      f"[完整指标]({folder / 'balance.json'})；[配对序列]({folder / 'paired_orirow.csv'})。参数："
                      + json.dumps(v['protocol'], ensure_ascii=False), ''])
    lines.extend([f"核对{len(identities)}个独立冻结来源/输入/产物，逐轮数值与退化数、方差恒等式均核验；[审计及分解]({out / 'audit.json'})。仅分析已有预测，不新增训练、推理或测试标签开发。", ''])
    temporary = out / 'V5_完整完成复盘.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(out / registration.get('completed_report_filename', 'V5_完整完成复盘.md'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    registration = read(args.root / 'registration.json')
    if args.context:
        context = read(args.context)
        allowed = {'completed_report_title', 'completed_report_filename', 'mechanism_description',
                   'implementation_limitations', 'attribution_note', 'next_decision_note', 'evaluation_description'}
        if set(context) - allowed:
            raise ValueError('Report context must not override scientific registration')
        registration.update(context)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        states = {family: job_status(args.root, family)
                  for family in registration['families']}
        control_job = args.root / 'jobs/control_comparisons/state.json'
        control_status = read(control_job)['status'] if control_job.exists() else None
        terminal = all(s in {'completed', 'failed', 'stopped'} for s in states.values())
        if terminal and control_status in {None, 'completed', 'failed', 'stopped'}:
            if control_status not in {None, 'completed'}:
                raise ValueError(f'Control comparison task did not succeed: {control_status}')
            build(args.root, registration, states)
            return 0
        if not args.wait:
            raise ValueError('Batch not terminal; use --wait for queue execution')
        stop.wait(15)
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
