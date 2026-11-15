"""Render source ablations and every completed dual-GRU integration from artifacts."""
import argparse
import json
from pathlib import Path
import time
from prediction_progress import atomic_json, digest

SOURCE_NAMES = {
    'full': '完整模型（本模型对照）',
    'mean_distance_only': 'UKT去协方差距离，只保留均值距离；其余计算保持原样',
    'no_distance_modulation': 'UKT关闭原乘性距离调制，保留W2及协方差传播',
    'linear_covariance_aggregation': 'UKT把值协方差聚合由α²改为α',
    'no_rasch': 'UKT关闭难度×KC调制；同时移除该配置的题目ID入口',
    'no_time': 'MTKT删除时间分支及当前题时间直加，内容分支权重置1',
    'no_cic': 'MTKT删除因果双卷积交互残差，保留attention与层归一化',
    'no_alibi': '删除ALiBi距离偏置，保留其余结构',
    'no_interaction_rasch': 'LEFOKT删除历史交互侧Rasch调制，保留当前查询题目条件',
}
FUSION_SENTENCES = {
    'qikt_gru_ukt_20261114': {
        'R0':'用UKT检索思路的均值距离查询严格过去响应，以有界logit补充双GRU输出。',
        'R1':'用UKT的W2距离与α²协方差传播检索过去响应，以有界logit补充双GRU输出。',
        'R2':'在UKT式W2历史检索补充分支中，按检索不确定性收缩对双GRU输出的修正。',
        'R3':'在UKT式W2历史检索中加入原乘性距离调制，再修正双GRU输出。'},
    'qikt_gru_ukt_v2_20261114': {
        'S0':'把UKT启发的标量题目难度×KC方向加入双GRU历史输入与当前解码，端到端训练。',
        'S1':'把UKT向量题目难度×KC方向加入双GRU历史输入与当前解码，端到端训练。',
        'S2':'将UKT向量Rasch用于有界输出修正，冻结双GRU，只训练新增因子。',
        'S3':'用UKT式W2检索补充输出，冻结双GRU，只训练检索分支，不加KL。'},
    'qikt_gru_ukt_v3_20261114': {
        'C0':'以题目向量＋KC增量替代UKT式乘法，加入双GRU输入和解码，作同容量加性对照。',
        'Q0':'把UKT向量Rasch只加到双GRU当前题解码的KC表示，历史输入结构保持原样。',
        'A0':'把UKT式当前题Rasch接入双GRU，从B0继续训练并以固定B0概率KL约束偏离。',
        'A1':'在当前题Rasch＋B0教师KL的双GRU上，增加训练中教师正确正负对的排序保护。'},
    'qikt_mtkt_cic_v1_20261115': {
        'M0Q':'将MTKT式CIC各卷积抽头都作用于当前题目GRU状态，作不新增跨位置访问的容量对照。',
        'M0C':'将MTKT式CIC各卷积抽头都作用于当前KC GRU状态，作不新增跨位置访问的容量对照。',
        'M0B':'在两个GRU分别加入当前状态CIC控制，名义参数量匹配双分支因果CIC。',
        'M1Q':'把MTKT因果双卷积交互作为残差加入题目GRU隐藏序列，再送入原解码器。',
        'M1C':'把MTKT因果双卷积交互作为残差加入KC GRU隐藏序列，再送入原解码器。',
        'M1B':'在题目和KC两个GRU隐藏序列分别加入独立MTKT式因果CIC残差。'},
    'qikt_lefokt_query_v1_20261115': {
        'L1Q':'借鉴LEFOKT当前题条件化，将标量题目难度×KC方向仅加到双GRU当前解码。',
        'L1H':'借鉴LEFOKT历史抽象，将题目GRU历史输入的显式题目通道乘β，保留当前题与KC/响应。',
        'L1B':'把LEFOKT启发的历史题目通道β=.5与当前题标量条件化同时接入双GRU。'},
}

FAILURES = {'std_reduced': '标准差未降', 'range_reduced': '极差未降',
    'pooled_preserved': '总体AUC下降', 'macro_preserved': '序列均值下降',
    'fixed_weak_preserved': '固定弱10%下降', 'fixed_weak20_preserved': '固定弱20%下降',
    'own_tail_preserved': '自身尾10%下降', 'own_tail20_preserved': '自身尾20%下降',
    'strong_preserved': '固定强25%下降'}


def table(headers, rows, numeric=None):
    numeric = numeric or {}
    best = {}
    for index, direction in numeric.items():
        values = [row[index] for row in rows if isinstance(row[index], (float, int))]
        if values:
            best[index] = (min if direction == 'min' else max)(values)
    result = ['| ' + ' | '.join(headers + ['相对baseline变化幅度（%）']) + ' |',
              '| ' + ' | '.join(['---'] * (len(headers) + 1)) + ' |']
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            text = f'{value:.12f}' if isinstance(value, float) else str(value)
            if index in best and value == best[index]:
                text = '**' + text + '**'
            cells.append(text.replace('|', '/').replace('\n', ' '))
        changes = []
        for index in numeric:
            value, baseline = row[index], rows[0][index]
            label = headers[index].replace(' ↑', '').replace(' ↓', '')
            if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
                change = '—'
            elif baseline == 0:
                change = '不适用（baseline为0）'
            else:
                percent = (value - baseline) / baseline * 100
                if label in ['标准差', '极差']:
                    arrow = '↑' if percent < 0 else '↓' if percent > 0 else '→'
                    change = f'{arrow}{abs(percent):.4f}%'
                else:
                    change = f'{percent:+.4f}%'
            changes.append(f'{label} {change}')
        cells.append('<br>'.join(changes))
        result.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(result) + '\n'


def render(repo, output):
    inputs = {}
    def read(path):
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())
    refs = repo / 'logs/advanced_reference_20261115'
    profile_path = refs / 'module_sequence_profiles/profiles.json'
    profiles = read(profile_path)['sections'] if profile_path.exists() else {}
    doc = '''# 三模型消融与双GRU模块融合：完整对照汇总

本文回答两个问题：原模型依赖什么机制，以及这些机制迁入我们的题目/KC双独立GRU后是否改善序列表现。每行均给出绝对指标与一句改动说明，保留全部已完成轮次，不只展示最好的一轮。列内最优加粗，**单列最好不等于联合有效**。

主指标表最后一列为相对首行baseline的变化幅度，保留四位小数，不是AUC百分点变化。前三表分别以各自完整模型为baseline，融合主表以双GRU B0为baseline；群体诊断和匹配表则按同行注明的对照计算，不跨群体混减。**标准差／极差用效果方向箭头：↑表示改善（数值缩小），↓表示变差（数值增大），→表示不变；例如标准差缩小10%写作↑10.0000%。** 箭头后的幅度为abs(当前值−baseline)/baseline×100%。其他指标仍按(当前值−baseline)/baseline×100%显示正负号；baseline为0的退化数量不计算相对百分比，未完成指标显示“—”。

所有数值均为nips_task34、fold0开发验证。总体AUC（pooled）为全部预测合并后计算；序列均值（macro）为可计分序列AUC等权平均；标准差为ddof=0；极差=max−min。主子集规则为有效目标≥20、正负少数类≥5；单类别序列不记为0.5。均值用于判断有没有普遍变差，标准差和极差用于衡量用户定义的序列不均衡。

**正式all_results.json.windowauclate_mean：下列新消融／融合均缺失。表中开发总体AUC不是正式测试指标，也不是训练指标。** 原生概念片段与严格题目窗口分清口径，不能直接跨表相减。

## 1. UKT 消融：固定权重推理干预，未删除模块重训

seed3407，原权重训练28 epoch，每行新增训练0 epoch；223341条原生概念展开目标，主子集773 UID。原生接口可能包含同题不同KC标签信息，不能当作严格题目预测。此前仅汇总总体AUC，本次用保存的47批预测和原始smasks/UID逐批核对标签后，在CPU补算了序列均值／标准差／极差；未重训或重新GPU推理。

'''
    fusion_names=['qikt_gru_ukt_20261114','qikt_gru_ukt_v2_20261114','qikt_gru_ukt_v3_20261114','qikt_mtkt_cic_v1_20261115','qikt_lefokt_query_v1_20261115','qikt_mtkt_time_v2_20261115','qikt_sequence_guided_v3_20261115','qikt_lefokt_additive_v3_20261115']
    completed_fusions=sum(1 for name in fusion_names for p in (repo/'logs'/name).glob('*/round*/balance.json') if p.parent.parent.name!='B0')
    native_count=7+len(list((repo/'logs/lefokt_extended_ablation_v2_fix01_20261115/training').glob('*/completion.json')))
    scope=f'累计已完成 **{native_count}次MTKT／LEFOKT源匹配重训、{completed_fusions}次双GRU融合／控制训练**，共享B0不重复计数。UKT源实验仍是既有权重推理干预，不冒称重训。\n\n'
    limitations='本阶段反复使用同一开发验证集筛选多种方案，存在选择偏差，未做确认性多重比较校正；配对区间不能替代跨seed或独立正式测试。时间方案沿用MTKT的当前事件时间语义：原始timestamps来自DateAnswered，当前题事件时刻作为已知输入；这里没有验证该时间在作答开始前是否可得，因此不能直接声称已验证作答前在线预测。当前响应标签及未来响应仍按严格链路隔离。标量加性LAI／LAQ不消费时间特征。\n\n'
    doc=doc.replace('所有数值均为nips_task34',scope+limitations+'所有数值均为nips_task34',1)
    ukt = read(refs / 'ukt_sensitivity_balance/summary.json')
    rows = []
    for v in ukt['results']:
        c = v['cohorts']['primary_20_5']
        rows.append([v['mode'], SOURCE_NAMES[v['mode']], '28 / 0', c['N'],
            v['native_validation_fragment_auc'], c['mean'], c['std'], c['range']])
    doc += table(['条件', '一句话改动', '历史／新增epoch', '序列数', '原生总体AUC ↑',
        '序列平均AUC ↑', '标准差 ↓', '极差 ↓'], rows,
        {4:'max', 5:'max', 6:'min', 7:'min'})
    doc += profiles.get('ukt', '')
    doc += '\n解释：去均值外的距离项或距离调制虽缩小部分离散指标，却明显降低总体和序列均值，属于可能“压低强项”的均衡，不能判有效。去Rasch还删除题目身份入口，不能把全部损失归因于乘法形式。use_CL=0，本配置没有对比学习收益证据。\n\n'
    for model, title, modes in [
        ('mtkt', '2. MTKT 消融：删除模块后匹配重训', ['full','no_time','no_cic','no_alibi']),
        ('lefokt_akt', '3. LEFOKT-AKT 消融：删除模块后匹配重训', ['full','no_alibi','no_interaction_rasch'])]:
        doc += f'## {title}\n\nseed3407；221348条严格开发题目目标、主子集773 UID。完整模型和删除模型均真实重训；排除当前题所有真实响应，每题多KC独立查询后平均。上下文最多199个过去展开KC＋当前KC。实际／选中epoch分列，不是新增epoch=0的推理干预。\n\n'
        rows = []
        extra = repo/'logs/lefokt_extended_ablation_v2_fix01_20261115'
        extra_names = {}
        if model == 'lefokt_akt' and (extra/'registration.json').exists():
            registration = read(extra/'registration.json')
            extra_names = registration['names']
            modes = modes + [m for m in registration['modes'] if (extra/'strict_questions'/m/'validation_question_results.json').exists()]
        for mode in modes:
            path = refs / f'runs/strict_questions/{model}_{mode}_strict_trained/validation_question_results.json'
            if mode not in ['full','no_time','no_cic','no_alibi','no_interaction_rasch']:
                path = extra/'strict_questions'/mode/'validation_question_results.json'
            v = read(path)
            c = v['cohorts']['primary_20_5']
            rows.append([mode, SOURCE_NAMES.get(mode, extra_names.get(mode,mode)), f"{v['completed_epochs']}/{v['selected_epoch']}",
                c['N'], v['validation_question_auc'], c['mean'], c['std'], c['range']])
        doc += table(['条件', '一句话改动', '实际／选中epoch', '序列数', '严格总体AUC ↑',
            '序列平均AUC ↑', '标准差 ↓', '极差 ↓'], rows,
            {4:'max', 5:'max', 6:'min', 7:'min'})
        doc += profiles.get(model, '')
        if model == 'mtkt':
            doc += '\n解释：加回CIC或时间路径在匹配对照上同时改善总体／均值／标准差／极差，但std配对区间跨0；这是单seed机制线索。加回ALiBi提高均值却扩大离散，不能优先作为公平性模块。先迁移CIC，完成后转向时间路径。\n\n'
        else:
            doc += '\n解释：去ALiBi缩小离散却损失总体和均值；去历史交互Rasch同时改善四项点指标，但固定原强25%仍下降约0.00010135，371条序列退化，其中55条下降>.01，未通过联合保护。因此迁移的是待验证的“当前题条件化＋历史抽象”设计，不是已证明稳健有效的模块。\n\n'
    doc += '''## 4. 与我们的双GRU结合：全部已完成轮次

方案编码U1/U2/U3分别对应UKT启发的第一／二／三批；round表示该方向的第几次独立尝试，并非稳定性seed复验。W2为同时比较均值与不确定性的Wasserstein距离；Rasch指题目因子与KC方向的乘积调制。\n\n基线B0为QIKT题目与KC两个独立GRU（version=gru，不是Mamba），seed42、训练27／选中17。221348条题目窗口预测、主子集773 orirow。GRU按200个题目保留上下文，参考模型按展开KC计，上下文预算不完全相同。以下融合间使用同一个B0及完全对齐目标；原验证选模／早停没有改。

联合点条件：标准差和极差都下降，且总体／序列均值、B0固定弱10%与20%、模型自身尾10%与20%、B0固定强25%均值不下降。失败项逐行注明；通过也仅为单seed开发候选，不代表正式测试或每条序列都不退化。控制含真实可训练容量，若有效也可记录容量候选，但不能称额外历史机制收益。LEFOKT历史通道β>0只是缩放，学习可补偿；β=0也仅删除显式题目通道，多KC注意力仍可能带间接题目信息。

'''
    groups = [
        ('UKT 启发：历史检索、不确定性、Rasch及保护组合',
         ['qikt_gru_ukt_20261114','qikt_gru_ukt_v2_20261114','qikt_gru_ukt_v3_20261114']),
        ('MTKT 启发：CIC接入位置与当前状态对照', ['qikt_mtkt_cic_v1_20261115']),
        ('LEFOKT 启发：当前查询与历史题目表示', ['qikt_lefokt_query_v1_20261115']),
    ]
    total = 0
    for title, roots in groups:
        doc += f'### {title}\n\n'
        rows, protections = [], []
        for index, name in enumerate(roots):
            root = repo / 'logs' / name
            registration = read(root / 'registration.json')
            for family in registration['families']:
                if family == 'B0' and index:
                    continue
                for path in sorted((root / family).glob('round*/balance.json')):
                    value = read(path)
                    training = value['training']
                    base = family == 'B0'
                    if not base:
                        total += 1
                    code = 'B0' if base else ('U'+str(index+1)+'-' if len(roots)>1 else '') + family + '/' + path.parent.name
                    prefix = 'UKT' if len(roots)>1 else 'MTKT' if family.startswith('M') else 'LEFOKT'
                    explanation = '原题目/KC双独立GRU，不增加参考模型模块。' if base else FUSION_SENTENCES[name][family]
                    settings = value['protocol'].get('model_settings', {})
                    params = ', '.join(k.replace('reference_fusion_', '').replace('reference_history_', '')+'='+str(v)
                        for k,v in settings.items() if k not in ['version','reference_fusion_mode','reference_fusion_bias','reference_cic_kernel','reference_cic_context'])
                    p = value['comparison']['candidate_validation_window_auc']
                    c = value['candidate']
                    failed = [FAILURES.get(k,k) for k,v in value['checks'].items() if not v]
                    result = '基线' if base else '通过' if value['exploratory_candidate_pass'] else '；'.join(failed)
                    control = family in registration.get('control_families', []) and not base
                    if control:
                        result = '容量对照；' + result
                    rows.append([code, explanation + ('；'+params if not base and params else ''),
                        f"{training['completed_epochs']}/{training['selected_epoch']}",p,c['macro_auc'],
                        c['std_ddof0'],c['range'],value['worsened'],result])
                    d = value['deltas']
                    protections.append([code, c['own_bottom10_mean'],c['own_bottom20_mean'],
                        value['baseline']['own_bottom10_mean']+d['fixed_baseline_bottom10_mean'],
                        value['baseline']['own_bottom20_mean']+d['fixed_baseline_bottom20_mean'],
                        .8231956701941118+d['fixed_baseline_top25_mean'],
                        value['decline_counts']['0.01'],str(value['paired_orirow_bootstrap_ci95']['std_delta'])])
        doc += table(['方案／轮次','来源机制及如何结合（一句话）','实际／选中epoch','总体AUC ↑',
            '序列平均AUC ↑','标准差 ↓','极差 ↓','退化序列数 ↓','联合结论／失败原因'],
            rows,{3:'max',4:'max',5:'min',6:'min',7:'min'})
        doc += '\n保护明细：固定组由B0定义；自身尾部按候选重新排序。配对95%区间为500次orirow重采样，不能代替跨seed验证。\n\n'
        doc += table(['方案／轮次','自身尾10% ↑','自身尾20% ↑','固定弱10% ↑','固定弱20% ↑',
            '固定强25% ↑','下降>.01数 ↓','Δstd配对95%区间'],protections,
            {1:'max',2:'max',3:'max',4:'max',5:'max',6:'min'})
    doc += f'''\n前述UKT启发、CIC及LEFOKT查询／历史三组已完成新增融合训练共 **{total} 次**（UKT启发34、CIC12、LEFOKT查询／历史9，均包含真实优化对照；B0只复用，不重复计数）。**联合候选0个**。CIC全部12条std高于B0；额外历史卷积尚无均衡收益证据。历史题目β衰减通常同时伤害弱组和总体，不能据此继续宣称去题目信息有利。

最接近的一条是L1Q/round2：std相对下降约0.085%、极差相对下降约0.398%，但总体AUC下降0.000006773612，仍未达标。UKT向量Rasch的部分均衡收益也能由加性同容量对照得到，不支持乘法机制独有归因。冻结主干或KL保护同样没有自动保住所有AUC。

## 5. 后续时间与序列结构迭代：结果与状态

这部分列出后续各批的绝对结果和执行状态；因找到候选而没有运行的轮次不算完成训练。每方向scale=.1/.5两轮上限，找到首个联合候选立即暂停，不自动扩seed／Peiyou。

'''
    root = repo / 'logs/qikt_mtkt_time_v2_20261115'
    descriptions = {'T2I':'把MTKT间隔／重复次数时间编码加入GRU的KC表示，影响历史输入与当前解码。',
        'T2D':'把MTKT时间编码与题目GRU状态拼接，以有界残差修正当前解码。',
        'T2G':'用MTKT时间编码查询严格过去的题目／KC响应，通过门控残差补充GRU读出（无ALiBi）。'}
    anchor=read(repo/'logs/qikt_gru_ukt_20261114/B0/round1/balance.json')
    c=anchor['candidate']
    rows = [['B0','原题目/KC双独立GRU。','—','已完成','27/17',
        anchor['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],'基线']]
    terminal = True
    for family, explanation in descriptions.items():
        statepath = root/'jobs'/family/'state.json'
        state = read(statepath).get('status') if statepath.exists() else 'queued'
        terminal &= state in ['completed','failed','stopped']
        queue = read(root/'jobs'/family/'queue_submission.json')
        completed = sorted((root/family).glob('round*/balance.json'))
        if not completed:
            rows.append([family,explanation,queue['job_id'],state,'—','—','—','—','—','未完成，不计为结果'])
        for path in completed:
            v=read(path)
            failed=[FAILURES.get(k,k) for k,ok in v['checks'].items() if not ok]
            t=v['training']; c=v['candidate']
            rows.append([family+'/'+path.parent.name,explanation,queue['job_id'],state,
                f"{t['completed_epochs']}/{t['selected_epoch']}",
                v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],
                '联合点通过' if v['exploratory_candidate_pass'] else '；'.join(failed)])
    doc += table(['方案','来源机制及如何结合（一句话）','GPU队列job ID','控制任务状态','实际／选中epoch',
        '总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓','联合结论'],rows,
        {5:'max',6:'max',7:'min',8:'min'})
    doc += '\n### 时间融合候选的受益序列与改进依据\n' + profiles.get('gru', '')
    guided = repo/'logs/qikt_sequence_guided_v3_20261115'
    if (guided/'registration.json').exists():
        registration=read(guided/'registration.json')
        doc+='\n### 基于受益序列的新结构实验：结果与取舍\n\n[预登记方案](18_序列受益分析驱动的GRU改进V3方案.md)。G3J将已有效的时间输入与当前解码以独立投影联合，输入／解码强度比固定5:1；G3S用当前KC严格前缀覆盖门控时间输入，覆盖越多修正越小，不使用真实AUC或完整响应路由。新结构同时与B0、已完成T2I最佳候选比较；旧候选不追加seed稳定性训练。\n\n'
        best=read(repo/'logs/qikt_mtkt_time_v2_20261115/T2I/round2/balance.json')
        anchor=read(repo/'logs/qikt_gru_ukt_20261114/B0/round1/balance.json')
        rows=[]
        for label,explanation,v in [('B0','原题目/KC双独立GRU',anchor),('T2I最佳已完成候选','时间信息加入输入，scale=.5，供进一步改进比较',best)]:
            c=v['candidate'];t=v['training']
            rows.append([label,explanation,f"{t['completed_epochs']}/{t['selected_epoch']}",v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],'基线' if label=='B0' else '联合点通过，单seed开发候选'])
        for family in ['G3J','G3S']:
            statepath=guided/'jobs'/family/'state.json'
            state=read(statepath).get('status') if statepath.exists() else 'queued'
            terminal=terminal and state in ['completed','failed','stopped']
            completed=sorted((guided/family).glob('round*/balance.json'))
            if not completed:
                rows.append([family,'MTKT时间机制：'+registration['family_names'][family]+'；'+state,'—','—','—','—','—','尚无完成结果'])
            for path in completed:
                v=read(path);c=v['candidate'];t=v['training']
                failed=[FAILURES.get(k,k) for k,ok in v['checks'].items() if not ok]
                rows.append([family+'/'+path.parent.name,('MTKT时间机制：' if family.startswith('G3') else '')+registration['family_names'][family],f"{t['completed_epochs']}/{t['selected_epoch']}",v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],'联合点通过' if v['exploratory_candidate_pass'] else '；'.join(failed)])
        doc+=table(['方案','结合方式与状态','实际／选中epoch','总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓','结论'],rows,{3:'max',4:'max',5:'min',6:'min'})
    additive=repo/'logs/qikt_lefokt_additive_v3_20261115'
    if (additive/'registration.json').exists():
        registration=read(additive/'registration.json')
        doc+='\n### LEFOKT加性迁移：完整源选择通过后接续\n\n[条件方案](21_LEFOKT加性迁移V3条件方案.md)；[完整源模块选择](20_LEFOKT完整消融与迁移选择.md)。只有完整源审计选择加性后才训练；若不满足则记录跳过。LAQ仅当前解码，LAI同时改变历史输入与当前解码；均保留标量题目ID、减去初始KC方向以保持零初始化。参数量匹配标量乘法，但优化几何不同，不称完全复刻源模型。\n\n'
        rows=[]
        anchor=read(repo/'logs/qikt_gru_ukt_20261114/B0/round1/balance.json')
        c=anchor['candidate'];rows.append(['B0','原双独立GRU','27/17',anchor['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],'基线'])
        for family in ['LAQ','LAI']:
            statepath=additive/'jobs'/family/'state.json'
            state=read(statepath).get('status') if statepath.exists() else 'queued'
            terminal=terminal and state in ['completed','failed','stopped']
            completed=sorted((additive/family).glob('round*/balance.json'))
            if not completed:
                summary=additive/family/'summary.json'
                explanation=read(summary).get('status',state) if summary.exists() else state
                rows.append([family,registration['family_names'][family]+'；'+explanation,'—','—','—','—','—','尚无完成训练结果'])
            for path in completed:
                v=read(path);c=v['candidate'];t=v['training']
                failed=[FAILURES.get(k,k) for k,ok in v['checks'].items() if not ok]
                rows.append([family+'/'+path.parent.name,('MTKT时间机制：' if family.startswith('G3') else '')+registration['family_names'][family],f"{t['completed_epochs']}/{t['selected_epoch']}",v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],'联合点通过' if v['exploratory_candidate_pass'] else '；'.join(failed)])
        doc+=table(['方案','结合方式与状态','实际／选中epoch','总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓','结论'],rows,{3:'max',4:'max',5:'min',6:'min'})
    doc+='\n候选不只对B0比较；[已完成候选的四项非支配比较](22_GRU候选的公平性取舍.md)同时列出T2I、T2D、G3J的取舍、个体退化和对最佳T2I的配对。G3J没有在标准差和弱10%上全面超过T2I，不能只按最高pooled选唯一赢家。\n'
    matched=repo/'logs/qikt_lefokt_additive_v3_20261115/matched_comparisons/summary.json'
    if matched.exists():doc+=read(matched)['report_fragment'].replace('上述三个对照','上述三类对照')
    placement=repo/'logs/qikt_lefokt_additive_v3_20261115/placement_comparison/input_vs_query_scale05/balance.json'
    if placement.exists():
        v=read(placement);d=v['deltas']
        doc+=f"\n同批次、同scale=.5的放置比较：LAI相对LAQ的Δ总体AUC={d['pooled_auc']:+.9f}、Δmacro={d['macro_auc']:+.9f}、Δstd={d['std_ddof0']:+.9f}、Δrange={d['range']:+.9f}，但按LAQ定义的固定强25%均值下降{abs(d['fixed_baseline_top25_mean']):.9f}，所以该直接比较仍未通过全部保护。它支持历史输入接入有利于四项主指标的线索，不证明对全部序列更安全。\n"
    doc += '''\n完整编排：GPU队列17真实预检→18独立吞吐校准→19/20/21三个有界方向；CPU队列127监控、128实时报告、129自动复盘。控制任务running不等于GPU实际训练；校准期间不运行科学训练，后续并发按本批实测吞吐与资源余量选择，共用既有GPU锁。

## 6. 原始证据与使用限制

- [UKT原保存预测及身份](../../logs/qikt_gru_ukt_20261114/ukt_sensitivity/summary.json)；[本次补算序列指标及逐批hash](../../logs/advanced_reference_20261115/ukt_sensitivity_balance/summary.json)。
- [MTKT源模型完整报告](01_MTKT消融与序列公平性.md)、[LEFOKT完整报告](02_LEFOKT消融与序列公平性.md)、[严格题目对齐与配对证据](04_严格题目评估与模块选择证据.md)。
- [UKT融合详细说明](../GRU_UKT实验整理/00_阅读入口与完成总账.md)、[CIC已完成完整复盘](../../logs/qikt_mtkt_cic_v1_20261115/completed_review/MTKT_CIC_GRU_V1_完整复盘.md)、[LEFOKT融合完整复盘](../../logs/qikt_lefokt_query_v1_20261115/completed_review/LEFOKT_Query_GRU_V1_完整复盘.md)。
- [下一批时间机制方案](12_MTKT时间信息融合V2方案.md)、[下一批实时结果](../../logs/qikt_mtkt_time_v2_20261115/MTKT_Time_GRU_V2_实时报告.md)。

各融合轮次的protocol.json、balance.json、paired_orirow.csv、保存模型、完整epoch检查点、窗口分批预测、命令和tsp状态均在对应logs根目录；底层产物在/data，不加入Git。旧汇总model字段曾误写public_mamba，实际version=gru／模型工厂为两个独立GRU，不能按那个旧字符串解释架构。本文由已有结果生成，当前任务继续执行。
'''
    output.write_text(doc)
    atomic_json(refs/'three_model_summary_manifest.json',{'input_sha256':inputs,
        'output':str(output),'output_sha256':digest(output),'completed_fusion_trainings':total,
        'new_batch_terminal':terminal,'formal_windowauclate_mean':None})
    return terminal


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('/root/pykt-toolkit'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    while True:
        done=render(a.repo,a.output)
        if done or not a.wait:
            break
        time.sleep(30)
