"""Publish a standalone AKT baseline/transfer table from completed results."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from prediction_progress import atomic_json

EXPLANATIONS={'baseline':'原始AKT；匹配的训练与评估基线',
    'ukt_cov':'UKT启发：正协方差Wasserstein历史匹配、alpha²聚合，残差补充AKT',
    'mtkt_cic':'MTKT双因果卷积门控交互，残差接入AKT因果隐状态',
    'mtkt_time':'MTKT时间设计：间隔与次数桶的小型解码修正',
    'lef_additive':'LEFOKT加性设计：保留题目/KC参数，替换AKT双侧Rasch乘法'}


def render(root):
    rows=[]
    for path in sorted((root/'runs').glob('*/strict_validation/result.json')):
        r=json.loads(path.read_text());r['run']=path.parents[1].name;rows.append(r)
    baseline=next((r for r in rows if r['mechanism']=='baseline'),None)
    rows.sort(key=lambda r:(r['mechanism']!='baseline',r['mechanism'],r['scale']))
    status={p.parent.name:json.loads(p.read_text()) for p in (root/'families').glob('*/state.json')}
    atomic_json(root/'summary.json',dict(results=rows,families=status,development_only=True))
    lines=['# AKT 三参考机制迁移：结果汇总','',
        '本文件自动更新。方案与参数依据见 [00_实验方案.md](00_实验方案.md)。所有结果为 nips_task34 / fold0 / seed3407，采用新训练的原生AKT作baseline。', '',
        '**指标口径：** 开发验证严格逐题预测，当前整题标签隔离；每个目标最多199个历史展开概念。主子集：目标数≥20、较少类≥5。标准差ddof=0。正式 `all_results.json.windowauclate_mean` 目前未计算，不能把本表 pooled AUC 当成该字段。', '',
        '↓/↑按指标好坏表示：标准差/极差缩减为↑，扩大为↓；pooled/macro使用带正负号百分比。逐列最优加粗，只表示已完成开发实验的点值。', '',
        '|方案|做了什么|scale|实际/选中epoch|pooled AUC|序列平均AUC|标准差|极差|相对baseline变化：pooled / 平均 / 标准差 / 极差|改善/退化序列|下降>0.01|初步候选|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|']
    fields=['validation_question_auc','mean','std','range']
    def vals(r):
        c=r['cohorts']['primary_20_5'];return [r['validation_question_auc'],c['mean'],c['std'],c['range']]
    best=[(max if i<2 else min)(vals(r)[i] for r in rows) for i in range(4)] if rows else []
    for r in rows:
        v=vals(r);text=[f'**{x:.6f}**' if x==best[i] else f'{x:.6f}' for i,x in enumerate(v)]
        b=r.get('balance');delta='—';counts='—';declines='—';candidate='baseline'
        if baseline and b:
            bv=vals(baseline);d=[(v[i]/bv[i]-1)*100 for i in range(4)]
            delta=' / '.join([f'{d[i]:+.3f}%' if i<2 else f'{"↑" if d[i]<0 else "↓" if d[i]>0 else "→"}{abs(d[i]):.3f}%' for i in range(4)])
            counts=f"{b['improved']}/{b['worsened']}";declines=str(b['decline_counts']['0.01']);candidate='是（单seed探索）' if b['exploratory_candidate_pass'] else '否'
        lines.append('|'+ '|'.join([r['run'],EXPLANATIONS[r['mechanism']],str(r['scale']),f"{r['completed_epochs']}/{r['selected_epoch']}",*text,delta,counts,declines,candidate])+'|')
    if not rows:lines.extend(['','尚无完成严格开发评估的训练。'])
    lines+=['','## 当前执行状态','']
    for family,s in status.items():lines.append(f"- {family}：{s['status']}；已完成{len(s['rounds'])}轮。"+(s.get('stop_reason','')))
    lines+=['','## 如何判断结果','',
        '初步候选需同时降低标准差和极差，并保住pooled、序列平均、自身尾部、基线固定弱10%与强25%群体均值。逐序列零退化另报，不由群体均值推断。完整保护指标、退化端点、500次配对bootstrap区间保存在每轮 strict_validation/result.json，并汇入本地 summary.json。', '',
        '时间方案假设当前事件时间已知；数据为DateAnswered，不能直接宣称答题前可部署。UKT为简化机制迁移，MTKT时间为桶特征迁移，均非完整参考模型。LEFOKT是在已有Rasch上做替换。最多每方向三轮；首个初步候选出现后暂停该方向。正式测试不用于本批参数筛选。', '',
        '原生训练片段仍保留原AKT的概念展开语义；本批严格评估避免当前题多KC标签泄漏。新模块均经过CPU因果性测试（激活零初始化残差后再翻转当前及未来标签），真实容器预检和有限梯度检查。', '']
    target=Path('docs/AKT公平性机制迁移/01_结果汇总.md');target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines));os.replace(temporary,target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    with (a.root/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);render(a.root)
