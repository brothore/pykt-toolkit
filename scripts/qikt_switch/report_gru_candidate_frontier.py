"""Report completed candidates against B0 and the strongest dispersion anchor."""
import argparse
import json
from pathlib import Path
from prediction_progress import atomic_json,digest
from render_three_model_summary import table
from analyze_strict_reference_ablations import evaluate_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo
    root=repo/'logs/advanced_reference_20261115/gru_candidate_frontier';root.mkdir(parents=True,exist_ok=True)
    sources=[('B0','原题目/KC双独立GRU','qikt_gru_ukt_20261114','B0',1),('T2I','时间编码加入KC输入','qikt_mtkt_time_v2_20261115','T2I',2),('T2D','时间编码只修正当前解码','qikt_mtkt_time_v2_20261115','T2D',1),('T2G','时间查询因果历史检索与门控','qikt_mtkt_time_v2_20261115','T2G',1),('G3J','共享时间编码：输入＋当前解码，比例5:1','qikt_sequence_guided_v3_20261115','G3J',1),('G3S','按因果KC覆盖收缩时间输入','qikt_sequence_guided_v3_20261115','G3S',1)]
    additive=repo/'logs/qikt_lefokt_additive_v3_20261115'
    for family in ['LAQ','LAI']:
        for path in sorted((additive/family).glob('round*/balance.json')):
            value=json.loads(path.read_text())
            if value['exploratory_candidate_pass']:
                number=int(path.parent.name.replace('round',''))
                sources.append((family+'/'+path.parent.name,'LEFOKT标量加性条件，'+('当前解码' if family=='LAQ' else '历史输入与解码'),'qikt_lefokt_additive_v3_20261115',family,number))
    records=[];identity={}
    for label,desc,folder,family,round_ in sources:
        path=repo/'logs'/folder/family/f'round{round_}'/'balance.json';v=json.loads(path.read_text());identity[str(path)]=digest(path);records.append({'label':label,'description':desc,'path':str(path),'value':v})
    def metrics(record):
        v=record['value'];c=v['candidate'];return [v['comparison']['candidate_validation_window_auc'],c['macro_auc'],-c['std_ddof0'],-c['range']]
    for r in records:
        m=metrics(r);r['dominated_by']=[x['label'] for x in records if x is not r and all(b>=a for a,b in zip(m,metrics(x))) and any(b>a for a,b in zip(m,metrics(x)))]
    def extended_metrics(record):
        v=record['value'];return metrics(record)+[-v['worsened'],-v['decline_counts']['0.01']]
    extended_frontier=[]
    for r in records:
        m=extended_metrics(r)
        dominated=any(all(b>=a for a,b in zip(m,extended_metrics(x))) and any(b>a for a,b in zip(m,extended_metrics(x))) for x in records if x is not r)
        if not dominated:extended_frontier.append(r['label'])
    rows=[];protection=[]
    for r in records:
        v=r['value'];c=v['candidate'];t=v['training'];d=v['deltas'];label=r['label']
        rows.append([label,r['description'],f"{t['completed_epochs']}/{t['selected_epoch']}",v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],v.get('substantive_development_signal',False),'被'+','.join(r['dominated_by'])+'支配' if r['dominated_by'] else '四项指标非支配'])
        protection.append([label,c['own_bottom10_mean'],c['own_bottom20_mean'],v['baseline']['own_bottom10_mean']+d['fixed_baseline_bottom10_mean'],.8231956701941118+d['fixed_baseline_top25_mean'],v['worsened'],v['decline_counts']['0.01'],str(v['paired_orirow_bootstrap_ci95']['std_delta'])])
    doc='''# 双GRU已完成候选：公平性与性能的取舍

nips_task34 / fold0 / seed42；开发题目窗口221348、primary773、std ddof=0。正式windowauclate_mean缺失。各方向达到首个联合点候选后已暂停，不自动追加seed／Peiyou。每表首行B0，幅度为相对变化；std/range的↑表示数值缩小。单seed配对区间不代表稳定性，实质开发信号也不等于正式测试显著结果。

“非支配”只比较pooled/macro越高、std/range越低这四项指标；没有将尾部、退化数量等也纳入定义，不能称全面最佳。\n\n'''
    doc+=table(['方案','改动','实际／选中epoch','总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓','实质开发信号','四项指标比较'],rows,{3:'max',4:'max',5:'min',6:'min'})
    doc+='\n## 尾部与个体退化\n\n保护的是群体均值，并未做到逐条序列均不退化。原强25%由B0固定定义。区间为500次orirow配对重采样。\n\n'
    doc+=table(['方案','自身尾10% ↑','自身尾20% ↑','B0固定弱10% ↑','B0固定强25% ↑','退化数 ↓','下降>.01数 ↓','Δstd95%区间'],protection,{1:'max',2:'max',3:'max',4:'max',5:'min',6:'min'})
    anchor=repo/'logs/qikt_mtkt_time_v2_20261115/T2I/round2/baseline/validation_window/window_predictions.tsv';comparisons=[]
    for family in ['G3J','G3S']:
        candidate=repo/f'logs/qikt_sequence_guided_v3_20261115/{family}/round1/baseline/validation_window/window_predictions.tsv'
        v=evaluate_pair(anchor,candidate,root/(family+'_vs_T2I'));comparisons.append({'family':family,'balance':v})
    doc+='''\n## 为什么暂不替换成唯一最佳方案

T2I对标准差改善最大；G3J的总体／序列均值更高、极差比T2I更小，但相对T2I的Δstd=+0.000135523，固定T2I弱10%及自身尾10%略降，所以不能称全面超过T2I。G3S相对T2I的Δpooled=−0.000967114、Δstd=+0.000496632，尽管极差更小，也没有继续加强该门控方向的依据。

按四项主指标，T2I、G3J、T2D各有取舍；T2D极差最小。G3S与T2G被G3J在这四项指标上支配，但不表示所有单条序列或全部保护指标也被支配。保留非支配候选与完整退化证据，稳定性仍按用户要求后置。

T2I有249条序列退化、36条下降>.01；G3J有236条退化、35条下降>.01。不能只用总体／弱组均值改善掩盖这些代价。后续选择需同时考虑用户对分散、极端值和个体退化的权衡，不在这里偷偷修改保护阈值。

[完整审计、数值与相对T2I配对](/root/pykt-toolkit/logs/advanced_reference_20261115/gru_candidate_frontier/summary.json)；[三模型及全部融合总表](13_三模型消融与全部GRU融合汇总.md)。
'''
    doc+='\n若再把退化总数、下降>.01数量也纳入非支配判断，当前集合为：'+', '.join(extended_frontier)+'。B0相对自身的退化数按定义为0，因此自然保留；这不意味着它的序列分散最小。这只是展示多目标取舍，不引入新的训练选模规则。\n'
    for r in records:
        if r['label'].startswith('LA'):
            v=r['value'];doc+=f"\n加性候选{r['label']}：{v['worsened']}条下降，其中{v['decline_counts']['0.01']}条下降>.01；最差ΔAUC={v['worst_delta']:+.9f}。幅度小但大幅退化更少时，也不能仅因四项主指标被支配就说它在所有保护维度都没有价值。仍未验证跨seed稳定性。\n"
    destination=repo/'docs/先进模型消融与GRU融合/22_GRU候选的公平性取舍.md';destination.write_text(doc)
    atomic_json(root/'summary.json',{'records':records,'four_metric_nondominated':[r['label'] for r in records if not r['dominated_by']],'extended_frontier_including_regressions':extended_frontier,'anchor_comparisons':comparisons,'input_sha256':identity,'analyzer_sha256':digest(Path(__file__)),'formal_windowauclate_mean':None,'document':str(destination)})
    print('Four-metric nondominated:',[r['label'] for r in records if not r['dominated_by']])


if __name__=='__main__':main()
