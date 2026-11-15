"""Conditional effects across the completed query/history/ALiBi factorial."""
import argparse
from itertools import product
import json
from pathlib import Path
from analyze_strict_reference_ablations import evaluate_pair
from prediction_progress import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo;source=repo/'logs/lefokt_extended_ablation_v2_fix01_20261115';old=repo/'logs/advanced_reference_20261115/runs/strict_questions';out=repo/'logs/advanced_reference_20261115/lefokt_factorial';out.mkdir(parents=True,exist_ok=True)
    mapping={(1,1,1):'full',(1,0,1):'no_interaction_rasch',(1,1,0):'no_alibi',(0,1,1):'no_query_rasch',(0,0,1):'no_both_rasch',(0,1,0):'no_query_rasch_no_alibi',(1,0,0):'no_interaction_rasch_no_alibi',(0,0,0):'no_both_rasch_no_alibi'}
    folders={flags:old/('lefokt_akt_'+mode+'_strict_trained') if mode in ['full','no_interaction_rasch','no_alibi'] else source/'strict_questions'/mode for flags,mode in mapping.items()}
    records=[];rows=[];names=['查询侧乘法','历史交互侧乘法','ALiBi']
    def arrow(delta,base):return ('↑' if delta<0 else '↓' if delta>0 else '→')+f'{abs(delta)/base*100:.4f}%'
    for index in range(3):
        others=[i for i in range(3) if i!=index]
        for values in product([0,1],repeat=2):
            off=[0,0,0]
            for i,v in zip(others,values):off[i]=v
            on=off.copy();on[index]=1;off=tuple(off);on=tuple(on)
            key=f'{index}_{values[0]}_{values[1]}'
            v=evaluate_pair(folders[off]/'window_predictions.tsv',folders[on]/'window_predictions.tsv',out/key);d=v['deltas']
            context='；'.join(names[i]+('开' if flag else '关') for i,flag in zip(others,values))
            records.append({'factor':names[index],'context':context,'baseline_off':mapping[off],'candidate_on':mapping[on],'balance':v})
            rows.append([names[index],context,f"{d['pooled_auc']:+.9f}",f"{d['macro_auc']:+.9f}",arrow(d['std_ddof0'],v['baseline']['std_ddof0']),arrow(d['range'],v['baseline']['range']),f"{d['fixed_baseline_bottom10_mean']:+.9f}",f"{d['fixed_baseline_top25_mean']:+.9f}",v['exploratory_candidate_pass']])
    headers=['加回因子','固定的其他条件','Δ总体AUC','Δ序列均值','std效果幅度','极差效果幅度','Δ固定弱10%','Δ固定强25%','联合点通过']
    doc='''# LEFOKT完整八格：条件效应而非单模块口号

nips_task34/fold0/seed3407，严格开发题目221348、primary773、std ddof=0，正式windowauclate_mean缺失。查询侧、历史交互侧、ALiBi各开关构成完整八格；两种加性替代另见源选择报告。

每行以“该因子关闭、另外两项保持表中状态”为独立baseline，比较加回因子后的变化。std／极差↑表示缩小、↓表示扩大，百分比以本行关闭条件为分母；弱／强组也由该行对照定义，不能直接跨行混减群体增益。单seed、重训优化路径不同，不提供跨seed稳定性保证。\n\n'''
    doc+='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    doc+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
    doc+='''\n## 解释边界

查询侧不只服务最后一题，它还参与各历史位置的query/key及最终解码。删除两侧乘法也会删除当前配置的题目ID入口，不能把恢复题目身份的全部增益归给乘法。历史交互作用必须分别在查询路径开／关时比较，不能只拿完整模型一条删除实验决定其普遍有用／无用。

ALiBi提高整体区分能力与降低序列离散是不同命题；请并看均值、分散和固定强／弱组。某些开关条件下std缩小可能主要来自强组下降，不等于补弱成功。原始每行完整指标、区间和配对序列见[全部条件效应](/root/pykt-toolkit/logs/advanced_reference_20261115/lefokt_factorial/summary.json)。源模块选择及其是否只为探索见[完整选择报告](20_LEFOKT完整消融与迁移选择.md)。
'''
    destination=repo/'docs/先进模型消融与GRU融合/23_LEFOKT八格条件效应.md';destination.write_text(doc);atomic_json(out/'summary.json',{'conditional_effects':records,'document':str(destination),'formal_windowauclate_mean':None})


if __name__=='__main__':main()
