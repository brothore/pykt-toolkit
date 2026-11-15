"""Paired strict-question module evidence, distinct from native fragments."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prediction_progress import atomic_json,digest
from audit_inputs import sequence_metrics
from sequence_balance_metrics import assess_balance
from report_advanced_references import table,NAMES


def read(p):return json.loads(p.read_text())
def evaluate_pair(base,candidate,out):
    out.mkdir(parents=True,exist_ok=True);identity={'base':digest(base),'candidate':digest(candidate),'analyzer':digest(Path(__file__))};cached=out/'balance.json'
    if cached.exists():
        saved=read(cached)
        if saved['input_identity']!=identity:raise ValueError('Strict comparison identity changed')
        return saved
    def frame(p):return pd.read_csv(p,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True)
    b=frame(base);c=frame(candidate);keys=['orirow','qidx','late_trues','questions','concepts']
    if not b[keys].equals(c[keys]):raise ValueError('Strict targets/labels/KCs differ')
    bc,bs=sequence_metrics(b);cc,cs=sequence_metrics(c);paired=cs.set_index('orirow').join(bs.set_index('orirow').auc.rename('baseline_auc'));paired['delta_auc']=paired.auc-paired.baseline_auc
    ba=float(roc_auc_score(b.late_trues,b.late_mean));ca=float(roc_auc_score(c.late_trues,c.late_mean));v=assess_balance(paired,ca-ba)
    v['checks'].update(fixed_weak20_preserved=v['deltas']['fixed_baseline_bottom20_mean']>=0,own_tail20_preserved=v['deltas']['own_bottom20_mean']>=0);v['exploratory_candidate_pass']=all(v['checks'].values());v['input_identity']=identity;v['pooled']={'baseline':ba,'candidate':ca};v['events']=len(b);v['formal_windowauclate_mean']=None
    paired.to_csv(out/'paired_orirow.csv');atomic_json(cached,v);return v


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();root=a.root;raw=root/'runs/strict_questions';evidence=root/'strict_analysis';evidence.mkdir(exist_ok=True);tasks=['mtkt_strict_saved','lefokt_akt_strict_saved','ukt_strict_saved']+[t+'_strict_trained' for t in ['mtkt_full','mtkt_no_time','mtkt_no_cic','mtkt_no_alibi','lefokt_akt_full','lefokt_akt_no_alibi','lefokt_akt_no_interaction_rasch']]
    while True:
        states={t:read(root/'jobs'/t/'state.json').get('status') if (root/'jobs'/t/'state.json').exists() else 'queued' for t in tasks};rows=[];comparisons=[];module_rows=[]
        for task in tasks:
            path=raw/task/'validation_question_results.json'
            if not path.exists():continue
            v=read(path);z=v['cohorts']['primary_20_5'];rows.append([task,v['model'],f"{v['seed']} / {v['completed_epochs']} / {v['selected_epoch']}",v['events'],v['validation_question_auc'],z['mean'],z['std'],z['range']])
        for model,modes in [('mtkt',['no_time','no_cic','no_alibi']),('lefokt_akt',['no_alibi','no_interaction_rasch'])]:
            full=raw/(model+'_full_strict_trained')/'window_predictions.tsv'
            if not full.exists():continue
            for mode in modes:
                removed=raw/(model+'_'+mode+'_strict_trained')/'window_predictions.tsv'
                if not removed.exists():continue
                v=evaluate_pair(full,removed,evidence/(model+'_'+mode+'_remove'));add=evaluate_pair(removed,full,evidence/(model+'_'+mode+'_addback'));comparisons.append({'model':model,'removed':mode,'remove':v,'addback':add})
                d=v['deltas'];module_rows.append([model,NAMES[mode],d['pooled_auc'],d['macro_auc'],d['std_ddof0'],d['range'],d['fixed_baseline_bottom10_mean'],d['fixed_baseline_top25_mean'],v['worsened'],str(v['paired_orirow_bootstrap_ci95']['std_delta']),add['exploratory_candidate_pass']])
        text='''# 严格当前题标签隔离：三参考模型及重训消融

本页只用nips/fold0开发数据，不用正式测试选参。每个题目qidx≥1计一次；多KC分别查询，任一查询中都排除当前题的所有真实响应，最后取均值。参考上下文最多199个过去展开KC＋一个当前KC；GRU上下文按题目数计，两者不等价。指标名为validation_question_auc，不能替代正式windowauclate_mean。

## 当前覆盖与绝对指标

seed／实际训练epoch／选中epoch；已有参考权重若缺少选中epoch元信息显示None，不补造。每列最优不等于联合目标通过。

'''+table(['任务','模型','seed/epoch/选中','题目目标数','严格验证pooled↑','macro↑','std↓','range↓'],rows,{4:'max',5:'max',6:'min',7:'min'} if rows else {})
        text+='\n## 相对匹配重训完整模型：删除模块的影响\n\n所有Δ为“删除后的模型−完整模型”；完整/删除均从相同seed和保存配置重训，并使用同一严格题目评估。std/range的正Δ表示删除后离散变大。最后一列检查反向“加回模块”是否同时改善std/range并保护pooled/macro、固定弱组/自身尾部/强组，不等于跨seed因果保证。\n\n'+table(['模型','删除内容','Δpooled','Δmacro','Δstd','Δrange','Δ固定弱10%','Δ固定强25%','退化数','Δstd95%区间','加回模块联合点通过'],module_rows)
        text+='\n## 状态\n\n'+table(['任务','状态'],[[t,states[t]] for t in tasks])+'\n\n未完成或失败项不当作完成；数据不足时不选“最佳模块”。原生概念片段报告、用户旧完整重叠滑窗正式结果与本页分开，不混减。后续GRU融合方案仍须独立预登记和实际训练验证。\n'
        (a.docs/'04_严格题目评估与模块选择证据.md').write_text(text)
        atomic_json(evidence/'summary.json',{'states':states,'completed_strict_results':len(rows),'expected':len(tasks),'comparisons':comparisons,'all_required_complete':all(s=='completed' for s in states.values()),'goal_complete':False,'next':'Scientifically review matched addback evidence and register multiple GRU module integrations; do not declare transfer success from source-model ablation'})
        if not a.wait or all(s in ['completed','failed','stopped'] for s in states.values()):return
        time.sleep(20)


if __name__=='__main__':main()
