"""Collect the completed factorial ablations and paired strict-question metrics."""
import argparse
import json
from pathlib import Path
import time
from analyze_strict_reference_ablations import evaluate_pair
from render_three_model_summary import table
from prediction_progress import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    reg=json.loads((a.root/'registration.json').read_text());ref=a.repo/'logs/advanced_reference_20261115/runs/strict_questions'
    while True:
        states={m:json.loads((a.root/'jobs'/m/'state.json').read_text()).get('status') if (a.root/'jobs'/m/'state.json').exists() else 'queued' for m in reg['modes']}
        rows=[];paired=[]
        cases=[('full',ref/'lefokt_akt_full_strict_trained'),('no_interaction_rasch',ref/'lefokt_akt_no_interaction_rasch_strict_trained'),('no_alibi',ref/'lefokt_akt_no_alibi_strict_trained')]+[(m,a.root/'strict_questions'/m) for m in reg['modes']]
        doc='''# LEFOKT-AKT补充消融：结果与保护指标

nips_task34 / fold0 / seed3407；每项从头匹配重训，沿用原选模／早停。全部采用同一严格题目链路，当前题所有真实响应排除；221348目标、primary773、std ddof=0。正式windowauclate_mean缺失。前三条件复用已完成的真实重训与预测，不计新训练。首行完整LEFOKT为baseline。标准差／极差幅度↑表示数值缩小、↓表示增大。\n\n'''
        for mode,folder in cases:
            path=folder/'validation_question_results.json'
            if not path.exists():continue
            v=json.loads(path.read_text());c=v['cohorts']['primary_20_5']
            rows.append([mode,reg['names'].get(mode,mode),f"{v['completed_epochs']}/{v['selected_epoch']}",v['validation_question_auc'],c['mean'],c['std'],c['range']])
            if mode!='full':
                balance=evaluate_pair(ref/'lefokt_akt_full_strict_trained/window_predictions.tsv',folder/'window_predictions.tsv',a.root/'analysis'/mode)
                paired.append({'mode':mode,'balance':balance})
        doc+=table(['条件','一句话改动','实际／选中epoch','总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓'],rows,{3:'max',4:'max',5:'min',6:'min'})
        doc+='\n## 配对保护与不确定性\n\n所有Δ相对完整LEFOKT；固定弱／强组由完整模型定义。区间为orirow配对重采样，不代表跨seed稳定性。\n\n'
        # This is a delta table: no baseline-relative percentage on zero deltas.
        heads=['条件','Δ固定弱10%','Δ固定弱20%','Δ固定强25%','退化数','下降>.01数','Δstd95%区间','联合点通过']
        doc+='| '+' | '.join(heads)+' |\n| '+' | '.join(['---']*len(heads))+' |\n'
        for x in paired:
            b=x['balance'];d=b['deltas'];values=[x['mode'],f"{d['fixed_baseline_bottom10_mean']:+.12f}",f"{d['fixed_baseline_bottom20_mean']:+.12f}",f"{d['fixed_baseline_top25_mean']:+.12f}",b['worsened'],b['decline_counts']['0.01'],b['paired_orirow_bootstrap_ci95']['std_delta'],b['exploratory_candidate_pass']]
            doc+='| '+' | '.join(map(str,values))+' |\n'
        doc+='\n## 队列状态\n\n'+ '\n'.join(f'- {m}：{s}' for m,s in states.items())+'\n\n见[预登记方案](16_LEFOKT补充消融V2方案.md)。缺失条件不能被当作完成；当前不足以判断LEFOKT没有公平性贡献。最终机制解释需要同时检查各匹配边与交互，不只选某一列最优。\n'
        destination=a.repo/'docs/先进模型消融与GRU融合/17_LEFOKT补充消融结果.md';destination.write_text(doc)
        terminal=all(s in ['completed','failed','stopped'] for s in states.values())
        atomic_json(a.root/'report_status.json',{'states':states,'strict_result_count':len(rows),'paired':paired,'all_terminal':terminal,'formal_windowauclate_mean':None,'document':str(destination)})
        if terminal:return
        time.sleep(30)


if __name__=='__main__':main()
