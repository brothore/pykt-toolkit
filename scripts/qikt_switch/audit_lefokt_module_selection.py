"""Require every registered source ablation before selecting a transfer design."""
import argparse
import json
from pathlib import Path
from analyze_strict_reference_ablations import evaluate_pair
from prediction_progress import atomic_json,digest
from render_three_model_summary import table


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=a.repo;root=repo/'logs/lefokt_extended_ablation_v2_fix01_20261115';reg=json.loads((root/'registration.json').read_text());old=repo/'logs/advanced_reference_20261115/runs/strict_questions'
    for mode in reg['modes']:
        state=json.loads((root/'jobs'/mode/'state.json').read_text())
        if state['status']!='completed':raise ValueError('Source factorial incomplete: '+mode)
    cases=[('full',old/'lefokt_akt_full_strict_trained'),('no_interaction_rasch',old/'lefokt_akt_no_interaction_rasch_strict_trained'),('no_alibi',old/'lefokt_akt_no_alibi_strict_trained')]+[(m,root/'strict_questions'/m) for m in reg['modes']]
    full=old/'lefokt_akt_full_strict_trained/window_predictions.tsv';records=[];rows=[]
    for mode,folder in cases:
        path=folder/'validation_question_results.json';v=json.loads(path.read_text());c=v['cohorts']['primary_20_5']
        if v['events']!=221348 or c['N']!=773 or v['fold']!=0 or v['seed']!=3407:raise ValueError('Source evaluation identity differs')
        b=evaluate_pair(full,folder/'window_predictions.tsv',a.output/'pairs'/mode)
        eligible=mode!='full' and all(b['checks'][k] for k in ['std_reduced','range_reduced','pooled_preserved','macro_preserved'])
        score=min(-b['deltas']['std_ddof0']/b['baseline']['std_ddof0'],-b['deltas']['range']/b['baseline']['range'])
        records.append({'mode':mode,'eligible_average_dispersion':eligible,'joint_pass':b['exploratory_candidate_pass'],'maximin_score':score,'balance':b,'result_sha256':digest(path)})
        rows.append([mode,reg['names'][mode],v['validation_question_auc'],c['mean'],c['std'],c['range'],b['exploratory_candidate_pass']])
    joint=[x for x in records if x['mode']!='full' and x['joint_pass']];pool=joint or [x for x in records if x['eligible_average_dispersion']]
    selected=max(pool,key=lambda x:x['maximin_score']) if pool else None
    value={'audit_passed':True,'passed':bool(selected and selected['mode']=='additive_rasch'),'selected_mode':selected['mode'] if selected else None,'selection_status':'source_joint_point_candidate' if joint else 'exploratory_only_no_source_joint_candidate','selection_rule':'Prefer all-protection joint points; otherwise exploratory fallback requires pooled/macro protected and std/range reduced. Rank minimum relative std/range reduction. No strong-group tolerance relaxed. Rule declared after first5newconditions observed, not blind preregistration.','expected_transfer_design':'additive_rasch','records':records,'source_conditions':len(cases),'formal_windowauclate_mean':None,'not_stability_proof':True}
    atomic_json(a.output/'module_selection_ready.json',value)
    doc='# LEFOKT完整补充消融后的迁移模块选择\n\n全10个条件包含完整八格与两种加性条件；前三条件复用已完成匹配重训。nips/fold0/seed3407，严格题目221348、primary773、std ddof=0，正式windowauclate_mean缺失。\n\n'+table(['条件','改动','总体AUC ↑','序列均值 ↑','标准差 ↓','极差 ↓','联合点通过'],rows,{2:'max',3:'max',4:'min',5:'min'})
    doc+=f"\n选择：{value['selected_mode']}；状态：{value['selection_status']}。本次预备加性迁移允许启动={value['passed']}。先优先联合通过条件；若没有，再从四项主指标有利的条件中选择std/range相对缩减的较小者最大的一项，明确仅作探索。强组退化不会被放宽或改称通过。规则是在5个新结果已观察后登记，带有探索选择偏差，不是盲法验证或稳定性证明。\n\n[完整审计及配对结果]({a.output/'module_selection_ready.json'})。如果实际选择不是加性，本批预备加性迁移不启动，需重新复盘；不能只因代码已写好就训练。\n"
    (repo/'docs/先进模型消融与GRU融合/20_LEFOKT完整消融与迁移选择.md').write_text(doc)


if __name__=='__main__':main()
