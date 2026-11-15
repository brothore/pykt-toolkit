"""Verify completed source ablations and publish standalone strict summaries."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prediction_progress import atomic_json,digest
from audit_inputs import sequence_metrics
from report_advanced_references import table,NAMES


def read(p):return json.loads(p.read_text())
def frame(p):return pd.read_csv(p,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);a=p.parse_args();r=a.root;docs=a.docs;required=[model+'_'+mode for model,modes in [('mtkt',['full','no_time','no_cic','no_alibi']),('lefokt_akt',['full','no_alibi','no_interaction_rasch'])] for mode in modes];audit={'source_stage_complete':False,'goal_complete':False,'training':{},'strict':{},'full_tensor_reproduction':{},'inputs_sha256':{}}
    for task in required:
        folder=r/'runs/retraining'/task;assert read(r/'jobs'/task/'state.json')['status']=='completed';done=read(folder/'completion.json');state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);assert state['finished'] and state['completed_epochs']==done['training']['completed_epochs'];assert state['optimizer']['state'];assert all(k in state['rng'] for k in ['python','numpy','torch','cuda','loader','sampler']);assert len(state['epochs'])==state['completed_epochs'];assert digest(folder/'best_model.ckpt')==done['selected_checkpoint_sha256'];config=read(folder/'config.json');assert digest(folder/(config['params']['emb_type']+'_model.ckpt'))==done['selected_checkpoint_sha256'];assert read(folder/'native_evaluation/full/result.json')['identity']['checkpoint']==done['selected_checkpoint_sha256'];audit['training'][task]={'completed_epochs':state['completed_epochs'],'selected_epoch':state['selected_epoch'],'checkpoint_sha256':done['selected_checkpoint_sha256'],'complete_epoch_model_optimizer_rng_saved':True};audit['inputs_sha256'][str(folder/'completion.json')]=digest(folder/'completion.json')
    for model in ['mtkt','lefokt_akt']:
        original=next(Path('/data/pykt-toolkit-latest/local_runs/nips0_best_20261104').glob(model+'/**/config.json'));params=read(original)['params'];name=params['emb_type']+'_model.ckpt';x=torch.load(original.parent/name,map_location='cpu',weights_only=True);y=torch.load(r/'runs/retraining'/(model+'_full')/name,map_location='cpu',weights_only=True);assert x.keys()==y.keys() and all(t.dtype==y[k].dtype and t.shape==y[k].shape and torch.equal(t,y[k]) for k,t in x.items());audit['full_tensor_reproduction'][model]={'exact_equal':True,'dtype_shape_checked':True,'tensors':len(x),'maximum_difference':0.,'old_file_sha256':digest(original.parent/name),'new_file_sha256':digest(r/'runs/retraining'/(model+'_full')/name)}
    baseline=read(Path('/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/baseline_ready.json'));b0=frame(Path(baseline['predictions']));keys=['orirow','qidx','late_trues','questions','concepts'];strict_tasks=['mtkt_strict_saved','lefokt_akt_strict_saved','ukt_strict_saved']+[t+'_strict_trained' for t in required];scores={}
    for task in strict_tasks:
        folder=r/'runs/strict_questions'/task;v=read(folder/'validation_question_results.json');assert read(r/'jobs'/task/'state.json')['status']=='completed';data=frame(folder/'window_predictions.tsv');assert data[keys].equals(b0[keys]);assert len(data)==v['events']==221348;assert not data.duplicated(['orirow','qidx']).any();cohorts,s=sequence_metrics(data);assert cohorts['primary_20_5']['N']==773
        for key in ['mean','std','range']:assert abs(cohorts['primary_20_5'][key]-v['cohorts']['primary_20_5'][key])<1e-12
        s=s[(s.n>=20)&(s.minority>=5)].set_index('orirow').sort_index();scores[task]=(v,s);audit['strict'][task]={'events':len(data),'primary_N':len(s),'checkpoint_sha256':v['identity']['weights'],'prediction_sha256':digest(folder/'window_predictions.tsv'),'inference_reused':bool(v.get('reused_identical_predictions'))};audit['inputs_sha256'][str(folder/'validation_question_results.json')]=digest(folder/'validation_question_results.json')
    for model,modes,filename in [('mtkt',['full','no_time','no_cic','no_alibi'],'01_MTKT消融与序列公平性.md'),('lefokt_akt',['full','no_alibi','no_interaction_rasch'],'02_LEFOKT消融与序列公平性.md')]:
        tasks=[model+'_'+m+'_strict_trained' for m in modes];v,base=scores[tasks[0]];order=base.sort_values('auc',kind='stable');n=len(order);weak10=order.index[:math.ceil(n*.1)];weak20=order.index[:math.ceil(n*.2)];strong=order.index[-math.ceil(n*.25):];mainrows=[];tailrows=[];riskrows=[]
        for mode,task in zip(modes,tasks):
            info,s=scores[task];metrics=info['cohorts']['primary_20_5'];t=audit['training'][model+'_'+mode];delta=s.auc-base.auc;arr=np.sort(s.auc.to_numpy());bal=read(r/'strict_analysis'/(model+'_'+mode+'_remove')/'balance.json') if mode!='full' else None
            mainrows.append([mode,NAMES[mode],f"{t['completed_epochs']}/{t['selected_epoch']}",info['validation_question_auc'],metrics['mean'],metrics['std'],metrics['std']-v['cohorts']['primary_20_5']['std'],metrics['range'],metrics['range']-v['cohorts']['primary_20_5']['range']])
            tailrows.append([mode,NAMES[mode],float(arr[:math.ceil(n*.1)].mean()),float(arr[:math.ceil(n*.2)].mean()),float(s.loc[weak10].auc.mean()),float(s.loc[weak20].auc.mean()),float(s.loc[strong].auc.mean())])
            riskrows.append([mode,NAMES[mode],int((delta< -1e-12).sum()),int((delta<-.001).sum()),int((delta<-.005).sum()),int((delta<-.01).sum()),float(delta.min()),str(bal['paired_orirow_bootstrap_ci95']['std_delta']) if bal else '[0, 0]','基线' if mode=='full' else '通过' if bal['exploratory_candidate_pass'] else '未通过'])
        section='\n## 严格题目评估与最终消融结论\n\n本节在全部7次匹配训练、10个严格结果全部完成后发布。10个结果含2个逐张量完全一致的完整模型推理复用，实际严格GPU推理8次，不算10次新推理。完整模型分别重训，未用历史最优权重冒充；MTKT82、LEF56个参数张量连dtype/shape都完全复现。\n\n本节baseline是对应参考模型的完整重训，不是GRU B0。每题独立查询其KC，排除当前题全部真实响应，题目目标221348个、主序列773；std ddof0。上下文最多199个过去展开KC＋当前KC，与GRU按200题计不同。seed3407/fold0，各行实际/选中epoch保留。正式windowauclate_mean缺失，均为严格开发验证。\n\n'
        section+=table(['条件','实际改动','实际/选中epoch','严格验证pooled↑','macro↑','std↓','Δstd↓','range↓','Δrange↓'],mainrows,{3:'max',4:'max',5:'min',6:'min',7:'min',8:'min'})+'\n固定弱/强组按本表完整模型定义，不与反向加回的群体混用。\n\n'+table(['条件','实际改动','自身尾10%↑','自身尾20%↑','固定弱10%↑','固定弱20%↑','固定强25%↑'],tailrows,{2:'max',3:'max',4:'max',5:'max',6:'max'})+'\n'+table(['条件','实际改动','退化数↓','下降>.001↓','下降>.005↓','下降>.01↓','最差增益↑','Δstd配对95%区间','相对完整模型联合保护'],riskrows,{2:'min',3:'min',4:'min',5:'min',6:'max'})
        if model=='mtkt':section+='\nCIC与时间分支加回时均有源模型点联合线索，但std配对区间跨0。按06的双分散指标规则，CIC优先；ALiBi提高均值却扩大离散/伤害尾部，不作为本次公平性首选。[CIC三位置与当前状态控制方案](07_MTKT模块选择与CIC融合V1方案.md)已登记，不等于GRU收益已确认。\n'
        else:section+='\n源模型没有全部保护通过的公平性模块。去交互Rasch相对最有希望，但固定强25%下降约.0001013，371条退化、55条下降>.01；不能称达标。选择“当前查询保留题目条件、历史响应共享化”作有界探索，不把ALiBi硬称公平性模块。[三种查询/历史融合方案](08_LEFOKT设计选择与查询历史融合V1方案.md)单独登记。去全部Rasch推理也去题目ID，仍不等于Rasch独有贡献。\n'
        section+=f'\n[源阶段完成审计]({r}/source_stage_audit.json)；[完整严格比较](04_严格题目评估与模块选择证据.md)。区间是orirow配对500次重采样，不是跨seed、独立学生或逐序列无退化证明。\n'
        target=docs/filename;old=target.read_text().split('\n## 严格题目评估与最终消融结论',1)[0];target.write_text(old+section)
    # Append common question-target comparison to the three-model report.
    allrows=[]
    for model in ['mtkt','ukt','lefokt_akt']:
        v,s=scores[model+'_strict_saved'];z=v['cohorts']['primary_20_5'];allrows.append([model,v['seed'],v['completed_epochs'],v['validation_question_auc'],z['mean'],z['std'],z['range'],float(np.sort(s.auc)[:math.ceil(len(s)*.1)].mean()),float(np.sort(s.auc)[:math.ceil(len(s)*.2)].mean())])
    _,bs=sequence_metrics(b0);bs=bs[(bs.n>=20)&(bs.minority>=5)];bvals=bs.auc.to_numpy();b0pooled=read(Path('/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/balance.json'))['comparison']['baseline_validation_window_auc'];allrows.append(['GRU B0',42,27,b0pooled,float(bvals.mean()),float(bvals.std(ddof=0)),float(np.ptp(bvals)),float(np.sort(bvals)[:math.ceil(len(bvals)*.1)].mean()),float(np.sort(bvals)[:math.ceil(len(bvals)*.2)].mean())])
    section='\n## 完成后的共同题目目标比较\n\n所有ID、题目、标签、KC串均逐行与GRU B0对齐。以下均是开发验证；源参考上下文按展开KC截取，B0按题截取，不能据跨架构差值孤立归因模块。GRU B0实际27/选中17；UKT未在本批重新训练，仅评估原28 epoch权重。\n\n'+table(['模型','seed','实际训练epoch','严格/原生GRU题目pooled↑','macro↑','std↓','range↓','自身尾10%↑','自身尾20%↑'],allrows,{3:'max',4:'max',5:'min',6:'min',7:'max',8:'max'})+'\nB0的macro/std优于这三个完整参考模型，MTKT pooled更高，UKT range更小。正式旧窗口推荐排名不直接外推到严格开发口径；新增模块需要在B0上独立测试，不能照搬整模型优势。\n'
    target=docs/'03_三个参考模型序列公平性比较.md';target.write_text(target.read_text().split('\n## 完成后的共同题目目标比较',1)[0]+section)
    audit.update(source_stage_complete=True,matched_training_runs=7,removed_module_training_runs=5,full_model_training_runs=2,strict_result_cases=10,actual_strict_gpu_evaluations=8,parameter_identical_prediction_reuses=2,labels_targets_match_gru_b0=True,formal_test_evaluated=False,next='Complete MTKT CIC and LEFOKT query/history GRU fusion trials, review results and iterate promising directions; goal remains active');atomic_json(r/'source_stage_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['training','strict','inputs_sha256','full_tensor_reproduction']},ensure_ascii=False))


if __name__=='__main__':main()
