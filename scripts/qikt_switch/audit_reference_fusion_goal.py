"""Verify scientific artifacts and report coverage before a human completion audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from audit_inputs import sequence_metrics
from sequence_balance_metrics import assess_balance
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo;ref=repo/'logs/advanced_reference_20261115';out=ref/'goal_final_audit';out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text());problems=[];verified=[]
    def require(condition,description):
        if not condition:problems.append(description)
    history=read(ref/'source_stage_audit.json');prefixes={}
    for name,expected in history['documents_sha256'].items():
        path=Path(name);text=path.read_text()
        if path.name.startswith('01_'):text=text.split('\n## 双GRU迁移后的更新',1)[0]
        if path.name.startswith('02_'):text=text.split('\n## 补充消融完成',1)[0]
        match=hashlib.sha256(text.encode()).hexdigest()==expected;prefixes[name]=match
        require(match,'Original archived report content changed: '+name)
    require(history['source_stage_complete'] and history['labels_targets_match_gru_b0'],'Original source stage not verified')
    source=repo/'logs/lefokt_extended_ablation_v2_fix01_20261115';registration=read(source/'registration.json')
    for mode in registration['modes']:
        require(read(source/'jobs'/mode/'state.json')['status']=='completed','Incomplete source: '+mode)
        result=read(source/'strict_questions'/mode/'validation_question_results.json');require(result['events']==221348 and result['cohorts']['primary_20_5']['N']==773,'Source target coverage differs: '+mode)
    selection=read(ref/'lefokt_complete_selection/module_selection_ready.json');require(selection['source_conditions']==10 and selection['audit_passed'],'Incomplete source choice audit')
    require(len(read(ref/'lefokt_factorial/summary.json')['conditional_effects'])==12,'Incomplete factorial effects')
    roots=['qikt_gru_ukt_20261114','qikt_gru_ukt_v2_20261114','qikt_gru_ukt_v3_20261114','qikt_mtkt_cic_v1_20261115','qikt_lefokt_query_v1_20261115','qikt_mtkt_time_v2_20261115','qikt_sequence_guided_v3_20261115','qikt_lefokt_additive_v3_20261115']
    basepath=repo/'logs/qikt_gru_ukt_20261114/B0/round1/baseline/validation_window/window_predictions.tsv';keys=['orirow','qidx','late_trues','questions','concepts']
    def frame(path):return pd.read_csv(path,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True)
    base=frame(basepath);basekeys=base[keys];counts={}
    _,base_sequences=sequence_metrics(base);base_auc=float(roc_auc_score(base.late_trues,base.late_mean))
    for name in roots:
        root=repo/'logs'/name;reg=read(root/'registration.json');count=0
        for family in reg['families']:
            if family=='B0':continue
            status=read(root/'jobs'/family/'state.json')['status'];require(status=='completed','Incomplete fusion family: '+name+'/'+family)
            paths=sorted((root/family).glob('round*/balance.json'))
            require(bool(paths),'Missing scientific results: '+name+'/'+family)
            require(len(paths)<=3,'Unbounded direction: '+name+'/'+family)
            found=False
            for path in paths:
                require(not found,'A further round ran after first candidate: '+str(path));v=read(path);found=v['exploratory_candidate_pass'];count+=1
                predictions=path.parent/'baseline/validation_window/window_predictions.tsv';current=frame(predictions)
                require(len(current)==221348 and not current.duplicated(['orirow','qidx']).any(),'Prediction coverage/duplicates: '+str(path))
                require(current[keys].equals(basekeys),'Target/label/KC alignment differs: '+str(path))
                pooled=float(roc_auc_score(current.late_trues,current.late_mean));cohorts,sequences=sequence_metrics(current);c=cohorts['primary_20_5'];require(c['N']==773,'Primary cohort differs: '+str(path))
                require(abs(pooled-v['comparison']['candidate_validation_window_auc'])<1e-10,'Pooled result differs: '+str(path))
                for field,key in [('mean','macro_auc'),('std','std_ddof0'),('range','range')]:require(abs(c[field]-v['candidate'][key])<1e-10,'Sequence metric differs '+key+': '+str(path))
                paired=sequences.set_index('orirow').join(base_sequences.set_index('orirow').auc.rename('baseline_auc'))
                paired['delta_auc']=paired.auc-paired.baseline_auc
                fresh=assess_balance(paired,pooled-base_auc)
                fresh['checks'].update(fixed_weak20_preserved=fresh['deltas']['fixed_baseline_bottom20_mean']>=0,own_tail20_preserved=fresh['deltas']['own_bottom20_mean']>=0)
                require(all(fresh['checks'].values())==v['exploratory_candidate_pass'],'Candidate gate differs: '+str(path))
                require(fresh['worsened']==v['worsened'] and fresh['decline_counts']==v['decline_counts'],'Individual regression counts differ: '+str(path))
                for key,value in fresh['deltas'].items():require(abs(value-v['deltas'][key])<1e-10,'Protection/dispersion delta differs '+key+': '+str(path))
                checkpoint=Path(v['training']['checkpoint_dir']);cfg=read(checkpoint/'config.json');version=cfg['model_config'].get('version',cfg['params'].get('version'));require(version=='gru','Non-GRU config: '+str(path))
                require(cfg['params']['fold']==0 and cfg['params']['seed']==42,'Fusion fold/seed differs: '+str(path))
                weights=checkpoint/'attn_model.ckpt';state=torch.load(weights,map_location='cpu',weights_only=True)
                qh=state['model.que_lstm_layer.weight_hh_l0'];ch=state['model.concept_lstm_layer.weight_hh_l0']
                require(qh.shape[0]==3*qh.shape[1] and ch.shape[0]==3*ch.shape[1],'Not GRU three-gate tensors: '+str(path))
                require(qh.untyped_storage().data_ptr()!=ch.untyped_storage().data_ptr(),'Shared GRU storage: '+str(path))
                verified.append({'trial':str(path),'pooled_recomputed':pooled,'primary_recomputed':c,'predictions_sha256':digest(predictions),'weights_sha256':digest(weights),'actual_version':version,'three_gate_independent_gru_storage':True,'candidate':found})
                del state,current
            counts[name]=count
    comparison=read(repo/'logs/qikt_lefokt_additive_v3_20261115/matched_comparisons/summary.json');la_count=counts[roots[-1]]
    require(len(comparison['comparisons'])==5*la_count,'Missing additive B0/multiplicative/frontier comparisons')
    for name in ['01_MTKT消融与序列公平性.md','02_LEFOKT消融与序列公平性.md','03_三个参考模型序列公平性比较.md','13_三模型消融与全部GRU融合汇总.md','20_LEFOKT完整消融与迁移选择.md','22_GRU候选的公平性取舍.md','23_LEFOKT八格条件效应.md','24_LEFOKT加性迁移的匹配比较.md']:
        path=repo/'docs/先进模型消融与GRU融合'/name;require(path.exists() and path.stat().st_size>100,'Missing required document '+name)
    report=(repo/'docs/先进模型消融与GRU融合/13_三模型消融与全部GRU融合汇总.md').read_text();require('加性迁移是否带来独立增益' in report,'Central report misses matched controls');require('### 哪些序列受益' in report and 'UID' in report,'Central report misses sequence beneficiaries')
    frontier=read(ref/'gru_candidate_frontier/summary.json');require(any(r['value'].get('substantive_development_signal') for r in frontier['records']),'No substantive development signal found')
    value={'audit_passed':not problems,'goal_complete':False,'requires_primary_agent_review':True,'problems':problems,'archived_report_prefixes_preserved':prefixes,'original_source_stage':{'matched_trainings':history['matched_training_runs'],'strict_cases':history['strict_result_cases']},'additional_lefokt_source_conditions':7,'complete_lefokt_source_conditions':10,'conditional_effects':12,'fusion_trial_counts':counts,'total_completed_fusion_trainings':len(verified),'recomputed_trials':verified,'source_choice':selection['selected_mode'],'source_choice_status':selection['selection_status'],'formal_windowauclate_mean':None,'stability_validated':False,'limits':'Single-seed development candidates; group mean protection, not zero individual regression. Historical UKT source interventions are not retrained ablations. References count expanded concepts; GRU counts questions.'}
    atomic_json(out/'audit.json',value);print('Completion evidence audit:',not problems,'trials',len(verified),'problems',len(problems))
    if problems:print('\n'.join(problems[:20]))


if __name__=='__main__':main()
