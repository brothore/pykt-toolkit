"""Reuse completed strict predictions only after exact inference-state checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import torch
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-spec',type=Path,required=True);p.add_argument('--target-spec',type=Path,required=True);a=p.parse_args()
    src=json.loads(a.source_spec.read_text());dst=json.loads(a.target_spec.read_text());source=Path(src['output']);out=Path(dst['output']);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((source/'validation_question_results.json').read_text());ca=json.loads(Path(src['checkpoint_dir'],'config.json').read_text());cb=json.loads(Path(dst['checkpoint_dir'],'config.json').read_text())
    for c in [ca,cb]:
        if c.pop('reference_ablation_mode','full')!='full':raise ValueError('Reuse is limited to complete reference models')
    if ca!=cb:raise ValueError('Inference configuration or seed differs')
    for k in ['raw_file','source','batch_size']:
        if src[k]!=dst[k]:raise ValueError('Raw data/source/batch execution differs')
    cpname=ca['params']['emb_type']+'_model.ckpt';wa=Path(src['checkpoint_dir'])/cpname;wb=Path(dst['checkpoint_dir'])/cpname
    x=torch.load(wa,map_location='cpu',weights_only=True);y=torch.load(wb,map_location='cpu',weights_only=True)
    if x.keys()!=y.keys() or any(t.dtype!=y[k].dtype or t.shape!=y[k].shape or not torch.equal(t,y[k]) for k,t in x.items()):raise ValueError('Inference parameter tensors are not identical')
    for code,checksum in old['identity']['code'].items():
        # Both tasks were registered for the same frozen inference implementation.
        original_plan=json.loads((a.target_spec.parent/'plan_initial_inference.json').read_text())
        paths=[Path(name) for name,value in original_plan['source_sha256'].items() if Path(name).name==code and value==checksum]
        if not paths or any(digest(path)!=checksum for path in paths):raise ValueError('Strict evaluation source differs')
    if digest(Path(src['raw_file']))!=old['identity']['raw'] or digest(Path(src['source']))!=old['identity']['source']:raise ValueError('Frozen data/model source changed')
    evidence={'method':'Exact inference parameter tensors, full configuration, source, input data and strict evaluation implementation identical; no new inference','tensor_count':len(x),'original_weights_sha256':digest(wa),'target_weights_sha256':digest(wb),'source_result_sha256':digest(source/'validation_question_results.json'),'source_task_spec_sha256':digest(a.source_spec),'target_task_spec_sha256':digest(a.target_spec),'training_runs_remain_distinct':True}
    for name in ['window_predictions.tsv','sequence_auc.csv']:
        target=out/name
        if target.exists():
            if digest(target)!=digest(source/name):raise ValueError('Target predictions already differ')
        else:shutil.copy2(source/name,target)
    stats=json.loads(Path(dst['checkpoint_dir'],'training_stats.json').read_text());result=dict(old);result.update(completed_epochs=stats['completed_epochs'],selected_epoch=stats['selected_epoch'],reused_identical_predictions=evidence,original_inference_identity=old['identity'],identity={**old['identity'],'spec':digest(a.target_spec),'weights':digest(wb),'reuse_source':digest(Path(__file__))});atomic_json(out/'reuse_evidence.json',evidence);atomic_json(out/'validation_question_results.json',result)


if __name__=='__main__':main()
