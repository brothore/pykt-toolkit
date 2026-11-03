"""Run one preregistered nips fold0 experiment; refuse CPU substitution.
No automatic sweeps or test-based parameter selection. Existing output is retained.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util
from audit_inputs import sha

VARIANTS={'baseline':{},'s1_item_residual':{'item_residual':1},'s2_history_drop005':{'history_response_dropout':.05},'s2_history_drop010':{'history_response_dropout':.1},'s3_separate_branches':{'separate_mamba_branches':1}}
VARIANTS.update({'s4_concept_memory':{'concept_memory_residual':1},
                 's5_pairwise_001':{'pairwise_loss_weight':.01},
                 's5_pairwise_005':{'pairwise_loss_weight':.05}})
VARIANTS.update({'s6_head_fusion_010':{'head_fusion_residual':1,'fusion_gate_scale':.1},
                 's6_head_fusion_005':{'head_fusion_residual':1,'fusion_gate_scale':.05}})
VARIANTS.update({'s7_overlap64':{'training_overlap':64}})
VARIANTS.update({'s2_history_drop001':{'history_response_dropout':.01},
                 's5_pairwise_0002':{'pairwise_loss_weight':.002}})
VARIANTS.update({'r2_mask005_rank005':{'history_response_dropout':.05,'pairwise_loss_weight':.05},
                 'r2_overlap64_rank005':{'training_overlap':64,'pairwise_loss_weight':.05}})
VARIANTS.update({'r3_frozen_memory_fusion_anchor1':{'concept_memory_residual':1,'head_fusion_residual':1,
    'freeze_backbone':1,'anchor_kl_weight':1.,'learning_rate':.001,'warm_start_baseline':1}})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=list(VARIANTS),required=True);ap.add_argument('--dataset',choices=['nips_task34','peiyou'],default='nips_task34');ap.add_argument('--fold',type=int,choices=range(5),default=0);ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103'
    neural_root=base/('neural' if args.dataset=='nips_task34' and args.fold==0 else f'neural_{args.dataset}_fold{args.fold}')
    out=neural_root/args.variant;out.mkdir(parents=True,exist_ok=True)
    config_name=args.dataset+'.json'
    cfg=json.loads((root/'configs/qikt_switch'/config_name).read_text());params=cfg['params'];other=params.pop('other_config');params.update(other)
    params.update(dataset_name=args.dataset,fold=args.fold)
    setting=dict(VARIANTS[args.variant]);overlap=setting.pop('training_overlap',0)
    warm_start=setting.pop('warm_start_baseline',0)
    params.update(use_trained=0,add_uuid=0,save_dir=str(out/'saved_model'));params.update(setting)
    child_env=os.environ.copy();child_env.pop('QIKT_SWITCH_OVERLAP_TRAIN',None)
    child_env.pop('QIKT_SWITCH_BASE_CHECKPOINT',None)
    warm_start_metadata=None
    if warm_start:
        selected=json.loads((neural_root/'baseline/round_summary.json').read_text())
        checkpoint=Path(selected['checkpoint_dir'])/'attn_model.ckpt'
        child_env['QIKT_SWITCH_BASE_CHECKPOINT']=str(checkpoint)
        warm_start_metadata={'path':str(checkpoint),'sha256':sha(checkpoint)}
    training_override=None
    if overlap:
        prepared=base/f'overlap_{args.dataset}_fold{args.fold}_{overlap}'
        if not (prepared/'manifest.json').exists():
            with open(out/'prepare_overlap.log','x') as log:
                subprocess.run([sys.executable,str(root/'scripts/qikt_switch/prepare_overlap.py'),'--dataset',args.dataset,'--fold',str(args.fold),'--overlap',str(overlap)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads((prepared/'manifest.json').read_text())
        source=prepared/'train_sequences.csv'
        if manifest['generated_sha256']!=sha(source) or not manifest['target_sets_equal']:
            raise SystemExit('Overlap input is not verified')
        child_env['QIKT_SWITCH_OVERLAP_TRAIN']=str(source)
        training_override={'path':str(source),'manifest':manifest}
    params.update(history_response_dropout=params.get('history_response_dropout',0.),item_residual=params.get('item_residual',0),separate_mamba_branches=params.get('separate_mamba_branches',0))
    # Keep all predeclared scientific settings; only the three experiment switches vary.
    command=[sys.executable,str(root/'scripts/qikt_switch/train_validation_only.py')]
    for k,v in params.items():command.extend(['--'+k,str(v)])
    (out/'requested_run.json').write_text(json.dumps({'command':command,'cwd':str(root/'examples'),'params':params,'training_override':training_override,'warm_start':warm_start_metadata,'post_training_test_inference':False,'status':'preflight_only' if args.preflight_only else 'requested'},indent=2))
    import torch
    checks={'cuda_available':torch.cuda.is_available(),'mamba_ssm_installed':importlib.util.find_spec('mamba_ssm') is not None,'torch':torch.__version__}
    (out/'preflight.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks),flush=True)
    if not all(checks[k] for k in ('cuda_available','mamba_ssm_installed')):
        raise SystemExit('Formal Mamba training unavailable; no CPU or GRU substitution was run.')
    if args.preflight_only:return
    # Permit only the next explicitly invoked experiment, not an automatic GPU queue.
    marker=out/'training_started.json'
    if marker.exists():raise SystemExit('Existing training record: inspect it before running again.')
    source_files=[root/'pykt/models/qikt_mamba.py',root/'pykt/models/qikt_switch_utils.py',
        root/'pykt/models/train_model.py',root/'pykt/models/evaluate_model.py',root/'pykt/models/init_model.py',
        root/'pykt/datasets/que_data_loader.py',root/'examples/wandb_train.py',root/'examples/wandb_qikt_mamba_train.py',
        root/'scripts/qikt_switch/train_validation_only.py',root/'scripts/qikt_switch/run_neural.py']
    (out/'trial_source_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in source_files},indent=2))
    marker.write_text(json.dumps({'pid':os.getpid(),'command':command},indent=2))
    with open(out/'training.log','x') as log:
        result=subprocess.run(command,cwd=root/'examples',env=child_env,stdout=log,stderr=subprocess.STDOUT)
    (out/'training_exit.json').write_text(json.dumps({'returncode':result.returncode},indent=2))
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
