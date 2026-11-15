"""Real training-data preflight and disposable matched-load GPU probes."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
from evaluate_validation import load_run,load_saved_model
from prediction_progress import atomic_json,digest
from cooperative_process import run_cooperative


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;ready=json.loads((root/'baseline_ready.json').read_text())
    import pandas as pd
    from compare_validation import compare
    from sequence_balance_metrics import assess_balance
    frame=pd.read_csv(ready['predictions'],sep='\t',dtype={'questions':str,'concepts':str})
    comparison,paired=compare(frame,frame,500,metric_key='validation_window_auc')
    control=assess_balance(paired,0.)
    folder=root/'B0/round1';training=json.loads((folder/'baseline/round_summary.json').read_text())
    control.update(protocol=json.loads((folder/'protocol.json').read_text()),comparison=comparison,training=training,substantive_development_signal=False)
    paired.to_csv(folder/'paired_orirow.csv');atomic_json(folder/'balance.json',control)
    config,paths=load_run(ready['checkpoint_dir']);baseline=load_saved_model(config,paths,device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg=config['data_config'];dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={})
    data=next(iter(DataLoader(dataset,batch_size=32)))
    with torch.no_grad():expected=baseline.predict_one_step(data)
    checks=[];probes=root/'resource_probes';probes.mkdir(exist_ok=True)
    module=importlib.import_module('pykt.models.init_model');module.device='cuda'
    for mode,fusion in [('mean','additive'),('w2','additive'),('w2','uncertainty'),('w2_legacy','additive')]:
        model_cfg=copy.deepcopy(config['model_config']);model_cfg.pop('learning_rate',None)
        model_cfg['other_config'].update(ukt_transfer_mode=mode,ukt_transfer_scale=.1,ukt_transfer_fusion=fusion)
        candidate=module.init_model('qikt_mamba',model_cfg,cfg,paths['emb_type'])
        result=candidate.load_state_dict(baseline.state_dict(),strict=False)
        if result.unexpected_keys or any(not k.startswith('model.ukt_transfer.') for k in result.missing_keys):raise ValueError('Unexpected transfer warm-start keys')
        candidate.model.eval()
        with torch.no_grad():actual=candidate.predict_one_step(data)
        error=float((actual-expected).abs().max())
        if error>1e-6:raise ValueError('Zero transfer prediction differs from GRU baseline')
        if mode=='w2_legacy':
            probe=probes/'gru_ukt_heavy';probe.mkdir(exist_ok=True)
            probe_config=copy.deepcopy(config);probe_config['model_config'].update(model_cfg)
            probe_config['params']['other_config']=model_cfg['other_config']
            probe_config['probe_metadata']={'scientific_training':False,'origin_baseline_sha256':digest(paths['checkpoint_path']),'purpose':'Disposable representative heavy UKT-transfer timing probe, zero correction before any probe update'}
            atomic_json(probe/'config.json',probe_config);torch.save(candidate.state_dict(),probe/'attn_model.ckpt')
        optimizer=torch.optim.Adam(candidate.parameters(),lr=1e-4)
        for _ in range(2):
            candidate.model.train();optimizer.zero_grad();_,loss=candidate.train_one_step(data)
            if not torch.isfinite(loss):raise ValueError('Nonfinite native training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in candidate.parameters()):raise ValueError('Nonfinite native gradients')
            optimizer.step()
        checks.append({'mode':mode,'fusion':fusion,'zero_residual_max_error':error,'finite_two_step_training':True,'UID_type':type(dataset.dori['uid']).__name__})
        del candidate,optimizer
    atomic_json(root/'native_transfer_preflight.json',{'passed':True,'training_fragments':32,'optimizer_steps_per_disposable_probe':2,'checks':checks,'baseline_checkpoint_sha256':digest(paths['checkpoint_path']),'scope':'Implementation check only, probe weights not scientific results'})
    del baseline;torch.cuda.empty_cache()
    results={}
    for name,checkpoint in [('dual_gru',Path(ready['checkpoint_dir'])),('gru_ukt_heavy',probes/'gru_ukt_heavy')]:
        out=root/'resource_calibration'/name;out.mkdir(parents=True,exist_ok=True)
        protocol={'workloads':['training','inference'],'concurrencies':[1,2,3],'repeats':[1,2],'mixed_phases':False,'worker':{'checkpoint_dir':str(checkpoint),'batch_size':32,'global_batches':120,'threads':2,'warmup':6},'scope':'New trainable dual GRU / representative heaviest UKT transfer, no scientific probe results'}
        atomic_json(out/'protocol.json',protocol)
        # The coordinator must release its own CUDA context before exclusive timing.
        results[name]={'checkpoint':str(checkpoint),'protocol':str(out/'protocol.json')}
    atomic_json(root/'probe_ready.json',{'passed':True,'cases':results})


if __name__=='__main__':main()
