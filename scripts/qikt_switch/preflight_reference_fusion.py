"""Disposable real-training-data CUDA checks; no scientific efficacy claim."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from evaluate_validation import load_run,load_saved_model
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline-ready',type=Path,required=True);a=p.parse_args();set_seed(42);a.root.mkdir(parents=True,exist_ok=True)
    ready=json.loads(a.baseline_ready.read_text());config,paths=load_run(ready['checkpoint_dir']);base=load_saved_model(config,paths,device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg=config['data_config'];dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={});data=next(iter(DataLoader(dataset,batch_size=32)))
    with torch.no_grad():expected=base.predict_one_step(data)
    factory=importlib.import_module('pykt.models.init_model');factory.device='cuda';results=[]
    for mode in ['cic_question','cic_concept','cic_both','time_input','time_decode','time_gate','alibi_logit','alibi_question','alibi_concept']:
        current=copy.deepcopy(config['model_config']);current.pop('learning_rate',None);current['other_config'].update(reference_fusion_mode=mode,reference_fusion_scale=.1,reference_fusion_bias='alibi',reference_cic_kernel=3)
        model=factory.init_model('qikt_mamba',current,cfg,paths['emb_type']);missing=model.load_state_dict(base.state_dict(),strict=False)
        if missing.unexpected_keys or any(not x.startswith('model.reference_fusion.') for x in missing.missing_keys):raise ValueError('Unexpected baseline mismatch')
        model.model.eval()
        with torch.no_grad():error=float((model.predict_one_step(data)-expected).abs().max())
        if error>1e-6:raise ValueError('Reference fusion zero initialization differs')
        optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
        for _ in range(2):
            model.model.train();optimizer.zero_grad();_,loss=model.train_one_step(data)
            if not torch.isfinite(loss):raise ValueError('Nonfinite reference fusion loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite reference fusion gradients')
            optimizer.step()
        results.append({'mode':mode,'zero_prediction_error':error,'two_training_steps_finite':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        atomic_json(a.root/'preflight_progress.json',{'completed':results,'scientific_training':False});del model,optimizer
    atomic_json(a.root/'preflight.json',{'passed':True,'checks':results,'training_fragments':32,'baseline_sha256':digest(paths['checkpoint_path']),'scientific_training':False,'scope':'Technical interface checks before module selection; no direction selected or efficacy result'})


if __name__=='__main__':main()
