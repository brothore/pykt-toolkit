"""Validate query/history variants and prepare independent throughput probes."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();set_seed(42);ready=json.loads((a.root/'baseline_ready.json').read_text());config,paths=load_run(ready['checkpoint_dir']);base=load_saved_model(config,paths,device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg=config['data_config'];ds=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={});data=next(iter(DataLoader(ds,batch_size=32)))
    with torch.no_grad():expected=base.predict_one_step(data)
    factory=importlib.import_module('pykt.models.init_model');factory.device='cuda';checks=[];probes={}
    for mode,weight in [('scalar_query',1.),('shared_history',.75),('shared_history',.5),('shared_history',0.),('shared_scalar',.5),('shared_history',1.)]:
        current=copy.deepcopy(config['model_config']);current.pop('learning_rate',None);current['other_config'].update(reference_fusion_mode=mode,reference_fusion_scale=.1,reference_history_weight=weight,reference_cic_context='past',reference_cic_kernel=3,reference_fusion_bias='none')
        model=factory.init_model('qikt_mamba',current,cfg,paths['emb_type']);missing=model.load_state_dict(base.state_dict(),strict=False)
        if missing.unexpected_keys or any(not k.startswith('model.reference_fusion.') for k in missing.missing_keys):raise ValueError('Unexpected base-state mismatch')
        model.model.eval()
        with torch.no_grad():error=float((model.predict_one_step(data)-expected).abs().max())
        identity_expected=mode=='scalar_query' or mode=='shared_history' and weight==1.
        if not torch.isfinite(torch.tensor(error)) or identity_expected and error>1e-6:raise ValueError('Invalid query/history initial prediction')
        if mode in ['scalar_query','shared_scalar']:
            folder=a.root/'resource_probes'/mode;folder.mkdir(parents=True,exist_ok=True);saved=copy.deepcopy(config);saved['model_config']=current;saved['params']['other_config']=current['other_config'];saved['probe_metadata']={'scientific_training':False,'baseline_sha256':digest(paths['checkpoint_path'])};atomic_json(folder/'config.json',saved);torch.save(model.state_dict(),folder/'attn_model.ckpt')
            atomic_json(folder/'protocol.json',{'workloads':['training','inference'],'concurrencies':[1,2,3],'repeats':[1,2],'mixed_phases':False,'worker':{'checkpoint_dir':str(folder),'batch_size':32,'global_batches':120,'threads':2,'warmup':6},'scope':'Query scalar and shared-history/query scalar representative dual-GRU loads'})
            probes[mode]={'checkpoint':str(folder),'protocol':str(folder/'protocol.json')}
        opt=torch.optim.Adam(model.parameters(),lr=.0001)
        for _ in range(2):
            model.model.train();opt.zero_grad();_,loss=model.train_one_step(data)
            if not torch.isfinite(loss):raise ValueError('Nonfinite query/history loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite query/history gradient')
            opt.step()
        checks.append({'mode':mode,'history_weight':weight,'initial_prediction_max_error':error,'identity_expected':identity_expected,'two_steps_finite':True,'parameters':sum(p.numel() for p in model.parameters())});del model,opt
    atomic_json(a.root/'native_preflight.json',{'passed':True,'checks':checks,'scientific_training':False,'training_fragments':32});atomic_json(a.root/'probe_ready.json',{'passed':True,'cases':probes})


if __name__=='__main__':main()
