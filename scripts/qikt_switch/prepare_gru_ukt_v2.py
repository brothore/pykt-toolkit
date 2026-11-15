"""Real-data checks and representative probes for Rasch/frozen transfers."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;set_seed(42)
    ready=json.loads((root/'baseline_ready.json').read_text());config,paths=load_run(ready['checkpoint_dir']);baseline=load_saved_model(config,paths,device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg=config['data_config'];dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={})
    data=next(iter(DataLoader(dataset,batch_size=32)))
    with torch.no_grad():expected=baseline.predict_one_step(data)
    factory=importlib.import_module('pykt.models.init_model');factory.device='cuda';checks=[];cases={}
    definitions=[('S0',{'ukt_rasch_mode':'input_scalar','ukt_rasch_scale':1.},False,None),
                 ('S1',{'ukt_rasch_mode':'input_vector','ukt_rasch_scale':1.},False,'gru_rasch_input'),
                 ('S2',{'ukt_rasch_mode':'output_vector','ukt_rasch_scale':.1,'ukt_rasch_freeze':1},True,None),
                 ('S3',{'ukt_transfer_mode':'w2','ukt_transfer_scale':.1,'ukt_transfer_freeze':1},True,'gru_w2_frozen')]
    for name,settings,freeze,case in definitions:
        model_cfg=copy.deepcopy(config['model_config']);model_cfg.pop('learning_rate',None);model_cfg['other_config'].update(settings)
        candidate=factory.init_model('qikt_mamba',model_cfg,cfg,paths['emb_type'])
        result=candidate.load_state_dict(baseline.state_dict(),strict=False)
        if result.unexpected_keys or any(not k.startswith(('model.rasch_transfer.','model.ukt_transfer.')) for k in result.missing_keys):raise ValueError('Unexpected new factor keys')
        candidate.model.eval()
        with torch.no_grad():error=float((candidate.predict_one_step(data)-expected).abs().max())
        if error>1e-6:raise ValueError('Zero initialization differs')
        if case:
            probe=root/'resource_probes'/case;probe.mkdir(parents=True,exist_ok=True)
            probe_config=copy.deepcopy(config);probe_config['model_config'].update(model_cfg);probe_config['params']['other_config']=model_cfg['other_config']
            probe_config['probe_metadata']={'scientific_training':False,'origin_baseline_sha256':digest(paths['checkpoint_path']),'freeze':freeze}
            atomic_json(probe/'config.json',probe_config);torch.save(candidate.state_dict(),probe/'attn_model.ckpt')
            protocol={'workloads':['training','inference'],'concurrencies':[1,2,3],'repeats':[1,2],'mixed_phases':False,'worker':{'checkpoint_dir':str(probe),'batch_size':32,'global_batches':120,'threads':2,'warmup':6},'scope':'New V2 representative trainable Rasch input and frozen W2'}
            atomic_json(probe/'protocol.json',protocol);cases[case]={'checkpoint':str(probe),'protocol':str(probe/'protocol.json')}
        optimizer=torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad],lr=.001 if freeze else .0001)
        for _ in range(2):
            candidate.model.train();optimizer.zero_grad();_,loss=candidate.train_one_step(data)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in candidate.parameters()):raise ValueError('Nonfinite gradients')
            optimizer.step()
        if freeze and any(not torch.equal(v,candidate.state_dict()[k]) for k,v in baseline.state_dict().items()):raise ValueError('Frozen GRU changed')
        checks.append({'family':name,'zero_error':error,'finite_two_steps':True,'frozen_backbone_unchanged':True if freeze else None,'trainable_parameters':sum(p.numel() for p in candidate.parameters() if p.requires_grad)})
        del candidate,optimizer
    atomic_json(root/'native_preflight.json',{'passed':True,'checks':checks,'training_fragments':32,'baseline_checkpoint_sha256':digest(paths['checkpoint_path']),'scope':'Disposable checks, no efficacy claim'})
    atomic_json(root/'probe_ready.json',{'passed':True,'cases':cases})


if __name__=='__main__':main()
