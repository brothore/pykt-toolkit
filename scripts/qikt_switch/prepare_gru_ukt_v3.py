"""Native-data preflight and probes for decoder placement/reference losses."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from pykt.models.qikt_baseline_anchor import install_baseline_anchor
from evaluate_validation import load_run,load_saved_model
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;set_seed(42)
    ready=json.loads((root/'baseline_ready.json').read_text());config,paths=load_run(ready['checkpoint_dir']);base=load_saved_model(config,paths,device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg=config['data_config'];dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={});data=next(iter(DataLoader(dataset,batch_size=32)))
    with torch.no_grad():expected=base.predict_one_step(data)
    factory=importlib.import_module('pykt.models.init_model');factory.device='cuda';checks=[];cases={}
    for name,mode,warm,case in [('C0','input_additive',False,None),('Q0','query_vector',False,'gru_rasch_query'),('A0','query_vector',True,None),('A1','query_vector',True,'gru_rasch_anchor_pair')]:
        model_cfg=copy.deepcopy(config['model_config']);model_cfg.pop('learning_rate',None);model_cfg['other_config'].update(ukt_rasch_mode=mode,ukt_rasch_scale=2. if name!='Q0' else 1.,ukt_rasch_warmstart=int(warm))
        model=factory.init_model('qikt_mamba',model_cfg,cfg,paths['emb_type']);missing=model.load_state_dict(base.state_dict(),strict=False)
        if missing.unexpected_keys or any(not k.startswith('model.rasch_transfer.') for k in missing.missing_keys):raise ValueError('Unexpected prototype keys')
        model.model.eval()
        with torch.no_grad():error=float((model.predict_one_step(data)-expected).abs().max())
        if error>1e-6:raise ValueError('Prototype zero initialization differs')
        objective={'initial_checkpoint':str(paths['checkpoint_path']),'baseline_anchor_kl':1.,'baseline_pair_weight':1. if name=='A1' else 0.}
        if case:
            folder=root/'resource_probes'/case;folder.mkdir(parents=True,exist_ok=True);probe=copy.deepcopy(config);probe['model_config'].update(model_cfg);probe['params']['other_config']=model_cfg['other_config'];probe['probe_metadata']={'scientific_training':False,'origin_baseline_sha256':digest(paths['checkpoint_path'])}
            if warm:probe['probe_training_objective']=objective
            atomic_json(folder/'config.json',probe);torch.save(model.state_dict(),folder/'attn_model.ckpt')
            protocol={'workloads':['training','inference'],'concurrencies':[1,2,3],'repeats':[1,2],'mixed_phases':False,'worker':{'checkpoint_dir':str(folder),'batch_size':32,'global_batches':120,'threads':2,'warmup':6},'scope':'V3 current-question modulation and heaviest KL+pair training objective'}
            atomic_json(folder/'protocol.json',protocol);cases[case]={'checkpoint':str(folder),'protocol':str(folder/'protocol.json')}
        teacher_metadata=install_baseline_anchor(model,objective) if warm else None
        keys=set(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.0001)
        for _ in range(2):
            model.model.train();opt.zero_grad();_,loss=model.train_one_step(data)
            if not torch.isfinite(loss):raise ValueError('Nonfinite native objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite native gradients')
            opt.step()
        if set(model.state_dict())!=keys:raise ValueError('Training teacher leaked into inference checkpoint')
        checks.append({'family':name,'zero_prediction_error':error,'finite_two_steps':True,'teacher_metadata':teacher_metadata,'student_tensor_count':len(keys)})
        del model,opt
    atomic_json(root/'native_preflight.json',{'passed':True,'checks':checks,'training_fragments':32,'baseline_checkpoint_sha256':digest(paths['checkpoint_path']),'scope':'Disposable training-data checks, not efficacy'})
    atomic_json(root/'probe_ready.json',{'passed':True,'cases':cases})


if __name__=='__main__':main()
