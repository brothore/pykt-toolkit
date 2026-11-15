"""Train a registered reference-model ablation, then evaluate selected weights."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from prediction_progress import atomic_json,digest
from reference_ablation_models import load_reference,intervene
from reference_training import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    from container_memory_budget import container_budget
    while True:
        budget=container_budget();probe=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10);free=int(probe.strip());admitted=budget['conservative_available_gib']>=10 and free>=6144
        atomic_json(out/'admission.json',{'admitted':admitted,'container':budget,'gpu_free_mib':free,'minimum_effective_host_gib':10,'minimum_gpu_free_mib':6144,'max_concurrent_trials':2})
        if admitted:break
        time.sleep(5)
    original=Path(spec['original_config']);config=json.loads(original.read_text());params=config['params'];cfg=config['data_config'];set_seed(params['seed'])
    if params['model_name']=='mtkt':
        from pykt.datasets.dkt_forget_dataloader import DktForgetDataset
        factory=lambda folds:DktForgetDataset(spec['data_file'],cfg['input_type'],folds)
    else:
        from pykt.datasets.data_loader import KTDataset
        factory=lambda folds:KTDataset(spec['data_file'],cfg['input_type'],folds,mode='test',aug_probs={})
    training=factory(set(cfg['folds'])-{params['fold']});validation=factory({params['fold']});batch=int(params['batch_size']);train_loader=DataLoader(training,batch_size=batch);valid_loader=DataLoader(validation,batch_size=batch)
    model,module=load_reference(spec['source'],config);patch=intervene(model,module,spec['mode'])
    if patch:(out/'intervention_forward.py').write_text(patch)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config['model_config']['learning_rate']))
    identity={'spec':digest(a.spec),'original_config':digest(original),'data':digest(Path(spec['data_file'])),'source':digest(Path(spec['source'])),'trainer':digest(Path(__file__).with_name('reference_training.py')),'adapter':digest(Path(__file__).with_name('reference_ablation_models.py')),'native_semantics':'BCE FP64; LEFOKT Rasch regularizer; Adam; sequential native fragments; delta>1e-3 selection, patience10, no test inference; complete epoch resume; no AMP/scheduler'}
    atomic_json(out/'identity.json',identity)
    config['reference_ablation_mode']=spec['mode'];atomic_json(out/'config.json',config)
    code=train(model,train_loader,valid_loader,optimizer,out,identity,max_epochs=int(params['num_epochs']))
    if code:return code
    shutil.copy2(out/'best_model.ckpt',out/(params['emb_type']+'_model.ckpt'))
    stats=json.loads((out/'training_stats.json').read_text());evalspec={'checkpoint_dir':str(out),'source':spec['source'],'data_file':spec['data_file'],'output':str(out/'native_evaluation'),'modes':['full'],'batch_size':32,'original_epochs':stats['completed_epochs']}
    atomic_json(out/'evaluation_spec.json',evalspec)
    subprocess.run([sys.executable,str(Path(__file__).with_name('reference_fragment_ablation.py')),'--spec',str(out/'evaluation_spec.json')],check=True)
    atomic_json(out/'completion.json',{'completed':True,'mode':spec['mode'],'training':stats,'selected_checkpoint_sha256':digest(out/'best_model.ckpt'),'test_evaluated':False,'formal_windowauclate_mean':None})


if __name__=='__main__':raise SystemExit(main())
