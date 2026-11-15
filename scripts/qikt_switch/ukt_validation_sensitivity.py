"""Read-only inference interventions on an existing UKT, native KT fragments.

These are sensitivities, not retrained ablations or question-window metrics.
Each completed inference batch is saved; original checkpoints are never written.
"""
import argparse
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import signal
import threading
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--reference-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);config_path=a.checkpoint_dir/'config.json';config=json.loads(config_path.read_text());cfg=config['data_config'];params=config['params'];weights=a.checkpoint_dir/(params['emb_type']+'_model.ckpt')
    root=Path(__file__).resolve().parents[2];original=root/'data'/params['dataset_name']/cfg['train_valid_file']
    identity={'config':digest(config_path),'weights':digest(weights),'data':digest(original),'source':digest(Path(__file__)),'ukt_source':digest(root/'pykt/models/ukt.py'),'reference_source':digest(a.reference_source),'fold':params['fold'],'batch_size':32}
    saved=a.output/'identity.json'
    if saved.exists() and json.loads(saved.read_text())!=identity:raise ValueError('Sensitivity identity changed')
    atomic_json(saved,identity)
    copied=a.output/'native_train_valid_sequences.csv'
    if not copied.exists():shutil.copy2(original,copied)
    if digest(copied)!=identity['data']:raise ValueError('Native data copy differs')
    from pykt.datasets.data_loader import KTDataset
    dataset=KTDataset(str(copied),cfg['input_type'],{params['fold']},mode='test',aug_probs={})
    module=importlib.import_module('pykt.models.ukt');module.device=torch.device('cuda')
    signature=inspect.signature(module.UKT.__init__);settings={k:v for k,v in config['model_config'].items() if k in signature.parameters}
    settings.update(n_question=cfg['num_c'],n_pid=cfg['num_q'],emb_type=params['emb_type'])
    state=torch.load(weights,map_location='cpu',weights_only=True)
    spec=importlib.util.spec_from_file_location('pykt.models.ukt_saved_reference',a.reference_source);reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module);reference_module.device=torch.device('cuda')
    ref_signature=inspect.signature(reference_module.UKT.__init__);ref_settings={k:v for k,v in settings.items() if k in ref_signature.parameters};reference=reference_module.UKT(**ref_settings).cuda();reference.load_state_dict(state);reference.eval()
    sample={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in next(iter(DataLoader(dataset,batch_size=32))).items()}
    native=module.UKT(**settings).cuda();native.load_state_dict(state);native.eval()
    with torch.no_grad():parity=float((native(sample)-reference(sample)).abs().max())
    if parity>1e-6:raise ValueError('Saved-source reference parity failed')
    del native,reference
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    original_distance=module.wasserstein_distance_matmul;original_attention=module.uattention
    results=[]
    modes=['full','mean_distance_only','no_distance_modulation','linear_covariance_aggregation','no_rasch']
    for mode in modes:
        directory=a.output/mode;directory.mkdir(exist_ok=True)
        progress_path=directory/'progress.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'next_batch':0}
        model=module.UKT(**settings).cuda();model.load_state_dict(state);model.eval()
        module.wasserstein_distance_matmul=original_distance;module.uattention=original_attention
        if mode=='mean_distance_only':
            def mean_distance(m1,c1,m2,c2):return (m1.square().sum(-1,keepdim=True)+m2.square().sum(-1,keepdim=True).transpose(-1,-2)-2*m1@m2.transpose(-1,-2))
            module.wasserstein_distance_matmul=mean_distance
        elif mode=='no_distance_modulation':
            with torch.no_grad():
                for layer in model.model.blocks_2:layer.masked_attn_head.gammas.fill_(-100)
        elif mode=='linear_covariance_aggregation':
            source=inspect.getsource(original_attention)
            if source.count('scores ** 2')!=1:raise ValueError('Unexpected covariance aggregation source')
            source=source.replace('scores ** 2','scores');namespace={};exec(compile(source,'<registered-linear-covariance-intervention>','exec'),module.__dict__,namespace);module.uattention=namespace['uattention']
            (directory/'intervention_source.py').write_text(source)
        elif mode=='no_rasch':
            with torch.no_grad():model.difficult_param.weight.zero_()
        with torch.no_grad():
            for index,batch in enumerate(DataLoader(dataset,batch_size=32)):
                if index<progress['next_batch']:continue
                batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                prediction=model(batch)[:,1:];mask=batch['smasks'].bool();labels=batch['shft_rseqs'][mask].cpu().numpy();score=prediction[mask].cpu().numpy()
                if not np.isfinite(score).all():raise ValueError('Nonfinite sensitivity scores')
                temporary=directory/f'batch_{index:05d}.tmp';target=temporary.with_suffix('.npz')
                with temporary.open('wb') as handle:np.savez_compressed(handle,labels=labels,scores=score)
                temporary.replace(target);progress={'next_batch':index+1};atomic_json(progress_path,progress)
                if stop.is_set():return 130
        parts=[np.load(p) for p in sorted(directory.glob('batch_*.npz'))];labels=np.concatenate([p['labels'] for p in parts]);scores=np.concatenate([p['scores'] for p in parts]);value={'mode':mode,'N_concept_expanded_targets':len(labels),'native_validation_fragment_auc':float(roc_auc_score(labels,scores)),'brier':float(np.mean((scores-labels)**2))};results.append(value);atomic_json(directory/'result.json',value);del model
    module.wasserstein_distance_matmul=original_distance;module.uattention=original_attention
    atomic_json(a.output/'summary.json',{'passed':True,'results':results,'reference_source_max_prediction_error':parity,'seed':params['seed'],'fold':params['fold'],'use_CL':settings.get('use_CL'),'source_identity':identity,'scope':'Inference-only interventions, native concept-expanded validation fragments; no retraining, no orirow/windowauclate_mean or GRU question-window comparison','formal_windowauclate_mean':None})


if __name__=='__main__':raise SystemExit(main())
