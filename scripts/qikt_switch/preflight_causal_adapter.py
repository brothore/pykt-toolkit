"""Real CUDA/native-data check before admitting frozen adapter experiments."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from evaluate_validation import load_run, load_saved_model, build_validation_dataset
from prediction_progress import atomic_json, digest
from train_validation_only import initialize_from_baseline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',default=['question','prefix','multiscale','supported'])
    parser.add_argument('--retrieval',action='store_true')
    parser.add_argument('--training-only',action='store_true',
                        help='Use training-fold fragments for the disposable optimizer check')
    parser.add_argument('--kl',type=float,default=0.)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; no CPU substitution')
    config,paths=load_run(args.baseline_checkpoint)
    baseline=load_saved_model(config,paths,device='cuda')
    if args.training_only:
        from pykt.datasets.que_data_loader import KTQueDataset
        cfg=config['data_config']
        dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],
            set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={})
    else:
        dataset=build_validation_dataset(config,paths)
    data=next(iter(DataLoader(Subset(dataset,range(32)),batch_size=32)))
    with torch.no_grad():expected=baseline.predict_one_step(data)
    module=importlib.import_module('pykt.models.init_model');module.device='cuda'
    results=[]
    for mode in args.modes:
        cfg=copy.deepcopy(config['model_config']);cfg.pop('learning_rate',None)
        if args.retrieval:
            cfg['other_config'].update(retrieval_adapter_mode=mode,retrieval_adapter_scale=.25,retrieval_adapter_kl=args.kl)
        else:cfg['other_config'].update(causal_adapter_mode=mode,causal_adapter_scale=.25,causal_adapter_kl=args.kl)
        candidate=module.init_model('qikt_mamba',cfg,config['data_config'],paths['emb_type'])
        initialize_from_baseline(candidate,paths['checkpoint_path']);candidate.model.eval()
        with torch.no_grad():actual=candidate.predict_one_step(data)
        error=float((actual-expected).abs().max())
        if error>1e-6:raise ValueError(f'Zero adapter prediction mismatch: {mode} {error}')
        snapshot={k:v.detach().clone() for k,v in baseline.state_dict().items()}
        optimizer=torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad],lr=.001)
        candidate.model.train();optimizer.zero_grad();_,loss=candidate.train_one_step(data)
        loss.backward();optimizer.step()
        for key,value in snapshot.items():
            if not torch.equal(value,candidate.state_dict()[key]):raise ValueError(f'Frozen backbone changed: {key}')
        results.append({'mode':mode,'max_zero_delta_prediction_error':error,
                        'loss':float(loss.detach()),'trainable_parameters':sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                        'native_UID_type':type(dataset.dori['uid']).__name__,'frozen_update_check':True})
        del candidate,optimizer,snapshot
    atomic_json(args.output,{'passed':True,'scope':f'32 native {"training" if args.training_only else "validation"} fragments; CUDA zero residual parity and one disposable optimizer step, not efficacy',
        'checkpoint_sha256':digest(paths['checkpoint_path']),'checks':results,'device':torch.cuda.get_device_name(),
        'source_sha256':digest(Path(__file__))})


if __name__=='__main__':main()
