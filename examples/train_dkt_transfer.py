"""Future independent-performance loss on an unchanged DKT student."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import log_loss,roc_auc_score
from torch.utils.data import DataLoader
from pykt.datasets.transfer_dataset import TransferDataset,MODES
from pykt.models import init_model,evaluate,train_model
from pykt.utils import set_seed
from train_dkt_first_action import response_diagnostics


def future_diagnostics(model,loader):
    model.eval();device=next(model.parameters()).device;predictions,labels=[],[]
    with torch.no_grad():
        for data in loader:
            output=model(data['cseqs'].to(device).long(),data['rseqs'].to(device).long())
            concepts=data['future_concepts'].to(device);valid=concepts>=0
            queried=output.gather(-1,concepts.clamp_min(0));p=(queried*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            mask=data['future_mask'];predictions.extend(p.cpu()[mask].tolist());labels.extend(data['future_labels'][mask].tolist())
    if not labels:return {'events':0}
    p,y=np.asarray(predictions),np.asarray(labels)
    return {'events':len(y),'positive_rate':float(y.mean()),'auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'nll':float(log_loss(y,p,labels=[0,1])),'brier':float(((p-y)**2).mean())}


@torch.backends.cudnn.flags(enabled=False)
def gradient_diagnostics(model,loader):
    model.eval();parameters=list(model.parameters());rows=[]
    for i,data in enumerate(loader):
        if i>=4:break
        _,parts=model.compute_loss(data,return_components=True);vectors={}
        for key,loss in parts.items():
            grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(parameters)
            vectors[key]=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(parameters,grads)])
        a,b=vectors['response'],vectors['transfer'];rows.append({'bce':float(parts['response'].detach()),'transfer':float(parts['transfer'].detach()),
            'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),'aux_norm_over_bce':float(b.norm()/a.norm().clamp_min(1e-12))})
    return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,default=0);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--num_epochs',type=int,default=20);parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--emb_size',type=int,default=200);parser.add_argument('--dropout',type=float,default=.2)
    parser.add_argument('--learning_rate',type=float,default=.001);parser.add_argument('--transfer_weight',type=float,default=.1)
    parser.add_argument('--transfer_loss',choices=['bce','brier'],default='bce')
    parser.add_argument('--pair_mode',choices=MODES,default='same_unassisted');parser.add_argument('--horizon',type=int,default=10)
    parser.add_argument('--save_dir',type=Path,required=True);args=parser.parse_args()
    save=args.save_dir.resolve();save.mkdir(parents=True,exist_ok=True)
    if (save/'config.json').exists():raise FileExistsError(f'Refusing overwrite {save}')
    os.chdir(ROOT/'examples');set_seed(args.seed);torch.set_num_threads(1)
    data=copy.deepcopy(json.loads((ROOT/'configs/data_config.json').read_text())['assist2009']);data['dpath']=str(ROOT/'data/assist2009')
    source=Path(data['dpath'])/data['train_valid_file'];train_folds=set(data['folds'])-{args.fold}
    train=TransferDataset(source,data['input_type'],train_folds,args.pair_mode,args.horizon)
    valid=TransferDataset(source,data['input_type'],{args.fold},args.pair_mode,args.horizon,mode='valid')
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0)
    valid_loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0)
    config={'emb_size':args.emb_size,'dropout':args.dropout,'transfer_weight':args.transfer_weight,'transfer_loss':args.transfer_loss}
    params=vars(args).copy();params.update(save_dir=str(save),model_name='dkt_transfer',dataset_name='assist2009',emb_type='qid',use_wandb=0,use_also=0)
    snapshot={'train_config':{'batch_size':args.batch_size,'num_epochs':args.num_epochs,'seq_len':data['maxlen'],'optimizer':'adam'},
              'model_config':config,'data_config':data,'params':params}
    (save/'config.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    model=init_model('dkt_transfer',config.copy(),data,'qid');assert sum(p.numel() for p in model.parameters())==395523
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    print('EXPERIMENT',json.dumps({**params,'training_pairs':train.stats,'validation_pairs':valid.stats}),flush=True)
    started=time.monotonic()
    results=train_model(model,train_loader,valid_loader,args.num_epochs,optimizer,str(save),save_model=True,
                        data_config=data,fold=args.fold,use_trained=0,also_config={'enable':False})
    elapsed=time.monotonic()-started
    model.load_state_dict(torch.load(save/'qid_model.ckpt',map_location=next(model.parameters()).device,weights_only=True))
    torch.save(model.export_dkt_state_dict(),save/'qid_dkt_backbone.ckpt')
    auc,acc=evaluate(model,valid_loader,'dkt_transfer');diagnostics=response_diagnostics(model,valid_loader)
    diagnostics['train_gradient']=gradient_diagnostics(model,train_loader);diagnostics['future_scopes']={}
    # Compare every model on the SAME four held-out supervision cohorts;
    # different positive-rate cohorts must not be mixed into one AUC.
    for mode in MODES:
        dataset=valid if mode==args.pair_mode else TransferDataset(source,data['input_type'],{args.fold},mode,args.horizon,mode='valid')
        diagnostics['future_scopes'][mode]=future_diagnostics(model,DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0))
    summary={'model':'dkt_transfer','dataset':'assist2009','fold':args.fold,'seed':args.seed,'epoch_budget':args.num_epochs,
             'best_epoch':int(results[-1]),'validation_auc':float(auc),'validation_accuracy':float(acc),'parameters':395523,
             'training_seconds':elapsed,'model_config':config,'pair_mode':args.pair_mode,'horizon':args.horizon,
             'train_sequences':len(train),'validation_sequences':len(valid),'train_pairs':train.stats,'valid_pairs':valid.stats,
             'validation_diagnostics':diagnostics,'test_evaluated':False}
    (save/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print('VALIDATION_SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
