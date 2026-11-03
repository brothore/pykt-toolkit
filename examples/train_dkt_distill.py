"""Parameter-identical DKT students trained with cross-fitted teacher loss."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import log_loss,roc_auc_score
from torch.utils.data import DataLoader
from pykt.datasets.distill_dataset import DistillDataset
from pykt.models import init_model,evaluate,train_model
from pykt.utils import set_seed
from train_dkt_first_action import response_diagnostics


def original_event_diagnostics(model,loader):
    model.eval();device=next(model.parameters()).device;p,y=[],[]
    with torch.no_grad():
        for data in loader:
            output=model(data['cseqs'].to(device).long(),data['rseqs'].to(device).long())
            pred=output.gather(-1,data['shft_cseqs'].to(device).long().unsqueeze(-1)).squeeze(-1).cpu()
            mask=data['original_event_mask'];p.extend(pred[mask].tolist());y.extend(data['shft_rseqs'][mask].tolist())
    return {'event_count':len(y),'auc':float(roc_auc_score(y,p)),'nll':float(log_loss(y,p,labels=[0,1])),
            'brier':float(np.mean((np.asarray(p)-np.asarray(y))**2))}


@torch.backends.cudnn.flags(enabled=False)
def kd_diagnostics(model,loader):
    model.eval();parameters=list(model.parameters());rows=[]
    for i,data in enumerate(loader):
        if i>=4:break
        _,parts=model.compute_loss(data,return_components=True)
        vectors={}
        for key,loss in parts.items():
            grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(parameters)
            vectors[key]=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(parameters,grads)])
        a,b=vectors['response'],vectors['distillation']
        rows.append({'bce':float(parts['response'].detach()),'kl':float(parts['distillation'].detach()),
                     'cosine':float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                     'kl_norm_over_bce':float(b.norm()/a.norm().clamp_min(1e-12))})
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=[0],default=0)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--num_epochs',type=int,default=20)
    parser.add_argument('--batch_size',type=int,default=32);parser.add_argument('--emb_size',type=int,default=200)
    parser.add_argument('--dropout',type=float,default=.2);parser.add_argument('--learning_rate',type=float,default=.001)
    parser.add_argument('--soft_targets',type=Path,required=True);parser.add_argument('--teacher_source',choices=['rich','thin'],default='rich')
    parser.add_argument('--kd_weight',type=float,default=.2);parser.add_argument('--kd_temperature',type=float,default=1.)
    parser.add_argument('--kd_mode',choices=['uniform','confidence'],default='uniform')
    parser.add_argument('--save_dir',type=Path,required=True);args=parser.parse_args()
    save=args.save_dir.resolve();soft=args.soft_targets.resolve();save.mkdir(parents=True,exist_ok=True)
    if (save/'config.json').exists():raise FileExistsError(f'Refusing overwrite {save}')
    os.chdir(ROOT/'examples');set_seed(args.seed);torch.set_num_threads(1)
    data=copy.deepcopy(json.loads((ROOT/'configs/data_config.json').read_text())['assist2009'])
    data['dpath']=str(ROOT/'data/assist2009');source=Path(data['dpath'])/data['train_valid_file']
    train=DistillDataset(source,data['input_type'],{1,2,3,4},soft,args.teacher_source)
    valid=DistillDataset(source,data['input_type'],{0},mode='valid')
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0)
    valid_loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0)
    config={'emb_size':args.emb_size,'dropout':args.dropout,'kd_weight':args.kd_weight,
            'kd_temperature':args.kd_temperature,'kd_mode':args.kd_mode}
    params=vars(args).copy();params.update(save_dir=str(save),soft_targets=str(soft),model_name='dkt_distill',
        dataset_name='assist2009',emb_type='qid',use_wandb=0,use_also=0)
    snapshot={'train_config':{'batch_size':args.batch_size,'num_epochs':args.num_epochs,'seq_len':data['maxlen'],'optimizer':'adam'},
              'model_config':config,'data_config':data,'params':params}
    (save/'config.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    model=init_model('dkt_distill',config.copy(),data,'qid')
    assert sum(p.numel() for p in model.parameters())==395523
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    print('EXPERIMENT',json.dumps(params),flush=True);started=time.monotonic()
    results=train_model(model,train_loader,valid_loader,args.num_epochs,optimizer,str(save),save_model=True,
                        data_config=data,fold=args.fold,use_trained=0,also_config={'enable':False})
    elapsed=time.monotonic()-started
    model.load_state_dict(torch.load(save/'qid_model.ckpt',map_location=next(model.parameters()).device,weights_only=True))
    torch.save(model.export_dkt_state_dict(),save/'qid_dkt_backbone.ckpt')
    auc,acc=evaluate(model,valid_loader,'dkt_distill')
    diagnostics=response_diagnostics(model,valid_loader)
    diagnostics['original_event']=original_event_diagnostics(model,valid_loader)
    diagnostics['train_kd']=kd_diagnostics(model,train_loader)
    output={'model':'dkt_distill','dataset':'assist2009','fold':args.fold,'seed':args.seed,'epoch_budget':args.num_epochs,
            'best_epoch':int(results[-1]),'validation_auc':float(auc),'validation_accuracy':float(acc),'training_seconds':elapsed,
            'parameters':sum(p.numel() for p in model.parameters()),'model_config':config,'teacher_source':args.teacher_source,
            'train_sequences':len(train),'validation_sequences':len(valid),'validation_diagnostics':diagnostics,'test_evaluated':False}
    (save/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print('VALIDATION_SUMMARY',json.dumps(output),flush=True)


if __name__=='__main__':main()
