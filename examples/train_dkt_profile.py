"""Fixed-prefix assessment profile supervision on the unchanged NIPS DKT."""
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
from torch.utils.data import DataLoader
from pykt.datasets.profile_dataset import ProfileDataset,profile_collate,MODES
from pykt.models import init_model,evaluate,train_model
from pykt.utils import set_seed


def profile_diagnostics(model,loader,shuffle_history=False):
    model.eval();rows=[]
    with torch.no_grad():
        for data in loader:
            device=next(model.parameters()).device;c=data['cseqs'].to(device).long();r=data['rseqs'].to(device).long()
            # A diagnostic sensitivity check, not an additional trained model.
            if shuffle_history:r=r.roll(1,0)
            output=model(c,r);p=model.profile_predictions(output,data).cpu().double()
            m=data['profile_mask'];y=data['profile_labels'].double();w=data['profile_mass'].double()
            for kind in (0,1,2):
                mask=m & (data['profile_kind']==kind)
                if mask.any():
                    weights=w[mask];pred=p[mask];labels=y[mask]
                    nll=torch.nn.functional.binary_cross_entropy(pred,labels,reduction='none')
                    rows.append((kind,float(weights.sum()),float((weights*(pred-labels)**2).sum()),float((weights*nll).sum()),float((weights*(pred-labels).abs()).sum()),int(mask.sum())))
    results={}
    for kind in (0,1,2):
        selected=[row for row in rows if row[0]==kind]
        if selected:
            mass=sum(row[1] for row in selected)
            results[str(kind)]={'groups':sum(row[5] for row in selected),'mass':mass,
                               'mse':sum(row[2] for row in selected)/mass,'nll':sum(row[3] for row in selected)/mass,'mae':sum(row[4] for row in selected)/mass}
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,default=0);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--num_epochs',type=int,default=20);parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--emb_size',type=int,default=200);parser.add_argument('--dropout',type=float,default=.2)
    parser.add_argument('--learning_rate',type=float,default=.001);parser.add_argument('--profile_weight',type=float,default=.1)
    parser.add_argument('--profile_mode',choices=MODES,default='score');parser.add_argument('--save_dir',type=Path,required=True);args=parser.parse_args()
    save=args.save_dir.resolve();save.mkdir(parents=True,exist_ok=True)
    if (save/'config.json').exists():raise FileExistsError(f'Refusing overwrite {save}')
    os.chdir(ROOT/'examples');set_seed(args.seed);torch.set_num_threads(1)
    data=copy.deepcopy(json.loads((ROOT/'configs/data_config.json').read_text())['nips_task34']);data['dpath']=str(ROOT/'data/nips_task34')
    source=Path(data['dpath'])/data['train_valid_file'];train_folds=set(data['folds'])-{args.fold}
    train=ProfileDataset(source,data['input_type'],train_folds,args.profile_mode)
    valid=ProfileDataset(source,data['input_type'],{args.fold},args.profile_mode,mode='valid')
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=profile_collate)
    valid_loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=profile_collate)
    config={'emb_size':args.emb_size,'dropout':args.dropout,'profile_weight':args.profile_weight}
    params=vars(args).copy();params.update(save_dir=str(save),model_name='dkt_profile',dataset_name='nips_task34',emb_type='qid',use_wandb=0,use_also=0)
    snapshot={'train_config':{'batch_size':args.batch_size,'num_epochs':args.num_epochs,'seq_len':data['maxlen'],'optimizer':'adam'},
              'model_config':config,'data_config':data,'params':params}
    (save/'config.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    # Construct only AFTER all datasets: no dataset construction consumes torch RNG.
    model=init_model('dkt_profile',config.copy(),data,'qid');assert sum(p.numel() for p in model.parameters())==355857
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    print('EXPERIMENT',json.dumps({**params,'training_profiles':train.stats,'validation_profiles':valid.stats}),flush=True)
    started=time.monotonic()
    results=train_model(model,train_loader,valid_loader,args.num_epochs,optimizer,str(save),save_model=True,
                        data_config=data,fold=args.fold,use_trained=0,also_config={'enable':False})
    elapsed=time.monotonic()-started
    model.load_state_dict(torch.load(save/'qid_model.ckpt',map_location=next(model.parameters()).device,weights_only=True))
    torch.save(model.export_dkt_state_dict(),save/'qid_dkt_backbone.ckpt')
    auc,acc=evaluate(model,valid_loader,'dkt_profile');diagnostics={}
    for mode in ('score','hierarchy','pseudo'):
        dataset=valid if mode==args.profile_mode else ProfileDataset(source,data['input_type'],{args.fold},mode,mode='valid')
        loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0,collate_fn=profile_collate)
        diagnostics[mode]=profile_diagnostics(model,loader)
        if mode=='hierarchy':diagnostics['hierarchy_rotated_responses']=profile_diagnostics(model,loader,True)
    summary={'model':'dkt_profile','dataset':'nips_task34','fold':args.fold,'seed':args.seed,'epoch_budget':args.num_epochs,
             'best_epoch':int(results[-1]),'validation_auc':float(auc),'validation_accuracy':float(acc),'parameters':355857,
             'training_seconds':elapsed,'model_config':config,'profile_mode':args.profile_mode,'train_profiles':train.stats,
             'valid_profiles':valid.stats,'validation_diagnostics':diagnostics,'test_evaluated':False}
    (save/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print('VALIDATION_SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
