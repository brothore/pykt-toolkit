"""Dataset-specific joint training; fixed response backbone and validation rules."""
import argparse,copy,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from pykt.datasets.joint_dataset import JointDataset,joint_collate
from pykt.models import init_model,evaluate,train_model
from pykt.utils import set_seed

SOURCES={
 'assist2009':{'action':'first_action_iterations_20261025/r3_balanced','process':'process_iterations_20261025/r4_time_reweight',
              'distillation':'distill_iterations_20261026/r1_rich','transfer':'transfer_iterations_20261026/r2_same_all'},
 'nips_task34':{'confidence':'confidence_iterations_20261025/r2_ordinal','option':'option_iterations_20261025/r1_direct','profile':'profile_iterations_20261027/r3_weight'}}
DEFAULT_WEIGHTS={'assist2009':{'action':.1,'time':.03,'attempt':.1,'hint':.1,'distillation':.2,'transfer':.1},
                 'nips_task34':{'confidence':.182,'option':.1,'profile':.03}}


def component_configs(dataset,stats):
    result={}
    for task,name in SOURCES[dataset].items():
        cfg=copy.deepcopy(json.loads((ROOT/'examples/saved_model'/name/'config.json').read_text())['model_config'])
        for key in ('emb_size','dropout'):cfg.pop(key,None)
        result[task]=cfg
    if dataset=='assist2009':
        result['action']['action_class_counts']=stats['first_action_stats']['class_counts']
        result['process']['process_stats']=stats['process_stats']
    else:
        result['confidence']['confidence_counts']=stats['confidence_stats']['counts']
        result['option'].update(option_keys=stats['option_stats']['option_keys'],option_prior=stats['option_stats']['option_prior'])
    return result


def diagnostics(model,loader,gradients=False):
    model.eval();totals={};masses={};gradient_rows=[];device=next(model.parameters()).device
    backbone=[p for name,p in model.named_parameters() if name.startswith(('interaction_emb.','lstm_layer.','out_layer.'))]
    for i,data in enumerate(loader):
        if gradients and i>=4:break
        with torch.set_grad_enabled(gradients), torch.backends.cudnn.flags(enabled=False if gradients else torch.backends.cudnn.enabled):
            parts=model.loss_components(data)
        # Loss means use different task masks. These batch summaries are only
        # fixed-loader diagnostics, never called a global event-weighted loss.
        for key,value in parts.items():totals[key]=totals.get(key,0)+float(value.detach());masses[key]=masses.get(key,0)+1
        if gradients:
            vectors={}
            for key,loss in parts.items():
                if key!='response' and not model.loss_weights[key]:continue
                grads=torch.autograd.grad(loss,backbone,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(backbone)
                vectors[key]=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(backbone,grads)])
            main=vectors['response'];row={}
            for key,g in vectors.items():
                if key=='response':continue
                row[key]={'cosine_to_response':float(F_cos(main,g)),
                          'weighted_norm_over_response':float(model.loss_weights[key]*g.norm()/main.norm().clamp_min(1e-12))}
            gradient_rows.append(row)
    result={'mean_batch_component_losses':{k:totals[k]/masses[k] for k in totals},'scope':'same fixed loader; batch means have task-specific effective denominators'}
    if gradient_rows:
        result['backbone_gradients_first_4_train_batches']={key:{metric:float(np.mean([row[key][metric] for row in gradient_rows])) for metric in gradient_rows[0][key]} for key in gradient_rows[0]}
    return result


def F_cos(a,b):return torch.nn.functional.cosine_similarity(a,b,dim=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=list(SOURCES),required=True)
    parser.add_argument('--fold',type=int,choices=[0],default=0);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--num_epochs',type=int,default=20);parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--weights_json',required=True);parser.add_argument('--save_dir',type=Path,required=True);args=parser.parse_args()
    save=args.save_dir.resolve();save.mkdir(parents=True,exist_ok=True)
    if (save/'config.json').exists():raise FileExistsError(f'Refusing overwrite {save}')
    weights=json.loads(args.weights_json);os.chdir(ROOT/'examples');set_seed(args.seed);torch.set_num_threads(1)
    data=copy.deepcopy(json.loads((ROOT/'configs/data_config.json').read_text())[args.dataset]);data['dpath']=str(ROOT/'data'/args.dataset)
    source=Path(data['dpath'])/data['train_valid_file']
    train=JointDataset(source,data['input_type'],set(data['folds'])-{args.fold},args.dataset)
    valid=JointDataset(source,data['input_type'],{args.fold},args.dataset,mode='valid',stats=train.stats)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=joint_collate)
    valid_loader=DataLoader(valid,batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=joint_collate)
    config={'emb_size':200,'dropout':.2,'dataset_name':args.dataset,'task_configs':component_configs(args.dataset,train.stats),'loss_weights':weights}
    name='dkt_joint_assist' if args.dataset=='assist2009' else 'dkt_joint_nips'
    params=vars(args).copy();params.update(save_dir=str(save),model_name=name,dataset_name=args.dataset,emb_type='qid',use_wandb=0,use_also=0)
    snapshot={'train_config':{'batch_size':args.batch_size,'num_epochs':args.num_epochs,'seq_len':data['maxlen'],'optimizer':'adam'},
              'model_config':config,'data_config':data,'params':params}
    (save/'config.json').write_text(json.dumps(snapshot,indent=2)+'\n');(save/'training_stats.json').write_text(json.dumps(train.stats,indent=2)+'\n')
    model=init_model(name,copy.deepcopy(config),data,'qid');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    print('EXPERIMENT',json.dumps(params),flush=True);started=time.monotonic()
    results=train_model(model,train_loader,valid_loader,args.num_epochs,optimizer,str(save),save_model=True,data_config=data,fold=0,use_trained=0,also_config={'enable':False})
    elapsed=time.monotonic()-started;model.load_state_dict(torch.load(save/'qid_model.ckpt',map_location=next(model.parameters()).device,weights_only=True))
    torch.save(model.export_dkt_state_dict(),save/'qid_dkt_backbone.ckpt');auc,acc=evaluate(model,valid_loader,name)
    info={'model':name,'dataset':args.dataset,'fold':0,'seed':args.seed,'epoch_budget':args.num_epochs,'best_epoch':int(results[-1]),
          'validation_auc':float(auc),'validation_accuracy':float(acc),'training_seconds':elapsed,'loss_weights':weights,
          'training_parameters':sum(p.numel() for p in model.parameters()),'inference_parameters':sum(v.numel() for v in model.export_dkt_state_dict().values()),
          'train_rows':len(train),'valid_rows':len(valid),'validation_diagnostics':diagnostics(model,valid_loader),
          'training_gradient_diagnostics':diagnostics(model,train_loader,True),'test_evaluated':False}
    (save/'validation.json').write_text(json.dumps(info,indent=2)+'\n');print('VALIDATION_SUMMARY',json.dumps(info),flush=True)


if __name__=='__main__':main()
