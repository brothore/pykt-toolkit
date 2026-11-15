"""Native reference BCE objectives with complete-epoch resume and provenance."""
import json
import signal
import threading
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from prediction_progress import atomic_json
from resumable_qikt import atomic_save,rng_state,restore_rng
from reference_ablation_models import predict


def to_device(batch,device):
    if isinstance(batch,(tuple,list)):data,gaps=batch
    else:data,gaps=batch,None
    return ({k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}, {k:v.to(device) for k,v in gaps.items()} if gaps else None)


def objective(model,data,gaps):
    regularizer=0.
    if model.model_name=='mtkt':output=model(data,gaps,train=True)[0]
    else:
        c=torch.cat((data['cseqs'][:,:1],data['shft_cseqs']),1).long();r=torch.cat((data['rseqs'][:,:1],data['shft_rseqs']),1).long();q=torch.cat((data['qseqs'][:,:1],data['shft_qseqs']),1).long()
        output,regularizer=model(c,r,q)
    mask=data['smasks'].bool();loss=torch.nn.functional.binary_cross_entropy(output[:,1:][mask].double(),data['shft_rseqs'][mask].double())+regularizer
    if not torch.isfinite(loss):raise ValueError('Nonfinite native reference loss')
    return loss


def train(model,train_loader,valid_loader,optimizer,output,identity,max_epochs=200,epoch_hook=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);device=next(model.parameters()).device;checkpoint=output/'resume.pt';stop=threading.Event()
    state={'identity':identity,'completed_epochs':0,'selected_epoch':-1,'best_validation_auc':0.,'steps':0,'finished':False,'epochs':[],'history':[]}
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Reference training identity changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'],train_loader)
    state['history'].append({'event':'resume' if checkpoint.exists() else 'start','from_epoch':state['completed_epochs'],'pid':__import__('os').getpid()})
    handlers={s:signal.signal(s,lambda *_:stop.set()) for s in [signal.SIGTERM,signal.SIGINT]}
    try:
        for epoch in range(state['completed_epochs']+1,max_epochs+1):
            if state['finished']:break
            began=time.perf_counter();model.train();losses=[];positions=0
            for i,batch in enumerate(train_loader):
                data,gaps=to_device(batch,device);optimizer.zero_grad();loss=objective(model,data,gaps);loss.backward();optimizer.step();state['steps']+=1;positions+=int(data['smasks'].sum());losses.append(float(loss.detach().cpu()))
                if i%20==0:atomic_json(output/'training_progress.json',{'phase':'training','epoch':epoch,'batch':i,'steps':state['steps'],'stop_requested':stop.is_set(),'resume_granularity':'Complete epoch; unfinished epoch is replayed'})
            training_seconds=time.perf_counter()-began;model.eval();labels=[];scores=[]
            with torch.no_grad():
                for batch in valid_loader:
                    data,gaps=to_device(batch,device);pred=predict(model,data,gaps)[:,1:];mask=data['smasks'].bool();labels.extend(data['shft_rseqs'][mask].cpu().tolist());scores.extend(pred[mask].cpu().tolist())
            auc=float(roc_auc_score(labels,scores))
            if auc>state['best_validation_auc']+1e-3:
                state.update(selected_epoch=epoch,best_validation_auc=auc);atomic_save(model.state_dict(),output/'best_model.ckpt')
            state.update(completed_epochs=epoch,finished=epoch-state['selected_epoch']>=10 or epoch==max_epochs)
            state['epochs'].append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_fragment_auc':auc,'selected_epoch':state['selected_epoch'],'training_seconds':training_seconds,'epoch_with_validation_seconds':time.perf_counter()-began,'training_scored_positions':positions,'training_positions_per_second':positions/training_seconds})
            if epoch_hook and epoch_hook(epoch):stop.set()
            state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(train_loader));atomic_save(state,checkpoint);atomic_json(output/'epoch_metrics.json',state['epochs'])
            atomic_json(output/'training_stats.json',{k:state[k] for k in ['completed_epochs','selected_epoch','best_validation_auc','finished','steps','history']})
            if stop.is_set():atomic_json(output/'training_progress.json',{'phase':'stopped','completed_epochs':epoch,'resume_granularity':'Complete epoch'});return 130
        atomic_json(output/'training_progress.json',{'phase':'completed','completed_epochs':state['completed_epochs']});return 0
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
