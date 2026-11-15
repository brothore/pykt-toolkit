"""QIKT-only native training semantics, with epoch-boundary resume.

No AMP, scheduler, gradient accumulation, ALSO, worker processes or post-training
test inference are used. These unsupported modes are rejected explicitly.
SIGTERM/SIGINT finish the current epoch, checkpoint, and exit 130.
"""
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from prediction_progress import atomic_json,digest


def atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(value,temporary);os.replace(temporary,path)


def rng_state(loader):
    generator=getattr(loader,'generator',None)
    sampler_generator=getattr(getattr(loader,'sampler',None),'generator',None)
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loader':generator.get_state() if generator is not None else None,
            'sampler':sampler_generator.get_state() if sampler_generator is not None else None}


def restore_rng(value,loader):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])
    for owner,key in [(loader,'loader'),(getattr(loader,'sampler',None),'sampler')]:
        generator=getattr(owner,'generator',None)
        if value[key] is not None:
            if generator is None:raise ValueError('Saved data-loader generator missing')
            generator.set_state(value[key].cpu())


def train_qikt(model,train_loader,valid_loader,num_epochs,opt,ckpt_path,test_loader=None,test_window_loader=None,
               save_model=False,data_config=None,fold=None,use_trained=0,accumulation_steps=1,also_config=None):
    from importlib import import_module
    native=import_module('pykt.models.train_model')
    if model.model_name!='qikt_mamba' or accumulation_steps!=1 or (also_config and also_config.get('enable')):
        raise ValueError('Resumable trainer supports plain QIKT, accumulation_steps=1, ALSO disabled')
    if not save_model or use_trained or test_loader is not None or test_window_loader is not None:
        raise ValueError('Resumable QIKT requires saved selection and separate test evaluation')
    if any(getattr(loader,'num_workers',0)!=0 for loader in [train_loader,valid_loader]):
        raise ValueError('Epoch-exact data-loader RNG restore currently requires num_workers=0')
    folder=Path(ckpt_path);path=folder/'training_resume.ckpt';best_path=folder/(model.emb_type+'_model.ckpt')
    objective_identity=None
    if os.environ.get('QIKT_V2_PROTOCOL'):
        from pykt.models.qikt_sequence_objectives import install_training_objective
        protocol_path=Path(os.environ['QIKT_V2_PROTOCOL']);protocol=json.loads(protocol_path.read_text())
        anchor_metadata=None
        if protocol.get('baseline_anchor_kl',0) or protocol.get('baseline_pair_weight',0):
            from pykt.models.qikt_baseline_anchor import install_baseline_anchor
            anchor_metadata=install_baseline_anchor(model,protocol)
        objective=install_training_objective(model,train_loader.dataset,protocol)
        atomic_json(folder/'training_objective.json',{'protocol':protocol,'metadata':objective.metadata(),'baseline_anchor':anchor_metadata})
        objective_identity={'protocol_sha256':digest(protocol_path),'training_metadata':digest(folder/'training_objective.json'),
                            'source':digest(Path(__file__).resolve().parents[2]/'pykt/models/qikt_sequence_objectives.py')}
        if anchor_metadata is not None:
            objective_identity['baseline_anchor_source']=digest(Path(__file__).resolve().parents[2]/'pykt/models/qikt_baseline_anchor.py')
            objective_identity['baseline_teacher_sha256']=digest(Path(protocol['initial_checkpoint']))
    identity={'config':digest(folder/'config.json'),'epochs':num_epochs,
              'objective':objective_identity,
              'sources':{str(p):digest(p) for p in [Path(__file__),Path(__file__).resolve().parents[2]/'pykt/models/qikt_mamba.py',Path(__file__).resolve().parents[2]/'pykt/models/qikt_switch_utils.py']}}
    if type(model).__module__.startswith('pykt.models.'):
        import inspect
        model_source=Path(inspect.getfile(type(model)))
        identity['sources'][str(model_source)]=digest(model_source)
    dependencies=getattr(model,'resume_source_dependencies',[])
    if dependencies:identity['additional_model_sources']={str(Path(p).resolve()):digest(Path(p)) for p in dependencies}
    state={'identity':identity,'completed_epochs':0,'best_epoch':-1,'best_auc':0.,'best_acc':0.,'steps':0,'elapsed_seconds':0.,'finished':False,'history':[]}
    if path.exists():
        if os.environ.get('QIKT_SWITCH_TRAIN_RESUME')!='1':raise ValueError('Training checkpoint exists; explicit resume required')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('Training configuration/source identity changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'],train_loader)
    # The checkpoint is authoritative; reconstruct this readable view on resume,
    # including completed re-entry after an interrupted metadata write.
    atomic_json(folder/'epoch_metrics.json',state.get('epoch_metrics',[]))
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    previous={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT]}
    started=time.perf_counter();previous_elapsed=state['elapsed_seconds']
    state['history'].append({'event':'resume' if path.exists() else 'start','pid':os.getpid(),'from_epoch':state['completed_epochs']})
    try:
        if state['finished']:return -1,-1,-1,-1,state['best_auc'],state['best_acc'],state['best_epoch']
        for epoch in range(state['completed_epochs']+1,num_epochs+1):
            losses=[];epoch_started=time.perf_counter()
            objective=getattr(model,'_sequence_objective',None)
            if objective is not None and hasattr(objective,'prepare_epoch'):
                preserved_rng=rng_state(train_loader)
                try:objective.prepare_epoch(model,train_loader,epoch,folder)
                finally:restore_rng(preserved_rng,train_loader)
            for index,batch in enumerate(train_loader):
                model.model.train();loss=native.model_forward(model,batch)
                opt.zero_grad();loss.backward();opt.step();state['steps']+=1;losses.append(float(loss.detach().cpu()))
                if index%20==0:atomic_json(folder/'training_progress.json',{'status':'stop_requested' if stopped else 'running','epoch':epoch,'batch':index,'optimizer_steps':state['steps'],'resume_granularity':'completed epoch'})
            auc,acc=native.evaluate(model,valid_loader,model.model_name)
            selected=auc>state['best_auc']+1e-3
            if selected:
                state.update(best_auc=float(auc),best_acc=float(acc),best_epoch=epoch)
                atomic_save(model.state_dict(),best_path)
            state.update(completed_epochs=epoch,elapsed_seconds=previous_elapsed+time.perf_counter()-started,
                         finished=epoch-state['best_epoch']>=10 or epoch==num_epochs,
                         model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(train_loader))
            epoch_metrics={'epoch':epoch,'validation_auc':float(auc),'validation_acc':float(acc),
                           'mean_train_loss':float(np.mean(losses)),'selected':bool(selected),
                           'selected_epoch':state['best_epoch'],'best_validation_auc':state['best_auc'],
                           'optimizer_steps':state['steps'],'epoch_elapsed_seconds':time.perf_counter()-epoch_started}
            state.setdefault('epoch_metrics',[]).append(epoch_metrics)
            atomic_save(state,path)
            atomic_json(folder/'epoch_metrics.json',state['epoch_metrics'])
            atomic_json(folder/'training_progress.json',{k:state[k] for k in ['completed_epochs','best_epoch','best_auc','best_acc','steps','elapsed_seconds','finished']})
            print(f"Epoch: {epoch}, validauc: {state['best_auc']:.4f}, validacc: {state['best_acc']:.4f}, best epoch: {state['best_epoch']}, best auc: {state['best_auc']:.4f}, train loss: {np.mean(losses)}",flush=True)
            print(f"Total Epoch Time: {time.perf_counter()-epoch_started:.2f}s",flush=True)
            print(json.dumps({'epoch_metrics':epoch_metrics}),flush=True)
            if stopped:raise SystemExit(130)
            if state['finished']:break
        return -1,-1,-1,-1,state['best_auc'],state['best_acc'],state['best_epoch']
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
