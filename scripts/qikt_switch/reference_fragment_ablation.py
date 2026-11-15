"""Resumable native-fragment inference interventions with UID balance reports.

Concept-expanded targets retain their native weighting. Not a question-window
comparison, retrained ablation or formal test evaluation.
"""
import argparse
import json
import signal
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from prediction_progress import PredictionProgress,atomic_json,digest
from reference_ablation_models import load_reference,intervene,predict
from audit_inputs import sequence_metrics
from sequence_balance_metrics import assess_balance


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text());root=Path(spec['output']);root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(spec['checkpoint_dir'],'config.json').read_text());cfg=config['data_config'];params=config['params'];cp=Path(spec['checkpoint_dir'])/(params['emb_type']+'_model.ckpt');set_seed(params['seed']);rawpath=Path(spec['data_file']);raw=pd.read_csv(rawpath);raw=raw[raw.fold==params['fold']].reset_index(drop=True)
    if params['model_name']=='mtkt':
        from pykt.datasets.dkt_forget_dataloader import DktForgetDataset
        dataset=DktForgetDataset(str(rawpath),cfg['input_type'],{params['fold']})
    else:
        from pykt.datasets.data_loader import KTDataset
        dataset=KTDataset(str(rawpath),cfg['input_type'],{params['fold']},mode='test',aug_probs={})
    assert len(raw)==len(dataset)
    identity={'spec':digest(a.spec),'config':digest(Path(spec['checkpoint_dir'],'config.json')),'checkpoint':digest(cp),'data':digest(rawpath),'reference_source':digest(Path(spec['source'])),'adapter':digest(Path(__file__).with_name('reference_ablation_models.py')),'runner':digest(Path(__file__))}
    results=[];stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    for mode in spec['modes']:
        out=root/mode;out.mkdir(exist_ok=True);model,module=load_reference(spec['source'],config,cp);patch=intervene(model,module,mode)
        if patch:(out/'intervention_forward.py').write_text(patch)
        model.eval();progress=PredictionProgress(out,{**identity,'mode':mode},resume=True)
        with torch.no_grad():
            for index,batch in enumerate(DataLoader(dataset,batch_size=spec['batch_size'])):
                start=index*spec['batch_size'];end=min(len(dataset),start+spec['batch_size'])
                if end<=progress.state['next_window']:continue
                if isinstance(batch,(tuple,list)):data,gaps=batch
                else:data,gaps=batch,None
                data={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in data.items()};gaps={k:v.cuda() for k,v in gaps.items()} if gaps else None
                pred=predict(model,data,gaps)[:,1:];mask=data['smasks'].bool();rows=[]
                if not torch.isfinite(pred[mask]).all():raise ValueError('Nonfinite reference prediction')
                pp=pred.cpu().numpy();mm=mask.cpu().numpy();rr=data['shft_rseqs'].cpu().numpy();qq=data['shft_qseqs'].cpu().numpy();cc=data['shft_cseqs'].cpu().numpy()
                for j in range(end-start):
                    uid=int(raw.iloc[start+j].uid)
                    for pos in np.flatnonzero(mm[j]):rows.append([uid,int((start+j)*200+pos+1),int(rr[j,pos]),float(pp[j,pos]),str(int(qq[j,pos])),str(int(cc[j,pos]))])
                progress.flush(rows,end)
                if stop.is_set():progress.mark('stopped');progress.close();return 130
        frame=pd.DataFrame(progress.records(),columns=['orirow','qidx','late_trues','late_mean','questions','concepts']);frame.to_csv(out/'predictions.tsv',sep='\t',index=False,float_format='%.17g');cohorts,sequences=sequence_metrics(frame);sequences.to_csv(out/'sequence_auc.csv',index=False)
        value={'mode':mode,'model':params['model_name'],'dataset':params['dataset_name'],'fold':params['fold'],'seed':params['seed'],'previous_training_epochs':spec['original_epochs'],'new_training_epochs':0,'native_validation_fragment_auc':float(roc_auc_score(frame.late_trues,frame.late_mean)),'brier':float(np.mean((frame.late_mean-frame.late_trues)**2)),'cohorts':cohorts,'events':len(frame),'scope':'Native concept-expanded fragments; UID sequence balance. Not question-window AUC, retraining or formal testing. Multiple-KC native label leakage audit pending.','formal_windowauclate_mean':None,'identity':identity}
        if mode!='full':
            baseline=pd.read_csv(root/'full/sequence_auc.csv').set_index('orirow');current=sequences.set_index('orirow');paired=current.join(baseline.auc.rename('baseline_auc'));paired['delta_auc']=paired.auc-paired.baseline_auc
            base=json.loads((root/'full/result.json').read_text());value['balance']=assess_balance(paired,value['native_validation_fragment_auc']-base['native_validation_fragment_auc']);paired.to_csv(out/'paired_orirow.csv')
        atomic_json(out/'result.json',value);results.append(value);progress.mark('completed');progress.close();del model
        atomic_json(root/'progress.json',{'completed_modes':[x['mode'] for x in results],'total_modes':len(spec['modes'])})
    atomic_json(root/'summary.json',{'completed':True,'results':results,'new_training':False,'goal_complete':False})


if __name__=='__main__':raise SystemExit(main())
