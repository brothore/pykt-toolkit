"""Strict causal per-question validation for native reference checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from sklearn.metrics import roc_auc_score
from pykt.utils import set_seed
from prediction_progress import PredictionProgress,atomic_json,digest
from reference_question_windows import CausalReferenceWindows
from reference_ablation_models import load_reference,predict
from reference_training import to_device
from audit_inputs import sequence_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text());out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(spec['checkpoint_dir'],'config.json').read_text());params=config['params'];set_seed(params['seed']);weight=Path(spec['checkpoint_dir'])/(params['emb_type']+'_model.ckpt');raw=pd.read_csv(spec['raw_file']);raw=raw[raw.fold==params['fold']].copy();dataset=CausalReferenceWindows(raw,length=200);model,module=load_reference(spec['source'],config,weight);model.eval()
    identity={'spec':digest(a.spec),'weights':digest(weight),'raw':digest(Path(spec['raw_file'])),'source':digest(Path(spec['source'])),'code':{name:digest(Path(__file__).with_name(name)) for name in ['evaluate_reference_questions.py','reference_question_windows.py','reference_ablation_models.py','prediction_progress.py']},'scope':'Development question targets, mean of independently queried target KCs, at most199 prior expanded concepts; current question response excluded in all KC queries'}
    progress=PredictionProgress(out,identity,resume=True);batch_size=int(spec.get('batch_size',64));start=progress.state['next_window'];loader=DataLoader(Subset(dataset,range(start,len(dataset))),batch_size=batch_size)
    with progress.signals(),torch.no_grad():
        for batch in loader:
            data,gaps,meta=batch;data,gaps=to_device((data,gaps),'cuda');prediction=predict(model,data,gaps);idx=meta['target_index'].cuda();scores=prediction.gather(1,idx[:,None]).squeeze(1).cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError('Nonfinite strict reference scores')
            rows=[[int(meta['orirow'][i]),int(meta['qidx'][i]),int(meta['kc'][i]),int(meta['label'][i]),float(score),meta['questions'][i],meta['concepts'][i]] for i,score in enumerate(scores)];start+=len(rows);progress.flush(rows,start)
            if progress.stopping:progress.mark('stopped');return 130
    expanded=pd.DataFrame(progress.records(),columns=['orirow','qidx','kc','late_trues','late_mean','questions','concepts']);expected=sum(len(np.fromstring(row.responses,sep=','))-1 for row in raw.itertuples())
    groups=expanded.groupby(['orirow','qidx'],sort=True)
    if (groups.late_trues.nunique()!=1).any():raise ValueError('KC labels differ within question')
    frame=groups.agg(late_trues=('late_trues','first'),late_mean=('late_mean','mean'),questions=('questions','first'),concepts=('concepts','first')).reset_index()
    if len(frame)!=expected or frame.duplicated(['orirow','qidx']).any():raise ValueError('Question coverage differs')
    cohorts,sequence=sequence_metrics(frame);frame.to_csv(out/'window_predictions.tsv',sep='\t',index=False,float_format='%.17g');sequence.to_csv(out/'sequence_auc.csv',index=False)
    stats=Path(spec['checkpoint_dir'])/'training_stats.json';training=json.loads(stats.read_text()) if stats.exists() else {'completed_epochs':spec['original_epochs'],'selected_epoch':None}
    atomic_json(out/'validation_question_results.json',{'model':params['model_name'],'dataset':params['dataset_name'],'fold':params['fold'],'seed':params['seed'],'completed_epochs':training['completed_epochs'],'selected_epoch':training.get('selected_epoch'),'validation_question_auc':float(roc_auc_score(frame.late_trues,frame.late_mean)),'events':len(frame),'expanded_target_queries':len(expanded),'cohorts':cohorts,'identity':identity,'formal_windowauclate_mean':None,'test_evaluated':False,'limitations':'Reference context counts200 expanded concepts; QIKT context counts200 questions. This is a strict-label-isolation development evaluator, not a claim of unchanged standard native fusion.'})
    progress.mark('completed');progress.close()


if __name__=='__main__':raise SystemExit(main())
