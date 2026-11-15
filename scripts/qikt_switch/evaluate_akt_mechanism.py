"""Resumable strict question-target AKT validation, paired to fresh baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from akt_experiment_io import build_model, predict
from reference_training import to_device
from reference_question_windows import CausalReferenceWindows
from prediction_progress import PredictionProgress, atomic_json, digest
from audit_inputs import sequence_metrics
from sequence_balance_metrics import assess_balance


def evaluate(run, baseline=None):
    run=Path(run); output=run/'strict_validation';output.mkdir(exist_ok=True)
    config=json.loads((run/'config.json').read_text());stats=json.loads((run/'training_complete.json').read_text())
    if not stats['completed']:raise ValueError('Training unfinished')
    torch.set_num_threads(2)
    weight=run/'best_model.ckpt';cfg=config['data_config'];rawpath=Path(cfg['dpath'])/cfg['train_valid_original_file_quelevel']
    raw=pd.read_csv(rawpath);raw=raw[raw.fold==config['params']['fold']]
    dataset=CausalReferenceWindows(raw,200);model=build_model(config)
    model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval()
    sources=[Path(__file__),Path(__file__).with_name('akt_experiment_io.py'),Path(__file__).with_name('reference_question_windows.py'),Path('pykt/models/akt.py')]
    if config['mechanism']!='baseline':sources.append(Path('pykt/models/akt_fairness.py'))
    identity={'config':digest(run/'config.json'),'weights':digest(weight),'raw':digest(rawpath),
        'sources':{str(p):digest(p) for p in sources},'scope':'nips fold0 development, per-target KC mean, strictly excludes current question labels; 200 expanded concepts'}
    progress=PredictionProgress(output,identity,resume=True);start=progress.state['next_window']
    loader=DataLoader(Subset(dataset,range(start,len(dataset))),batch_size=64,num_workers=0)
    with progress.signals(),torch.no_grad():
        for batch in loader:
            data,gaps,meta=batch;data,gaps=to_device((data,gaps),'cuda')
            prediction,_=predict(model,data,gaps)
            scores=prediction.gather(1,meta['target_index'].cuda()[:,None]).squeeze(1).cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError('Nonfinite strict scores')
            rows=[[int(meta['orirow'][i]),int(meta['qidx'][i]),int(meta['kc'][i]),int(meta['label'][i]),float(v),meta['questions'][i],meta['concepts'][i]] for i,v in enumerate(scores)]
            start+=len(rows);progress.flush(rows,start)
            if progress.stopping:progress.mark('stopped');progress.close();return 130
    expanded=pd.DataFrame(progress.records(),columns=['orirow','qidx','kc','late_trues','late_mean','questions','concepts'])
    groups=expanded.groupby(['orirow','qidx'],sort=True)
    if (groups.late_trues.nunique()!=1).any():raise ValueError('Conflicting target labels')
    frame=groups.agg(late_trues=('late_trues','first'),late_mean=('late_mean','mean'),questions=('questions','first'),concepts=('concepts','first')).reset_index()
    expected=sum(len(str(row.responses).split(','))-1 for row in raw.itertuples())
    if len(frame)!=expected:raise ValueError('Incomplete question coverage')
    cohorts,sequences=sequence_metrics(frame);frame.to_csv(output/'window_predictions.tsv',sep='\t',index=False,float_format='%.17g');sequences.to_csv(output/'sequence_auc.csv',index=False)
    pooled=float(roc_auc_score(frame.late_trues,frame.late_mean))
    result=dict(completed=True,model='akt',mechanism=config['mechanism'],scale=config['scale'],
        dataset=config['params']['dataset_name'],fold=config['params']['fold'],seed=config['params']['seed'],
        completed_epochs=stats['epoch'],selected_epoch=stats['selected_epoch'],validation_question_auc=pooled,
        cohorts=cohorts,events=len(frame),expanded_queries=len(expanded),identity=identity,
        windowauclate_mean=None,formal_test_evaluated=False,
        note='Development strict-question AUC is not the standard formal windowauclate_mean. Current time features assume event timestamp known; DateAnswered is not a question-start timestamp.')
    if baseline:
        bdir=Path(baseline)/'strict_validation';base=json.loads((bdir/'result.json').read_text())
        bp=pd.read_csv(bdir/'window_predictions.tsv',sep='\t');keys=['orirow','qidx','late_trues','questions','concepts']
        if not frame[keys].astype(str).equals(bp[keys].astype(str)):raise ValueError('Baseline target/label/KC identity mismatch')
        bs=pd.read_csv(bdir/'sequence_auc.csv').set_index('orirow');paired=sequences.set_index('orirow').join(bs.auc.rename('baseline_auc'),validate='one_to_one')
        if paired.baseline_auc.isna().any() or len(paired)!=len(bs):raise ValueError('Sequence mismatch')
        paired['delta_auc']=paired.auc-paired.baseline_auc;paired.to_csv(output/'paired_orirow.csv')
        result['balance']=assess_balance(paired,pooled-base['validation_question_auc'])
    atomic_json(output/'result.json',result);progress.mark('completed');progress.close();return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path);a=p.parse_args()
    raise SystemExit(evaluate(a.run,a.baseline))
