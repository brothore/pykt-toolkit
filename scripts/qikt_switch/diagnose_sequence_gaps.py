"""Read-only V2 validation phenotypes and causal simple-reference gaps.

Complete-sequence label summaries are post-hoc diagnostics, never routing or
training weights. SIGTERM/SIGINT stop between UIDs; saved UID rows are reused.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import signal
import threading
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    raw_path=root/'data/nips_task34/train_valid_quelevel.csv'
    paths=[base/'neural/baseline/validation_window/orirow_auc.csv']+[base/f'seed_replication/neural_seed{s}/baseline/validation_window/orirow_auc.csv' for s in [43,44]]
    identity={str(p):digest(p) for p in [raw_path,*paths,Path(__file__)]}
    checkpoint=out/'uid_progress.json'
    progress=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'input_sha256':identity,'rows':{}}
    if progress['input_sha256']!=identity:raise ValueError('Diagnostic source/input changed')
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    raw=pd.read_csv(raw_path);train=raw[raw.fold!=0];valid=raw[raw.fold==0]
    qn=Counter();qc=Counter();kn=Counter();kc=Counter()
    for row in train.itertuples():
        for q,c,r in zip(row.questions.split(','),row.concepts.split(','),row.responses.split(',')):
            answer=int(r);qn[int(q)]+=1;qc[int(q)]+=answer
            for concept in set(map(int,c.split('_'))):kn[concept]+=1;kc[concept]+=answer
    qcut=float(np.quantile(list(qn.values()),.25));kcut=float(np.quantile(list(kn.values()),.25))
    frames=[pd.read_csv(p).set_index('orirow') for p in paths]
    eligible=frames[0][(frames[0].n>=20)&(frames[0].minority>=5)].index
    aucs=pd.concat([f.auc.rename(f'seed{s}') for f,s in zip(frames,[42,43,44])],axis=1).loc[eligible]
    weak=(aucs<=aucs.quantile(.25)).all(axis=1);strong=(aucs>=aucs.quantile(.75)).all(axis=1)
    variability=aucs.std(axis=1,ddof=0);varcut=float(variability.quantile(.75))
    global_prior=sum(qc.values())/sum(qn.values())
    for row in valid.itertuples():
        uid=int(row.uid)
        if uid not in eligible or str(uid) in progress['rows']:continue
        qs=list(map(int,row.questions.split(',')));ys=list(map(int,row.responses.split(',')));cs=[set(map(int,c.split('_'))) for c in row.concepts.split(',')]
        past_n=defaultdict(int);past_c=defaultdict(int);predictions=[];supports=[]
        for i,(q,concepts,y) in enumerate(zip(qs,cs,ys)):
            if i:
                qprior=(qc[q]+global_prior*2)/(qn[q]+2)
                history=np.mean([(past_c[c]+1)/(past_n[c]+2) for c in concepts])
                predictions.append(.5*qprior+.5*history);supports.append(np.mean([past_n[c] for c in concepts]))
            for c in concepts:past_n[c]+=1;past_c[c]+=y
        record={'orirow':uid,'targets':len(ys)-1,'correctness_posthoc':float(np.mean(ys[1:])),
                'switch_rate_posthoc':float(np.mean(np.diff(ys)!=0)),'distinct_kcs':len(set.union(*cs)),
                'rare_question_fraction':float(np.mean([qn[q]<=qcut for q in qs[1:]])),
                'rare_kc_fraction':float(np.mean([np.mean([kn[c]<=kcut for c in concepts]) for concepts in cs[1:]])),
                'mean_prefix_kc_support':float(np.mean(supports)),
                'baseline_mean_auc':float(aucs.loc[uid].mean()),'baseline_seed_std':float(variability.loc[uid]),
                'stable_weak':bool(weak.loc[uid]),'stable_strong':bool(strong.loc[uid]),'high_seed_variability':bool(variability.loc[uid]>=varcut),
                'simple_reference_auc':float(roc_auc_score(ys[1:],predictions))}
        record['reference_gap']=record['simple_reference_auc']-record['baseline_mean_auc']
        progress['rows'][str(uid)]=record
        if len(progress['rows'])%25==0 or stop.is_set():atomic_json(checkpoint,progress)
        if stop.is_set():return 130
    atomic_json(checkpoint,progress)
    table=pd.DataFrame(list(progress['rows'].values())).sort_values('orirow');temporary=out/'uid_table.tmp';table.to_csv(temporary,index=False);temporary.replace(out/'uid_table.csv')
    masks={'stable_weak':table.stable_weak,'stable_strong':table.stable_strong,'unstable':table.high_seed_variability,
           'short':table.targets<50,'long':table.targets>=400,'rare_kc':table.rare_kc_fraction>=.25,
           'low_correctness':table.correctness_posthoc<.4,'little_relevant_history':table.mean_prefix_kc_support<1,
           'short_and_rare':(table.targets<50)&(table.rare_kc_fraction>=.25),'reference_outperforms':table.reference_gap>0}
    groups={}
    for name,mask in masks.items():
        group=table[mask];groups[name]={'N':len(group),'baseline_macro_auc':float(group.baseline_mean_auc.mean()) if len(group) else None,
            'reference_gap_mean':float(group.reference_gap.mean()) if len(group) else None,'stable_weak_count':int(group.stable_weak.sum()),
            'support_for_training_group':len(group)>=50}
    atomic_json(out/'analysis.json',{'scope':'Nips fold0 validation primary_20_5 only. Labels/whole-sequence features and baseline AUC are explanatory, never training UID weights or inference gates.',
        'input_sha256':identity,'training_frequency_thresholds':{'question_q25':qcut,'kc_q25':kcut},'groups':groups,
        'heuristics':'Diagnostic grouping thresholds fixed by script before execution; not a fitted causal explanation.',
        'reference':'Fixed .5 training-only smoothed item prior + .5 causal KC prefix posterior; no validation fitting.',
        'table_sha256':digest(out/'uid_table.csv'),'completed_orirows':len(table),'test_evaluated':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
