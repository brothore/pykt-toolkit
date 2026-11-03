"""CPU validation probes: all features exist before the predicted response.
This is mechanism screening, not a replacement for QIKT-Mamba retraining.
Fold 2/3/4 fit item priors; fold 1 fits logistic heads; fold 0 validates.
No test labels or frozen model scores enter fitting or selection.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from audit_inputs import sequence_metrics,sha

FEATURES=['item_logit','prefix_logit','previous_answer','concept_prefix_logit','log_concept_count','prefix_switch','previous_x_switch','item_x_prefix','concept_x_item']
VARIANTS={
 'p0_item':[0],
 'p1_item_ability':[0,1],
 'p2_recent_answer':[0,1,2],
 'p3_concept_memory':[0,1,3,4,7,8],
 'p4_adaptive_history':[0,1,2,3,4,5,6,7,8],
}


def item_stats(rows,num_q):
    count=np.zeros(num_q);correct=np.zeros(num_q)
    for row in rows.itertuples():
        q=np.fromstring(row.questions,sep=',',dtype=int);y=np.fromstring(row.responses,sep=',',dtype=int)
        np.add.at(count,q,1);np.add.at(correct,q,y)
    mean=correct.sum()/count.sum()
    return logit(np.clip((correct+20*mean)/(count+20),1e-5,1-1e-5)),mean,count


def events(rows,prior,mean):
    xs=[];ys=[];uids=[];indices=[]
    for row in rows.itertuples():
        q=np.fromstring(row.questions,sep=',',dtype=int);y=np.fromstring(row.responses,sep=',',dtype=int)
        cs=[tuple(map(int,c.split('_'))) for c in row.concepts.split(',')]
        skill={};total=0.;sw=0
        for i,(qi,yi,ci) in enumerate(zip(q,y,cs)):
            if i:
                # All prefix quantities exclude yi, including the switching gate.
                ability=logit((total+10*mean)/(i+10))
                counts=[skill.get(c,(0.,0))[1] for c in ci]
                rates=[(skill.get(c,(0.,0))[0]+5*mean)/(skill.get(c,(0.,0))[1]+5) for c in ci]
                concept=logit(np.mean(rates));switch=(sw+1)/(max(0,i-1)+2)
                prev=2*y[i-1]-1;item=prior[qi]
                xs.append([item,ability,prev,concept,np.log1p(np.mean(counts)),switch,prev*switch,item*ability,concept*item])
                ys.append(yi);uids.append(row.uid);indices.append(i)
            total+=yi
            for c in ci:
                s,n=skill.get(c,(0.,0));skill[c]=(s+yi,n+1)
            if i and yi!=y[i-1]:sw+=1
    return np.asarray(xs),pd.DataFrame(dict(orirow=uids,qidx=indices,late_trues=ys))


def fit_head(x,y,cols,weight=None):
    model=LogisticRegression(C=1.,max_iter=200,solver='lbfgs',random_state=42)
    model.fit(x[:,cols],y,sample_weight=weight)
    return model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='nips_task34');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/f'logs/qikt_switch_20261103/cpu_probe_{args.dataset}';out.mkdir(exist_ok=True)
    path=root/f'data/{args.dataset}/train_valid_quelevel.csv';d=pd.read_csv(path)
    assert d.uid.is_unique
    qmax=max(max(map(int,v.split(','))) for v in d.questions)+1
    prior,mean,count=item_stats(d[d.fold.isin([2,3,4])],qmax)
    x,train=events(d[d.fold==1],prior,mean);v,valid=events(d[d.fold==0],prior,mean)
    assert set(train.orirow).isdisjoint(valid.orirow)
    result={'purpose':'validation-only causal mechanism probes; no QIKT-Mamba training or performance claim','dataset':args.dataset,'fold':0,'seed':42,'neural_epochs':0,'train_data_sha256':sha(path),'script_sha256':sha(Path(__file__)),'prior_folds':[2,3,4],'head_fit_fold':1,'validation_fold':0,'features':FEATURES,'runs':{}}
    # Group definitions fixed from labels solely for diagnostic reporting, never model inputs.
    group=[]
    for uid,g in valid.groupby('orirow',sort=False):
        y=g.late_trues.to_numpy();group.append((uid,np.mean(y[1:]!=y[:-1])))
    rates=pd.Series(dict(group));cuts=rates.quantile([.25,.75]).to_numpy()
    models={}
    for name,cols in VARIANTS.items():
        model=fit_head(x,train.late_trues,cols);p=model.predict_proba(v[:,cols])[:,1]
        frame=valid.assign(late_mean=p);cohorts,seq=sequence_metrics(frame)
        metrics={'validation_auc':float(roc_auc_score(valid.late_trues,p)),'validation_logloss':float(log_loss(valid.late_trues,p)),
                 'cohorts':cohorts,'features':[FEATURES[i] for i in cols],'coefficients':model.coef_[0].tolist(),'intercept':model.intercept_[0],
                 'low_switch_macro_auc':float(seq.loc[seq.orirow.isin(rates[rates<=cuts[0]].index),'auc'].mean()),
                 'high_switch_macro_auc':float(seq.loc[seq.orirow.isin(rates[rates>=cuts[1]].index),'auc'].mean())}
        seq.to_csv(out/f'{name}_sequences.csv',index=False);result['runs'][name]=metrics;models[name]=(model,cols,p)
        print(name,metrics['validation_auc'],metrics['cohorts']['primary_20_5'],flush=True)
    # Objective screening on same causal features: row balance and positive/negative pair balance.
    cols=VARIANTS['p4_adaptive_history'];sizes=train.groupby('orirow').late_trues.transform('size').to_numpy()
    positives=train.groupby('orirow').late_trues.transform('sum').to_numpy();negatives=sizes-positives
    row_weights=1./sizes;row_weights/=row_weights.mean()
    class_count=np.where(train.late_trues.to_numpy()==1,positives,negatives)
    pair_weights=1./np.maximum(class_count,1);pair_weights/=pair_weights.mean()
    for name,weights in [('p5_row_balanced',row_weights),('p6_row_class_balanced',pair_weights)]:
        model=fit_head(x,train.late_trues,cols,weights);p=model.predict_proba(v[:,cols])[:,1]
        frame=valid.assign(late_mean=p);cohorts,seq=sequence_metrics(frame)
        result['runs'][name]={'validation_auc':float(roc_auc_score(valid.late_trues,p)),'validation_logloss':float(log_loss(valid.late_trues,p)),'cohorts':cohorts,
            'features':[FEATURES[i] for i in cols],'coefficients':model.coef_[0].tolist(),'intercept':model.intercept_[0],
            'low_switch_macro_auc':float(seq.loc[seq.orirow.isin(rates[rates<=cuts[0]].index),'auc'].mean()),'high_switch_macro_auc':float(seq.loc[seq.orirow.isin(rates[rates>=cuts[1]].index),'auc'].mean())}
        seq.to_csv(out/f'{name}_sequences.csv',index=False)
        print(name,result['runs'][name]['validation_auc'],cohorts['primary_20_5'],flush=True)
    (out/'validation_results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'item_priors.npz',logits=prior,count=count)

if __name__=='__main__':main()
