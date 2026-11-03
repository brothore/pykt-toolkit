"""Compare aligned validation predictions, with orirow-cluster resampling.
No test selection or substitution for formal windowauclate_mean.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit_inputs import sequence_metrics,sha


class ClusterAUC:
    """Exact weighted AUC with fixed predictions, including tied scores."""
    def __init__(self,labels,scores,cluster):
        order=np.argsort(scores,kind='stable');p=np.asarray(scores)[order]
        self.labels=np.asarray(labels,dtype=float)[order];self.cluster=np.asarray(cluster,dtype=int)[order]
        self.starts=np.r_[0,np.flatnonzero(p[1:]!=p[:-1])+1]

    def __call__(self,cluster_weights):
        weights=np.asarray(cluster_weights)[self.cluster]
        positive=np.add.reduceat(weights*self.labels,self.starts)
        negative=np.add.reduceat(weights*(1-self.labels),self.starts)
        denom=positive.sum()*negative.sum()
        if not denom:return float('nan')
        return float(np.dot(positive,np.cumsum(negative)-.5*negative)/denom)


def compare(base,candidate,bootstrap=500,metric_key='validation_auc'):
    keys=['orirow','qidx']
    for d in [base,candidate]:
        if d.duplicated(keys).any():raise ValueError('Duplicate prediction keys')
        if not np.isfinite(d.late_mean).all():raise ValueError('Nonfinite prediction')
    base=base.sort_values(keys).reset_index(drop=True);candidate=candidate.sort_values(keys).reset_index(drop=True)
    required=['orirow','qidx','late_trues','questions','concepts']
    if len(base)!=len(candidate) or not base[required].astype(str).equals(candidate[required].astype(str)):
        raise ValueError('Validation event IDs, questions, concepts or labels differ')
    bc,bs=sequence_metrics(base);cc,cs=sequence_metrics(candidate)
    bs=bs.set_index('orirow');cs=cs.set_index('orirow');paired=cs.join(bs.auc.rename('baseline_auc'))
    paired['delta_auc']=paired.auc-paired.baseline_auc
    primary=(paired.n>=20)&(paired.minority>=5);p=paired[primary];delta=p.delta_auc.to_numpy()
    clusters,unique=pd.factorize(base.orirow,sort=True);k=len(unique)
    fa=ClusterAUC(base.late_trues.to_numpy(),base.late_mean.to_numpy(),clusters)
    fb=ClusterAUC(candidate.late_trues.to_numpy(),candidate.late_mean.to_numpy(),clusters)
    rng=np.random.default_rng(42);pooled=[];macro=[]
    for _ in range(bootstrap):
        weights=np.bincount(rng.integers(k,size=k),minlength=k)
        pooled.append(fb(weights)-fa(weights));macro.append(rng.choice(delta,len(delta),replace=True).mean())
    quantiles=p.baseline_auc.quantile([.25,.75]).to_numpy();groups={}
    for name,mask in [('baseline_low',p.baseline_auc<=quantiles[0]),('baseline_middle',p.baseline_auc.between(*quantiles,inclusive='neither')),('baseline_high',p.baseline_auc>=quantiles[1])]:
        x=p[mask];groups[name]={'N':len(x),'baseline_macro_auc':float(x.baseline_auc.mean()),'candidate_macro_auc':float(x.auc.mean()),'mean_delta':float(x.delta_auc.mean()),'worsened':int((x.delta_auc< -1e-12).sum())}
    ba=float(roc_auc_score(base.late_trues,base.late_mean));ca=float(roc_auc_score(candidate.late_trues,candidate.late_mean))
    result={'metric_scope':metric_key,'N_events':len(base),'N_orirows':k,
        'baseline_'+metric_key:ba,'candidate_'+metric_key:ca,'pooled_delta':ca-ba,
        'baseline_cohorts':bc,'candidate_cohorts':cc,'primary_paired':{'N':len(p),'improved':int((delta>1e-12).sum()),'worsened':int((delta< -1e-12).sum()),'unchanged':int((abs(delta)<=1e-12).sum()),'mean_delta':float(delta.mean()),'min_delta':float(delta.min()),'max_delta':float(delta.max())},
        'baseline_performance_groups':groups,'bootstrap':{'resampling_unit':'orirow; not proof of independent student identities','repetitions':bootstrap,'seed':42,'pooled_delta_ci95':np.nanquantile(pooled,[.025,.975]).tolist(),'primary_macro_delta_ci95':np.quantile(macro,[.025,.975]).tolist()},
        'strict_primary_no_regression_pass':bool((delta>=-1e-12).all()),
        'strict_pooled_improvement_and_primary_no_regression_pass':bool(ca>ba and (delta>=-1e-12).all())}
    return result,paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--bootstrap',type=int,default=500);args=ap.parse_args()
    a=pd.read_csv(args.baseline,sep='\t',dtype={'questions':str,'concepts':str});b=pd.read_csv(args.candidate,sep='\t',dtype={'questions':str,'concepts':str})
    result,paired=compare(a,b,args.bootstrap);result['prediction_sha256']={'baseline':sha(args.baseline),'candidate':sha(args.candidate)}
    args.output_dir.mkdir(parents=True,exist_ok=True);paired.to_csv(args.output_dir/'paired_orirow.csv');(args.output_dir/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
