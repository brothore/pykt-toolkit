"""Paired sequence-balance metrics with fixed and model-specific tails."""
import numpy as np


def assess_balance(paired,pooled_delta,bootstrap=500):
    p=paired[(paired.n>=20)&(paired.minority>=5)].sort_index()
    if len(p)<3:raise ValueError('At least three primary paired sequences required')
    a=p.baseline_auc.to_numpy();b=p.auc.to_numpy();delta=b-a;n=len(a)
    order=np.argsort(a,kind='stable');tail=lambda f:max(1,int(np.ceil(n*f)))
    weak10=order[:tail(.1)];weak20=order[:tail(.2)];strong=order[-tail(.25):]
    def metrics(v):
        s=np.sort(v)
        return {'macro_auc':float(v.mean()),'std_ddof0':float(v.std(ddof=0)),'range':float(np.ptp(v)),
                'q95_minus_q05':float(np.quantile(v,.95)-np.quantile(v,.05)),
                'own_bottom10_mean':float(s[:tail(.1)].mean()),'own_bottom20_mean':float(s[:tail(.2)].mean()),
                'range_without_min_and_max':float(s[-2]-s[1])}
    base=metrics(a);candidate=metrics(b);changes={k:candidate[k]-base[k] for k in base}
    changes.update(fixed_baseline_bottom10_mean=float(delta[weak10].mean()),fixed_baseline_bottom20_mean=float(delta[weak20].mean()),fixed_baseline_top25_mean=float(delta[strong].mean()),pooled_auc=float(pooled_delta))
    checks={'std_reduced':changes['std_ddof0']<0,'range_reduced':changes['range']<0,
            'pooled_preserved':pooled_delta>=0,'macro_preserved':changes['macro_auc']>=0,
            'fixed_weak_preserved':changes['fixed_baseline_bottom10_mean']>=0,
            'own_tail_preserved':changes['own_bottom10_mean']>=0,
            'strong_preserved':changes['fixed_baseline_top25_mean']>=0}
    rng=np.random.default_rng(42);draws=[]
    for _ in range(bootstrap):
        idx=rng.integers(0,n,size=n);x=a[idx];y=b[idx];d=y-x
        weak=np.isin(idx,weak10);high=np.isin(idx,strong)
        draws.append([y.std(ddof=0)-x.std(ddof=0),d.mean(),d[weak].mean() if weak.any() else np.nan,d[high].mean() if high.any() else np.nan])
    intervals={name:np.nanquantile(np.asarray(draws)[:,i],[.025,.975]).tolist() for i,name in enumerate(['std_delta','macro_delta','fixed_bottom10_delta','fixed_top25_delta'])}
    return {'cohort':'primary_20_5','N':n,'baseline':base,'candidate':candidate,'deltas':changes,'checks':checks,
            'exploratory_candidate_pass':all(checks.values()),'strict_goal_pass':all(checks.values()) and bool((delta>=-1e-12).all()),
            'improved':int((delta>1e-12).sum()),'worsened':int((delta< -1e-12).sum()),'tied':int((np.abs(delta)<=1e-12).sum()),
            'decline_counts':{str(t):int((delta< -t).sum()) for t in [.001,.005,.01]},'worst_delta':float(delta.min()),
            'baseline_min_orirow':str(p.index[np.argmin(a)]),'baseline_max_orirow':str(p.index[np.argmax(a)]),
            'candidate_min_orirow':str(p.index[np.argmin(b)]),'candidate_max_orirow':str(p.index[np.argmax(b)]),
            'paired_orirow_bootstrap_ci95':intervals,'uncertainty_note':'Exploratory orirow resampling, not independent student/seed proof; raw range endpoints and sensitivity retained.'}
