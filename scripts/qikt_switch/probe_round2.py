"""R2 predeclared validation-only weak objective mixing after R1 collapse.
Never loads test predictions or test data. Same fold protocol as round 1.
"""
from causal_probe import *

root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';out=base/'cpu_probe_nips_round2';out.mkdir(exist_ok=True)
protocol={'dataset':'nips_task34','fold':0,'seed':42,'neural_epochs':0,
 'purpose':'CPU causal probe R2; not a QIKT-Mamba performance result',
 'reason':'R1 full per-row/class balancing raises macro AUC but sharply lowers pooled AUC; test weak mixing and report every row regression',
 'variants':{'r2_class_mix_005':.05,'r2_class_mix_015':.15,'r2_class_mix_030':.30},
 'selection':'validation diagnostic only; no test selection or promotion to formal neural result'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
d=pd.read_csv(root/'data/nips_task34/train_valid_quelevel.csv');qmax=max(max(map(int,v.split(','))) for v in d.questions)+1
prior,mean,count=item_stats(d[d.fold.isin([2,3,4])],qmax);x,t=events(d[d.fold==1],prior,mean);v,valid=events(d[d.fold==0],prior,mean)
cols=VARIANTS['p4_adaptive_history'];head=fit_head(x,t.late_trues,cols);ref=head.predict_proba(v[:,cols])[:,1]
_,refseq=sequence_metrics(valid.assign(late_mean=ref));refseq=refseq.set_index('orirow');refauc=roc_auc_score(valid.late_trues,ref)
sizes=t.groupby('orirow').late_trues.transform('size').to_numpy();pos=t.groupby('orirow').late_trues.transform('sum').to_numpy()
w=1/np.maximum(np.where(t.late_trues.to_numpy()==1,pos,sizes-pos),1);w/=w.mean()
results={'protocol':protocol,'baseline_validation_auc':refauc,'baseline_cohorts':sequence_metrics(valid.assign(late_mean=ref))[0],'runs':{}}
rng=np.random.default_rng(42)
for name,alpha in protocol['variants'].items():
 model=fit_head(x,t.late_trues,cols,(1-alpha)+alpha*w);p=model.predict_proba(v[:,cols])[:,1]
 cohorts,seq=sequence_metrics(valid.assign(late_mean=p));seq=seq.set_index('orirow');delta=seq.auc-refseq.auc
 primary=(seq.n>=20)&(seq.minority>=5);dv=delta[primary].to_numpy();ci=np.quantile([rng.choice(dv,len(dv),replace=True).mean() for _ in range(1000)],[.025,.975])
 seq['baseline_auc']=refseq.auc;seq['delta_auc']=delta;seq.to_csv(out/f'{name}_paired.csv')
 result={'validation_auc':float(roc_auc_score(valid.late_trues,p)),'pooled_delta':float(roc_auc_score(valid.late_trues,p)-refauc),'cohorts':cohorts,
  'primary_paired':{'N':len(dv),'improved':int((dv>1e-12).sum()),'worsened':int((dv< -1e-12).sum()),'unchanged':int((abs(dv)<=1e-12).sum()),'mean_delta':float(dv.mean()),'min_delta':float(dv.min()),'max_delta':float(dv.max()),'bootstrap_macro_delta_ci95':ci.tolist()},
  'strict_no_row_regression_pass':bool((dv>=-1e-12).all()),'coefficients':model.coef_[0].tolist(),'intercept':model.intercept_[0]}
 results['runs'][name]=result;print(name,json.dumps(result),flush=True)
(out/'validation_results.json').write_text(json.dumps(results,indent=2))
