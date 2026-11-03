"""Predeclared small validation-only convex mixture grid; no test inference."""
import json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from audit_inputs import sha

ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['r2','r3'],default='r2');args=ap.parse_args()
root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';out=base/f'refinement_{args.stage}/mixtures';out.mkdir(parents=True,exist_ok=True)
names=['baseline','s2_history_drop005','s5_pairwise_005','s7_overlap64'];frames={};metadata={}
for name in names:
 p=base/f'neural/{name}/validation_window/window_predictions.tsv';frames[name]=pd.read_csv(p,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True)
 metadata[name]={'predictions_sha256':sha(p),'training':json.loads((base/f'neural/{name}/round_summary.json').read_text())}
a=frames['baseline'];keys=['orirow','qidx','late_trues','questions','concepts']
for name,f in frames.items():assert f[keys].equals(a[keys]),name
labels=a.late_trues.to_numpy();groups=list(a.groupby('orirow',sort=True).indices.items());eligible=[(uid,idx) for uid,idx in groups if len(np.unique(labels[idx]))==2]
ns=np.array([len(i) for _,i in eligible]);minor=np.array([min(labels[i].sum(),len(i)-labels[i].sum()) for _,i in eligible]);primary=(ns>=20)&(minor>=5)
preds={name:f.late_mean.to_numpy() for name,f in frames.items()}

def rowauc(p):return np.array([roc_auc_score(labels[idx],p[idx]) for _,idx in eligible])
baserow=rowauc(preds['baseline']);baseauc=roc_auc_score(labels,preds['baseline']);bv=baserow[primary]
lo=bv<=np.quantile(bv,.25);hi=bv>=np.quantile(bv,.75)

def measure(weights):
 p=sum(preds[k]*v for k,v in weights.items());auc=roc_auc_score(labels,p);rv=rowauc(p)[primary];delta=rv-bv
 result={'components':weights,'validation_window_auc':float(auc),'pooled_delta':float(auc-baseauc),'primary_macro_auc':float(rv.mean()),'primary_std':float(rv.std(ddof=0)),'primary_range':float(np.ptp(rv)),
 'primary_improved':int((delta>1e-12).sum()),'primary_worsened':int((delta< -1e-12).sum()),'primary_tied':int((abs(delta)<=1e-12).sum()),'worst_primary_delta':float(delta.min()),'baseline_low_mean_delta':float(delta[lo].mean()),'baseline_high_mean_delta':float(delta[hi].mean()),'strict_primary_no_regression_pass':bool((delta>=-1e-12).all())}
 result['joint_distribution_screen_pass']=bool(auc>baseauc and rv.std(ddof=0)<=bv.std(ddof=0) and np.ptp(rv)<=np.ptp(bv) and delta[lo].mean()>=0 and delta[hi].mean()>=0)
 return result,p,delta

settings=[]
for name in names[1:]:
 for alpha in [.25,.5,.75]:settings.append({'baseline':1-alpha,name:alpha})
for left,right in [('s2_history_drop005','s5_pairwise_005'),('s7_overlap64','s5_pairwise_005'),('s2_history_drop005','s7_overlap64')]:
 for alpha in [.25,.5,.75]:
  for beta in [.1,.25,.5]:
   if alpha+beta<=1:settings.append({'baseline':1-alpha-beta,left:alpha,right:beta})
if args.stage=='r3':
 settings=[{'baseline':.15,'s7_overlap64':.1,'s5_pairwise_005':.75},
           {'baseline':.15,'s2_history_drop005':.1,'s5_pairwise_005':.75},
           {'s2_history_drop005':.05,'s7_overlap64':.2,'s5_pairwise_005':.75}]
protocol={'stage':args.stage,'scope':'developmental window-validation mixture screen; not a final non-regression acceptance gate','dataset':'nips_task34','fold':0,'seed':42,'settings':settings,'metadata':metadata,'baseline_auc':float(baseauc),'baseline_primary_std':float(bv.std(ddof=0)),'baseline_primary_range':float(np.ptp(bv)),'test_read':False,
 'r3_rationale':'R2 large mask/overlap corrections improve pooled AUC but violate spread constraints. Test only three ranking-dominated mixtures: ranking weight .75, smaller corrective weights; overlap offsets adverse extreme-tail behavior of masking. Coefficients are global, with no per-UID gates; no test labels read.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
for index,weights in enumerate(settings):
 result,p,delta=measure(weights);result['id']=index;results.append(result);print(json.dumps(result),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2));pd.DataFrame([{**r,'components':json.dumps(r['components'])} for r in results]).to_csv(out/'results.csv',index=False)
passing=[r for r in results if r['joint_distribution_screen_pass']]
summary={'evaluated':len(results),'joint_distribution_screen_passes':len(passing),'strict_primary_no_regression_passes':sum(r['strict_primary_no_regression_pass'] for r in results),'best_pooled':max(results,key=lambda r:r['pooled_delta']),'joint_screen_candidates':sorted(passing,key=lambda r:r['pooled_delta'],reverse=True),'objective_achieved':False,'test_read':False}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
