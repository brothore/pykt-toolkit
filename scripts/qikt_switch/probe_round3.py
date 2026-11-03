"""CPU validation R3: add a small within-fragment ranking objective to BCE.
Same disjoint prior/head/validation folds as R1. No test data are loaded.
"""
from causal_probe import *
from scipy.optimize import minimize
from scipy.special import expit

root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';out=base/'cpu_probe_nips_round3';out.mkdir(exist_ok=True)
protocol={'dataset':'nips_task34','fold':0,'seed':42,'neural_epochs':0,'purpose':'CPU linear probe; not a QIKT-Mamba result',
 'reason':'R2 class reweighting reduces pooled AUC; retain token BCE and add a small ranking term',
 'variants':{'r3_pairwise_001':.01,'r3_pairwise_005':.05},'max_pairs_per_fragment':256,
 'fragment_definition':'group by uid and floor(qidx/200); causal features still use full prefix, unlike neural 200-step inputs',
 'training_data_sha256':sha(root/'data/nips_task34/train_valid_quelevel.csv'),'source_sha256':sha(Path(__file__)),
 'prior_folds':[2,3,4],'head_fit_fold':1,'validation_fold':0,'test_loaded':False}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
d=pd.read_csv(root/'data/nips_task34/train_valid_quelevel.csv');qmax=max(max(map(int,v.split(','))) for v in d.questions)+1
prior,mean,count=item_stats(d[d.fold.isin([2,3,4])],qmax);x,t=events(d[d.fold==1],prior,mean);v,valid=events(d[d.fold==0],prior,mean)
cols=VARIANTS['p4_adaptive_history'];x=x[:,cols];v=v[:,cols];y=t.late_trues.to_numpy();yv=valid.late_trues.to_numpy()
head=LogisticRegression(C=1.,max_iter=200,solver='lbfgs',random_state=42).fit(x,y)
initial=np.r_[head.coef_[0],head.intercept_[0]];ref=head.predict_proba(v)[:,1];refauc=roc_auc_score(yv,ref)
refcohorts,refseq=sequence_metrics(valid.assign(late_mean=ref));refseq=refseq.set_index('orirow')
rng=np.random.default_rng(42);diffs=[];weights=[];eligible=0
for _,g in t.assign(fragment=t.qidx//200).groupby(['orirow','fragment'],sort=False):
 pos=g.index[g.late_trues==1].to_numpy();neg=g.index[g.late_trues==0].to_numpy()
 if not len(pos) or not len(neg):continue
 n=min(256,len(pos)*len(neg));eligible+=1
 if len(pos)*len(neg)<=256:
  a=np.repeat(pos,len(neg));b=np.tile(neg,len(pos))
 else:a=rng.choice(pos,n);b=rng.choice(neg,n)
 diffs.append(x[a]-x[b]);weights.append(np.full(n,1./n))
delta=np.concatenate(diffs);pw=np.concatenate(weights)/eligible

def objective(theta,alpha):
 w=theta[:-1];z=x@w+theta[-1];loss=np.mean(np.logaddexp(0,z)-y*z)+.5*np.sum(w*w)/len(y)
 error=expit(z)-y;grad=np.r_[x.T@error/len(y)+w/len(y),error.mean()]
 margins=delta@w;loss+=alpha*np.dot(pw,np.logaddexp(0,-margins));grad[:-1]-=alpha*delta.T@(pw*expit(-margins))
 return loss,grad

results={'protocol':protocol,'training_fragments_with_pairs':eligible,'training_sampled_pairs':len(delta),'baseline_validation_auc':float(refauc),'baseline_cohorts':refcohorts,'runs':{}}
# Finite-difference directional derivative checks the objective before fitting.
direction=np.random.default_rng(4).normal(size=len(initial));eps=1e-5
f,grad=objective(initial,.01);finite=(objective(initial+eps*direction,.01)[0]-objective(initial-eps*direction,.01)[0])/(2*eps)
assert np.isclose(finite,grad@direction,rtol=1e-5,atol=1e-7)
results['gradient_check']={'finite_difference':float(finite),'analytic':float(grad@direction)}
for name,alpha in protocol['variants'].items():
 fitted=minimize(objective,initial,args=(alpha,),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12,'gtol':1e-7})
 if not fitted.success:raise RuntimeError(fitted.message)
 p=expit(v@fitted.x[:-1]+fitted.x[-1]);cohorts,seq=sequence_metrics(valid.assign(late_mean=p));seq=seq.set_index('orirow')
 seq['baseline_auc']=refseq.auc;seq['delta_auc']=seq.auc-refseq.auc;seq.to_csv(out/f'{name}_paired.csv')
 primary=(seq.n>=20)&(seq.minority>=5);dv=seq.loc[primary,'delta_auc'].to_numpy()
 result={'validation_auc':float(roc_auc_score(yv,p)),'pooled_delta':float(roc_auc_score(yv,p)-refauc),'cohorts':cohorts,
 'validation_logloss':float(log_loss(yv,p)),'primary_paired':{'N':len(dv),'improved':int((dv>1e-12).sum()),'worsened':int((dv< -1e-12).sum()),'unchanged':int((abs(dv)<=1e-12).sum()),'mean_delta':float(dv.mean()),'min_delta':float(dv.min()),'max_delta':float(dv.max())},
 'strict_no_row_regression_pass':bool((dv>=-1e-12).all()),'optimizer_success':bool(fitted.success),'optimizer_steps':int(fitted.nit),'coefficients':fitted.x.tolist()}
 results['runs'][name]=result;print(name,json.dumps(result),flush=True)
(out/'validation_results.json').write_text(json.dumps(results,indent=2))
