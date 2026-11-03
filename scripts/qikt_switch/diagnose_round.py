"""Post-hoc validation diagnostics; labels here are never model features."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from causal_probe import item_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';directory=base/'neural'/args.variant
    paired=pd.read_csv(directory/'comparison/paired_orirow.csv').set_index('orirow');raw=pd.read_csv(root/'data/nips_task34/train_valid_quelevel.csv')
    prior,mean,count=item_stats(raw[raw.fold!=0],948);prediction=pd.read_csv(base/'neural/baseline/validation/validation_predictions.tsv',sep='\t')
    item_auc={}
    for uid,g in prediction.groupby('orirow'):
        if g.late_trues.nunique()==2:item_auc[uid]=roc_auc_score(g.late_trues,prior[g.questions.to_numpy(dtype=int)])
    features=[]
    for r in raw[raw.fold==0].itertuples():
        y=np.fromstring(r.responses,sep=',',dtype=int);p=y.mean();switch=np.mean(y[1:]!=y[:-1]);null=2*y.sum()*(len(y)-y.sum())/(len(y)*(len(y)-1))
        features.append(dict(orirow=r.uid,raw_length=len(y),correctness=p,switch_rate=switch,excess_switch=switch-null,train_only_item_auc=item_auc.get(r.uid,np.nan)))
    data=paired.join(pd.DataFrame(features).set_index('orirow'),rsuffix='_raw');primary=data[(data.n>=20)&(data.minority>=5)].copy()
    # switch_rate already existed from scored positions; use the full raw sequence value explicitly.
    switch_column='switch_rate_raw' if 'switch_rate_raw' in primary else 'switch_rate'
    groups={}
    for q in [.05,.10,.25]:
        for tail,mask in [('bottom',primary.baseline_auc<=primary.baseline_auc.quantile(q)),('top',primary.baseline_auc>=primary.baseline_auc.quantile(1-q))]:
            g=primary[mask];groups[f'baseline_{tail}_{int(q*100)}pct']={'N':len(g),'mean_delta':float(g.delta_auc.mean()),'worsened':int((g.delta_auc< -1e-12).sum()),'min_delta':float(g.delta_auc.min())}
    bins=pd.qcut(primary[switch_column],4,labels=False,duplicates='drop')
    for label,g in primary.groupby(bins):groups[f'raw_switch_quartile_{int(label)}']={'N':len(g),'mean_delta':float(g.delta_auc.mean()),'worsened':int((g.delta_auc< -1e-12).sum()),'switch_mean':float(g[switch_column].mean())}
    out=directory/'diagnostics';out.mkdir(exist_ok=True);primary.to_csv(out/'primary_rows.csv')
    result={'scope':'post-hoc validation label structure only; not causal attribution or deployable gate features','groups':groups,
      'lowest_baseline_rows':primary.nsmallest(5,'baseline_auc').reset_index().to_dict('records'),
      'worst_deltas':primary.nsmallest(5,'delta_auc').reset_index().to_dict('records')}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4));color=primary[switch_column]
    points=axes[0].scatter(primary.baseline_auc,primary.delta_auc,c=color,cmap='viridis',s=12,alpha=.7)
    axes[0].set(xlabel='Baseline orirow AUC',ylabel='Candidate minus baseline AUC',title='Fixed validation orirows')
    axes[1].scatter(primary[switch_column],primary.delta_auc,s=12,alpha=.6);axes[1].set(xlabel='Full raw-sequence switch rate',ylabel='Candidate minus baseline AUC',title='Post-hoc structure diagnostic')
    for ax in axes:ax.axhline(0,color='grey',linewidth=.8)
    for uid,row in primary.nsmallest(3,'delta_auc').iterrows():axes[0].annotate(str(uid),(row.baseline_auc,row.delta_auc),fontsize=7)
    fig.colorbar(points,ax=axes[0],label='Raw switch rate');fig.suptitle(f'{args.variant}: QIKT-Mamba / nips fold0 seed42');fig.tight_layout();fig.savefig(out/'paired_deltas.png',dpi=160);plt.close(fig)
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
