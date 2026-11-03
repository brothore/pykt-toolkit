"""Audit portable predictions against local IDs, order, labels and concepts.
Reconstructed window metrics are explicitly marked, not historical all_results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sequence_metrics(d, score='late_mean'):
    rows=[]
    for key,g in d.groupby('orirow',sort=False):
        y=g.late_trues.to_numpy(); pos=int(y.sum()); n=len(y)
        if 0 < pos < n:
            rows.append(dict(orirow=int(key),n=n,minority=min(pos,n-pos),
                auc=float(roc_auc_score(y,g[score])),switch_rate=float(np.mean(y[1:]!=y[:-1])) if n>1 else None))
    s=pd.DataFrame(rows,columns=['orirow','n','minority','auc','switch_rate'])
    cohorts={}
    for name,mask in [('all_two_class',s.n>0),('primary_20_5',(s.n>=20)&(s.minority>=5)),('strict_50_10',(s.n>=50)&(s.minority>=10))]:
        v=s.loc[mask,'auc'].to_numpy()
        if not len(v):
            cohorts[name]=dict(N=0,mean=None,std=None,range=None,q05=None,q95=None)
            continue
        cohorts[name]=dict(N=len(v),mean=float(v.mean()),std=float(v.std(ddof=0)),range=float(np.ptp(v)),q05=float(np.quantile(v,.05)),q95=float(np.quantile(v,.95)))
    return cohorts,s


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=a.parse_args()
    root=args.root; base=root/'logs/qikt_switch_20261103'; hand=base/'handoff'; out=base/'input_audit';out.mkdir(exist_ok=True)
    report={}
    for ds,runs in [('nips_task34',['run108','run109']),('peiyou',['run022','run069'])]:
        local_path=root/f'data/{ds}/test_quelevel.csv';local=pd.read_csv(local_path).set_index('uid')
        assert local.index.is_unique
        for run in runs:
            path=hand/f'predictions/{ds}_{run}.tsv.gz';d=pd.read_csv(path,sep='\t',dtype={'concepts':str}).sort_values(['orirow','qidx'])
            assert not d.duplicated(['orirow','qidx']).any()
            counts={'sequences':0,'events':0,'question_mismatch':0,'label_mismatch':0,'concept_mismatch':0,'incomplete_sequences':0}
            for uid,g in d.groupby('orirow',sort=False):
                row=local.loc[uid];idx=g.qidx.to_numpy(dtype=int)
                qs=np.array(row.questions.split(','),dtype=int);ys=np.array(row.responses.split(','),dtype=int)
                cs=[','.join(sorted(x.split('_'),key=int)) for x in row.concepts.split(',')]
                # Late fusion repeats a question once per concept (e.g. "451,451").
                observed=[]
                for value in g.questions:
                    members=list(map(int,str(value).split(',')))
                    assert len(set(members))==1, ('mixed question fusion',uid,value)
                    observed.append(members[0])
                counts['question_mismatch']+=int(np.sum(qs[idx]!=np.asarray(observed)))
                counts['label_mismatch']+=int(np.sum(ys[idx]!=g.late_trues.to_numpy()))
                counts['concept_mismatch']+=sum(cs[i]!=','.join(sorted(str(c).split(','),key=int)) for i,c in zip(idx,g.concepts))
                counts['incomplete_sequences']+=int(not np.array_equal(idx,np.arange(1,len(qs))))
                counts['sequences']+=1;counts['events']+=len(g)
            cohorts,seq=sequence_metrics(d);seq.to_csv(out/f'{ds}_{run}_sequences.csv',index=False)
            cfg=json.loads((hand/f'configs/{ds}_{run}.json').read_text())
            result={'dataset':ds,'model':'qikt_mamba/public_mamba','fold':cfg['params']['fold'],'seed':cfg['params']['seed'],
                'configured_max_epochs':cfg['train_config']['num_epochs'],'completed_epochs':None,
                'metric_provenance':'reconstructed from frozen late_mean window predictions; historical all_results.json and completed epochs unavailable',
                'windowauclate_mean':float(roc_auc_score(d.late_trues,d.late_mean)), 'cohorts':cohorts,'alignment':counts,
                'local_data_sha256':sha(local_path),'frozen_predictions_sha256':sha(path)}
            target=out/f'{ds}_{run}';target.mkdir(exist_ok=True);(target/'all_results.json').write_text(json.dumps(result,indent=2))
            report[ds+'_'+run]=result
            print(ds,run,counts,'windowauclate_mean',result['windowauclate_mean'],flush=True)
            assert not any(counts[k] for k in ['question_mismatch','label_mismatch','concept_mismatch','incomplete_sequences'])
            assert counts['sequences']==len(local)
    (out/'alignment.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
