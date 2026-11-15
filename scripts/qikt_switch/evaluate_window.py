"""Memory-bounded native-question window inference for frozen QIKT checkpoints.

QIKT consumes all KCs of a question jointly, producing one question score.
Per-question mean fusion is therefore an identity; repeated question IDs in
legacy output are serialization of multiple KCs, not extra predictions.
Compute windowauclate_mean from the exported per-question scores explicitly.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from contextlib import nullcontext
from prediction_progress import PredictionProgress,atomic_json
from sklearn.metrics import roc_auc_score
from evaluate_validation import load_run,load_saved_model,sha256_file
from audit_inputs import sequence_metrics


class QuestionWindows(Dataset):
    def __init__(self,raw,length,max_concepts):
        if length<3:raise ValueError('Window length must be >=3')
        if not raw.uid.is_unique:raise ValueError('UIDs must be unique')
        self.length=length;self.max_concepts=max_concepts;self.rows=[];self.windows=[]
        for row in raw.itertuples():
            q=np.fromstring(row.questions,sep=',',dtype=np.int64);r=np.fromstring(row.responses,sep=',',dtype=np.float32)
            concepts=[list(map(int,c.split('_'))) for c in row.concepts.split(',')]
            if not len(q)==len(r)==len(concepts) or not np.isin(r,[0,1]).all():raise ValueError('Invalid aligned question/response/KC sequence')
            if any(len(c)>max_concepts for c in concepts):raise ValueError('Too many KCs')
            c=np.full((len(q),max_concepts),-1,dtype=np.int64)
            for i,values in enumerate(concepts):c[i,:len(values)]=values
            t=np.fromstring(row.timestamps,sep=',',dtype=np.int64) if hasattr(row,'timestamps') else np.zeros(len(q),dtype=np.int64)
            if len(t)!=len(q):raise ValueError('Timestamp alignment differs')
            index=len(self.rows);self.rows.append(dict(uid=int(row.uid),q=q,c=c,r=r,t=t))
            if len(q)<3:raise ValueError('Raw evaluation sequence violates preprocessing minimum length')
            first=min(len(q),length);self.windows.append((index,first,True))
            self.windows.extend((index,stop,False) for stop in range(length+1,len(q)+1))

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        row_idx,stop,first=self.windows[index];row=self.rows[row_idx];start=max(0,stop-self.length);size=stop-start
        q=np.full(self.length,-1,dtype=np.int64);r=np.full(self.length,-1,dtype=np.float32);c=np.full((self.length,self.max_concepts),-1,dtype=np.int64);t=np.full(self.length,-1,dtype=np.int64)
        for target,key in [(q,'q'),(r,'r'),(c,'c'),(t,'t')]:target[:size]=row[key][start:stop]
        mask=(r[:-1]!=-1)&(r[1:]!=-1);select=mask.copy()
        if not first:select[:]=False;select[size-2]=True
        data={'uid':torch.tensor(row['uid']),'masks':torch.from_numpy(mask),'smasks':torch.from_numpy(select),
              'cseqs':torch.from_numpy(c[:-1]),'shft_cseqs':torch.from_numpy(c[1:])}
        for values,key in [(q,'qseqs'),(r,'rseqs'),(t,'tseqs')]:
            data[key]=torch.from_numpy(values[:-1]*mask);data['shft_'+key]=torch.from_numpy(values[1:]*mask)
        positions=np.full(self.length-1,-1,dtype=np.int64);positions[:size-1]=np.arange(start+1,stop)
        return data,{'orirow':row['uid'],'qidx':torch.from_numpy(positions)}


def evaluate_window_checkpoint(checkpoint_dir,output_dir,batch_size=32,device='auto',source='test',mixture_path=None,resume=False,checkpoint_batches=0):
    started=time.perf_counter()
    config,paths=load_run(checkpoint_dir);cfg=config['data_config']
    if paths['emb_type'] not in ('attn','iekt'):raise ValueError('Window input path currently supports attn/iekt QIKT only')
    if source not in ('test','validation'):raise ValueError('Unknown prediction source')
    raw_path=(paths['data_dir']/cfg.get('test_original_file_quelevel','test_quelevel.csv')) if source=='test' else paths['raw_path']
    raw=pd.read_csv(raw_path)
    if source=='validation':raw=raw[raw.fold==paths['fold']].copy()
    dataset=QuestionWindows(raw,int(cfg.get('maxlen',200)),int(cfg['max_concepts']))
    progress=None
    if checkpoint_batches:
        if checkpoint_batches<1:raise ValueError('checkpoint_batches must be positive')
        identity={'config':sha256_file(Path(checkpoint_dir)/'config.json'),'checkpoint':sha256_file(paths['checkpoint_path']),
            'raw_data':sha256_file(raw_path),'source':source,'batch_size':batch_size,'device':device,
            'mixture':sha256_file(mixture_path) if mixture_path else None,
            'code':{str(p):sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('prediction_progress.py'),Path(__file__).with_name('evaluate_validation.py'),Path(__file__).resolve().parents[2]/'pykt/models/qikt_mamba.py',Path(__file__).resolve().parents[2]/'pykt/models/qikt_switch_utils.py']}}
        progress=PredictionProgress(output_dir,identity,resume)
    components=[]
    if mixture_path:
        spec=json.loads(Path(mixture_path).read_text())
        for component in spec['components']:
            weight=float(component['weight'])
            if weight<0 or not np.isfinite(weight):raise ValueError('Invalid mixture weight')
            if not weight:continue
            current,cp=load_run(component['checkpoint_dir'])
            if any(cp[k]!=paths[k] for k in ('dataset','fold','seed','emb_type')) or any(current['data_config'][k]!=cfg[k] for k in ('num_q','num_c','max_concepts','maxlen')):
                raise ValueError('Mixture component configuration differs')
            digest=sha256_file(cp['checkpoint_path'])
            if component.get('checkpoint_sha256',digest)!=digest:raise ValueError('Frozen component checkpoint changed')
            stats=cp['checkpoint_dir']/'training_stats.json';training=json.loads(stats.read_text()) if stats.exists() else {}
            components.append({'model':load_saved_model(current,cp,device=device),'weight':weight,'checkpoint_dir':str(cp['checkpoint_dir']),'checkpoint_sha256':digest,'completed_epochs':training.get('completed_epochs'),'selected_epoch':cp['actual_epoch']})
        if not components or not np.isclose(sum(c['weight'] for c in components),1.,atol=1e-12,rtol=0):raise ValueError('Mixture weights must sum to one')
    else:
        components=[{'model':load_saved_model(config,paths,device=device),'weight':1.}]
    model=components[0]['model'];records=[]
    next_window=progress.state['next_window'] if progress else 0
    remaining=Subset(dataset,range(next_window,len(dataset)))
    with progress.signals() if progress else nullcontext(),torch.no_grad():
        for batch_index,(data,meta) in enumerate(DataLoader(remaining,batch_size=batch_size,shuffle=False)):
            p=sum(c['weight']*c['model'].predict_one_step(data).detach().cpu().numpy().astype(np.float64) for c in components)
            mask=data['smasks'].numpy();labels=data['shft_rseqs'].numpy();qs=data['shft_qseqs'].numpy();cs=data['shft_cseqs'].numpy()
            for b,i in zip(*np.where(mask)):
                concepts=[int(x) for x in cs[b,i] if x!=-1];question=int(qs[b,i])
                records.append((int(meta['orirow'][b]),int(meta['qidx'][b,i]),int(labels[b,i]),float(p[b,i]),','.join([str(question)]*len(concepts)),','.join(map(str,reversed(concepts)))))
            next_window+=len(meta['orirow'])
            if progress and ((batch_index+1)%checkpoint_batches==0 or progress.stopping):
                progress.flush(records,next_window);records=[]
                if progress.stopping:raise SystemExit(130)
        if progress:
            progress.flush(records,next_window);records=progress.records()
    frame=pd.DataFrame(records,columns=['orirow','qidx','late_trues','late_mean','questions','concepts'])
    if frame.duplicated(['orirow','qidx']).any():raise ValueError('A question was scored multiple times')
    # Validate all exported positions against the original full question sequences.
    groups={int(uid):g.sort_values('qidx') for uid,g in frame.groupby('orirow')}
    for row in dataset.rows:
        g=groups[row['uid']];idx=g.qidx.to_numpy();assert np.array_equal(idx,np.arange(1,len(row['q'])))
        assert np.array_equal(g.late_trues.to_numpy(),row['r'][idx])
        assert all(set(map(int,str(value).split(',')))=={int(q)} for value,q in zip(g.questions,row['q'][idx]))
        assert all(sorted(map(int,str(value).split(',')))==sorted(int(x) for x in row['c'][i] if x!=-1) for value,i in zip(g.concepts,idx))
    # Native QIKT has one score per original question; the mean-fusion group is a singleton.
    grouped=frame.groupby(['orirow','qidx'],sort=False).agg(late_trues=('late_trues','first'),late_mean=('late_mean','mean')).reset_index()
    auc=float(roc_auc_score(grouped.late_trues,grouped.late_mean)) if grouped.late_trues.nunique()==2 else None
    cohorts,seq=sequence_metrics(frame)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output_dir/'window_predictions.tsv',sep='\t',index=False,float_format='%.17g');seq.to_csv(output_dir/'orirow_auc.csv',index=False)
    metric_key='windowauclate_mean' if source=='test' else 'validation_window_auc'
    result={'dataset':paths['dataset'],'model':paths['model_name'],'version':paths['version'],'fold':paths['fold'],'seed':paths['seed'],
       'configured_epochs':paths['configured_epochs'],'selected_epoch':paths['actual_epoch'],metric_key:auc,'cohorts':cohorts,'source':source,
       'events':len(frame),'orirows':len(groups),'windows':len(dataset),'window_length':dataset.length,
       'coverage':'Every original question position 1..n-1, exactly once; IDs/labels/KCs aligned against raw sequences',
       'score_provenance':'Native QIKT joint-KC question prediction, explicit per-(orirow,qidx) mean fusion; singleton groups. Repeated question IDs only serialize KCs.',
       'metric_provenance':'ROC AUC recomputed from exported mean-fused original-question window scores; not copied/renamed from validation_auc or window_testauc',
       'device':str(next(model.parameters()).device),'raw_data_sha256':sha256_file(raw_path),'checkpoint_sha256':sha256_file(paths['checkpoint_path'])}
    stats=paths['checkpoint_dir']/'training_stats.json'
    result['completed_epochs']=json.loads(stats.read_text()).get('completed_epochs') if stats.exists() else None
    if mixture_path:
        result['model']='qikt_mamba_prediction_mixture'
        result['components']=[{k:v for k,v in c.items() if k!='model'} for c in components]
        result['mixture_spec_sha256']=sha256_file(mixture_path)
        result['completed_epochs']=None;result['selected_epoch']=None
    result['elapsed_seconds']=time.perf_counter()-started
    result['test_evaluated']=source=='test'
    atomic_json(output_dir/('all_results.json' if source=='test' else 'validation_window_results.json'),result)
    if progress:progress.mark('completed');progress.close()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');ap.add_argument('--source',choices=['test','validation'],default='test');ap.add_argument('--mixture-json',type=Path);ap.add_argument('--resume',action='store_true');ap.add_argument('--checkpoint-batches',type=int,default=0);args=ap.parse_args()
    print(json.dumps(evaluate_window_checkpoint(args.checkpoint_dir,args.output_dir,args.batch_size,args.device,args.source,args.mixture_json,args.resume,args.checkpoint_batches)),flush=True)

if __name__=='__main__':main()
