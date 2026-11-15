"""Prequential residual state: score t before ingesting response t."""
from collections import deque
import numpy as np
import pandas as pd
from scipy.special import expit


def calibrate_frame(frame, mode, prior, radius, memory=200):
    if mode not in {'global','kc','bucket','mixed','joint'} or not np.isfinite([prior,radius]).all() or prior<=0 or radius<=0 or memory<1:
        raise ValueError('Invalid prefix residual protocol')
    result=frame.sort_values(['orirow','qidx']).reset_index(drop=True).copy()
    if result.duplicated(['orirow','qidx']).any():raise ValueError('Duplicate prediction keys')
    if not result.late_mean.between(0,1).all() or not result.late_trues.isin([0,1]).all():raise ValueError('Invalid score/label')
    output=np.empty(len(result));max_history=0
    for _,group in result.groupby('orirow',sort=False):
        history=deque();totals={}
        def adjust(keys, error, fisher, direction):
            for key in keys:
                e,v,n=totals.get(key,(0.,0.,0))
                totals[key]=((e+direction*error,v+direction*fisher,n+direction)
                             if n+direction else (0.,0.,0))
        def correction(key):
            e,v,n=totals.get(key,(0.,0.,0))
            if n<0:raise ValueError('Negative history support')
            return e/(max(0.,v)+prior)
        for row in group.itertuples():
            time=int(row.qidx);p=float(row.late_mean)
            while history and history[0][0]<time-memory:
                _,keys,error,fisher=history.popleft();adjust(keys,error,fisher,-1)
            if history and history[-1][0]>=time:raise ValueError('History includes current/future response')
            concepts=sorted(set(map(int,str(row.concepts).split(','))))
            if not concepts or min(concepts)<0:raise ValueError('Invalid KC identity')
            bucket=min(4,int(p*5))
            kc_keys=[('kc',c) for c in concepts];joint_keys=[('joint',c,bucket) for c in concepts]
            global_delta=correction(('global',));kc_delta=np.mean([correction(key) for key in kc_keys])
            bucket_delta=correction(('bucket',bucket));joint_delta=np.mean([correction(key) for key in joint_keys])
            delta={'global':global_delta,'kc':kc_delta,'bucket':bucket_delta,
                   'mixed':.5*(kc_delta+bucket_delta),'joint':joint_delta}[mode]
            safe=np.clip(p,1e-6,1-1e-6)
            output[row.Index]=expit(np.log(safe)-np.log1p(-safe)+np.clip(delta,-radius,radius))
            max_history=max(max_history,len(history))
            # The current response is consumed only after emitting its score.
            error=float(row.late_trues)-p;fisher=p*(1-p)
            keys=[('global',),('bucket',bucket),*kc_keys,*joint_keys]
            adjust(keys,error,fisher,1);history.append((time,keys,error,fisher))
    result['late_mean']=output
    return result,{'scope':'Strictly prior scored events, current prediction precedes current label update',
                   'memory_positions':memory,'maximum_prior_events_used':max_history,
                   'uses_position_zero':False,'state_resets_at_each_orirow':True,
                   'additional_streaming_memory':True,'offline_fitting':False}
