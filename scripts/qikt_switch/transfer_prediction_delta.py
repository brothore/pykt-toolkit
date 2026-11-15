"""Fixed score-space residual transfer; no current/future response is read."""
import numpy as np
import pandas as pd
from scipy.special import expit


def transfer(base,student,anchor,mode,strength):
    if mode not in {'raw','uncertain','prefix_center'} or not np.isfinite(strength) or strength<0:
        raise ValueError('Invalid prediction transfer settings')
    keys=['orirow','qidx','late_trues','questions','concepts']
    frames=[f.sort_values(['orirow','qidx']).reset_index(drop=True) for f in [base,student,anchor]]
    if any(f.duplicated(['orirow','qidx']).any() or not f[keys].equals(frames[0][keys]) for f in frames):
        raise ValueError('Prediction identities differ')
    for frame in frames:
        if not frame.late_mean.between(0,1).all():raise ValueError('Invalid probability')
    logits=lambda p:np.log(np.clip(p,1e-6,1-1e-6))-np.log1p(-np.clip(p,1e-6,1-1e-6))
    b,s,a=[frame.late_mean.to_numpy() for frame in frames]
    delta=logits(s)-logits(b)
    if mode=='uncertain':delta=delta*(4*a*(1-a))**2
    if mode=='prefix_center':
        mean=np.zeros(len(delta))
        for _,group in frames[0].groupby('orirow',sort=False):
            indices=group.index.to_numpy();values=pd.Series(delta[indices])
            total=values.rolling(200,min_periods=1).sum().shift(1).fillna(0).to_numpy()
            count=np.minimum(np.arange(len(indices)),200)
            mean[indices]=total/(count+10)
        delta=delta-mean
    result=frames[0].copy()
    result['late_mean']=expit(logits(a)+strength*delta).astype(np.float32).astype(np.float64)
    return result
