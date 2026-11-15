"""Build one fixed prediction average without fitting or evaluating weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from prediction_progress import atomic_json, digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());identity={'spec':digest(args.spec),
        'inputs':{p:digest(Path(p)) for p in spec['predictions']},'source':digest(Path(__file__))}
    manifest=args.output.with_suffix('.identity.json')
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['identity']!=identity or digest(args.output)!=old['output_sha256']:raise ValueError('Average identity changed')
        return
    weights=np.asarray(spec['weights'],dtype=float)
    if not len(weights)==len(spec['predictions']) or not np.isfinite(weights).all() or (weights<0).any() or abs(weights.sum()-1)>1e-12:
        raise ValueError('Invalid fixed weights')
    frames=[pd.read_csv(p,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True) for p in spec['predictions']]
    keys=['orirow','qidx','late_trues','questions','concepts'];base=frames[0].copy()
    for frame in frames:
        if frame.duplicated(['orirow','qidx']).any() or not frame[keys].equals(base[keys]):raise ValueError('Prediction alignment differs')
        if not np.isfinite(frame.late_mean).all() or not frame.late_mean.between(0,1).all():raise ValueError('Invalid probability')
    values=np.stack([frame.late_mean.to_numpy() for frame in frames])
    if spec['mode']=='logit':
        values=np.clip(values,1e-6,1-1e-6);values=np.log(values)-np.log1p(-values)
    elif spec['mode']!='probability':raise ValueError('Unknown fixed averaging mode')
    mean=np.sum(values*weights[:,None],axis=0)
    base['late_mean']=expit(mean) if spec['mode']=='logit' else mean
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');base.to_csv(temporary,sep='\t',index=False,float_format='%.17g');temporary.replace(args.output)
    atomic_json(manifest,{'identity':identity,'output_sha256':digest(args.output),'events':len(base),
                         'scope':'Fixed averaging only; no fitting, model training, inference or test evaluation'})


if __name__=='__main__':main()
