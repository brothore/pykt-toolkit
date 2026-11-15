"""Descriptive pair attribution of a registered fixed mixture; never fit a gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prediction_progress import atomic_json, digest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--paired', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); p = json.loads(args.protocol.read_text())
    frames = []
    for name in ('baseline_predictions','anchor_predictions','candidate_predictions'):
        frames.append(pd.read_csv(p[name], sep='\t', dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True))
    baseline, anchor, candidate = frames; keys = ['orirow','qidx','late_trues','questions','concepts']
    for frame in frames:
        if frame.duplicated(['orirow','qidx']).any() or not frame[keys].equals(baseline[keys]):
            raise ValueError('Prediction identities differ')
    weight = p['rounds'][0]['candidate_weight']
    mixture = (1-weight)*anchor.late_mean.to_numpy()+weight*candidate.late_mean.to_numpy()
    expected = pd.read_csv(args.paired).set_index('orirow')
    rank = lambda margin: (margin>0).astype(float)+.5*(margin==0)
    records = []
    for uid, group in anchor.groupby('orirow',sort=False):
        label = group.late_trues.to_numpy(); positive = label==1; negative = label==0
        if len(label)<20 or min(positive.sum(),negative.sum())<5: continue
        ix = group.index.to_numpy(); a = group.late_mean.to_numpy(); b = candidate.late_mean.to_numpy()[ix]; m = mixture[ix]
        am = a[positive,None]-a[None,negative]; bm = b[positive,None]-b[None,negative]; mm = m[positive,None]-m[None,negative]
        old, new = rank(am), rank(mm); gain = new-old
        auc = float(new.mean())
        if abs(auc-float(expected.loc[uid,'auc']))>1e-12: raise ValueError('Mixture differs from registered result')
        records.append({'orirow':int(uid),'pairs':int(gain.size),'anchor_auc':float(old.mean()),'mixture_auc':auc,
                        'delta_vs_anchor':float(gain.mean()),'corrected_pair_mass':float(gain[gain>0].sum()),
                        'damaged_pair_mass':float(-gain[gain<0].sum()),
                        'corrected_anchor_margin_median':float(np.median(np.abs(am[gain>0]))) if (gain>0).any() else None,
                        'damaged_anchor_margin_median':float(np.median(np.abs(am[gain<0]))) if (gain<0).any() else None,
                        'anchor_source_disagreement_pairs':int((rank(am)!=rank(bm)).sum())})
    if len(records)!=773: raise ValueError('Unexpected primary cohort')
    atomic_json(args.output, {'scope':'Descriptive validation primary773; no training/routing or causal claim',
        'mixture_weight':weight,'N':len(records),'improved_vs_anchor':sum(r['delta_vs_anchor']>1e-12 for r in records),
        'worsened_vs_anchor':sum(r['delta_vs_anchor']< -1e-12 for r in records),
        'macro_delta_vs_anchor':float(np.mean([r['delta_vs_anchor'] for r in records])),
        'corrected_pair_mass':sum(r['corrected_pair_mass'] for r in records),
        'damaged_pair_mass':sum(r['damaged_pair_mass'] for r in records),
        'note':'Pair mass is interaction-pair weighted; it cannot replace equal-sequence macro gains.',
        'per_orirow':records,'input_sha256':{str(Path(p[n])):digest(Path(p[n])) for n in ('baseline_predictions','anchor_predictions','candidate_predictions')}})


if __name__ == '__main__': main()
