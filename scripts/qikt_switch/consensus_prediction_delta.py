"""Fixed causal score algebra of two existing corrections; no gate fitting."""
import numpy as np
from scipy.special import expit


def consensus(base, first, second, mode, amount):
    if mode not in {'mixture', 'agreement', 'amplified'} or not np.isfinite(amount):
        raise ValueError('Invalid consensus settings')
    if (mode in {'mixture', 'agreement'} and not 0 <= amount <= 1) or (mode == 'amplified' and amount < 0):
        raise ValueError('Invalid consensus coefficient')
    frames = [f.sort_values(['orirow', 'qidx']).reset_index(drop=True) for f in [base, first, second]]
    keys = ['orirow', 'qidx', 'late_trues', 'questions', 'concepts']
    if any(f.duplicated(['orirow', 'qidx']).any() or not f[keys].equals(frames[0][keys]) for f in frames):
        raise ValueError('Prediction identities differ')
    for frame in frames:
        if not frame.late_mean.between(0, 1).all():
            raise ValueError('Invalid probabilities')
    def logit(frame):
        p = frame.late_mean.to_numpy().clip(1e-6, 1-1e-6)
        return np.log(p)-np.log1p(-p)
    b, a, h = [logit(f) for f in frames]
    d, e = a-b, h-b
    if mode == 'mixture': delta = (1-amount)*d+amount*e
    elif mode == 'agreement': delta = np.where(d*e > 0, d, amount*d)
    else: delta = amount*(d+e)/2
    output = frames[0].copy()
    output['late_mean'] = expit(b+delta).astype(np.float32).astype(np.float64)
    return output
