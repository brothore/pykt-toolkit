"""Pure diagnostics for paired sequence AUCs; no fitting or model selection."""
import numpy as np


def diagnose(paired):
    p = paired[(paired.n >= 20) & (paired.minority >= 5)].copy()
    if p.empty or not np.isfinite(p[['auc', 'baseline_auc', 'delta_auc']]).all().all():
        raise ValueError('Missing finite primary sequence AUCs')
    b = p.baseline_auc.to_numpy(); a = p.auc.to_numpy(); d = p.delta_auc.to_numpy()
    if not np.allclose(a - b, d, atol=1e-12, rtol=0):
        raise ValueError('Paired sequence AUC deltas disagree')
    covariance = float(np.mean((b - b.mean()) * (d - d.mean())))
    variance_delta = float(a.var(ddof=0) - b.var(ddof=0))
    identity = 2 * covariance + float(d.var(ddof=0))
    if not np.isclose(variance_delta, identity, atol=1e-12, rtol=0):
        raise ValueError('Variance identity failed')
    groups = {}
    for fraction in [.05, .25]:
        lo, hi = np.quantile(b, [fraction, 1 - fraction])
        for name, mask in [('low', b <= lo), ('high', b >= hi)]:
            values = d[mask]
            groups[f'baseline_{name}_{fraction:g}'] = {
                'N': len(values), 'mean_delta': float(values.mean()),
                'worsened': int((values < -1e-12).sum()),
            }
    edges = np.quantile(p.switch_rate, [.25, .5, .75])
    bins = np.searchsorted(edges, p.switch_rate, side='left')
    switch_groups = []
    for i in range(4):
        values = d[bins == i]
        switch_groups.append({'quartile': i + 1, 'N': len(values),
                              'mean_delta': float(values.mean()) if len(values) else None})
    return {
        'N': len(p), 'baseline_std_ddof0': float(b.std(ddof=0)),
        'candidate_std_ddof0': float(a.std(ddof=0)),
        'baseline_range': float(np.ptp(b)), 'candidate_range': float(np.ptp(a)),
        'variance_delta': variance_delta, 'twice_baseline_delta_covariance': 2 * covariance,
        'delta_variance': float(d.var(ddof=0)), 'variance_identity_residual': variance_delta - identity,
        'baseline_groups': groups, 'switch_quartiles': switch_groups,
        'improved': int((d > 1e-12).sum()), 'worsened': int((d < -1e-12).sum()),
        'strict_no_regression': bool((d >= -1e-12).all()),
        'scope': 'Validation-only primary orirows. Switch rates describe complete label sequences post hoc; no causal claim, per-UID model rules, test reads or candidate selection.',
    }
