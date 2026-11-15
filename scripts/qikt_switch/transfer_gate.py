"""Screen fixed methods for Peiyou; never refit them using Nips test outcomes."""
import math


def assess_transfer(outcome):
    c = outcome['comparison']
    if c['metric_scope'] != 'windowauclate_mean':
        raise ValueError('Transfer requires formal windowauclate_mean comparison')
    measured = outcome['result']['windowauclate_mean']
    if not math.isfinite(measured) or abs(measured - c['candidate_windowauclate_mean']) > 1e-12:
        raise ValueError('Formal result and aligned comparison disagree')
    ci = c['bootstrap']['pooled_delta_ci95']
    groups = c['baseline_performance_groups']
    checks = {
        'main_metric_increased': c['pooled_delta'] > 0,
        'orirow_bootstrap_lower_bound_positive': math.isfinite(ci[0]) and ci[0] > 0,
        'baseline_low_group_mean_preserved': groups['baseline_low']['mean_delta'] >= 0,
        'baseline_high_group_mean_preserved': groups['baseline_high']['mean_delta'] >= 0,
    }
    base = c['baseline_cohorts']['primary_20_5']
    candidate = c['candidate_cohorts']['primary_20_5']
    return {
        'candidate': outcome['candidate'],
        'eligible_for_peiyou': all(checks.values()),
        'checks': checks,
        'primary_std_nonincreasing': candidate['std'] <= base['std'],
        'primary_range_nonincreasing': candidate['range'] <= base['range'],
        'primary_worsened': c['primary_paired']['worsened'],
        'strict_primary_no_regression_pass': c['strict_primary_no_regression_pass'],
        'objective_achieved': False,
        'note': 'Fixed-method transfer screening only; no Nips refitting. Orirow bootstrap is not training-seed replication or proof of individual protection.',
    }
