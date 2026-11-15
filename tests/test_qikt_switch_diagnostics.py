import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/qikt_switch'))
from validation_diagnostics import diagnose


def paired():
    baseline = np.array([.5, .6, .8, .9])
    delta = np.array([.1, .05, -.05, -.1])
    return pd.DataFrame({'n': 100, 'minority': 20, 'baseline_auc': baseline,
                         'auc': baseline + delta, 'delta_auc': delta,
                         'switch_rate': [.1, .2, .3, .4]})


def test_lower_variance_does_not_hide_harm_to_high_baseline_sequences():
    result = diagnose(paired())
    assert result['candidate_std_ddof0'] < result['baseline_std_ddof0']
    assert result['variance_delta'] < 0 and abs(result['variance_identity_residual']) < 1e-12
    assert result['twice_baseline_delta_covariance'] < 0 and result['delta_variance'] > 0
    assert result['worsened'] == 2 and not result['strict_no_regression']
    assert result['baseline_groups']['baseline_high_0.25']['mean_delta'] < 0


def test_stopped_trial_analysis_resumes_checkpoint_without_rewriting_it(tmp_path):
    neural = tmp_path / 'neural'; trial = neural / 'scheme'; window = trial / 'validation_window'
    window.mkdir(parents=True); source = window / 'paired_orirow.csv'; paired().to_csv(source, index=False)
    (trial / 'round_summary.json').write_text(json.dumps({'variant': 'scheme', 'dataset': 'nips_task34',
        'fold': 0, 'seed': 42, 'model': 'qikt_mamba/public_mamba', 'completed_epochs': 14, 'selected_epoch': 4}))
    out = tmp_path / 'out'
    command = [sys.executable, str(ROOT / 'scripts/qikt_switch/analyze_validation_trials.py'),
               '--neural-root', str(neural), '--output-dir', str(out), '--workers', '1']
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    subprocess.run(command, env=env, check=True, capture_output=True)
    checkpoint = out / 'scheme.json'; before = checkpoint.stat().st_mtime_ns
    state_path = out / 'state.json'; state = json.loads(state_path.read_text())
    state['status'] = 'stopped'; state['exit_code'] = 130; state_path.write_text(json.dumps(state))
    subprocess.run(command + ['--resume'], env=env, check=True, capture_output=True)
    assert checkpoint.stat().st_mtime_ns == before
    state = json.loads(state_path.read_text())
    assert state['status'] == 'completed' and state['exit_code'] == 0
    assert [h['event'] for h in state['history']][-2:] == ['resume', 'completed']
    changed = paired(); changed.loc[0, 'n'] = 101; changed.to_csv(source, index=False)
    invalid = subprocess.run(command + ['--resume'], env=env, capture_output=True, text=True)
    assert invalid.returncode != 0 and 'Inputs or analysis code changed' in invalid.stderr
    assert checkpoint.stat().st_mtime_ns == before
