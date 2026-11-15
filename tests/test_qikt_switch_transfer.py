import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'))
from transfer_gate import assess_transfer
import run_confirmation_pipeline as pipeline


def outcome():
    return {'candidate': 'fixed', 'result': {'windowauclate_mean': .81}, 'comparison': {
        'metric_scope': 'windowauclate_mean', 'candidate_windowauclate_mean': .81,
        'pooled_delta': .01, 'bootstrap': {'pooled_delta_ci95': [.002, .018]},
        'baseline_performance_groups': {'baseline_low': {'mean_delta': .01}, 'baseline_high': {'mean_delta': .001}},
        'baseline_cohorts': {'primary_20_5': {'std': .1, 'range': .5}},
        'candidate_cohorts': {'primary_20_5': {'std': .11, 'range': .49}},
        'primary_paired': {'worsened': 2}, 'strict_primary_no_regression_pass': False}}


def test_gate_preserves_individual_failures_and_does_not_modify_frozen_results():
    o = outcome(); original = copy.deepcopy(o); result = assess_transfer(o)
    assert result['eligible_for_peiyou']
    assert not result['objective_achieved'] and not result['strict_primary_no_regression_pass']
    assert not result['primary_std_nonincreasing'] and result['primary_worsened'] == 2
    assert o == original


@pytest.mark.parametrize('failure', ['auc', 'uncertainty', 'nan', 'low', 'high'])
def test_no_expensive_transfer_without_nips_evidence(failure):
    o = outcome(); c = o['comparison']
    if failure == 'auc': c['pooled_delta'] = 0
    elif failure == 'uncertainty': c['bootstrap']['pooled_delta_ci95'][0] = -.001
    elif failure == 'nan': c['bootstrap']['pooled_delta_ci95'][0] = float('nan')
    else: c['baseline_performance_groups']['baseline_' + failure]['mean_delta'] = -.001
    assert not assess_transfer(o)['eligible_for_peiyou']


def test_gate_requires_exact_main_metric():
    o = outcome(); o['comparison']['metric_scope'] = 'validation_window_auc'
    with pytest.raises(ValueError): assess_transfer(o)
    o = outcome(); o['result']['windowauclate_mean'] = .8
    with pytest.raises(ValueError): assess_transfer(o)


def test_controller_handoff_keeps_existing_inference_and_does_not_launch_duplicate(tmp_path, monkeypatch):
    script = tmp_path / 'evaluate_window.py'; target = tmp_path / 'all_results.json'
    script.write_text('import time,sys\ntime.sleep(.5)\nopen(sys.argv[1],"w").write("{}")\n')
    command = [sys.executable, str(script), str(target)]
    child = subprocess.Popen(command)
    original_sleep = time.sleep
    monkeypatch.setattr(pipeline.time, 'sleep', lambda delay: original_sleep(.02))
    def no_duplicate(*args): raise AssertionError('Attempted to restart running inference')
    monkeypatch.setattr(pipeline, 'run_command', no_duplicate)
    try:
        pipeline.run_or_attach_inference(command, target, tmp_path / 'inference.log')
        assert child.wait(timeout=5) == 0 and target.exists()
        assert json.loads((tmp_path / 'inference_attachment.json').read_text())['pid'] == child.pid
    finally:
        if child.poll() is None: child.terminate(); child.wait()


def test_inference_discovery_never_reads_unrelated_process_argv(tmp_path,monkeypatch):
    unrelated=tmp_path/'999';unrelated.mkdir();(unrelated/'comm').write_text('rg\n')
    original_iterdir=Path.iterdir;original_read=Path.read_bytes
    def iterdir(path):
        return iter([unrelated]) if str(path)=='/proc' else original_iterdir(path)
    def read(path):
        if path==unrelated/'cmdline':raise AssertionError('Unrelated process argv must not be read')
        return original_read(path)
    monkeypatch.setattr(Path,'iterdir',iterdir);monkeypatch.setattr(Path,'read_bytes',read)
    target=tmp_path/'result.json'
    monkeypatch.setattr(pipeline,'run_command',lambda *_:target.write_text('{}'))
    pipeline.run_or_attach_inference([sys.executable,'evaluate_window.py'],target,tmp_path/'inference.log')
    assert target.exists()
