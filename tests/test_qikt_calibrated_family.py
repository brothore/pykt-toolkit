"""A failed calibration must block launches; a registered signal must skip them."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run_controller(tmp_path, status, signal_found=False):
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'status': status}))
    result = tmp_path / 'calibration.json'
    result.write_text('{}')  # Skip/failure must happen before recommendations are read.
    signals = tmp_path / 'upstream'
    if signal_found:
        folder = signals / 'I4' / 'round2'
        folder.mkdir(parents=True)
        (folder / 'balance.json').write_text(json.dumps({'substantive_development_signal': True}))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'calibration_state': str(state), 'calibration_result': str(result),
        'admission_record': str(tmp_path / 'admission.json'),
        'signal_roots': [str(signals)], 'result': str(tmp_path / 'summary.json'),
        'command': [sys.executable, '-c', 'raise RuntimeError("must never launch")'],
    }))
    script = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch/run_calibrated_family.py'
    return subprocess.run([sys.executable, str(script), '--spec', str(spec)],
                          capture_output=True, text=True, timeout=15,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})


def test_failed_dependency_does_not_launch(tmp_path):
    run = run_controller(tmp_path, 'failed')
    assert run.returncode != 0
    assert 'Calibration requires review' in run.stderr
    assert not (tmp_path / 'summary.json').exists()


def test_upstream_signal_skips_before_cuda_or_calibration_selection(tmp_path):
    run = run_controller(tmp_path, 'completed', signal_found=True)
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['status'] == 'skipped_existing_development_signal'
    assert result['executed'] is False and len(result['signals']) == 1


def test_cross_batch_shared_limit_until_peer_science_is_terminal(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
    from run_calibrated_family import effective_slots
    rec={phase:{'recommended_concurrency':2} for phase in ['training','inference']};peer=tmp_path/'peer.json';spec={'shared_slot_limit':1,'shared_peer_states':[str(peer)]}
    assert effective_slots(rec,spec)==1
    peer.write_text(json.dumps({'status':'running'}));assert effective_slots(rec,spec)==1
    peer.write_text(json.dumps({'status':'completed'}));assert effective_slots(rec,spec)==2
    assert effective_slots(rec,{})==2
