import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


def load_runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('discovery_runner', scripts / 'run_neural.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda _: object())
    return module


def test_discovery_parameters_reach_actual_training_command_without_launching(tmp_path, monkeypatch):
    runner = load_runner(monkeypatch)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'model_settings': {'item_residual': 1, 'history_response_dropout': .005}}))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['run_neural', '--variant', 'baseline', '--resumable',
                                     '--objective-protocol', str(protocol), '--output-root', str(output), '--preflight-only'])
    runner.main()
    requested = json.loads((output / 'baseline/requested_run.json').read_text())
    assert requested['params']['item_residual'] == 1
    assert requested['params']['history_response_dropout'] == .005
    command = requested['command']
    assert command[command.index('--item_residual') + 1] == '1'
    assert not (output / 'baseline/training_started.json').exists()


def test_discovery_protocol_cannot_override_data_split(tmp_path, monkeypatch):
    runner = load_runner(monkeypatch)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'model_settings': {'fold': 1}}))
    monkeypatch.setattr(sys, 'argv', ['run_neural', '--variant', 'baseline', '--resumable',
                                     '--objective-protocol', str(protocol), '--output-root', str(tmp_path / 'output'), '--preflight-only'])
    with pytest.raises(ValueError, match='Unsupported discovery model settings'):
        runner.main()


def test_frequency_rank_composition_requires_opt_in_and_has_finite_gradient(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    from test_qikt_switch_frozen import model, batch
    from pykt.models.qikt_sequence_objectives import install_training_objective
    sample = batch()
    sample['uid'] = torch.tensor([1])
    labels, concepts = sample['rshft'], sample['cshft']

    class Data:
        aug_probs = {}
        dori = {'uid': torch.tensor([1]), 'rseqs': torch.cat([labels[:, :1], labels], 1),
                'smasks': torch.ones(1, labels.shape[1] + 1, dtype=torch.long),
                'cseqs': torch.cat([concepts[:, :1], concepts], 1)}

        def __len__(self):
            return 1

    protocol = {'frequency_alpha': .1, 'rank_weight': .002}
    with pytest.raises(ValueError, match='explicit discovery opt-in'):
        install_training_objective(model(), Data(), protocol)
    composed = model()
    install_training_objective(composed, Data(), dict(protocol, allow_frequency_rank_combination=True))
    _, loss = composed.train_one_step(sample, process=False)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value.grad).all() for value in composed.parameters() if value.grad is not None)
