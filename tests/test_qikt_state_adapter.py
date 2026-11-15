import importlib
import sys
from pathlib import Path
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from test_qikt_causal_adapter import adapter
from test_qikt_switch_frozen import model, batch
from pykt.models.qikt_causal_adapter import CausalCorrection, baseline_state_features

MODES = ['state_only', 'state_multiscale', 'state_interaction']


def test_baseline_state_uses_only_detached_probabilities():
    heads = [torch.tensor([[v]], requires_grad=True) for v in [.2, .4, .7]]
    outputs = dict(zip(['y_question_all', 'y_concept_all', 'y_concept_next'], heads))
    features = baseline_state_features(outputs, torch.tensor([[0.]], requires_grad=True))
    assert features.shape == (1, 1, 6) and not features.requires_grad
    assert torch.allclose(features, torch.tensor([[[.2, .4, .7, .5, .5, 1.]]]))


def test_default_module_keeps_old_rng_parameters_and_layout():
    class Original(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 64, bias=False)
            self.hidden = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
            self.output = nn.Linear(64, 1)
            nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    torch.manual_seed(71); old = Original(); after = torch.get_rng_state()
    torch.manual_seed(71); new = CausalCorrection(4)
    assert torch.equal(after, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())


@pytest.mark.parametrize('mode', MODES)
def test_state_adapter_zero_parity_freezing_causality_and_checkpoint(mode, tmp_path):
    teacher = model(); teacher.model.eval(); candidate = adapter(mode, kl=1.)
    missing = candidate.load_state_dict(teacher.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('model.causal_adapter.') for k in missing.missing_keys)
    expected = teacher.predict_one_step(batch(), process=False)
    assert torch.equal(expected, candidate.predict_one_step(batch(), process=False))
    optimizer = torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        candidate.model.train(); optimizer.zero_grad()
        _, loss = candidate.train_one_step(batch(), process=False)
        loss.backward(); optimizer.step()
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in teacher.state_dict().items())
    assert all(p.grad is None for k, p in candidate.named_parameters() if not k.startswith('model.causal_adapter.'))
    candidate.model.eval()
    reference = candidate.predict_one_step(batch(), process=False)
    assert not torch.equal(expected, reference)
    for time in range(1, 5):
        changed = batch(); changed['cr'][:, time:] = 1 - changed['cr'][:, time:]
        changed['rshft'] = changed['cr'][:, 1:].float()
        actual = candidate.predict_one_step(changed, process=False)
        assert torch.equal(reference[:, :time], actual[:, :time])
    module = importlib.import_module('pykt.models.init_model'); old_device = module.device
    try:
        module.device = 'cpu'
        cfg = dict(emb_size=4, dropout=.3, version='lstm', other_config={
            'output_mode': 'an_irt', 'causal_adapter_mode': mode,
            'causal_adapter_scale': .5, 'causal_adapter_kl': 1.})
        restored = module.init_model('qikt_mamba', cfg,
            {'num_q': 3, 'num_c': 2, 'max_concepts': 2, 'emb_path': ''}, 'attn')
        path = tmp_path / 'checkpoint.pt'; torch.save(candidate.state_dict(), path)
        restored.load_state_dict(torch.load(path, weights_only=True)); restored.model.eval()
        assert torch.equal(reference, restored.predict_one_step(batch(), process=False))
    finally:
        module.device = old_device
