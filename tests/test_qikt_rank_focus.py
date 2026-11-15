import numpy as np
import pytest
import torch
from pykt.models.qikt_rank_gradient_cache import risk_focus_weights


@pytest.mark.parametrize('mode', ['exponential', 'variance'])
def test_unclipped_weights_match_risk_objective_gradient(mode):
    risk = torch.tensor([.45, .5, .55], dtype=torch.double, requires_grad=True)
    strength = 2.
    objective = (torch.logsumexp(strength*risk, 0)-np.log(3))/strength if mode == 'exponential' else risk.mean()+strength/2*risk.var(unbiased=False)
    gradient = torch.autograd.grad(objective, risk)[0].numpy()
    actual = risk_focus_weights(dict(enumerate(risk.detach().numpy())), mode, strength)
    assert np.allclose(np.array(list(actual.values()))/3, gradient, atol=1e-12)


@pytest.mark.parametrize('mode', ['off', 'exponential', 'variance'])
def test_weights_are_bounded_and_zero_focus_is_identity(mode):
    risks = {1: 0., 2: .5, 3: 1.}
    assert risk_focus_weights(risks, mode, 0) == {1: 1., 2: 1., 3: 1.}
    focused = risk_focus_weights(risks, mode, 10000)
    assert all(.5 <= v <= 1.5 for v in focused.values())
    assert focused[3] >= focused[2] >= focused[1]
    assert risk_focus_weights({}, mode, 1) == {}


@pytest.mark.parametrize('mode,strength', [('unknown', 1), ('variance', -1), ('off', float('nan'))])
def test_invalid_focus_fails_even_for_empty_cohort(mode, strength):
    with pytest.raises(ValueError): risk_focus_weights({}, mode, strength)
