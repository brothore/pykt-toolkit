"""CPU-only tests for the composable AKT reference-mechanism transfers.

Torch is restricted to one thread and CUDA is hidden before importing torch so
these tests cannot create a GPU context.  The real-data pre-check is run by the
main agent, not here.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

from pykt.models.akt import AKT
from pykt.models.akt_fairness import (
    MECHANISMS,
    AKTFairness,
    MTKTCausalInteraction,
    UKTCovarianceResidual,
)


AKT_KW = dict(
    n_question=7,
    n_pid=10,
    d_model=8,
    n_blocks=1,
    dropout=0.0,
    d_ff=16,
    kq_same=1,
    final_fc_dim=16,
    num_attn_heads=2,
    emb_type="qid",
)

Q = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 0, 1, 2, 3, 4]])
R = torch.tensor([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]])
PID = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 0, 1]])
GAPS = {
    "rgaps": torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long),
    "sgaps": torch.tensor([[0, 0, 1, 1, 2, 2], [1, 1, 0, 0, 2, 2]], dtype=torch.long),
    "pcounts": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long),
}
NON_BASELINE = tuple(mode for mode in MECHANISMS if mode != "baseline")


def make(mechanism=None):
    """Fresh model; the same seed gives identical inherited AKT weights."""
    torch.manual_seed(42)
    if mechanism is None:
        return AKT(**AKT_KW)
    return AKTFairness(mechanism=mechanism, scale=0.1, **AKT_KW)


def invoke(model, q=Q, r=R, pid=PID):
    gaps = GAPS if model.mechanism == "mtkt_time" else None
    return model(q, r, pid, gaps=gaps)


def copy_shared_akt_weights(dst, src):
    """Load only the inherited AKT tensors into a mechanism model."""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    shared = {key: value for key, value in src_state.items() if key in dst_state}
    missing, unexpected = dst.load_state_dict(shared, strict=False)
    assert not unexpected
    assert all(key in dst_state for key in shared)
    return dst


def test_baseline_is_bit_identical_to_original_akt():
    base = make().eval()
    fair = make("baseline").eval()
    assert isinstance(fair, AKT)
    assert fair.state_dict().keys() == base.state_dict().keys()

    with torch.no_grad():
        base_two = base(Q, R, PID)
        fair_two = fair(Q, R, PID)
        assert torch.equal(base_two[0], fair_two[0])
        assert torch.equal(base_two[1], fair_two[1])

        base_three = base(Q, R, PID, qtest=True)
        fair_three = fair(Q, R, PID, qtest=True)
        assert len(base_three) == len(fair_three) == 3
        for left, right in zip(base_three, fair_three):
            assert torch.equal(left, right)
        assert fair_three[2].shape == (Q.size(0), Q.size(1), 2 * AKT_KW["d_model"])


def test_zero_residual_mechanisms_match_akt_at_initialisation():
    base = make().eval()
    with torch.no_grad():
        expected = base(Q, R, PID)[0]
        expected_concat = base(Q, R, PID, qtest=True)[2]

    for mode in ("ukt_cov", "mtkt_cic", "mtkt_time"):
        model = make(mode).eval()
        copy_shared_akt_weights(model, base)
        with torch.no_grad():
            preds, reg = invoke(model)
            preds_qtest, reg_qtest, concat_q = model(
                Q, R, PID, qtest=True, gaps=GAPS if mode == "mtkt_time" else None
            )
        assert torch.equal(preds, expected), mode
        assert torch.equal(preds_qtest, expected), mode
        assert torch.equal(concat_q, expected_concat), mode
        assert torch.isfinite(reg).all(), mode


def test_lef_additive_keeps_parameters_and_starts_identical():
    base = make().eval()
    model = make("lef_additive").eval()
    copy_shared_akt_weights(model, base)
    model.reset_initial_buffers()

    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in base.parameters()
    )
    assert model.q_diff_origin.shape == model.q_embed_diff.weight.shape
    assert model.qa_diff_origin.shape == model.qa_embed_diff.weight.shape
    assert not model.q_diff_origin.requires_grad
    assert not model.qa_diff_origin.requires_grad
    assert torch.equal(model.q_diff_origin, model.q_embed_diff.weight.detach())
    assert torch.equal(model.qa_diff_origin, model.qa_embed_diff.weight.detach())

    with torch.no_grad():
        # AKT's difficulty embedding starts at zero, so both forms have a
        # zero PID modulation at construction even for non-zero problem ids.
        assert torch.equal(base(Q, R, PID)[0], model(Q, R, PID)[0])

        # Removing the multiplication changes behaviour once difficulty/skill
        # directions are non-zero: that is expected for the additive transfer.
        model.difficult_param.weight.fill_(0.3)
        assert not torch.equal(model(Q, R, PID)[0], base(Q, R, PID)[0])


@pytest.mark.parametrize("mode", MECHANISMS)
def test_response_at_current_and_future_positions_cannot_change_prefix(mode):
    model = make(mode).eval()
    # Activate residuals: zero initial output would conceal a query-label leak.
    with torch.no_grad():
        if mode == 'ukt_cov': model.ukt_cov.out.weight.normal_(0, .1)
        if mode == 'mtkt_cic': model.cic.conv3.weight.normal_(0, .1)
        if mode == 'mtkt_time': model.time_residual.net[-1].weight.normal_(0, .1)
    with torch.no_grad():
        preds = invoke(model)[0]

    length = R.size(1)
    for start in range(length):
        flipped = R.clone()
        flipped[:, start:] = 1 - flipped[:, start:]
        with torch.no_grad():
            changed = invoke(model, r=flipped)[0]
        # Prediction at ``start`` and all earlier positions must ignore the
        # response at ``start`` and every future response.
        assert torch.equal(preds[:, : start + 1], changed[:, : start + 1]), (
            mode,
            start,
        )


@pytest.mark.parametrize("mode", NON_BASELINE)
def test_nonbaseline_finite_backward_and_learnable(mode):
    base = make().eval()
    model = make(mode)
    copy_shared_akt_weights(model, base)
    if mode == "lef_additive":
        model.reset_initial_buffers()

    model.eval()
    with torch.no_grad():
        before = invoke(model)[0].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(3):
        model.train()
        optimizer.zero_grad()
        preds, reg = invoke(model)
        loss = F.binary_cross_entropy(
            preds.clamp(1e-6, 1.0 - 1e-6), R.float()
        ) + reg
        assert torch.isfinite(loss), mode
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, mode
        assert all(torch.isfinite(grad).all() for grad in grads), mode
        assert any(grad.abs().sum() > 0 for grad in grads), mode
        optimizer.step()

    model.eval()
    with torch.no_grad():
        after = invoke(model)[0]
    assert not torch.allclose(before, after, atol=1e-7, rtol=0), mode


@pytest.mark.parametrize("mode", MECHANISMS)
def test_state_dict_restore_round_trip(mode):
    base = make().eval()
    model = make(mode)
    copy_shared_akt_weights(model, base)
    if mode == "lef_additive":
        model.reset_initial_buffers()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    optimizer.zero_grad()
    preds, reg = invoke(model)
    (F.binary_cross_entropy(preds.clamp(1e-6, 1.0 - 1e-6), R.float()) + reg).backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        expected = invoke(model)[0]

    restored = make(mode)
    missing, unexpected = restored.load_state_dict(model.state_dict())
    assert not missing and not unexpected
    restored.eval()
    with torch.no_grad():
        actual = invoke(restored)[0]
    assert torch.equal(expected, actual), mode


def test_time_mechanism_validates_full_length_integer_gaps():
    model = make("mtkt_time").eval()
    with pytest.raises(ValueError):
        model(Q, R, PID)
    with pytest.raises(ValueError):
        model(Q, R, PID, gaps={"rgaps": GAPS["rgaps"]})
    with pytest.raises(ValueError):
        model(Q, R, PID, gaps=dict(GAPS, rgaps=GAPS["rgaps"].float()))
    with pytest.raises(ValueError):
        model(Q, R, PID, gaps=dict(GAPS, sgaps=GAPS["sgaps"][:, :3]))
    with pytest.raises(ValueError):
        model(Q, R, PID, gaps=dict(GAPS, pcounts=-GAPS["pcounts"]))


def test_real_cic_is_causal_convolution_and_finite():
    module = MTKTCausalInteraction(8, hidden=8, kernel1=1, kernel2=3, dropout=0.0)
    assert isinstance(module.conv1, torch.nn.Conv1d)
    assert isinstance(module.conv2, torch.nn.Conv1d)
    with torch.no_grad():
        module.conv3.weight.normal_()
        module.conv3.bias.normal_()
    x = torch.randn(2, 6, 8)
    out = module(x)
    changed = x.clone()
    changed[:, 3:] += 10.0
    with torch.no_grad():
        changed_out = module(changed)
    assert torch.equal(out[:, :3], changed_out[:, :3])
    out.square().sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()
    )


def test_ukt_covariance_residual_stays_finite_on_large_inputs():
    module = UKTCovarianceResidual(8, scale=0.1)
    with torch.no_grad():
        module.out.weight.normal_()
        module.out.bias.normal_()
    x = torch.randn(2, 6, 8) * 1000.0
    out = module(x)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()
    )
