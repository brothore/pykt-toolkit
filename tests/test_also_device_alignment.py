import pytest
import torch

from ALSO.also import ALSO


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this regression test")
def test_also_step_aligns_pi_to_model_device():
    model = torch.nn.Linear(4, 1).cuda()
    optimizer = ALSO(
        params=model.parameters(),
        n_groups=2,
        batch_size=2,
        loss_scale=1.0,
        mode="optimistic",
        lr=1e-3,
    )

    def closure(pi_selected, scale):
        losses = torch.ones(pi_selected.shape[0], device=model.weight.device) * scale
        return losses, losses.mean().item()

    groups_indexes = torch.tensor([0, 1], device=model.weight.device, dtype=torch.long)

    loss = optimizer.step(closure=closure, groups_indexes=groups_indexes)

    assert isinstance(loss, float)
    assert optimizer.pi.device == model.weight.device
