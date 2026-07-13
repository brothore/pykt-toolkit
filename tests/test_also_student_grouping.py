import pytest
import torch

from pykt.models.train_model import _build_student_group_metadata, _prepare_also_group_ids


class _Dataset:
    def __init__(self, uids):
        self.dori = {"uid": uids}


class _Loader:
    def __init__(self, uids):
        self.dataset = _Dataset(uids)


def test_student_groups_are_dense_and_injective_for_sparse_uids():
    mapping, sequence_counts = _build_student_group_metadata(_Loader([3675, 0, 3851, 3675]))

    assert mapping == {0: 0, 3675: 1, 3851: 2}
    assert torch.equal(sequence_counts, torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64))

    group_ids, n_groups = _prepare_also_group_ids(
        {"uid": torch.tensor([3851, 3675, 0])}, "student_id", torch.device("cpu"), mapping
    )
    assert n_groups == 3
    assert torch.equal(group_ids, torch.tensor([2, 1, 0]))


def test_student_group_count_must_match_training_fold():
    with pytest.raises(ValueError, match="expected also_n_groups=3"):
        _build_student_group_metadata(_Loader([0, 3675, 3851]), configured_n_groups=3082)
