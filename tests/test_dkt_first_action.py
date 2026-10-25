import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pykt.datasets.first_action_dataset import FirstActionDataset, build_first_action_sidecar
from pykt.models.dkt import DKT
from pykt.models.dkt_first_action import DKTFirstAction
from pykt.models.train_model import cal_loss, model_forward


@pytest.fixture
def batch():
    return {'cseqs': torch.tensor([[0, 1, 0], [1, 0, 0]]),
            'rseqs': torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
            'shft_cseqs': torch.tensor([[1, 0, 1], [0, 1, 0]]),
            'shft_rseqs': torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
            'masks': torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            'smasks': torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            'action_labels': torch.tensor([[1, 0, 2], [0, 3, -100]]),
            'action_mask': torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)}


def test_zero_weight_matches_original_dkt_and_preserves_rng(batch):
    torch.manual_seed(41)
    baseline = DKT(2, 8, dropout=.2)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(41)
    new = DKTFirstAction(2, 8, dropout=.2, action_weight=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, new.state_dict()[name])
    baseline.eval()
    new.eval()
    expected = baseline(batch['cseqs'], batch['rseqs'].long())
    assert torch.equal(expected, new(batch['cseqs'], batch['rseqs'].long()))
    selected = expected.gather(-1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    old_loss = cal_loss(baseline, [selected], batch['rseqs'], batch['shft_rseqs'], batch['smasks'])
    new_loss = new.compute_loss({k: v for k, v in batch.items() if not k.startswith('action_')})
    assert torch.equal(old_loss, new_loss)


def test_zero_weight_training_step_matches_dkt_with_dropout(batch):
    torch.manual_seed(71)
    baseline = DKT(2, 8, dropout=.2).train()
    torch.manual_seed(71)
    new = DKTFirstAction(2, 8, dropout=.2, action_weight=0).train()
    old_opt = torch.optim.Adam(baseline.parameters(), lr=.001)
    new_opt = torch.optim.Adam(new.parameters(), lr=.001)
    rng = torch.random.get_rng_state()
    predicted = baseline(batch['cseqs'], batch['rseqs'].long()).gather(
        -1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss = cal_loss(baseline, [predicted], batch['rseqs'], batch['shft_rseqs'], batch['smasks'])
    loss.backward()
    old_opt.step()
    torch.random.set_rng_state(rng)
    new.compute_loss(batch).backward()
    new_opt.step()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name


@pytest.mark.parametrize('mode', ['joint', 'conditional'])
def test_auxiliary_gradients_and_checkpoint_roundtrip(batch, mode, tmp_path):
    model = DKTFirstAction(2, 8, dropout=0, action_mode=mode, action_balance=.5,
                           action_class_counts=[10, 8, 4, 2])
    loss = model_forward(model, batch)
    loss.backward()
    assert model.lstm_layer.weight_ih_l0.grad.abs().sum() > 0
    assert model.action_out.weight.grad.abs().sum() > 0
    assert torch.isfinite(loss)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    clone = DKTFirstAction(2, 8, dropout=0, action_mode=mode)
    clone.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(model.action_probabilities(batch), clone.action_probabilities(batch))


@pytest.mark.parametrize('mode', ['joint', 'conditional'])
def test_inference_never_reads_target_answers_or_actions(batch, mode):
    model = DKTFirstAction(2, 8, dropout=0, action_mode=mode).eval()
    before = model.action_probabilities(batch)
    altered = copy.deepcopy(batch)
    altered['shft_rseqs'] = 1 - altered['shft_rseqs']
    altered['action_labels'].fill_(3)
    altered['action_mask'].fill_(False)
    assert torch.equal(before, model.action_probabilities(altered))
    assert torch.allclose(before.sum(-1), torch.ones_like(before[..., 0]))
    if mode == 'conditional':
        response = model(batch['cseqs'], batch['rseqs'].long()).gather(
            -1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
        assert torch.equal(before[..., 0], response)


def test_future_history_changes_cannot_change_earlier_predictions(batch):
    model = DKTFirstAction(2, 8, dropout=0).eval()
    before = model(batch['cseqs'], batch['rseqs'].long())
    changed = batch['rseqs'].long().clone()
    changed[:, 2] = 1 - changed[:, 2]
    assert torch.equal(before[:, :2], model(batch['cseqs'], changed)[:, :2])


def test_exported_backbone_loads_in_unmodified_dkt(batch):
    model = DKTFirstAction(2, 8, dropout=0).eval()
    baseline = DKT(2, 8, dropout=0).eval()
    baseline.load_state_dict(model.export_dkt_state_dict(), strict=True)
    assert torch.equal(model(batch['cseqs'], batch['rseqs'].long()),
                       baseline(batch['cseqs'], batch['rseqs'].long()))


@pytest.mark.parametrize('mode', ['joint', 'conditional'])
def test_empty_masks_return_differentiable_zero(batch, mode):
    model = DKTFirstAction(2, 8, dropout=0, action_mode=mode)
    batch['smasks'].fill_(False)
    loss = model.compute_loss(batch)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(model.compute_loss(batch, return_per_sample=True), torch.zeros(2, dtype=torch.float64))


def test_unknown_action_is_ignored_even_if_external_mask_is_true():
    model = DKTFirstAction(2, 8, dropout=0)
    logits = torch.randn(1, 2, 4, requires_grad=True)
    loss = model._action_loss(logits, torch.tensor([[1, -100]]), torch.ones(1, 2, dtype=torch.bool), False)
    assert torch.allclose(loss, F.cross_entropy(logits[:, :1].reshape(1, 4), torch.tensor([1])))
    with pytest.raises(ValueError):
        model._action_loss(logits, torch.tensor([[1, 8]]), torch.ones(1, 2, dtype=torch.bool), False)


def _fixture_data(directory):
    mapping = {'uid': {'10': 0, '20': 1}, 'questions': {'100': 0, '101': 1, '102': 2},
               'concepts': {'11': 0, '12': 1}}
    (directory / 'keyid2idx.json').write_text(json.dumps(mapping))
    raw = []
    # Event 3 crosses a chunk boundary; event 5 repeats question 100 as a real new attempt.
    for i, (uid, q, c, r, a) in enumerate([
        (10, 100, '11', 1, 0), (10, 101, '12', 0, 0), (10, 102, '11_12', 0, 1),
        (10, 101, '11', 0, 2), (10, 100, '12', 1, 0),
        (20, 100, '11', 1, 0), (20, 101, '12', 0, 2), (20, 102, '11', 0, 1)]):
        raw.append({'order_id': 10000000+i, 'user_id': uid, 'problem_id': q,
                    'skill_id': c, 'correct': r, 'first_action': a})
    pd.DataFrame(raw).to_csv(directory / 'skill_builder_data_corrected_collapsed.csv', index=False)
    originals = [
        {'uid': 0, 'fold': 1, 'questions': '0,1,2,2,1,0', 'concepts': '0,1,0,1,0,1',
         'responses': '1,0,0,0,0,1', 'is_repeat': '0,0,0,1,0,0'},
        {'uid': 1, 'fold': 0, 'questions': '0,1,2', 'concepts': '0,1,0',
         'responses': '1,0,0', 'is_repeat': '0,0,0'}]
    pd.DataFrame(originals).to_csv(directory / 'train_valid.csv', index=False)
    seq = []
    for row in originals:
        for start in range(0, len(row['responses'].split(',')), 3):
            segment = {'fold': row['fold'], 'uid': row['uid'], 'selectmasks': '1,1,1'}
            segment.update({key: ','.join(row[key].split(',')[start:start+3])
                            for key in ['questions', 'concepts', 'responses', 'is_repeat']})
            seq.append(segment)
    pd.DataFrame(seq).to_csv(directory / 'train_valid_sequences.csv', index=False)


def test_event_alignment_masks_copies_and_respects_folds(tmp_path):
    _fixture_data(tmp_path)
    dataset = FirstActionDataset(tmp_path / 'train_valid_sequences.csv', ['questions', 'concepts'], {1})
    assert len(dataset) == 2
    assert dataset.action_labels.tolist() == [[1, 2], [3, 0]]
    assert dataset.action_class_counts == [1, 1, 1, 1]
    with np.load(tmp_path / 'dkt_first_action/targets_v1.npz') as sidecar:
        assert not sidecar['mask'][1, 0]  # copied concept at the next chunk start
    validation = FirstActionDataset(tmp_path / 'train_valid_sequences.csv', ['questions', 'concepts'], {0})
    assert validation.action_class_counts == [0, 0, 1, 1]
    assert set(dataset.dori['uid']).isdisjoint(validation.dori['uid'])


def test_balancing_uses_only_supplied_training_counts():
    model = DKTFirstAction(2, 8, action_mode='conditional', action_balance=.5,
                           action_class_counts=[10000, 100, 25, 4])
    expected = torch.tensor([100., 25., 4.]).rsqrt()
    assert torch.allclose(model.action_class_weights, expected / expected.mean())


def test_train_initializer_routes_only_new_model_to_sidecar(tmp_path):
    from pykt.datasets.init_dataset import init_dataset4train
    _fixture_data(tmp_path)
    config = {'assist2009': {'dpath': str(tmp_path), 'train_valid_file': 'train_valid_sequences.csv',
                             'input_type': ['questions', 'concepts'], 'folds': [0, 1]}}
    train, valid, *_ = init_dataset4train('assist2009', 'dkt_first_action', config, 0, 2)
    old_train, old_valid, *_ = init_dataset4train('assist2009', 'dkt', config, 0, 2)
    assert isinstance(train.dataset, FirstActionDataset)
    assert not isinstance(old_train.dataset, FirstActionDataset)
    assert torch.equal(next(iter(train))['cseqs'], next(iter(old_train))['cseqs'])


def test_sidecar_detects_misalignment_and_rebuilds_changed_raw_labels(tmp_path):
    _fixture_data(tmp_path)
    sidecar = build_first_action_sidecar(tmp_path)
    raw_path = tmp_path / 'skill_builder_data_corrected_collapsed.csv'
    raw = pd.read_csv(raw_path)
    raw.loc[1, 'first_action'] = 2
    raw.to_csv(raw_path, index=False)
    build_first_action_sidecar(tmp_path)
    with np.load(sidecar) as data:
        assert data['labels'][0, 1] == 3
    seq_path = tmp_path / 'train_valid_sequences.csv'
    seq = pd.read_csv(seq_path)
    seq.loc[0, 'questions'] = '1,1,2'
    seq.to_csv(seq_path, index=False)
    with pytest.raises(ValueError, match='chunk mismatch'):
        build_first_action_sidecar(tmp_path)


def test_stale_legacy_cache_is_rejected_without_modifying_it(tmp_path):
    _fixture_data(tmp_path)
    source = tmp_path / 'train_valid_sequences.csv'
    FirstActionDataset(source, ['questions', 'concepts'], {1})
    cache_path = tmp_path / 'train_valid_sequences.csv_1.pkl'
    cached = pd.read_pickle(cache_path)
    cached['cseqs'][0, 0] = 1
    pd.to_pickle(cached, cache_path)
    before = cache_path.read_bytes()
    with pytest.raises(ValueError, match='Stale KTDataset cache'):
        FirstActionDataset(source, ['questions', 'concepts'], {1})
    assert cache_path.read_bytes() == before
