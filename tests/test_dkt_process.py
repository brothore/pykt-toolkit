import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pykt.datasets.process_dataset import ProcessDataset, process_targets
from pykt.models.dkt import DKT
from pykt.models.dkt_process import DKTProcess
from test_dkt_first_action import _fixture_data


@pytest.fixture
def batch():
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    return {'cseqs': torch.tensor([[0, 1, 0], [1, 0, 0]]),
            'rseqs': torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
            'shft_cseqs': torch.tensor([[1, 0, 1], [0, 1, 0]]),
            'shft_rseqs': torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
            'masks': mask.clone(), 'smasks': mask.clone(),
            'process_time': torch.tensor([[2., 3., 4.], [1., 2., float('nan')]]),
            'process_attempt': torch.tensor([[2, 1, 3], [1, 0, -100]]),
            'process_hint': torch.tensor([[1., 0., 0.], [0., 1., float('nan')]]),
            **{f'process_{key}_mask': mask.clone() for key in ['time', 'attempt', 'hint']}}


def test_zero_aux_matches_dkt_rng_forward_and_adam_update(batch):
    torch.manual_seed(77)
    old = DKT(2, 8, .2)
    rng_after_init = torch.random.get_rng_state()
    torch.manual_seed(77)
    new = DKTProcess(2, 8, .2, time_weight=0, attempt_weight=0, hint_weight=0)
    assert torch.equal(rng_after_init, torch.random.get_rng_state())
    a, b = torch.optim.Adam(old.parameters()), torch.optim.Adam(new.parameters())
    rng = torch.random.get_rng_state()
    selected = old(batch['cseqs'], batch['rseqs'].long()).gather(-1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss = F.binary_cross_entropy(selected[batch['smasks']].double(), batch['shft_rseqs'][batch['smasks']].double())
    loss.backward(); a.step()
    torch.random.set_rng_state(rng)
    new_loss = new.compute_loss({k: v for k, v in batch.items() if not k.startswith('process_')})
    assert torch.equal(loss, new_loss)
    new_loss.backward(); b.step()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name


@pytest.mark.parametrize('mode', ['joint', 'conditional'])
def test_gradients_masks_roundtrip_and_export(batch, mode, tmp_path):
    model = DKTProcess(2, 8, dropout=0, process_mode=mode)
    loss = model.compute_loss(batch)
    loss.backward()
    for name in ['time_out', 'attempt_out', 'hint_out', 'lstm_layer']:
        assert sum(p.grad.abs().sum() for p in getattr(model, name).parameters()) > 0
    path = tmp_path / 'state.pt'
    torch.save(model.state_dict(), path)
    clone = DKTProcess(2, 8, dropout=0, process_mode=mode)
    clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model.compute_loss(batch), clone.compute_loss(batch))
    old = DKT(2, 8, dropout=0)
    old.load_state_dict(model.export_dkt_state_dict(), strict=True)
    assert torch.equal(old(batch['cseqs'], batch['rseqs'].long()), model(batch['cseqs'], batch['rseqs'].long()))
    for key in ['time', 'attempt', 'hint']:
        batch[f'process_{key}_mask'].fill_(False)
    _, parts = model.compute_loss(batch, return_components=True)
    assert all(parts[key] == 0 for key in ['time', 'attempt', 'hint'])
    batch['smasks'].fill_(False)
    assert model.compute_loss(batch) == 0
    assert torch.equal(model.compute_loss(batch, return_per_sample=True), torch.zeros(2, dtype=torch.float64))


def test_predictions_ignore_future_response_and_process_targets(batch):
    model = DKTProcess(2, 8, dropout=0).eval()
    before = model.process_predictions(batch)
    changed = copy.deepcopy(batch)
    changed['shft_rseqs'] = 1 - changed['shft_rseqs']
    for key in changed:
        if key.startswith('process_'):
            changed[key].fill_(0)
    after = model.process_predictions(changed)
    assert all(torch.equal(before[key], after[key]) for key in before)
    changed['rseqs'][:, -1] = 1 - changed['rseqs'][:, -1]
    after = model.process_predictions(changed)
    assert all(torch.equal(before[key][:, :-1], after[key][:, :-1]) for key in before)


def test_conditional_losses_exclude_correct_responses(batch):
    model = DKTProcess(2, 8, dropout=0, process_mode='conditional')
    _, before = model.compute_loss(batch, return_components=True)
    correct = batch['shft_rseqs'] == 1
    batch['process_attempt'][correct] = 3
    batch['process_hint'][correct] = 1
    _, after = model.compute_loss(batch, return_components=True)
    assert torch.equal(before['attempt'], after['attempt'])
    assert torch.equal(before['hint'], after['hint'])


def test_process_label_cleaning_masks_and_valid_zero():
    raw = pd.DataFrame({'ms_first_response': [-1, 0, 1000, 4000000, 2000, np.nan],
                        'attempt_count': [0, 1, 2, 3824, -1, np.nan],
                        'hint_count': [0, 0, 1, 0, 5, 0], 'hint_total': [0, 3, 3, 3, 3, 3],
                        'first_action': [0, 0, 1, 2, 0, 0]})
    fields = process_targets(raw)
    assert fields['process_time_mask'].tolist() == [False, False, True, False, True, False]
    assert fields['process_attempt'].tolist()[:4] == [0, 1, 2, 3]
    assert fields['process_attempt_mask'].tolist() == [True, True, True, True, False, False]
    assert fields['process_hint_mask'].tolist() == [False, True, True, False, False, True]


def _process_fixture(tmp_path):
    _fixture_data(tmp_path)
    source = tmp_path / 'skill_builder_data_corrected_collapsed.csv'
    raw = pd.read_csv(source)
    raw['ms_first_response'] = [1000, 2000, 3000, 4000, 5000, 600000, 700000, 800000]
    raw['attempt_count'] = [1, 2, 0, 3, 1, 100, 100, 100]
    raw['hint_count'] = [0, 0, 1, 0, 0, 1, 1, 1]
    raw['hint_total'] = 3
    raw.to_csv(source, index=False)
    return source


def test_event_alignment_stats_and_fold_isolation(tmp_path):
    _process_fixture(tmp_path)
    path = tmp_path / 'train_valid_sequences.csv'
    train = ProcessDataset(path, ['questions', 'concepts'], {1})
    valid = ProcessDataset(path, ['questions', 'concepts'], {0}, mode='valid', process_stats=train.process_stats)
    assert train.fields['process_attempt'].tolist() == [[2, 0], [3, 1]]
    assert train.fields['process_attempt_mask'].all()
    assert train.process_stats['attempt_counts'] == [1, 1, 1, 1]
    assert train.process_stats == valid.process_stats
    assert set(train.dori['uid']).isdisjoint(valid.dori['uid'])
    assert not any(key.startswith('action_') for key in train[0])
    # Event spanning chunks is copied at next chunk's first position, which
    # cannot reappear as an auxiliary target.
    assert train.process_stats['time_count'] == 4
    before = copy.deepcopy(train.process_stats)
    source = tmp_path / 'skill_builder_data_corrected_collapsed.csv'
    raw = pd.read_csv(source)
    raw.loc[raw.user_id == 20, 'ms_first_response'] = 999999
    raw.to_csv(source, index=False)
    rebuilt = ProcessDataset(path, ['questions', 'concepts'], {1})
    assert rebuilt.process_stats == before
    with pytest.raises(ValueError, match='training-fold'):
        ProcessDataset(path, ['questions', 'concepts'], {0}, mode='valid')


def test_supplied_counts_define_weights_and_normalization():
    model = DKTProcess(2, 8, process_balance=.5, process_stats={
        'time_center': 3, 'time_scale': 2, 'attempt_counts': [4, 9, 16, 25], 'hint_counts': [4, 16]})
    expected = torch.tensor([4., 9., 16., 25.]).rsqrt()
    assert torch.allclose(model.attempt_class_weights, expected / expected.mean())
    assert model.process_time_center == 3
    assert model.process_time_scale == 2


@pytest.mark.parametrize('field,value', [('time', float('nan')), ('attempt', 1.5), ('hint', 2.)])
def test_invalid_active_targets_rejected(batch, field, value):
    model = DKTProcess(2, 8, dropout=0)
    batch[f'process_{field}'] = batch[f'process_{field}'].float()
    batch[f'process_{field}'][0, 0] = value
    with pytest.raises(ValueError, match=f'process_{field}'):
        model.compute_loss(batch)


def test_initializer_preserves_original_and_previous_variant(tmp_path):
    from pykt.datasets.init_dataset import init_dataset4train
    _process_fixture(tmp_path)
    config = {'assist2009': {'dpath': str(tmp_path), 'train_valid_file': 'train_valid_sequences.csv',
                             'input_type': ['questions', 'concepts'], 'folds': [0, 1]}}
    train, valid, *_ = init_dataset4train('assist2009', 'dkt_process', config, 0, 2)
    old, *_ = init_dataset4train('assist2009', 'dkt', config, 0, 2)
    assert isinstance(train.dataset, ProcessDataset)
    assert train.dataset.process_stats == valid.dataset.process_stats
    assert torch.equal(next(iter(train))['rseqs'], next(iter(old))['rseqs'])
