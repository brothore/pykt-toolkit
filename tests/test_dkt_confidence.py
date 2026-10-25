import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pykt.datasets.confidence_dataset import ConfidenceDataset, build_confidence_sidecar, confidence_labels
from pykt.models.dkt import DKT
from pykt.models.dkt_confidence import DKTConfidence
from pykt.models.train_model import model_forward


@pytest.fixture
def batch():
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    return {'cseqs': torch.tensor([[0, 1, 0], [1, 0, 0]]),
            'rseqs': torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
            'shft_cseqs': torch.tensor([[1, 0, 1], [0, 1, 0]]),
            'shft_rseqs': torch.tensor([[0., 1., 0.], [1., 0., 0.]]),
            'masks': mask.clone(), 'smasks': mask.clone(),
            'confidence_labels': torch.tensor([[0, 4, 2], [1, -100, -100]]),
            'confidence_mask': mask.clone()}


@pytest.mark.parametrize('mode', ['categorical', 'ordinal', 'conditional'])
def test_zero_aux_preserves_rng_forward_and_adam_step(batch, mode):
    torch.manual_seed(19)
    old = DKT(2, 8, .2)
    rng_init = torch.random.get_rng_state()
    torch.manual_seed(19)
    new = DKTConfidence(2, 8, .2, confidence_weight=0, confidence_mode=mode)
    assert torch.equal(rng_init, torch.random.get_rng_state())
    rng = torch.random.get_rng_state()
    a, b = torch.optim.Adam(old.parameters()), torch.optim.Adam(new.parameters())
    selected = old(batch['cseqs'], batch['rseqs'].long()).gather(-1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss = F.binary_cross_entropy(selected[batch['smasks']].double(), batch['shft_rseqs'][batch['smasks']].double())
    loss.backward(); a.step()
    torch.random.set_rng_state(rng)
    new_loss = new.compute_loss({k: v for k, v in batch.items() if not k.startswith('confidence_')})
    assert torch.equal(loss, new_loss)
    new_loss.backward(); b.step()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name


@pytest.mark.parametrize('mode', ['categorical', 'ordinal', 'conditional'])
def test_gradients_empty_masks_and_roundtrip(batch, mode, tmp_path):
    model = DKTConfidence(2, 8, dropout=0, confidence_mode=mode)
    loss = model_forward(model, batch)
    loss.backward()
    assert model.confidence_out.weight.grad.abs().sum() > 0
    assert model.lstm_layer.weight_ih_l0.grad.abs().sum() > 0
    path = tmp_path / 'checkpoint'
    torch.save(model.state_dict(), path)
    clone = DKTConfidence(2, 8, dropout=0, confidence_mode=mode)
    clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(loss, clone.compute_loss(batch))
    old = DKT(2, 8, dropout=0)
    old.load_state_dict(model.export_dkt_state_dict(), strict=True)
    assert torch.equal(old(batch['cseqs'], batch['rseqs'].long()), model(batch['cseqs'], batch['rseqs'].long()))
    batch['confidence_mask'].fill_(False)
    _, parts = model.compute_loss(batch, return_components=True)
    assert parts['confidence'] == 0
    batch['smasks'].fill_(False)
    assert model.compute_loss(batch) == 0
    assert torch.equal(model.compute_loss(batch, return_per_sample=True), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize('mode', ['categorical', 'ordinal', 'conditional'])
def test_no_future_target_leakage_and_normalized_joint(batch, mode):
    model = DKTConfidence(2, 8, dropout=0, confidence_mode=mode).eval()
    before = model.confidence_probabilities(batch)
    changed = copy.deepcopy(batch)
    changed['shft_rseqs'] = 1 - changed['shft_rseqs']
    changed['confidence_labels'].fill_(4)
    changed['confidence_mask'].fill_(False)
    after = model.confidence_probabilities(changed)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert torch.allclose(before['confidence_probs'].sum(-1), torch.ones(2, 3))
    assert torch.allclose(before['joint_probs'].sum((-1, -2)), torch.ones(2, 3))
    p = model(batch['cseqs'], batch['rseqs'].long()).gather(-1, batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(before['joint_probs'][..., 1, :].sum(-1), p)
    changed['rseqs'][:, -1] = 1 - changed['rseqs'][:, -1]
    after = model.confidence_probabilities(changed)
    assert all(torch.equal(before[k][:, :-1], after[k][:, :-1]) for k in before)


def test_ordinal_loss_penalizes_distance():
    model = DKTConfidence(2, 8, confidence_mode='ordinal')
    label = torch.tensor([[0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    near = torch.tensor([[[0., 5., 0., 0., 0.]]])
    far = torch.tensor([[[0., 0., 0., 0., 5.]]])
    assert model._confidence_loss(near, label, label, mask, False) < model._confidence_loss(far, label, label, mask, False)


@pytest.mark.parametrize('value', [1.5, float('nan'), 9])
def test_invalid_active_labels_raise_but_masked_nan_is_safe(batch, value):
    model = DKTConfidence(2, 8, dropout=0)
    batch['confidence_labels'] = batch['confidence_labels'].float()
    batch['confidence_labels'][0, 0] = value
    with pytest.raises(ValueError, match='confidence_labels'):
        model.compute_loss(batch)
    batch['confidence_mask'][0, 0] = False
    assert torch.isfinite(model.compute_loss(batch))


def _fixture(directory):
    metadata = directory / 'metadata'; metadata.mkdir()
    mapping = {'uid': {'10': 0, '20': 1}, 'questions': {'100': 0, '101': 1, '102': 2},
               'concepts': {'11': 0, '12': 1}}
    (directory / 'keyid2idx.json').write_text(json.dumps(mapping))
    pd.DataFrame({'SubjectId': [11, 12, 99], 'Level': [3, 3, 2]}).to_csv(metadata / 'subject_metadata.csv', index=False)
    pd.DataFrame({'QuestionId': [100, 101, 102], 'SubjectId': ['[11,99]', '[12]', '[11,12]']}).to_csv(metadata / 'question_metadata_task_3_4.csv', index=False)
    rows, answers = [], []
    for i, (uid, q, r, confidence) in enumerate([(10, 100, 1, 0), (10, 101, 0, 25), (10, 102, 0, 100),
                                                (10, 101, 1, np.nan), (10, 100, 1, 75),
                                                (20, 100, 1, 100), (20, 101, 0, 50), (20, 100, 0, np.nan)]):
        rows.append({'UserId': uid, 'QuestionId': q, 'IsCorrect': r, 'AnswerId': 1000+i})
        answers.append({'AnswerId': 1000+i, 'DateAnswered': f'2020-01-01 00:{max(0,i-1):02d}:00', 'Confidence': confidence})
    pd.DataFrame(rows).to_csv(directory / 'train_task_3_4.csv', index=False)
    pd.DataFrame(answers).to_csv(metadata / 'answer_metadata_task_3_4.csv', index=False)
    ts = pd.to_datetime(pd.DataFrame(answers).DateAnswered).dt.as_unit('ms').astype('int64').tolist()
    originals = [dict(uid=0, fold=1, questions='0,1,2,2,1,0', concepts='0,1,0,1,1,0', responses='1,0,0,0,1,1',
                      timestamps=','.join(map(str, [ts[0],ts[1],ts[2],ts[2],ts[3],ts[4]])), is_repeat='0,0,0,1,0,0'),
                 dict(uid=1, fold=0, questions='0,1,0', concepts='0,1,0', responses='1,0,0',
                      timestamps=','.join(map(str,ts[5:])), is_repeat='0,0,0')]
    pd.DataFrame(originals).to_csv(directory / 'train_valid.csv', index=False)
    chunks = []
    for row in originals:
        for start in range(0,len(row['responses'].split(',')),3):
            chunk = {'uid':row['uid'],'fold':row['fold'],'selectmasks':'1,1,1'}
            chunk.update({key:','.join(row[key].split(',')[start:start+3])
                          for key in ['questions','concepts','responses','timestamps','is_repeat']})
            chunks.append(chunk)
    pd.DataFrame(chunks).to_csv(directory / 'train_valid_sequences.csv', index=False)


def test_label_mapping_keeps_zero_and_masks_missing():
    assert confidence_labels([0,25,50,75,100,np.nan,-1,5]).tolist() == [0,1,2,3,4,-100,-100,-100]


def test_alignment_ties_retries_copies_counts_and_folds(tmp_path):
    _fixture(tmp_path)
    source = tmp_path / 'train_valid_sequences.csv'
    train = ConfidenceDataset(source,['questions','concepts'],{1})
    valid = ConfidenceDataset(source,['questions','concepts'],{0})
    assert train.labels.tolist() == [[1,4],[-100,3]]
    assert train.confidence_counts == [0,1,0,1,1]
    assert train.mask.tolist() == [[True,True],[False,True]]
    assert set(train.dori['uid']).isdisjoint(valid.dori['uid'])
    with np.load(tmp_path / 'dkt_confidence/targets_v1.npz') as data:
        assert not data['mask'][1,0]
        assert data['events'][0,0] != data['events'][1,-1]  # real repeated question
        assert data['labels'][0,0] == 0  # valid confidence zero, chunk-start not a target


def test_changed_metadata_invalidates_cache_and_tie_order_is_verified(tmp_path):
    _fixture(tmp_path)
    path = build_confidence_sidecar(tmp_path)
    source = tmp_path / 'metadata/answer_metadata_task_3_4.csv'
    raw = pd.read_csv(source); raw.loc[1,'Confidence']=100; raw.to_csv(source,index=False)
    build_confidence_sidecar(tmp_path)
    with np.load(path) as data:
        assert data['labels'][0,1] == 4
    # First two events have tied times; swapping primary row order must fail
    # against existing sequence rather than silently assigning labels.
    primary = tmp_path / 'train_task_3_4.csv'
    raw = pd.read_csv(primary); raw.iloc[[1,0]+list(range(2,len(raw)))].to_csv(primary,index=False)
    with pytest.raises(ValueError,match='complete confidence sequence mismatch'):
        build_confidence_sidecar(tmp_path)


def test_class_weights_only_use_supplied_training_counts():
    model = DKTConfidence(2,8,confidence_balance=.5,confidence_counts=[1,4,9,16,25])
    expected = torch.tensor([1.,4.,9.,16.,25.]).rsqrt()
    assert torch.allclose(model.confidence_class_weights,expected/expected.mean())


def test_initializer_routes_only_new_name(tmp_path):
    from pykt.datasets.init_dataset import init_dataset4train
    _fixture(tmp_path)
    config={'nips_task34':{'dpath':str(tmp_path),'train_valid_file':'train_valid_sequences.csv',
                          'input_type':['questions','concepts'],'folds':[0,1]}}
    new,_,*_=init_dataset4train('nips_task34','dkt_confidence',config,0,2)
    old,_,*_=init_dataset4train('nips_task34','dkt',config,0,2)
    assert isinstance(new.dataset,ConfidenceDataset)
    assert not isinstance(old.dataset,ConfidenceDataset)
    assert torch.equal(next(iter(new))['rseqs'],next(iter(old))['rseqs'])
