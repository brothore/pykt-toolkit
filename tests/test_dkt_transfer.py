import copy
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from pykt.models.dkt import DKT
from pykt.models.dkt_transfer import DKTTransfer
from pykt.datasets.transfer_dataset import TransferDataset
from test_dkt_process import _process_fixture


@pytest.fixture
def batch():
    mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    return {'cseqs':torch.tensor([[0,1,0],[1,0,0]]),'rseqs':torch.tensor([[1.,0.,1.],[0.,1.,0.]]),
            'shft_cseqs':torch.tensor([[1,0,1],[0,1,0]]),'shft_rseqs':torch.tensor([[0.,1.,0.],[1.,0.,0.]]),
            'masks':mask.clone(),'smasks':mask.clone(),
            'future_concepts':torch.tensor([[[0,1,-1],[1,-1,-1],[0,-1,-1]],[[1,-1,-1],[0,1,-1],[-1,-1,-1]]]),
            'future_labels':torch.tensor([[1.,0.,1.],[0.,1.,float('nan')]]),'future_mask':mask.clone()}


def test_exact_params_rng_and_zero_weight_adam_parity(batch):
    torch.manual_seed(55);old=DKT(2,8,.2);rng_init=torch.random.get_rng_state()
    torch.manual_seed(55);new=DKTTransfer(2,8,.2,transfer_weight=0)
    assert torch.equal(rng_init,torch.random.get_rng_state())
    assert list(old.state_dict())==list(new.state_dict())
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
    a,b=torch.optim.Adam(old.parameters()),torch.optim.Adam(new.parameters());rng=torch.random.get_rng_state()
    p=old(batch['cseqs'],batch['rseqs'].long()).gather(-1,batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss=F.binary_cross_entropy(p[batch['smasks']].double(),batch['shft_rseqs'][batch['smasks']].double())
    loss.backward();a.step();torch.random.set_rng_state(rng)
    other=new.compute_loss({k:v for k,v in batch.items() if not k.startswith('future_')})
    assert torch.equal(loss,other);other.backward();b.step()
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key])


@pytest.mark.parametrize('loss_type',['bce','brier'])
def test_future_query_uses_current_existing_concept_outputs(batch,loss_type,tmp_path):
    m=DKTTransfer(2,8,0,transfer_loss=loss_type)
    p=m(batch['cseqs'],batch['rseqs'].long())
    queries=p.gather(-1,batch['future_concepts'].clamp_min(0))
    weights=batch['future_concepts']>=0
    means=(queries*weights).sum(-1)/weights.sum(-1).clamp_min(1)
    mask=batch['future_mask'];target=batch['future_labels'][mask].double()
    expected=F.binary_cross_entropy(means[mask].double(),target) if loss_type=='bce' else ((means[mask].double()-target)**2).mean()
    _,parts=m.compute_loss(batch,return_components=True);assert torch.allclose(expected,parts['transfer'])
    parts['transfer'].backward();assert m.lstm_layer.weight_ih_l0.grad.abs().sum()>0
    path=tmp_path/'state';torch.save(m.state_dict(),path);old=DKT(2,8,0)
    old.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert torch.equal(old(batch['cseqs'],batch['rseqs'].long()),p)
    batch['future_mask'].fill_(False);_,parts=m.compute_loss(batch,return_components=True);assert parts['transfer']==0
    batch['smasks'].fill_(False);assert m.compute_loss(batch)==0
    assert torch.equal(m.compute_loss(batch,return_per_sample=True),torch.zeros(2,dtype=torch.float64))


def test_future_labels_cannot_change_inference_or_earlier_states(batch):
    m=DKTTransfer(2,8,0).eval();before=m(batch['cseqs'],batch['rseqs'].long())
    changed=copy.deepcopy(batch);changed['future_labels'].fill_(0);changed['future_concepts'].fill_(1)
    assert torch.equal(before,m(changed['cseqs'],changed['rseqs'].long()))
    changed['rseqs'][:,-1]=1-changed['rseqs'][:,-1]
    assert torch.equal(before[:,:-1],m(changed['cseqs'],changed['rseqs'].long())[:,:-1])


@pytest.mark.parametrize('field,value',[('future_labels',float('nan')),('future_labels',.5),('future_concepts',2),('future_concepts',1.5)])
def test_invalid_active_targets_and_masked_padding(batch,field,value):
    m=DKTTransfer(2,8,0);batch[field]=batch[field].float()
    if field=='future_concepts':batch[field][0,0,0]=value
    else:batch[field][0,0]=value
    with pytest.raises(ValueError):m.compute_loss(batch)
    batch['future_mask'][0,0]=False;assert torch.isfinite(m.compute_loss(batch))


def _transfer_fixture(directory):
    source=_process_fixture(directory);raw=pd.read_csv(source);raw['template_id']=1;raw.to_csv(source,index=False)
    return source


def test_future_pairing_not_same_question_anchors_last_concept_and_horizon(tmp_path):
    _transfer_fixture(tmp_path);source=tmp_path/'train_valid_sequences.csv'
    d=TransferDataset(source,['questions','concepts'],{1},'same_unassisted',10)
    assert d.mask.all()
    assert d.labels.tolist()==[[0.,1.],[1.,1.]]
    assert d.distances.tolist()==[[1,3],[2,1]]
    assert not torch.any(d.anchor_events[d.mask]==d.target_events[d.mask])
    # The second chunk starts with the final concept of an event that began
    # in the preceding chunk; it is a legitimate completed-event anchor.
    assert int(d.anchor_events[1,0])==10000002
    short=TransferDataset(source,['questions','concepts'],{1},'same_unassisted',1)
    assert short.mask.tolist()==[[True,False],[False,True]]


def test_all_events_remove_hint_conditioning_and_label_only_future_fields(tmp_path):
    raw_path=_transfer_fixture(tmp_path);source=tmp_path/'train_valid_sequences.csv'
    a=TransferDataset(source,['questions','concepts'],{1},'same_all',10)
    assert a.labels.tolist()==[[0.,0.],[0.,1.]]
    before=a.base.dori['rseqs'].clone();raw=pd.read_csv(raw_path)
    raw.loc[4,'hint_count']=1;raw.to_csv(raw_path,index=False)
    b=TransferDataset(source,['questions','concepts'],{1},'same_all',10)
    assert b.labels[1,1]==0 and torch.equal(before,b.base.dori['rseqs'])
    # Changing validation student's process cannot alter training targets.
    target=b.labels.clone();raw=pd.read_csv(raw_path);raw.loc[raw.user_id==20,'hint_count']=2;raw.to_csv(raw_path,index=False)
    c=TransferDataset(source,['questions','concepts'],{1},'same_all',10)
    assert torch.equal(target,c.labels)


def test_cross_template_requires_exact_concept_set_and_separate_question(tmp_path):
    raw_path=_transfer_fixture(tmp_path);raw=pd.read_csv(raw_path);raw.loc[3:4,'template_id']=2;raw.to_csv(raw_path,index=False)
    d=TransferDataset(tmp_path/'train_valid_sequences.csv',['questions','concepts'],{1},'cross_all',10)
    assert d.mask.tolist()==[[True,True],[False,False]]
    assert d.distances[0].tolist()==[3,3]
    assert d.labels[0].tolist()==[0.,1.]


def test_framework_dispatch_and_fold_separation(batch,tmp_path):
    from pykt.models import init_model
    from pykt.models.train_model import model_forward
    m=init_model('dkt_transfer',{'emb_size':8,'dropout':0},{'num_c':2,'emb_path':''},'qid')
    assert torch.equal(model_forward(m,batch),m.compute_loss(batch))
    _transfer_fixture(tmp_path);p=tmp_path/'train_valid_sequences.csv'
    train=TransferDataset(p,['questions','concepts'],{1},'next_event');valid=TransferDataset(p,['questions','concepts'],{0},'next_event')
    assert set(train.dori['uid']).isdisjoint(valid.dori['uid'])
    assert train.distances[train.mask].eq(1).all()
