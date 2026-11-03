import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pykt.datasets.option_dataset import OptionDataset
from pykt.models.dkt import DKT
from pykt.models.dkt_option import DKTOption
from pykt.models.train_model import model_forward
from test_dkt_confidence import _fixture


@pytest.fixture
def batch():
    mask = torch.tensor([[1,1,1],[1,1,0]], dtype=torch.bool)
    return {'cseqs':torch.tensor([[0,1,0],[1,0,0]]),'rseqs':torch.tensor([[1.,0.,1.],[0.,1.,0.]]),
            'shft_cseqs':torch.tensor([[1,0,1],[0,1,0]]),'shft_qseqs':torch.tensor([[0,1,0],[1,0,2]]),
            'shft_rseqs':torch.tensor([[0.,1.,0.],[1.,0.,0.]]),'masks':mask.clone(),'smasks':mask.clone(),
            'option_labels':torch.tensor([[1,1,3],[1,2,-100]]),'option_mask':mask.clone()}


def model(mode='direct', **kwargs):
    return DKTOption(2,8,3,dropout=kwargs.pop('dropout',0),option_mode=mode,option_keys=[0,1,-1],
                     option_prior=[[0,.2,.3,.5],[.2,0,.6,.2],[.25]*4],**kwargs)


@pytest.mark.parametrize('mode',['direct','conditional','prior','binary_control'])
def test_zero_weight_rng_forward_and_adam_update(batch,mode):
    torch.manual_seed(31);old=DKT(2,8,.2);rng_init=torch.random.get_rng_state()
    torch.manual_seed(31);new=model(mode,dropout=.2,option_weight=0)
    assert torch.equal(rng_init,torch.random.get_rng_state())
    a,b=torch.optim.Adam(old.parameters()),torch.optim.Adam(new.parameters());rng=torch.random.get_rng_state()
    p=old(batch['cseqs'],batch['rseqs'].long()).gather(-1,batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss=F.binary_cross_entropy(p[batch['smasks']].double(),batch['shft_rseqs'][batch['smasks']].double())
    loss.backward();a.step();torch.random.set_rng_state(rng)
    newloss=new.compute_loss({k:v for k,v in batch.items() if not k.startswith('option_') and k!='shft_qseqs'})
    assert torch.equal(loss,newloss)
    newloss.backward();b.step()
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k


@pytest.mark.parametrize('mode',['direct','conditional','prior','binary_control'])
def test_aux_gradients_roundtrip_export_and_empty(batch,mode,tmp_path):
    m=model(mode);loss,parts=m.compute_loss(batch,return_components=True)
    parts['option'].backward(retain_graph=True)
    assert m.lstm_layer.weight_ih_l0.grad.abs().sum()>0
    assert m.option_question_emb.weight.grad.abs().sum()>0
    assert torch.equal(loss,model_forward(m,batch))
    path=tmp_path/'state';torch.save(m.state_dict(),path);clone=model(mode)
    clone.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert torch.equal(loss,clone.compute_loss(batch))
    old=DKT(2,8,0);old.load_state_dict(m.export_dkt_state_dict(),strict=True)
    assert torch.equal(old(batch['cseqs'],batch['rseqs'].long()),m(batch['cseqs'],batch['rseqs'].long()))
    batch['option_mask'].fill_(False);_,parts=m.compute_loss(batch,return_components=True);assert parts['option']==0
    batch['smasks'].fill_(False);assert m.compute_loss(batch)==0
    assert torch.equal(m.compute_loss(batch,return_per_sample=True),torch.zeros(2,dtype=torch.float64))


@pytest.mark.parametrize('mode',['direct','conditional','prior','binary_control'])
def test_no_future_choice_or_answer_leakage_and_probabilities(batch,mode):
    m=model(mode).eval();before=m.option_probabilities(batch)
    changed=copy.deepcopy(batch);changed['shft_rseqs']=1-changed['shft_rseqs'];changed['option_labels'].fill_(2)
    changed['CorrectAnswer']=torch.zeros_like(batch['option_labels']);changed['option_mask'].fill_(False)
    after=m.option_probabilities(changed)
    assert torch.equal(before['option_probs'],after['option_probs'])
    assert torch.allclose(before['option_probs'].sum(-1),torch.ones(2,3))
    assert not before['eligible_question_mask'][1,-1]
    if mode in ('conditional','prior'):
        keys=m.option_keys[batch['shft_qseqs']].clamp_min(0);known=before['eligible_question_mask']
        p=m(batch['cseqs'],batch['rseqs'].long()).gather(-1,batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
        chosen=before['option_probs'].gather(-1,keys.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(chosen[known],p[known])
    changed['rseqs'][:,-1]=1-changed['rseqs'][:,-1]
    assert torch.equal(before['option_probs'][:,:-1],m.option_probabilities(changed)['option_probs'][:,:-1])


def test_conditional_excludes_correct_events_and_prior_is_residual(batch):
    m=model('conditional');_,a=m.compute_loss(batch,return_components=True)
    correct=batch['shft_rseqs']==1;batch['option_labels'][correct]=3
    _,b=m.compute_loss(batch,return_components=True);assert torch.equal(a['option'],b['option'])
    prior=model('prior');prior.option_out.weight.data.zero_();prior.option_out.bias.data.zero_()
    p=prior.option_probabilities(batch)['option_probs'][0,0]
    wrong=p.clone();wrong[0]=0;wrong/=wrong.sum()
    assert torch.allclose(wrong,torch.tensor([0.,.2,.3,.5]),atol=1e-6)


def test_binary_control_has_no_wrong_option_identity_supervision(batch):
    m=model('binary_control');a=m.compute_loss(batch)
    # All three wrong targets refer to question 0, with correct key 0.
    wrong=(batch['shft_rseqs']==0)&batch['option_mask']
    batch['option_labels'][wrong]=2
    assert torch.equal(a,m.compute_loss(batch))


def test_direct_loss_decomposes_into_binary_control_and_wrong_identity(batch):
    direct=model('direct');control=model('binary_control');conditional=model('conditional')
    control.load_state_dict(direct.state_dict());conditional.load_state_dict(direct.state_dict())
    _,a=direct.compute_loss(batch,return_components=True)
    _,b=control.compute_loss(batch,return_components=True)
    _,c=conditional.compute_loss(batch,return_components=True)
    mask=batch['option_mask']&batch['masks']&batch['smasks']
    fraction=((batch['shft_rseqs']==0)&mask).sum()/mask.sum()
    assert torch.allclose(a['option'],b['option']+fraction*c['option'],atol=1e-6)


@pytest.mark.parametrize('value',[1.5,float('nan'),9])
def test_invalid_active_targets_but_safe_masked_nan(batch,value):
    m=model();batch['option_labels']=batch['option_labels'].float();batch['option_labels'][0,0]=value
    with pytest.raises(ValueError,match='option_labels'):m.compute_loss(batch)
    batch['option_mask'][0,0]=False;assert torch.isfinite(m.compute_loss(batch))


def option_fixture(directory):
    _fixture(directory);path=directory/'train_task_3_4.csv';r=pd.read_csv(path)
    r['CorrectAnswer']=r.QuestionId.map({100:1,101:2,102:3})
    r['AnswerValue']=np.where(r.IsCorrect==1,r.CorrectAnswer,r.CorrectAnswer%4+1)
    r.to_csv(path,index=False)


def test_training_only_keys_counts_ties_retries_and_copies(tmp_path):
    option_fixture(tmp_path);source=tmp_path/'train_valid_sequences.csv'
    train=OptionDataset(source,['questions','concepts'],{1})
    valid=OptionDataset(source,['questions','concepts'],{0},mode='valid',option_stats=train.option_stats)
    assert train.option_stats['option_keys']==[0,1,2]
    assert train.counts['eligible']==4 and train.counts['wrong']==2
    assert train.labels.tolist()==[[2,3],[1,0]]
    assert train.option_stats==valid.option_stats
    assert set(train.dori['uid']).isdisjoint(valid.dori['uid'])
    assert not any(k.startswith('confidence_') for k in train[0])
    with pytest.raises(ValueError,match='training-fold'):OptionDataset(source,['questions','concepts'],{0},mode='valid')
    raw_path=tmp_path/'train_task_3_4.csv';raw=pd.read_csv(raw_path)
    # Validation answer-key and choice change must not fit training priors.
    selected=raw.UserId==20;raw.loc[selected,'CorrectAnswer']=4
    raw.loc[selected,'AnswerValue']=np.where(raw.loc[selected,'IsCorrect']==1,4,1)
    raw.to_csv(raw_path,index=False)
    rebuilt=OptionDataset(source,['questions','concepts'],{1});assert rebuilt.option_stats==train.option_stats
    shifted=OptionDataset(source,['questions','concepts'],{0},mode='valid',option_stats=train.option_stats)
    assert shifted.counts['eligible']==0


def test_unstable_keys_and_invalid_option_consistency(tmp_path):
    option_fixture(tmp_path);path=tmp_path/'train_task_3_4.csv';raw=pd.read_csv(path)
    raw.loc[3,['CorrectAnswer','AnswerValue']]=[3,3];raw.to_csv(path,index=False)
    train=OptionDataset(tmp_path/'train_valid_sequences.csv',['questions','concepts'],{1})
    assert train.option_stats['option_keys']==[0,-1,2]
    assert train.option_stats['unstable_questions']==1 and train.counts['eligible']==2
    raw.loc[1,'AnswerValue']=raw.loc[1,'CorrectAnswer'];raw.to_csv(path,index=False)
    with pytest.raises(ValueError,match='IsCorrect'):OptionDataset(tmp_path/'train_valid_sequences.csv',['questions','concepts'],{1})


def test_new_loader_routing_leaves_dkt_unchanged(tmp_path):
    from pykt.datasets.init_dataset import init_dataset4train
    option_fixture(tmp_path)
    config={'nips_task34':{'dpath':str(tmp_path),'train_valid_file':'train_valid_sequences.csv','input_type':['questions','concepts'],'folds':[0,1]}}
    train,valid,*_=init_dataset4train('nips_task34','dkt_option',config,0,2)
    old,_,*_=init_dataset4train('nips_task34','dkt',config,0,2)
    assert isinstance(train.dataset,OptionDataset) and train.dataset.option_stats==valid.dataset.option_stats
    assert torch.equal(next(iter(train))['cseqs'],next(iter(old))['cseqs'])
