import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from pykt.models.dkt import DKT
from pykt.models.dkt_profile import DKTProfile
from pykt.models.train_model import model_forward
from pykt.datasets.profile_dataset import aggregate_targets,candidate_sessions,ProfileDataset,profile_collate,build_profile_sidecar
from test_dkt_confidence import _fixture


def batch():
    mask=torch.ones(2,3,dtype=torch.bool)
    return {'cseqs':torch.tensor([[0,1,0],[1,0,1]]),'rseqs':torch.tensor([[1.,0.,1.],[0.,1.,0.]]),
            'shft_cseqs':torch.tensor([[1,0,1],[0,1,0]]),'shft_rseqs':torch.tensor([[0.,1.,0.],[1.,0.,1.]]),
            'masks':mask.clone(),'smasks':mask.clone(),'profile_anchors':torch.tensor([[0,0],[1,1]]),
            'profile_concepts':torch.tensor([[[.5,.5],[1.,0.]],[[0.,1.],[.25,.75]]]),
            'profile_labels':torch.tensor([[.5,1.],[0.,.25]]),'profile_mass':torch.tensor([[.5,.5],[.3,.7]]),
            'profile_mask':torch.ones(2,2,dtype=torch.bool)}


def test_zero_weight_rng_loss_adam_and_state_parity():
    data=batch();torch.manual_seed(72);old=DKT(2,8,.2);rng=torch.random.get_rng_state()
    torch.manual_seed(72);new=DKTProfile(2,8,.2,profile_weight=0)
    assert torch.equal(rng,torch.random.get_rng_state());assert list(old.state_dict())==list(new.state_dict())
    a,b=torch.optim.Adam(old.parameters()),torch.optim.Adam(new.parameters());rng=torch.random.get_rng_state()
    p=old(data['cseqs'],data['rseqs'].long()).gather(-1,data['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss=F.binary_cross_entropy(p.double(),data['shft_rseqs'].double());loss.backward();a.step()
    torch.random.set_rng_state(rng);other=new.compute_loss({k:v for k,v in data.items() if not k.startswith('profile_')})
    assert torch.equal(loss,other);other.backward();b.step()
    for key in old.state_dict():assert torch.equal(old.state_dict()[key],new.state_dict()[key])


def test_loss_uses_one_frozen_prefix_and_exact_group_mass(tmp_path):
    data=batch();model=DKTProfile(2,8,0);output=model(data['cseqs'],data['rseqs'].long())
    p=model.profile_predictions(output,data)
    assert torch.equal(p[0,0],output[0,0].mean())
    assert torch.equal(p[1,0],output[1,1,1])
    expected=(F.binary_cross_entropy(p.double(),data['profile_labels'].double(),reduction='none')*data['profile_mass'].double()).sum()/2
    loss,parts=model.compute_loss(data,return_components=True);assert torch.allclose(parts['profile'],expected)
    assert torch.equal(model_forward(model,data),loss)
    parts['profile'].backward();assert model.lstm_layer.weight_ih_l0.grad.abs().sum()>0
    old=DKT(2,8,0);old.load_state_dict(model.export_dkt_state_dict(),strict=True)
    assert torch.equal(output,old(data['cseqs'],data['rseqs'].long()))
    changed=copy.deepcopy(data);changed['rseqs'][0,1:]=1-changed['rseqs'][0,1:]
    assert torch.equal(p[0],model.profile_predictions(model(changed['cseqs'],changed['rseqs'].long()),changed)[0])
    changed['profile_labels'].zero_();assert torch.equal(output,model(changed['cseqs'],data['rseqs'].long()))
    per=model.compute_loss(data,return_per_sample=True);assert per.shape==(2,)


@pytest.mark.parametrize('field,value',[('profile_anchors',3),('profile_anchors',.5),('profile_concepts',float('nan')),('profile_labels',1.1),('profile_labels',float('nan')),('profile_mass',-1)])
def test_active_invalid_fields_and_masked_nan(field,value):
    data=batch();data[field]=data[field].float();data[field][0,0]=value
    model=DKTProfile(2,8,0)
    with pytest.raises(ValueError):model.compute_loss(data)
    data['profile_mask'][0,0]=False;assert torch.isfinite(model.compute_loss(data))


def test_empty_groups_and_inactive_main_anchor():
    data=batch();model=DKTProfile(2,8,0);data['profile_mask'].zero_()
    for key in ('profile_concepts','profile_labels','profile_mass'):data[key].fill_(float('nan'))
    _,parts=model.compute_loss(data,return_components=True);assert parts['profile']==0
    data['smasks'].zero_();assert model.compute_loss(data)==0
    data=batch();data['smasks'][0,0]=False
    with pytest.raises(ValueError):model.compute_loss(data)
    data=batch()
    for key in [k for k in data if k.startswith('profile_')]:data[key]=data[key][:,:0]
    assert torch.isfinite(model.compute_loss(data))


def test_hierarchy_fractional_exposure_avoids_duplicate_event_mass():
    targets=aggregate_targets([[0,1],[1]],[1,0],[5,6],2)
    assert np.allclose(targets[0][1],[.25,.75]);assert targets[0][2]==.5
    assert targets[1][3]==.25;assert targets[2][3]==.75
    assert targets[1][2]==1;assert targets[2][2]==pytest.approx(1/3)
    assert sum(row[3] for row in targets[1:])==1
    assert sum(row[2]*row[3] for row in targets[1:])==pytest.approx(.5)


def event(i,q=1,quiz=1,ts=None):
    return {'q':q,'event':i,'key':(quiz,0),'valid_quiz':True,'ts':i*60000 if ts is None else ts}


def test_candidate_sessions_exclude_ties_revisits_retries_and_tail():
    events=[event(0,0,0),event(1,1,1),event(2,2,1),event(3,3,2)]
    good,excluded=candidate_sessions(events);assert good==[(1,3)];assert excluded['right_censored_user_tail']==1
    tied=copy.deepcopy(events);tied[0]['ts']=tied[1]['ts'];assert not candidate_sessions(tied)[0]
    duplicate=copy.deepcopy(events);duplicate[2]['q']=1;assert not candidate_sessions(duplicate)[0]
    revisited=events+[event(4,4,1),event(5,5,3)];assert not candidate_sessions(revisited)[0]


def fixture(directory,split=False):
    _fixture(directory)
    meta=directory/'metadata'
    subjects=pd.read_csv(meta/'subject_metadata.csv');subjects['ParentId']=[99,98,3];subjects.to_csv(meta/'subject_metadata.csv',index=False)
    answers=pd.read_csv(meta/'answer_metadata_task_3_4.csv');answers['QuizId']=[0,0,1,1,2,0,1,2];answers.to_csv(meta/'answer_metadata_task_3_4.csv',index=False)
    original=pd.read_csv(directory/'train_valid.csv');chunks=[]
    for row in original.to_dict('records'):
        n=len(row['responses'].split(','));row['selectmasks']=','.join(['1']*n+['-1']*(6-n))
        for key in ['questions','concepts','responses','timestamps','is_repeat']:row[key]=','.join(row[key].split(',')+['-1']*(6-n))
        chunks.append(row)
    if split:
        chunks=[]
        for row in original.to_dict('records'):
            n=len(row['responses'].split(','))
            for start in range(0,n,3):
                chunk={'uid':row['uid'],'fold':row['fold'],'selectmasks':'1,1,1'}
                chunk.update({k:','.join(row[k].split(',')[start:start+3]) for k in ['questions','concepts','responses','timestamps','is_repeat']});chunks.append(chunk)
    pd.DataFrame(chunks).to_csv(directory/'train_valid_sequences.csv',index=False)


def test_raw_event_alignment_multiconcept_complete_block_fold_and_collate(tmp_path):
    fixture(tmp_path);source=tmp_path/'train_valid_sequences.csv'
    train=ProfileDataset(source,['questions','concepts'],{1},'hierarchy');valid=ProfileDataset(source,['questions','concepts'],{0},'hierarchy',mode='valid')
    assert train.stats['sessions']==1;assert valid.stats['sessions']==0
    record=train.records[0];assert record['profile_anchors'].unique().tolist()==[1]
    assert record['profile_mass'].sum()==1
    root=record['profile_kind']==0;assert record['profile_labels'][root].tolist()==[.5]
    assert torch.allclose(record['profile_concepts'][root],torch.tensor([[.25,.75]]))
    data=profile_collate([train[0],valid[0]]);assert data['profile_mask'][1].sum()==0
    assert torch.isfinite(DKTProfile(2,8,0).compute_loss(data))
    manifest=tmp_path/'dkt_profile/profiles_v1.json';before=json.loads(manifest.read_text())['sources']
    answers=pd.read_csv(tmp_path/'metadata/answer_metadata_task_3_4.csv');answers.loc[2,'QuizId']=0;answers.to_csv(tmp_path/'metadata/answer_metadata_task_3_4.csv',index=False)
    build_profile_sidecar(tmp_path);assert json.loads(manifest.read_text())['sources']!=before


def test_cross_chunk_block_is_excluded_not_partially_supervised(tmp_path):
    fixture(tmp_path,split=True)
    data=ProfileDataset(tmp_path/'train_valid_sequences.csv',['questions','concepts'],{1})
    assert data.stats['sessions']==0
