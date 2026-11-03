import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from pykt.models.dkt import DKT
from pykt.models.dkt_distill import DKTdistill, RichHistoryTeacher
from pykt.datasets.distill_dataset import HistoryProcessDataset, DistillDataset, build_history_sidecar
from test_dkt_process import _process_fixture


@pytest.fixture
def batch():
    mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
    return {'cseqs':torch.tensor([[0,1,0],[1,0,0]]),'rseqs':torch.tensor([[1.,0.,1.],[0.,1.,0.]]),
            'shft_cseqs':torch.tensor([[1,0,1],[0,1,0]]),'shft_rseqs':torch.tensor([[0.,1.,0.],[1.,0.,0.]]),
            'masks':mask.clone(),'smasks':mask.clone(),'teacher_probs':torch.tensor([[.2,.8,.3],[.9,.7,float('nan')]]),
            'teacher_mask':mask.clone()}


def test_student_exact_state_rng_and_zero_kd_adam_equivalence(batch):
    torch.manual_seed(29);old=DKT(2,8,.2);rng_init=torch.random.get_rng_state()
    torch.manual_seed(29);new=DKTdistill(2,8,.2,kd_weight=0)
    assert torch.equal(rng_init,torch.random.get_rng_state())
    assert list(old.state_dict())==list(new.state_dict())
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
    a,b=torch.optim.Adam(old.parameters()),torch.optim.Adam(new.parameters());rng=torch.random.get_rng_state()
    p=old(batch['cseqs'],batch['rseqs'].long()).gather(-1,batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    loss=F.binary_cross_entropy(p[batch['smasks']].double(),batch['shft_rseqs'][batch['smasks']].double())
    loss.backward();a.step();torch.random.set_rng_state(rng)
    newloss=new.compute_loss({k:v for k,v in batch.items() if not k.startswith('teacher_')})
    assert torch.equal(loss,newloss);newloss.backward();b.step()
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key])


@pytest.mark.parametrize('mode',['uniform','confidence'])
def test_detached_teacher_gradient_roundtrip_and_empty(batch,mode,tmp_path):
    m=DKTdistill(2,8,0,kd_mode=mode,kd_temperature=2)
    batch['teacher_probs'].requires_grad_();loss,parts=m.compute_loss(batch,return_components=True)
    parts['distillation'].backward(retain_graph=True)
    assert m.lstm_layer.weight_ih_l0.grad.abs().sum()>0
    assert batch['teacher_probs'].grad is None
    path=tmp_path/'state';torch.save(m.state_dict(),path);old=DKT(2,8,0)
    old.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert torch.equal(old(batch['cseqs'],batch['rseqs'].long()),m(batch['cseqs'],batch['rseqs'].long()))
    batch['teacher_mask'].fill_(False);_,parts=m.compute_loss(batch,return_components=True);assert parts['distillation']==0
    batch['smasks'].fill_(False);assert m.compute_loss(batch)==0
    assert torch.equal(m.compute_loss(batch,return_per_sample=True),torch.zeros(2,dtype=torch.float64))


def test_confidence_zero_and_equal_distributions(batch):
    m=DKTdistill(2,8,0,kd_mode='confidence');batch['teacher_probs'].fill_(.5)
    _,parts=m.compute_loss(batch,return_components=True);assert parts['distillation']==0
    m.kd_mode='uniform'
    with torch.no_grad():batch['teacher_probs']=m(batch['cseqs'],batch['rseqs'].long()).gather(-1,batch['shft_cseqs'].unsqueeze(-1)).squeeze(-1)
    _,parts=m.compute_loss(batch,return_components=True);assert abs(parts['distillation'])<1e-10


@pytest.mark.parametrize('value',[float('nan'),-.2,1.2])
def test_invalid_active_soft_target_and_safe_masked_nan(batch,value):
    m=DKTdistill(2,8,0);batch['teacher_probs'][0,0]=value
    with pytest.raises(ValueError,match='teacher_probs'):m.compute_loss(batch)
    batch['teacher_mask'][0,0]=False;assert torch.isfinite(m.compute_loss(batch))


def test_teacher_zero_history_matches_original_and_causal():
    torch.manual_seed(14);old=DKT(2,8,0);rng=torch.random.get_rng_state()
    torch.manual_seed(14);teacher=RichHistoryTeacher(2,8,0)
    assert torch.equal(rng,torch.random.get_rng_state())
    c=torch.tensor([[0,1,0]]);r=torch.tensor([[1,0,1]]);features=torch.zeros(1,3,11)
    assert torch.equal(old(c,r),teacher(c,r,features))
    before=teacher(c,r,features);features[:,-1]=10
    assert torch.equal(before[:,:-1],teacher(c,r,features)[:,:-1])
    with pytest.raises(ValueError,match='align'):teacher(c,r,torch.zeros(1,2,11))


def test_history_alignment_is_input_not_target_and_fold_norm_isolation(tmp_path):
    source=_process_fixture(tmp_path);path=tmp_path/'train_valid_sequences.csv'
    train=HistoryProcessDataset(path,['questions','concepts'],{1})
    prediction=HistoryProcessDataset(path,['questions','concepts'],{1},mode='predict',normalizer={'0':[0,1],'8':[0,1]})
    assert torch.allclose(prediction.features[0,:,0],torch.log1p(torch.tensor([1.,2.])))
    assert prediction.teacher_mask.tolist()==[[True,True],[True,True]]
    before=train.features.clone();normalizer=copy.deepcopy(train.normalizer)
    raw=pd.read_csv(source)
    # Last raw event for this training student is a target only, never an
    # input-history position in any chunk. Its process must not affect inputs.
    raw.loc[4,'ms_first_response']=100000
    # Held-out students must never determine normalization.
    raw.loc[raw.user_id==20,'ms_first_response']=999999
    raw.to_csv(source,index=False)
    rebuilt=HistoryProcessDataset(path,['questions','concepts'],{1})
    assert rebuilt.normalizer==normalizer and torch.equal(before,rebuilt.features)
    with pytest.raises(ValueError,match='normalizer'):
        HistoryProcessDataset(path,['questions','concepts'],{0},mode='predict')
    valid=HistoryProcessDataset(path,['questions','concepts'],{0},mode='valid',normalizer=normalizer)
    assert set(train.dori['uid']).isdisjoint(valid.dori['uid'])


def test_oof_source_alignment_and_validation_no_targets(tmp_path):
    _process_fixture(tmp_path);source=tmp_path/'train_valid_sequences.csv';history=build_history_sidecar(tmp_path)
    with np.load(history) as data:
        folds,uids=data['folds'],data['uids'];mask=data['starts'][:,1:]&(folds[:,None]!=0)
        shape=mask.shape
    archive=tmp_path/'soft_targets.npz'
    with archive.open('wb') as stream:np.savez(stream,rich_probs=np.full(shape,.7,dtype=np.float32),thin_probs=np.full(shape,.6,dtype=np.float32),mask=mask,folds=folds,uids=uids)
    archive.with_suffix('.json').write_text(json.dumps({'sources':json.loads(history.with_suffix('.json').read_text())['sources']}))
    train=DistillDataset(source,['questions','concepts'],{1},soft_targets=archive)
    valid=DistillDataset(source,['questions','concepts'],{0},mode='valid')
    assert train.mask.all() and not valid.mask.any()
    assert not any('history_process'==key for key in train[0])
    assert torch.all(train.probabilities==.7)
    metadata=json.loads(archive.with_suffix('.json').read_text());metadata['sources']['raw']='stale'
    archive.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='Stale'):DistillDataset(source,['questions','concepts'],{1},soft_targets=archive)


def test_framework_routes_student_without_extra_parameters(batch):
    from pykt.models import init_model
    from pykt.models.train_model import model_forward
    m=init_model('dkt_distill',{'emb_size':8,'dropout':0,'kd_weight':.2},
                 {'num_c':2,'emb_path':''},'qid')
    assert isinstance(m,DKTdistill)
    assert list(m.state_dict())==list(DKT(2,8,0).state_dict())
    assert torch.equal(model_forward(m,batch),m.compute_loss(batch))
