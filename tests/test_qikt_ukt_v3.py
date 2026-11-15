import importlib
import json
import random
import signal
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from test_qikt_switch_frozen import batch
from test_qikt_ukt_transfer import make as baseline
from pykt.models.qikt_ukt_rasch import QIKTRaschTransfer
from pykt.models.qikt_baseline_anchor import install_baseline_anchor,confident_pair_penalty
from pykt.models.qikt_pair_protected import pair_preservation_loss
from train_validation_only import initialize_from_baseline


def make(mode,warm=False):
    return QIKTRaschTransfer(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='gru',other_config={'output_mode':'an_irt','ukt_rasch_mode':mode,'ukt_rasch_scale':2.,'ukt_rasch_warmstart':warm})


@pytest.mark.parametrize('mode',['input_additive','query_vector'])
def test_new_placements_zero_parity_and_future_causality(mode):
    teacher=baseline();torch.manual_seed(42);student=make(mode);teacher.model.eval();student.model.eval()
    assert all(torch.equal(v,student.state_dict()[k]) for k,v in teacher.state_dict().items())
    assert torch.equal(teacher.predict_one_step(batch(),process=False),student.predict_one_step(batch(),process=False))
    with torch.no_grad():student.model.rasch_transfer.difficulty.weight.fill_(.2)
    reference=student.predict_one_step(batch(),process=False)
    for t in range(1,5):
        changed=batch();changed['cr'][:,t:]=1-changed['cr'][:,t:];changed['rshft']=changed['cr'][:,1:].float()
        assert torch.equal(reference[:,:t],student.predict_one_step(changed,process=False)[:,:t])


def test_query_only_placement_keeps_both_gru_history_inputs_identical():
    teacher=baseline();torch.manual_seed(42);student=make('query_vector');teacher.model.eval();student.model.eval()
    with torch.no_grad():student.model.rasch_transfer.difficulty.weight.fill_(.5)
    captured={};handles=[]
    def capture(key):
        def hook(_,args):captured[key]=args[0].detach().clone()
        return hook
    for label,m in [('b',teacher),('s',student)]:
        for branch in ['que_lstm_layer','concept_lstm_layer']:handles.append(getattr(m.model,branch).register_forward_pre_hook(capture((label,branch))))
    teacher.predict_one_step(batch(),process=False);student.predict_one_step(batch(),process=False)
    for h in handles:h.remove()
    assert all(torch.equal(captured[('b',branch)],captured[('s',branch)]) for branch in ['que_lstm_layer','concept_lstm_layer'])


def test_vectorized_pair_loss_matches_original_value_and_gradients():
    torch.manual_seed(3);t=torch.rand(3,6,dtype=torch.double);s=torch.rand(3,6,dtype=torch.double,requires_grad=True);y=torch.tensor([[1,0,1,0,1,0],[1,1,1,1,1,1],[1,0,0,1,0,1]]);mask=torch.ones_like(y,dtype=torch.bool);mask[2,-1]=False
    first=confident_pair_penalty(t,s,y,mask,.1);second=pair_preservation_loss(torch.logit(t),torch.logit(s),y,mask,.1)
    assert torch.allclose(first,second,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(first,s,retain_graph=True)[0],torch.autograd.grad(second,s)[0],atol=1e-12)


def test_baseline_teacher_is_frozen_and_not_in_student_checkpoint(tmp_path):
    b=baseline();path=tmp_path/'base.pt';torch.save(b.state_dict(),path);m=make('query_vector',True);initialize_from_baseline(m,path)
    expected_keys=set(m.state_dict());m.model.eval();_,plain=m.train_one_step(batch(),process=False)
    meta=install_baseline_anchor(m,{'initial_checkpoint':str(path),'baseline_anchor_kl':3.,'baseline_pair_weight':.1})
    teacher=next(c.cell_contents for c in m.train_one_step.__func__.__closure__ if isinstance(c.cell_contents,QIKTRaschTransfer))
    original={k:v.clone() for k,v in teacher.state_dict().items()};_,anchored=m.train_one_step(batch(),process=False);assert torch.allclose(plain,anchored,atol=1e-12)
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(3):m.model.train();opt.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();opt.step()
    assert set(m.state_dict())==expected_keys and not meta['teacher_in_inference_state']
    assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in original.items())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())


class Rows(Dataset):
    def __init__(self):self.aug_probs={};self.dori={'uid':[1,2],'rseqs':batch()['cr'].repeat(2,1),'smasks':torch.ones(2,5,dtype=torch.long)}
    def __len__(self):return 2
    def __getitem__(self,i):
        b=batch();return {'uid':i+1,'qseqs':b['cq'][0,:-1],'shft_qseqs':b['cq'][0,1:],'cseqs':b['cc'][0,:-1],'shft_cseqs':b['cc'][0,1:],'rseqs':b['cr'][0,:-1],'shft_rseqs':b['cr'][0,1:].float(),'masks':torch.ones(4,dtype=torch.bool),'smasks':torch.ones(4,dtype=torch.bool),'tseqs':torch.zeros(4,dtype=torch.long),'shft_tseqs':torch.zeros(4,dtype=torch.long)}


def test_anchor_teacher_reconstruction_keeps_epoch_resume_exact(tmp_path,monkeypatch):
    from resumable_qikt import train_qikt
    native=importlib.import_module('pykt.models.train_model');monkeypatch.setattr(native,'model_forward',lambda m,d:m.train_one_step(d)[1]);monkeypatch.setattr(native,'evaluate',lambda *_:(.6,.5))
    path=tmp_path/'base.pt';torch.save(baseline().state_dict(),path)
    def setup(folder,seed):
        folder.mkdir(exist_ok=True);(folder/'config.json').write_text('{}');proto=folder/'protocol.json'
        proto.write_text(json.dumps({'alpha':0.,'rank_weight':0.,'initial_checkpoint':str(path),'baseline_anchor_kl':1.,'baseline_pair_weight':.1}))
        monkeypatch.setenv('QIKT_V2_PROTOCOL',str(proto));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        m=make('query_vector',True);initialize_from_baseline(m,path);opt=torch.optim.Adam(m.parameters(),lr=.01);loader=DataLoader(Rows(),batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(123));return m,opt,loader
    full=tmp_path/'full';a,oa,la=setup(full,42);train_qikt(a,la,la,3,oa,str(full),save_model=True);expected={k:v.clone() for k,v in a.state_dict().items()};rng=torch.get_rng_state()
    resumed=tmp_path/'resumed';b,ob,lb=setup(resumed,42)
    def stop(*_):signal.raise_signal(signal.SIGTERM);return .6,.5
    monkeypatch.setattr(native,'evaluate',stop)
    with pytest.raises(SystemExit):train_qikt(b,lb,lb,3,ob,str(resumed),save_model=True)
    monkeypatch.setattr(native,'evaluate',lambda *_:(.6,.5));monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1');c,oc,lc=setup(resumed,999);train_qikt(c,lc,lc,3,oc,str(resumed),save_model=True)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(rng,torch.get_rng_state())
