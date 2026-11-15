import sys
from pathlib import Path
import importlib
import pytest
import torch
from pykt.models.qikt_mamba import QIKT_MAMBA
from pykt.models.qikt_reference_fusion import QIKTReferenceFusion,CausalTimeEncoding,alibi_slopes
sys.path.insert(0,str(Path(__file__).parent))
from test_qikt_switch_frozen import batch

MODES=['cic_question','cic_concept','cic_both','time_input','time_decode','time_gate','time_input_decode','time_sparse_input','scalar_additive_query','scalar_additive_input','alibi_logit','alibi_question','alibi_concept']
def data():
    x=batch();x['m']=torch.ones(1,4,dtype=torch.bool);x['ct']=torch.arange(5)[None]*60000;return x

def make(mode=None,**config):
    torch.manual_seed(42);cls=QIKT_MAMBA if mode is None else QIKTReferenceFusion
    return cls(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='gru',other_config={'output_mode':'an_irt','reference_fusion_mode':mode or 'off','reference_fusion_scale':.1,**config})


@pytest.mark.parametrize('mode',MODES)
def test_zero_initialization_finite_gradients_causality_and_restore(mode,tmp_path):
    b=make();m=make(mode);b.model.eval();m.model.eval();expected=b.predict_one_step(data(),process=False)
    assert all(torch.equal(t,m.state_dict()[k]) for k,t in b.state_dict().items())
    assert torch.equal(expected,m.predict_one_step(data(),process=False))
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(2):
        m.model.train();opt.zero_grad();_,loss=m.train_one_step(data(),process=False);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    m.model.eval();pred=m.predict_one_step(data(),process=False)
    assert not torch.equal(pred,expected)
    for t in range(1,5):
        other=data();other['cr'][:,t:]=1-other['cr'][:,t:];other['rshft']=other['cr'][:,1:].float()
        assert torch.equal(pred[:,:t],m.predict_one_step(other,process=False)[:,:t])
        other=data();other['ct'][:,t+1:]+=100000
        assert torch.equal(pred[:,:t],m.predict_one_step(other,process=False)[:,:t])
    restored=make(mode);restored.load_state_dict(m.state_dict());restored.model.eval();assert torch.equal(pred,restored.predict_one_step(data(),process=False))


def test_multi_kc_time_features_match_hand_computed_categories():
    encoder=CausalTimeEncoding(width=4);c=torch.tensor([[[0,-1],[1,-1],[0,1]]]);t=torch.tensor([[0,60000,120000]]);valid=torch.ones(1,3,dtype=torch.bool);f=encoder.features(c,t,valid)
    assert f.shape==(1,3,79)
    # At third question KC0 has gap2min -> round(log2(3))+1=3;
    # KC1 has gap1min ->2. Each occurred once -> count category1.
    assert f[0,2,2]==.5 and f[0,2,3]==.5
    assert f[0,2,32+2]==1
    assert f[0,2,64+1]==1
    assert f[0,0,0]==1 and f[0,0,32]==1 and f[0,0,64]==1


def test_alibi_matches_native_power_of_two_rule():
    assert alibi_slopes(4)==[.25,.0625,.015625,.00390625]
    assert len(alibi_slopes(3))==3


def test_native_factory_routes_reference_and_baseline_defaults_unchanged():
    module=importlib.import_module('pykt.models.init_model');previous=module.device;module.device='cpu'
    try:
        cfg={'emb_size':4,'dropout':.3,'version':'gru','other_config':{'output_mode':'an_irt','reference_fusion_mode':'alibi_logit','reference_fusion_bias':'none'}}
        m=module.init_model('qikt_mamba',cfg,{'num_q':3,'num_c':2,'max_concepts':2,'emb_path':''},'attn');assert isinstance(m,QIKTReferenceFusion)
        assert m.model.reference_fusion.readout.use_bias is False
    finally:module.device=previous


@pytest.mark.parametrize('mode',['cic_both','time_gate','time_input_decode','time_sparse_input','scalar_additive_query','scalar_additive_input','alibi_logit','scalar_query','shared_history','shared_scalar'])
def test_complete_epoch_resume_preserves_new_modules_optimizer_and_rng(mode,tmp_path,monkeypatch):
    import json,random,signal
    import numpy as np
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
    from resumable_qikt import train_qikt
    from test_qikt_ukt_v3 import Rows
    monkeypatch.delenv('QIKT_V2_PROTOCOL',raising=False)
    native=importlib.import_module('pykt.models.train_model');monkeypatch.setattr(native,'model_forward',lambda m,d:m.train_one_step(d)[1]);monkeypatch.setattr(native,'evaluate',lambda *_:(.6,.5))
    def setup(path,seed):
        path.mkdir(exist_ok=True);(path/'config.json').write_text('{}');random.seed(seed);np.random.seed(seed);m=make(mode);torch.manual_seed(seed)
        opt=torch.optim.Adam(m.parameters(),lr=.01);loader=DataLoader(Rows(),batch_size=1,shuffle=True,generator=torch.Generator().manual_seed(123));return m,opt,loader
    full=tmp_path/'full';a,oa,la=setup(full,42);train_qikt(a,la,la,3,oa,str(full),save_model=True);rng=torch.get_rng_state()
    split=tmp_path/'split';b,ob,lb=setup(split,42)
    def stop(*_):signal.raise_signal(signal.SIGTERM);return .6,.5
    monkeypatch.setattr(native,'evaluate',stop)
    with pytest.raises(SystemExit):train_qikt(b,lb,lb,3,ob,str(split),save_model=True)
    monkeypatch.setattr(native,'evaluate',lambda *_:(.6,.5));monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1');c,oc,lc=setup(split,999);train_qikt(c,lc,lc,3,oc,str(split),save_model=True)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())


def test_cic_current_control_matches_parameter_count_and_excludes_extra_history():
    from pykt.models.qikt_reference_fusion import CausalInteractionConv
    torch.manual_seed(42);past=CausalInteractionConv(4,dropout=0.,context='past');torch.manual_seed(42);current=CausalInteractionConv(4,dropout=0.,context='current')
    assert sum(p.numel() for p in past.parameters())==sum(p.numel() for p in current.parameters())
    with torch.no_grad():past.out.weight.fill_(.1);current.out.weight.fill_(.1)
    x=torch.randn(1,5,4);changed=x.clone();changed[:,:3]+=10
    assert torch.equal(current(x)[:,3:],current(changed)[:,3:])
    assert not torch.equal(past(x)[:,3:],past(changed)[:,3:])
    current(x).square().sum().backward()
    assert torch.isfinite(current.long.weight.grad).all() and current.long.weight.grad.abs().sum()>0


@pytest.mark.parametrize('mode',['scalar_query','shared_history','shared_scalar'])
def test_lefokt_query_only_design_keeps_current_identity_and_causal_shared_history(mode):
    b=make();m=make(mode);b.model.eval();m.model.eval();expected=b.predict_one_step(data(),process=False)
    if mode=='scalar_query':assert torch.equal(expected,m.predict_one_step(data(),process=False))
    else:assert not torch.equal(expected,m.predict_one_step(data(),process=False))
    seen=[];hook=m.model.que_lstm_layer.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    m.predict_one_step(data(),process=False);hook.remove()
    if mode in ['shared_history','shared_scalar']:
        assert torch.equal(seen[0][...,:4],torch.zeros_like(seen[0][...,:4]))
        assert torch.equal(seen[0][...,8:12],torch.zeros_like(seen[0][...,8:12]))
    # Both historical GRUs remain independent; current question is still visible.
    assert m.model.que_lstm_layer is not m.model.concept_lstm_layer
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(2):
        m.model.train();opt.zero_grad();_,loss=m.train_one_step(data(),process=False);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
    m.model.eval();pred=m.predict_one_step(data(),process=False)
    for t in range(1,5):
        other=data();other['cr'][:,t:]=1-other['cr'][:,t:];other['rshft']=other['cr'][:,1:].float();assert torch.equal(pred[:,:t],m.predict_one_step(other,process=False)[:,:t])
    other=data();other['cq'][:,1:]=(other['cq'][:,1:]+1)%3
    assert not torch.equal(pred,m.predict_one_step(other,process=False))
    restored=make(mode);restored.load_state_dict(m.state_dict());restored.model.eval();assert torch.equal(pred,restored.predict_one_step(data(),process=False))


def test_shared_history_weight_one_is_forward_baseline_and_half_is_partial():
    base=make();same=make('shared_history',reference_history_weight=1.);partial=make('shared_history',reference_history_weight=.5)
    for m in [base,same,partial]:m.model.eval()
    assert torch.equal(base.predict_one_step(data(),process=False),same.predict_one_step(data(),process=False))
    assert not torch.equal(base.predict_one_step(data(),process=False),partial.predict_one_step(data(),process=False))
    assert base.state_dict().keys()==same.state_dict().keys()==partial.state_dict().keys()


def test_sparse_time_gate_uses_only_causal_kc_coverage():
    from pykt.models.qikt_reference_fusion import ReferenceFusion
    fusion=ReferenceFusion(4,'time_sparse_input',.5)
    c=torch.tensor([[[0],[0],[0],[0]]]);t=torch.tensor([[0,60000,120000,180000]])
    valid=torch.ones(1,4,dtype=torch.bool);embedding=torch.zeros(1,4,8)
    a=fusion.context(embedding,c,torch.zeros(1,4),t,valid)
    b=fusion.context(embedding,c,torch.ones(1,4),t,valid)
    assert torch.equal(a,b)
    assert torch.allclose(a[0,:,-1],torch.tensor([1.,.5,1/3,1/3]))
    changed=c.clone();changed[:,-1]=1
    future=fusion.context(embedding,changed,torch.zeros(1,4),t,valid)
    assert torch.equal(a[:,:3],future[:,:3])


def test_scalar_additive_has_matched_parameter_count_and_keeps_item_channel():
    from pykt.models.qikt_ukt_rasch import RaschFactors
    additive=RaschFactors(3,2,4,'input_scalar_additive',.5)
    multiplicative=RaschFactors(3,2,4,'input_scalar',.5)
    assert sum(p.numel() for p in additive.parameters())==sum(p.numel() for p in multiplicative.parameters())
    q=torch.tensor([[0,1]]);c=torch.tensor([[[0],[0]]])
    assert torch.equal(additive.input_delta(q,c),torch.zeros(1,2,4))
    with torch.no_grad():
        additive.difficulty.weight[0].fill_(.2);additive.difficulty.weight[1].fill_(.6)
        additive.concept_delta.weight.add_(.3)
    result=additive.input_delta(q,c)
    assert torch.allclose(result,torch.tensor([[[.25]*4,[.45]*4]]))
    result.sum().backward()
    assert additive.difficulty.weight.grad.abs().sum()>0
    assert additive.concept_delta.weight.grad.abs().sum()>0
    assert not additive.concept_origin.requires_grad
