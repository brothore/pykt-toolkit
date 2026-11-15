import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parent))
from test_qikt_switch_frozen import batch
from pykt.models.qikt_mamba import QIKT_MAMBA
from pykt.models.qikt_ukt_transfer import QIKTUKTTransfer,diagonal_w2,UKTReadout


def make(mode='off',fusion='additive'):
    torch.manual_seed(42)
    cls=QIKT_MAMBA if mode=='off' else QIKTUKTTransfer
    return cls(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='gru',other_config={
        'output_mode':'an_irt','ukt_transfer_mode':mode,'ukt_transfer_scale':.1,'ukt_transfer_fusion':fusion})


def test_gru_alias_matches_legacy_initialization_and_predictions():
    a=make();torch.manual_seed(42)
    b=QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',other_config={'output_mode':'an_irt'})
    a.model.eval();b.model.eval()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(a.predict_one_step(batch(),process=False),b.predict_one_step(batch(),process=False))


def test_actual_model_factory_routes_saved_transfer_configuration():
    import importlib
    module=importlib.import_module('pykt.models.init_model');previous=module.device;module.device='cpu'
    try:
        cfg={'emb_size':4,'dropout':.3,'version':'gru','other_config':{'output_mode':'an_irt','ukt_transfer_mode':'w2','ukt_transfer_scale':.1,'ukt_transfer_fusion':'uncertainty'}}
        restored=module.init_model('qikt_mamba',cfg,{'num_q':3,'num_c':2,'max_concepts':2,'emb_path':''},'attn')
        assert isinstance(restored,QIKTUKTTransfer)
        restored.load_state_dict(make('w2','uncertainty').state_dict())
        restored.model.eval();assert torch.isfinite(restored.predict_one_step(batch(),process=False)).all()
    finally:module.device=previous


def test_w2_matches_direct_diagonal_gaussian_formula():
    mean=torch.tensor([[[0.,1.],[2.,3.]]]);var=torch.tensor([[[1.,4.],[4.,9.]]])
    actual=diagonal_w2(mean,var)
    expected=((mean[:,:,None]-mean[:,None,:])**2+(var.sqrt()[:,:,None]-var.sqrt()[:,None,:])**2).sum(-1)
    assert torch.allclose(actual,expected)
    with pytest.raises(ValueError):diagonal_w2(mean,-var)


@pytest.mark.parametrize('mode',['mean','w2','w2_legacy'])
def test_readout_strict_history_covariance_and_padding_gradients(mode):
    torch.manual_seed(42);readout=UKTReadout(4,mode,width=4)
    x=torch.randn(1,5,4,requires_grad=True);r=torch.tensor([[0,1,0,1,0]]);valid=torch.tensor([[True,True,True,False,False]])
    with torch.no_grad():readout.out[-1].weight.fill_(.01)
    delta,cov,attention=readout(x,r,valid)
    assert torch.equal(attention.triu(),torch.zeros_like(attention))
    assert torch.equal(attention[:,:,3:],torch.zeros_like(attention[:,:,3:]))
    assert torch.equal(cov[:,0],torch.zeros_like(cov[:,0]))
    v=torch.ones(1,5,4) if mode=='mean' else torch.nn.functional.elu(readout.cov(x)+readout.response_cov(r))+1+1e-6
    assert torch.allclose(cov,attention.square()@v)
    delta.sum().backward();assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in readout.parameters())


@pytest.mark.parametrize('mode,fusion',[('mean','additive'),('w2','additive'),('w2','uncertainty'),('w2_legacy','additive')])
def test_zero_initial_parity_causality_training_and_reconstruction(mode,fusion,tmp_path):
    teacher=make();candidate=make(mode,fusion);teacher.model.eval();candidate.model.eval()
    assert isinstance(candidate.model.que_lstm_layer,torch.nn.GRU)
    assert candidate.model.que_lstm_layer is not candidate.model.concept_lstm_layer
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in teacher.state_dict().items())
    expected=teacher.predict_one_step(batch(),process=False)
    assert torch.equal(expected,candidate.predict_one_step(batch(),process=False))
    optimizer=torch.optim.Adam(candidate.parameters(),lr=.01)
    for _ in range(3):
        candidate.model.train();optimizer.zero_grad();_,loss=candidate.train_one_step(batch(),process=False)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters());optimizer.step()
    candidate.model.eval();reference=candidate.predict_one_step(batch(),process=False)
    assert not torch.equal(reference,expected)
    for t in range(1,5):
        changed=batch();changed['cr'][:,t:]=1-changed['cr'][:,t:];changed['rshft']=changed['cr'][:,1:].float()
        assert torch.equal(reference[:,:t],candidate.predict_one_step(changed,process=False)[:,:t])
    path=tmp_path/'model.pt';torch.save(candidate.state_dict(),path)
    restored=make(mode,fusion);restored.load_state_dict(torch.load(path,weights_only=True));restored.model.eval()
    assert torch.equal(reference,restored.predict_one_step(batch(),process=False))
