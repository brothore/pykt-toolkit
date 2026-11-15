import importlib
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parent))
from test_qikt_switch_frozen import batch
from test_qikt_ukt_transfer import make as base_model
from pykt.models.qikt_ukt_rasch import RaschFactors,QIKTRaschTransfer
from pykt.models.qikt_ukt_transfer import QIKTUKTTransfer


def make(mode,freeze=False):
    torch.manual_seed(42)
    cfg={'output_mode':'an_irt','ukt_rasch_mode':mode,'ukt_rasch_scale':.25 if freeze else 1.,'ukt_rasch_freeze':freeze}
    return QIKTRaschTransfer(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='gru',other_config=cfg)


def test_multikc_modulation_and_padding_values():
    f=RaschFactors(3,2,4,'input_vector',2.)
    with torch.no_grad():f.difficulty.weight.fill_(1);f.concept_delta.weight[0].fill_(2);f.concept_delta.weight[1].fill_(4)
    q=torch.tensor([[0,1,2]]);c=torch.tensor([[[0,1],[1,-1],[-1,-1]]])
    assert torch.equal(f.input_delta(q,c),torch.tensor([[[6.,6.,6.,6.],[8.,8.,8.,8.],[0.,0.,0.,0.]]]))


@pytest.mark.parametrize('mode,freeze',[('input_scalar',False),('input_vector',False),('output_vector',True)])
def test_parity_training_causality_and_freeze(mode,freeze):
    teacher=base_model();teacher.model.eval();m=make(mode,freeze);m.model.eval()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in teacher.state_dict().items())
    expected=teacher.predict_one_step(batch(),process=False)
    assert torch.equal(m.predict_one_step(batch(),process=False),expected)
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        m.model.train();optimizer.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());optimizer.step()
    m.model.eval();reference=m.predict_one_step(batch(),process=False);assert not torch.equal(reference,expected)
    for t in range(1,5):
        changed=batch();changed['cr'][:,t:]=1-changed['cr'][:,t:];changed['rshft']=changed['cr'][:,1:].float()
        assert torch.equal(reference[:,:t],m.predict_one_step(changed,process=False)[:,:t])
    if freeze:
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in teacher.state_dict().items())
        assert all(not p.requires_grad for k,p in m.named_parameters() if not k.startswith('model.rasch_transfer.'))


def test_actual_factory_and_strict_saved_configuration_reconstruction():
    module=importlib.import_module('pykt.models.init_model');old=module.device;module.device='cpu'
    try:
        cfg={'emb_size':4,'dropout':.3,'version':'gru','other_config':{'output_mode':'an_irt','ukt_rasch_mode':'input_vector','ukt_rasch_scale':1.}}
        m=module.init_model('qikt_mamba',cfg,{'num_q':3,'num_c':2,'max_concepts':2,'emb_path':''},'attn');m.load_state_dict(make('input_vector').state_dict());m.model.eval()
        assert torch.isfinite(m.predict_one_step(batch(),process=False)).all()
    finally:module.device=old


def test_frozen_w2_warmstart_updates_only_new_parameters(tmp_path):
    from train_validation_only import initialize_from_baseline
    teacher=base_model();teacher.model.eval();path=tmp_path/'baseline.ckpt';torch.save(teacher.state_dict(),path)
    m=QIKTUKTTransfer(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='gru',other_config={'output_mode':'an_irt','ukt_transfer_mode':'w2','ukt_transfer_freeze':1})
    initialize_from_baseline(m,path);m.model.eval();assert torch.equal(m.predict_one_step(batch(),process=False),teacher.predict_one_step(batch(),process=False))
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        m.model.train();optimizer.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();optimizer.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in teacher.state_dict().items())
    assert all(not p.requires_grad for k,p in m.named_parameters() if not k.startswith('model.ukt_transfer.'))
