import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qikt_switch_frozen import model, batch
from pykt.models.qikt_causal_adapter import QIKTCausalAdapter, causal_history_features


def adapter(mode='multiscale', kl=0.):
    return QIKTCausalAdapter(num_q=3, num_c=2, emb_size=4, dropout=.3,
        emb_type='attn', version='lstm', other_config={'output_mode':'an_irt',
        'causal_adapter_mode':mode,'causal_adapter_scale':.5,'causal_adapter_kl':kl})


def test_features_have_exact_past_only_values():
    c=torch.tensor([[[0,0],[0,-1],[1,-1],[0,1]]]);r=torch.tensor([[1,0,1,0]])
    f=causal_history_features(c,r,2)
    assert f.shape==(1,4,16)
    assert f[0,1,14]==1 and f[0,1,15]==1
    assert f[0,2,14]==0 and f[0,2,15]==0
    assert torch.allclose(f[0,1,7],torch.log(torch.tensor(2.)))
    for time in range(4):
        changed=r.clone();changed[:,time:]=1-changed[:,time:]
        assert torch.equal(f[:,:time+1],causal_history_features(c,changed,2)[:,:time+1])


def test_recent_features_respect_eight_and_thirty_two_boundaries():
    c=torch.zeros((1,40,1),dtype=torch.long);r=torch.zeros((1,40));r[:,0]=1
    f=causal_history_features(c,r,1)
    assert torch.allclose(f[0,8,6],torch.logit(torch.tensor(.2)))
    assert torch.allclose(f[0,9,6],torch.logit(torch.tensor(.1)))
    assert torch.allclose(f[0,33,10],torch.logit(torch.tensor(1/34)))
    assert torch.isfinite(f).all()


@pytest.mark.parametrize('mode',['question','prefix','multiscale','supported','uncertain','disagreement','low_support'])
def test_zero_adapter_parity_and_only_adapter_updates(mode):
    teacher=model();teacher.model.eval();m=adapter(mode)
    missing=m.load_state_dict(teacher.state_dict(),strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('model.causal_adapter.') for k in missing.missing_keys)
    expected=teacher.predict_one_step(batch(),process=False)
    assert torch.equal(expected,m.predict_one_step(batch(),process=False))
    state={k:v.clone() for k,v in teacher.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        m.model.train();optimizer.zero_grad();_,loss=m.train_one_step(batch(),process=False)
        loss.backward();optimizer.step()
    assert all(torch.equal(value,m.state_dict()[key]) for key,value in state.items())
    outputs,_=m.predict_one_step(batch(),process=False,return_details=True)
    assert outputs['causal_delta'].abs().max()<=.5
    assert not torch.equal(outputs['y'],expected)


def test_kl_anchors_to_frozen_probabilities():
    m=adapter(kl=1.)
    with torch.no_grad():m.model.causal_adapter.output.bias.fill_(1.)
    _,anchored=m.train_one_step(batch(),process=False)
    m.causal_kl=0
    _,plain=m.train_one_step(batch(),process=False)
    assert anchored>plain


@pytest.mark.parametrize('mode',['uncertain','disagreement','low_support'])
def test_gates_match_registered_inputs_and_stay_bounded(mode):
    m=adapter(mode)
    with torch.no_grad():m.model.causal_adapter.output.bias.fill_(.7)
    outputs,current=m.predict_one_step(batch(),process=False,return_details=True)
    if mode=='uncertain':
        p=torch.sigmoid(outputs['causal_baseline_logit']);gate=4*p*(1-p)
    elif mode=='disagreement':
        heads=torch.stack([outputs['y_question_all'],outputs['y_concept_all'],outputs['y_concept_next']],-1)
        gate=((heads.max(-1).values-heads.min(-1).values)/.5).clamp(0,1)
    else:
        f=causal_history_features(current['cc'],current['cr'],2)[:,1:]
        gate=5/(torch.expm1(f[...,1]).clamp_min(0)+5)
    expected=.5*torch.tanh(torch.tensor(.7))*gate
    assert torch.allclose(outputs['causal_delta'],expected)
    assert torch.all((gate>=0)&(gate<=1))


def test_saved_configuration_reconstructs_new_model(tmp_path):
    from pykt.models.init_model import init_model
    import importlib
    module=importlib.import_module('pykt.models.init_model');old=module.device;module.device='cpu'
    try:
        cfg=dict(emb_size=4,dropout=.3,version='lstm',other_config={'output_mode':'an_irt','causal_adapter_mode':'prefix'})
        m=init_model('qikt_mamba',cfg,{'num_q':3,'num_c':2,'max_concepts':2,'emb_path':''},'attn')
        assert isinstance(m,QIKTCausalAdapter)
    finally:module.device=old
