import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_qikt_switch_frozen import model,batch
from pykt.models.qikt_retrieval_adapter import QIKTRetrievalAdapter


def adapter(mode='content',kl=0.):
    return QIKTRetrievalAdapter(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',
        other_config={'output_mode':'an_irt','retrieval_adapter_mode':mode,'retrieval_adapter_scale':.5,'retrieval_adapter_kl':kl})


@pytest.mark.parametrize('mode',['content','kc_bias','multi_scale'])
def test_initial_parity_frozen_update_and_dropout_scope(mode):
    teacher=model();teacher.model.eval();m=adapter(mode)
    keys=m.load_state_dict(teacher.state_dict(),strict=False)
    assert not keys.unexpected_keys and all(k.startswith('model.retrieval_adapter.') for k in keys.missing_keys)
    m.model.eval();expected=teacher.predict_one_step(batch(),process=False)
    assert torch.equal(m.predict_one_step(batch(),process=False),expected)
    state={k:v.clone() for k,v in teacher.state_dict().items()};opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        m.model.train();assert m.model.retrieval_adapter.training and not m.model.que_emb.training
        opt.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();opt.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
    outputs,_=m.predict_one_step(batch(),process=False,return_details=True)
    assert outputs['retrieval_delta'].abs().max()<=.5


@pytest.mark.parametrize('mode',['content','kc_bias','multi_scale'])
def test_current_and_future_responses_are_masked_in_nonzero_adapter(mode):
    m=adapter(mode);m.model.eval()
    with torch.no_grad():m.model.retrieval_adapter.output.weight.fill_(.05)
    data=batch();expected=m.predict_one_step(data,process=False)
    for t in range(1,5):
        changed={k:v.clone() for k,v in data.items()};changed['cr'][:,t:]=1-changed['cr'][:,t:]
        actual=m.predict_one_step(changed,process=False)
        assert torch.equal(actual[:,:t],expected[:,:t])


def test_retrieval_kl_and_checkpoint_configuration():
    m=adapter('multi_scale',1.);m.model.eval()
    with torch.no_grad():m.model.retrieval_adapter.output.bias.fill_(1.)
    _,anchored=m.train_one_step(batch(),process=False);m.retrieval_kl=0.
    _,plain=m.train_one_step(batch(),process=False)
    assert anchored>plain
    import importlib
    module=importlib.import_module('pykt.models.init_model');old=module.device;module.device='cpu'
    try:
        loaded=module.init_model('qikt_mamba',dict(emb_size=4,dropout=.3,version='lstm',other_config={'output_mode':'an_irt','retrieval_adapter_mode':'multi_scale'}),{'num_q':3,'num_c':2,'max_concepts':2,'emb_path':''},'attn')
        assert isinstance(loaded,QIKTRetrievalAdapter)
    finally:module.device=old
