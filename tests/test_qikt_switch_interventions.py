"""CPU integration checks using the existing GRU version.
These do not validate the Mamba CUDA implementation or model efficacy.
"""
from pathlib import Path
import torch
import pytest
from pykt.models.qikt_mamba import QIKTNet, QIKT_MAMBA


def fixture():
    q=torch.tensor([[0,1,2,1,0]])
    c=torch.tensor([[[0,-1],[1,-1],[0,1],[1,-1],[0,-1]]])
    r=torch.tensor([[1,0,1,0,1]])
    data=dict(cq=q,cc=c,cr=r,qshft=q[:,1:],cshft=c[:,1:],rshft=r[:,1:].float(),sm=torch.ones(1,4,dtype=torch.bool))
    return q,c,r,data


def net(cls=QIKTNet,embedding="iekt",**other):
    return cls(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type=embedding,version='lstm',other_config=other)


@pytest.mark.parametrize("embedding",["iekt","attn"])
def test_disabled_options_exact_snapshot_rng_forward_gradient_parity(embedding):
    # Remove only the unused CUDA import, retaining the exact frozen GRU path.
    source=(Path(__file__).resolve().parents[1]/'tests/fixtures/qikt_mamba_reference.py').read_text()
    namespace={'__name__':'pykt.models._snapshot_qikt','__package__':'pykt.models'}
    exec(compile(source.replace('from mamba_ssm import Mamba',''),'<frozen_qikt_snapshot>','exec'),namespace)
    torch.manual_seed(18);old=net(namespace['QIKTNet'],embedding=embedding);old_rng=torch.random.get_rng_state()
    torch.manual_seed(18);new=net(embedding=embedding);assert torch.equal(old_rng,torch.random.get_rng_state())
    assert list(old.state_dict())==list(new.state_dict())
    q,c,r,data=fixture();rng=torch.random.get_rng_state();a=old(q,c,r,data);torch.random.set_rng_state(rng);b=new(q,c,r,data)
    for k in a:assert torch.equal(a[k],b[k])
    sum(v.sum() for v in a.values()).backward();sum(v.sum() for v in b.values()).backward()
    for (na,pa),(nb,pb) in zip(old.named_parameters(),new.named_parameters()):
        assert na==nb
        if pa.grad is not None:assert torch.equal(pa.grad,pb.grad)


@pytest.mark.parametrize("embedding",["iekt","attn"])
def test_neutral_history_keeps_targets_and_eval_restores_full_feedback(embedding):
    m=net(embedding=embedding,history_response_dropout=1.);q,c,r,data=fixture();other=1-r
    rng=torch.random.get_rng_state();a=m(q,c,r,data);torch.random.set_rng_state(rng);b=m(q,c,other,data)
    for k in a:assert torch.equal(a[k],b[k])
    assert torch.equal(data['rshft'],r[:,1:].float())
    m.eval();a=m(q,c,r,data);b=m(q,c,other,data)
    assert not torch.equal(a['y_question_all'],b['y_question_all'])


@pytest.mark.parametrize("embedding",["iekt","attn"])
def test_future_response_cannot_change_past_predictions(embedding):
    m=net(embedding=embedding).eval();q,c,r,data=fixture();a=m(q,c,r,data);changed=r.clone();changed[:,2:]=1-changed[:,2:]
    b=m(q,c,changed,data)
    for k in a:assert torch.equal(a[k][:,:2],b[k][:,:2])


@pytest.mark.parametrize("embedding",["iekt","attn"])
def test_item_residual_zero_start_learns_from_existing_training_loss(embedding):
    torch.manual_seed(42)
    m=QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type=embedding,version='lstm',other_config={'output_mode':'an_irt','item_residual':True})
    torch.manual_seed(42)
    baseline=QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type=embedding,version='lstm',other_config={'output_mode':'an_irt'})
    baseline.model.eval()
    for key,value in baseline.model.state_dict().items():
        assert torch.equal(value,m.model.state_dict()[key])
    _,_,_,data=fixture();m.model.eval()
    assert torch.equal(m.model.item_residual.weight,torch.zeros(3,1))
    y=m.predict_one_step(data,process=False)
    assert torch.equal(y,baseline.predict_one_step(data,process=False))
    y_again=m.predict_one_step({**data,'rshft':1-data['rshft']},process=False)
    assert torch.equal(y,y_again)
    _,loss=m.train_one_step(data,process=False);loss.backward()
    assert m.model.item_residual.weight.grad.abs().sum()>0


def test_separate_mamba_starts_identical_without_changing_common_rng_or_weights():
    import importlib.util
    if importlib.util.find_spec('mamba_ssm') is None:
        pytest.skip('Mamba installation is required for constructor control')
    def build(separate):
        torch.manual_seed(42)
        return QIKTNet(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type='attn',version='public_mamba',other_config={'separate_mamba_branches':separate})
    shared=build(False);rng=torch.random.get_rng_state();separate=build(True)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert shared.que_lstm_layer is shared.concept_lstm_layer
    assert separate.que_lstm_layer is not separate.concept_lstm_layer
    for key,value in shared.state_dict().items():assert torch.equal(value,separate.state_dict()[key]),key
    for a,b in zip(separate.que_lstm_layer.parameters(),separate.concept_lstm_layer.parameters()):assert a.data_ptr()!=b.data_ptr()
