import math
import torch
import pytest
from pykt.models.qikt_switch_utils import causal_concept_features, fragment_pairwise_loss
from pykt.models.qikt_mamba import QIKT_MAMBA


def test_memory_hand_calculation_duplicates_padding_and_future_invariance():
    c=torch.tensor([[[0,0],[1,-1],[0,1],[-1,-1],[0,-1]]])
    r=torch.tensor([[1,0,1,0,0]])
    f=causal_concept_features(c,r,2,strength=2.)
    # Prefix for event2: KC0 has one correct, KC1 one incorrect => mean .5.
    assert f[0,0].tolist()==[0.,0.]
    assert f[0,1].tolist()==[0.,0.]
    assert f[0,2,0]==0
    assert torch.allclose(f[0,2,1],torch.tensor(math.log(2.)))
    assert f[0,3].tolist()==[0.,0.]
    assert torch.allclose(f[0,4,0],torch.logit(torch.tensor(.75)))
    missing=r.float();missing[:,3]=float('nan')
    assert torch.equal(f,causal_concept_features(c,missing,2,strength=2.))
    for i in range(5):
        changed=r.clone();changed[:,i:]=1-changed[:,i:]
        assert torch.equal(f[:,:i+1],causal_concept_features(c,changed,2,strength=2.)[:,:i+1])


def test_pairwise_exact_formula_fragment_weighting_mask_and_gradient():
    logits=torch.tensor([[1.,-1.,0.],[0.,2.,-2.]],requires_grad=True)
    p=logits.sigmoid();y=torch.tensor([[1.,0.,float('nan')],[1.,1.,0.]])
    mask=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    loss=fragment_pairwise_loss(p,y,mask)
    expected=(torch.nn.functional.softplus(torch.tensor(-2.))+(torch.nn.functional.softplus(torch.tensor(-2.))+torch.nn.functional.softplus(torch.tensor(-4.)))/2)/2
    assert torch.allclose(loss,expected)
    loss.backward();assert logits.grad[0,0]<0 and logits.grad[0,1]>0 and logits.grad[0,2]==0
    # No defined positive-negative pair yields a finite differentiable zero.
    one=torch.tensor([[.2,.4]],requires_grad=True)
    zero=fragment_pairwise_loss(one,torch.ones_like(one),torch.ones_like(one,dtype=torch.bool))
    zero.backward();assert zero==0 and torch.equal(one.grad,torch.zeros_like(one))
    padded=torch.tensor([[float('nan')]],requires_grad=True)
    assert fragment_pairwise_loss(padded,padded,torch.zeros_like(padded,dtype=torch.bool))==0


def test_memory_residual_zero_start_causal_prediction_and_gradient():
    def model(**cfg):
        torch.manual_seed(42)
        return QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type='attn',version='lstm',other_config={'output_mode':'an_irt',**cfg})
    plain=model();m=model(concept_memory_residual=True,pairwise_loss_weight=.01)
    q=torch.tensor([[0,1,2,1,0]]);c=torch.tensor([[[0,-1],[1,-1],[0,1],[1,-1],[0,-1]]]);r=torch.tensor([[1,0,1,0,1]])
    data=dict(cq=q,cc=c,cr=r,qshft=q[:,1:],cshft=c[:,1:],rshft=r[:,1:].float(),sm=torch.ones(1,4,dtype=torch.bool))
    plain.model.eval();m.model.eval()
    assert torch.equal(plain.predict_one_step(data,process=False),m.predict_one_step(data,process=False))
    for k,v in plain.model.state_dict().items():assert torch.equal(v,m.model.state_dict()[k])
    _,loss=m.train_one_step(data,process=False);loss.backward()
    assert m.model.concept_memory_residual.weight.grad.abs().sum()>0
    with torch.no_grad():m.model.concept_memory_residual.weight.fill_(.5)
    a=m.predict_one_step(data,process=False);changed=r.clone();changed[:,2:]=1-changed[:,2:]
    b=m.predict_one_step({**data,'cr':changed,'rshft':1-data['rshft']},process=False)
    assert torch.equal(a[:,:2],b[:,:2])
    _,base=plain.train_one_step(data,process=False)
    ranked=model(pairwise_loss_weight=.01);ranked.model.eval();p,combined=ranked.train_one_step(data,process=False)
    expected=base+.01*fragment_pairwise_loss(p,data['rshft'],data['sm'])
    assert torch.allclose(combined,expected)


@pytest.mark.parametrize('value',[-1,float('nan'),float('inf')])
def test_invalid_pairwise_coefficient_rejected(value):
    with pytest.raises(ValueError,match='pairwise_loss_weight'):
        QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,emb_type='iekt',version='lstm',other_config={'pairwise_loss_weight':value})


def test_fusion_zero_start_bounded_correction_and_no_target_dependency():
    def model(**cfg):
        torch.manual_seed(42)
        return QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=0,emb_type='attn',version='lstm',other_config={'output_mode':'an_irt',**cfg})
    plain=model();m=model(head_fusion_residual=True,fusion_gate_scale=.1)
    q=torch.tensor([[0,1,2,1,0]]);c=torch.tensor([[[0,-1],[1,-1],[0,1],[1,-1],[0,-1]]]);r=torch.tensor([[1,0,1,0,1]])
    data=dict(cq=q,cc=c,cr=r,qshft=q[:,1:],cshft=c[:,1:],rshft=r[:,1:].float(),sm=torch.ones(1,4,dtype=torch.bool))
    plain.model.eval();m.model.eval()
    assert torch.equal(plain.predict_one_step(data,process=False),m.predict_one_step(data,process=False))
    for k,v in plain.model.state_dict().items():assert torch.equal(v,m.model.state_dict()[k])
    _,loss=m.train_one_step(data,process=False);loss.backward()
    assert m.model.head_fusion_residual.weight.grad.abs().sum()>0
    with torch.no_grad():m.model.head_fusion_residual.weight.fill_(100.)
    outputs,_=m.predict_one_step(data,process=False,return_details=True)
    p=outputs['y'];p_other=m.predict_one_step({**data,'rshft':1-data['rshft']},process=False)
    assert torch.equal(p,p_other)
    # Saturated positive gates add exactly scale * sum(branch logits).
    inverse=lambda x:torch.log(x/(1-x+1e-8)+1e-8)
    total=sum(inverse(outputs[k]) for k in ['y_question_all','y_concept_all','y_concept_next'])
    assert torch.allclose(p,torch.sigmoid(1.1*total),atol=1e-7)
