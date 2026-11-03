import sys
from pathlib import Path
import torch
import pytest
from pykt.models.qikt_mamba import QIKT_MAMBA
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from train_validation_only import initialize_from_baseline


def model(**cfg):
    torch.manual_seed(42)
    return QIKT_MAMBA(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',other_config={'output_mode':'an_irt',**cfg})


def batch():
    q=torch.tensor([[0,1,2,1,0]]);c=torch.tensor([[[0,-1],[1,-1],[0,1],[1,-1],[0,-1]]]);r=torch.tensor([[1,0,1,0,1]])
    return dict(cq=q,cc=c,cr=r,qshft=q[:,1:],cshft=c[:,1:],rshft=r[:,1:].float(),sm=torch.ones(1,4,dtype=torch.bool))


def test_frozen_warmstart_has_deterministic_teacher_and_never_updates_backbone(tmp_path):
    baseline=model();baseline.model.eval();path=tmp_path/'base.ckpt';torch.save(baseline.state_dict(),path)
    m=model(freeze_backbone=True,concept_memory_residual=True,head_fusion_residual=True,anchor_kl_weight=1.)
    initialize_from_baseline(m,path);m.model.train()
    assert not m.model.que_emb.att_dropout.training and not m.model.dropout_layer.training
    expected=baseline.predict_one_step(batch(),process=False)
    assert torch.equal(m.predict_one_step(batch(),process=False),expected)
    backbone={k:v.clone() for k,v in m.state_dict().items() if not any(x in k for x in ['concept_memory_residual','head_fusion_residual'])}
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();opt.step()
    for k,v in backbone.items():assert torch.equal(v,m.state_dict()[k]),k
    assert all(p.grad is None for name,p in m.named_parameters() if 'residual' not in name)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==15
    outputs,_=m.predict_one_step(batch(),process=False,return_details=True)
    assert torch.equal(torch.sigmoid(outputs['baseline_logit']),expected)
    assert not torch.equal(outputs['y'],expected)


def test_anchor_is_kl_to_fixed_teacher_not_to_current_labels():
    m=model(freeze_backbone=True,concept_memory_residual=True,anchor_kl_weight=2.)
    with torch.no_grad():m.model.concept_memory_residual.bias.fill_(.5)
    outputs,_=m.predict_one_step(batch(),process=False,return_details=True)
    student=outputs['fusion_logit'];teacher=outputs['baseline_logit'];t=torch.sigmoid(teacher)
    kl=((torch.nn.functional.softplus(student)-t*student)-(torch.nn.functional.softplus(teacher)-t*teacher)).mean()
    _,anchored=m.train_one_step(batch(),process=False)
    m.model.anchor_kl_weight=0.;_,unanchored=m.train_one_step(batch(),process=False)
    assert kl>0 and torch.allclose(anchored-unanchored,2*kl.double(),atol=1e-7)


@pytest.mark.parametrize('cfg',[{'freeze_backbone':True},{'anchor_kl_weight':1.},
 {'freeze_backbone':True,'head_fusion_residual':True,'history_response_dropout':.1}])
def test_invalid_frozen_refinement_rejected(cfg):
    with pytest.raises(ValueError):model(**cfg)
