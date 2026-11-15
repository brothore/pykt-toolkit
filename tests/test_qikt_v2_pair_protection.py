import sys
from pathlib import Path
import torch
from pykt.models.qikt_pair_protected import pair_preservation_loss,QIKTProtectedAdapter
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_qikt_switch_frozen import batch


def test_confident_correct_pair_margin_and_gradient_direction():
    teacher=torch.logit(torch.tensor([[.8,.2]]));student=torch.logit(torch.tensor([[.7,.3]])).requires_grad_()
    loss=pair_preservation_loss(teacher,student,torch.tensor([[1,0]]),torch.ones(1,2,dtype=torch.bool))
    assert torch.allclose(loss,torch.tensor(.2),atol=1e-6)
    loss.backward();assert student.grad[0,0]<0 and student.grad[0,1]>0
    wrong=pair_preservation_loss(-teacher,student.detach().requires_grad_(),torch.tensor([[1,0]]),torch.ones(1,2,dtype=torch.bool));assert wrong==0


def test_pair_protection_keeps_backbone_frozen_and_restores_checkpoint(tmp_path):
    config={'output_mode':'an_irt','v2_adapter_kl':1.,'v2_pair_weight':1.}
    m=QIKTProtectedAdapter(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',other_config=config)
    initial={k:v.clone() for k,v in m.state_dict().items() if 'prefix_adapter' not in k}
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):opt.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();opt.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
    path=tmp_path/'model.ckpt';torch.save(m.state_dict(),path)
    other=QIKTProtectedAdapter(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',other_config=config);other.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(m.predict_one_step(batch(),process=False),other.predict_one_step(batch(),process=False))
