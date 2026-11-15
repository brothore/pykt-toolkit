import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_qikt_switch_frozen import model,batch
from pykt.models.qikt_prefix_adapter import QIKTPrefixAdapter,prefix_features


def adapter(train_projection=False,kl=0.):
    return QIKTPrefixAdapter(num_q=3,num_c=2,emb_size=4,dropout=.3,emb_type='attn',version='lstm',
        other_config={'output_mode':'an_irt','v2_projection_trainable':train_projection,'v2_adapter_kl':kl,'v2_adapter_scale':.05})


def test_prefix_features_exclude_current_future_labels_and_deduplicate_kcs():
    c=torch.tensor([[[0,0],[0,-1],[1,-1],[0,1]]]);r=torch.tensor([[1,0,1,0]])
    values=prefix_features(c,r,2);changed=r.clone();changed[:,1:]=1-changed[:,1:]
    assert torch.equal(values[:,:2],prefix_features(c,changed,2)[:,:2])
    assert torch.allclose(values[0,1,1],torch.log(torch.tensor(2.)))
    assert values[0,1,3]==1 and values[0,0,3]==0


def test_zero_adapter_matches_teacher_and_optimization_never_updates_backbone():
    baseline=model();baseline.model.eval();m=adapter();baseline_state=baseline.state_dict()
    missing=m.load_state_dict(baseline_state,strict=False)
    assert all(k.startswith('model.prefix_adapter.') for k in missing.missing_keys)
    expected=baseline.predict_one_step(batch(),process=False)
    assert torch.equal(expected,m.predict_one_step(batch(),process=False))
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==353
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        m.model.train();optimizer.zero_grad();_,loss=m.train_one_step(batch(),process=False);loss.backward();optimizer.step()
    assert all(torch.equal(value,m.state_dict()[key]) for key,value in baseline_state.items())
    out,_=m.predict_one_step(batch(),process=False,return_details=True)
    assert float(out['v2_delta'].abs().max())<=.05
    assert not torch.equal(out['y'],expected)


def test_round2_only_adds_projection_trainability_and_keeps_backbone_frozen():
    m=adapter(True,1.)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==417
    assert all(not p.requires_grad for name,p in m.named_parameters() if not name.startswith('model.prefix_adapter.'))
    _,loss=m.train_one_step(batch(),process=False);loss.backward()
    assert torch.isfinite(loss)
