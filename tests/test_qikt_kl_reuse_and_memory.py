import sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from test_qikt_causal_adapter import adapter
from test_qikt_switch_frozen import batch
from pykt.models.qikt_mamba import QIKT_MAMBA
from container_memory_budget import container_budget


def test_kl_reuses_one_forward_and_matches_duplicate_forward_loss_gradient_and_rng():
    torch.manual_seed(42);m=adapter(kl=1.);m.model.train()
    with torch.no_grad():m.model.causal_adapter.output.bias.fill_(.3)
    initial_rng=torch.get_rng_state().clone()
    m.zero_grad();_,native=QIKT_MAMBA.train_one_step(m,batch(),process=False)
    outputs,current=m.predict_one_step(batch(),process=False,return_details=True)
    mask=current['sm'].bool();b=outputs['causal_baseline_logit'][mask];s=outputs['causal_student_logit'][mask];p=torch.sigmoid(b)
    old=native+((F.softplus(s)-p*s)-(F.softplus(b)-p*b)).mean();old.backward()
    grads={k:v.grad.clone() for k,v in m.named_parameters() if v.grad is not None}
    after_old=torch.get_rng_state().clone();m.zero_grad();torch.set_rng_state(initial_rng)
    calls=[];original=m.model.forward
    def count(*a,**kw):calls.append(1);return original(*a,**kw)
    m.model.forward=count
    _,new=m.train_one_step(batch(),process=False);new.backward()
    assert len(calls)==1 and torch.allclose(old,new,atol=1e-10,rtol=0)
    assert torch.equal(after_old,torch.get_rng_state())
    for k,v in m.named_parameters():
        if k in grads:assert torch.allclose(v.grad,grads[k],atol=1e-6,rtol=1e-5),k
    assert m._causal_anchor is None and not m._collect_causal_anchor


def test_clean_cache_is_only_half_credited_and_dirty_pages_are_excluded(tmp_path):
    unit=1024**3
    (tmp_path/'memory.limit_in_bytes').write_text(str(80*unit))
    (tmp_path/'memory.usage_in_bytes').write_text(str(76*unit))
    (tmp_path/'memory.stat').write_text(f'total_inactive_file {24*unit}\ntotal_dirty {unit}\ntotal_writeback {unit}\n')
    value=container_budget(tmp_path)
    assert value['container_headroom_gib']==4 and value['conservative_available_gib']==15
    (tmp_path/'memory.usage_in_bytes').write_text(str(79*unit))
    (tmp_path/'memory.stat').write_text('total_inactive_file 0\n')
    assert container_budget(tmp_path)['conservative_available_gib']==1
