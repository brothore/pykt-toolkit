import importlib.util
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from reference_ablation_models import intervene


def test_lefokt_interaction_ablation_retains_query_rasch():
    path=Path('/data/pykt-toolkit-latest/pykt/models/lefokt_akt.py')
    spec=importlib.util.spec_from_file_location('lefokt_cpu_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.LEFOKT_AKT(3,5,8,1,0.,d_ff=8,num_attn_heads=2,emb_type='qid_noforgetting');model.eval()
    with torch.no_grad():model.difficult_param.weight.fill_(.2)
    source=intervene(model,module,'no_interaction_rasch')
    assert source.count('qa_embed_data = qa_embed_data')==2
    assert 'q_embed_data = q_embed_data + pid_embed_data' in source
    q=torch.tensor([[0,1,2,0]]);r=torch.tensor([[0,1,0,1]]);pid=torch.tensor([[0,1,2,3]])
    assert torch.isfinite(model(q,r,pid)[0]).all()


def test_native_reference_epoch_resume_is_exact(tmp_path):
    from torch.utils.data import DataLoader
    from reference_training import train
    path=Path('/data/pykt-toolkit-latest/pykt/models/lefokt_akt.py')
    spec=importlib.util.spec_from_file_location('lefokt_resume_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def make():
        torch.manual_seed(42)
        return module.LEFOKT_AKT(3,5,8,1,.2,d_ff=8,num_attn_heads=2,emb_type='qid_noforgetting')
    samples=[]
    for i in range(4):
        samples.append({'cseqs':torch.tensor([0,1,2]),'shft_cseqs':torch.tensor([1,2,0]),'qseqs':torch.tensor([0,1,2]),'shft_qseqs':torch.tensor([1,2,3]),'rseqs':torch.tensor([i%2,1,0]).float(),'shft_rseqs':torch.tensor([1,0,(i+1)%2]).float(),'smasks':torch.ones(3,dtype=torch.bool)})
    loader=DataLoader(samples,batch_size=2);a=make();oa=torch.optim.Adam(a.parameters(),lr=.001)
    assert train(a,loader,loader,oa,tmp_path/'full',{'id':'test'},max_epochs=3)==0
    b=make();ob=torch.optim.Adam(b.parameters(),lr=.001)
    assert train(b,loader,loader,ob,tmp_path/'split',{'id':'test'},max_epochs=3,epoch_hook=lambda n:n==1)==130
    torch.manual_seed(999);c=make();oc=torch.optim.Adam(c.parameters(),lr=.001)
    assert train(c,loader,loader,oc,tmp_path/'split',{'id':'test'},max_epochs=3)==0
    for key,value in a.state_dict().items():assert torch.equal(value,c.state_dict()[key]),key
    full=torch.load(tmp_path/'full/resume.pt',weights_only=False);split=torch.load(tmp_path/'split/resume.pt',weights_only=False)
    assert torch.equal(full['rng']['torch'],split['rng']['torch'])
    assert full['selected_epoch']==split['selected_epoch']


def test_lefokt_query_factorial_and_additive_preserve_identity_and_restore(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    path=Path('/data/pykt-toolkit-latest/pykt/models/lefokt_akt.py')
    spec=importlib.util.spec_from_file_location('lefokt_factorial_cpu_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def make():
        torch.manual_seed(42)
        return module.LEFOKT_AKT(3,5,8,1,0.,d_ff=8,num_attn_heads=2,emb_type='qid_alibi')
    q=torch.tensor([[0,1,2,0]]);r=torch.tensor([[0,1,0,1]]);pid=torch.tensor([[0,1,2,3]])
    modes=['no_query_rasch','no_both_rasch','no_query_rasch_no_alibi',
        'no_interaction_rasch_no_alibi','no_both_rasch_no_alibi','additive_rasch','additive_rasch_no_alibi']
    for mode in modes:
        base=make().eval();model=make().eval();patch=intervene(model,module,mode)
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in base.parameters())
        assert torch.equal(base(q,r,pid)[0],model(q,r,pid)[0]) or mode.endswith('_no_alibi')
        if mode.startswith('no_query'):
            assert 'q_embed_data = q_embed_data + pid_embed_data' not in patch
            assert 'qa_embed_data = qa_embed_data + pid_embed_data' in patch
        if mode.startswith('no_both'):
            assert 'qa_embed_data = qa_embed_data + pid_embed_data' not in patch
        if mode.startswith('additive'):
            with torch.no_grad():model.difficult_param.weight.fill_(.3)
            changed_pid=(pid+1)%5
            with torch.no_grad():model.difficult_param.weight[1].fill_(.9)
            assert not torch.equal(model(q,r,pid)[0],model(q,r,changed_pid)[0])
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        score,reg=model(q,r,pid);loss=score.square().mean()+reg;loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();clone=make().eval();intervene(clone,module,mode);clone.load_state_dict(model.state_dict())
        assert torch.equal(model(q,r,pid)[0],clone(q,r,pid)[0])
