import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from pykt.datasets.indexed_que_dataset import IndexedQueDataset
from pykt.models.qikt_rank_gradient_cache import RankGradientCache,rank_risk_gradient,digest


class Data(Dataset):
    def __init__(self):
        self.aug_probs={};self.dori={'uid':torch.tensor([1,1,2,3]),
            'rseqs':torch.tensor([[0.,1.,1.,-1.],[1.,0.,0.,-1.],[0.,1.,0.,-1.],[0.,1.,1.,-1.]]),
            'smasks':torch.tensor([[1,1,1,-1]]*4)}
    def __len__(self):return 4
    def __getitem__(self,i):
        return {'uid':self.dori['uid'][i],'x':torch.tensor([[.2],[.8],[.5]])+i*.1,
            'shft_rseqs':self.dori['rseqs'][i,1:],'smasks':torch.tensor([True,True,False])}


class Model(torch.nn.Module):
    model_name='qikt_mamba';emb_type='attn'
    def __init__(self):super().__init__();self.model=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(1,1))
    def predict_one_step(self,data):
        assert '_fragment_index' not in data
        return self.model(data['x']).sigmoid().squeeze(-1)
    def get_loss(self,p,y,mask):return F.binary_cross_entropy(p[mask].double(),y[mask].double())
    def train_one_step(self,data,process=True,return_all=False):
        p=self.predict_one_step(data);return p,self.get_loss(p,data['shft_rseqs'],data['smasks'])
    def batch_to_device(self,data,process=True):return {'rshft':data['shft_rseqs'],'sm':data['smasks']}


@pytest.mark.parametrize('mode',['logistic','smooth_auc'])
def test_analytic_full_pair_gradient_matches_autograd(mode):
    z=torch.tensor([-.4,.8,.3,-.2],dtype=torch.double,requires_grad=True);y=np.array([1,1,0,0])
    difference=z[2:][None,:]-z[:2][:,None]
    loss=F.softplus(difference).mean() if mode=='logistic' else torch.sigmoid(difference/.5).mean()
    loss.backward();risk,gradient=rank_risk_gradient(z.detach().numpy(),y,mode,.5)
    assert abs(float(loss.detach())-risk)<1e-12
    assert np.max(np.abs(z.grad.numpy()-gradient))<1e-12


def test_index_wrapper_preserves_native_sampler_and_rng():
    data=Data();original=DataLoader(data,batch_size=2)
    wrapped=DataLoader(IndexedQueDataset(data),batch_size=2,sampler=original.sampler,generator=original.generator)
    torch.manual_seed(42);a=list(original);expected_rng=torch.get_rng_state()
    torch.manual_seed(42);b=list(wrapped)
    assert torch.equal(expected_rng,torch.get_rng_state())
    assert torch.equal(torch.cat([v['_fragment_index'] for v in b]),torch.arange(4))
    for first,second in zip(a,b):
        assert all(torch.equal(value,second[key]) for key,value in first.items())


@pytest.mark.parametrize('mode',['logistic','smooth_auc'])
@pytest.mark.parametrize('support',[0.,5.])
@pytest.mark.parametrize('focus',['off','exponential','variance'])
def test_cached_epoch_mean_value_and_gradient_match_full_uid_risk(tmp_path,mode,support,focus):
    dataset=IndexedQueDataset(Data());loader=DataLoader(dataset,batch_size=2);model=Model()
    cache=RankGradientCache(dataset,{'cached_rank_mode':mode,'cached_rank_reliability':support,
        'cached_rank_focus_mode':focus,'cached_rank_focus_strength':1.,
        'gradient_cache_directory':str(tmp_path/'archive')},{1:4,2:2,3:2})
    folder=tmp_path/'checkpoint';folder.mkdir();cache.prepare_epoch(model,loader,1,folder)
    p=cache.cache['anchor_probabilities'].clone().requires_grad_();z=torch.logit(p.double().clamp(1e-6,1-1e-6))
    y=dataset.dori['rseqs'][:,1:];mask=cache.valid;risks=[]
    for uid in [1,2]:
        ix=[i for i,u in enumerate(dataset.dori['uid']) if int(u)==uid]
        score=z[ix][mask[ix]];labels=y[ix][mask[ix]];diff=score[labels==0][None,:]-score[labels==1][:,None]
        risk=F.softplus(diff).mean() if mode=='logistic' else torch.sigmoid(diff/.5).mean()
        risks.append(cache.weights[uid]*cache.cache['risk_focus_weights'][uid]*risk)
    expected=torch.stack(risks).mean();surrogate=0.
    for batch in loader:
        indices=batch['_fragment_index'];surrogate=surrogate+cache.batch_surrogate(p[indices],batch['smasks'],indices,batch['uid'].tolist())/len(loader)
    assert abs(float(surrogate.detach())-float(expected.detach()))<1e-7
    a=torch.autograd.grad(expected,p,retain_graph=True)[0];b=torch.autograd.grad(surrogate,p)[0]
    assert torch.allclose(a,b,atol=2e-7,rtol=2e-6)
    assert torch.equal(b[:,2],torch.zeros(4))
    old_sha=digest(tmp_path/'archive/epoch_001.pt')
    model.predict_one_step=lambda *_:(_ for _ in ()).throw(AssertionError('Must reuse verified cache'))
    cache.prepare_epoch(model,loader,1,folder);assert digest(tmp_path/'archive/epoch_001.pt')==old_sha
    with torch.no_grad():model.model[1].weight.add_(.1)
    with pytest.raises(ValueError,match='identity differs'):cache.prepare_epoch(model,loader,1,folder)


def test_padding_nan_and_wrong_fragment_mapping(tmp_path):
    data=IndexedQueDataset(Data());loader=DataLoader(data,batch_size=2);model=Model()
    cache=RankGradientCache(data,{'gradient_cache_directory':str(tmp_path/'archive')},{1:4,2:2,3:2});folder=tmp_path/'ckpt';folder.mkdir()
    cache.prepare_epoch(model,loader,1,folder);p=cache.cache['anchor_probabilities'][:2].clone();p[:,2]=float('nan');p.requires_grad_()
    loss=cache.batch_surrogate(p,cache.valid[:2],[0,1],[1,1]);assert torch.isfinite(loss)
    loss.backward();assert torch.isfinite(p.grad).all()
    with pytest.raises(ValueError,match='UID mapping'):cache.batch_surrogate(p,cache.valid[:2],[0,1],[2,2])


def test_native_list_uids_match_tensor_uids_in_cache_identity_and_coverage(tmp_path):
    first=Data();second=Data();second.dori['uid']=second.dori['uid'].tolist()
    a=RankGradientCache(IndexedQueDataset(first),{'gradient_cache_directory':str(tmp_path/'first')},{1:4,2:2,3:2})
    b=RankGradientCache(IndexedQueDataset(second),{'gradient_cache_directory':str(tmp_path/'second')},{1:4,2:2,3:2})
    assert a.data_identity==b.data_identity
    assert np.array_equal(a.uids,b.uids)
    loader=DataLoader(IndexedQueDataset(second),batch_size=2);model=Model();folder=tmp_path/'ckpt';folder.mkdir()
    b.prepare_epoch(model,loader,1,folder)
    for batch in loader:
        rows=batch['_fragment_index'];p=b.cache['anchor_probabilities'][rows].clone().requires_grad_()
        loss=b.batch_surrogate(p,batch['smasks'],rows,batch['uid'].tolist());loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(p.grad).all()


def test_zero_rank_index_wrapper_keeps_native_training_identical():
    from pykt.models.qikt_sequence_objectives import install_training_objective
    def run(indexed):
        torch.manual_seed(42);data=Data();old=DataLoader(data,batch_size=2)
        loader=DataLoader(IndexedQueDataset(data),batch_size=2,sampler=old.sampler,generator=old.generator) if indexed else old
        model=Model();opt=torch.optim.Adam(model.parameters(),lr=.01)
        if indexed:install_training_objective(model,loader.dataset,{'indexed_training_fragments':True,'rank_weight':0.})
        for _ in range(2):
            for batch in loader:
                model.model.train();_,loss=model.train_one_step(batch);opt.zero_grad();loss.backward();opt.step()
        return copy.deepcopy(model.state_dict()),torch.get_rng_state()
    a,ra=run(False);b,rb=run(True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert torch.equal(ra,rb)


@pytest.mark.parametrize('focus',['off','exponential','variance'])
def test_cached_rank_epoch_resume_matches_optimizer_and_rng(tmp_path,monkeypatch,focus):
    import importlib,json,random,signal,sys
    sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
    from resumable_qikt import train_qikt
    native=importlib.import_module('pykt.models.train_model')
    monkeypatch.setattr(native,'model_forward',lambda model,data:model.train_one_step(data)[1])
    def evaluate(*args):
        random.random();np.random.rand();torch.rand(1);return .6,.5
    monkeypatch.setattr(native,'evaluate',evaluate)
    def setup(folder,seed):
        folder.mkdir(exist_ok=True);(folder/'config.json').write_text('{}')
        protocol=folder/'protocol.json'
        if not protocol.exists():protocol.write_text(json.dumps({'rank_weight':.1,'epoch_rank_cache':True,'cached_rank_focus_mode':focus,'cached_rank_focus_strength':1.,'gradient_cache_directory':str(folder/'anchors')}))
        monkeypatch.setenv('QIKT_V2_PROTOCOL',str(protocol));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        loader=DataLoader(IndexedQueDataset(Data()),batch_size=2);model=Model();opt=torch.optim.Adam(model.parameters(),lr=.01)
        return model,opt,loader
    folder=tmp_path/'continuous';a,oa,la=setup(folder,42);train_qikt(a,la,la,3,oa,str(folder),save_model=True)
    expected=copy.deepcopy(a.state_dict());rng=(random.random(),np.random.rand(),torch.rand(1))
    folder=tmp_path/'resumed';b,ob,lb=setup(folder,42)
    def stop_after_eval(*args):
        value=evaluate(*args);signal.raise_signal(signal.SIGTERM);return value
    monkeypatch.setattr(native,'evaluate',stop_after_eval)
    with pytest.raises(SystemExit) as stopped:train_qikt(b,lb,lb,3,ob,str(folder),save_model=True)
    assert stopped.value.code==130
    monkeypatch.setattr(native,'evaluate',evaluate);monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1')
    c,oc,lc=setup(folder,999);train_qikt(c,lc,lc,3,oc,str(folder),save_model=True)
    assert all(torch.equal(expected[k],c.state_dict()[k]) for k in expected)
    actual=(random.random(),np.random.rand(),torch.rand(1))
    assert actual[:2]==rng[:2] and torch.equal(actual[2],rng[2])
