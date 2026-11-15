from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from pykt.datasets.complete_uid_sampler import CompleteUIDBatchSampler
from pykt.models.qikt_complete_uid_objective import CompleteUIDObjective


class ToyDataset(SimpleNamespace):
    def __len__(self): return len(self.dori['uid'])


def fixture():
    # UID1 has two separate, individually single-class fragments: only a
    # complete-UID objective can compare their positives and negatives.
    d=ToyDataset(dori={'uid':torch.tensor([1,1,2]),
        'rseqs':torch.tensor([[0.,1.,1.],[1.,0.,0.],[0.,1.,0.]]),
        'smasks':torch.ones(3,3)})
    sampler=CompleteUIDBatchSampler(d,2)
    return d,sampler


def test_sampler_keeps_all_fragments_and_rng_restores():
    d,sampler=fixture();torch.manual_seed(42);rng=torch.get_rng_state()
    first=list(sampler);torch.set_rng_state(rng);assert list(sampler)==first
    assert sorted(i for b in first for i in b)==[0,1,2]
    assert any(0 in b and 1 in b for b in first)


def test_complete_rank_includes_cross_fragment_pairs_and_gradients():
    d,sampler=fixture();obj=CompleteUIDObjective(d,sampler,{}, {1:4,2:2})
    p=torch.tensor([[.6,.7],[.4,.3],[.8,.2]],dtype=torch.double,requires_grad=True)
    y=d.dori['rseqs'][:,1:];mask=torch.ones_like(y,dtype=torch.bool)
    got=obj.rank_loss(p,y,mask,[1,1,2]);scores=torch.logit(p)
    expected=(F.softplus(scores[1][None,:]-scores[0][:,None]).mean()+
              F.softplus(scores[2,1]-scores[2,0]))/2
    assert torch.allclose(got,expected,atol=1e-12)
    ga=torch.autograd.grad(got,p,retain_graph=True)[0];gb=torch.autograd.grad(expected,p)[0]
    assert torch.allclose(ga,gb,atol=1e-12)
    with pytest.raises(ValueError,match='Incomplete UID'):
        obj.rank_loss(p[:1],y[:1],mask[:1],[1])


def test_smooth_auc_risk_matches_manual_complete_auc_surrogate():
    d,sampler=fixture();obj=CompleteUIDObjective(d,sampler,{'complete_rank_mode':'smooth_auc','complete_rank_temperature':.5},{1:4,2:2})
    p=torch.tensor([[.6,.7],[.4,.3],[.8,.2]],dtype=torch.double,requires_grad=True)
    y=d.dori['rseqs'][:,1:];mask=torch.ones_like(y,dtype=torch.bool);scores=torch.logit(p)
    expected=(torch.sigmoid((scores[1][None,:]-scores[0][:,None])/.5).mean()+torch.sigmoid((scores[2,1]-scores[2,0])/.5))/2
    result=obj.rank_loss(p,y,mask,[1,1,2]);assert torch.allclose(result,expected,atol=1e-12)
    result.backward();assert torch.isfinite(p.grad).all()


def test_padding_nan_does_not_enter_complete_ranking():
    d,sampler=fixture();obj=CompleteUIDObjective(d,sampler,{}, {1:4,2:2})
    p=torch.tensor([[.6,.7,float('nan')],[.4,.3,float('nan')],[.8,.2,float('nan')]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[1.,1.,-1.],[0.,0.,-1.],[1.,0.,-1.]])
    result=obj.rank_loss(p,y,y!=-1,[1,1,2]);assert torch.isfinite(result)
    result.backward();assert torch.isfinite(p.grad).all()


def test_single_class_uid_has_connected_zero_ranking():
    d=ToyDataset(dori={'uid':torch.tensor([1]),'rseqs':torch.tensor([[0.,1.,1.]]),'smasks':torch.ones(1,3)})
    sampler=CompleteUIDBatchSampler(d,1);obj=CompleteUIDObjective(d,sampler,{}, {1:2})
    p=torch.tensor([[.7,.8]],requires_grad=True);y=torch.ones(1,2)
    loss=obj.rank_loss(p,y,torch.ones(1,2,dtype=torch.bool),[1]);assert loss==0
    loss.backward();assert torch.equal(p.grad,torch.zeros_like(p))


def test_complete_objective_requires_compatible_sampler():
    from pykt.models.qikt_sequence_objectives import install_training_objective
    d,sampler=fixture();model=SimpleNamespace()
    with pytest.raises(ValueError,match='matching batch sampler'):
        install_training_objective(model,d,{'full_uid_batch_size':2,'rank_weight':.01})


def test_complete_training_hook_keeps_native_bce_and_adds_correct_rank():
    from pykt.models.qikt_sequence_objectives import install_training_objective
    d,sampler=fixture();d._qikt_complete_uid_sampler=sampler
    class Model:
        def __init__(self):self.prediction=torch.tensor([[.6,.7],[.4,.3],[.8,.2]],dtype=torch.double,requires_grad=True)
        def get_loss(self,p,y,mask):return F.binary_cross_entropy(p[mask],y[mask].double())
        def train_one_step(self,data,process=True,return_all=False):return self.prediction,self.get_loss(self.prediction,data['y'],data['mask'])
        def batch_to_device(self,data,process=True):return {'rshft':data['y'],'sm':data['mask']}
    data={'uid':d.dori['uid'],'y':d.dori['rseqs'][:,1:],'mask':torch.ones(3,2,dtype=torch.bool)}
    for weight in [0.,.01]:
        model=Model();native=model.get_loss(model.prediction,data['y'],data['mask'])
        obj=install_training_objective(model,d,{'full_uid_batch_size':2,'rank_weight':weight})
        _,loss=model.train_one_step(data)
        expected=native+weight*obj.complete.rank_loss(model.prediction,data['y'],data['mask'],[1,1,2])
        assert torch.allclose(loss,expected,atol=1e-12)
        if not weight:assert torch.equal(loss,native)
        assert obj.metadata()['rank_eligible_orirows']==2


def test_uid_sampler_epoch_resume_preserves_updates_and_rng():
    import copy
    from torch.utils.data import DataLoader
    class Data(ToyDataset):
        def __getitem__(self,index):
            return {'uid':self.dori['uid'][index], 'y':self.dori['rseqs'][index,1:],
                    'x':torch.tensor([[.2],[.8]])+index*.1}
    d,_=fixture();dataset=Data(dori=d.dori)
    def setup():
        sampler=CompleteUIDBatchSampler(dataset,1)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0)
        model=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(1,1))
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        objective=CompleteUIDObjective(dataset,sampler,{}, {1:4,2:2})
        return loader,model,opt,objective
    def epoch(state):
        loader,model,opt,objective=state
        for data in loader:
            prediction=model(data['x']).sigmoid().squeeze(-1);y=data['y']
            loss=F.binary_cross_entropy(prediction,y)+.03*objective.rank_loss(prediction,y,torch.ones_like(y,dtype=torch.bool),data['uid'].tolist())
            opt.zero_grad();loss.backward();opt.step()
    torch.manual_seed(42);continuous=setup();epoch(continuous);epoch(continuous)
    expected=copy.deepcopy(continuous[1].state_dict());expected_rng=torch.get_rng_state()
    torch.manual_seed(42);partial=setup();epoch(partial)
    saved=copy.deepcopy({'model':partial[1].state_dict(),'optimizer':partial[2].state_dict(),'rng':torch.get_rng_state()})
    resumed=setup();resumed[1].load_state_dict(saved['model']);resumed[2].load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    epoch(resumed)
    assert all(torch.equal(expected[k],value) for k,value in resumed[1].state_dict().items())
    assert torch.equal(expected_rng,torch.get_rng_state())
