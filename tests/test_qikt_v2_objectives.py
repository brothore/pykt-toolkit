from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import importlib
import signal
import pytest
import torch

from pykt.models.qikt_sequence_objectives import SequenceObjective,install_training_objective
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_qikt_switch_frozen import model,batch


def dataset():
    d=SimpleNamespace(aug_probs={},dori={'uid':torch.tensor([1,1,2]),
        'rseqs':torch.tensor([[0,1,0,-1],[1,0,1,-1],[1,0,-1,-1]]),
        'smasks':torch.tensor([[1,1,1,-1],[1,1,1,-1],[1,1,-1,-1]])})
    class View:
        aug_probs=d.aug_probs;dori=d.dori
        def __len__(self):return 3
    return View()


def test_complete_uid_bce_has_equal_total_mass_despite_multiple_fragments():
    objective=SequenceObjective(dataset(),alpha=1);objective.uids=[1,1,2]
    p=torch.tensor([[.8,.2,float('nan')],[.2,.8,float('nan')],[.2,float('nan'),float('nan')]],requires_grad=True)
    y=torch.tensor([[1.,0.,-1.],[0.,1.,-1.],[0.,-1.,-1.]]);mask=y!=-1
    loss=objective.equal_sequence_bce(p,y,mask)
    assert torch.allclose(loss,-torch.log(torch.tensor(.8,dtype=torch.double)))
    loss.backward()
    assert torch.allclose(p.grad[:2].nan_to_num().abs().sum(),p.grad[2].nan_to_num().abs().sum())
    assert objective.counts=={1:4,2:1}


def test_alpha_zero_preserves_native_loss_rng_and_gradient_exactly():
    a=model();b=deepcopy(a);sample=batch();sample['uid']=torch.tensor([1])
    install_training_objective(b,dataset(),{'alpha':0.,'rank_weight':0.})
    torch.manual_seed(100);pa,la=a.train_one_step(sample,process=False);la.backward();rng=torch.get_rng_state()
    torch.manual_seed(100);pb,lb=b.train_one_step(sample,process=False);lb.backward()
    assert torch.equal(pa,pb) and torch.equal(la,lb) and torch.equal(rng,torch.get_rng_state())
    for aa,bb in zip(a.parameters(),b.parameters()):
        if aa.grad is not None:assert torch.equal(aa.grad,bb.grad)


def test_sequence_pair_loss_ignores_single_class_and_has_correct_direction():
    objective=SequenceObjective(dataset(),rank_weight=.01);objective.uids=[1,1,2]
    y=torch.tensor([[1,0],[0,1],[0,0]]);mask=torch.ones_like(y,dtype=torch.bool)
    good=torch.tensor([[.9,.1],[.1,.9],[.1,.2]],requires_grad=True)
    bad=1-good
    assert objective.sequence_weighted_fragment_rank(good,y,mask)<objective.sequence_weighted_fragment_rank(bad,y,mask)
    objective.sequence_weighted_fragment_rank(good,y,mask).backward()
    assert torch.equal(good.grad[2],torch.zeros(2))


def test_v2_training_epoch_resume_restores_objective_and_optimizer(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
    from resumable_qikt import train_qikt
    native=importlib.import_module('pykt.models.train_model')
    class Loader:
        num_workers=0;generator=None
        def __iter__(self):
            for uid in [1,1,2]:
                d=batch();d['uid']=torch.tensor([uid]);yield d
        def __len__(self):return 3
    # SequenceObjective needs a normal dataset length, not a loader's batch count.
    class Data:
        aug_probs={}
        dori={'uid':torch.tensor([1,1,2]),'rseqs':torch.tensor([[1,0,1,0,1]]*3),'smasks':torch.ones(3,5,dtype=torch.long)}
        def __len__(self):return 3
    loader=Loader();loader.dataset=Data()
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'alpha':.25,'rank_weight':.01}))
    monkeypatch.setenv('QIKT_V2_PROTOCOL',str(protocol));monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1')
    monkeypatch.setattr(native,'model_forward',lambda m,d:m.train_one_step(d,process=False)[1])
    monkeypatch.setattr(native,'evaluate',lambda *args:(.6,.5))
    a=model();oa=torch.optim.Adam(a.parameters(),lr=.001);folder=tmp_path/'full';folder.mkdir();(folder/'config.json').write_text('{}')
    train_qikt(a,loader,loader,3,oa,str(folder),save_model=True)
    b=model();ob=torch.optim.Adam(b.parameters(),lr=.001);resumed=tmp_path/'resumed';resumed.mkdir();(resumed/'config.json').write_text('{}')
    def stop(*args):signal.raise_signal(signal.SIGTERM);return .6,.5
    monkeypatch.setattr(native,'evaluate',stop)
    with pytest.raises(SystemExit):train_qikt(b,loader,loader,3,ob,str(resumed),save_model=True)
    monkeypatch.setattr(native,'evaluate',lambda *args:(.6,.5));c=model();oc=torch.optim.Adam(c.parameters(),lr=.001)
    train_qikt(c,loader,loader,3,oc,str(resumed),save_model=True)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
