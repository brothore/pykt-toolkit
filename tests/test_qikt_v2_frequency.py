from types import SimpleNamespace
import torch
from pykt.models.qikt_frequency_objective import FrequencyObjective


def dataset(flip=False):
    ids=torch.tensor([0]*20+[1]*8+[2]*2).reshape(5,6)
    concepts=torch.cat([torch.zeros(5,1,dtype=torch.long),ids],1).unsqueeze(-1).repeat(1,1,2)
    labels=torch.tensor([[0,1,0,1,0,1,0]]*5)
    if flip:labels=1-labels
    return SimpleNamespace(dori={'rseqs':labels,'smasks':torch.ones(5,7,dtype=torch.long),'cseqs':concepts})


def test_frequency_is_label_independent_counts_duplicate_kcs_once_and_normalizes_globally():
    a=FrequencyObjective(dataset());b=FrequencyObjective(dataset(True))
    assert a.counts.tolist()==[20,8,2]
    assert torch.equal(a.weights,b.weights) and a.normalizer==b.normalizer
    assert a.weights[2]>a.weights[0] and float(a.weights.max())<=2
    context=a.event_weights(dataset().dori['cseqs'][:,1:,:])
    assert torch.allclose((context/a.normalizer).mean(),torch.tensor(1.,dtype=torch.double))


def test_cap_one_matches_unweighted_bce_and_gradient():
    data=dataset();a=FrequencyObjective(data,cap=1)
    a.context=a.event_weights(data.dori['cseqs'][:,1:,:])
    p=torch.full((5,6),.6,dtype=torch.double,requires_grad=True);labels=data.dori['rseqs'][:,1:].double();mask=torch.ones_like(labels,dtype=torch.bool)
    weighted=a.weighted_bce(p,labels,mask);native=torch.nn.functional.binary_cross_entropy(p,labels)
    assert torch.equal(weighted,native)
    ga,=torch.autograd.grad(weighted,p,retain_graph=True);gb,=torch.autograd.grad(native,p)
    assert torch.equal(ga,gb)
