import importlib
import os
from pathlib import Path
import random
import signal
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from resumable_qikt import train_qikt


class TinyQIKT(torch.nn.Module):
    model_name='qikt_mamba'
    emb_type='attn'
    def __init__(self):
        super().__init__();self.model=torch.nn.Sequential(torch.nn.Linear(2,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))


def setup(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=TinyQIKT();opt=torch.optim.Adam(model.parameters(),lr=.01)
    loader=DataLoader(TensorDataset(torch.arange(12,dtype=torch.float).reshape(6,2)/10),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(123))
    return model,opt,loader


def test_epoch_resume_matches_uninterrupted_native_training_with_rng_and_adam(tmp_path,monkeypatch):
    native=importlib.import_module('pykt.models.train_model')
    monkeypatch.setattr(native,'model_forward',lambda model,data: model.model(data[0]).square().mean())
    def evaluate(*args):
        random.random();np.random.rand();torch.rand(1)
        return .6,.5
    monkeypatch.setattr(native,'evaluate',evaluate)
    a,oa,la=setup();original=tmp_path/'native';original.mkdir();(original/'config.json').write_text('{}')
    native.train_model(a,la,la,3,oa,str(original),save_model=True)
    expected_rng=(random.random(),np.random.rand(),torch.rand(1))
    b,ob,lb=setup();folder=tmp_path/'resumed';folder.mkdir();(folder/'config.json').write_text('{}')
    def stop_after_validation(*args):
        result=evaluate(*args);signal.raise_signal(signal.SIGTERM);return result
    monkeypatch.setattr(native,'evaluate',stop_after_validation)
    with pytest.raises(SystemExit) as stopped:
        train_qikt(b,lb,lb,3,ob,str(folder),save_model=True)
    assert stopped.value.code==130
    checkpoint=torch.load(folder/'training_resume.ckpt',weights_only=False)
    assert checkpoint['completed_epochs']==1 and checkpoint['steps']==3
    monkeypatch.setattr(native,'evaluate',evaluate);monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1')
    c,oc,lc=setup(999)
    train_qikt(c,lc,lc,3,oc,str(folder),save_model=True)
    for key,value in a.state_dict().items():assert torch.equal(value,c.state_dict()[key]),key
    actual_rng=(random.random(),np.random.rand(),torch.rand(1))
    assert expected_rng[:2]==actual_rng[:2] and torch.equal(expected_rng[2],actual_rng[2])
    assert torch.load(folder/'training_resume.ckpt',weights_only=False)['finished']
    # Completed re-entry performs no optimization.
    before={k:v.clone() for k,v in c.state_dict().items()};train_qikt(c,lc,lc,3,oc,str(folder),save_model=True)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in before.items())


def test_raw_epoch_metrics_distinguish_ignored_improvements_and_recover_view(tmp_path,monkeypatch):
    native=importlib.import_module('pykt.models.train_model')
    monkeypatch.setattr(native,'model_forward',lambda model,data: model.model(data[0]).square().mean())
    scores=iter([.7,.7005,.699])
    monkeypatch.setattr(native,'evaluate',lambda *args:(next(scores),.5))
    model,opt,loader=setup();folder=tmp_path/'run';folder.mkdir();(folder/'config.json').write_text('{}')
    train_qikt(model,loader,loader,3,opt,str(folder),save_model=True)
    import json
    metrics=json.loads((folder/'epoch_metrics.json').read_text())
    assert [row['validation_auc'] for row in metrics]==[.7,.7005,.699]
    assert [row['best_validation_auc'] for row in metrics]==[.7,.7,.7]
    assert [row['selected'] for row in metrics]==[True,False,False]
    assert [row['selected_epoch'] for row in metrics]==[1,1,1]
    checkpoint=torch.load(folder/'training_resume.ckpt',weights_only=False)
    assert checkpoint['epoch_metrics']==metrics
    (folder/'epoch_metrics.json').unlink()
    monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1')
    train_qikt(model,loader,loader,3,opt,str(folder),save_model=True)
    assert json.loads((folder/'epoch_metrics.json').read_text())==metrics
