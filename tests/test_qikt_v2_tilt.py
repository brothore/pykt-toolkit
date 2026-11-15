import importlib
import json
from pathlib import Path
import signal
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from pykt.models.qikt_tilt_objective import bounded_tilt_weights
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))


def test_tilt_keeps_mean_one_strict_bounds_and_increases_attention_monotonically():
    weights=bounded_tilt_weights([.1,.5,1.,1000.],.5,2.)
    assert np.isclose(weights.mean(),1.,atol=1e-12)
    assert weights.min()>=.5 and weights.max()<=2 and np.all(np.diff(weights)>=0)
    assert np.allclose(bounded_tilt_weights([1,2,3],0),1)


def test_tilt_epoch_measurement_and_resume_restore_rng_sampler_and_adam(tmp_path,monkeypatch):
    from test_qikt_switch_validation_export import _write_fixture
    from pykt.datasets.que_data_loader import KTQueDataset
    from pykt.models.qikt_mamba import QIKT_MAMBA
    from resumable_qikt import train_qikt
    (tmp_path/'fixture').mkdir();data,checkpoint,_,_=_write_fixture(tmp_path/'fixture')
    cfg=json.loads((checkpoint/'config.json').read_text());model_config=cfg['model_config'];model_config['dropout']=.2
    dataset=KTQueDataset(str(data/'train_valid_sequences_quelevel.csv'),['questions','concepts'],{0},5,2,mode='train')
    def setup():
        torch.manual_seed(42);m=QIKT_MAMBA(num_q=10,num_c=5,**model_config)
        return m,torch.optim.Adam(m.parameters(),lr=.001),DataLoader(dataset,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(123))
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'tilt_mix':.25,'tilt_temperature':.5,'tilt_cap':2}))
    monkeypatch.setenv('QIKT_V2_PROTOCOL',str(protocol));monkeypatch.setenv('QIKT_SWITCH_TRAIN_RESUME','1')
    native=importlib.import_module('pykt.models.train_model');monkeypatch.setattr(native,'evaluate',lambda *a:(.6,.5))
    full=tmp_path/'full';full.mkdir();(full/'config.json').write_text('{}');a,oa,la=setup();train_qikt(a,la,la,3,oa,str(full),save_model=True)
    resumed=tmp_path/'resumed';resumed.mkdir();(resumed/'config.json').write_text('{}');b,ob,lb=setup()
    def stop(*a):signal.raise_signal(signal.SIGTERM);return .6,.5
    monkeypatch.setattr(native,'evaluate',stop)
    with pytest.raises(SystemExit):train_qikt(b,lb,lb,3,ob,str(resumed),save_model=True)
    monkeypatch.setattr(native,'evaluate',lambda *a:(.6,.5));c,oc,lc=setup();train_qikt(c,lc,lc,3,oc,str(resumed),save_model=True)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    for epoch in [1,2,3]:
        x=json.loads((full/f'tilt_weights_epoch{epoch:03d}.json').read_text());y=json.loads((resumed/f'tilt_weights_epoch{epoch:03d}.json').read_text())
        assert x['uid_weights']==y['uid_weights'] and x['uid_risks']==y['uid_risks']
