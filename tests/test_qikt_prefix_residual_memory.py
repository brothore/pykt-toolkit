import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from prefix_residual_memory import calibrate_frame


def frame(uid=1):
    return pd.DataFrame({'orirow':[uid]*6,'qidx':range(1,7),'late_mean':[.5]*6,
                         'late_trues':[1,1,0,0,1,0],'questions':['1']*6,'concepts':['0,0']*6})


@pytest.mark.parametrize('mode',['global','kc','bucket','mixed','joint'])
def test_current_and_future_labels_never_change_current_or_past_outputs(mode):
    original=frame();actual,_=calibrate_frame(original,mode,1.,1.)
    for t in range(len(original)):
        altered=original.copy();altered.loc[t:,'late_trues']=1-altered.loc[t:,'late_trues']
        changed,_=calibrate_frame(altered,mode,1.,1.)
        assert np.array_equal(actual.late_mean[:t+1],changed.late_mean[:t+1])


def test_kcs_are_deduplicated_and_uids_do_not_share_state():
    a=frame();b=frame(2);both,_=calibrate_frame(pd.concat([a,b],ignore_index=True),'kc',1.,1.)
    single,_=calibrate_frame(a,'kc',1.,1.)
    assert np.array_equal(both.late_mean[:6],both.late_mean[6:])
    assert np.array_equal(single.late_mean,both.late_mean[:6])
    a['concepts']='0';dedup,_=calibrate_frame(a,'kc',1.,1.)
    assert np.array_equal(single.late_mean,dedup.late_mean)


def test_memory_evicts_old_positions_and_score_uses_only_past():
    actual,meta=calibrate_frame(frame(),'global',1.,1.,memory=1)
    assert actual.late_mean.iloc[0]==.5
    assert np.isclose(actual.late_mean.iloc[1],1/(1+np.exp(-.5/1.25)))
    assert np.isclose(actual.late_mean.iloc[3],1/(1+np.exp(.5/1.25)))
    assert meta['maximum_prior_events_used']==1


def test_unseen_kc_has_no_adjustment_and_zero_errors_keep_scores():
    a=frame();a.loc[1:,'concepts']='1';actual,_=calibrate_frame(a,'kc',1.,1.)
    assert actual.late_mean.iloc[1]==.5
    a['late_mean']=a.late_trues.astype(float)
    actual,_=calibrate_frame(a,'global',1.,1.)
    assert np.allclose(actual.late_mean,np.clip(a.late_mean,1e-6,1-1e-6))
