import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from test_qikt_transfer_delta import data
from consensus_prediction_delta import consensus


def test_mixture_endpoints_and_zero_amplification_preserve_fp32_scores():
    b=data();a=data();h=data();a['late_mean']=np.array([.2,.6,.3,.7,.4],np.float32).astype(float)
    h['late_mean']=np.array([.3,.5,.2,.6,.5],np.float32).astype(float)
    assert np.array_equal(consensus(b,a,h,'mixture',0).late_mean,a.late_mean)
    assert np.array_equal(consensus(b,a,h,'mixture',1).late_mean,h.late_mean)
    assert np.array_equal(consensus(b,a,h,'amplified',0).late_mean,b.late_mean)


def test_disagreement_zero_reverts_only_disagreeing_positions():
    b=data();b['late_mean']=.5;a=b.copy();h=b.copy()
    a['late_mean']=np.array([.6,.6,.4,.4,.5],np.float32).astype(float)
    h['late_mean']=np.array([.7,.4,.3,.6,.6],np.float32).astype(float)
    assert np.array_equal(consensus(b,a,h,'agreement',0).late_mean,np.array([.6,.5,.4,.5,.5],np.float32).astype(float))


@pytest.mark.parametrize('mode,amount',[('mixture',.25),('agreement',.5),('amplified',1.5)])
def test_scores_ignore_labels_and_future_predictions(mode,amount):
    b=data();a=data();h=data();a['late_mean']=a.late_mean+.05;h['late_mean']=h.late_mean-.03
    expected=consensus(b,a,h,mode,amount)
    for frame in [b,a,h]:frame.late_trues=1-frame.late_trues
    assert np.array_equal(expected.late_mean,consensus(b,a,h,mode,amount).late_mean)
    for frame in [b,a,h]:frame.loc[3:,'late_mean']=.99
    assert np.array_equal(expected.late_mean[:3],consensus(b,a,h,mode,amount).late_mean[:3])


def test_misaligned_labels_and_invalid_coefficients_fail():
    b=data();a=data();a.loc[0,'late_trues']=1
    with pytest.raises(ValueError):consensus(b,a,b,'mixture',.5)
    with pytest.raises(ValueError):consensus(b,b,b,'agreement',-1)
