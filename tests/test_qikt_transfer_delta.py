import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from transfer_prediction_delta import transfer


def data(uid=1):
    p=np.array([.1,.7,.4,.8,.3],dtype=np.float32).astype(float)
    return pd.DataFrame({'orirow':[uid]*5,'qidx':range(1,6),'late_trues':[0,1,0,1,1],
                         'questions':['0']*5,'concepts':['0']*5,'late_mean':p})


def test_zero_and_identity_transfer_keep_native_fp32_ties():
    base=data();student=base.copy();student['late_mean']=np.array([.5]*5,dtype=np.float32).astype(float)
    assert np.array_equal(transfer(base,student,base,'raw',0).late_mean,base.late_mean)
    assert np.array_equal(transfer(base,student,base,'raw',1).late_mean,student.late_mean)


def test_prefix_prediction_difference_never_reads_future_or_labels():
    base=data();student=base.copy();student['late_mean']=[.2,.6,.5,.7,.4]
    actual=transfer(base,student,base,'prefix_center',2)
    for t in range(5):
        changed=student.copy();changed.loc[t+1:,'late_mean']=.99
        again=transfer(base,changed,base,'prefix_center',2)
        assert np.array_equal(again.late_mean[:t+1],actual.late_mean[:t+1])
    b=base.copy();s=student.copy();b.late_trues=1-b.late_trues;s.late_trues=b.late_trues
    assert np.array_equal(transfer(b,s,b,'prefix_center',2).late_mean,actual.late_mean)


def test_prefix_state_isolated_between_uids():
    b=pd.concat([data(1),data(2)],ignore_index=True);s=b.copy();s['late_mean']=s.late_mean+.05
    result=transfer(b,s,b,'prefix_center',1)
    assert np.array_equal(result.late_mean[:5],result.late_mean[5:])
