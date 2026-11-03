"""Guard against current/future response leakage in the CPU mechanism probe."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]/'scripts/qikt_switch'
sys.path.insert(0,str(P))
from causal_probe import events


def row(y):
    return pd.DataFrame([dict(uid=7,questions='0,1,2,1,0',concepts='0,1,0_1,1,0',responses=','.join(map(str,y)))])


def test_prediction_features_exclude_current_and_future_responses():
    prior=np.array([-.5,.4,.8])
    original=[1,0,1,0,1]
    x,meta=events(row(original),prior,.6)
    for t in range(1,5):
        changed=original[:t]+[1-y for y in original[t:]]
        alternate,other=events(row(changed),prior,.6)
        np.testing.assert_array_equal(x[:t],alternate[:t])
        assert other.iloc[t-1].late_trues!=meta.iloc[t-1].late_trues


def test_question_features_change_when_current_question_changes():
    prior=np.array([-.5,.4,.8]);d=row([1,0,1,0,1]);x,_=events(d,prior,.6)
    d.loc[0,'questions']='0,2,2,1,0';altered,_=events(d,prior,.6)
    assert x[0,0]!=altered[0,0]
    np.testing.assert_array_equal(x[0,1:7],altered[0,1:7])
