import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from sequence_balance_metrics import assess_balance


def paired(after):
    before=np.array([.5,.6,.8,.9]);after=np.array(after)
    return pd.DataFrame({'n':[100]*4,'minority':[20]*4,'baseline_auc':before,'auc':after,'delta_auc':after-before},index=[10,20,30,40])


def test_rejects_variance_reduction_obtained_by_harming_strong_sequences():
    result=assess_balance(paired([.6,.7,.8,.8]),.001,bootstrap=40)
    assert result['deltas']['std_ddof0']<0 and result['deltas']['range']<0
    assert not result['checks']['strong_preserved'] and not result['exploratory_candidate_pass']
    assert result['worsened']==1 and not result['strict_goal_pass']


def test_fixed_and_own_tail_improvement_with_strong_preservation():
    result=assess_balance(paired([.6,.65,.8,.9]),.001,bootstrap=40)
    assert result['exploratory_candidate_pass'] and result['strict_goal_pass']
    assert result['deltas']['fixed_baseline_bottom10_mean']>0
    assert result['deltas']['own_bottom10_mean']>0 and result['worsened']==0
    assert result['baseline_min_orirow']=='10'
