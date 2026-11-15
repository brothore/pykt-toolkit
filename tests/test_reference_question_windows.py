import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from reference_question_windows import CausalReferenceWindows,gap_categories


def raw(responses='0,1,0,1'):
    return pd.DataFrame([{'uid':7,'questions':'1,2,3,4','concepts':'0,1_2,0,2','responses':responses,'timestamps':'0,60000,120000,180000'}])


def test_current_question_other_kc_response_cannot_leak():
    a=CausalReferenceWindows(raw());b=CausalReferenceWindows(raw('0,0,1,0'))
    for i in [0,1]:
        data,gaps,meta=a[i];changed,other,cm=b[i]
        assert meta['qidx']==1
        assert data['shft_rseqs'][meta['target_index']-1]==0
        assert all(torch.equal(v,changed[k]) for k,v in data.items())
        assert all(torch.equal(v,other[k]) for k,v in gaps.items())
        assert data['masks'].sum()==1  # first question history, one current KC


def test_counts_targets_and_gap_context_bounds():
    d=CausalReferenceWindows(raw(),length=3)
    assert len(d)==4
    for i in range(len(d)):
        data,gaps,meta=d[i]
        assert data['smasks'].sum()==1
        assert 1<=meta['target_index']<=2
        assert data['shft_rseqs'][meta['target_index']-1]==0
    assert gap_categories([0,0,1],[0,60000,120000]).tolist()==[[0,0,0],[2,2,1],[0,2,0]]
