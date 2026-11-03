import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pykt.datasets.que_data_loader import KTQueDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from evaluate_window import QuestionWindows


def test_streamed_windows_equal_native_loader_and_score_each_target_once(tmp_path):
    raw=pd.DataFrame([dict(uid=4,fold=-1,questions='0,1,2,3,4,5,6',concepts='0,1,0_1,1,0,1,0',responses='1,0,1,0,1,0,1',timestamps='0,1,2,3,4,5,6'),dict(uid=5,fold=-1,questions='0,1,2',concepts='0,1,0',responses='0,1,0',timestamps='0,1,2')])
    streamed=QuestionWindows(raw,length=4,max_concepts=2);native_rows=[]
    for r in raw.itertuples():
        fields={k:getattr(r,k).split(',') for k in ['questions','concepts','responses','timestamps']};n=len(fields['questions'])
        for stop in [min(n,4),*range(5,n+1)]:
            start=max(0,stop-4);size=stop-start;first=start==0
            row=dict(uid=r.uid,fold=-1)
            for k,v in fields.items():row[k]=','.join(v[start:stop]+['-1']*(4-size))
            selected=[1]*size if first else [-1]*(size-1)+[1]
            row['selectmasks']=','.join(map(str,selected+[-1]*(4-size)));native_rows.append(row)
    path=tmp_path/'native_windows.csv';pd.DataFrame(native_rows).to_csv(path,index=False)
    native=KTQueDataset(str(path),['questions','concepts'],{-1},2,2,mode='test')
    assert len(streamed)==len(native)
    scored=[]
    for i in range(len(native)):
        a,meta=streamed[i];b=native[i]
        for k in ['qseqs','shft_qseqs','cseqs','shft_cseqs','rseqs','shft_rseqs','tseqs','shft_tseqs','masks','smasks']:
            assert torch.equal(a[k],b[k]),k
        scored.extend((meta['orirow'],int(x)) for x in meta['qidx'][a['smasks']])
    assert scored==[(4,i) for i in range(1,7)]+[(5,1),(5,2)]


def test_window_metric_is_recomputed_from_original_question_predictions(tmp_path):
    import json
    from test_qikt_switch_validation_export import _write_fixture
    from evaluate_window import evaluate_window_checkpoint
    data,checkpoint,raw,_=_write_fixture(tmp_path)
    raw.to_csv(data/'test_quelevel.csv',index=False)
    result=evaluate_window_checkpoint(checkpoint,tmp_path/'window_export',batch_size=3,device='cpu')
    predictions=pd.read_csv(tmp_path/'window_export/window_predictions.tsv',sep='\t')
    expected=sum(len(v.split(','))-1 for v in raw.responses)
    assert len(predictions)==expected and result['events']==expected
    assert not predictions.duplicated(['orirow','qidx']).any()
    from sklearn.metrics import roc_auc_score
    actual=json.loads((tmp_path/'window_export/all_results.json').read_text())
    assert np.isclose(actual['windowauclate_mean'],roc_auc_score(predictions.late_trues,predictions.late_mean),atol=1e-14)
    assert actual['cohorts']['primary_20_5']['N']==1
    # Changing the final observed answer must not change its own prediction.
    last_uid=int(raw.uid.iloc[-1]);before=predictions[predictions.orirow==last_uid].iloc[-1].late_mean
    row=raw.iloc[-1].copy();labels=row.responses.split(',');labels[-1]=str(1-int(labels[-1]));raw.loc[raw.index[-1],'responses']=','.join(labels);raw.to_csv(data/'test_quelevel.csv',index=False)
    evaluate_window_checkpoint(checkpoint,tmp_path/'changed',batch_size=3,device='cpu')
    after=pd.read_csv(tmp_path/'changed/window_predictions.tsv',sep='\t');after=after[after.orirow==last_uid].iloc[-1].late_mean
    assert before==after


def test_validation_windows_do_not_read_test_data_or_emit_test_metric(tmp_path):
    from test_qikt_switch_validation_export import _write_fixture
    from evaluate_window import evaluate_window_checkpoint
    data,checkpoint,raw,_=_write_fixture(tmp_path)
    assert not (data/'test_quelevel.csv').exists()
    result=evaluate_window_checkpoint(checkpoint,tmp_path/'validation_window',batch_size=3,device='cpu',source='validation')
    assert 'validation_window_auc' in result and 'windowauclate_mean' not in result
    assert result['test_evaluated'] is False
    assert (tmp_path/'validation_window/validation_window_results.json').exists()
    assert not (tmp_path/'validation_window/all_results.json').exists()
    assert result['events']==sum(len(s.split(','))-1 for s in raw.responses)


def test_single_class_sequences_are_not_assigned_artificial_auc():
    from audit_inputs import sequence_metrics
    frame=pd.DataFrame(dict(orirow=[1,1,2,2],qidx=[1,2,1,2],late_trues=[1,1,0,0],late_mean=[.2,.7,.1,.8]))
    cohorts,rows=sequence_metrics(frame)
    assert rows.empty
    assert all(v['N']==0 and v['mean'] is None and v['std'] is None and v['range'] is None for v in cohorts.values())


def test_fixed_mixture_replays_identical_components_and_checks_weights(tmp_path):
    import json
    from test_qikt_switch_validation_export import _write_fixture
    from evaluate_window import evaluate_window_checkpoint
    _,checkpoint,_,_=_write_fixture(tmp_path)
    single=evaluate_window_checkpoint(checkpoint,tmp_path/'single',batch_size=3,device='cpu',source='validation')
    spec=tmp_path/'mixture.json';spec.write_text(json.dumps({'components':[{'checkpoint_dir':str(checkpoint),'weight':.25},{'checkpoint_dir':str(checkpoint),'weight':.75}]}))
    combined=evaluate_window_checkpoint(checkpoint,tmp_path/'mixture',batch_size=3,device='cpu',source='validation',mixture_path=spec)
    assert combined['model']=='qikt_mamba_prediction_mixture'
    assert combined['validation_window_auc']==single['validation_window_auc']
    a=pd.read_csv(tmp_path/'single/window_predictions.tsv',sep='\t');b=pd.read_csv(tmp_path/'mixture/window_predictions.tsv',sep='\t')
    assert a.equals(b)
    spec.write_text(json.dumps({'components':[{'checkpoint_dir':str(checkpoint),'weight':.2}]}))
    import pytest
    with pytest.raises(ValueError,match='sum to one'):
        evaluate_window_checkpoint(checkpoint,tmp_path/'invalid',device='cpu',source='validation',mixture_path=spec)
