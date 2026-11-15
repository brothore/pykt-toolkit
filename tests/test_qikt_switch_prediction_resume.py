import json
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from prediction_progress import PredictionProgress
from evaluate_window import evaluate_window_checkpoint


def test_chunk_identity_and_integrity_are_verified(tmp_path):
    p=PredictionProgress(tmp_path,{'checkpoint':'a'})
    p.flush([[1,2,0,.5,'3','4']],32);p.close()
    with pytest.raises(ValueError,match='changed'):
        PredictionProgress(tmp_path,{'checkpoint':'b'},resume=True)
    chunk=tmp_path/'prediction_progress/chunk_000000.json';chunk.write_text('[]')
    with pytest.raises(ValueError,match='corrupted'):
        PredictionProgress(tmp_path,{'checkpoint':'a'},resume=True)


def test_stopped_inference_resumes_without_duplicates_or_prediction_changes(tmp_path,monkeypatch):
    from test_qikt_switch_validation_export import _write_fixture
    _,checkpoint,_,_=_write_fixture(tmp_path)
    reference=evaluate_window_checkpoint(checkpoint,tmp_path/'reference',batch_size=3,device='cpu',source='validation')
    original=PredictionProgress.flush
    def flush_then_stop(self,rows,next_window):
        original(self,rows,next_window);self.stopping=True
    monkeypatch.setattr(PredictionProgress,'flush',flush_then_stop)
    with pytest.raises(SystemExit) as stopped:
        evaluate_window_checkpoint(checkpoint,tmp_path/'resumed',3,'cpu','validation',checkpoint_batches=1)
    assert stopped.value.code==130
    state=json.loads((tmp_path/'resumed/prediction_progress/state.json').read_text())
    assert state['status']=='stopped' and state['next_window']==3
    monkeypatch.setattr(PredictionProgress,'flush',original)
    result=evaluate_window_checkpoint(checkpoint,tmp_path/'resumed',3,'cpu','validation',resume=True,checkpoint_batches=1)
    a=pd.read_csv(tmp_path/'reference/window_predictions.tsv',sep='\t')
    b=pd.read_csv(tmp_path/'resumed/window_predictions.tsv',sep='\t')
    assert a.equals(b) and result['validation_window_auc']==reference['validation_window_auc']
