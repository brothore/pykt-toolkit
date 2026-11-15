import importlib.util
import json
from pathlib import Path
import sys


def test_queued_jobs_without_worker_state_are_not_failed_or_terminal(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts/qikt_switch'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('review_batch_queue_test',scripts/'review_discovery_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.job_status(tmp_path,'U1')=='queued'
    p=tmp_path/'jobs/U1';p.mkdir(parents=True)
    for status in ['running','completed','failed','stopped']:
        (p/'state.json').write_text(json.dumps({'status':status}))
        assert module.job_status(tmp_path,'U1')==status
