import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
import resource_admission as module


def test_one_monitor_timeout_retries_query_without_launching_training(tmp_path,monkeypatch):
    calls=[]
    def query(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise subprocess.TimeoutExpired(args[0],10)
        return SimpleNamespace(returncode=0,stdout='0, 20000\n',stderr='')
    monkeypatch.setattr(module.subprocess,'run',query);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    module.wait_for_capacity(tmp_path/'admission.json',min_host_available_mib=0)
    assert len(calls)==2 and json.loads((tmp_path/'admission.json').read_text())['admitted']


def test_three_monitor_failures_stop_with_record_and_never_admit(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],10)
    monkeypatch.setattr(module.subprocess,'run',fail);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='Three consecutive'):module.wait_for_capacity(tmp_path/'admission.json')
    state=json.loads((tmp_path/'admission.json').read_text());assert state['consecutive_query_failures']==3 and state['admitted'] is False


def test_saved_gru_summary_does_not_inherit_hardcoded_mamba_label():
    from advance_experiment import describe_saved_architecture
    x=describe_saved_architecture({'model_config':{'version':'gru'}})
    assert x['model']=='qikt_mamba/gru' and x['architecture']=='two independent GRUs'
    assert describe_saved_architecture({'model_config':{'version':'lstm'}})['architecture']=='two independent GRUs'
