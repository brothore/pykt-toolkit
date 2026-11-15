import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts/qikt_switch'


def test_native_audit_gate_rejects_failed_and_accepts_passed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('v3_family', SCRIPTS/'run_discovery_family.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    state = tmp_path/'state.json'; result = tmp_path/'result.json'
    design = {'evaluation_gate': {'state':str(state), 'result':str(result)}}
    state.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(RuntimeError, match='failed or stopped'):
        module.require_evaluation_gate(design)
    result.write_text(json.dumps({'passed':False}))
    with pytest.raises(RuntimeError, match='did not pass'):
        module.require_evaluation_gate(design)
    result.write_text(json.dumps({'passed':True}))
    module.require_evaluation_gate(design)


def test_controller_waits_for_child_safe_stop(tmp_path):
    ready = tmp_path/'ready'; saved = tmp_path/'saved'
    child = (
        'import signal,time,pathlib,sys; '
        f'p=pathlib.Path({str(saved)!r}); '
        'signal.signal(signal.SIGTERM,lambda *_:(p.write_text("checkpoint"),sys.exit(130))); '
        f'pathlib.Path({str(ready)!r}).write_text("ready"); '
        'time.sleep(20)'
    )
    parent = ('import sys; '+f'sys.path.insert(0,{str(SCRIPTS)!r}); '+
              'from cooperative_process import run_cooperative; '+
              f'run_cooperative([sys.executable,"-c",{child!r}],check=True)')
    process = subprocess.Popen([sys.executable,'-c',parent])
    try:
        deadline = time.monotonic()+8
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert ready.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8)==130
        assert saved.read_text()=='checkpoint'
    finally:
        if process.poll() is None: process.kill(); process.wait()


def test_evidence_dependency_requires_successful_artifact(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('v3_evidence', SCRIPTS/'run_after_evidence.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = tmp_path/'result.json'; state = tmp_path/'state.json'
    dep = tmp_path/'dep.json'; protocol = tmp_path/'protocol.json'; output = tmp_path/'out'
    dep.write_text(json.dumps({'artifacts':[{'result':str(result),'state':str(state),'require_passed':True,'required_values':{'status':'completed'}}]}))
    protocol.write_text('{}')
    monkeypatch.setattr(sys,'argv',['run_after_evidence','--dependency',str(dep),'--protocol',str(protocol),'--output-dir',str(output)])
    state.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(RuntimeError,match='terminated without'):
        module.main()
    result.write_text(json.dumps({'passed':False}))
    with pytest.raises(RuntimeError,match='did not pass'):
        module.main()
    calls = []; monkeypatch.setattr(module,'run_cooperative',lambda *a,**kw:calls.append(a))
    result.write_text(json.dumps({'passed':True,'status':'failed'}))
    with pytest.raises(RuntimeError,match='unexpected status'):
        module.main()
    result.write_text(json.dumps({'passed':True,'status':'completed'}))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw:'')
    monkeypatch.setattr(sys,'argv',sys.argv+['--gpu-lock',str(tmp_path/'gpu.lock')])
    module.main()
    assert len(calls)==1
    assert str(result) in json.loads((output/'dependencies_verified.json').read_text())
    assert json.loads((output/'gpu_admission.json').read_text())['status']=='exclusive_slot_released'
