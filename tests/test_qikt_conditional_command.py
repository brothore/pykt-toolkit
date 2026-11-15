import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/qikt_switch/run_conditional_command.py'


def spec(tmp_path, status='completed', signal_found=False):
    state=tmp_path/'state.json';state.write_text(json.dumps({'status':status}))
    root=tmp_path/'upstream';folder=root/'method/round1';folder.mkdir(parents=True)
    (folder/'balance.json').write_text(json.dumps({'substantive_development_signal':signal_found}))
    result=tmp_path/'result.json';marker=tmp_path/'marker'
    value={'dependencies':[str(state)],'signal_roots':[str(root)],'result':str(result),
           'cwd':str(tmp_path),'command':[sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).write_text("executed")']}
    path=tmp_path/'spec.json';path.write_text(json.dumps(value))
    return path,result,marker


def run(path):
    return subprocess.run([sys.executable,str(SCRIPT),'--spec',str(path)],capture_output=True,text=True,
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=10)


def test_completed_dependency_runs_fixed_command(tmp_path):
    path,_,marker=spec(tmp_path)
    assert run(path).returncode==0
    assert marker.read_text()=='executed'


def test_substantive_signal_skips_command_with_source_identity(tmp_path):
    path,result,marker=spec(tmp_path,signal_found=True)
    assert run(path).returncode==0
    assert not marker.exists()
    value=json.loads(result.read_text())
    assert value['executed'] is False and len(value['signals'][0]['sha256'])==64


def test_failed_dependency_never_runs_command(tmp_path):
    path,_,marker=spec(tmp_path,status='failed')
    assert run(path).returncode!=0
    assert not marker.exists()


def test_waiting_task_stops_without_touching_upstream(tmp_path):
    path,_,marker=spec(tmp_path,status='running')
    process=subprocess.Popen([sys.executable,str(SCRIPT),'--spec',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    time.sleep(.2);process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=5)==130
    assert not marker.exists()
    assert json.loads((tmp_path/'state.json').read_text())['status']=='running'
