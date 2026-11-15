import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_completed_queue_task_reuses_verified_output_and_rejects_changed_input(tmp_path):
    root=Path(__file__).resolve().parents[1];source=tmp_path/'input.txt';source.write_text('frozen')
    output=tmp_path/'result.json';plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'task_id':'test','command':[sys.executable,'-c',f'from pathlib import Path; Path({str(output)!r}).write_text("{{}}")'],
        'cwd':str(root),'resources':{'gpu':False},'input_sha256':{str(source):hashlib.sha256(source.read_bytes()).hexdigest()},'source_sha256':{},'outputs':[str(output)]}))
    command=[sys.executable,str(root/'scripts/qikt_switch/queued_task.py'),'--plan',str(plan),'--resume']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    subprocess.run(command,check=True,env=env,capture_output=True)
    before=output.stat().st_mtime_ns
    subprocess.run(command,check=True,env=env,capture_output=True)
    assert output.stat().st_mtime_ns==before
    source.write_text('changed')
    failed=subprocess.run(command,env=env,capture_output=True)
    assert failed.returncode!=0 and output.stat().st_mtime_ns==before
    assert json.loads((tmp_path/'state.json').read_text())['status']=='failed'
