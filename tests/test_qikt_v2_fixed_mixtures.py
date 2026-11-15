import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd


def test_fixed_mixtures_skip_unavailable_oof_and_reuse_verified_results(tmp_path):
    root=Path(__file__).resolve().parents[1];rows=[]
    for uid in [1,2,3]:
        for i in range(30):rows.append({'orirow':uid,'qidx':i+1,'late_trues':i%2,'late_mean':.7 if i%2 else .3,'questions':str(i%3),'concepts':'0'})
    base=tmp_path/'base.tsv';pd.DataFrame(rows).to_csv(base,sep='\t',index=False)
    other=tmp_path/'other.tsv';frame=pd.DataFrame(rows);frame['late_mean']=.9*frame.late_mean+.05;frame.to_csv(other,sep='\t',index=False)
    parent=tmp_path/'parent.json';parent.write_text(json.dumps({'status':'completed'}));protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'dependencies':[str(parent)],'predictions':{'baseline':str(base),'other':str(other)},'mixtures':[{'id':'fixed','weights':{'baseline':.5,'other':.5}}],'oof_manifest':str(tmp_path/'missing_oof.json'),'family_summaries':{}}))
    output=tmp_path/'report';command=[sys.executable,str(root/'scripts/qikt_switch/run_v2_fixed_mixtures.py'),'--protocol',str(protocol),'--output-dir',str(output)]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    subprocess.run(command,check=True,capture_output=True,env=env)
    audit=json.loads((output/'oof_prerequisite_audit.json').read_text())
    assert audit['status']=='not_run_missing_oof' and not audit['learned_gate_trained']
    checkpoint=output/'fixed.json';before=checkpoint.stat().st_mtime_ns
    subprocess.run(command,check=True,capture_output=True,env=env);assert checkpoint.stat().st_mtime_ns==before
    frame['late_mean']+=.001;frame.to_csv(other,sep='\t',index=False)
    rejected=subprocess.run(command,capture_output=True,env=env)
    assert rejected.returncode!=0 and checkpoint.stat().st_mtime_ns==before
