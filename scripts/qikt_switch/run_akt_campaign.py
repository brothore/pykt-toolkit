"""One bounded AKT family per tsp job; fail closed on unfinished prerequisites."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prediction_progress import atomic_json, digest

SCALES = {'baseline': [0.], 'ukt_cov': [.1,.3,1.], 'mtkt_cic': [.1,.3,1.],
          'mtkt_time': [.1,.5,1.], 'lef_additive': [.1,.03,.3]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',choices=SCALES,required=True)
    a=p.parse_args();root=a.root.resolve();family=a.family;folder=root/'families'/family;folder.mkdir(parents=True,exist_ok=True)
    baseline=root/'runs/baseline_r1';script=Path(__file__).resolve().parent
    state=dict(family=family,pid=os.getpid(),started=time.time(),status='running',rounds=[],tsp_socket=os.getenv('TS_SOCKET'))
    if (folder/'state.json').exists():
        old=json.loads((folder/'state.json').read_text());state['previous_state']=old
        if old['status']=='completed':return
    atomic_json(folder/'state.json',state)
    try:
        manifest=json.loads((root/'source_manifest.json').read_text())
        for source,sha256 in manifest.items():
            if digest(Path(source))!=sha256:raise ValueError(f'Campaign source changed: {source}; preserve run and issue a versioned fix')
        ready=json.loads((root/'calibration_sync/ready.json').read_text())
        if not ready['passed']:raise ValueError('Calibration failed')
        if not json.loads((root/'preflight.json').read_text())['passed']:raise ValueError('Native preflight failed')
        if family!='baseline' and not (baseline/'strict_validation/result.json').exists():raise ValueError('Fresh baseline incomplete')
        cfg=json.loads((root/'baseline_config.json').read_text());config_identity=digest(root/'baseline_config.json')
        for round_number,scale in enumerate(SCALES[family],1):
            run=root/'runs'/f'{family}_r{round_number}';run.mkdir(parents=True,exist_ok=True)
            config={**cfg,'mechanism':family,'scale':scale};cp=run/'planned_config.json'
            if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Planned config changed')
            atomic_json(cp,config)
            # tsp concurrency is selected by measured equal-work throughput.
            # A second lock prevents manual/resume invocations duplicating a run.
            with (run/'.run.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                from container_memory_budget import container_budget
                budget=container_budget()
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                atomic_json(run/'admission.json',dict(container=budget,gpu_free_mib=free,
                    required_gpu_headroom_mib=6144,required_conservative_host_gib=6,measured_concurrency=ready['selected']))
                if budget['conservative_available_gib']<10 or free<11000:raise RuntimeError('Insufficient headroom for AKT peak plus safety budget')
                cmd=[sys.executable,str(script/'train_akt_mechanism.py'),'--config',str(cp),'--output',str(run)]
                atomic_json(run/'task.json',dict(parent_family=family,round=round_number,command=cmd,
                    tsp_socket=os.getenv('TS_SOCKET'),baseline_config_sha256=config_identity,started=time.time()))
                subprocess.run(cmd,check=True)
                if not (run/'strict_validation/result.json').exists():
                    cmd=[sys.executable,str(script/'evaluate_akt_mechanism.py'),'--run',str(run)]
                    if family!='baseline':cmd.extend(['--baseline',str(baseline)])
                    subprocess.run(cmd,check=True)
            result=json.loads((run/'strict_validation/result.json').read_text())
            candidate=result.get('balance',{}).get('exploratory_candidate_pass',False)
            state['rounds'].append(dict(round=round_number,scale=scale,run=str(run),candidate=candidate))
            atomic_json(folder/'state.json',state)
            subprocess.run([sys.executable,str(script/'report_akt_campaign.py'),'--root',str(root)],check=True)
            if candidate:
                state['stop_reason']='First exploratory candidate found; remaining scales paused';break
        state.update(status='completed',ended=time.time())
        atomic_json(folder/'state.json',state)
        subprocess.run([sys.executable,str(script/'report_akt_campaign.py'),'--root',str(root)],check=True)
    except BaseException as e:
        state.update(status='stopped' if isinstance(e,subprocess.CalledProcessError) and e.returncode==130 else 'failed',error=repr(e),ended=time.time())
        atomic_json(folder/'state.json',state);raise


if __name__=='__main__':main()
