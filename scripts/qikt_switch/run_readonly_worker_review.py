"""Bounded dsh read-only audit with saved prompt/events/exit metadata."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    prompt=a.output/'prompt.txt';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    began=time.time()
    with prompt.open('r') as inp,(a.output/'events.jsonl').open('w') as events,(a.output/'diagnostics.log').open('w') as diagnostics:
        try:
            result=subprocess.run(['/root/.local/bin/dsh','headless','--json','-'],stdin=inp,stdout=events,stderr=diagnostics,cwd=a.repo,env=env,timeout=180)
            status={'status':'completed' if result.returncode==0 else 'failed','exit_code':result.returncode}
        except subprocess.TimeoutExpired:
            status={'status':'timed_out','timeout_seconds':180}
    atomic_json(a.output/'worker_exit.json',{**status,'elapsed_seconds':time.time()-began,'prompt_sha256':digest(prompt),'model_identity':'Not assumed; inspect actual events if reported','review_not_accepted_without_primary_verification':True})


if __name__=='__main__':main()
