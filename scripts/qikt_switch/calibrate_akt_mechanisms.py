"""Bounded single/dual throughput of equal native AKT work on a real batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from akt_experiment_io import build_model, loaders, predict
from reference_training import to_device
from prediction_progress import atomic_json


def worker(config, output, steps, barrier):
    torch.set_num_threads(2)
    model = build_model(config); data, gaps = to_device(next(iter(loaders(config)[0])), 'cuda')
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    def step():
        opt.zero_grad(); p, reg = predict(model, data, gaps); m = data['smasks'].bool()
        loss = torch.nn.functional.binary_cross_entropy(p[:, 1:][m].double(), data['shft_rseqs'][m].double()) + reg
        loss.backward(); opt.step()
    for _ in range(3): step()
    torch.cuda.synchronize()
    output.with_suffix('.ready').touch()
    while not barrier.exists(): time.sleep(.02)
    start = time.perf_counter()
    for _ in range(steps): step()
    torch.cuda.synchronize()
    atomic_json(output, dict(seconds=time.perf_counter()-start, steps=steps,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--steps',type=int,default=120);p.add_argument('--barrier',type=Path);a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    if a.worker: worker(cfg,a.output,a.steps,a.barrier);return
    from container_memory_budget import container_budget
    a.output.mkdir(parents=True,exist_ok=True);cells=[]
    for repeat in [1,2]:
        for concurrency in [1,2]:
            processes=[];start=time.perf_counter();barrier=a.output/f'{repeat}_{concurrency}.start'
            if barrier.exists():barrier.unlink()
            for i in range(concurrency):
                path=a.output/f'{repeat}_{concurrency}_{i}.json';log=path.with_suffix('.log').open('w')
                if path.with_suffix('.ready').exists():path.with_suffix('.ready').unlink()
                cmd=[sys.executable,__file__,'--config',str(a.config),'--output',str(path),'--worker','--steps',str(a.steps//concurrency),'--barrier',str(barrier)]
                processes.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),path,log))
            while not all(path.with_suffix('.ready').exists() for _,path,_ in processes):
                if any(proc.poll() is not None for proc,_,_ in processes):raise RuntimeError('Calibration worker failed before barrier')
                time.sleep(.1)
            barrier.touch()
            for proc,path,log in processes:
                rc=proc.wait();log.close()
                if rc:raise RuntimeError(f'Calibration failed: {path}, exit {rc}')
            values=[json.loads(path.read_text()) for _,path,_ in processes]
            cells.append(dict(repeat=repeat,concurrency=concurrency,workers=values,
                effective_batch_per_second=a.steps/max(v['seconds'] for v in values),
                elapsed_with_load=time.perf_counter()-start,container=container_budget()))
            atomic_json(a.output/'cells.json',cells)
    means={c:sum(v['effective_batch_per_second'] for v in cells if v['concurrency']==c)/2 for c in [1,2]}
    selected=min(c for c in means if means[c]>=.95*max(means.values()))
    atomic_json(a.output/'ready.json',dict(passed=True,selected=selected,mean_batch_per_second=means,
        limitation='Repeated real training batch; synchronized start and equal total measured batches/cell, no validation/window inference calibration. No universal optimum claim.'))


if __name__=='__main__':main()
