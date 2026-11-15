"""Bounded, resumable measurement cells for native QIKT concurrency."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from prediction_progress import atomic_json, digest
from container_memory_budget import container_budget


def allocation(total, count):
    if total % count: raise ValueError('Work must divide evenly across workers')
    return [list(range(i,total,count)) for i in range(count)]


def shortlist(measurements):
    grouped = {}
    for m in measurements: grouped.setdefault(m['concurrency'],[]).append(m['steps_per_second'])
    medians = {c:statistics.median(v) for c,v in grouped.items()}
    best = max(medians.values())
    return min(c for c,value in medians.items() if value>=.95*best), medians


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    protocol=json.loads(args.protocol.read_text()); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    scripts=Path(__file__).resolve().parent
    identity={'protocol':digest(args.protocol),'sources':{str(p):digest(p) for p in [Path(__file__),scripts/'resource_benchmark_worker.py']}}
    ip=out/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity: raise ValueError('Calibration identity changed')
    atomic_json(ip,identity)
    children=[]; stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        for child in children:
            if child.poll() is None: child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def gpu_pids():
        x=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        return {int(line.strip()) for line in x.splitlines() if line.strip().isdigit()}
    def sample():
        x=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
        utilization,used,free,total=map(int,x.strip().split(','))
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
        io={}
        for child in children:
            path=Path(f'/proc/{child.pid}/io')
            try:io[str(child.pid)]={k:int(v) for k,v in (line.split(':') for line in path.read_text().splitlines())}
            except (FileNotFoundError,ProcessLookupError):pass
        return {'clock':time.perf_counter(),'gpu_utilization':utilization,'used_mib':used,'free_mib':free,
                'total_mib':total,'host_available_mib':mem['MemAvailable']/1024,'loadavg':os.getloadavg(),'worker_io':io,
                **container_budget()}
    def cell(name,workloads,repeat,batches):
        nonlocal children
        folder=out/name/f'repeat{repeat}'; result=folder/'result.json'
        if result.exists():return json.loads(result.read_text())
        if stopping:raise SystemExit(130)
        if gpu_pids():raise RuntimeError('Foreign GPU workload present; defer calibration without stopping it')
        folder.mkdir(parents=True,exist_ok=True); release=folder/'release.json'
        # Incomplete cells are fresh remeasurements, not optimizer resumption.
        if release.exists():release.unlink()
        children=[]; handles=[]; started=time.perf_counter()
        try:
            for i,(workload,assigned) in enumerate(zip(workloads,batches)):
                worker=folder/f'worker{i}';worker.mkdir(exist_ok=True)
                for stale in ['ready.json','result.json']:
                    if (worker/stale).exists():(worker/stale).unlink()
                spec=dict(protocol['worker'],workload=workload,worker=i,batches=assigned,release=str(release))
                atomic_json(worker/'spec.json',spec);handle=(worker/'execution.log').open('a');handles.append(handle)
                env=dict(os.environ,OMP_NUM_THREADS=str(spec['threads']),MKL_NUM_THREADS=str(spec['threads']),OPENBLAS_NUM_THREADS=str(spec['threads']),CUBLAS_WORKSPACE_CONFIG=':4096:2')
                children.append(subprocess.Popen([sys.executable,str(scripts/'resource_benchmark_worker.py'),'--spec',str(worker/'spec.json')],env=env,stdout=handle,stderr=subprocess.STDOUT))
            atomic_json(folder/'processes.json',{'pids':[c.pid for c in children],'command_source':str(scripts/'resource_benchmark_worker.py'),'started_clock':started})
            samples=[];released=None
            with (folder/'resources.jsonl').open('a',buffering=1) as log:
                while True:
                    observation=sample();samples.append(observation);log.write(json.dumps(observation)+'\n')
                    foreign=gpu_pids()-{c.pid for c in children}
                    if foreign:raise RuntimeError(f'Foreign GPU process entered measurement: {sorted(foreign)}')
                    if stopping:raise SystemExit(130)
                    if released is None and all((folder/f'worker{i}/ready.json').exists() for i in range(len(children))):
                        released=time.perf_counter();atomic_json(release,{'clock':released})
                    if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Benchmark worker failed; inspect its log')
                    if all(c.poll() is not None for c in children):break
                    if time.perf_counter()-started>900:raise RuntimeError('Measurement cell exceeded predeclared 15-minute budget')
                    time.sleep(.5)
            results=[json.loads((folder/f'worker{i}/result.json').read_text()) for i in range(len(children))]
            if released is None or any(x['status']!='completed' or x['steps']!=len(batch) for x,batch in zip(results,batches)):
                raise ValueError('Incomplete measurement cannot be ranked')
            elapsed=max(x['timed_end'] for x in results)-released;steps=sum(x['steps'] for x in results)
            steady=[s for s in samples if released<=s['clock']<=max(x['timed_end'] for x in results)]
            value={'name':name,'workloads':workloads,'concurrency':len(workloads),'repeat':repeat,'steps':steps,
                'scored_positions':sum(x['scored_positions'] for x in results),'steady_seconds':elapsed,
                'steps_per_second':steps/elapsed,'scored_positions_per_second':sum(x['scored_positions'] for x in results)/elapsed,
                'end_to_end_seconds':time.perf_counter()-started,'device_sampled_peak_mib':max(x['used_mib'] for x in samples),
                'allocator_peak_sum_mib':sum(x['allocator_peak_reserved_mib'] for x in results),
                'max_worker_reserved_mib':max(x['allocator_peak_reserved_mib'] for x in results),
                'minimum_host_available_mib':min(x['host_available_mib'] for x in samples),
                'minimum_container_headroom_gib':min(x['container_headroom_gib'] for x in samples),
                'steady_gpu_utilization_mean':statistics.mean(x['gpu_utilization'] for x in steady) if steady else None,
                'workers':results}
            atomic_json(result,value);return value
        finally:
            for child in children:
                if child.poll() is None:child.send_signal(signal.SIGTERM)
            for child in children:child.wait()
            for handle in handles:handle.close()
            children=[]
    all_results=[];recommendations={}
    for workload in protocol.get('workloads',['training','ranking','inference']):
        measured=[];consecutive_declines=0
        for concurrency in protocol.get('concurrencies',[1,2,3,4,6,8]):
            obs=sample()
            if obs['conservative_available_gib']<2*concurrency+4:
                atomic_json(out/'cgroup_growth_boundary.json',{'concurrency_not_started':concurrency,'observation':obs,
                            'criterion':'2GiB per worker plus4GiB reserve, credit half clean inactive cache; raw charge also retained'})
                if not measured:raise RuntimeError('Insufficient cgroup headroom for first calibration cell')
                break
            if measured:
                obs=sample();per_worker=max(x['max_worker_reserved_mib'] for x in measured)
                if concurrency*per_worker+3072>obs['total_mib'] or obs['host_available_mib']<16384:
                    break
            current=[]
            for repeat in protocol.get('repeats',[1,2]):
                value=cell(f'{workload}_c{concurrency}',[workload]*concurrency,repeat,
                           allocation(protocol['worker']['global_batches'],concurrency))
                current.append(value);measured.append(value);all_results.append(value)
            _,medians=shortlist(measured)
            if statistics.median(x['steps_per_second'] for x in current)<.95*max(medians.values()):consecutive_declines+=1
            else:consecutive_declines=0
            atomic_json(out/'progress.json',{'results':all_results,'current_workload':workload,'current_concurrency':concurrency})
            if consecutive_declines>=2:break
        choice,medians=shortlist(measured)
        recommendations[workload]={'recommended_concurrency':choice,'median_steps_per_second':medians,
            'criterion':'Lowest concurrency within 5% of best tested median; not a global optimum claim'}
    # Equal heterogeneous work: 60 training + 60 inference, split vs overlap.
    mixed=[];n=protocol['worker']['global_batches']//2
    for repeat in (protocol.get('repeats',[1,2]) if protocol.get('mixed_phases',True) else []):
        train=cell('mixed_serial_training',['training'],repeat,[list(range(n))])
        infer=cell('mixed_serial_inference',['inference'],repeat,[list(range(n))])
        parallel=cell('mixed_parallel',['training','inference'],repeat,[list(range(n)),list(range(n))])
        mixed.append({'repeat':repeat,'serial_steady_seconds':train['steady_seconds']+infer['steady_seconds'],
            'parallel_steady_seconds':parallel['steady_seconds'],'serial_end_to_end_seconds':train['end_to_end_seconds']+infer['end_to_end_seconds'],
            'parallel_end_to_end_seconds':parallel['end_to_end_seconds']})
    final={'identity':identity,'recommendations':recommendations,'measurements':all_results,'mixed_phase_comparison':mixed,
           'status':'completed','limitations':['Short warmed-cache measurement, not full-epoch proof.','Allocator peaks exclude external CUDA allocations; device peaks are sampled.','Different workloads are ranked separately.']}
    atomic_json(out/'summary.json',final)
    lines=['# 资源分配实测结果','','固定120个batch，batch32；同类工作按相同有效工作量比较。每档两次；训练、排序和推理分开排名。','']
    for workload,rec in recommendations.items():
        lines.extend([f'## {workload}：建议并发{rec["recommended_concurrency"]}','','|并发|中位step/s ↑|最低/最高step/s|端到端中位秒|采样设备峰值MiB|allocator峰值之和MiB|','|---|---|---|---|---|---|'])
        best=max(rec['median_steps_per_second'].values())
        for c,median in rec['median_steps_per_second'].items():
            m=[x for x in all_results if x['workloads'][0]==workload and x['concurrency']==c]
            score=f'{median:.5f}';score='**'+score+'**' if median==best else score
            lines.append(f'|{c}|{score}|{min(x["steps_per_second"] for x in m):.5f}/{max(x["steps_per_second"] for x in m):.5f}|{statistics.median(x["end_to_end_seconds"] for x in m):.2f}|{max(x["device_sampled_peak_mib"] for x in m)}|{max(x["allocator_peak_sum_mib"] for x in m):.1f}|')
        lines.append('')
    lines.extend(['## 混合阶段：相同60步训练＋60步推理','','|复测|顺序稳态秒|并行稳态秒|顺序端到端秒|并行端到端秒|','|---|---|---|---|---|'])
    for m in mixed:lines.append('|'+ '|'.join(str(m[k]) for k in ['repeat','serial_steady_seconds','parallel_steady_seconds','serial_end_to_end_seconds','parallel_end_to_end_seconds'])+'|')
    lines.extend(['','建议选取在实测最佳中位吞吐5%内的最低并发；没有声称全局唯一最佳。完整有效计分位置吞吐、每worker延迟、资源、身份和恢复记录见summary.json及各cell。短工作量、热缓存和负载变化可能影响适用范围；实际新任务需继续核对吞吐。'])
    (out/'资源分配实测报告.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
