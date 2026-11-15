"""Memory admission for GPU phases; tsp bounds concurrent experiments.

GPU utilization measures busy time, not remaining compute capacity. Allow a
bounded concurrent trial even at 100% busy time, then assess actual throughput.
Callers may explicitly request a stricter utilization threshold.
"""
import datetime
import subprocess
import time
from prediction_progress import atomic_json


def wait_for_capacity(record,min_free_mib=6144,max_utilization=100,min_host_available_mib=8192):
    query_failures=0
    while True:
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            if result.returncode:raise RuntimeError('GPU resource query failed: '+result.stderr[-500:])
        except (subprocess.TimeoutExpired,RuntimeError) as error:
            query_failures+=1
            atomic_json(record,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'admitted':False,'status':'resource_observation_unavailable','consecutive_query_failures':query_failures,'error':str(error),'action':'No training launch; bounded retry of the same read-only query'})
            if query_failures>=3:raise RuntimeError('Three consecutive GPU resource query failures; preserve task checkpoints and request review') from error
            time.sleep(2)
            continue
        query_failures=0
        lines=result.stdout.strip().splitlines()
        if len(lines)!=1:raise ValueError('Admission currently requires one visible GPU')
        utilization,free=map(int,lines[0].split(','))
        with open('/proc/meminfo') as handle:
            available=next(int(line.split()[1])//1024 for line in handle if line.startswith('MemAvailable:'))
        admitted=free>=min_free_mib and utilization<=max_utilization and available>=min_host_available_mib
        atomic_json(record,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'admitted':admitted,'gpu_utilization':utilization,
            'gpu_free_mib':free,'host_available_mib':available,'budget':{'min_gpu_free_mib':min_free_mib,'max_start_utilization':max_utilization,'min_host_available_mib':min_host_available_mib},
            'policy':'Start a phase when headroom exists; do not stop or signal other workloads.'})
        if admitted:return
        time.sleep(5)
