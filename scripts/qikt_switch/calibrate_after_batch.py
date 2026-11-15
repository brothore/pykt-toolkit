"""Defer matched frozen-workload resource tests until current batches end."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
from cooperative_process import run_cooperative
from prediction_progress import atomic_json,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());out=args.output_dir;out.mkdir(exist_ok=True)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    for dependency in spec.get('dependencies',[]):
        while not stop.is_set():
            p=Path(dependency['result']);s=Path(dependency['state'])
            if p.exists():
                if not json.loads(p.read_text()).get('passed'):raise ValueError('Implementation preflight failed')
                break
            if s.exists() and json.loads(s.read_text())['status'] in {'failed','stopped','completed'}:
                raise RuntimeError('Implementation preflight terminated without passing')
            stop.wait(5)
    if stop.is_set():return 130
    while not stop.is_set():
        states={}
        for root in map(Path,spec['wait_roots']):
            for family in json.loads((root/'registration.json').read_text())['families']:
                p=root/'jobs'/family/'state.json';states[str(p)]=json.loads(p.read_text())['status'] if p.exists() else 'queued'
        atomic_json(out/'progress.json',{'status':'waiting_for_natural_completion','states':states})
        if all(s in {'completed','failed','stopped'} for s in states.values()):break
        stop.wait(15)
    if stop.is_set():return 130
    results={}
    with Path(spec['gpu_lock']).open('a') as lock:
        while not stop.is_set():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:stop.wait(5)
        while not stop.is_set():
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
            if not active:break
            atomic_json(out/'progress.json',{'status':'waiting_for_gpu_contexts_to_finish','observed_pids':active});stop.wait(5)
        if stop.is_set():return 130
        for case in spec['cases']:
            folder=Path(case['round_root']);summary=json.loads((folder/'baseline/round_summary.json').read_text())
            window=json.loads((folder/'baseline/validation_window/validation_window_results.json').read_text())
            checkpoint=Path(summary['checkpoint_dir'])
            if digest(checkpoint/'attn_model.ckpt')!=window['checkpoint_sha256']:raise ValueError('Completed source checkpoint changed')
            destination=out/case['name'];destination.mkdir(exist_ok=True)
            protocol={'workloads':['training','inference'],'concurrencies':[1,2,3],'repeats':[1,2],'mixed_phases':False,
                'worker':{'checkpoint_dir':str(checkpoint),'batch_size':32,'global_batches':120,'threads':2,'warmup':6},
                'source_checkpoint_sha256':window['checkpoint_sha256'],'source_summary_sha256':digest(folder/'baseline/round_summary.json'),
                'scope':'Disposable matched workload, no scientific weights retained; dataset/fold/seed and selected epoch inherited from source'}
            pp=destination/'protocol.json'
            if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Completed source/protocol identity differs')
            atomic_json(pp,protocol)
            run_cooperative([sys.executable,str(Path(__file__).with_name('calibrate_resources.py')),'--protocol',str(pp),'--output-dir',str(destination/'measurements')],check=True)
            results[case['name']]=json.loads((destination/'measurements/summary.json').read_text())
    atomic_json(out/'summary.json',{'status':'completed','cases':results,
                'scope':'Only these frozen architectures; no general global optimum or scientific efficacy claim'})
    lines=['# 当前冻结负载资源分配实测','','每类固定120个batch、batch32、6步warmup，每档两次；训练与推理分开。优先选距最高中位吞吐5%内的最低并发。当前任务自然完成后才测试，没有停止旧任务。','']
    for name,value in results.items():
        lines.append('## '+name+'\n')
        lines.append((out/name/'measurements/资源分配实测报告.md').read_text())
    lines.append('限制：训练样本为固定seed置乱分配，与原生SequentialSampler的完整epoch次序不同；仅是相同形状固定工作量的代表性吞吐。不同冻结结构、额外KL前向和完整主干不能相互替代结论；完整阶段包含加载/保存开销，不能只看GPU利用率或显存。')
    (out/'资源分配实测总报告.md').write_text('\n'.join(lines)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
