"""One disposable native QIKT workload; never saves scientific model weights."""
import argparse
import importlib
import json
from pathlib import Path
import resource
import signal
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate
from evaluate_validation import load_run, load_saved_model
from evaluate_window import QuestionWindows
from prediction_progress import atomic_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args(); spec = json.loads(args.spec.read_text())
    out = args.spec.parent; started = time.perf_counter(); stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(spec['threads'])
    from pykt.utils import set_seed
    set_seed(42+spec['worker'])
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()
    config, paths = load_run(spec['checkpoint_dir']); cfg = config['data_config']
    model = load_saved_model(config, paths, device='cuda')
    workload = spec['workload']; batch_size = spec['batch_size']
    if workload == 'inference':
        raw = pd.read_csv(paths['raw_path']); raw = raw[raw.fold==paths['fold']]
        dataset = QuestionWindows(raw, cfg.get('maxlen',200), cfg['max_concepts'])
        indices = np.linspace(0,len(dataset)-1,spec['global_batches']*batch_size,dtype=int)
        opt = None
    else:
        from pykt.datasets.que_data_loader import KTQueDataset
        dataset = KTQueDataset(str(paths['fragment_path']), cfg['input_type'],
            set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={})
        indices = np.random.default_rng(42).permutation(len(dataset))
        if len(indices)<spec['global_batches']*batch_size:
            raise ValueError('Insufficient unique training batches')
        if workload == 'ranking':
            from pykt.models.qikt_sequence_objectives import install_training_objective
            install_training_objective(model,dataset,{'alpha':.02,'rank_weight':.0005})
        if config.get('probe_training_objective'):
            from pykt.models.qikt_baseline_anchor import install_baseline_anchor
            install_baseline_anchor(model,config['probe_training_objective'])
        opt = torch.optim.Adam(model.parameters(),lr=1e-4)
    native = importlib.import_module('pykt.models.train_model')
    def batch(number):
        selected = indices[number*batch_size:(number+1)*batch_size]
        data = [dataset[int(i)] for i in selected]
        return default_collate([item[0] for item in data] if workload=='inference' else data)
    def step(data):
        if opt is None:
            model.model.eval()
            with torch.no_grad(): result = model.predict_one_step(data)
            result.detach().cpu()  # native exporters copy batch predictions to host
        else:
            model.model.train(); loss = native.model_forward(model,data)
            opt.zero_grad(); loss.backward(); opt.step()
            if not np.isfinite(float(loss.detach().cpu())): raise ValueError('Nonfinite benchmark loss')
    numbers = spec['batches']; warm = batch(numbers[0])
    for _ in range(spec['warmup']):
        if stopping: raise SystemExit(130)
        step(warm)
    torch.cuda.synchronize(); ready = time.perf_counter()
    atomic_json(out/'ready.json',{'pid':__import__('os').getpid(),'ready_seconds':ready-started})
    release = Path(spec['release'])
    while not release.exists():
        if stopping: raise SystemExit(130)
        time.sleep(.01)
    timed_start = time.perf_counter(); tokens = 0; count = 0
    for number in numbers:
        if stopping: break
        data = batch(number); tokens += int(data['smasks'].sum()); step(data); count += 1
    torch.cuda.synchronize(); ended = time.perf_counter()
    atomic_json(out/'result.json',{'status':'stopped' if stopping else 'completed','workload':workload,
        'worker':spec['worker'],'steps':count,'scored_positions':tokens,'timed_start':timed_start,
        'timed_end':ended,'timed_seconds':ended-timed_start,'startup_warmup_seconds':ready-started,
        'allocator_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'allocator_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'host_max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'cpu_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,
        'runtime_flags':{'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'CUDA_LAUNCH_BLOCKING':__import__('os').environ.get('CUDA_LAUNCH_BLOCKING'),
            'CUBLAS_WORKSPACE_CONFIG':__import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG')}})
    if stopping: raise SystemExit(130)


if __name__=='__main__': main()
