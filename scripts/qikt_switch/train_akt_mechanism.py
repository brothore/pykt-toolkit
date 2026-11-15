"""Matched AKT training with complete-epoch Adam/RNG/early-stop resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import torch
from sklearn.metrics import roc_auc_score
from pykt.utils import set_seed
from akt_experiment_io import build_model, loaders, predict
from reference_training import to_device
from prediction_progress import atomic_json, digest
from resumable_qikt import atomic_save, rng_state, restore_rng


def train(config, output, epoch_hook=None, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('akt_experiment_io.py'),
               Path('pykt/models/akt.py'), Path(__file__).with_name('resumable_qikt.py'),
               Path(__file__).with_name('reference_training.py')]
    if config['mechanism'] != 'baseline':
        sources.append(Path('pykt/models/akt_fairness.py'))
    cfg = config['data_config']; data = Path(cfg['dpath']) / cfg['train_valid_file']
    identity = {'config': config, 'sources': {str(p): digest(p) for p in sources}, 'data': digest(data)}
    set_seed(config['params']['seed']); torch.set_num_threads(2)
    model = build_model(config, device); train_loader, valid_loader = loaders(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model_config']['learning_rate'])
    checkpoint = output / 'resume.pt'
    state = dict(identity=identity, completed_epochs=0, selected_epoch=-1, best_validation_auc=0.,
                 steps=0, finished=False, epochs=[], history=[])
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume identity differs: do not overwrite an existing run')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], train_loader)
    atomic_json(output / 'config.json', config)
    atomic_json(output / 'identity.json', identity)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for index, source in enumerate(sources):
        target=snapshot/f'{index}_{source.name}'
        if target.exists() and digest(target)!=digest(source):raise ValueError('Frozen source snapshot changed')
        if not target.exists():shutil.copy2(source,target)
    state['history'].append(dict(event='resume' if checkpoint.exists() else 'start',
                                pid=os.getpid(), time=time.time(), from_epoch=state['completed_epochs']))
    atomic_json(output / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': str(device), 'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'command': __import__('sys').argv, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'CUDA_LAUNCH_BLOCKING': os.getenv('CUDA_LAUNCH_BLOCKING'), 'scheduler': None, 'AMP': False,
        'resume_granularity': 'Complete epoch; unfinished epoch replayed from last committed state'})
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    handlers = {s: signal.signal(s, stop) for s in [signal.SIGTERM, signal.SIGINT]}
    try:
        for epoch in range(state['completed_epochs'] + 1, config['params']['num_epochs'] + 1):
            if state['finished']: break
            began = time.perf_counter(); model.train(); loss_sum = 0.; positions = 0
            for i, batch in enumerate(train_loader):
                d, g = to_device(batch, device); optimizer.zero_grad()
                pred, regularizer = predict(model, d, g); mask = d['smasks'].bool()
                loss = torch.nn.functional.binary_cross_entropy(pred[:, 1:][mask].double(),
                    d['shft_rseqs'][mask].double()) + regularizer
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); optimizer.step(); state['steps'] += 1
                loss_sum += float(loss.detach()); positions += int(mask.sum())
                if i % 20 == 0:
                    atomic_json(output / 'progress.json', dict(phase='training', epoch=epoch, batch=i,
                        batches=len(train_loader), pid=os.getpid(), stop_requested=stopping, updated=time.time()))
            training_seconds = time.perf_counter() - began
            model.eval(); y, p = [], []
            with torch.no_grad():
                for batch in valid_loader:
                    d, g = to_device(batch, device); pred, _ = predict(model, d, g); mask = d['smasks'].bool()
                    y.extend(d['shft_rseqs'][mask].cpu().tolist()); p.extend(pred[:, 1:][mask].cpu().tolist())
            auc = float(roc_auc_score(y, p))
            if auc > state['best_validation_auc'] + .001:
                state.update(selected_epoch=epoch, best_validation_auc=auc)
                atomic_save(model.state_dict(), output / 'best_model.ckpt')
            state.update(completed_epochs=epoch, finished=epoch - state['selected_epoch'] >= 10 or epoch == config['params']['num_epochs'])
            state['epochs'].append(dict(epoch=epoch, loss=loss_sum / len(train_loader), validation_fragment_auc=auc,
                selected_epoch=state['selected_epoch'], training_seconds=training_seconds,
                total_seconds=time.perf_counter() - began, training_positions=positions,
                positions_per_second=positions/training_seconds))
            state.update(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(train_loader))
            atomic_save(state, checkpoint)
            stats = {k: state[k] for k in ['completed_epochs','selected_epoch','best_validation_auc','finished','steps','history']}
            atomic_json(output / 'training_stats.json', stats); atomic_json(output / 'epoch_metrics.json', state['epochs'])
            print(json.dumps(state['epochs'][-1]), flush=True)
            if epoch_hook and epoch_hook(epoch): stopping = True
            if stopping:
                atomic_json(output / 'progress.json', dict(phase='stopped', epoch=epoch)); return 130
        atomic_json(output / 'training_complete.json', dict(completed=True, epoch=state['completed_epochs'],
            selected_epoch=state['selected_epoch'], weights_sha256=digest(output/'best_model.ckpt'), ended=time.time()))
        atomic_json(output / 'progress.json', dict(phase='completed', epoch=state['completed_epochs']))
        return 0
    finally:
        for s, handler in handlers.items(): signal.signal(s, handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try: code = train(json.loads(a.config.read_text()), a.output)
    except BaseException as e:
        a.output.mkdir(parents=True, exist_ok=True)
        atomic_json(a.output/'failure.json', dict(error=repr(e), time=time.time())); raise
    raise SystemExit(code)
