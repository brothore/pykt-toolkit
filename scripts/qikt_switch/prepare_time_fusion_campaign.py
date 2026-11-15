"""Real-container checks and independent throughput probes for temporal fusion."""
import argparse
import copy
import importlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pykt.utils import set_seed
from evaluate_validation import load_run, load_saved_model
from prediction_progress import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['time_input','time_decode','time_gate'])
    parser.add_argument('--eligibility-result', type=Path)
    args = parser.parse_args()
    root = args.root
    if args.eligibility_result:
        selection = json.loads(args.eligibility_result.read_text())
        atomic_json(root / 'source_selection_evidence.json', {
            'result': str(args.eligibility_result), 'sha256': digest(args.eligibility_result),
            'selection': selection})
        if not selection.get('passed'):
            skipped = {'passed': False, 'status': 'skipped_source_selection', 'cases': {}}
            atomic_json(root / 'native_preflight.json', skipped)
            atomic_json(root / 'probe_ready.json', skipped)
            return
    set_seed(42)
    ready = json.loads((root / 'baseline_ready.json').read_text())
    config, paths = load_run(ready['checkpoint_dir'])
    baseline = load_saved_model(config, paths, device='cuda')
    from pykt.datasets.que_data_loader import KTQueDataset
    cfg = config['data_config']
    dataset = KTQueDataset(str(paths['fragment_path']), cfg['input_type'],
        set(cfg['folds']) - {paths['fold']}, cfg['num_c'], cfg['max_concepts'],
        mode='train', aug_probs={})
    data = next(iter(DataLoader(dataset, batch_size=32)))
    with torch.no_grad():
        expected = baseline.predict_one_step(data)
    factory = importlib.import_module('pykt.models.init_model')
    factory.device = 'cuda'
    checks, probes = [], {}
    for mode in args.modes:
        current = copy.deepcopy(config['model_config'])
        current.pop('learning_rate', None)
        current['other_config'].update(reference_fusion_mode=mode,
            reference_fusion_scale=.1, reference_fusion_bias='none')
        model = factory.init_model('qikt_mamba', current, cfg, paths['emb_type'])
        missing = model.load_state_dict(baseline.state_dict(), strict=False)
        if missing.unexpected_keys or any(not k.startswith('model.reference_fusion.') for k in missing.missing_keys):
            raise ValueError('Unexpected baseline-state mismatch')
        model.model.eval()
        with torch.no_grad():
            error = float((model.predict_one_step(data) - expected).abs().max())
        if not torch.isfinite(torch.tensor(error)) or error > 1e-6:
            raise ValueError('Temporal module must start at baseline predictions')
        folder = root / 'resource_probes' / mode
        folder.mkdir(parents=True, exist_ok=True)
        saved = copy.deepcopy(config)
        saved['model_config'] = current
        saved['params']['other_config'] = current['other_config']
        saved['probe_metadata'] = {'scientific_training': False,
            'baseline_sha256': digest(paths['checkpoint_path'])}
        atomic_json(folder / 'config.json', saved)
        torch.save(model.state_dict(), folder / 'attn_model.ckpt')
        atomic_json(folder / 'protocol.json', {
            'workloads': ['training', 'inference'], 'concurrencies': [1, 2, 3],
            'repeats': [1, 2], 'mixed_phases': False,
            'worker': {'checkpoint_dir': str(folder), 'batch_size': 32,
                'global_batches': 120, 'threads': 2, 'warmup': 6},
            'scope': 'Registered dual-GRU fusion variants measured separately'})
        probes[mode] = {'checkpoint': str(folder), 'protocol': str(folder / 'protocol.json')}
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        for _ in range(2):
            model.model.train()
            optimizer.zero_grad()
            _, loss = model.train_one_step(data)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite temporal loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite temporal gradient')
            optimizer.step()
        checks.append({'mode': mode, 'zero_error': error, 'two_steps_finite': True})
        del model, optimizer
    atomic_json(root / 'native_preflight.json', {'passed': True,
        'checks': checks, 'training_fragments': 32, 'scientific_training': False})
    atomic_json(root / 'probe_ready.json', {'passed': True, 'cases': probes})


if __name__ == '__main__':
    main()
