"""Validation-only, isolated DKT process-supervision experiments."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader

from pykt.datasets import init_dataset4train
from pykt.models import evaluate, init_model, train_model
from pykt.utils import set_seed
from train_dkt_first_action import response_diagnostics


def process_diagnostics(model, loader, train_dataset):
    truth = {key: [] for key in ('time', 'attempt', 'hint')}
    pred = {key: [] for key in truth}
    concept_truth = {key: [] for key in truth}
    failure_truth = {key: [] for key in ('attempt', 'hint')}
    failure_pred = {key: [] for key in failure_truth}
    losses = {key: [] for key in ('response', 'time', 'attempt', 'hint')}
    model.eval()
    with torch.no_grad():
        for data in loader:
            p = {key: value.cpu() for key, value in model.process_predictions(data).items()}
            _, components = model.compute_loss(data, return_components=True)
            for key, value in components.items():
                losses[key].append(float(value))
            for key, prediction in [('time', 'time_prediction'), ('attempt', 'attempt_probs'), ('hint', 'hint_prob')]:
                mask = data[f'process_{key}_mask'] & data['masks'] & data['smasks']
                if key != 'time':
                    failed = mask & (data['shft_rseqs'] == 0)
                    failure_truth[key].extend(data[f'process_{key}'][failed].tolist())
                    failure_pred[key].extend(p[prediction][failed].tolist())
                if model.process_mode == 'conditional' and key != 'time':
                    mask &= data['shft_rseqs'] == 0
                truth[key].extend(data[f'process_{key}'][mask].tolist())
                pred[key].extend(p[prediction][mask].tolist())
                concept_truth[key].extend(data['shft_cseqs'][mask].tolist())
    stats = model.process_stats
    counts_a = np.asarray(stats['attempt_counts'], dtype=float)
    counts_h = np.asarray(stats['hint_counts'], dtype=float)
    pa = np.maximum(counts_a, 1) / np.maximum(counts_a, 1).sum()
    ph = np.maximum(counts_h, 1) / np.maximum(counts_h, 1).sum()
    attempt_truth = np.asarray(truth['attempt'], dtype=int)
    attempt_pred = np.asarray(pred['attempt'])
    hint_truth = np.asarray(truth['hint'], dtype=int)
    hint_pred = np.asarray(pred['hint'])
    time_truth, time_pred = np.asarray(truth['time']), np.asarray(pred['time'])
    # A stronger diagnostic than global prevalence: training-fold concept-only
    # priors. This distinguishes predictable item/concept effects from history.
    concepts = train_dataset.dori['cseqs'][:, 1:].long()
    failure = train_dataset.dori['rseqs'][:, 1:] == 0
    concept_baseline = {}
    for key in truth:
        fields = train_dataset.fields
        mask = fields[f'process_{key}_mask'].clone()
        if model.process_mode == 'conditional' and key != 'time':
            mask &= failure
        c = concepts[mask].numpy()
        targets = fields[f'process_{key}'][mask].numpy()
        if key == 'time':
            n = np.bincount(c, minlength=model.num_c)
            sums = np.bincount(c, weights=targets, minlength=model.num_c)
            estimates = (sums + 10 * stats['time_center']) / (n + 10)
            concept_baseline['time_concept_mean_log_mae'] = float(np.abs(time_truth - estimates[concept_truth[key]]).mean())
        else:
            prior = pa if key == 'attempt' else ph
            k = len(prior)
            counts = np.zeros((model.num_c, k))
            np.add.at(counts, (c, targets.astype(int)), 1)
            estimates = (counts + 10 * prior) / (counts.sum(-1, keepdims=True) + 10)
            y = np.asarray(truth[key], dtype=int)
            concept_baseline[f'{key}_concept_prior_nll'] = float(-np.log(estimates[np.asarray(concept_truth[key]), y]).mean())
    return {
        **concept_baseline,
        'matched_failure_subset': {
            'attempt_count': len(failure_truth['attempt']),
            'hint_count': len(failure_truth['hint']),
            'attempt_macro_f1': float(f1_score(failure_truth['attempt'], np.asarray(failure_pred['attempt']).argmax(-1),
                                              labels=[0, 1, 2, 3], average='macro', zero_division=0)),
            'hint_auc': float(roc_auc_score(failure_truth['hint'], failure_pred['hint'])),
        },
        'counts': {key: len(value) for key, value in truth.items()},
        'mean_batch_loss_components': {key: float(np.mean(value)) for key, value in losses.items()},
        'time_log_mae': float(mean_absolute_error(time_truth, time_pred)),
        'time_train_mean_log_mae': float(np.abs(time_truth - stats['time_center']).mean()),
        'attempt_macro_f1': float(f1_score(attempt_truth, attempt_pred.argmax(-1), labels=[0, 1, 2, 3], average='macro', zero_division=0)),
        'attempt_nll': float(log_loss(attempt_truth, attempt_pred, labels=[0, 1, 2, 3])),
        'attempt_train_prior_nll': float(-np.log(pa[attempt_truth]).mean()),
        'hint_auc': float(roc_auc_score(hint_truth, hint_pred)) if len(set(hint_truth)) == 2 else None,
        'hint_nll': float(log_loss(hint_truth, hint_pred, labels=[0, 1])),
        'hint_train_prior_nll': float(-np.log(ph[hint_truth]).mean()),
    }


@torch.backends.cudnn.flags(enabled=False)
def gradient_diagnostics(model, loader, max_batches=4):
    """Compare auxiliary gradients with response gradients on validation only.

    No optimizer steps; no validation values affect parameters/statistics.
    """
    model.eval()
    shared = list(model.interaction_emb.parameters()) + list(model.lstm_layer.parameters())
    results = {name: [] for name in ('time', 'attempt', 'hint')}
    for batch_index, data in enumerate(loader):
        if batch_index >= max_batches:
            break
        _, components = model.compute_loss(data, return_components=True)
        gradients = {}
        for key, loss in components.items():
            if not loss.requires_grad:
                gradients[key] = torch.cat([torch.zeros_like(p).reshape(-1) for p in shared])
                continue
            grad = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
            gradients[key] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                        for p, g in zip(shared, grad)])
        reference = gradients['response']
        for key in results:
            other = gradients[key]
            results[key].append({'cosine': float(torch.nn.functional.cosine_similarity(reference, other, dim=0)),
                                 'norm_over_response': float(other.norm() / reference.norm().clamp_min(1e-12))})
    return {key: {metric: float(np.mean([row[metric] for row in rows]))
                  for metric in ('cosine', 'norm_over_response')} for key, rows in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['dkt', 'dkt_process'], default='dkt_process')
    parser.add_argument('--dataset_name', choices=['assist2009'], default='assist2009')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--time_weight', type=float, default=.1)
    parser.add_argument('--attempt_weight', type=float, default=.1)
    parser.add_argument('--hint_weight', type=float, default=.1)
    parser.add_argument('--process_mode', choices=['joint', 'conditional'], default='joint')
    parser.add_argument('--process_balance', type=float, default=0.)
    parser.add_argument('--save_dir', type=Path, required=True)
    args = parser.parse_args()
    save = args.save_dir.resolve()
    os.chdir(ROOT / 'examples')
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {save}')
    set_seed(args.seed)
    torch.set_num_threads(1)
    all_data = copy.deepcopy(json.loads((ROOT / 'configs/data_config.json').read_text()))
    data = all_data[args.dataset_name]
    data['dpath'] = str((ROOT / 'examples' / data['dpath']).resolve())
    train, valid, *_ = init_dataset4train(args.dataset_name, args.model_name, all_data,
                                         args.fold, args.batch_size, args=args,
                                         aug_probs={'truncate': 0, 'duplicate': 0, 'shuffle': 0, 'random_rev': 0})
    config = {'emb_size': args.emb_size, 'dropout': args.dropout}
    if args.model_name == 'dkt_process':
        config.update(time_weight=args.time_weight, attempt_weight=args.attempt_weight, hint_weight=args.hint_weight,
                      process_mode=args.process_mode, process_balance=args.process_balance,
                      process_stats=train.dataset.process_stats)
    params = vars(args).copy()
    params.update(save_dir=str(save), emb_type='qid', use_wandb=0, use_also=0)
    snapshot = {'train_config': {'batch_size': args.batch_size, 'num_epochs': args.num_epochs,
                                 'seq_len': data['maxlen'], 'optimizer': 'adam'},
                'model_config': config, 'data_config': data, 'params': params}
    (save / 'config.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    model = init_model(args.model_name, config.copy(), data, 'qid')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('EXPERIMENT', json.dumps(snapshot), flush=True)
    started = time.monotonic()
    results = train_model(model, train, valid, args.num_epochs, optimizer, str(save), save_model=True,
                          data_config=data, fold=args.fold, use_trained=0, also_config={'enable': False})
    train_seconds = time.monotonic() - started
    model.load_state_dict(torch.load(save / 'qid_model.ckpt', weights_only=True,
                                     map_location=next(model.parameters()).device))
    loader = DataLoader(valid.dataset, batch_size=128, shuffle=False, num_workers=0)
    auc, acc = evaluate(model, loader, args.model_name)
    diagnostics = response_diagnostics(model, loader)
    if args.model_name == 'dkt_process':
        torch.save(model.export_dkt_state_dict(), save / 'qid_dkt_backbone.ckpt')
        diagnostics.update(process_diagnostics(model, loader, train.dataset))
        diagnostics['shared_gradient_diagnostics'] = gradient_diagnostics(model, loader)
    result = {'model': args.model_name, 'dataset': args.dataset_name, 'fold': args.fold, 'seed': args.seed,
              'epoch_budget': args.num_epochs, 'best_epoch': int(results[-1]),
              'validation_auc': float(auc), 'validation_accuracy': float(acc),
              'training_seconds': train_seconds, 'model_config': config,
              'parameters': sum(p.numel() for p in model.parameters()),
              'train_sequences': len(train.dataset), 'validation_sequences': len(valid.dataset),
              'validation_diagnostics': diagnostics, 'test_evaluated': False}
    (save / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('VALIDATION_SUMMARY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
