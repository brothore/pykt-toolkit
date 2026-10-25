"""Isolated first-action experiment runner; defaults to validation-only training."""
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
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from torch.utils.data import DataLoader

from pykt.datasets import init_dataset4train
from pykt.models import evaluate, init_model, train_model
from pykt.utils import set_seed


def action_diagnostics(model, loader):
    truths, predictions, failure_truths, failure_predictions = [], [], [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            probs = model.action_probabilities(data).cpu()
            labels = data['action_labels']
            mask = data['action_mask'] & data['smasks'] & data['masks'] & (labels >= 0)
            truths.extend(labels[mask].tolist())
            predictions.extend(probs.argmax(-1)[mask].tolist())
            failure = mask & (labels > 0)
            failure_truths.extend((labels[failure] - 1).tolist())
            failure_predictions.extend(probs[..., 1:].argmax(-1)[failure].tolist())
    return {
        'event_count': len(truths),
        'class_counts': np.bincount(truths, minlength=4).tolist(),
        'confusion_matrix': confusion_matrix(truths, predictions, labels=list(range(4))).tolist(),
        'macro_f1': float(f1_score(truths, predictions, labels=list(range(4)), average='macro', zero_division=0)),
        'failure_macro_f1': float(f1_score(failure_truths, failure_predictions,
                                        labels=list(range(3)), average='macro', zero_division=0)),
        'failure_confusion_matrix': confusion_matrix(failure_truths, failure_predictions,
                                                    labels=list(range(3))).tolist(),
    }


def response_diagnostics(model, loader):
    predictions, truths = [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for data in loader:
            c, r = data['cseqs'].to(device).long(), data['rseqs'].to(device).long()
            y = model(c, r).gather(-1, data['shft_cseqs'].to(device).long().unsqueeze(-1)).squeeze(-1).cpu()
            mask = data['smasks'] & data['masks']
            predictions.extend(y[mask].tolist())
            truths.extend(data['shft_rseqs'][mask].tolist())
    return {'response_nll': float(log_loss(truths, predictions, labels=[0, 1])),
            'response_brier': float(np.mean((np.asarray(predictions) - np.asarray(truths)) ** 2))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['dkt', 'dkt_first_action'], default='dkt_first_action')
    parser.add_argument('--dataset_name', choices=['assist2009'], default='assist2009')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--action_weight', type=float, default=.1)
    parser.add_argument('--action_mode', choices=['joint', 'conditional'], default='joint')
    parser.add_argument('--action_balance', type=float, default=0.)
    parser.add_argument('--action_hidden', type=int, default=100)
    parser.add_argument('--save_dir', type=Path, required=True)
    args = parser.parse_args()
    save = args.save_dir.resolve()
    os.chdir(ROOT / 'examples')
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite an experiment: {save}')
    set_seed(args.seed)
    torch.set_num_threads(1)
    data_all = json.loads((ROOT / 'configs/data_config.json').read_text())
    data_all = copy.deepcopy(data_all)
    data = data_all[args.dataset_name]
    data['dpath'] = str((ROOT / 'examples' / data['dpath']).resolve())
    train_loader, valid_loader, *_ = init_dataset4train(
        args.dataset_name, args.model_name, data_all, args.fold, args.batch_size,
        aug_probs={'truncate': 0, 'duplicate': 0, 'shuffle': 0, 'random_rev': 0})
    model_config = {'emb_size': args.emb_size, 'dropout': args.dropout}
    if args.model_name == 'dkt_first_action':
        model_config.update(action_weight=args.action_weight, action_mode=args.action_mode,
                            action_balance=args.action_balance, action_hidden=args.action_hidden,
                            action_class_counts=train_loader.dataset.action_class_counts)
    params = vars(args).copy()
    params['save_dir'] = str(save)
    params['emb_type'] = 'qid'
    params['use_wandb'] = 0
    params['use_also'] = 0
    snapshot = {'train_config': {'batch_size': args.batch_size, 'num_epochs': args.num_epochs,
                                 'seq_len': data['maxlen'], 'optimizer': 'adam'},
                'model_config': model_config, 'data_config': data, 'params': params}
    (save / 'config.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    model = init_model(args.model_name, model_config.copy(), data, 'qid')
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('EXPERIMENT', json.dumps(params), flush=True)
    print('MODEL_CONFIG', json.dumps(model_config), flush=True)
    print('PARAMETERS', sum(p.numel() for p in model.parameters()), flush=True)
    started = time.monotonic()
    results = train_model(model, train_loader, valid_loader, args.num_epochs, opt,
                          str(save), save_model=True, data_config=data, fold=args.fold,
                          use_trained=0, also_config={'enable': False})
    model.load_state_dict(torch.load(save / 'qid_model.ckpt', map_location=next(model.parameters()).device,
                                     weights_only=True))
    if args.model_name == 'dkt_first_action':
        torch.save(model.export_dkt_state_dict(), save / 'qid_dkt_backbone.ckpt')
    val = DataLoader(valid_loader.dataset, batch_size=128, shuffle=False, num_workers=0)
    auc, acc = evaluate(model, val, args.model_name)
    diagnostics = response_diagnostics(model, val)
    if args.model_name == 'dkt_first_action':
        diagnostics.update(action_diagnostics(model, val))
    summary = {'status': 'trained_and_reloaded', 'model': args.model_name, 'dataset': args.dataset_name,
               'fold': args.fold, 'seed': args.seed, 'epoch_budget': args.num_epochs,
               'best_epoch': int(results[-1]), 'validation_auc': float(auc), 'validation_accuracy': float(acc),
               'elapsed_seconds': time.monotonic() - started, 'model_config': model_config,
               'parameters': sum(p.numel() for p in model.parameters()),
               'train_sequences': len(train_loader.dataset), 'validation_sequences': len(valid_loader.dataset),
               'validation_diagnostics': diagnostics, 'test_evaluated': False}
    (save / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('VALIDATION_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
