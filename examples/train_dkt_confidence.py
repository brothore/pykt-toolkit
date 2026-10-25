"""NIPS confidence experiments: validation only until checkpoints are frozen."""
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
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from pykt.datasets import init_dataset4train
from pykt.datasets.confidence_dataset import ConfidenceDataset
from pykt.models import evaluate, init_model, train_model
from pykt.utils import set_seed
from train_dkt_first_action import response_diagnostics


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def response_coverage_diagnostics(model, loader):
    truths, predictions, observed = [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for data in loader:
            p = model(data['cseqs'].to(device).long(), data['rseqs'].to(device).long())
            p = p.gather(-1, data['shft_cseqs'].to(device).long().unsqueeze(-1)).squeeze(-1).cpu()
            # Base main metric retains original concept expansion. Diagnostic
            # groups use first original event positions, including missingness.
            truths.append(data['shft_rseqs'].numpy())
            predictions.append(p.numpy())
            observed.append(data['confidence_mask'].numpy())
    y, p = np.concatenate(truths), np.concatenate(predictions)
    observed = np.concatenate(observed)
    events = loader.dataset.event_mask.numpy()
    return {'observed': {'count': int(observed.sum()), 'auc': safe_auc(y[observed], p[observed])},
            'missing': {'count': int((events & ~observed).sum()),
                        'auc': safe_auc(y[events & ~observed], p[events & ~observed])}}


def confidence_diagnostics(model, loader, train_dataset):
    labels, responses, predictions, joints, concepts, questions = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            probs = {key: value.cpu() for key, value in model.confidence_probabilities(data).items()}
            mask = data['confidence_mask'] & data['masks'] & data['smasks']
            labels.extend(data['confidence_labels'][mask].tolist())
            responses.extend(data['shft_rseqs'][mask].tolist())
            predictions.extend(probs['confidence_probs'][mask].tolist())
            joints.extend(probs['joint_probs'][mask].tolist())
            concepts.extend(data['shft_cseqs'][mask].tolist())
            questions.extend(data['shft_qseqs'][mask].tolist())
    y, r = np.asarray(labels, dtype=int), np.asarray(responses, dtype=int)
    p, joint = np.asarray(predictions), np.asarray(joints)
    counts = np.maximum(np.asarray(train_dataset.confidence_counts, dtype=float), 1)
    prior = counts / counts.sum()
    result = {'count': len(y), 'class_counts': np.bincount(y, minlength=5).tolist(),
              'confidence_nll': float(log_loss(y, p, labels=[0, 1, 2, 3, 4])),
              'confidence_prior_nll': float(-np.log(prior[y]).mean()),
              'confidence_macro_f1': float(f1_score(y, p.argmax(-1), labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)),
              'confidence_level_mae': float(np.abs(p @ np.arange(5) - y).mean()),
              'prior_level_mae': float(np.abs(prior @ np.arange(5) - y).mean()),
              'joint_nll': float(-np.log(np.clip(joint[np.arange(len(y)), r, y], 1e-12, 1)).mean())}
    for key, selected, values in [('concept', concepts, train_dataset.dori['cseqs']),
                                   ('question', questions, train_dataset.dori['qseqs'])]:
        indices = values[:, 1:][train_dataset.mask].long().numpy()
        truth = train_dataset.labels[train_dataset.mask].numpy()
        table = np.zeros((max(int(values.max()), max(selected)) + 1, 5))
        np.add.at(table, (indices, truth), 1)
        table = (table + 10 * prior) / (table.sum(-1, keepdims=True) + 10)
        result[f'{key}_prior_nll'] = float(-np.log(table[np.asarray(selected), y]).mean())
    for name, event, probability in [
        ('high_confidence_wrong', (r == 0) & (y >= 3), joint[:, 0, 3:].sum(-1)),
        ('low_confidence_correct', (r == 1) & (y <= 1), joint[:, 1, :2].sum(-1))]:
        result[name] = {'count': int(event.sum()), 'auc': safe_auc(event, probability),
                        'brier': float(((probability - event) ** 2).mean())}
    return result


@torch.backends.cudnn.flags(enabled=False)
def gradient_diagnostics(model, loader):
    model.eval()
    shared = list(model.interaction_emb.parameters()) + list(model.lstm_layer.parameters())
    rows = []
    for i, data in enumerate(loader):
        if i >= 4:
            break
        _, parts = model.compute_loss(data, return_components=True)
        vectors = {}
        for key, loss in parts.items():
            if not loss.requires_grad:
                vectors[key] = torch.cat([torch.zeros_like(p).reshape(-1) for p in shared])
                continue
            grad = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
            vectors[key] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                      for p, g in zip(shared, grad)])
        a, b = vectors['response'], vectors['confidence']
        rows.append([float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                     float(b.norm() / a.norm().clamp_min(1e-12))])
    means = np.mean(rows, axis=0)
    return {'cosine': float(means[0]), 'aux_norm_over_response': float(means[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['dkt', 'dkt_confidence'], default='dkt_confidence')
    parser.add_argument('--dataset_name', choices=['nips_task34'], default='nips_task34')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--confidence_weight', type=float, default=.1)
    parser.add_argument('--confidence_mode', choices=['categorical', 'ordinal', 'conditional'], default='categorical')
    parser.add_argument('--confidence_balance', type=float, default=0.)
    parser.add_argument('--save_dir', type=Path, required=True)
    args = parser.parse_args()
    save = args.save_dir.resolve()
    os.chdir(ROOT / 'examples')
    save.mkdir(parents=True, exist_ok=True)
    if (save / 'config.json').exists():
        raise FileExistsError(f'Refusing to overwrite experiment {save}')
    set_seed(args.seed)
    torch.set_num_threads(1)
    all_data = copy.deepcopy(json.loads((ROOT / 'configs/data_config.json').read_text()))
    data = all_data[args.dataset_name]
    data['dpath'] = str((ROOT / 'examples' / data['dpath']).resolve())
    train, valid, *_ = init_dataset4train(args.dataset_name, args.model_name, all_data, args.fold,
                                         args.batch_size, aug_probs={'truncate': 0, 'duplicate': 0, 'shuffle': 0, 'random_rev': 0})
    config = {'emb_size': args.emb_size, 'dropout': args.dropout}
    if args.model_name == 'dkt_confidence':
        config.update(confidence_weight=args.confidence_weight, confidence_mode=args.confidence_mode,
                      confidence_balance=args.confidence_balance, confidence_counts=train.dataset.confidence_counts)
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
    trained = train_model(model, train, valid, args.num_epochs, optimizer, str(save), save_model=True,
                          data_config=data, fold=args.fold, use_trained=0, also_config={'enable': False})
    elapsed = time.monotonic() - started
    model.load_state_dict(torch.load(save / 'qid_model.ckpt', map_location=next(model.parameters()).device, weights_only=True))
    auc, acc = evaluate(model, valid, args.model_name)
    if args.model_name == 'dkt':
        validation_dataset = ConfidenceDataset(Path(data['dpath']) / data['train_valid_file'], data['input_type'],
                                               {args.fold}, mode='valid', aug_probs={})
    else:
        validation_dataset = valid.dataset
    loader = DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=0)
    diagnostics = response_diagnostics(model, loader)
    diagnostics['coverage_groups'] = response_coverage_diagnostics(model, loader)
    if args.model_name == 'dkt_confidence':
        torch.save(model.export_dkt_state_dict(), save / 'qid_dkt_backbone.ckpt')
        diagnostics['confidence'] = confidence_diagnostics(model, loader, train.dataset)
        diagnostics['gradient'] = gradient_diagnostics(model, loader)
    output = {'model': args.model_name, 'dataset': args.dataset_name, 'fold': args.fold, 'seed': args.seed,
              'epoch_budget': args.num_epochs, 'best_epoch': int(trained[-1]), 'validation_auc': float(auc),
              'validation_accuracy': float(acc), 'training_seconds': elapsed, 'model_config': config,
              'parameters': sum(p.numel() for p in model.parameters()),
              'train_sequences': len(train.dataset), 'validation_sequences': len(valid.dataset),
              'validation_diagnostics': diagnostics, 'test_evaluated': False}
    (save / 'validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print('VALIDATION_SUMMARY', json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
