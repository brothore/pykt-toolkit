"""Validation-only DKT selected-option experiments."""
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
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader

from pykt.datasets import init_dataset4train
from pykt.models import evaluate, init_model, train_model
from pykt.utils import set_seed
from train_dkt_first_action import response_diagnostics


def option_diagnostics(model, loader, train_dataset):
    labels, probabilities, keys, responses, questions, main_prob = [], [], [], [], [], []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for data in loader:
            out = model.option_probabilities(data)['option_probs'].cpu()
            mask = data['option_mask'] & data['masks'] & data['smasks']
            q = data['shft_qseqs'].long()
            p = model(data['cseqs'].to(device).long(), data['rseqs'].to(device).long()).gather(
                -1, data['shft_cseqs'].to(device).long().unsqueeze(-1)).squeeze(-1).cpu()
            labels.extend(data['option_labels'][mask].tolist()); probabilities.extend(out[mask].tolist())
            questions.extend(q[mask].tolist()); responses.extend(data['shft_rseqs'][mask].tolist())
            keys.extend(model.option_keys.cpu()[q[mask]].tolist()); main_prob.extend(p[mask].tolist())
    y, q, key, r = map(lambda x: np.asarray(x, dtype=int), (labels, questions, keys, responses))
    p, main = np.asarray(probabilities), np.asarray(main_prob)
    p /= p.sum(-1, keepdims=True)
    wrong = r == 0
    distractor = p.copy(); distractor[np.arange(len(y)), key] = 0
    distractor /= distractor.sum(-1, keepdims=True)
    priors = np.asarray(train_dataset.option_stats['option_prior'])[q]
    reference = (1-main[:, None]) * priors
    reference[np.arange(len(y)), key] = main
    return {'events': len(y), 'wrong_events': int(wrong.sum()),
            'choice_nll': float(log_loss(y, p, labels=[0, 1, 2, 3])),
            'choice_accuracy': float((p.argmax(-1) == y).mean()),
            'wrong_option_nll': float(-np.log(np.clip(distractor[wrong, y[wrong]], 1e-12, 1)).mean()),
            'wrong_option_accuracy': float((distractor[wrong].argmax(-1) == y[wrong]).mean()),
            'wrong_option_macro_f1': float(f1_score(y[wrong], distractor[wrong].argmax(-1),
                                                  labels=[0, 1, 2, 3], average='macro', zero_division=0)),
            'question_wrong_prior_nll': float(-np.log(np.clip(priors[wrong, y[wrong]], 1e-12, 1)).mean()),
            'question_wrong_prior_accuracy': float((priors[wrong].argmax(-1) == y[wrong]).mean()),
            'main_plus_question_prior_choice_nll': float(-np.log(np.clip(reference[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'correct_option_vs_main_probability_mae': float(np.abs(p[np.arange(len(y)), key]-main).mean())}


@torch.backends.cudnn.flags(enabled=False)
def gradient_diagnostics(model, loader):
    model.eval()
    shared = list(model.interaction_emb.parameters()) + list(model.lstm_layer.parameters())
    rows = []
    for i, data in enumerate(loader):
        if i >= 4: break
        _, parts = model.compute_loss(data, return_components=True)
        vectors = {}
        for key, loss in parts.items():
            grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True) if loss.requires_grad else [None]*len(shared)
            vectors[key] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(shared, grads)])
        a, b = vectors['response'], vectors['option']
        rows.append([float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                     float(b.norm()/a.norm().clamp_min(1e-12))])
    result = np.mean(rows, axis=0)
    return {'cosine': float(result[0]), 'aux_norm_over_response': float(result[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=20); parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--emb_size', type=int, default=200); parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--option_weight', type=float, default=.1)
    parser.add_argument('--option_mode', choices=['direct', 'conditional', 'prior', 'binary_control'], default='direct')
    parser.add_argument('--save_dir', type=Path, required=True)
    args = parser.parse_args(); save = args.save_dir.resolve()
    os.chdir(ROOT/'examples'); save.mkdir(parents=True, exist_ok=True)
    if (save/'config.json').exists(): raise FileExistsError(f'Refusing overwrite: {save}')
    set_seed(args.seed); torch.set_num_threads(1)
    all_data = copy.deepcopy(json.loads((ROOT/'configs/data_config.json').read_text()))
    data = all_data['nips_task34']; data['dpath'] = str((ROOT/'examples'/data['dpath']).resolve())
    train, valid, *_ = init_dataset4train('nips_task34', 'dkt_option', all_data, args.fold, args.batch_size, aug_probs={})
    stats = train.dataset.option_stats
    config = {'emb_size': args.emb_size, 'dropout': args.dropout, 'num_q': data['num_q'],
              'option_weight': args.option_weight, 'option_mode': args.option_mode,
              'option_keys': stats['option_keys'], 'option_prior': stats['option_prior']}
    params = vars(args).copy(); params.update(save_dir=str(save), model_name='dkt_option', dataset_name='nips_task34',
                                            emb_type='qid', use_wandb=0, use_also=0)
    snapshot = {'train_config': {'batch_size': args.batch_size, 'num_epochs': args.num_epochs, 'seq_len': data['maxlen'], 'optimizer': 'adam'},
                'model_config': config, 'data_config': data, 'params': params}
    (save/'config.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    model = init_model('dkt_option', config.copy(), data, 'qid')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print('EXPERIMENT', json.dumps({**params, 'train_counts': train.dataset.counts, 'valid_counts': valid.dataset.counts}), flush=True)
    started = time.monotonic()
    result = train_model(model, train, valid, args.num_epochs, optimizer, str(save), save_model=True,
                         data_config=data, fold=args.fold, use_trained=0, also_config={'enable': False})
    elapsed = time.monotonic()-started
    model.load_state_dict(torch.load(save/'qid_model.ckpt', map_location=next(model.parameters()).device, weights_only=True))
    torch.save(model.export_dkt_state_dict(), save/'qid_dkt_backbone.ckpt')
    auc, acc = evaluate(model, valid, 'dkt_option')
    loader = DataLoader(valid.dataset, batch_size=128, shuffle=False, num_workers=0)
    diagnostics = response_diagnostics(model, loader)
    diagnostics['option'] = option_diagnostics(model, loader, train.dataset)
    diagnostics['gradient'] = gradient_diagnostics(model, loader)
    summary = {'model': 'dkt_option', 'dataset': 'nips_task34', 'fold': args.fold, 'seed': args.seed,
               'epoch_budget': args.num_epochs, 'best_epoch': int(result[-1]), 'validation_auc': float(auc),
               'validation_accuracy': float(acc), 'training_seconds': elapsed,
               'parameters': sum(p.numel() for p in model.parameters()), 'model_config': config,
               'train_sequences': len(train.dataset), 'validation_sequences': len(valid.dataset),
               'train_counts': train.dataset.counts, 'validation_counts': valid.dataset.counts,
               'validation_diagnostics': diagnostics, 'test_evaluated': False}
    (save/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('VALIDATION_SUMMARY', json.dumps({k:v for k,v in summary.items() if k!='model_config'}), flush=True)


if __name__ == '__main__': main()
