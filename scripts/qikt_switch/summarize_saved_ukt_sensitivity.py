"""Recover UID balance from saved sensitivity scores without new inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pykt.datasets.data_loader import KTDataset
from audit_inputs import sequence_metrics
from prediction_progress import atomic_json, digest


def summarize(source, output):
    summary = json.loads((source / 'summary.json').read_text())
    rawpath = source / 'native_train_valid_sequences.csv'
    if digest(rawpath) != summary['source_identity']['data']:
        raise ValueError('Saved sensitivity data identity differs')
    raw = pd.read_csv(rawpath)
    raw = raw[raw.fold == summary['fold']].reset_index(drop=True)
    dataset = KTDataset(str(rawpath), ['questions', 'concepts'],
        {summary['fold']}, mode='test', aug_probs={})
    if len(dataset) != len(raw):
        raise ValueError('Fragment metadata alignment differs')
    ids, labels, hashes = [], [], {}
    for index, batch in enumerate(DataLoader(dataset, batch_size=32)):
        mask = batch['smasks'].bool().numpy()
        target = batch['shft_rseqs'].numpy()[mask]
        labels.append(target)
        uids = raw.iloc[index*32:index*32+len(mask)].uid.to_numpy()
        ids.append(np.broadcast_to(uids[:, None], mask.shape)[mask])
    labels, ids = np.concatenate(labels), np.concatenate(ids)
    results = []
    for result in summary['results']:
        mode = result['mode']
        files = sorted((source / mode).glob('batch_*.npz'))
        if len(files) != len(DataLoader(dataset, batch_size=32)):
            raise ValueError('Saved batch coverage differs')
        scores = []
        offset = 0
        for path in files:
            with np.load(path) as part:
                count = len(part['labels'])
                if not np.array_equal(part['labels'], labels[offset:offset+count]):
                    raise ValueError('Saved labels differ from reconstructed order')
                scores.append(part['scores'])
                offset += count
            hashes[str(path)] = digest(path)
        score = np.concatenate(scores)
        if offset != len(ids):
            raise ValueError('UID/prediction coverage differs')
        if abs(roc_auc_score(labels, score) - result['native_validation_fragment_auc']) > 1e-12:
            raise ValueError('Reconstructed pooled AUC differs from saved result')
        frame = pd.DataFrame({'orirow': ids, 'qidx': np.arange(len(ids)),
            'late_trues': labels, 'late_mean': score})
        cohorts, sequences = sequence_metrics(frame)
        output.mkdir(parents=True, exist_ok=True)
        sequences.to_csv(output / (mode + '_sequence_auc.csv'), index=False)
        results.append({**result, 'cohorts': cohorts})
    value = {'analysis_source_sha256': digest(Path(__file__)), 'results': results, 'source_summary_sha256': digest(source / 'summary.json'),
        'data_sha256': digest(rawpath), 'saved_batch_sha256': hashes,
        'label_and_uid_alignment_verified': True, 'new_training': False,
        'new_gpu_inference': False, 'formal_windowauclate_mean': None,
        'scope': 'Original native concept fragments; not strict question windows or retrained ablations'}
    atomic_json(output / 'summary.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.output)
