"""Exhaustive validation audit against native preprocessing, loader and evaluate.

No test labels are read. Resume at completed UID shards; native evaluate is
unmodified. The hook records its actual question scores for an aligned check.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from evaluate_validation import load_run, load_saved_model
from evaluate_window import QuestionWindows
from prediction_progress import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.preprocess.split_datasets_que import generate_window_sequences
from pykt.models import evaluate
from audit_inputs import sequence_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    config, paths = load_run(args.checkpoint_dir)
    cfg = config['data_config']
    identity = {'reference': digest(args.reference), 'checkpoint': digest(paths['checkpoint_path']),
                'raw': digest(paths['raw_path']), 'source': digest(Path(__file__)),
                'native_sources': {str(p): digest(p) for p in [ROOT/'pykt/preprocess/split_datasets_que.py',
                    ROOT/'pykt/datasets/que_data_loader.py', ROOT/'pykt/models/evaluate_model.py']}}
    ip = out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Audit identity changed')
    atomic_json(ip, identity)
    raw = pd.read_csv(paths['raw_path']); raw = raw[raw.fold == paths['fold']].copy()
    reference = pd.read_csv(args.reference, sep='\t')
    if reference.duplicated(['orirow', 'qidx']).any(): raise ValueError('Duplicate reference')
    reference = reference.set_index(['orirow', 'qidx'])
    model = load_saved_model(config, paths, device='cuda')
    original = model.predict_one_step
    summaries = []
    for start in range(0, len(raw), 16):
        shard = out/f'shard_{start:06d}.json'
        if shard.exists(): summaries.append(json.loads(shard.read_text())); continue
        block = raw.iloc[start:start+16]
        native_rows = generate_window_sequences(block, list(block.columns), int(cfg.get('maxlen', 200)))
        file = out/f'native_{start:06d}.csv'; native_rows.to_csv(file, index=False)
        native = KTQueDataset(str(file), cfg['input_type'], [paths['fold']],
                              cfg['num_c'], cfg['max_concepts'], mode='test')
        stream = QuestionWindows(block, int(cfg.get('maxlen', 200)), cfg['max_concepts'])
        if len(native) != len(stream): raise ValueError('Window count differs')
        keys = []; labels = []
        for i in range(len(native)):
            a = native[i]; b, meta = stream[i]
            for key in ['qseqs','shft_qseqs','cseqs','shft_cseqs','rseqs','shft_rseqs','masks','smasks']:
                if not torch.equal(a[key], b[key]): raise ValueError(f'Tensor mismatch: {start}/{i}/{key}')
            mask = a['smasks'].bool()
            keys.extend((int(meta['orirow']), int(q)) for q in meta['qidx'][mask])
            labels.extend(a['shft_rseqs'][mask].tolist())
        scores = []
        def capture(data, *pos, **kw):
            prediction = original(data, *pos, **kw)
            scores.extend(prediction.detach().cpu()[data['smasks'].cpu().bool()].tolist())
            return prediction
        model.predict_one_step = capture
        try:
            auc, acc = evaluate(model, DataLoader(native, batch_size=32, shuffle=False, num_workers=0), model.model_name)
        finally:
            model.predict_one_step = original
        selected = reference.loc[keys]
        if not np.array_equal(labels, selected.late_trues.to_numpy()): raise ValueError('Labels differ')
        scores = np.asarray(scores); expected = selected.late_mean.to_numpy()
        error = float(np.max(np.abs(scores-expected)))
        expected_auc = float(roc_auc_score(labels, expected))
        if error > 1e-6:
            raise ValueError(f'Prediction discrepancy: max={error}, AUC={auc-expected_auc}')
        pd.DataFrame({'orirow':[k[0] for k in keys], 'qidx':[k[1] for k in keys],
                      'late_trues':labels, 'late_mean':scores}).to_csv(out/f'scores_{start:06d}.tsv', sep='\t', index=False)
        record = {'start':start,'uids':list(map(int,block.uid)),'events':len(scores),
                  'windows':len(native),'max_prediction_error':error,'native_auc':float(auc),
                  'reference_auc':expected_auc,'native_accuracy':float(acc)}
        atomic_json(shard, record); summaries.append(record)
        # Only temporary audit material; preserve completed summaries and identities.
        for p in out.glob(file.name+'*'): p.unlink()
    count = sum(s['events'] for s in summaries)
    if count != len(reference): raise ValueError('Coverage mismatch')
    scores = pd.concat([pd.read_csv(out/f'scores_{s["start"]:06d}.tsv', sep='\t') for s in summaries])
    native_cohorts, _ = sequence_metrics(scores)
    ref = reference.reset_index()
    reference_cohorts, _ = sequence_metrics(ref)
    native_auc = float(roc_auc_score(scores.late_trues, scores.late_mean))
    reference_auc = float(roc_auc_score(ref.late_trues, ref.late_mean))
    deltas = {k:native_cohorts['primary_20_5'][k]-reference_cohorts['primary_20_5'][k]
              for k in ('mean','std','range')}
    if abs(native_auc-reference_auc)>1e-6 or abs(deltas['std'])>1e-5 or abs(deltas['range'])>1e-4:
        raise ValueError(f'Metric drift exceeds numerical audit budget: {deltas}')
    result = {'passed':True,'source':'validation','test_evaluated':False,
              'scope':'Exhaustive native preprocessing + KTQueDataset tensors + native evaluate scores',
              'events':count,'windows':sum(s['windows'] for s in summaries),
              'uids':sum(len(s['uids']) for s in summaries),
              'max_prediction_error':max(s['max_prediction_error'] for s in summaries),
              'native_validation_pooled_auc':native_auc, 'reference_validation_pooled_auc':reference_auc,
              'native_minus_reference_primary_metrics':deltas,
              'exact_metric_parity':native_auc==reference_auc and all(v==0 for v in deltas.values()),
              'numerical_budget':{'max_prediction_error':1e-6,'pooled_auc':1e-6,'std':1e-5,'range':1e-4},
              'limitation':'Generic question loader is None for QIKT; this verifies its supported native question-level path, not an unmodified late-fusion script output.',
              'identity':identity}
    atomic_json(out/'audit_result.json', result)


if __name__ == '__main__': main()
