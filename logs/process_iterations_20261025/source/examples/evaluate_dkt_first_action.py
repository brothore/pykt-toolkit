"""Evaluate frozen DKT/first-action checkpoints using unchanged KT metrics."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from pykt.datasets.data_loader import KTDataset
from pykt.models import evaluate, evaluate_question, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    save = args.save_dir.resolve()
    if (save / 'all_results.json').exists() or any(save.glob('qid_*predictions.txt')):
        raise FileExistsError('This frozen checkpoint already has test results; refusing to append predictions')
    snapshot = json.loads((save / 'config.json').read_text())
    data = snapshot['data_config']
    params = snapshot['params']
    name = params['model_name']
    if name not in ('dkt', 'dkt_first_action', 'dkt_process'):
        raise ValueError('This runner only supports DKT, first-action and process variants')
    torch.set_num_threads(1)
    model = load_model(name, copy.deepcopy(snapshot['model_config']), data, 'qid', str(save))
    model.eval()
    started = time.monotonic()
    result = {}
    cases = [
        ('test_file', False, 'testauc', 'testacc', 'qid_test_predictions.txt'),
        ('test_window_file', False, 'window_testauc', 'window_testacc', 'qid_test_window_predictions.txt'),
        ('test_question_file', True, 'oriauc', 'oriacc', 'qid_test_question_predictions.txt'),
        ('test_question_window_file', True, 'windowauc', 'windowacc', 'qid_test_question_window_predictions.txt'),
    ]
    for file_key, qtest, auc_key, acc_key, prediction_file in cases:
        dataset = KTDataset(str(Path(data['dpath']) / data[file_key]), data['input_type'], {-1}, qtest=qtest)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if qtest:
            auc, acc = evaluate_question(model, loader, name, ['late_fusion'], str(save / prediction_file))
            result.update({auc_key + key: float(value) for key, value in auc.items()})
            result.update({acc_key + key: float(value) for key, value in acc.items()})
        else:
            auc, acc = evaluate(model, loader, name, save_path=str(save / prediction_file))
            result[auc_key], result[acc_key] = float(auc), float(acc)
        print('FINISHED', file_key, json.dumps(result), flush=True)
    assert 'windowauclate_mean' in result
    (save / 'all_results.json').write_text(json.dumps(result, indent=2) + '\n')
    (save / 'test_runtime.json').write_text(json.dumps({'seconds': time.monotonic() - started}) + '\n')
    print('PRIMARY_METRIC', result['windowauclate_mean'], flush=True)


if __name__ == '__main__':
    main()
