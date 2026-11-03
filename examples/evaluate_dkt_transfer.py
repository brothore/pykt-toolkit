"""Frozen transfer DKT checkpoints: unchanged primary window/question metric."""
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
from pykt.models import evaluate_question, load_model


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128); args = parser.parse_args()
    save = args.save_dir.resolve(); prediction = save/'qid_test_question_window_predictions.txt'
    if prediction.exists() or (save/'all_results.json').exists(): raise FileExistsError('Frozen test results already exist')
    config = json.loads((save/'config.json').read_text()); data = config['data_config']; name = config['params']['model_name']
    if name not in ('dkt', 'dkt_transfer'): raise ValueError('Expected DKT/transfer model')
    torch.set_num_threads(1)
    model = load_model(name, copy.deepcopy(config['model_config']), data, 'qid', str(save)); model.eval()
    dataset = KTDataset(str(Path(data['dpath'])/data['test_question_window_file']), data['input_type'], {-1}, qtest=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    started = time.monotonic()
    auc, acc = evaluate_question(model, loader, name, ['late_fusion'], str(prediction))
    result = {'windowauc'+key: float(value) for key,value in auc.items()}
    result.update({'windowacc'+key: float(value) for key,value in acc.items()})
    assert 'windowauclate_mean' in result
    (save/'all_results.json').write_text(json.dumps(result, indent=2)+'\n')
    (save/'test_runtime.json').write_text(json.dumps({'seconds': time.monotonic()-started,
         'scope': 'primary question-window only; original evaluate_question and late_fusion', 'batch_size': args.batch_size}, indent=2)+'\n')
    print('PRIMARY_METRIC', result['windowauclate_mean'], flush=True)


if __name__ == '__main__': main()
