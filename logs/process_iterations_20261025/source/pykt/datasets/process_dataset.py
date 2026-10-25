"""Training-only ASSIST process targets; original KT files stay unchanged."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .first_action_dataset import FirstActionDataset, build_first_action_sidecar


def process_targets(raw):
    """Construct targets independently of folds; fit no statistics here.

    Zero attempts is a real recorded category. Time excludes zero, negatives
    and >1h intervals. Hint non-use is only meaningful when hints were offered;
    scaffold-first records are excluded from hint supervision.
    """
    def numeric(key):
        return pd.to_numeric(raw[key], errors='coerce').to_numpy(dtype=float)
    ms, attempts = numeric('ms_first_response'), numeric('attempt_count')
    hints, available = numeric('hint_count'), numeric('hint_total')
    first = numeric('first_action')
    tm = np.isfinite(ms) & (ms > 0) & (ms <= 3600000)
    am = np.isfinite(attempts) & (attempts >= 0) & (attempts == np.floor(attempts))
    hm = (np.isfinite(hints) & (hints >= 0) & (hints == np.floor(hints)) &
          np.isfinite(available) & (available > 0) & (hints <= available) &
          np.isin(first, [0, 1]))
    return {
        'process_time': np.log1p(np.where(tm, ms, 0) / 1000).astype(np.float32),
        'process_attempt': np.where(am, np.minimum(attempts, 3), 0).astype(np.int64),
        'process_hint': np.where(hm, hints > 0, 0).astype(np.float32),
        'process_time_mask': tm, 'process_attempt_mask': am, 'process_hint_mask': hm,
    }


class ProcessDataset(Dataset):
    def __init__(self, file_path, input_type, folds, mode='train', aug_probs=None,
                 process_stats=None, process_mode='joint'):
        if process_mode not in ('joint', 'conditional'):
            raise ValueError('process_mode must be joint or conditional')
        # Reuse the independently validated raw-event alignment, including
        # full sequence equality and stale legacy-cache checks. Drop action
        # fields: only process targets are exposed to this model.
        aligned = FirstActionDataset(file_path, input_type, folds, mode, aug_probs)
        self.base = aligned.base
        directory = Path(file_path).resolve().parent
        sidecar = build_first_action_sidecar(directory)
        with np.load(sidecar, allow_pickle=False) as stored:
            all_events = stored['events'].copy()
            selected = np.isin(stored['folds'], list(folds))
        # First expanded concept of each original event, across chunk borders.
        starts = np.zeros_like(all_events, dtype=bool)
        seen = set()
        for row_index, row in enumerate(all_events):
            for column, event in enumerate(row):
                if event >= 0 and int(event) not in seen:
                    starts[row_index, column] = True
                    seen.add(int(event))
        events = all_events[selected, 1:]
        eligible = torch.from_numpy(starts[selected, 1:]) & self.base.dori['masks'] & self.base.dori['smasks']
        raw = pd.read_csv(directory / 'skill_builder_data_corrected_collapsed.csv', low_memory=False)
        # Only train/validation events enter the target table; never test data.
        raw = raw[raw.order_id.isin(set(all_events[all_events >= 0].tolist()))].copy()
        if raw.order_id.duplicated().any():
            raise ValueError('Ambiguous process event identity')
        targets = process_targets(raw)
        lookup = pd.Index(raw.order_id.to_numpy(dtype=np.int64))
        positions = lookup.get_indexer(events.reshape(-1))
        if ((positions < 0) & (events.reshape(-1) >= 0)).any():
            raise ValueError('Missing original process event')
        self.fields = {}
        for name, values in targets.items():
            array = values[np.maximum(positions, 0)].reshape(events.shape).copy()
            if name.endswith('_mask'):
                array &= events >= 0
                tensor = torch.from_numpy(array) & eligible
            else:
                array[events < 0] = 0
                tensor = torch.from_numpy(array)
            self.fields[name] = tensor
        self.event_ids = torch.from_numpy(events.copy())
        # Validation receives training statistics explicitly. Requiring this
        # makes accidental validation-derived balancing/normalization fail.
        if process_stats is None:
            if mode != 'train':
                raise ValueError('Validation must receive training-fold process_stats')
            self.process_stats = self.fit_stats(process_mode)
        else:
            self.process_stats = json.loads(json.dumps(process_stats))

    def fit_stats(self, process_mode):
        f = self.fields
        times = f['process_time'][f['process_time_mask']].double()
        if times.numel() == 0:
            raise ValueError('No valid training response times')
        failure = self.base.dori['rseqs'][:, 1:] == 0
        am, hm = f['process_attempt_mask'].clone(), f['process_hint_mask'].clone()
        if process_mode == 'conditional':
            am &= failure
            hm &= failure
        return {'time_center': times.mean().item(),
                'time_scale': max(times.std(unbiased=False).item(), 1e-6),
                'attempt_counts': torch.bincount(f['process_attempt'][am], minlength=4).tolist(),
                'hint_counts': torch.bincount(f['process_hint'][hm].long(), minlength=2).tolist(),
                'time_count': times.numel(), 'process_mode': process_mode}

    @property
    def dori(self):
        return self.base.dori

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = self.base[index]
        batch.update({name: value[index] for name, value in self.fields.items()})
        return batch
