"""ASSIST completed-process history and cross-fitted distillation targets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .first_action_dataset import FirstActionDataset, build_first_action_sidecar
from .process_dataset import process_targets


def build_history_sidecar(directory):
    import fcntl
    directory = Path(directory).resolve()
    event_path = build_first_action_sidecar(directory)
    sources = json.loads(event_path.with_suffix('.json').read_text())['sources']
    output = directory / 'dkt_distill'; output.mkdir(exist_ok=True)
    archive, manifest = output/'history_v1.npz', output/'history_v1.json'
    with (output/'.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if archive.exists() and manifest.exists() and json.loads(manifest.read_text()).get('sources') == sources:
            return archive
        with np.load(event_path, allow_pickle=False) as stored:
            events, uids, folds = stored['events'].copy(), stored['uids'].copy(), stored['folds'].copy()
        raw = pd.read_csv(directory/'skill_builder_data_corrected_collapsed.csv', low_memory=False)
        raw = raw[raw.order_id.isin(set(events[events >= 0].tolist()))].copy()
        if raw.order_id.duplicated().any(): raise ValueError('Ambiguous ASSIST history event')
        targets = process_targets(raw)
        hints = pd.to_numeric(raw.hint_count, errors='coerce').to_numpy(dtype=float)
        available = pd.to_numeric(raw.hint_total, errors='coerce').to_numpy(dtype=float)
        values = np.zeros((len(raw), 11), dtype=np.float32)
        values[:, 0] = targets['process_time']; values[:, 1] = targets['process_time_mask']
        for category in range(4):
            values[:, 2+category] = (targets['process_attempt'] == category) & targets['process_attempt_mask']
        values[:, 6] = targets['process_attempt_mask']
        values[:, 7] = targets['process_hint']
        values[:, 8] = np.log1p(np.where(targets['process_hint_mask'], hints, 0))
        values[:, 9] = targets['process_hint_mask']
        values[:, 10] = np.isfinite(available) & (available > 0)
        lookup = pd.Index(raw.order_id.astype(np.int64))
        position = lookup.get_indexer(events.reshape(-1))
        if ((position < 0) & (events.reshape(-1) >= 0)).any(): raise ValueError('Missing history event')
        features = values[np.maximum(position, 0)].reshape(*events.shape, 11)
        features[events < 0] = 0
        starts = np.zeros_like(events, dtype=bool); seen = set()
        for i, row in enumerate(events):
            for j, event in enumerate(row):
                if event >= 0 and int(event) not in seen:
                    starts[i, j] = True; seen.add(int(event))
        with archive.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, features=features, events=events, starts=starts, uids=uids, folds=folds)
        archive.with_suffix('.npz.tmp').replace(archive)
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'version': 1, 'sources': sources, 'feature_dim': 11,
            'features': ['log_time','time_valid','attempt_0','attempt_1','attempt_2','attempt_3plus',
                         'attempt_valid','hint_used','log_hint_count','hint_valid','hint_available'],
            'normalization': 'fit only on unique input-history events in teacher fitting folds'}, indent=2)+'\n')
        temporary.replace(manifest)
    return archive


class HistoryProcessDataset(Dataset):
    def __init__(self, file_path, input_type, folds, mode='train', normalizer=None):
        source = Path(file_path).resolve()
        aligned = FirstActionDataset(source, input_type, folds, mode=mode, aug_probs={})
        self.base = aligned.base
        archive = build_history_sidecar(source.parent)
        with np.load(archive, allow_pickle=False) as data:
            selected = np.isin(data['folds'], list(folds))
            self.row_indices = np.flatnonzero(selected)
            self.features = data['features'][selected, :-1].copy()
            input_ids = data['events'][selected, :-1].copy()
            self.teacher_mask = torch.from_numpy(data['starts'][selected, 1:].copy())
            uids = data['uids'][selected]
        if not np.array_equal(uids, np.asarray(self.base.dori['uid'])): raise ValueError('History student order mismatch')
        self.teacher_mask &= self.base.dori['masks'] & self.base.dori['smasks']
        if normalizer is None:
            if mode != 'train': raise ValueError('Prediction requires teacher-fit-fold normalizer')
            usable = input_ids.reshape(-1) >= 0
            flat_ids = input_ids.reshape(-1)[usable]
            _, first = np.unique(flat_ids, return_index=True)
            unique_features = self.features.reshape(-1, 11)[usable][first]
            normalizer = {}
            for column, valid in [(0, 1), (8, 9)]:
                values = unique_features[unique_features[:, valid] > 0, column].astype(float)
                normalizer[str(column)] = [float(values.mean()) if len(values) else 0.,
                                            max(float(values.std()), 1e-6) if len(values) else 1.]
        self.normalizer = json.loads(json.dumps(normalizer))
        for column, valid in [(0, 1), (8, 9)]:
            center, scale = self.normalizer[str(column)]
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid history normalizer')
            self.features[..., column] = np.where(self.features[..., valid] > 0,
                                                  (self.features[..., column]-center)/scale, 0)
        self.features *= self.base.dori['masks'].numpy()[..., None]
        self.features = torch.from_numpy(self.features)

    @property
    def dori(self): return self.base.dori

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        data = self.base[index]
        data.update(history_process=self.features[index], teacher_event_mask=self.teacher_mask[index],
                    row_index=int(self.row_indices[index]))
        return data


class DistillDataset(Dataset):
    def __init__(self, file_path, input_type, folds, soft_targets=None, teacher_source='rich', mode='train'):
        if teacher_source not in ('rich', 'thin'): raise ValueError('Unknown teacher source')
        source = Path(file_path).resolve()
        aligned = FirstActionDataset(source, input_type, folds, mode=mode, aug_probs={})
        self.base = aligned.base
        shape = self.base.dori['masks'].shape
        history = build_history_sidecar(source.parent)
        with np.load(history, allow_pickle=False) as data:
            selected = np.isin(data['folds'], list(folds))
            self.event_mask = torch.from_numpy(data['starts'][selected, 1:].copy())
        self.event_mask &= self.base.dori['masks'] & self.base.dori['smasks']
        if soft_targets is None:
            if mode == 'train': raise ValueError('Training requires out-of-fold teacher targets')
            self.probabilities = torch.full(shape, .5); self.mask = torch.zeros(shape, dtype=torch.bool)
        else:
            metadata = json.loads(Path(soft_targets).with_suffix('.json').read_text())
            current = json.loads(build_history_sidecar(source.parent).with_suffix('.json').read_text())['sources']
            if metadata['sources'] != current: raise ValueError('Stale teacher soft-target sources')
            with np.load(soft_targets, allow_pickle=False) as data:
                selected = np.isin(data['folds'], list(folds))
                if not np.array_equal(data['uids'][selected], np.asarray(self.base.dori['uid'])):
                    raise ValueError('OOF target student order mismatch')
                self.probabilities = torch.from_numpy(data[teacher_source+'_probs'][selected].copy())
                self.mask = torch.from_numpy(data['mask'][selected].copy())
            if self.probabilities.shape != shape: raise ValueError('OOF target shape mismatch')
            self.mask &= self.base.dori['masks'] & self.base.dori['smasks']
            if not torch.isfinite(self.probabilities[self.mask]).all(): raise ValueError('Missing OOF teacher probabilities')

    @property
    def dori(self): return self.base.dori

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        data = self.base[index]
        data.update(teacher_probs=self.probabilities[index], teacher_mask=self.mask[index], original_event_mask=self.event_mask[index])
        return data
