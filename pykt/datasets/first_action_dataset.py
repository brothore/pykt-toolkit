"""ASSIST first-action targets, aligned by original events without rewriting KT data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loader import KTDataset


def _signature(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ints(value):
    return [int(x) for x in str(value).split(',')]


def build_first_action_sidecar(data_dir):
    """Return immutable labels for train_valid sequences; never build test labels.

    Reconstruct each student's complete original event order, check every mapped
    question/concept/response against the existing sequence, and only then slice.
    Consecutive retries of the same question remain distinct via order_id.
    """
    import fcntl

    directory = Path(data_dir).resolve()
    sources = {
        'raw': directory / 'skill_builder_data_corrected_collapsed.csv',
        'mapping': directory / 'keyid2idx.json',
        'original': directory / 'train_valid.csv',
        'sequences': directory / 'train_valid_sequences.csv',
    }
    output = directory / 'dkt_first_action'
    output.mkdir(exist_ok=True)
    archive, manifest = output / 'targets_v1.npz', output / 'targets_v1.json'
    with (output / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signatures = {name: _signature(path) for name, path in sources.items()}
        if archive.exists() and manifest.exists():
            stored = json.loads(manifest.read_text())
            if stored.get('version') == 1 and stored.get('sources') == signatures:
                return archive

        mapping = json.loads(sources['mapping'].read_text())
        original = pd.read_csv(sources['original'])
        sequences = pd.read_csv(sources['sequences'])
        allowed_uids = set(original.uid.astype(int))
        inverse_uid = {str(raw): int(mapped) for raw, mapped in mapping['uid'].items()
                       if int(mapped) in allowed_uids}
        raw = pd.read_csv(sources['raw'], dtype=str)
        raw = raw.dropna(subset=['user_id', 'problem_id', 'skill_id', 'correct', 'order_id'])
        raw = raw[raw.user_id.isin(inverse_uid)].copy()
        if raw.order_id.duplicated().any():
            raise ValueError('Expected unique ASSIST order_id; refusing ambiguous event alignment')
        raw['_raw_index'] = raw.index
        raw['_uid'] = raw.user_id.map(inverse_uid)
        full = {}
        for uid, group in raw.groupby('_uid', sort=False):
            group = group.sort_values(['order_id', '_raw_index'], kind='stable')
            qs, cs, rs, labels, starts, event_ids = [], [], [], [], [], []
            for row in group.itertuples(index=False):
                response = int(row.correct)
                if response not in (0, 1):
                    raise ValueError('Non-binary raw response in eligible ASSIST records')
                if response == 1 and row.first_action == '0':
                    label = 0
                elif response == 0 and row.first_action in ('0', '1', '2'):
                    label = int(row.first_action) + 1
                else:
                    label = -100  # Unknown/inconsistent behavior is not a success label.
                concepts = [int(mapping['concepts'][c]) for c in row.skill_id.split('_')]
                question = int(mapping['questions'][row.problem_id])
                for j, concept in enumerate(concepts):
                    qs.append(question)
                    cs.append(concept)
                    rs.append(response)
                    labels.append(label)
                    starts.append(j == 0 and label >= 0)
                    event_ids.append(int(row.order_id))
            full[int(uid)] = (qs, cs, rs, labels, starts, event_ids)

        for row in original.itertuples(index=False):
            expected = full[int(row.uid)]
            if any(_ints(getattr(row, name)) != expected[i]
                   for i, name in enumerate(('questions', 'concepts', 'responses'))):
                raise ValueError(f'Raw/processed complete sequence mismatch for mapped uid={row.uid}')

        label_rows, start_rows, event_rows, uid_rows, folds = [], [], [], [], []
        cursors = {}
        for row in sequences.itertuples(index=False):
            uid = int(row.uid)
            qs, cs, rs = _ints(row.questions), _ints(row.concepts), _ints(row.responses)
            valid_n = next((i for i, x in enumerate(rs) if x == -1), len(rs))
            if any(x != -1 for x in rs[valid_n:]):
                raise ValueError('Only right padding is supported')
            start = cursors.get(uid, 0)
            end = start + valid_n
            expected = full[uid]
            if any(values[:valid_n] != expected[i][start:end]
                   for i, values in enumerate((qs, cs, rs))):
                raise ValueError(f'Raw/processed chunk mismatch for mapped uid={uid}, offset={start}')
            starts = expected[4][start:end]
            if hasattr(row, 'is_repeat'):
                repeats = _ints(row.is_repeat)[:valid_n]
                # All four labels are valid in this dataset. Unknown labels may be masked.
                expected_repeats = [0 if (j == 0 or expected[5][j-1] != expected[5][j]) else 1
                                    for j in range(start, end)]
                if repeats != expected_repeats:
                    raise ValueError('Processed is_repeat disagrees with original event boundaries')
            padding = len(rs) - valid_n
            label_rows.append(expected[3][start:end] + [-100] * padding)
            start_rows.append(starts + [False] * padding)
            event_rows.append(expected[5][start:end] + [-1] * padding)
            uid_rows.append(uid)
            folds.append(int(row.fold))
            cursors[uid] = end
        if len(set(map(len, label_rows))) != 1:
            raise ValueError('Sequence lengths must match')
        with archive.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, labels=np.asarray(label_rows, dtype=np.int64),
                                mask=np.asarray(start_rows, dtype=bool),
                                events=np.asarray(event_rows, dtype=np.int64),
                                uids=np.asarray(uid_rows, dtype=np.int64),
                                folds=np.asarray(folds, dtype=np.int64))
        archive.with_suffix('.npz.tmp').replace(archive)
        metadata = {'version': 1, 'sources': signatures, 'rows': len(sequences),
                    'students': len(allowed_uids), 'raw_events': len(raw),
                    'classes': ['first_correct', 'attempt_incorrect', 'hint_first', 'scaffold_first'],
                    'note': 'Only train/validation students; class weighting must use training folds only.'}
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(manifest)
    return archive


class FirstActionDataset(Dataset):
    """A transparent wrapper; standard KTDataset and its caches are untouched."""
    def __init__(self, file_path, input_type, folds, mode='train', aug_probs=None):
        if aug_probs and any(float(value) != 0 for value in aug_probs.values()):
            raise ValueError('First-action alignment currently requires augmentation disabled')
        file_path = Path(file_path).resolve()
        if file_path.name != 'train_valid_sequences.csv':
            raise ValueError('First-action supervision is supported only for train_valid_sequences.csv')
        sidecar = build_first_action_sidecar(file_path.parent)
        self.base = KTDataset(str(file_path), input_type, folds, mode=mode, aug_probs=aug_probs)
        with np.load(sidecar, allow_pickle=False) as data:
            selected = np.isin(data['folds'], list(folds))
            self.action_labels = torch.from_numpy(data['labels'][selected, 1:].copy())
            self.action_mask = torch.from_numpy(data['mask'][selected, 1:].copy())
            self.event_ids = torch.from_numpy(data['events'][selected, 1:].copy())
            uids = data['uids'][selected]
        if len(uids) != len(self.base) or not np.array_equal(uids, np.asarray(self.base.dori['uid'])):
            raise ValueError('Base dataset and sidecar student order disagree')
        # The legacy pickle cache has no source hash. A reprocessed CSV with the
        # same students must not silently pair new actions with old responses.
        current = pd.read_csv(file_path)
        current = current[current['fold'].isin(folds)]
        for tensor_key, csv_key in [('qseqs', 'questions'), ('cseqs', 'concepts'), ('rseqs', 'responses')]:
            expected = torch.tensor([_ints(value) for value in current[csv_key]],
                                    dtype=self.base.dori[tensor_key].dtype)
            if not torch.equal(expected, self.base.dori[tensor_key]):
                raise ValueError(f'Stale KTDataset cache for {csv_key}; regenerate that cache before training')
        selected_targets = torch.tensor([_ints(value) for value in current['selectmasks']])[:, 1:] != -1
        if not torch.equal(selected_targets, self.base.dori['smasks']):
            raise ValueError('Stale KTDataset cache for selectmasks; regenerate that cache before training')
        self.action_mask &= self.base.dori['masks'] & self.base.dori['smasks']
        selected_labels = self.action_labels[self.action_mask]
        self.action_class_counts = torch.bincount(selected_labels, minlength=4).tolist()

    @property
    def dori(self):
        return self.base.dori

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        data = self.base[index]
        data['action_labels'] = self.action_labels[index]
        data['action_mask'] = self.action_mask[index]
        return data
