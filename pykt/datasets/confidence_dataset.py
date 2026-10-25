"""NIPS task34 confidence labels, aligned to original AnswerId events."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loader import KTDataset
from .first_action_dataset import _ints, _signature


def confidence_labels(values):
    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    valid = np.isin(values, [0, 25, 50, 75, 100])
    return np.where(valid, np.where(valid, values, 0) / 25, -100).astype(np.int64)


def build_confidence_sidecar(data_dir):
    """Verify complete q/c/r/time sequences before slicing labels into chunks.

    Match timestamp ties using original primary CSV order, exactly as NIPS
    preprocessing does. Only train/validation students receive target labels.
    """
    import fcntl
    directory = Path(data_dir).resolve()
    sources = {'primary': directory / 'train_task_3_4.csv',
               'answers': directory / 'metadata/answer_metadata_task_3_4.csv',
               'questions': directory / 'metadata/question_metadata_task_3_4.csv',
               'subjects': directory / 'metadata/subject_metadata.csv',
               'mapping': directory / 'keyid2idx.json', 'original': directory / 'train_valid.csv',
               'sequences': directory / 'train_valid_sequences.csv'}
    output = directory / 'dkt_confidence'
    output.mkdir(exist_ok=True)
    archive, manifest = output / 'targets_v1.npz', output / 'targets_v1.json'
    with (output / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        signatures = {key: _signature(path) for key, path in sources.items()}
        if archive.exists() and manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved.get('version') == 1 and saved.get('sources') == signatures:
                return archive
        mapping = json.loads(sources['mapping'].read_text())
        original, sequences = pd.read_csv(sources['original']), pd.read_csv(sources['sequences'])
        allowed = set(original.uid.astype(int))
        uids = {int(raw): int(mapped) for raw, mapped in mapping['uid'].items() if int(mapped) in allowed}
        raw = pd.read_csv(sources['primary'])
        raw['_raw_index'] = np.arange(len(raw))
        raw = raw[raw.UserId.isin(uids)].copy()
        if raw.AnswerId.duplicated().any():
            raise ValueError('Duplicate primary AnswerId; ambiguous confidence alignment')
        answer = pd.read_csv(sources['answers'])
        if answer.AnswerId.duplicated().any():
            raise ValueError('Duplicate metadata AnswerId')
        raw = raw.merge(answer[['AnswerId', 'DateAnswered', 'Confidence']], on='AnswerId',
                        how='left', validate='one_to_one', indicator=True, sort=False)
        if not raw['_merge'].eq('both').all():
            raise ValueError('Missing answer metadata')
        raw['_timestamp'] = pd.to_datetime(raw.DateAnswered, errors='raise').dt.as_unit('ms').astype('int64')
        raw['_uid'] = raw.UserId.map(uids)
        raw['_label'] = confidence_labels(raw.Confidence)
        subjects = pd.read_csv(sources['subjects'])
        level3 = set(subjects.loc[subjects.Level == 3, 'SubjectId'].astype(int))
        questions = pd.read_csv(sources['questions'])
        if questions.QuestionId.duplicated().any():
            raise ValueError('Duplicate question metadata')
        # Preserve the original preprocessor's integer set-intersection order.
        concept_map = {int(row.QuestionId): [int(mapping['concepts'][str(c)])
                       for c in set(ast.literal_eval(row.SubjectId)) & level3]
                       for row in questions.itertuples(index=False)}
        full = {}
        for uid, group in raw.groupby('_uid', sort=False):
            group = group.sort_values(['_timestamp', '_raw_index'], kind='stable')
            qs, cs, rs, ts, labels, starts, events = [], [], [], [], [], [], []
            # to_dict avoids pandas renaming leading-underscore fields.
            for row in group.to_dict('records'):
                concepts = concept_map[int(row['QuestionId'])]
                if not concepts:
                    raise ValueError('No level-3 concept for raw NIPS question')
                question = int(mapping['questions'][str(int(row['QuestionId']))])
                response = int(row['IsCorrect'])
                if response not in (0, 1):
                    raise ValueError('Nonbinary NIPS response')
                for j, concept in enumerate(concepts):
                    qs.append(question); cs.append(concept); rs.append(response)
                    ts.append(int(row['_timestamp'])); labels.append(int(row['_label']))
                    starts.append(j == 0); events.append(int(row['AnswerId']))
            full[int(uid)] = (qs, cs, rs, ts, labels, starts, events)
        keys = ('questions', 'concepts', 'responses', 'timestamps')
        for row in original.itertuples(index=False):
            expected = full[int(row.uid)]
            if any(_ints(getattr(row, key)) != expected[i] for i, key in enumerate(keys)):
                raise ValueError(f'Raw/processed complete confidence sequence mismatch uid={row.uid}')
        label_rows, masks, event_rows, uid_rows, folds, cursors = [], [], [], [], [], {}
        for row in sequences.itertuples(index=False):
            uid = int(row.uid)
            values = [_ints(getattr(row, key)) for key in keys]
            n = next((i for i, value in enumerate(values[2]) if value == -1), len(values[2]))
            if any(value != -1 for value in values[2][n:]):
                raise ValueError('Expected right padding')
            start, expected = cursors.get(uid, 0), full[uid]
            end = start + n
            if any(value[:n] != expected[i][start:end] for i, value in enumerate(values)):
                raise ValueError(f'Raw/processed confidence chunk mismatch uid={uid}, offset={start}')
            if hasattr(row, 'is_repeat'):
                repeats = [int(not expected[5][i]) for i in range(start, end)]
                if _ints(row.is_repeat)[:n] != repeats:
                    raise ValueError('NIPS is_repeat disagrees with raw event identity')
            padding = len(values[2]) - n
            label_rows.append(expected[4][start:end] + [-100] * padding)
            masks.append(expected[5][start:end] + [False] * padding)
            event_rows.append(expected[6][start:end] + [-1] * padding)
            uid_rows.append(uid); folds.append(int(row.fold)); cursors[uid] = end
        with archive.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, labels=np.asarray(label_rows, dtype=np.int64),
                                mask=np.asarray(masks, dtype=bool), events=np.asarray(event_rows, dtype=np.int64),
                                uids=np.asarray(uid_rows, dtype=np.int64), folds=np.asarray(folds, dtype=np.int64))
        archive.with_suffix('.npz.tmp').replace(archive)
        saved = {'version': 1, 'sources': signatures, 'students': len(allowed), 'raw_events': len(raw),
                 'observed_confidence': int((raw['_label'] >= 0).sum()),
                 'classes': [0, 25, 50, 75, 100], 'tie_breaker': 'primary CSV original row index',
                 'note': 'Train/validation only; confidence zero is valid, missing is -100.'}
        tmp = manifest.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(saved, indent=2) + '\n'); tmp.replace(manifest)
    return archive


class ConfidenceDataset(Dataset):
    def __init__(self, file_path, input_type, folds, mode='train', aug_probs=None):
        if aug_probs and any(float(value) != 0 for value in aug_probs.values()):
            raise ValueError('Confidence alignment requires augmentation disabled')
        source = Path(file_path).resolve()
        if source.name != 'train_valid_sequences.csv':
            raise ValueError('Confidence training requires train_valid_sequences.csv')
        sidecar = build_confidence_sidecar(source.parent)
        self.base = KTDataset(str(source), input_type, folds, mode=mode, aug_probs=aug_probs)
        with np.load(sidecar, allow_pickle=False) as stored:
            selected = np.isin(stored['folds'], list(folds))
            self.labels = torch.from_numpy(stored['labels'][selected, 1:].copy())
            self.mask = torch.from_numpy(stored['mask'][selected, 1:].copy())
            self.event_ids = torch.from_numpy(stored['events'][selected, 1:].copy())
            uids = stored['uids'][selected]
        if not np.array_equal(uids, np.asarray(self.base.dori['uid'])):
            raise ValueError('Confidence sidecar and legacy cache uid order differ')
        current = pd.read_csv(source)
        current = current[current.fold.isin(folds)]
        for tensor, column in [('qseqs', 'questions'), ('cseqs', 'concepts'), ('rseqs', 'responses'), ('tseqs', 'timestamps')]:
            expected = torch.tensor([_ints(value) for value in current[column]], dtype=self.base.dori[tensor].dtype)
            if not torch.equal(expected, self.base.dori[tensor]):
                raise ValueError(f'Stale KTDataset cache for {column}')
        select = torch.tensor([_ints(value) for value in current.selectmasks])[:, 1:] != -1
        if not torch.equal(select, self.base.dori['smasks']):
            raise ValueError('Stale KTDataset cache for selectmasks')
        self.event_mask = self.mask & self.base.dori['masks'] & self.base.dori['smasks']
        self.mask = self.event_mask & (self.labels >= 0)
        self.confidence_counts = torch.bincount(self.labels[self.mask], minlength=5).tolist()
        self.coverage = float(self.mask.sum() / self.event_mask.sum().clamp_min(1))

    @property
    def dori(self):
        return self.base.dori

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        data = self.base[index]
        data.update(confidence_labels=self.labels[index], confidence_mask=self.mask[index])
        return data
