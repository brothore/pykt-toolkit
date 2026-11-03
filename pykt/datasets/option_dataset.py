"""NIPS selected-option targets with training-fold-only stable answer keys."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .confidence_dataset import ConfidenceDataset


class OptionDataset(Dataset):
    def __init__(self, file_path, input_type, folds, mode='train', aug_probs=None, option_stats=None):
        # Reuse original AnswerId/time-tie alignment and full q/c/r/time checks.
        aligned = ConfidenceDataset(file_path, input_type, folds, mode, aug_probs)
        self.base = aligned.base
        self.event_ids = aligned.event_ids
        self.event_mask = aligned.event_mask.clone()
        directory = Path(file_path).resolve().parent
        raw = pd.read_csv(directory / 'train_task_3_4.csv',
                          usecols=['AnswerId', 'AnswerValue', 'CorrectAnswer', 'IsCorrect'])
        # Limit all newly constructed targets to this dataset's fold events.
        ids = self.event_ids.numpy()
        raw = raw[raw.AnswerId.isin(set(ids[ids >= 0].tolist()))].copy()
        if raw.AnswerId.duplicated().any():
            raise ValueError('Duplicate AnswerId in option source')
        lookup = pd.Index(raw.AnswerId)
        positions = lookup.get_indexer(ids.reshape(-1))
        if ((positions < 0) & (ids.reshape(-1) >= 0)).any():
            raise ValueError('Missing option AnswerId')
        selected, correct, responses = [], [], []
        for key, output in [('AnswerValue', selected), ('CorrectAnswer', correct), ('IsCorrect', responses)]:
            values = pd.to_numeric(raw[key], errors='coerce').to_numpy(dtype=float)
            output.append(values[np.maximum(positions, 0)].reshape(ids.shape))
        selected, correct, responses = selected[0], correct[0], responses[0]
        valid = np.isin(selected, [1, 2, 3, 4]) & np.isin(correct, [1, 2, 3, 4]) & np.isin(responses, [0, 1])
        event_mask = self.event_mask.numpy()
        if ((selected == correct)[valid & event_mask].astype(int) != responses[valid & event_mask]).any():
            raise ValueError('Selected/correct option does not match IsCorrect')
        base_response = self.base.dori['rseqs'][:, 1:].numpy()
        if (responses[event_mask] != base_response[event_mask]).any():
            raise ValueError('Aligned option event and KT response disagree')
        self.labels = torch.from_numpy(np.where(valid, selected - 1, -100).astype(np.int64))
        actual_keys = torch.from_numpy(np.where(valid, correct - 1, -1).astype(np.int64))
        self.raw_mask = self.event_mask & torch.from_numpy(valid)
        q = self.base.dori['qseqs'][:, 1:].long()
        # keyid2idx defines all question ids, including questions not seen in
        # this fold. Unknown keys stay -1; validation cannot fit them.
        num_q = len(json.loads((directory / 'keyid2idx.json').read_text())['questions'])
        if option_stats is None:
            if mode != 'train':
                raise ValueError('Validation requires training-fold option_stats')
            keys = torch.full((num_q,), -1, dtype=torch.long)
            key_counts = torch.zeros(num_q, 4, dtype=torch.long)
            active_q, active_k = q[self.raw_mask], actual_keys[self.raw_mask]
            key_counts.index_put_((active_q, active_k), torch.ones_like(active_q), accumulate=True)
            stable = (key_counts > 0).sum(-1) == 1
            keys[stable] = key_counts[stable].argmax(-1)
            stable_mask = self.raw_mask & stable[q]
            wrong_mask = stable_mask & (self.base.dori['rseqs'][:, 1:] == 0)
            counts = torch.zeros(num_q, 4, dtype=torch.float64)
            wrong_q, wrong_choice = q[wrong_mask], self.labels[wrong_mask]
            counts.index_put_((wrong_q, wrong_choice), torch.ones_like(wrong_q, dtype=torch.float64), accumulate=True)
            # Smoothed within-question distractor prior. Known correct key
            # always excluded, even if its raw global position is popular.
            candidates = torch.ones(num_q, 4, dtype=torch.float64)
            candidates[stable, keys[stable]] = 0
            prior = counts + candidates  # one pseudo-count per wrong option
            prior /= prior.sum(-1, keepdim=True)
            self.option_stats = {'option_keys': keys.tolist(), 'option_prior': prior.tolist(),
                                 'question_wrong_counts': counts.long().tolist(),
                                 'stable_questions': int(stable.sum()),
                                 'unstable_questions': int(((key_counts > 0).sum(-1) > 1).sum()),
                                 'unknown_questions': int((key_counts.sum(-1) == 0).sum()),
                                 'source': 'training-fold eligible original events only',
                                 'smoothing': 'one pseudo-count per distractor'}
        else:
            self.option_stats = copy.deepcopy(option_stats)
        keys = torch.tensor(self.option_stats['option_keys'], dtype=torch.long)
        if keys.numel() != num_q:
            raise ValueError('Training option keys do not match question mapping')
        self.mask = self.raw_mask & (keys[q] >= 0) & (actual_keys == keys[q])
        self.wrong_mask = self.mask & (self.base.dori['rseqs'][:, 1:] == 0)
        self.counts = {'raw_events': int(self.event_mask.sum()), 'eligible': int(self.mask.sum()),
                       'wrong': int(self.wrong_mask.sum()),
                       'excluded_unstable_unknown_or_key_mismatch': int((self.raw_mask & ~self.mask).sum()),
                       'option_counts': torch.bincount(self.labels[self.mask], minlength=4).tolist()}

    @property
    def dori(self):
        return self.base.dori

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        data = self.base[index]
        data.update(option_labels=self.labels[index], option_mask=self.mask[index])
        return data
