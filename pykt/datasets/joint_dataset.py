"""Joint supervision over the existing ASSIST and NIPS task-3/4 datasets.

This module is deliberately a *composition* layer.  It does not preprocess,
re-slice, re-order, or otherwise reinterpret any raw data.  Each auxiliary
signal is produced by the already-validated dataset wrapper for that signal,
and this layer refuses to combine them unless their shared KT base agrees
exactly (uid, q/c/r sequences, masks and smasks).

The original KT fields returned by ``self.base[index]`` are never replaced.
Only auxiliary keys that do not exist in the base batch are added.

Supported dataset names
-----------------------
``assist2009`` (aliases: ``assist``, ``assist09``)
    FirstActionDataset + ProcessDataset + TransferDataset + DistillDataset
``nips_task34`` (aliases: ``nips``, ``nips34``, ``nips_task3_4``)
    ConfidenceDataset + OptionDataset + ProfileDataset(profile_mode='hierarchy')

Training statistics
-------------------
For ``mode='train'`` the process/option statistics are fitted here from the
training folds only and mirrored into ``self.stats``.  For ``mode='valid'`` a
``stats`` dictionary produced by a training ``JointDataset`` is mandatory; the
validation loader never fits statistics on its own events.

Distillation cache
------------------
The default ASSIST teacher cache is the rich OOF bundle used by
``examples/train_dkt_distill.py``::

    examples/saved_model/distill_iterations_20261026/teachers/soft_targets.npz

That bundle only contains out-of-fold probabilities for folds 1-4.  Fold 0
has no teacher labels, so a validation JointDataset on fold 0 falls back to
``DistillDataset(..., soft_targets=None)`` (all-false teacher mask), exactly as
the original validation entry point did.  A validation JointDataset on folds
1-4 does load the OOF cache because those rows have real teacher labels.

Collation
---------
``JointDataset.collate_fn`` (also exposed as ``JointDataset.collate`` and the
module-level ``joint_collate``) dispatches to ``profile_collate`` whenever the
batch contains variable-length ``profile_*`` fields, and to ``default_collate``
otherwise.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .confidence_dataset import ConfidenceDataset
from .distill_dataset import DistillDataset
from .first_action_dataset import FirstActionDataset
from .option_dataset import OptionDataset
from .process_dataset import ProcessDataset
from .profile_dataset import ProfileDataset, profile_collate
from .transfer_dataset import MODES as TRANSFER_MODES, TransferDataset

__all__ = ['JointDataset', 'joint_collate']

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEACHER_CANDIDATES = (
    _PROJECT_ROOT / 'examples' / 'saved_model' / 'distill_iterations_20261026' /
    'teachers' / 'soft_targets.npz',
    _PROJECT_ROOT / 'logs' / 'distill_iterations_20261026' / 'teachers' /
    'soft_targets.npz',
)

_DATASET_ALIASES = {
    'assist': 'assist2009',
    'assist09': 'assist2009',
    'assist2009': 'assist2009',
    'nips': 'nips_task34',
    'nips34': 'nips_task34',
    'nips_task3_4': 'nips_task34',
    'nips_task34': 'nips_task34',
}

# qseqs exists only when ``questions`` is part of input_type.  The other five
# fields are required by the alignment contract in all supported configurations.
_SHARED_REQUIRED = ('cseqs', 'rseqs', 'uid', 'masks', 'smasks')
_SHARED_OPTIONAL = ('qseqs', 'tseqs', 'utseqs')


def _canonical_dataset_name(dataset_name):
    key = str(dataset_name).strip().lower()
    if key not in _DATASET_ALIASES:
        raise ValueError(
            f'Unknown dataset_name={dataset_name!r}; expected one of '
            f'{sorted(set(_DATASET_ALIASES.values()))}'
        )
    return _DATASET_ALIASES[key]


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_same(reference, other, label):
    """Exact value+shape equality; dtypes are checked for tensor/array values."""
    reference = _as_numpy(reference)
    other = _as_numpy(other)
    if reference.shape != other.shape:
        raise AssertionError(
            f'{label} shape mismatch: {reference.shape} != {other.shape}'
        )
    if reference.dtype != other.dtype:
        raise AssertionError(
            f'{label} dtype mismatch: {reference.dtype} != {other.dtype}'
        )
    if not np.array_equal(reference, other):
        raise AssertionError(f'{label} values differ; refusing to guess alignment')


def _check_shared_base(base, component, component_name):
    """Require one component's KT base to be row-for-row identical to base."""
    other = component.dori
    for key in _SHARED_REQUIRED:
        if key not in base or key not in other:
            raise AssertionError(
                f'{component_name}: shared base field {key!r} is missing'
            )
        _assert_same(base[key], other[key], f'{component_name}.{key}')
    for key in _SHARED_OPTIONAL:
        in_base, in_other = key in base, key in other
        if in_base != in_other:
            raise AssertionError(
                f'{component_name}: shared base field {key!r} exists on only '
                'one side; refusing to infer alignment'
            )
        if in_base:
            _assert_same(base[key], other[key], f'{component_name}.{key}')
    if len(component) != len(base['rseqs']):
        raise AssertionError(
            f'{component_name}: length {len(component)} != base length '
            f'{len(base["rseqs"])}'
        )


def _resolve_teacher_path(teacher_path):
    if teacher_path is None:
        return None
    path = Path(teacher_path).expanduser()
    if path.is_dir():
        path = path / 'soft_targets.npz'
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f'Teacher soft-target cache not found: {path}')
    return path


def _default_teacher_path():
    for candidate in _DEFAULT_TEACHER_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def _teacher_has_selected_targets(teacher_path, folds):
    """True iff the OOF cache contains at least one labelled selected row."""
    with np.load(teacher_path, allow_pickle=False) as data:
        if 'mask' not in data or 'folds' not in data:
            # Let DistillDataset produce its own precise validation error.
            return True
        selected = np.isin(data['folds'], list(folds))
        return bool(np.asarray(data['mask'])[selected].any())


def _require_stats(stats, key, dataset_name):
    if stats is None:
        raise ValueError(
            f'{dataset_name} validation requires stats from a training '
            'JointDataset; refusing to fit validation statistics'
        )
    if not isinstance(stats, dict) or stats.get(key) is None:
        raise ValueError(
            f'{dataset_name} validation requires stats[{key!r}] from the '
            'training folds'
        )


def _check_process_stats(stats):
    required = {'time_center', 'time_scale', 'attempt_counts', 'hint_counts',
                'process_mode'}
    missing = required - set(stats)
    if missing:
        raise ValueError(f'process_stats missing keys: {sorted(missing)}')
    if stats['process_mode'] not in ('joint', 'conditional'):
        raise ValueError('process_stats has an invalid process_mode')


def _check_option_stats(stats):
    required = {'option_keys', 'option_prior'}
    missing = required - set(stats)
    if missing:
        raise ValueError(f'option_stats missing keys: {sorted(missing)}')


class JointDataset(Dataset):
    """Combine the existing per-signal datasets without changing their code.

    Parameters
    ----------
    file_path : str | Path
        ``train_valid_sequences.csv`` for the selected dataset.
    input_type : list[str] | str
        Same input types passed to ``KTDataset`` (normally questions+concepts).
    folds : iterable[int] | int
        Fold selection.  Row order is whatever the existing loaders produce.
    dataset_name : str
        ``assist2009`` or ``nips_task34`` (plus documented aliases).
    mode : {'train', 'valid'}
        ``valid`` requires ``stats`` from a training ``JointDataset``.
    stats : dict | None
        Training statistics.  Ignored/refitted for ``mode='train'``; mandatory
        for ``mode='valid'``.
    teacher_path : str | Path | None
        ASSIST OOF teacher ``.npz`` or its directory.  ``None`` uses the rich
        cache documented in the module docstring.
    transfer_mode : str
        Forwarded to ``TransferDataset`` (default ``same_all``, horizon 10).
    """

    def __init__(self, file_path, input_type, folds, dataset_name,
                 mode='train', stats=None, teacher_path=None,
                 transfer_mode='same_all'):
        self.dataset_name = _canonical_dataset_name(dataset_name)
        source = Path(file_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f'Joint dataset file not found: {source}')
        if source.name != 'train_valid_sequences.csv':
            raise ValueError(
                'Joint supervision currently supports train_valid_sequences.csv '
                'only; test files are intentionally rejected'
            )
        self.file_path = str(source)
        if isinstance(input_type, str):
            input_type = [input_type]
        self.input_type = list(input_type)
        if isinstance(folds, (int, np.integer)):
            folds = [int(folds)]
        self.folds = tuple(sorted({int(fold) for fold in folds}))
        if not self.folds:
            raise ValueError('folds must not be empty')
        self.mode = str(mode).strip().lower()
        if self.mode not in ('train', 'valid'):
            raise ValueError("mode must be 'train' or 'valid'")
        self.transfer_mode = str(transfer_mode)
        if self.transfer_mode not in TRANSFER_MODES:
            raise ValueError(
                f'transfer_mode must be one of {TRANSFER_MODES}, '
                f'got {transfer_mode!r}'
            )
        self.horizon = 10
        self.stats_ignored_on_train = self.mode == 'train' and stats is not None

        train_stats = None
        if self.mode == 'valid':
            train_stats = stats
            if train_stats is None:
                raise ValueError(
                    f'{self.dataset_name} validation requires stats from a '
                    'training JointDataset; refusing to fit validation statistics'
                )
            # Keep a copy so later caller mutation cannot change this loader.
            train_stats = copy.deepcopy(train_stats)

        if self.dataset_name == 'assist2009':
            self._build_assist(source, train_stats, teacher_path)
        else:
            if teacher_path is not None:
                raise ValueError('teacher_path is only used for ASSIST datasets')
            self._build_nips(source, train_stats)

        self.components = dict(self._components)
        self.base = self._components[self._base_component].base
        self._base_keys = frozenset(self.base[0].keys()) if len(self.base) else frozenset()
        self._aux_keys = self._discover_aux_keys()
        self._check_shared_bases()
        self._check_cross_component_alignment()

        if self.mode == 'train':
            self.stats = self._build_train_stats()
            self.validation_info = None
        else:
            # Exactly the training statistics: validation never adds fitted
            # values of its own.  Validation metadata lives beside them.
            self.stats = train_stats
            self.validation_info = {
                'dataset_name': self.dataset_name,
                'folds': list(self.folds),
                'rows': len(self.base),
            }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_assist(self, source, train_stats, teacher_path):
        process_stats = None
        if self.mode == 'valid':
            _require_stats(train_stats, 'process_stats', self.dataset_name)
            _check_process_stats(train_stats['process_stats'])
            process_stats = train_stats['process_stats']

        first_action = FirstActionDataset(
            source, self.input_type, self.folds, mode=self.mode
        )
        process = ProcessDataset(
            source, self.input_type, self.folds, mode=self.mode,
            process_stats=process_stats
        )
        transfer = TransferDataset(
            source, self.input_type, self.folds,
            pair_mode=self.transfer_mode, horizon=self.horizon, mode=self.mode
        )

        explicit = _resolve_teacher_path(teacher_path)
        teacher = explicit
        if teacher is None and self.mode == 'train':
            teacher = _default_teacher_path()
            if teacher is None:
                raise FileNotFoundError(
                    'ASSIST training requires the OOF rich teacher cache; pass '
                    'teacher_path=... explicitly. Looked for: '
                    + ', '.join(str(p) for p in _DEFAULT_TEACHER_CANDIDATES)
                )
        elif teacher is None:
            # Validation may run without distillation; only load the default
            # cache when the selected folds actually have OOF labels.
            teacher = _default_teacher_path()

        distill_soft = None
        if teacher is not None:
            if self.mode == 'train' or _teacher_has_selected_targets(teacher, self.folds):
                distill_soft = teacher
        self.teacher_path = str(teacher) if teacher is not None else None
        self.distill_teacher_path = str(distill_soft) if distill_soft is not None else None

        distill = DistillDataset(
            source, self.input_type, self.folds, soft_targets=distill_soft,
            teacher_source='rich', mode=self.mode
        )

        self._components = {
            'first_action': first_action,
            'process': process,
            'transfer': transfer,
            'distill': distill,
        }
        self._base_component = 'first_action'
        self.first_action = first_action
        self.process = process
        self.transfer = transfer
        self.distill = distill

    def _build_nips(self, source, train_stats):
        option_stats = None
        if self.mode == 'valid':
            _require_stats(train_stats, 'option_stats', self.dataset_name)
            _check_option_stats(train_stats['option_stats'])
            option_stats = train_stats['option_stats']

        confidence = ConfidenceDataset(
            source, self.input_type, self.folds, mode=self.mode
        )
        option = OptionDataset(
            source, self.input_type, self.folds, mode=self.mode,
            option_stats=option_stats
        )
        profile = ProfileDataset(
            source, self.input_type, self.folds,
            profile_mode='hierarchy', mode=self.mode
        )

        self._components = {
            'confidence': confidence,
            'option': option,
            'profile': profile,
        }
        self._base_component = 'confidence'
        self.confidence = confidence
        self.option = option
        self.profile = profile

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_shared_bases(self):
        base = self.base.dori
        for name, component in self._components.items():
            _check_shared_base(base, component, name)

    def _check_cross_component_alignment(self):
        if self.dataset_name == 'assist2009':
            first = self._components['first_action']
            process = self._components['process']
            transfer = self._components['transfer']
            if not torch.equal(first.event_ids, process.event_ids):
                raise AssertionError(
                    'FirstActionDataset and ProcessDataset event IDs differ'
                )
            expected_anchors = first.event_ids[:, :-1]
            if transfer.anchor_events.shape != first.event_ids.shape:
                raise AssertionError(
                    'TransferDataset anchors and FirstAction event IDs have '
                    'different shapes'
                )
            # Transfer anchor at output position j is the event E[j] (the
            # completed input event); FirstAction event id at j is E[j+1].
            if not torch.equal(transfer.anchor_events[:, 1:], expected_anchors):
                raise AssertionError(
                    'TransferDataset anchors do not match FirstAction event IDs'
                )
            self._alignment_evidence = {
                'event_ids': 'FirstActionDataset == ProcessDataset',
                'transfer_anchors': 'TransferDataset.anchor_events[:, 1:] == '
                                    'FirstActionDataset.event_ids[:, :-1]',
                'teacher_loaded': self.distill_teacher_path is not None,
            }
        else:
            confidence = self._components['confidence']
            option = self._components['option']
            if not torch.equal(confidence.event_ids, option.event_ids):
                raise AssertionError(
                    'ConfidenceDataset and OptionDataset event IDs differ'
                )
            self._alignment_evidence = {
                'event_ids': 'ConfidenceDataset == OptionDataset',
                'profile_mode': 'hierarchy',
            }

    def _discover_aux_keys(self):
        """Collect only non-base keys from each component, rejecting collisions."""
        base_keys = self._base_keys
        aux = {}
        owner = {}
        for name, component in self._components.items():
            if len(component):
                probe = component[0]
            else:  # pragma: no cover - supported datasets are never empty
                probe = {}
            keys = [key for key in probe if key not in base_keys]
            if not keys:
                raise AssertionError(
                    f'{name} exposes no auxiliary fields beyond the base batch'
                )
            for key in keys:
                if key in owner:
                    raise AssertionError(
                        f'auxiliary field {key!r} is produced by both '
                        f'{owner[key]} and {name}'
                    )
                owner[key] = name
            aux[name] = tuple(keys)
        return aux

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _build_train_stats(self):
        stats = {
            'dataset_name': self.dataset_name,
            'file_path': self.file_path,
            'input_type': list(self.input_type),
            'folds': list(self.folds),
            'mode': 'train',
            'transfer_mode': (self.transfer_mode
                              if self.dataset_name == 'assist2009' else None),
            'horizon': self.horizon if self.dataset_name == 'assist2009' else None,
            'rows': len(self.base),
            'components': list(self._components),
            'alignment': dict(self._alignment_evidence),
            'shared_base_fields': list(_SHARED_REQUIRED) + [
                key for key in _SHARED_OPTIONAL if key in self.base.dori
            ],
            'provided_stats_ignored_on_train': self.stats_ignored_on_train,
        }
        if self.dataset_name == 'assist2009':
            first = self._components['first_action']
            process = self._components['process']
            transfer = self._components['transfer']
            distill = self._components['distill']
            stats['first_action_stats'] = {
                'class_counts': list(first.action_class_counts),
                'supervised_events': int(first.action_mask.sum()),
                'uid_rows': len(first),
            }
            stats['process_stats'] = copy.deepcopy(process.process_stats)
            stats['transfer_stats'] = copy.deepcopy(transfer.stats)
            teacher_mask = distill.mask
            stats['distill_stats'] = {
                'teacher_source': 'rich',
                'teacher_path': self.distill_teacher_path,
                'supervised_events': int(teacher_mask.sum()),
                'coverage': (float(teacher_mask.sum() /
                                   distill.event_mask.sum().clamp_min(1))),
                'probability_mean': (float(
                    distill.probabilities[teacher_mask].mean()
                ) if bool(teacher_mask.any()) else None),
                'original_event_mask_events': int(distill.event_mask.sum()),
            }
        else:
            confidence = self._components['confidence']
            option = self._components['option']
            profile = self._components['profile']
            stats['confidence_stats'] = {
                'counts': list(confidence.confidence_counts),
                'coverage': float(confidence.coverage),
                'events': int(confidence.event_mask.sum()),
            }
            stats['option_stats'] = copy.deepcopy(option.option_stats)
            stats['option_counts'] = copy.deepcopy(option.counts)
            stats['profile_stats'] = copy.deepcopy(profile.stats)
        return stats

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    @property
    def dori(self):
        return self.base.dori

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        # Start from the untouched main KT task.  Base keys are never written.
        sample = self.base[index]
        for name, component in self._components.items():
            part = component[index]
            for key in self._aux_keys[name]:
                if key in sample:
                    raise AssertionError(
                        f'auxiliary field {key!r} from {name} would overwrite '
                        'an existing base field; refusing duplicate supervision'
                    )
                sample[key] = part[key]
        return sample


def joint_collate(batch):
    """Dispatch collation while preserving ProfileDataset's variable lengths."""
    uses_profile = any(
        any(str(key).startswith('profile_') for key in row) for row in batch
    )
    if uses_profile:
        return profile_collate(batch)
    return default_collate(batch)


# Both spellings are intentionally available to DataLoader callers/tests.
JointDataset.collate = staticmethod(joint_collate)
JointDataset.collate_fn = staticmethod(joint_collate)
