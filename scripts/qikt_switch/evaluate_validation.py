"""Evaluate a saved qikt_mamba checkpoint on its validation fold and export
per-question predictions.

This script deliberately reports ``validation_auc``: a direct token/question
level ROC AUC on the held-out validation fold, reconstructed from the original
``train_valid_sequences_quelevel.csv`` fragments and their original
``selectmasks``.  It is *not* ``windowauclate_mean`` and must not be reported as
that metric.

The original fragments repeat a ``uid`` once per chunk.  For every fragment we
recover the absolute question index with

    absolute_qidx = cumulative_valid_length_of_previous_fragments(uid)
                   + shifted_local_position

where ``shifted_local_position`` is the 1-based position after the causal shift
used by the original ``KTQueDataset``.  Questions, concepts and labels are
checked token-by-token against the complete sequence in
``train_valid_quelevel.csv``.  Short tail chunks that the original preprocessing
dropped are therefore never invented, and padding is never scored.

The native device is used (CUDA for public_mamba). The model is built/loaded through the original
``pykt.models.load_model`` wrapper and the validation fold is read through the
original ``KTQueDataset(mode="test")`` without augmentation.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

# The script is normally launched from the repository root, but Python puts the
# script directory (not the cwd) on sys.path.  Add the repository root before
# importing pykt so ``--help`` works from a clean shell too.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models import load_model

PREDICTION_COLUMNS = [
    "orirow",
    "qidx",
    "late_trues",
    "late_mean",
    "questions",
    "concepts",
]
PREDICTION_FILE = "validation_predictions.tsv"
ORIROW_FILE = "orirow_auc.csv"
RESULT_FILE = "validation_results.json"

# Cohort names intentionally match the existing audit/export convention.
COHORT_DEFINITIONS = {
    "all_two_class": "every orirow with at least one positive and one negative label",
    "primary_20_5": "orirow n>=20 and minority-class count>=5",
    "strict_50_10": "orirow n>=50 and minority-class count>=10",
}


def sha256_file(path):
    """Return the SHA256 hex digest of *path*."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_existing_path(value):
    """Resolve a possibly relative path recorded in a training snapshot."""
    if value is None or str(value) == "":
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), ROOT, ROOT / "examples"):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def _read_json(path):
    with open(path, "r", encoding="utf8") as handle:
        return json.load(handle)


def _positive_int(value, name):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _find_actual_epoch(checkpoint_dir, config):
    """Return an explicitly recorded completed/best epoch, otherwise None.

    No inference is made from filenames, logs, or test-result files.  The
    snapshot written by the original trainer does not record the actual epoch,
    so the normal result is ``None``.
    """
    for key in ("actual_epoch", "completed_epochs", "best_epoch"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, f"config.json:{key}"
    params = config.get("params") or {}
    for key in ("actual_epoch", "completed_epochs", "best_epoch"):
        value = params.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, f"config.json:params.{key}"

    # Only a training-side sidecar is considered.  all_results.json is not read
    # here because this validation-only step must not inspect test results.
    sidecar = Path(checkpoint_dir) / "training_stats.json"
    if sidecar.is_file():
        try:
            stats = _read_json(sidecar)
        except (OSError, ValueError):
            stats = {}
        for key in ("actual_epoch", "completed_epochs", "best_epoch", "epoch"):
            value = stats.get(key) if isinstance(stats, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value, f"training_stats.json:{key}"
    return None, None


def load_run(checkpoint_dir):
    """Load the checkpoint snapshot and resolve all input/checkpoint paths."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    config = _read_json(config_path)

    params = config.get("params") or {}
    data_config = copy.deepcopy(config.get("data_config") or {})
    if not data_config:
        raise ValueError("config.json has no data_config block")
    if "dpath" not in data_config:
        raise ValueError("config.json data_config has no dpath")

    data_dir = _resolve_existing_path(data_config["dpath"])
    if data_dir is None or not data_dir.is_dir():
        raise FileNotFoundError(f"resolved data directory does not exist: {data_dir}")
    data_config["dpath"] = str(data_dir)

    fragment_name = data_config.get(
        "train_valid_file_quelevel", "train_valid_sequences_quelevel.csv"
    )
    raw_name = data_config.get(
        "train_valid_original_file_quelevel", "train_valid_quelevel.csv"
    )
    fragment_path = (data_dir / fragment_name).resolve()
    raw_path = (data_dir / raw_name).resolve()
    for path in (fragment_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing validation input: {path}")

    fold = params.get("fold", data_config.get("fold"))
    if fold is None:
        raise ValueError("config.json does not record the validation fold")
    fold = int(fold)

    model_name = params.get("model_name") or "qikt_mamba"
    if model_name != "qikt_mamba":
        raise ValueError(f"this exporter is scoped to qikt_mamba, got {model_name!r}")

    emb_type = params.get("emb_type") or "attn"
    checkpoint_path = checkpoint_dir / f"{emb_type}_model.ckpt"
    if not checkpoint_path.is_file():
        fallback = checkpoint_dir / "attn_model.ckpt"
        if fallback.is_file():
            emb_type = "attn"
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(
                f"missing model checkpoint: {checkpoint_path} (also no attn_model.ckpt)"
            )

    train_config = config.get("train_config") or {}
    configured_epochs = train_config.get("num_epochs", params.get("num_epochs"))
    if configured_epochs is not None:
        configured_epochs = _positive_int(configured_epochs, "configured_epochs")
    actual_epoch, actual_epoch_source = _find_actual_epoch(checkpoint_dir, config)

    seed = params.get("seed")
    seed = int(seed) if seed is not None else None
    dataset = params.get("dataset_name") or data_dir.name
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("config.json has no model_config block")

    paths = {
        "checkpoint_dir": checkpoint_dir,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "data_dir": data_dir,
        "fragment_path": fragment_path,
        "raw_path": raw_path,
        "fragment_name": fragment_name,
        "raw_name": raw_name,
        "dataset": dataset,
        "model_name": model_name,
        "emb_type": emb_type,
        "version": model_config.get("version"),
        "fold": fold,
        "seed": seed,
        "configured_epochs": configured_epochs,
        "actual_epoch": actual_epoch,
        "actual_epoch_source": actual_epoch_source,
    }
    return config, paths


def load_saved_model(config, paths, device="auto"):
    """Build and load the whole qikt_mamba wrapper on the native device.

    ``pykt.models.load_model`` is used unchanged; ``torch.load`` is temporarily
    given a CPU ``map_location`` so GPU-trained checkpoints can be restored on a
    CPU host.
    """
    init_model_module = importlib.import_module("pykt.models.init_model")
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if paths["version"] == "public_mamba" and selected_device == "cpu":
        raise ValueError("public_mamba requires native CUDA kernels; CPU substitution is disabled")

    data_config = copy.deepcopy(config["data_config"])
    data_config["dpath"] = str(paths["data_dir"])
    data_config.setdefault("emb_path", "")
    model_config = copy.deepcopy(config["model_config"])
    # These are supplied explicitly by init_model for qikt_mamba.  Removing any
    # duplicates keeps snapshots from older trainers loadable.
    for key in ("num_q", "num_c", "max_concepts", "emb_type", "emb_path", "device"):
        model_config.pop(key, None)

    # Force the module-level device used by the original init_model, then put it
    # back so importing this module does not change unrelated callers.
    previous_device = init_model_module.device
    init_model_module.device = selected_device

    real_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return real_load(*args, **kwargs)

    torch.load = cpu_load
    try:
        model = load_model(
            paths["model_name"],
            model_config,
            data_config,
            paths["emb_type"],
            str(paths["checkpoint_dir"]),
        )
    finally:
        torch.load = real_load
        init_model_module.device = previous_device

    model.to(selected_device)
    # QueBaseModel.train is its fitting API, so Module.eval on the wrapper
    # would call train(False) with the wrong signature. Match native evaluate.
    model.model.eval()
    return model


def build_validation_dataset(config, paths):
    """Build the original no-augmentation KTQueDataset for the validation fold."""
    data_config = config["data_config"]
    if "input_type" not in data_config:
        raise ValueError("config.json data_config has no input_type")
    if "num_c" not in data_config or "max_concepts" not in data_config:
        raise ValueError("config.json data_config needs num_c and max_concepts")
    return KTQueDataset(
        str(paths["fragment_path"]),
        input_type=data_config["input_type"],
        folds={int(paths["fold"])},
        concept_num=int(data_config["num_c"]),
        max_concepts=int(data_config["max_concepts"]),
        mode="test",
        aug_probs=None,
    )


def _parse_int_list(value):
    return [int(item) for item in str(value).split(",")]


def _parse_str_list(value):
    return [str(item) for item in str(value).split(",")]


def _valid_length(responses):
    """Number of leading non-padding events (padding value is -1)."""
    size = 0
    while size < len(responses) and int(responses[size]) != -1:
        size += 1
    if any(int(value) != -1 for value in responses[size:]):
        raise ValueError("responses contain non-padding values after padding started")
    return size


def _concept_row(token, max_concepts):
    token = str(token)
    if token == "-1":
        return [-1] * max_concepts
    values = [int(part) for part in token.split("_")]
    if len(values) > max_concepts:
        raise ValueError(
            f"concept {token!r} has more than max_concepts={max_concepts} ids"
        )
    return values + [-1] * (max_concepts - len(values))


def _build_fragment_metadata(config, paths):
    """Read complete sequences and the validation fragments.

    Returns a list aligned with ``KTQueDataset`` rows.  Every fragment is
    checked against the complete original sequence before any prediction is
    made.  Absolute qidx values are recovered from cumulative valid lengths.
    """
    raw = pd.read_csv(paths["raw_path"])
    fragments = pd.read_csv(paths["fragment_path"])
    if "uid" not in raw or "uid" not in fragments:
        raise ValueError("validation inputs must contain uid")
    if not raw["uid"].is_unique:
        raise ValueError("train_valid_quelevel.csv contains duplicate uid values")

    full_by_uid = {}
    for _, row in raw.iterrows():
        uid = int(row["uid"])
        full_by_uid[uid] = {
            "questions": _parse_int_list(row["questions"]),
            "concepts": _parse_str_list(row["concepts"]),
            "responses": _parse_int_list(row["responses"]),
        }

    fold = int(paths["fold"])
    if "fold" not in fragments:
        raise ValueError("train_valid_sequences_quelevel.csv must contain fold")
    selected = fragments[fragments["fold"].astype(int) == fold].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"no validation fragments for fold {fold}")

    offsets = {}
    metadata = []
    seen_uids = set()
    fragment_counts = {}
    for index, row in selected.iterrows():
        uid = int(row["uid"])
        questions = _parse_int_list(row["questions"])
        concepts = _parse_str_list(row["concepts"])
        responses = _parse_int_list(row["responses"])
        selectmasks = _parse_int_list(row["selectmasks"])
        length = len(questions)
        if not (len(concepts) == len(responses) == len(selectmasks) == length):
            raise ValueError(f"fragment {index} uid={uid} has inconsistent field lengths")
        if uid not in full_by_uid:
            raise ValueError(f"fragment {index} uid={uid} is absent from the complete file")

        size = _valid_length(responses)
        for field, values in (
            ("questions", questions),
            ("concepts", concepts),
            ("selectmasks", selectmasks),
        ):
            if any(str(value) != "-1" for value in values[size:]):
                raise ValueError(
                    f"fragment {index} uid={uid} has non-padding {field} after valid length"
                )

        offset = offsets.get(uid, 0)
        full = full_by_uid[uid]
        if offset + size > len(full["questions"]):
            raise ValueError(f"fragment {index} uid={uid} exceeds the complete sequence")
        expected = {
            "questions": full["questions"][offset : offset + size],
            "concepts": full["concepts"][offset : offset + size],
            "responses": full["responses"][offset : offset + size],
        }
        actual = {
            "questions": questions[:size],
            "concepts": concepts[:size],
            "responses": responses[:size],
        }
        for field in ("questions", "concepts", "responses"):
            if actual[field] != expected[field]:
                for position, (got, want) in enumerate(
                    zip(actual[field], expected[field])
                ):
                    if got != want:
                        raise ValueError(
                            f"fragment {index} uid={uid} {field} mismatch at complete "
                            f"index {offset + position}: {got!r} != {want!r}"
                        )
                raise ValueError(
                    f"fragment {index} uid={uid} {field} length mismatch: "
                    f"{len(actual[field])} != {len(expected[field])}"
                )

        metadata.append(
            {
                "row": index,
                "uid": uid,
                "offset": offset,
                "size": size,
                "questions": questions,
                "concepts": concepts,
                "responses": responses,
                "selectmasks": selectmasks,
                "full": full,
            }
        )
        offsets[uid] = offset + size
        if uid not in seen_uids:
            seen_uids.add(uid)
            fragment_counts[uid] = 1
        else:
            fragment_counts[uid] += 1

    dropped_events = 0
    dropped_sequences = 0
    for uid in seen_uids:
        dropped = len(full_by_uid[uid]["questions"]) - offsets[uid]
        if dropped:
            dropped_events += dropped
            dropped_sequences += 1

    stats = {
        "raw_sequences": len(seen_uids),
        "raw_events": int(sum(len(full_by_uid[uid]["questions"]) for uid in seen_uids)),
        "fragments": len(metadata),
        "fragment_events": int(sum(item["size"] for item in metadata)),
        "uids_with_multiple_fragments": int(
            sum(count > 1 for count in fragment_counts.values())
        ),
        "dropped_tail_sequences": dropped_sequences,
        "dropped_tail_events": dropped_events,
    }
    return metadata, stats


def _dataset_row_matches(data, batch_index, meta, max_concepts):
    """Assert that the dataset row still represents the validated fragment.

    KTQueDataset returns causal-shifted tensors of length ``L-1``.  Compare the
    history (``qseqs``/``cseqs``/``rseqs``) and the shifted target positions
    against the fragment taken from the complete sequence.
    """
    size = meta["size"]
    if size > 1:
        expected_c = [
            _concept_row(token, max_concepts) for token in meta["concepts"][: size - 1]
        ]
        expected_c_shift = [
            _concept_row(token, max_concepts) for token in meta["concepts"][1:size]
        ]
        if data["qseqs"][batch_index, : size - 1].tolist() != meta["questions"][: size - 1]:
            raise ValueError(f"dataset question row {meta['row']} does not match fragment")
        if (
            data["shft_qseqs"][batch_index, : size - 1].tolist()
            != meta["questions"][1:size]
        ):
            raise ValueError(
                f"dataset shifted question row {meta['row']} does not match fragment"
            )
        if data["rseqs"][batch_index, : size - 1].tolist() != meta["responses"][: size - 1]:
            raise ValueError(f"dataset response row {meta['row']} does not match fragment")
        if (
            data["shft_rseqs"][batch_index, : size - 1].tolist()
            != meta["responses"][1:size]
        ):
            raise ValueError(
                f"dataset shifted response row {meta['row']} does not match fragment"
            )
        if data["cseqs"][batch_index, : size - 1].tolist() != expected_c:
            raise ValueError(f"dataset concept row {meta['row']} does not match fragment")
        if data["shft_cseqs"][batch_index, : size - 1].tolist() != expected_c_shift:
            raise ValueError(
                f"dataset shifted concept row {meta['row']} does not match fragment"
            )


def _collect_records(model, dataset, metadata, max_concepts, batch_size):
    """Run the original batch_to_device/predict_one_step path and map rows."""
    if len(dataset) != len(metadata):
        raise ValueError(
            f"dataset rows ({len(dataset)}) do not match validation fragments "
            f"({len(metadata)}); a stale preprocessing cache may be present"
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = []
    seen = {}
    duplicate_positions = 0
    duplicate_prediction_mismatches = 0
    cursor = 0
    model.model.eval()
    with torch.no_grad():
        for data in loader:
            new_data = model.batch_to_device(data, process=True)
            predictions = model.predict_one_step(new_data, process=False)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            scores = predictions.detach().cpu().numpy()
            smasks = new_data["sm"].detach().cpu().numpy()
            if scores.shape != smasks.shape:
                raise ValueError(
                    f"prediction shape {scores.shape} != smask shape {smasks.shape}"
                )
            batch_rows = data["qseqs"].shape[0]
            for batch_index in range(batch_rows):
                meta = metadata[cursor + batch_index]
                _dataset_row_matches(data, batch_index, meta, max_concepts)
                expected_smask = np.asarray(
                    [int(value) != -1 for value in meta["selectmasks"][1:]],
                    dtype=bool,
                )
                if not np.array_equal(smasks[batch_index], expected_smask):
                    raise ValueError(
                        f"loader smask does not match original selectmasks for "
                        f"fragment {meta['row']} uid={meta['uid']}"
                    )
                for local_index in range(smasks.shape[1]):
                    if not smasks[batch_index, local_index]:
                        continue
                    shifted_position = local_index + 1
                    if shifted_position >= meta["size"]:
                        raise ValueError(
                            f"selected padding position in fragment {meta['row']}"
                        )
                    label = int(meta["responses"][shifted_position])
                    if label not in (0, 1):
                        raise ValueError(
                            f"non-binary label {label} in fragment {meta['row']}"
                        )
                    absolute_qidx = meta["offset"] + shifted_position
                    full = meta["full"]
                    question = int(full["questions"][absolute_qidx])
                    concept = str(full["concepts"][absolute_qidx])
                    if question != int(meta["questions"][shifted_position]):
                        raise ValueError("fragment question disagrees with complete sequence")
                    if concept != str(meta["concepts"][shifted_position]):
                        raise ValueError("fragment concept disagrees with complete sequence")
                    if label != int(full["responses"][absolute_qidx]):
                        raise ValueError("fragment label disagrees with complete sequence")
                    score = float(scores[batch_index, local_index])
                    key = (meta["uid"], absolute_qidx)
                    record = {
                        "orirow": meta["uid"],
                        "qidx": absolute_qidx,
                        "late_trues": label,
                        "late_mean": score,
                        "questions": question,
                        "concepts": concept,
                    }
                    if key in seen:
                        duplicate_positions += 1
                        previous = seen[key]
                        if (
                            previous["late_trues"] != label
                            or previous["questions"] != question
                            or previous["concepts"] != concept
                        ):
                            raise ValueError(
                                f"duplicate question {key} disagrees on question/concept/label"
                            )
                        if not np.isclose(
                            previous["late_mean"], score, rtol=1e-6, atol=1e-7
                        ):
                            duplicate_prediction_mismatches += 1
                        continue
                    seen[key] = record
                    records.append(record)
            cursor += batch_rows
    if cursor != len(metadata):
        raise ValueError(f"processed {cursor} fragments, expected {len(metadata)}")

    frame = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["orirow", "qidx"], kind="mergesort").reset_index(
            drop=True
        )
    stats = {
        "selected_positions": int(duplicate_positions + len(frame)),
        "scored_questions": int(len(frame)),
        "duplicate_positions": int(duplicate_positions),
        "duplicate_prediction_mismatches": int(duplicate_prediction_mismatches),
    }
    return frame, stats


def _summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "N": 0,
            "mean": None,
            "std": None,
            "range": None,
            "q05": None,
            "q95": None,
        }
    return {
        "N": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "range": float(np.ptp(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _per_orirow_auc(frame):
    rows = []
    for uid, group in frame.groupby("orirow", sort=True):
        labels = group["late_trues"].to_numpy(dtype=int)
        n = int(len(labels))
        positives = int(labels.sum())
        minority = int(min(positives, n - positives))
        auc = None
        if 0 < positives < n:
            auc = float(roc_auc_score(labels, group["late_mean"].to_numpy(dtype=float)))
        rows.append(
            {
                "orirow": int(uid),
                "n": n,
                "positives": positives,
                "minority": minority,
                "auc": auc,
                "defined": auc is not None,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["orirow", "n", "positives", "minority", "auc", "defined"],
    )


def _cohorts(per_orirow):
    if per_orirow.empty:
        empty = _summary([])
        return {
            "all_two_class": dict(empty),
            "primary_20_5": dict(empty),
            "strict_50_10": dict(empty),
        }
    masks = {
        "all_two_class": per_orirow["defined"],
        "primary_20_5": (per_orirow["n"] >= 20) & (per_orirow["minority"] >= 5),
        "strict_50_10": (per_orirow["n"] >= 50) & (per_orirow["minority"] >= 10),
    }
    result = {}
    for name, mask in masks.items():
        values = per_orirow.loc[mask, "auc"].dropna().to_numpy(dtype=float)
        result[name] = _summary(values)
    return result


def evaluate_checkpoint(checkpoint_dir, output_dir, batch_size=128, device="auto"):
    """Evaluate a checkpoint and write the TSV/CSV/JSON validation export."""
    batch_size = _positive_int(batch_size, "batch_size")
    if batch_size == 0:
        raise ValueError("batch_size must be positive")
    config, paths = load_run(checkpoint_dir)
    metadata, fragment_stats = _build_fragment_metadata(config, paths)

    model = load_saved_model(config, paths, device=device)
    dataset = build_validation_dataset(config, paths)
    max_concepts = int(config["data_config"]["max_concepts"])
    frame, prediction_stats = _collect_records(
        model, dataset, metadata, max_concepts, batch_size
    )

    per_orirow = _per_orirow_auc(frame)
    cohorts = _cohorts(per_orirow)

    labels = frame["late_trues"].to_numpy(dtype=int) if not frame.empty else np.array([])
    scores = (
        frame["late_mean"].to_numpy(dtype=float) if not frame.empty else np.array([])
    )
    validation_auc = None
    if labels.size and np.unique(labels).size == 2:
        validation_auc = float(roc_auc_score(labels, scores))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / PREDICTION_FILE
    orirow_path = output_dir / ORIROW_FILE
    result_path = output_dir / RESULT_FILE

    frame.to_csv(
        prediction_path,
        sep="\t",
        index=False,
        columns=PREDICTION_COLUMNS,
        float_format="%.10g",
    )
    per_orirow.to_csv(orirow_path, index=False, float_format="%.10g")

    counts = {
        **fragment_stats,
        **prediction_stats,
        "dataset_rows": int(len(dataset)),
        "padding_scored": False,
        "test_files_read": [],
    }
    result = {
        "device": str(next(model.parameters()).device),
        "dataset": paths["dataset"],
        "model": paths["model_name"],
        "emb_type": paths["emb_type"],
        "version": paths["version"],
        "fold": int(paths["fold"]),
        "seed": paths["seed"],
        "configured_epochs": paths["configured_epochs"],
        "actual_epoch": paths["actual_epoch"],
        "actual_epoch_source": paths["actual_epoch_source"],
        "primary_metric": "validation_auc",
        "validation_auc": validation_auc,
        "validation_auc_defined": validation_auc is not None,
        "cohorts": cohorts,
        "cohort_definitions": COHORT_DEFINITIONS,
        "counts": counts,
        "alignment": {
            "question_mismatch": 0,
            "concept_mismatch": 0,
            "label_mismatch": 0,
            "selectmask_definition": "original KTQueDataset smasks = selectmasks[1:] != -1",
            "qidx_recovery": (
                "absolute_qidx = cumulative_valid_length_of_previous_fragments(uid) "
                "+ shifted_local_position; shifted_local_position is 1-based after "
                "the original causal shift"
            ),
            "padding_ignored": True,
            "unique_per_uid_qidx": not bool(
                prediction_stats["duplicate_positions"]
            ),
        },
        "metric_provenance": (
            "direct validation-fold ROC AUC over reconstructed per-question "
            "predictions from the original fragments and selectmasks; this is "
            "not windowauclate_mean"
        ),
        "score_source": (
            "QIKT_MAMBA.predict_one_step(...)->y, the fused score used by the "
            "original native evaluate()"
        ),
        "windowauclate_mean": None,
        "windowauclate_mean_note": (
            "not computed; validation_auc must not be reported as "
            "windowauclate_mean"
        ),
        "label_usage": (
            "labels are export targets only; model inputs are the original "
            "KTQueDataset question/concept tensors and shifted prior responses"
        ),
        "causal_input_order": (
            "prediction at absolute qidx uses history before that event; current "
            "and future labels are not model inputs"
        ),
        "inputs": {
            "train_valid_sequences_quelevel": str(paths["fragment_path"]),
            "train_valid_quelevel": str(paths["raw_path"]),
        },
        "input_sha256": {
            paths["fragment_name"]: sha256_file(paths["fragment_path"]),
            paths["raw_name"]: sha256_file(paths["raw_path"]),
        },
        "checkpoint": str(paths["checkpoint_path"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint_path"]),
        "config_sha256": sha256_file(paths["config_path"]),
        "outputs": {
            "predictions_tsv": PREDICTION_FILE,
            "orirow_auc_csv": ORIROW_FILE,
            "validation_results_json": RESULT_FILE,
        },
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    with open(result_path, "w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved qikt_mamba checkpoint on its validation fold and "
            "export per-question validation predictions (validation_auc, not "
            "windowauclate_mean)."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="directory containing config.json and <emb_type>_model.ckpt (attn_model.ckpt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for validation_predictions.tsv, orirow_auc.csv and validation_results.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="inference batch size (default: 128)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = evaluate_checkpoint(
        args.checkpoint_dir, args.output_dir, batch_size=args.batch_size, device=args.device
    )
    print(
        json.dumps(
            {
                "validation_auc": result["validation_auc"],
                "scored_questions": result["counts"]["scored_questions"],
                "cohorts": {
                    name: values["N"] for name, values in result["cohorts"].items()
                },
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
