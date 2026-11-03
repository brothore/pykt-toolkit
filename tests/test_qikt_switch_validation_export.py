"""CPU fixture checks for the qikt_mamba validation export.

The fixture uses the existing GRU path (``version="lstm"``) of the real
``QIKT_MAMBA`` class, not a stand-in Mamba implementation.  It exercises the
original loader/selectmask behaviour with multiple fragments for the same uid
and a dropped short tail fragment.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from pykt.models.qikt_mamba import QIKT_MAMBA

P = Path(__file__).resolve().parents[1] / "scripts/qikt_switch"
sys.path.insert(0, str(P))
import evaluate_validation as ev


def _tokens(values):
    return ",".join(str(value) for value in values)


def _full_sequences():
    specs = {
        10: [1, 0, 1, 0, 1, 0, 1],
        11: [1] * 6,
        12: [0, 1, 0, 1],
        13: [0, 0, 1],
        14: [i % 2 for i in range(29)],
    }
    rows = []
    for uid, responses in specs.items():
        length = len(responses)
        questions = [(i * 2) % 9 for i in range(length)]
        concepts = ["0_1" if i % 7 == 0 else str(i % 3) for i in range(length)]
        timestamps = [1000 + i for i in range(length)]
        rows.append(
            {
                "fold": 0,
                "uid": uid,
                "questions": _tokens(questions),
                "concepts": _tokens(concepts),
                "responses": _tokens(responses),
                "timestamps": _tokens(timestamps),
            }
        )
    return pd.DataFrame(rows)


def _fragments(full, maxlen=4, min_seq_len=3):
    rows = []
    fields = ["questions", "concepts", "responses", "timestamps"]
    for _, source in full.iterrows():
        sequences = {key: str(source[key]).split(",") for key in fields}
        length = len(sequences["responses"])
        start = 0
        rest = length
        while rest >= maxlen:
            item = {key: sequences[key][start : start + maxlen] for key in fields}
            item.update(fold=0, uid=int(source.uid), selectmasks=["1"] * maxlen)
            rows.append(item)
            start += maxlen
            rest -= maxlen
        # This is the original preprocessing rule: short tails are dropped.
        if rest < min_seq_len:
            continue
        pad = maxlen - rest
        item = {key: sequences[key][start:] + ["-1"] * pad for key in fields}
        item.update(fold=0, uid=int(source.uid), selectmasks=["1"] * rest + ["-1"] * pad)
        rows.append(item)

    output = []
    for item in rows:
        row = {"fold": item["fold"], "uid": item["uid"]}
        for key in fields + ["selectmasks"]:
            row[key] = _tokens(item[key])
        output.append(row)
    return pd.DataFrame(output)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    full = _full_sequences()
    fragments = _fragments(full)
    full.to_csv(data_dir / "train_valid_quelevel.csv", index=False)
    fragments.to_csv(data_dir / "train_valid_sequences_quelevel.csv", index=False)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    model_config = {
        "emb_size": 8,
        "dropout": 0.0,
        "emb_type": "attn",
        "version": "lstm",
        "mlp_layer_num": 1,
        "num_attn_head": 2,
        "other_config": {
            "output_mode": "an",
            "output_q_all_lambda": 1,
            "output_c_all_lambda": 1,
            "output_c_next_lambda": 1,
        },
    }
    torch.manual_seed(7)
    model = QIKT_MAMBA(num_q=10, num_c=5, **model_config)
    torch.save(model.state_dict(), checkpoint_dir / "attn_model.ckpt")

    data_config = {
        "dpath": str(data_dir),
        "num_q": 10,
        "num_c": 5,
        "input_type": ["questions", "concepts"],
        "max_concepts": 2,
        "maxlen": 4,
        "emb_path": "",
        "folds": [0, 1, 2, 3, 4],
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
    }
    config = {
        "train_config": {"num_epochs": 7, "batch_size": 2, "optimizer": "adam"},
        "model_config": model_config,
        "data_config": data_config,
        "params": {
            "model_name": "qikt_mamba",
            "dataset_name": "fixture",
            "fold": 0,
            "emb_type": "attn",
            "seed": 42,
            "num_epochs": 7,
            "learning_rate": 1e-3,
            "save_dir": str(checkpoint_dir),
        },
    }
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf8"
    )
    return data_dir, checkpoint_dir, full, fragments


EXPECTED_QIDX = {
    10: [1, 2, 3, 5, 6],
    11: [1, 2, 3],
    12: [1, 2, 3],
    13: [1, 2],
    14: [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27],
}


def _native_order(fragments):
    """Expected (uid, qidx, label) in KTQueDataset/DataLoader order."""
    offsets = {}
    records = []
    selected = fragments[fragments["fold"].astype(int) == 0]
    for _, row in selected.iterrows():
        uid = int(row.uid)
        questions = [int(value) for value in str(row.questions).split(",")]
        responses = [int(value) for value in str(row.responses).split(",")]
        selectmasks = [int(value) for value in str(row.selectmasks).split(",")]
        length = len(questions)
        size = 0
        while size < length and responses[size] != -1:
            size += 1
        offset = offsets.get(uid, 0)
        for local_index in range(1, size):
            if selectmasks[local_index] != -1:
                records.append((uid, offset + local_index, responses[local_index]))
        offsets[uid] = offset + size
    return records


def test_export_recovers_qidx_across_fragments_and_ignores_padding(tmp_path):
    _, checkpoint_dir, full, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    result = ev.evaluate_checkpoint(checkpoint_dir, output_dir, batch_size=2)

    predictions = pd.read_csv(output_dir / ev.PREDICTION_FILE, sep="\t")
    assert list(predictions.columns) == ev.PREDICTION_COLUMNS
    for uid, qidxs in EXPECTED_QIDX.items():
        got = predictions.loc[predictions.orirow == uid, "qidx"].tolist()
        assert got == qidxs, (uid, got, qidxs)

    assert len(predictions) == 34
    assert predictions.duplicated(["orirow", "qidx"]).sum() == 0
    assert not (predictions.qidx == 0).any()
    assert result["counts"]["scored_questions"] == 34
    assert result["counts"]["fragments"] == 12
    assert result["counts"]["dropped_tail_events"] == 3
    assert result["counts"]["test_files_read"] == []
    assert result["alignment"]["padding_ignored"] is True

    raw = full.set_index("uid")
    for _, row in predictions.iterrows():
        source = raw.loc[int(row.orirow)]
        questions = [int(value) for value in str(source.questions).split(",")]
        concepts = str(source.concepts).split(",")
        responses = [int(value) for value in str(source.responses).split(",")]
        qidx = int(row.qidx)
        assert int(row.late_trues) == responses[qidx]
        assert int(row.questions) == questions[qidx]
        assert str(row.concepts) == concepts[qidx]


def test_predictions_and_auc_match_direct_native_evaluate(tmp_path):
    _, checkpoint_dir, _, fragments = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    result = ev.evaluate_checkpoint(checkpoint_dir, output_dir, batch_size=3)

    config, paths = ev.load_run(checkpoint_dir)
    model = ev.load_saved_model(config, paths)
    dataset = ev.build_validation_dataset(config, paths)

    native = model.evaluate(dataset, batch_size=3)
    native_auc = native["auc"]
    assert result["validation_auc"] == pytest.approx(native_auc, abs=1e-9)
    assert result["primary_metric"] == "validation_auc"
    assert result["windowauclate_mean"] is None

    native_predictions = model.predict(dataset, batch_size=3)
    native_scores = native_predictions["y"]
    native_trues = native_predictions["ts"]
    expected = _native_order(fragments)
    assert len(expected) == len(native_scores) == 34
    np.testing.assert_array_equal(
        native_trues, np.asarray([label for _, _, label in expected], dtype=float)
    )

    predictions = pd.read_csv(output_dir / ev.PREDICTION_FILE, sep="\t")
    predictions = predictions.set_index(["orirow", "qidx"]).sort_index()
    expected_scores = np.asarray(
        [predictions.loc[(uid, qidx), "late_mean"] for uid, qidx, _ in expected],
        dtype=float,
    )
    np.testing.assert_allclose(native_scores, expected_scores, rtol=1e-6, atol=1e-7)
    assert result["validation_auc"] == pytest.approx(
        roc_auc_score(native_trues, native_scores), abs=1e-9
    )


def test_single_class_orirow_auc_is_undefined_and_cohorts_are_explicit(tmp_path):
    _, checkpoint_dir, _, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    result = ev.evaluate_checkpoint(checkpoint_dir, output_dir, batch_size=2)

    per_orirow = pd.read_csv(output_dir / ev.ORIROW_FILE)
    assert len(per_orirow) == 5
    single = per_orirow[per_orirow.orirow == 11].iloc[0]
    assert not bool(single.defined)
    assert np.isnan(single.auc)
    assert int(single.n) == 3
    assert int(single.minority) == 0

    cohorts = result["cohorts"]
    assert cohorts["all_two_class"]["N"] == 4
    assert cohorts["primary_20_5"]["N"] == 1
    assert cohorts["strict_50_10"]["N"] == 0
    assert cohorts["strict_50_10"]["mean"] is None
    primary = per_orirow[per_orirow.orirow == 14].iloc[0]
    assert int(primary.n) == 21
    assert int(primary.minority) == 7
    assert int(cohorts["primary_20_5"]["N"]) == 1


def test_result_records_provenance_epochs_and_hashes(tmp_path):
    data_dir, checkpoint_dir, _, _ = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    result = ev.evaluate_checkpoint(checkpoint_dir, output_dir, batch_size=2)

    assert result["dataset"] == "fixture"
    assert result["model"] == "qikt_mamba"
    assert result["fold"] == 0
    assert result["seed"] == 42
    assert result["configured_epochs"] == 7
    assert result["actual_epoch"] is None
    assert result["actual_epoch_source"] is None
    assert result["checkpoint_sha256"] == _sha256(checkpoint_dir / "attn_model.ckpt")
    assert result["config_sha256"] == _sha256(checkpoint_dir / "config.json")
    assert result["input_sha256"]["train_valid_sequences_quelevel.csv"] == _sha256(
        data_dir / "train_valid_sequences_quelevel.csv"
    )
    assert result["input_sha256"]["train_valid_quelevel.csv"] == _sha256(
        data_dir / "train_valid_quelevel.csv"
    )
    assert len(result["checkpoint_sha256"]) == 64
    assert result["counts"]["dataset_rows"] == 12


def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as excinfo:
        ev.main(["--help"])
    assert excinfo.value.code == 0
    assert "--checkpoint-dir" in capsys.readouterr().out
