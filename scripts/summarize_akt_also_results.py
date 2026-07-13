#!/usr/bin/env python3
"""Summarize AKT/ALSO runs and flag checkpoints missing student-level metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run",
    "checkpoint_dir",
    "status",
    "use_also",
    "also_grouping_mode",
    "batch_size",
    "also_pi_lr",
    "also_pi_decay",
    "also_loss_scale",
    "also_alpha",
    "also_mode",
    "overall_dataset_auc",
    "student_stats_mean",
    "student_stats_std",
    "student_stats_range",
    "validauc",
    "best_epoch",
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("saved_model"))
    parser.add_argument("--output", type=Path, default=Path("logs/akt_also_results_summary.csv"))
    args = parser.parse_args()

    rows: list[dict] = []
    for config_path in sorted(args.root.rglob("config.json")):
        config = load_json(config_path)
        params = config.get("params", {})
        if params.get("model_name") != "akt" or params.get("dataset_name") != "assist2009":
            continue

        run = config_path.parent.parents[1].name
        if not (run.startswith("akt_also_") or run.startswith("akt_student_also_v2")):
            continue

        metrics = load_json(config_path.parent / "overall_stats_output.json")
        results = load_json(config_path.parent / "all_results.json")
        if metrics and results:
            status = "evaluated"
        elif (config_path.parent / "qid_model.ckpt").is_file():
            status = "evaluation_missing"
        else:
            status = "training_or_incomplete"

        row = {
            "run": run,
            "checkpoint_dir": str(config_path.parent),
            "status": status,
            "use_also": params.get("use_also"),
            "also_grouping_mode": params.get("also_grouping_mode"),
            "batch_size": params.get("batch_size"),
            "also_pi_lr": params.get("also_pi_lr"),
            "also_pi_decay": params.get("also_pi_decay"),
            "also_loss_scale": params.get("also_loss_scale"),
            "also_alpha": params.get("also_alpha"),
            "also_mode": params.get("also_mode"),
            "overall_dataset_auc": metrics.get("overall_dataset_auc"),
            "student_stats_mean": metrics.get("student_stats_mean"),
            "student_stats_std": metrics.get("student_stats_std"),
            "student_stats_range": metrics.get("student_stats_range"),
            "validauc": results.get("validauc"),
            "best_epoch": results.get("best_epoch"),
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["status"], row["run"], row["checkpoint_dir"]))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {len(rows)} AKT rows to {args.output}: {counts}")


if __name__ == "__main__":
    main()
