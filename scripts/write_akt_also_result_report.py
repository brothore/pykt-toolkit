#!/usr/bin/env python3
"""Render the current AKT/ALSO result CSV as a Markdown experiment report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


def number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: str) -> str:
    parsed = number(value)
    return "-" if parsed != parsed else f"{parsed:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("logs/akt_also_results_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/AKT_ALSO_实验结果汇总.md"))
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    pending = [row for row in rows if row.get("status") != "evaluated"]
    evaluated.sort(
        key=lambda row: (
            -number(row.get("student_stats_mean", "")),
            number(row.get("student_stats_std", "")),
            -number(row.get("overall_dataset_auc", "")),
        )
    )

    lines = [
        "# AKT + ALSO Experiment Results",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Evaluation Criteria",
        "",
        "Rank evaluated configurations by higher student-level mean AUC, then lower student-level standard deviation, while monitoring overall dataset AUC and range.",
        "",
        "## Completed Runs",
        "",
        "| Run | ALSO | Batch | pi_lr | pi_decay | Loss scale | Alpha | Mode | Overall AUC | Student mean | Student std | Student range |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in evaluated:
        lines.append(
            "| {run} | {use_also} | {batch_size} | {also_pi_lr} | {also_pi_decay} | "
            "{also_loss_scale} | {also_alpha} | {also_mode} | {overall} | {mean} | {std} | {range_} |".format(
                run=row.get("run", "-"),
                use_also=row.get("use_also", "-"),
                batch_size=row.get("batch_size", "-"),
                also_pi_lr=row.get("also_pi_lr", "-"),
                also_pi_decay=row.get("also_pi_decay", "-"),
                also_loss_scale=row.get("also_loss_scale", "-") or "-",
                also_alpha=row.get("also_alpha", "-") or "-",
                also_mode=row.get("also_mode", "-") or "-",
                overall=fmt(row.get("overall_dataset_auc", "")),
                mean=fmt(row.get("student_stats_mean", "")),
                std=fmt(row.get("student_stats_std", "")),
                range_=fmt(row.get("student_stats_range", "")),
            )
        )
    if not evaluated:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Pending Or Incomplete Runs", ""])
    if pending:
        for row in pending:
            lines.append(
                f"- `{row.get('run')}` / `{Path(row.get('checkpoint_dir', '')).name}`: {row.get('status')}"
            )
    else:
        lines.append("All discovered runs have student-level evaluation artifacts.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `overall_dataset_auc` is calculated from the saved student-level prediction artifact and is tracked alongside the model-level test AUC in `all_results.json`.",
            "- A student range of 1.0 indicates that at least one student has AUC 0.0 and another has AUC 1.0; interpret it together with sample counts and IQR.",
            "- Historical `akt_abl_*` runs from the old implementation are deliberately excluded.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(evaluated)} completed and {len(pending)} pending runs")


if __name__ == "__main__":
    main()
