#!/usr/bin/env python3
"""Render the current AKT/ALSO result CSV as a Markdown experiment report."""

from __future__ import annotations

import argparse
import csv
import math
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


def config_name(row: dict[str, str]) -> str:
    """Make the parameter setting identifiable when multiple rows share a run name."""
    if row.get("use_also") != "1":
        return f"Baseline, batch={row.get('batch_size', '-') }"

    parts = [
        "ALSO",
        f"batch={row.get('batch_size', '-')}",
        f"pi_lr={row.get('also_pi_lr', '-')}",
        f"pi_decay={row.get('also_pi_decay', '-')}",
    ]
    if row.get("also_loss_scale"):
        parts.append(f"loss_scale={row['also_loss_scale']}")
    if row.get("also_alpha") and row["also_alpha"] != "1.0":
        parts.append(f"alpha={row['also_alpha']}")
    if row.get("also_mode") and row["also_mode"] != "optimistic":
        parts.append(f"mode={row['also_mode']}")
    return ", ".join(parts)


def best_row(rows: list[dict[str, str]], key: str, reverse: bool) -> dict[str, str] | None:
    valid = [row for row in rows if number(row.get(key, "")) == number(row.get(key, ""))]
    return (max if reverse else min)(valid, key=lambda row: number(row[key])) if valid else None


def best_metric_value(rows: list[dict[str, str]], key: str, reverse: bool) -> float:
    values = [number(row.get(key, "")) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan")
    return max(values) if reverse else min(values)


def fmt_metric(row: dict[str, str], key: str, best_value: float) -> str:
    value = number(row.get(key, ""))
    rendered = fmt(row.get(key, ""))
    if not math.isnan(value) and not math.isnan(best_value) and math.isclose(value, best_value):
        return f"**{rendered}**"
    return rendered


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
    best_overall_auc = best_metric_value(evaluated, "overall_dataset_auc", reverse=True)
    best_student_mean = best_metric_value(evaluated, "student_stats_mean", reverse=True)
    best_student_std = best_metric_value(evaluated, "student_stats_std", reverse=False)
    best_student_range = best_metric_value(evaluated, "student_stats_range", reverse=False)

    lines = [
        "# AKT + ALSO 阶段性实验报告",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "本实验已将 ALSO 适配到 AKT 的训练流程：按真实学生 ID 建立无碰撞分组，并按每名学生的序列数归一化 ALSO 损失与学生级评测。",
        "",
        "## AKT 运行 ALSO 的代码适配说明",
        "",
        "1. **逐序列损失接口**：AKT 原训练流程使用 batch 标量损失；为满足 ALSO 对组损失向量与 closure 的要求，训练前向新增逐序列 BCE loss 输出，并由 ALSO 对该向量进行加权更新。",
        "2. **学生级无碰撞分组**：从训练折的 `uid` 构建 `uid -> group_id` 一一映射，一名学生对应一个 ALSO group；不再使用可能碰撞的取模分组，也不会把验证或测试学生并入训练 group。",
        "3. **序列切分归一化**：同一学生被切为多条序列时，每条序列损失除以该学生的训练序列数，使学生总贡献近似相等，避免长序列学生在 min-max 目标中被重复放大。",
        "4. **训练参数贯通**：AKT 启动器和通用训练器接入 `use_also`、`grouping_mode`、`n_groups`、`mode`、`alpha`、`pi_lr`、`pi_decay` 与 `loss_scale`，保留 `PYTHONPATH=$PWD python -m examples.wandb_akt_train ...` 的启动方式。",
        "5. **论文超参数与变体**：`n_groups` 由训练学生数推断，loss scale 可按 `n_groups / batch_size` 设置；同时支持 optimistic 的 `alpha` 消融与 `descent-ascent` 变体。",
        "6. **一条龙评测修复**：预测子进程继承项目 `PYTHONPATH`，并修复预测结束后错误恢复旧参数的问题，使训练、预测和学生级统计可连续完成。",
        "7. **学生级结果追踪**：自动汇总 overall AUC、学生 AUC 均值、标准差和极差，并检测缺失 `all_results.json` 或 `overall_stats_output.json` 的实验。",
        "",
        "## 状态与评价目标",
        "",
        f"当前已完成 **{len(evaluated)}** 个新代码实验的学生级评测；另有 **{len(pending)}** 个实验仍在训练或评测中，不能将其视为负结果。",
        "",
        "排序优先级为更高的学生级平均 AUC，其次为更低的学生级标准差；同时监控 overall AUC。学生级极差仅用于诊断异常值，不单独作为优劣依据。",
        "",
        "## 已完成学生级评测",
        "",
        "| 配置 | Overall AUC | 学生平均 AUC | 学生 AUC 标准差 | 学生 AUC 极差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in evaluated:
        lines.append(
            "| {config} | {overall} | {mean} | {std} | {range_} |".format(
                config=config_name(row),
                overall=fmt_metric(row, "overall_dataset_auc", best_overall_auc),
                mean=fmt_metric(row, "student_stats_mean", best_student_mean),
                std=fmt_metric(row, "student_stats_std", best_student_std),
                range_=fmt_metric(row, "student_stats_range", best_student_range),
            )
        )
    if not evaluated:
        lines.append("| - | - | - | - | - |")

    lines.extend(
        [
            "",
            "加粗数值为当前已完成实验中该列的最优值：Overall AUC 与学生平均 AUC 取最大，学生标准差与极差取最小；并列最优值均加粗。",
        ]
    )

    baseline_rows = [row for row in evaluated if row.get("use_also") == "0"]
    also_rows = [row for row in evaluated if row.get("use_also") == "1"]
    best_mean = best_row(evaluated, "student_stats_mean", reverse=True)
    lowest_std = best_row(evaluated, "student_stats_std", reverse=False)
    best_also_mean = best_row(also_rows, "student_stats_mean", reverse=True)
    lowest_also_std = best_row(also_rows, "student_stats_std", reverse=False)

    lines.extend(["", "## 阶段性分析", ""])
    if best_mean:
        lines.append(
            f"- **当前最高学生平均 AUC**：{config_name(best_mean)}，"
            f"学生平均 AUC={fmt(best_mean.get('student_stats_mean', ''))}。"
        )
    if lowest_std:
        lines.append(
            f"- **当前最低学生标准差**：{config_name(lowest_std)}，"
            f"标准差={fmt(lowest_std.get('student_stats_std', ''))}。"
        )
    if best_also_mean and lowest_also_std:
        lines.append(
            f"- **已完成 ALSO 中的取舍**：最高学生平均 AUC 来自 {config_name(best_also_mean)} "
            f"({fmt(best_also_mean.get('student_stats_mean', ''))})；最低标准差来自 "
            f"{config_name(lowest_also_std)} ({fmt(lowest_also_std.get('student_stats_std', ''))})。"
        )
    if baseline_rows and also_rows:
        baseline = best_row(baseline_rows, "student_stats_mean", reverse=True)
        if baseline and best_also_mean:
            mean_gap = number(best_also_mean["student_stats_mean"]) - number(baseline["student_stats_mean"])
            auc_gap = number(best_also_mean["overall_dataset_auc"]) - number(baseline["overall_dataset_auc"])
            lines.append(
                f"- **与当前最佳 baseline 的比较**：已完成 ALSO 的最佳学生平均 AUC 仍低 "
                f"{abs(mean_gap):.6f}，overall AUC 低 {abs(auc_gap):.6f}；因此现有结果尚未证明 ALSO 达到"
                "“提高学生平均 AUC 且降低离散度”的目标。"
            )
    if evaluated and all(number(row.get("student_stats_range", "")) == 1.0 for row in evaluated):
        lines.append("- **极差诊断**：所有已完成配置的学生级 AUC 极差均为 1.0，说明极端学生 AUC 仍同时出现 0 与 1；本轮消融尚未改善该指标。")

    lines.extend(["", "## 待完成实验", ""])
    if pending:
        for row in pending:
            lines.append(
                f"- `{config_name(row)}`（{row.get('status')}）"
            )
    else:
        lines.append("所有发现的新代码实验均已生成学生级评测产物。")

    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- `overall_dataset_auc` 从保存的学生级预测产物计算，并与 `all_results.json` 中的模型级测试指标一并记录。",
            "- 极差为 1.0 表示至少有学生 AUC 为 0.0，且至少有学生 AUC 为 1.0；应结合学生样本数和四分位数解释。",
            "- 历史 `akt_abl_*` 旧代码训练结果已按要求排除。",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with {len(evaluated)} completed and {len(pending)} pending runs")


if __name__ == "__main__":
    main()
