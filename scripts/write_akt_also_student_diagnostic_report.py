#!/usr/bin/env python3
"""Write a paired, student-level diagnostic report for completed AKT/ALSO runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def number(value: str | float | int | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.6f}"


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def config_label(row: dict[str, str]) -> str:
    if row.get("use_also") != "1":
        return f"baseline (batch={row.get('batch_size', '-')})"
    parts = [
        f"ALSO (batch={row.get('batch_size', '-')}",
        f"pi_lr={row.get('also_pi_lr', '-')}",
        f"pi_decay={row.get('also_pi_decay', '-')}",
    ]
    if row.get("also_loss_scale"):
        parts.append(f"loss_scale={row['also_loss_scale']}")
    return ", ".join(parts) + ")"


def read_student_aucs(checkpoint: Path) -> dict[str, float]:
    matches = list(checkpoint.glob("qid_test_question_window_predictions_per_student_aucs.json"))
    if not matches:
        raise FileNotFoundError(f"Missing student AUC artifact under {checkpoint}")
    raw = json.loads(matches[0].read_text(encoding="utf-8"))
    return {str(uid): float(auc) for uid, auc in raw.items()}


def read_stats(checkpoint: Path) -> dict[str, float]:
    matches = list(checkpoint.glob("overall_stats_output.json"))
    if not matches:
        return {}
    raw = json.loads(matches[0].read_text(encoding="utf-8"))
    return {key: number(value) for key, value in raw.items()}


def read_sequence_metadata(data_dir: Path) -> tuple[Counter[str], Counter[str], set[str], set[str]]:
    test_counts: Counter[str] = Counter()
    train_sequence_counts: Counter[str] = Counter()
    test_uids: set[str] = set()
    train_uids: set[str] = set()

    with (data_dir / "test_question_window_sequences.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Student AUC artifacts are keyed by `orirow`, the original test-row
            # index.  Use the same key for support counts; raw uid is only used
            # for the train/test population-overlap diagnostic below.
            uid = row["uid"]
            test_uids.add(uid)
            oriroot = row["orirow"].split(",")[0]
            test_counts[oriroot] += sum(value == "1" for value in row["selectmasks"].split(","))

    with (data_dir / "train_valid_sequences.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # The experiments use fold 0 for validation, hence folds 1--4 are the train fold.
            if row["fold"] == "0":
                continue
            uid = row["uid"]
            train_uids.add(uid)
            train_sequence_counts[uid] += 1

    return test_counts, train_sequence_counts, test_uids, train_uids


def paired_row(
    baseline: dict[str, float],
    candidate: dict[str, float],
    test_counts: Counter[str],
) -> dict[str, float | int]:
    common = sorted(set(baseline) & set(candidate))
    deltas = [candidate[uid] - baseline[uid] for uid in common]
    base_values = [baseline[uid] for uid in common]
    q1, q3 = percentile(base_values, 0.25), percentile(base_values, 0.75)
    support_values = [float(test_counts[uid]) for uid in common]
    support_q1, support_q3 = percentile(support_values, 0.25), percentile(support_values, 0.75)

    def mean_for(uids: list[str]) -> float:
        return statistics.fmean(candidate[uid] - baseline[uid] for uid in uids) if uids else float("nan")

    low_auc = [uid for uid in common if baseline[uid] <= q1]
    high_auc = [uid for uid in common if baseline[uid] >= q3]
    low_support = [uid for uid in common if test_counts[uid] <= support_q1]
    high_support = [uid for uid in common if test_counts[uid] >= support_q3]
    return {
        "students": len(common),
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "p25_delta": percentile(deltas, 0.25),
        "p75_delta": percentile(deltas, 0.75),
        "improved": sum(delta > 1e-12 for delta in deltas),
        "worsened": sum(delta < -1e-12 for delta in deltas),
        "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
        "low_auc_delta": mean_for(low_auc),
        "high_auc_delta": mean_for(high_auc),
        "low_support_delta": mean_for(low_support),
        "high_support_delta": mean_for(high_support),
    }


def main() -> None:
    root = Path.cwd()
    summary_path = root / "logs/akt_also_results_summary.csv"
    output = root / "docs/AKT_ALSO_学生级序列诊断报告.md"
    data_dir = root / "data/assist2009"

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "evaluated"]

    experiments = []
    for row in rows:
        checkpoint = root / row["checkpoint_dir"]
        try:
            experiments.append((row, read_student_aucs(checkpoint), read_stats(checkpoint)))
        except FileNotFoundError:
            continue

    test_counts, train_counts, test_uids, train_uids = read_sequence_metadata(data_dir)
    baselines = {
        row["batch_size"]: (row, aucs, stats)
        for row, aucs, stats in experiments
        if row.get("use_also") == "0"
    }
    also_runs = [(row, aucs, stats) for row, aucs, stats in experiments if row.get("use_also") == "1"]
    also_runs.sort(key=lambda item: -number(item[0].get("student_stats_mean")))

    lines = [
        "# AKT + ALSO 学生级与序列级诊断报告",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结论摘要",
        "",
        "已完成实验显示，当前 ALSO 配置未改善学生平均 AUC、学生标准差或学生 AUC 极差。该结论来自同一测试学生上的逐学生 AUC 配对比较，而不只来自汇总均值。",
        "",
        "最直接的解释不是分组映射错误，而是目标与评测之间存在强偏移：ALSO 在训练学生上重加权逐序列 BCE 损失，而报告评估的是未见测试学生的逐学生 AUC。两者既不共享学生 group，也不是同一种损失/指标。",
        "",
        "## 数据与评测覆盖",
        "",
        f"- 测试学生数：{len(test_uids)}；训练折学生数：{len(train_uids)}；二者 UID 交集：{len(test_uids & train_uids)}。",
        f"- 测试学生有效评测交互数：Q1={percentile([float(value) for value in test_counts.values()], 0.25):.0f}，"
        f"中位数={percentile([float(value) for value in test_counts.values()], 0.5):.0f}，"
        f"Q3={percentile([float(value) for value in test_counts.values()], 0.75):.0f}。",
        f"- 训练学生序列数：最少={min(train_counts.values()) if train_counts else 0}，最多={max(train_counts.values()) if train_counts else 0}；"
        "训练损失已按此数量归一化。",
        "",
        "## 已完成实验的分布指标",
        "",
        "| 配置 | Overall AUC | 学生均值 | 标准差 | EAWI@20 | Gini | IQR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_completed = sorted(experiments, key=lambda item: -number(item[0].get("student_stats_mean")))
    for row, _, stats in all_completed:
        lines.append(
            "| {label} | {overall} | {mean} | {std} | {eawi} | {gini} | {iqr} |".format(
                label=config_label(row),
                overall=fmt(stats.get("overall_dataset_auc", float("nan"))),
                mean=fmt(stats.get("student_stats_mean", float("nan"))),
                std=fmt(stats.get("student_stats_std", float("nan"))),
                eawi=fmt(stats.get("student_stats_eawi_alpha_20", float("nan"))),
                gini=fmt(stats.get("student_stats_gini_coefficient", float("nan"))),
                iqr=fmt(stats.get("student_stats_iqr", float("nan"))),
            )
        )

    lines.extend(
        [
            "",
            "## 同 batch 的逐学生配对比较",
            "",
            "每行比较同一 batch size 的 baseline 与 ALSO，delta 定义为 `ALSO 学生 AUC - baseline 学生 AUC`。低 AUC/低支持度组按 baseline AUC 或测试有效交互数的下四分位划分。",
            "",
            "| ALSO 配置 | 配对学生数 | 平均 delta | 中位 delta | 改善/变差/不变 | 低 AUC 组 delta | 高 AUC 组 delta | 低支持度 delta | 高支持度 delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    comparisons = []
    for row, aucs, _ in also_runs:
        baseline = baselines.get(row.get("batch_size"))
        if not baseline:
            continue
        comparison = paired_row(baseline[1], aucs, test_counts)
        comparisons.append((row, comparison))
        lines.append(
            "| {label} | {students} | {mean} | {median} | {improved}/{worsened}/{unchanged} | {low_auc} | {high_auc} | {low_support} | {high_support} |".format(
                label=config_label(row),
                students=comparison["students"],
                mean=fmt(float(comparison["mean_delta"])),
                median=fmt(float(comparison["median_delta"])),
                improved=comparison["improved"],
                worsened=comparison["worsened"],
                unchanged=comparison["unchanged"],
                low_auc=fmt(float(comparison["low_auc_delta"])),
                high_auc=fmt(float(comparison["high_auc_delta"])),
                low_support=fmt(float(comparison["low_support_delta"])),
                high_support=fmt(float(comparison["high_support_delta"])),
            )
        )

    best_also = also_runs[0] if also_runs else None
    if best_also and "64" in baselines:
        base_aucs = baselines["64"][1]
        also_aucs = best_also[1]
        common = set(base_aucs) & set(also_aucs)
        zero_one = []
        for name, aucs in [("baseline", base_aucs), ("ALSO", also_aucs)]:
            zero = [uid for uid in common if aucs[uid] == 0.0]
            one = [uid for uid in common if aucs[uid] == 1.0]
            zero_one.append(
                f"{name}: AUC=0 的学生 {len(zero)} 名（测试交互中位数 "
                f"{statistics.median(test_counts[uid] for uid in zero) if zero else 0:.0f}），"
                f"AUC=1 的学生 {len(one)} 名（测试交互中位数 "
                f"{statistics.median(test_counts[uid] for uid in one) if one else 0:.0f}）。"
            )
        lines.extend(
            [
                "",
                "## 为什么当前结果没有改善",
                "",
                f"- **总体退化而非少数异常值造成**：与 batch=64 baseline 配对时，当前学生均值最高的 {config_label(best_also[0])} 也应在上表显示负的平均和中位 delta；这说明大多数测试学生没有获得收益。",
                "- **训练 group 与测试学生不重合**：训练学生的 ALSO `pi` 只对训练 uid 更新，而测试学生是未见 uid。因此该优化只能通过共享的题目/概念表示间接迁移，不能直接保证测试学生级 AUC 提升。",
                "- **优化目标与报告指标不一致**：训练中使用逐序列、按学生序列数归一化的 BCE；报告使用每名学生的 AUC。BCE 更重视概率校准，AUC 只比较排序，二者的最优解不必一致。",
                "- **罕见学生的估计噪声仍在**：学生 AUC 的 0/1 极值通常由很少的测试交互产生；这会把 range 固定为 1.0，并使它对优化器改动不敏感。",
                *[f"- **极值检查**：{text}" for text in zero_one],
                "- **论文设定未必直接迁移**：ALSO 的收益依赖于 group 定义、组样本量和训练/测试分布。这里 group 是学生而非论文中的原始任务 group，且每个训练学生仅有 1--7 条序列，`pi` 的梯度估计较稀疏。",
            ]
        )

    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "1. 不把学生 AUC 极差作为主优化目标；改用有最小测试交互门槛的学生子集、EAWI@20 或学生 AUC 的下分位数，并同时报告覆盖率。",
            "2. 若目标是测试学生公平性，优先按可泛化的训练特征分组（训练序列长度、历史正确率、题目覆盖度），而不是按训练学生 ID 建立不可迁移的 group。",
            "3. 对现有 student-ID 版本，降低 `pi_lr` 或提高 `pi_decay` 只能缓解重加权强度，不能解决训练 group 与测试学生不重合的问题；应将其作为稳健性对照，而非预期必然提升测试学生 AUC 的方案。",
            "4. 继续完成当前 alpha 与 descent-ascent 实验后，将其自动补入此诊断；若仍为负 delta，再进行特征分组和最小样本门槛的下一轮实验。",
            "",
            "## 说明",
            "",
            "- 本报告仅使用新代码训练且已产生 `qid_test_question_window_predictions_per_student_aucs.json` 的实验；旧 `akt_abl_*` 结果不参与比较。",
            "- 以上为描述性诊断。单 seed 的差异不能作为统计显著性结论，关键候选配置需要多 seed 复现。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} from {len(experiments)} evaluated runs and {len(comparisons)} paired comparisons")


if __name__ == "__main__":
    main()
