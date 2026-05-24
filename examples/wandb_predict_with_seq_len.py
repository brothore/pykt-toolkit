"""
wandb_predict_with_seq_len.py

固定长度（seq_len ± tolerance）学生子集上的不平衡性评估。
和 wandb_predict_with_unbalance.py 的差别：
  - 推理只跑 test_question_window 通路（论文 Table 3 实验所需的那一路）；
  - 落盘 txt 后, 不再对全体学生算 1 套指标, 而是按多个 target_seq_len 各取
    L_i ∈ seq_len * (1 ± tolerance) 的子集, 分别算: overall_auc / mean / std /
    range / iqr / q1 / q3 / Gini / EAWI(alpha=1,2,3)。
  - 复用 cal_unbalance.py 里的 safe_roc_auc / sliding_window_collect /
    calculate_wealth_metrics / gini_coefficient, 不重复造轮子。
"""
import os
import time
import json
import copy
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from cal_unbalance import (
    safe_roc_auc,
    sliding_window_collect,
    calculate_wealth_metrics,
    ENABLE_SLIDING_WINDOW,
    WINDOW_SIZE,
)
from pykt.models import evaluate, evaluate_question, load_model  # noqa: F401
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_seq_lens(s):
    if not s:
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


# ────────────────────────── 学生级指标解析 ──────────────────────────

def build_student_records(file_path):
    """
    读 txt -> 返回 {orirow: dict(L=, trues=np.array, scores=np.array, auc=)}。
    L = 该学生在 txt 里的行数 = 学生真实交互长度 (orirow 来自原 test.csv 的行号,
    1 学生 1 行, 所以 orirow 等价于 uid; 见 split_datasets.generate_question_sequences:392)。

    AUC 的计算口径与 cal_unbalance.parse_and_calculate_aucs_from_file 保持一致:
    ENABLE_SLIDING_WINDOW=True 时对该学生序列做 stride=1 的窗口汇集后再算 AUC。
    """
    data = pd.read_csv(file_path, sep=r"\s+")
    required = ["orirow", "late_trues", "late_mean"]
    if not all(c in data.columns for c in required):
        data.columns = data.columns.str.strip()
        if not all(c in data.columns for c in required):
            raise ValueError(f"txt 缺列 {required}, 实际列: {list(data.columns)}")

    data["late_trues"] = pd.to_numeric(data["late_trues"], errors="coerce")
    data["late_mean"] = pd.to_numeric(data["late_mean"], errors="coerce")
    data = data.dropna(subset=["late_trues", "late_mean"])

    students = {}
    for orirow, group in data.groupby("orirow"):
        trues = group["late_trues"].values
        scores = group["late_mean"].values
        L = len(trues)

        if ENABLE_SLIDING_WINDOW:
            ct, cs = sliding_window_collect(trues, scores, window_size=WINDOW_SIZE)
        else:
            ct, cs = trues.tolist(), scores.tolist()

        if len(np.unique(ct)) >= 2 and len(ct) > 0:
            auc = float(safe_roc_auc(ct, cs))
        else:
            auc = 0.5

        students[int(orirow)] = {
            "L": int(L),
            "auc": auc,
            "trues": np.asarray(ct),
            "scores": np.asarray(cs),
        }
    return students


def bucket_metrics(students, target_L, tol):
    """
    对一个固定长度 target_L 在 students 里筛选 |L - target_L| / target_L <= tol 的学生子集,
    计算桶内 overall AUC + 学生 AUC 的 mean/std/max/min/range/iqr/q1/q3 + Gini/EAWI。
    """
    lo = target_L * (1.0 - tol)
    hi = target_L * (1.0 + tol)
    chosen = [(orirow, s) for orirow, s in students.items() if lo <= s["L"] <= hi]

    info = {
        "target_seq_len": int(target_L),
        "tolerance": float(tol),
        "len_range": [float(lo), float(hi)],
        "n_students": len(chosen),
        "student_uids": [orirow for orirow, _ in chosen],
        "student_lengths": [s["L"] for _, s in chosen],
    }
    if not chosen:
        return info

    flat_trues = np.concatenate([s["trues"] for _, s in chosen])
    flat_scores = np.concatenate([s["scores"] for _, s in chosen])
    if len(np.unique(flat_trues)) >= 2:
        info["overall_dataset_auc"] = float(safe_roc_auc(flat_trues, flat_scores))
    else:
        info["overall_dataset_auc"] = 0.5

    aucs = np.array([s["auc"] for _, s in chosen], dtype=float)
    mean_auc = float(aucs.mean())
    std_auc = float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0
    max_auc = float(aucs.max())
    min_auc = float(aucs.min())
    range_auc = max_auc - min_auc

    avg_w, gini, eawi = calculate_wealth_metrics(aucs.tolist(), alphas=[1.0, 2.0, 3.0])
    q1 = float(np.percentile(aucs, 25))
    q3 = float(np.percentile(aucs, 75))

    info.update({
        "student_stats_mean": mean_auc,
        "student_stats_std": std_auc,
        "student_stats_max": max_auc,
        "student_stats_min": min_auc,
        "student_stats_range": float(range_auc),
        "student_stats_iqr": float(q3 - q1),
        "student_stats_q1": q1,
        "student_stats_q3": q3,
        "student_stats_average_auc": float(avg_w),
        "student_stats_gini_coefficient": float(gini),
        **{f"student_stats_{k}": float(v) for k, v in eawi.items()},
    })
    return info


def evaluate_seq_len_buckets(file_path, target_seq_lens, tolerance,
                             out_json=None, out_csv=None):
    students = build_student_records(file_path)
    buckets = [bucket_metrics(students, L, tolerance) for L in target_seq_lens]

    # 学生级 AUC 落盘 (含 L), 给后续画图/复查
    per_student_rows = [
        {"orirow": orirow, "L": s["L"], "auc": s["auc"]}
        for orirow, s in sorted(students.items())
    ]
    base_dir = os.path.dirname(file_path)
    if out_csv is None:
        out_csv = os.path.join(base_dir, "student_auc_with_length.csv")
    if out_json is None:
        out_json = os.path.join(base_dir, "seq_len_bucket_stats.json")

    pd.DataFrame(per_student_rows).to_csv(out_csv, index=False)
    print(f"[ok] 学生 AUC + 长度落盘 -> {out_csv}")

    payload = {
        "source_file": os.path.abspath(file_path),
        "window_size": WINDOW_SIZE,
        "sliding_window_enabled": bool(ENABLE_SLIDING_WINDOW),
        "n_students_total": len(students),
        "tolerance": float(tolerance),
        "target_seq_lens": [int(x) for x in target_seq_lens],
        "buckets": buckets,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda x: round(x, 6) if isinstance(x, float) else x)
    print(f"[ok] 桶级统计落盘 -> {out_json}")

    # 控制台简报
    print("\n=== seq_len buckets summary ===")
    print(f"  total students in txt : {len(students)}")
    for b in buckets:
        if b["n_students"] == 0:
            print(f"  L={b['target_seq_len']:>6d} (±{tolerance*100:.0f}%): no student")
            continue
        print(
            f"  L={b['target_seq_len']:>6d} (±{tolerance*100:.0f}%): "
            f"n={b['n_students']:>4d}  "
            f"overall_auc={b['overall_dataset_auc']:.4f}  "
            f"mean={b['student_stats_mean']:.4f}  "
            f"std={b['student_stats_std']:.4f}  "
            f"range={b['student_stats_range']:.4f}  "
            f"gini={b['student_stats_gini_coefficient']:.4f}"
        )
    return payload


# ────────────────────────── 推理入口 ──────────────────────────

def main(params):
    start_total = time.time()
    print(f"--- [Time] Start: {_now()} ---")

    if params["use_wandb"] == 1:
        import wandb
        os.environ["WANDB_API_KEY"] = wandb_config["api_key"]
        wandb.init(project="wandb_predict_seq_len")

    save_dir = params["save_dir"]
    batch_size = params["bz"]
    fusion_type = params["fusion_type"].split(",")
    target_seq_lens = parse_seq_lens(params["seq_lens"])
    tolerance = params["tolerance"]

    # 1) 配置加载
    t1 = time.time()
    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
            model_config.pop(remove_item, None)
        trained_params = config["params"]
        fold = trained_params["fold"]
        model_name = trained_params["model_name"]
        dataset_name = trained_params["dataset_name"]
        emb_type = trained_params["emb_type"]
        if model_name in ["saint", "sakt", "atdkt"]:
            model_config["seq_len"] = config["train_config"]["seq_len"]

    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time", "dbakt"]:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name == "lpkt":
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"]

    if model_name not in ["dimkt"]:
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = \
            init_test_datasets(data_config, model_name, batch_size)
    else:
        diff_level = trained_params["difficult_levels"]
        test_loader, test_window_loader, test_question_loader, test_question_window_loader = \
            init_test_datasets(data_config, model_name, batch_size, diff_level=diff_level)
    print(f"[Time] data loaders: {time.time() - t1:.2f}s")

    # 2) 加载模型
    t2 = time.time()
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    print(f"[Time] model load : {time.time() - t2:.2f}s")

    # rkt 需要预加载 rel
    rel = None
    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        ds_short = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if ds_short in ["algebra2005", "bridge2algebra2006"]:
            rel = pd.read_pickle(os.path.join(dpath, "phi_dict" + folds_str + ".pkl"))
        else:
            rel = pd.read_pickle(os.path.join(dpath, "phi_array" + folds_str + ".pkl"))

    # 3) 推理: 优先 question_window, 缺啥退啥
    #    两路 txt 都含 orirow / late_trues / late_mean, build_student_records 兼容。
    rkt_kw = {"rel": rel} if model.model_name == "rkt" else {}
    candidates = [
        ("test_question_window_file", test_question_window_loader,
         "_test_question_window_predictions.txt", "evaluate_question"),
        ("test_window_file",          test_window_loader,
         "_test_window_predictions.txt",          "evaluate"),
        ("test_question_file",        test_question_loader,
         "_test_question_predictions.txt",        "evaluate_question"),
        ("test_file",                 test_loader,
         "_test_predictions.txt",                 "evaluate"),
    ]
    save_path, eval_source = None, None
    for cfg_key, loader, suffix, eval_kind in candidates:
        if loader is None or cfg_key not in data_config:
            continue
        save_path = os.path.join(save_dir, model.emb_type + suffix)
        eval_source = cfg_key
        t3 = time.time()
        if eval_kind == "evaluate_question":
            aucs, accs = evaluate_question(
                model, loader, model_name,
                fusion_type=fusion_type, save_path=save_path,
            )
        else:
            aucs, accs = evaluate(
                model, loader, model_name, save_path=save_path, **rkt_kw,
            )
        print(f"[Time] {eval_source} eval: {time.time() - t3:.2f}s")
        print(f"[{eval_source}] testaucs: {aucs}")
        print(f"[{eval_source}] testaccs: {accs}")
        break
    if save_path is None:
        raise RuntimeError(
            "当前 dataset 没有任何可用的 test loader / test_*_file, 无法评估。"
        )

    # 4) 桶级指标
    t4 = time.time()
    out_json = os.path.join(save_dir, "seq_len_bucket_stats.json")
    out_csv = os.path.join(save_dir, "student_auc_with_length.csv")
    payload = evaluate_seq_len_buckets(
        save_path, target_seq_lens, tolerance,
        out_json=out_json, out_csv=out_csv,
    )
    print(f"[Time] seq_len bucketing  : {time.time() - t4:.2f}s")

    # 5) wandb 上报: 把每个桶的关键指标拍平上报
    if params["use_wandb"] == 1:
        flat = {
            "source_file": payload["source_file"],
            "n_students_total": payload["n_students_total"],
            "tolerance": payload["tolerance"],
        }
        for b in payload["buckets"]:
            L = b["target_seq_len"]
            flat[f"L{L}_n_students"] = b["n_students"]
            for k, v in b.items():
                if k.startswith("student_stats_") or k == "overall_dataset_auc":
                    flat[f"L{L}_{k}"] = v
        wandb.log(flat)

    print("-" * 30)
    print(f"Total time: {time.time() - start_total:.2f}s (end {_now()})")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument(
        "--seq_lens", type=str, default="20000,30000",
        help="逗号分隔的多个目标长度, 例: 20000,30000",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="长度容差比例, 默认 0.05 即 ±5%%",
    )
    parser.add_argument(
        "--from_txt", type=str, default=None,
        help="跳过推理, 直接读已有 *_test_question_window_predictions.txt 算桶指标",
    )

    args = parser.parse_args()
    print(args)

    if args.from_txt:
        # 旁路: 已经有 txt 时直接做桶级指标, 不重新推理
        evaluate_seq_len_buckets(
            args.from_txt,
            parse_seq_lens(args.seq_lens),
            args.tolerance,
        )
    else:
        main(vars(args))
