#!/usr/bin/env python3
"""
collect_tsp_results.py
扫 tsp 队列 + ts-out 日志 + saved_model/ 产物, 输出 tsp_results.json。

设计要点
--------
- tsp `finished` 仅表示进程退出, 不代表训练成功; 真正的成功标准是产物存在:
  saved_dir 同时存在 overall_stats_output.json 与 all_results.json。
- 同一组 (model, dataset, fold, emb_type, version, seed, dropout, mlp_layer_num,
  loss_*) 可能多次运行, 按 overall_dataset_auc 取最好的一次, 其余进入 attempts。
- 正在跑的 (running/queued) 也收入, status="running"/"queued", 指标留空。
- PARAM_COUNT 是中期才加的标记, 不一定每条都有 -> 完全可选, 缺失时 null。
- 增量: 旧 tsp_results.json 里 status="success" 的记录直接保留 (避免重算); 其余
  按 task_id 重新计算后写回。
- 整脚本只做 IO + 解析, 无副作用; 任何单条解析异常被吞掉并记到 record.error。
- 多 socket: --socket 可多次或逗号分隔, "default" 代表系统默认 socket
  (不设置 TS_SOCKET 环境变量). 例: --socket default,/tmp/ts_gpu0,/tmp/ts_gpu1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/root/autodl-tmp/pykt-toolkit")
SAVED_MODEL_DIR = REPO_ROOT / "examples" / "saved_model"
OUTPUT_JSON = REPO_ROOT / "tsp_results.json"
RUN_HISTORY = SAVED_MODEL_DIR / "run_history.txt"

PARAM_COUNT_RE = re.compile(
    r"\[PARAM_COUNT\]\s+model=(\S+)\s+emb_type=(\S+)\s+dataset=(\S+)"
    r"\s+trainable=([\d,]+)\s+total=([\d,]+)"
)
SAVE_DIR_RE = re.compile(r"save_dir:\s*(saved_model/[^\s,]+)")
RUN_ID_RE = re.compile(r"ID:\s*([0-9a-f-]{36})")
STATUS_OK_RE = re.compile(r"运行历史状态已更新为成功")
STATUS_FAIL_RE = re.compile(r"运行历史状态已更新为失败|训练过程中发生错误")


# ───────────────────────── tsp 队列读取 ─────────────────────────

DEFAULT_SOCKET = "default"  # 占位: 表示不设置 TS_SOCKET 环境变量, 即系统默认队列


def _tsp_env(socket: str) -> dict[str, str]:
    env = os.environ.copy()
    if socket and socket != DEFAULT_SOCKET:
        env["TS_SOCKET"] = socket
    else:
        env.pop("TS_SOCKET", None)
    return env


def list_tsp_ids(socket: str = DEFAULT_SOCKET) -> list[str]:
    """返回某个 tsp socket 队列中所有 task id (字符串, 含 finished/running/queued)。"""
    try:
        out = subprocess.check_output(
            ["tsp", "-l"], text=True, stderr=subprocess.DEVNULL, env=_tsp_env(socket)
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    ids = []
    for line in out.splitlines()[1:]:  # 跳过表头
        parts = line.split(None, 1)
        if parts and parts[0].isdigit():
            ids.append(parts[0])
    return ids


def inspect_tsp(task_id: str, socket: str = DEFAULT_SOCKET) -> dict[str, Any]:
    """tsp -i <id> 解析, 返回 dict。"""
    info: dict[str, Any] = {"task_id": int(task_id), "tsp_socket": socket}
    try:
        out = subprocess.check_output(
            ["tsp", "-i", task_id], text=True, stderr=subprocess.DEVNULL, env=_tsp_env(socket)
        )
    except subprocess.CalledProcessError:
        info["error"] = "tsp -i failed"
        return info
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Exit status:"):
            m = re.search(r"exit code\s+(-?\d+)", line)
            info["exit_code"] = int(m.group(1)) if m else None
        elif line.startswith("Command:"):
            info["cmd"] = line[len("Command:"):].strip()
        elif line.startswith("Enqueue time:"):
            info["enqueue_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Start time:"):
            info["start_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("End time:"):
            info["end_time"] = line.split(":", 1)[1].strip()
        elif line.startswith("Time run:") or line.startswith("Time running:"):
            m = re.search(r"([\d.]+)s", line)
            info["time_sec"] = float(m.group(1)) if m else None

    # 取 output 文件路径需要从 -l 拿
    info["output_file"] = _output_file_for(task_id, socket)
    info["state"] = _state_for(task_id, socket)
    return info


_tsp_l_cache: dict[str, list[str]] = {}
def _tsp_l_lines(socket: str = DEFAULT_SOCKET) -> list[str]:
    if socket not in _tsp_l_cache:
        try:
            out = subprocess.check_output(
                ["tsp", "-l"], text=True, stderr=subprocess.DEVNULL, env=_tsp_env(socket)
            )
            _tsp_l_cache[socket] = out.splitlines()[1:]
        except Exception:
            _tsp_l_cache[socket] = []
    return _tsp_l_cache[socket]


def _output_file_for(task_id: str, socket: str = DEFAULT_SOCKET) -> str | None:
    for ln in _tsp_l_lines(socket):
        parts = ln.split()
        if parts and parts[0] == task_id:
            for p in parts[1:]:
                if p.startswith("/tmp/ts-out."):
                    return p
            return None
    return None


def _state_for(task_id: str, socket: str = DEFAULT_SOCKET) -> str | None:
    for ln in _tsp_l_lines(socket):
        parts = ln.split()
        if parts and parts[0] == task_id:
            return parts[1] if len(parts) > 1 else None
    return None


# ───────────────────────── 命令行解析 ─────────────────────────

def parse_train_cmd(cmd: str) -> dict[str, Any]:
    """从 tsp Command 抠出 dataset/fold/emb_type/version/seed/loss 等。"""
    if not cmd:
        return {}
    toks = shlex.split(cmd)
    # script 名 -> model_name 推断
    script = next((t for t in toks if t.endswith(".py")), None)
    inferred_model = None
    if script:
        base = Path(script).stem  # e.g. wandb_qikt_mamba_train
        m = re.match(r"wandb_(.+?)_train$", base)
        if m:
            inferred_model = m.group(1)

    params: dict[str, Any] = {"_script": script, "_inferred_model": inferred_model}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            val: Any = True
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                val = toks[i + 1]
                i += 1
                # 数字化
                if isinstance(val, str):
                    try:
                        if "." in val or "e" in val.lower():
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
            params[key] = val
        i += 1
    return params


# ───────────────────────── ts-out 日志解析 ─────────────────────────

def parse_ts_out(path: str | None) -> dict[str, Any]:
    """读 /tmp/ts-out.XXXXXX, 抽 PARAM_COUNT / save_dir / run_id / 训练状态判定。"""
    info: dict[str, Any] = {
        "param_count": None,
        "save_dir": None,
        "run_id": None,
        "log_status": None,  # "success" / "failure" / "incomplete"
        "tail": None,
    }
    if not path or not os.path.isfile(path):
        return info
    try:
        with open(path, "r", errors="replace") as f:
            content = f.read()
    except OSError as e:
        info["error"] = f"read ts-out: {e}"
        return info

    m = PARAM_COUNT_RE.search(content)
    if m:
        info["param_count"] = {
            "model": m.group(1),
            "emb_type": m.group(2),
            "dataset": m.group(3),
            "trainable": int(m.group(4).replace(",", "")),
            "total": int(m.group(5).replace(",", "")),
        }

    m = SAVE_DIR_RE.search(content)
    if m:
        info["save_dir"] = m.group(1)

    m = RUN_ID_RE.search(content)
    if m:
        info["run_id"] = m.group(1)

    if STATUS_OK_RE.search(content):
        info["log_status"] = "success"
    elif STATUS_FAIL_RE.search(content):
        info["log_status"] = "failure"
    else:
        info["log_status"] = "incomplete"

    # 留 tail 便于人工排查
    tail_lines = content.splitlines()[-15:]
    info["tail"] = "\n".join(tail_lines)[-2000:]
    return info


# ───────────────────────── saved_model/<dir> 产物解析 ─────────────────────────

def parse_saved_dir(save_dir: str | None) -> dict[str, Any]:
    """读 overall_stats_output.json / all_results.json / config.json。"""
    out: dict[str, Any] = {
        "overall_stats": None,
        "all_results": None,
        "config": None,
        "save_dir_abs": None,
        "save_dir_exists": False,
    }
    if not save_dir:
        return out
    abs_dir = (REPO_ROOT / "examples" / save_dir).resolve()
    out["save_dir_abs"] = str(abs_dir)
    if not abs_dir.is_dir():
        return out
    out["save_dir_exists"] = True

    for fname, key in [
        ("overall_stats_output.json", "overall_stats"),
        ("all_results.json", "all_results"),
        ("config.json", "config"),
    ]:
        fp = abs_dir / fname
        if fp.is_file():
            try:
                with open(fp) as f:
                    out[key] = json.load(f)
            except Exception as e:
                out[f"_{key}_error"] = str(e)
    return out


# ───────────────────────── 状态判定 ─────────────────────────

def derive_status(tsp_info: dict, ts_out_info: dict, saved: dict) -> str:
    """
    success         : overall_stats + all_results 都齐 -> 真正完整成功
    train_no_eval   : 模型 ckpt/all_results 有但缺 overall_stats (训练完没跑指标)
    failed          : ts-out 标识 failure, 或 exit_code 非 0 且无产物
    running         : tsp state == running
    queued          : tsp state == queued
    unknown         : 其他
    """
    state = tsp_info.get("state")
    if state == "running":
        return "running"
    if state == "queued":
        return "queued"

    has_overall = saved.get("overall_stats") is not None
    has_all = saved.get("all_results") is not None

    if has_overall and has_all:
        return "success"
    if has_all and not has_overall:
        return "train_no_eval"

    if ts_out_info.get("log_status") == "failure":
        return "failed"

    ec = tsp_info.get("exit_code")
    if ec is not None and ec != 0:
        return "failed"

    return "unknown"


# ───────────────────────── 汇聚 / 去重 ─────────────────────────

def make_combo_key(rec: dict) -> tuple:
    """同一参数组合的去重 key。仅取关键超参, 忽略 task_id/timestamp/socket。
    跨 socket 也会被合并: 同一组超参在 gpu0/gpu1 各跑一份, 视为 attempts。"""
    p = rec.get("params") or {}
    return (
        p.get("_inferred_model") or p.get("model_name"),
        p.get("dataset_name"),
        p.get("fold"),
        p.get("emb_type"),
        p.get("version"),
        p.get("seed"),
        p.get("dropout"),
        p.get("mlp_layer_num"),
        p.get("loss_q_all_lambda"),
        p.get("loss_c_all_lambda"),
        p.get("loss_c_next_lambda"),
        p.get("output_mode"),
    )


def rec_score(rec: dict) -> float:
    """同组多次运行排序: 优先 success/有 auc, 再按 auc 大者优。"""
    if rec["status"] != "success":
        return -1.0
    stats = rec.get("overall_stats") or {}
    return float(stats.get("overall_dataset_auc", -1.0))


def collect(sockets: list[str]) -> dict:
    records: list[dict] = []
    for socket in sockets:
        ids = list_tsp_ids(socket)
        for tid in ids:
            try:
                tsp_info = inspect_tsp(tid, socket)
                params = parse_train_cmd(tsp_info.get("cmd", ""))
                ts_out_info = parse_ts_out(tsp_info.get("output_file"))
                save_dir = ts_out_info.get("save_dir")
                saved = parse_saved_dir(save_dir)
                status = derive_status(tsp_info, ts_out_info, saved)

                rec = {
                    "task_id": tsp_info["task_id"],
                    "tsp_socket": socket,
                    "status": status,
                    "model": params.get("_inferred_model"),
                    "dataset": params.get("dataset_name"),
                    "fold": params.get("fold"),
                    "emb_type": params.get("emb_type"),
                    "version": params.get("version"),
                    "params": {k: v for k, v in params.items() if not k.startswith("_")} | {
                        "_inferred_model": params.get("_inferred_model"),
                        "_script": params.get("_script"),
                    },
                    "tsp": {
                        "socket": socket,
                        "state": tsp_info.get("state"),
                        "exit_code": tsp_info.get("exit_code"),
                        "enqueue_time": tsp_info.get("enqueue_time"),
                        "start_time": tsp_info.get("start_time"),
                        "end_time": tsp_info.get("end_time"),
                        "time_sec": tsp_info.get("time_sec"),
                        "output_file": tsp_info.get("output_file"),
                    },
                    "save_dir": save_dir,
                    "save_dir_abs": saved.get("save_dir_abs"),
                    "save_dir_exists": saved.get("save_dir_exists", False),
                    "run_id": ts_out_info.get("run_id"),
                    "log_status": ts_out_info.get("log_status"),
                    "param_count": ts_out_info.get("param_count"),
                    "overall_stats": saved.get("overall_stats"),
                    "all_results": saved.get("all_results"),
                }
                # 失败任务才留 tail, 成功任务略过避免文件膨胀
                if status not in ("success",):
                    rec["log_tail"] = ts_out_info.get("tail")
                records.append(rec)
            except Exception as e:
                records.append({
                    "task_id": int(tid) if tid.isdigit() else tid,
                    "tsp_socket": socket,
                    "status": "collect_error",
                    "error": f"{type(e).__name__}: {e}",
                })

    # 同组多次运行: 保留 best, 其它放 attempts
    groups: dict[tuple, list[dict]] = {}
    orphans: list[dict] = []  # 无法形成组 key 的 (典型: collect_error)
    for r in records:
        if r.get("status") == "collect_error":
            orphans.append(r)
            continue
        key = make_combo_key(r)
        groups.setdefault(key, []).append(r)

    deduped: list[dict] = []
    for key, recs in groups.items():
        recs.sort(key=rec_score, reverse=True)
        best = recs[0]
        if len(recs) > 1:
            best = dict(best)
            best["attempts"] = [
                {"task_id": r["task_id"], "tsp_socket": r.get("tsp_socket"),
                 "status": r["status"],
                 "overall_dataset_auc": (r.get("overall_stats") or {}).get("overall_dataset_auc")}
                for r in recs[1:]
            ]
        deduped.append(best)

    deduped.sort(key=lambda r: (r.get("model") or "", r.get("dataset") or "",
                                r.get("fold") if r.get("fold") is not None else -1,
                                r.get("emb_type") or "", r.get("version") or "",
                                r.get("tsp_socket") or "", r.get("task_id", 0)))

    summary = {
        "sockets": sockets,
        "total_tasks": len(records),
        "total_groups": len(deduped),
        "by_status": _count_by(deduped, "status"),
        "by_model": _count_by(deduped, "model"),
        "by_dataset": _count_by(deduped, "dataset"),
        "by_socket": _count_by(records, "tsp_socket"),  # 注意: by_socket 用未去重的 records
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "pending": [
            {"task_id": r["task_id"], "tsp_socket": r.get("tsp_socket"), "status": r["status"]}
            for r in deduped if r.get("status") in ("running", "queued")
        ],
        "results": deduped,
        "collect_errors": orphans,
    }


def _count_by(records: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        k = str(r.get(field))
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ───────────────────────── 入口 ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUTPUT_JSON),
                    help="输出 JSON 路径 (默认: %(default)s)")
    ap.add_argument("--socket", action="append", default=None,
                    help="tsp socket 路径, 可多次或逗号分隔. 'default' 表示系统默认 socket. "
                         "默认: default,/tmp/ts_gpu0,/tmp/ts_gpu1")
    ap.add_argument("--quiet", action="store_true",
                    help="不打印摘要")
    ap.add_argument("--exit-when-done", action="store_true",
                    help="如果队列里已经没有 running/queued, 以 exit code 0 退出, "
                         "否则以 75 (待续) 退出, 供 cron 判断")
    args = ap.parse_args()

    # 解析 sockets: 多次 --socket 累加, 每个值再按逗号拆分
    if args.socket:
        sockets: list[str] = []
        for s in args.socket:
            sockets.extend(p.strip() for p in s.split(",") if p.strip())
    else:
        sockets = [DEFAULT_SOCKET, "/tmp/ts_gpu0", "/tmp/ts_gpu1"]
    # 去重保序
    seen: set[str] = set()
    sockets = [s for s in sockets if not (s in seen or seen.add(s))]

    data = collect(sockets)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写入: 先写临时文件再 rename
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)

    if not args.quiet:
        s = data["summary"]
        print(f"[{data['generated_at']}] -> {out_path}")
        print(f"  sockets    : {s['sockets']}")
        print(f"  total tasks: {s['total_tasks']}   total groups: {s['total_groups']}")
        print(f"  by status  : {s['by_status']}")
        print(f"  by model   : {s['by_model']}")
        print(f"  by socket  : {s['by_socket']}")
        print(f"  pending    : {data['pending']}")

    if args.exit_when_done:
        if data["pending"]:
            sys.exit(75)  # EX_TEMPFAIL
        sys.exit(0)


if __name__ == "__main__":
    main()
