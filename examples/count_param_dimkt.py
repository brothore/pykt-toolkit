"""
绕过 dataloader, 只构造 DIMKT 模型并打印 [PARAM_COUNT].
DIMKT 在 init_dataset4train 里依赖难度分级预处理产物 (train_valid_sequences_quelevel_diff_*.csv),
没这些文件时正常 train 入口会在 dataloader 阶段挂掉, 拿不到参数量.
本脚本仅做 init_model + sum(p.numel()), 完全不读数据, 用于补全参数量统计.
"""
import argparse
import copy
import json
import os

import torch

from pykt.models import init_model

device = "cpu" if not torch.cuda.is_available() else "cuda"


def main(params):
    dataset_name = params["dataset_name"]
    model_name = "dimkt"
    emb_type = params["emb_type"]

    with open("../configs/data_config.json") as fin:
        data_config = copy.deepcopy(json.load(fin))[dataset_name]

    model_config = {
        "dropout": params["dropout"],
        "emb_size": params["emb_size"],
        "batch_size": params["batch_size"],
        "num_steps": params["num_steps"],
        "difficult_levels": params["difficult_levels"],
    }
    model = init_model(model_name, model_config, data_config, emb_type)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[PARAM_COUNT] model={model_name} emb_type={emb_type} "
        f"dataset={dataset_name} trainable={n_trainable:,} total={n_total:,}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--difficult_levels", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=199)
    args = parser.parse_args()
    main(vars(args))
