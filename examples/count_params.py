"""Count trainable parameters of AKT and qikt_mamba(FairKT) under capacity-matching grids.

Usage:
    cd examples && python count_params.py --dataset_name assist2009
    cd examples && python count_params.py --dataset_name assist2009 --csv params_assist2009.csv
"""
import argparse
import json
import os
import sys
import csv
import copy

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pykt.models import init_model


AKT_GRID = [
    {"tag": "AKT-default", "d_model": 256, "d_ff": 512,  "n_blocks": 4, "num_attn_heads": 8},
    {"tag": "AKT-M",       "d_model": 256, "d_ff": 1024, "n_blocks": 4, "num_attn_heads": 8},
    {"tag": "AKT-L1",      "d_model": 384, "d_ff": 768,  "n_blocks": 4, "num_attn_heads": 8},
    {"tag": "AKT-L2",      "d_model": 256, "d_ff": 512,  "n_blocks": 8, "num_attn_heads": 8},
    {"tag": "AKT-XL",      "d_model": 512, "d_ff": 1024, "n_blocks": 4, "num_attn_heads": 8},
]

FAIRKT_GRID = [
    {"tag": "FairKT-default(emb300_mlp2)", "emb_size": 300, "mlp_layer_num": 2, "num_attn_head": 1},
    {"tag": "FairKT-main(emb256_mlp1)",   "emb_size": 256, "mlp_layer_num": 1, "num_attn_head": 1},
    {"tag": "FairKT-main(emb256_mlp2)",   "emb_size": 256, "mlp_layer_num": 2, "num_attn_head": 1},
    {"tag": "FairKT-S(emb128_mlp1)",      "emb_size": 128, "mlp_layer_num": 1, "num_attn_head": 1},
    {"tag": "FairKT-S(emb128_mlp2)",      "emb_size": 128, "mlp_layer_num": 2, "num_attn_head": 1},
    {"tag": "FairKT-S(emb200_mlp1)",      "emb_size": 200, "mlp_layer_num": 1, "num_attn_head": 1},
    {"tag": "FairKT-S(emb200_mlp2)",      "emb_size": 200, "mlp_layer_num": 2, "num_attn_head": 1},
]


def build_akt_config(cfg, dropout=0.2):
    return {
        "d_model": cfg["d_model"],
        "d_ff": cfg["d_ff"],
        "n_blocks": cfg["n_blocks"],
        "num_attn_heads": cfg["num_attn_heads"],
        "dropout": dropout,
    }


def build_fairkt_config(cfg, dropout=0.4):
    other_config = {
        "output_mode": "an",
        "loss_q_all_lambda": 0,
        "loss_c_all_lambda": 0,
        "loss_q_next_lambda": 0,
        "loss_c_next_lambda": 0,
        "output_q_all_lambda": 1,
        "output_c_all_lambda": 1,
        "output_q_next_lambda": 0,
        "output_c_next_lambda": 1,
    }
    return {
        "emb_size": cfg["emb_size"],
        "mlp_layer_num": cfg["mlp_layer_num"],
        "num_attn_head": cfg["num_attn_head"],
        "dropout": dropout,
        "version": "an",
        "other_config": other_config,
    }


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--data_config", type=str, default="../configs/data_config.json")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config_all = json.load(f)
    if args.dataset_name not in data_config_all:
        raise SystemExit(f"dataset {args.dataset_name} not in {args.data_config}")
    data_config = data_config_all[args.dataset_name]

    rows = []

    for cfg in AKT_GRID:
        mc = build_akt_config(cfg)
        try:
            model = init_model("akt", copy.deepcopy(mc), data_config, emb_type="qid")
            tr, tot = count_params(model)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[ERR] {cfg['tag']}: {e}")
            continue
        row = {"model": "akt", "tag": cfg["tag"], "trainable": tr, "total": tot, **mc}
        rows.append(row)
        print(f"{cfg['tag']:<16} trainable={tr:>12,}  total={tot:>12,}  config={mc}")

    for cfg in FAIRKT_GRID:
        mc = build_fairkt_config(cfg)
        try:
            model = init_model("qikt_mamba", copy.deepcopy(mc), data_config, emb_type="attn")
            tr, tot = count_params(model)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[ERR] {cfg['tag']}: {e}")
            continue
        row = {"model": "qikt_mamba", "tag": cfg["tag"], "trainable": tr, "total": tot,
               "emb_size": mc["emb_size"], "mlp_layer_num": mc["mlp_layer_num"],
               "num_attn_head": mc["num_attn_head"]}
        rows.append(row)
        print(f"{cfg['tag']:<16} trainable={tr:>12,}  total={tot:>12,}  "
              f"emb_size={mc['emb_size']} mlp_layer_num={mc['mlp_layer_num']} num_attn_head={mc['num_attn_head']}")

    if args.csv:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
