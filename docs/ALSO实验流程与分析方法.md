# ALSO 实验流程与分析方法

## 1. 实验背景

ALSO（Adaptive Loss Scaling Optimizer）是一种面向分布鲁棒优化的训练策略，通过对不同样本组（学生/序列）赋予自适应权重，旨在提升模型在子群体上的公平性。

在知识追踪（Knowledge Tracing）场景中，我们使用 **pyKT 框架** + **DKT 模型** + **assist2009 数据集** 进行实验，目标是观察 ALSO 能否提升学生级别的 AUC 均值（`student_stats_mean`），同时降低学生间 AUC 的标准差（`student_stats_std`）和极差（`student_stats_range`）。

### 实验配置矩阵

| 配置名 | 优化器 | 分组方式 | 说明 |
|---|---|---|---|
| `baseline` | 标准 Adam | 无 | 对照组 |
| `student_id` | ALSO | 按学生 UID 分组 | 每个学生一个组 |
| `seq` | ALSO | 按 batch 内序列分组 | 每条序列一个组 |

---

## 2. 环境准备

### 2.1 激活环境

```bash
# 确保在仓库根目录
cd /root/autodl-tmp/pykt-toolkit

# 确认 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 2.2 超参数配置

所有实验使用统一的 DKT 超参：

| 参数 | 值 | 说明 |
|---|---|---|
| `--dataset_name` | `assist2009` | 数据集 |
| `--model_name` | `dkt` | 模型架构 |
| `--emb_type` | `qid` | 嵌入类型 |
| `--seed` | `42` | 随机种子 |
| `--fold` | `0` | 交叉验证折 |
| `--dropout` | `0.2` | Dropout 比例 |
| `--emb_size` | `200` | 嵌入维度 |
| `--learning_rate` | `1e-3` | 学习率 |
| `--batch_size` | `32` | 批次大小 |
| `--num_epochs` | `100` | 最大训练轮数 |
| `--use_wandb` | `0` | 禁用 WandB |
| `--add_uuid` | `0` | 不添加 UUID |
| `--use_trained` | `0` | 从头训练 |

ALSO 专属参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `--use_also` | `1` | 启用 ALSO 优化器 |
| `--also_grouping_mode` | `student_id` 或 `seq` | 分组模式 |

---

## 3. 启动实验

### 3.1 日志命名规范

为避免日志覆盖，使用**唯一日志路径** + **时间戳日志目录**：

```bash
# 日志目录命名：logs/{实验标识}_{轮次}/
# 示例：
LOGDIR="logs/dkt_round2"
mkdir -p "$LOGDIR"
```

### 3.2 后台启动命令

三组实验可同时并行启动（不同 GPU 或共享 GPU）：

```bash
cd /root/autodl-tmp/pykt-toolkit
LOGDIR="logs/dkt_round2"
mkdir -p "$LOGDIR"

# ==================== Baseline ====================
nohup python -m examples.wandb_dkt_train \
    --dataset_name assist2009 \
    --model_name dkt \
    --emb_type qid \
    --save_dir saved_model/dkt_round2_baseline \
    --seed 42 --fold 0 \
    --dropout 0.2 --emb_size 200 \
    --learning_rate 1e-3 --batch_size 32 \
    --num_epochs 100 \
    --use_wandb 0 --add_uuid 0 --use_trained 0 \
    --use_also 0 \
    > "$LOGDIR/baseline.log" 2>&1 &

# ==================== Student ID ====================
nohup python -m examples.wandb_dkt_train \
    --dataset_name assist2009 \
    --model_name dkt \
    --emb_type qid \
    --save_dir saved_model/dkt_round2_student_id \
    --seed 42 --fold 0 \
    --dropout 0.2 --emb_size 200 \
    --learning_rate 1e-3 --batch_size 32 \
    --num_epochs 100 \
    --use_wandb 0 --add_uuid 0 --use_trained 0 \
    --use_also 1 --also_grouping_mode student_id \
    > "$LOGDIR/student_id.log" 2>&1 &

# ==================== Seq ====================
nohup python -m examples.wandb_dkt_train \
    --dataset_name assist2009 \
    --model_name dkt \
    --emb_type qid \
    --save_dir saved_model/dkt_round2_seq \
    --seed 42 --fold 0 \
    --dropout 0.2 --emb_size 200 \
    --learning_rate 1e-3 --batch_size 32 \
    --num_epochs 100 \
    --use_wandb 0 --add_uuid 0 --use_trained 0 \
    --use_also 1 --also_grouping_mode seq \
    > "$LOGDIR/seq.log" 2>&1 &

echo "三组实验已启动，查看进度: tail -f $LOGDIR/*.log"
```

### 3.3 监控运行状态

```bash
# 查看进程
ps -ef | grep 'python -m examples.wandb_dkt_train'

# 实时查看日志
tail -f logs/dkt_round2/baseline.log
tail -f logs/dkt_round2/student_id.log
tail -f logs/dkt_round2/seq.log

# 查看各实验 epoch 进度
for f in baseline student_id seq; do
    echo -n "$f: "; grep -c 'Epoch:' logs/dkt_round2/$f.log
done
```

---

## 4. 结果分析

### 4.1 分析结果位置

训练完成后，每个实验目录下会生成以下关键文件：

| 文件 | 路径 | 说明 |
|---|---|---|
| 训练日志 | `logs/dkt_round2/baseline.log` | 完整训练输出 |
| 模型 checkpoint | `saved_model/dkt_round2_baseline/.../qid_model.ckpt` | 最佳模型权重 |
| 预测结果（JSON） | `.../overall_stats_output.json` | 公平性评估指标 |
| 预测结果（CSV） | `.../student_auc_output.csv` | 每个学生的 AUC |

### 4.2 关键指标速查

JSON 文件 (`overall_stats_output.json`) 中重要指标：

```bash
cat saved_model/dkt_round2_baseline/*/overall_stats_output.json
```

| JSON Key | 含义 | 期望方向 |
|---|---|---|
| `overall_dataset_auc` | 整体测试集 AUC | ↑ 越高越好 |
| `student_stats_mean` | 学生级别 AUC 均值 | ↑ 越高越好 |
| `student_stats_std` | 学生级别 AUC 标准差 | ↓ 越低越公平 |
| `student_stats_range` | 学生级别 AUC 极差 (max-min) | ↓ 越低越公平 |
| `student_stats_iqr` | 学生 AUC 四分位距 | ↓ 越低越公平 |
| `student_stats_gini_coefficient` | 学生 AUC 基尼系数 | ↓ 越低越公平 |
| `student_stats_eawi_alpha_10` | EAWI 公平性指数 (α=10) | ↑ 越高越好 |
| `student_stats_eawi_alpha_20` | EAWI 公平性指数 (α=20) | ↑ 越高越好 |
| `student_stats_eawi_alpha_30` | EAWI 公平性指数 (α=30) | ↑ 越高越好 |

### 4.3 从训练日志提取指标

```bash
# 提取最佳验证 AUC
grep 'Epoch.*best auc' logs/dkt_round2/baseline.log | tail -5

# 提取最终统计行
grep 'fold.*modelname' logs/dkt_round2/baseline.log

# 提取 early stopping 信息
grep 'Early stopping\|Training Finished' logs/dkt_round2/*.log
```

### 4.4 批量汇总所有指标

```bash
# 读取所有实验的总结文件
echo "=== Baseline ===" && cat saved_model/dkt_round2_baseline/*/overall_stats_output.json
echo "=== Student ID ===" && cat saved_model/dkt_round2_student_id/*/overall_stats_output.json
echo "=== Seq ===" && cat saved_model/dkt_round2_seq/*/overall_stats_output.json
```

---

## 5. 第一轮实验结果（参考）

| 指标 | Baseline | Student ID | Seq |
|---|---|---|---|
| Val AUC | 0.8309 | 0.8141 | 0.8085 |
| Overall Test AUC | 0.7184 | 0.6868 | 0.6816 |
| **AUC Mean** ↑ | **0.5803** | 0.5695 | 0.5676 |
| **AUC Std** ↓ | 0.2010 | 0.2016 | **0.1976** |
| **AUC Range** ↓ | 1.0 | 1.0 | 1.0 |
| **AUC IQR** ↓ | 0.2091 | 0.1915 | **0.1853** |
| **Gini** ↓ | **0.1876** | 0.1912 | 0.1882 |

### 初步结论

- Baseline 在整体 AUC 和均值上最优
- Seq 模式在 Std 和 IQR 上略有改善（标准差降低 ~1.7%），但以牺牲整体 AUC 为代价
- 需要调整 ALSO 超参（`pi_lr`、`pi_decay`、`alpha`）继续探索

---

## 6. 超参调优指南

需要调整 ALSO 的关键参数（在 `pykt/models/train_model.py` 的 `_create_also_optimizer` 中配置）：

| 参数 | 默认值 | 建议调参范围 | 说明 |
|---|---|---|---|
| `pi_lr` | 1e-3 | 1e-4 ~ 1e-2 | π 更新学习率，降低可减少噪声 |
| `pi_decay` | 1e-2 | 1e-3 ~ 1e-1 | π 正则化强度，增大可让分布更均匀 |
| `alpha` | 1.0 | 0.5 ~ 2.0 | Optimistic mode 负动量系数 |
| `loss_scale` | auto (n_groups/bs) | 0.1 ~ 10.0 | 手动设定 loss 缩放 |

### 调参实验记录模板

| 轮次 | pi_lr | pi_decay | AUC Mean | Std | IQR | 备注 |
|---|---|---|---|---|---|---|
| R1 | 1e-3 | 1e-2 | 0.5695 | 0.2016 | 0.1915 | 默认值 |
| R2 | 5e-4 | 5e-2 | ? | ? | ? | 待测试 |

---

## 7. 快速启动脚本模板

将以下内容保存为 `run_experiments.sh`：

```bash
#!/bin/bash
set -e
cd /root/autodl-tmp/pykt-toolkit

ROUND=${1:-"test"}                    # 实验轮次，如 round2
PI_LR=${2:-"1e-3"}                    # ALSO pi_lr
PI_DECAY=${3:-"1e-2"}                 # ALSO pi_decay

LOGDIR="logs/dkt_${ROUND}"
mkdir -p "$LOGDIR"

COMMON="--dataset_name assist2009 --model_name dkt --emb_type qid --seed 42 --fold 0 --dropout 0.2 --emb_size 200 --learning_rate 1e-3 --batch_size 32 --num_epochs 100 --use_wandb 0 --add_uuid 0 --use_trained 0"

echo "=== 启动实验轮次: $ROUND ==="
echo "日志目录: $LOGDIR"
echo "pi_lr=$PI_LR pi_decay=$PI_DECAY"

# Baseline
nohup python -m examples.wandb_dkt_train $COMMON --use_also 0 --save_dir "saved_model/dkt_${ROUND}_baseline" > "$LOGDIR/baseline.log" 2>&1 &

# Student ID
nohup python -m examples.wandb_dkt_train $COMMON --use_also 1 --also_grouping_mode student_id --save_dir "saved_model/dkt_${ROUND}_student_id" > "$LOGDIR/student_id.log" 2>&1 &

# Seq
nohup python -m examples.wandb_dkt_train $COMMON --use_also 1 --also_grouping_mode seq --save_dir "saved_model/dkt_${ROUND}_seq" > "$LOGDIR/seq.log" 2>&1 &

echo "已启动，监控: tail -f $LOGDIR/*.log"
```

使用方式：

```bash
# 默认参数
bash run_experiments.sh round2

# 自定义 ALSO 超参
bash run_experiments.sh round2 5e-4 5e-2
```
