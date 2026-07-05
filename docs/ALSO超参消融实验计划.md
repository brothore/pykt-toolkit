# ALSO 超参消融实验计划

## 1. 问题诊断

### 第一轮 Student ID 结果 vs Baseline

| 指标 | Baseline | Student ID (R1) | 差距 |
|---|---|---|---|
| Val AUC | 0.8309 | 0.8141 | -2.0% |
| Overall Test AUC | 0.7184 | 0.6868 | -4.4% |
| AUC Mean | 0.5803 | 0.5695 | -1.9% |
| AUC Std | 0.2010 | 0.2016 | +0.3% (变差) |

### 根因分析

问题出在 `loss_scale = n_groups / batch_size` 的计算上：

```
n_groups = data_config["students_num_train"] = 3082
batch_size = 32
loss_scale = 3082 / 32 ≈ 96.3
```

这个巨大的缩放因子导致了三个连锁问题：

#### 问题 1：模型梯度被放大 ~96 倍
```python
# ALSO 内部：closure 中 losses = losses * scale
# scale ≈ 96.3，导致 model.backward() 梯度放大 96 倍
```
虽然 ALSO 内部有独立 Adam，但梯度过大仍会导致训练不稳定、收敛过快、泛化差。

#### 问题 2：π 更新极度稀疏且噪声大
```
π 向量维度: 3082
每 batch 活跃学生: ~30（仅占 1%）
活跃条目获得巨大梯度（loss * scale ≈ 96x）
非活跃 3052 个条目仅靠 pi_decay 拉向均匀分布
→ π 分布振荡、不稳定
```

#### 问题 3：π 正则化信号太弱
```
pi_lr = 1e-3（过大，配合 96x 梯度 → 步子太大）
pi_decay = 1e-2（过小，无法有效平滑稀疏更新）
```

### 解决方案

**核心思路**：缩小 `n_groups`，让 `loss_scale` 回归合理范围（1~10），同时让 π 更新更密集。

---

## 2. 新增 CLI 参数

在开始消融前，需要在 `examples/wandb_dkt_train.py` 中添加三个新参数：

```python
parser.add_argument("--also_n_groups", type=int, default=None, help="手动指定 ALSO 分组数")
parser.add_argument("--also_pi_lr", type=float, default=1e-3, help="π 学习率")
parser.add_argument("--also_pi_decay", type=float, default=1e-2, help="π 正则化系数")
```

并在 `examples/wandb_train.py` 的 `also_config` 中确保这些参数被读取（已有 `params.get("also_pi_lr", 1e-3)` 等，只需确保 CLI 参数能流到 `params` 字典）。

---

## 3. 消融实验设计

### 实验矩阵

所有实验使用 **Student ID 模式**，保持其他超参与第一轮一致。

#### Phase A：n_groups 消融（最关键的变量）

`n_groups` 直接控制 `loss_scale = n_groups/bs`，对训练稳定性有决定性影响。

| 实验 ID | n_groups | loss_scale | pi_lr | pi_decay | 假设 |
|---|---|---|---|---|---|
| **R1** (已跑) | 3082 | 96.3 | 1e-3 | 1e-2 | 对照组（过大） |
| **A1** | 50 | 1.56 | 1e-3 | 1e-2 | dense π, 低 scale |
| **A2** | 100 | 3.13 | 1e-3 | 1e-2 | 中等分组 |
| **A3** | 200 | 6.25 | 1e-3 | 1e-2 | 中等偏大 |
| **A4** | 500 | 15.6 | 1e-3 | 1e-2 | 验证大分组边界 |

**预期**：A1-A3 明显优于 R1，A4 开始退化。

#### Phase B：pi_lr 消融（用 A 中最佳 n_groups）

| 实验 ID | n_groups | pi_lr | pi_decay | 假设 |
|---|---|---|---|---|
| **B1** | best_A | 5e-4 | 1e-2 | 降低 π 更新步长 |
| **B2** | best_A | 2e-4 | 1e-2 | 更保守的 π 更新 |
| **B3** | best_A | 5e-3 | 1e-2 | 激进更新（验证是否过冲） |

#### Phase C：pi_decay 消融（用 A+B 最佳组合）

| 实验 ID | n_groups | pi_lr | pi_decay | 假设 |
|---|---|---|---|---|
| **C1** | best | best_B | 5e-2 | 更强正则化 |
| **C2** | best | best_B | 1e-1 | 极强正则化（趋近均匀） |

### Phase D：最佳组合汇总

| 实验 ID | n_groups | pi_lr | pi_decay | loss_scale |
|---|---|---|---|---|
| **D** | best_A | best_B | best_C | - |

---

## 4. 日志命名规范

每轮实验使用独立目录，确保不覆盖：

```
logs/dkt_ablation/
├── a1_n50_pilr1e3_pid1e2.log      # A1
├── a2_n100_pilr1e3_pid1e2.log     # A2
├── a3_n200_pilr1e3_pid1e2.log     # A3
├── a4_n500_pilr1e3_pid1e2.log     # A4
├── b1_n{best}_pilr5e4_pid1e2.log  # B1
├── b2_n{best}_pilr2e4_pid1e2.log  # B2
├── b3_n{best}_pilr5e3_pid1e2.log  # B3
├── c1_n{best}_pilr{best}_pid5e2.log
├── c2_n{best}_pilr{best}_pid1e1.log
└── d_best.log                      # D
```

save_dir 对应命名：
```
saved_model/dkt_ablation_a1_n50/
saved_model/dkt_ablation_a2_n100/
...
```

---

## 5. 部分实施：快速启动 Phase A

```bash
cd /root/autodl-tmp/pykt-toolkit
LOGDIR="logs/dkt_ablation"
mkdir -p "$LOGDIR"

BASE="--dataset_name assist2009 --model_name dkt --emb_type qid --seed 42 --fold 0 --dropout 0.2 --emb_size 200 --learning_rate 1e-3 --batch_size 32 --num_epochs 100 --use_wandb 0 --add_uuid 0 --use_trained 0 --use_also 1 --also_grouping_mode student_id"

# A1: n_groups=50
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 50  --save_dir saved_model/dkt_ablation_a1_n50  > "$LOGDIR/a1_n50_pilr1e3_pid1e2.log"  2>&1 &

# A2: n_groups=100
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 100 --save_dir saved_model/dkt_ablation_a2_n100 > "$LOGDIR/a2_n100_pilr1e3_pid1e2.log" 2>&1 &

# A3: n_groups=200
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 200 --save_dir saved_model/dkt_ablation_a3_n200 > "$LOGDIR/a3_n200_pilr1e3_pid1e2.log" 2>&1 &

# A4: n_groups=500
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 500 --save_dir saved_model/dkt_ablation_a4_n500 > "$LOGDIR/a4_n500_pilr1e3_pid1e2.log" 2>&1 &
```

---

## 6. 预期结果与分析框架

### 成功标志

与 Baseline 相比，期望看到：
- `student_stats_mean`↑（学生平均 AUC 提升）
- `student_stats_std`↓（学生间差异缩小）
- `student_stats_range`↓（极差缩小）
- `overall_dataset_auc` 不能大幅低于 0.7184

### 分析模板

```bash
# 批量提取 Phase A 结果
for exp in a1_n50 a2_n100 a3_n200 a4_n500; do
    echo "=== $exp ==="
    cat saved_model/dkt_ablation_${exp}/*/overall_stats_output.json 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print(f\"AUC={d['overall_dataset_auc']:.4f} Mean={d['student_stats_mean']:.4f} Std={d['student_stats_std']:.4f}\")"
done
```

### 决策逻辑

```
Phase A 完成后：
  ├── 最佳 n_groups → 进入 Phase B
  └── 如果所有都差 → 降低 pi_lr 重跑 Phase A

Phase B 完成后：
  ├── 最佳 (n_groups, pi_lr) → 进入 Phase C
  └── Phase C 完成后 → 最佳三参数组合
```

---

## 7. 执行状态

> ✅ Phase A 完成 (2026-07-06)
> ⏳ Phase A2 (batch_size 消融) 待执行

---

## 8. Phase A 实验结果

| 实验 | n_groups | bs | loss_scale | Val AUC | Overall AUC | Mean ↑ | Std ↓ | IQR ↓ | Gini ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | - | 32 | - | **0.8309** | **0.7184** | **0.5803** | 0.2010 | 0.2091 | 0.1876 |
| R1 (student_id) | 3082 | 32 | 96.3 | 0.8141 | 0.6868 | 0.5695 | 0.2016 | 0.1915 | 0.1912 |
| A1 | 50 | 32 | 1.56 | 0.8085 | 0.6827 | 0.5671 | 0.1986 | 0.1868 | 0.1892 |
| A2 | 100 | 32 | 3.13 | 0.8087 | 0.6827 | 0.5662 | 0.2020 | 0.1877 | 0.1927 |
| A3 | 200 | 32 | 6.25 | 0.8100 | 0.6846 | 0.5663 | 0.1994 | 0.1874 | 0.1902 |
| A4 | 500 | 32 | 15.6 | 0.8101 | 0.6860 | 0.5668 | 0.1989 | **0.1837** | 0.1894 |
| **A5** | 3082 | **64** | 48.2 | **0.8188** | **0.6987** | **0.5750** | **0.1983** | 0.1923 | **0.1856** |

### 关键发现
- A5 (bs=64) 五项指标在 ALSO 中最佳，说明增大 batch_size 比缩 n_groups 更有效
- n_groups 从 50→500，指标差异不大，说明 loss_scale 在 1.5~15 区间内 ALSO 表现接近
- Baseline 仍全面碾压所有 ALSO 配置（差距约 2pp Overall AUC）

---

## 9. Phase A2: batch_size 消融

基于 Phase A 发现 bs 对 ALSO 影响显著，在 Phase B (pi_lr) 之前先做 batch_size 消融。

固定 n_groups=3082（原始学生数），变化 batch_size：

| 实验 | n_groups | bs | loss_scale | 假设 |
|---|---|---|---|---|
| A5 (已有) | 3082 | 64 | 48.2 | Phase A 最优 |
| A6 | 3082 | 16 | 192.6 | 极端 scale，预计最差 |
| A7 | 3082 | 128 | 24.1 | 更大 bs，scale 更低 |

增加一组 n_groups=500 配合大 bs（消融 n_groups×bs 交互）：

| 实验 | n_groups | bs | loss_scale | 假设 |
|---|---|---|---|---|
| A8 | 500 | 128 | 3.91 | 最佳组合候选 |

### 启动命令

```bash
cd /root/autodl-tmp/pykt-toolkit
LOGDIR="logs/dkt_ablation"
mkdir -p "$LOGDIR"

BASE="--dataset_name assist2009 --model_name dkt --emb_type qid --seed 42 --fold 0 --dropout 0.2 --emb_size 200 --learning_rate 1e-3 --num_epochs 100 --use_wandb 0 --add_uuid 0 --use_trained 0 --use_also 1 --also_grouping_mode student_id --also_n_groups 3082"

# A6: bs=16
nohup python -m examples.wandb_dkt_train $BASE --batch_size 16 --save_dir saved_model/dkt_abl_a6_n3082_bs16 > "$LOGDIR/a6_n3082_bs16.log" 2>&1 &

# A7: bs=128
nohup python -m examples.wandb_dkt_train $BASE --batch_size 128 --save_dir saved_model/dkt_abl_a7_n3082_bs128 > "$LOGDIR/a7_n3082_bs128.log" 2>&1 &

# A8: n=500, bs=128
nohup python -m examples.wandb_dkt_train $BASE --also_n_groups 500 --batch_size 128 --save_dir saved_model/dkt_abl_a8_n500_bs128 > "$LOGDIR/a8_n500_bs128.log" 2>&1 &

echo "Phase A2 launched: A6 A7 A8"
```
