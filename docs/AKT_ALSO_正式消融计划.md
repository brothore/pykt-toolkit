# AKT + ALSO 正式消融计划

## 固定设置

- 数据集：assist2009，fold=0，seed=42；
- 模型：AKT qid，`d_model=256`，模型学习率 `1e-4`；
- 分组：训练折学生的一对一映射（2465 组），不允许合并学生；
- 主指标：`student_stats_mean` 最大化、`student_stats_std` 与 `student_stats_range` 最小化；
- 约束指标：`overall_dataset_auc` 不出现明显退化；
- 每个实验使用独立 run ID、日志文件及 `save_dir`。

## 阶段安排

| 阶段 | 参数 | 配置 | 目的 |
|---|---|---|---|
| 0 | Baseline | bs=64 | 当前 v2 队列中运行，提供无 ALSO 对照。 |
| 1 | `pi_lr` | `3e-5, 1e-4, 3e-4, 1e-3`，bs=64 | 确定困难学生权重更新的合适速度。 |
| 2 | `batch_size` | `32, 64, 128`；每个 batch 有 Baseline 与 ALSO | 判断更稳定的学生损失估计是否改善公平性。 |
| 3 | `pi_decay` | `1e-3, 1e-2, 1e-1`，bs=64 | 控制权重分布偏离均匀先验的程度。 |
| 4 | `loss_scale` | 默认值的 `0.5x, 1x, 2x`，bs=64 | 检验论文 mini-batch 缩放在 KT 上的适配性。 |
| 5 | `alpha`、`mode` | `alpha=0/0.5/1.0`，以及 `descent-ascent` | 比较 optimistic 更新及其负动量。 |

阶段 1 的 `1e-4`、`3e-4` 与阶段 0 baseline 已由 `scripts/run_akt_student_also_ablation.sh` 排队。`scripts/run_akt_also_full_ablation.sh` 会等待该队列结束后继续其余配置。

## 并行策略

- `bs=64` 单任务约占 9.6 GiB 显存，可两项并行；
- `bs=32` 的 Baseline/ALSO 配对并行；
- `bs=128` 可能接近 17 GiB，单独执行；
- 每批完成后检查监控日志；若出现 OOM，下一批降为单任务执行，而不重跑已成功的任务。

## 启动与监控

```bash
setsid bash scripts/run_akt_also_full_ablation.sh </dev/null \
  > logs/akt_also_full_launcher.log 2>&1 &

setsid bash scripts/monitor_akt_also.sh </dev/null \
  > logs/akt_also_monitor_launcher.log 2>&1 &
```

调度器和监控器均自动生成时间戳文件名。监控器记录 GPU 显存、进程、最新 epoch、异常以及最终学生级指标。

训练后公平性预测会以项目根目录加入子进程 `PYTHONPATH` 的方式运行；预测失败只记录该配置失败，不会阻断后续消融。
