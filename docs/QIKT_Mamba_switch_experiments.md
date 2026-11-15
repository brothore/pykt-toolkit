# QIKT-Mamba 切换实验

本目录对应的实现包含可独立开关的历史反馈 dropout、题目残差、独立 Mamba 分支、因果知识点记忆、片段排序损失和融合残差。所有开关默认关闭；冻结主干的残差微调支持从基线权重初始化和 KL 约束。

## 环境与训练

使用 Python 3.12、PyTorch 2.8 CUDA 环境，安装项目依赖后运行：

```bash
bash scripts/quick_run/install_mamba.sh
python scripts/qikt_switch/check_cuda_environment.py
python scripts/qikt_switch/run_neural.py --dataset nips_task34 --fold 0 --variant baseline
```

安装脚本优先使用 `packages/` 下匹配 ABI 的 wheel，也可通过 `PYKT_WHEEL_DIR` 指定本地目录。正式训练要求 CUDA 和 Mamba，不自动降级到 CPU。训练预设保存在 `configs/qikt_switch/`，支持 `nips_task34` 和 `peiyou`；其他方案可通过 `run_neural.py --help` 查看。数据应按项目原有方式准备，输出保存在本地 `logs/qikt_switch_20261103/`，已有训练记录不会自动覆盖或重启。

训练入口保留原来的验证集选模和早停逻辑，关闭训练结束后的自动测试推理。训练窗口重叠只作用于训练折，检查目标集合一致性；验证折不变。冻结残差方案要求对应数据集的基线已完成并导出 `round_summary.json`。

## 评估与检查

`evaluate_validation.py` 导出原始验证片段；`evaluate_window.py` 使用初始窗口和滚动窗口逐题预测。通过 `--source validation` 做方案选择，冻结方案后才使用 `--source test`。预测输出包含对齐信息，`compare_validation.py` 检查标签及题目对应关系，汇总逐序列 AUC、总体 AUC、标准差（ddof=0）、极差和退化数量。混合预测使用固定的全局系数，不能根据测试标签或 UID 调整。

正式测试的主要指标是 `all_results.json` 中的 `windowauclate_mean`。验证窗口的 `validation_window_auc` 仅用于验证集比较，不能代替主要指标。报告需注明数据集、模型、fold、seed、完成 epoch 和选中 epoch。总体 AUC 提升不足以证明每条序列均未退化。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_qikt_switch_*.py
```

测试覆盖关闭开关时的基线一致性、因果性、冻结主干、窗口对齐、混合预测和指标比较。`tests/fixtures/qikt_mamba_reference.py` 是提交 `14c32de` 的 GRU 路径回归参考，不含数据或权重。

实验数据、权重、日志、结果表及本地进度报告不随代码提交。`audit_inputs.py` 和 `probe_round*.py` 是分析用户本地交接材料的辅助工具，另需本地交接包；常规训练和单模型评估不依赖该交接包。串行控制脚本需要已有实验记录，运行前应检查对应输入与阶段，避免与正在运行的训练争用 GPU。

确认流程先在 nips 上完成候选冻结与正式测试。只有主要指标提升、orirow 重采样增量区间下界大于零且低/高基线表现组平均 AUC 不下降的固定方案，才进入 peiyou；不满足时记录结论并跳过昂贵的迁移训练。标准差、极差与逐序列退化另行报告，不能把迁移筛选通过当作目标已达成。测试结果只控制是否投入迁移计算，不用于修改方案或系数。

多实例应使用独立输出目录，并综合 GPU 利用率、显存、CPU、内存和存储吞吐判断是否并行。显存空闲但 GPU 算力已饱和时仍采用串行。控制进程支持接管已有窗口推理，不重启正在运行的模型进程。

验证集逐序列诊断可与 GPU 任务并行执行：

```bash
TS_SOCKET=/tmp/pykt-qikt-cpu.sock TS_SLOTS=4 \
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
tsp -N 4 -L qikt-validation-diagnostics python scripts/qikt_switch/analyze_validation_trials.py \
  --neural-root logs/qikt_switch_20261103/neural \
  --output-dir logs/qikt_switch_20261103/validation_diagnostics --workers 4
```

该工具只读已完成的窗口验证配对表，分解 `Var(candidate)-Var(baseline) = 2Cov(baseline, delta)+Var(delta)`，报告低/高基线表现组及切换率分组，帮助识别标准差缩小是否伴随强序列退化。它不选择模型、不读取测试数据。GPU 队列与 CPU 队列的资源预算需合并考虑。

分析支持 SIGTERM/SIGINT 安全停止，在途分方案计算结束后保存检查点并返回 130；相同参数增加 `--resume` 从已完成方案恢复。输入或分析源码身份变化时拒绝复用旧检查点，应使用新目录。输出目录用文件锁防止重复执行，JSON 原子写入；`state.json` 保存配置、源码/输入/产物校验值、环境、PID 和恢复历史。调度时另保存 `queue_submission.json`，记录 `tsp` job ID、socket、父流程和资源预算。这是分方案恢复，不是训练状态恢复；现有训练流程仍按原方式完成。

长窗口评估可启用 `evaluate_window.py --checkpoint-batches 64 --resume`。已提交的预测块与进度原子保存；SIGTERM/SIGINT 在当前 batch 结束后保存并退出 130，恢复时检查数据、模型、配置、源码和预测块校验值，跳过已完成窗口。异常掉电最多重算最后一段未提交 batch；batch 大小固定，以保持恢复前后的批次边界。此功能默认关闭，已有调用方式保持兼容。

`queued_task.py --plan PATH --resume` 执行固定的单命令任务计划，记录进程与退出状态，拒绝输入/源码变化及并发重复执行；完成任务再次提交时先检查产物，不重复计算。通过对应队列的 `tsp -k JOB_ID` 请求停止，再次提交相同命令可恢复。恢复粒度由任务本身决定。`report_component_trials.py` 在每个前置评估完成后立刻进行 CPU 对比，与后续 GPU 评估重叠；失败或停止的前置任务会触发人工复核，不会被视为有效结果。

提前排队只针对已确定的有限任务；冻结的参数与科学判断条件仍然有效。队列和完整任务清单保存在本地产物中，Codex 离线不会阻止已排任务衔接。

新种子训练可使用 `run_neural.py --seed 43 --output-root PATH --resumable --resume`。新目录首次执行不需要既有检查点；已有训练记录必须存在真实的 `training_resume.ckpt`，否则拒绝把从头重跑当作恢复。恢复器只支持当前 QIKT 协议：Adam、无梯度累积/ALSO/AMP/调度器、DataLoader workers=0。它保留原生 `auc > best + .001` 的选模更新和 10 epoch 早停，在 epoch 边界保存模型、优化器、随机数及 loader/sampler generator、训练进度和选模状态。SIGTERM/SIGINT 等当前 epoch 完成后退出 130；硬中断最多重做当前未提交 epoch。原有训练默认入口不改变。

`replicate_trial.py` 依次进行上述训练、原始验证导出和可恢复的验证窗口评估，只读训练/验证数据；非基线方案先要求同 seed 基线成功。`report_replications.py` 随每个任务完成保存复验记录，并用原已冻结的系数比较两个混合方案，不重新拟合参数。多 seed 一致性仍不能证明每条序列均受保护。
