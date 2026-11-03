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
