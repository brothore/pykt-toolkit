# ASSIST 联合监督：五轮迭代记录

独立模型 `dkt_joint_assist`，assist2009 / fold0 / seed42 / 最多20 epoch，标准验证AUC选模，主指标 `windowauclate_mean`。

选定联合版 `r5_ablation_b`（有效任务 `action, time, attempt, hint, distillation`），主指标 **0.754702007**。相对原 DKT +0.003871801，学生配对自举95%区间 [+0.002617888, +0.005073551]；相对此前验证最优单方案 +0.000488921，区间 [-0.000425544, +0.001464477]。区间包含0，尚未证实比此前单方案更有效。

完整任务组合中验证最高的是 `r2_half`，其主指标 0.754849181。在原基线、此前单方案和所有联合轮次之间，标准验证 AUC 最高版本为 `r5_ablation_b`；主测试值不用于改选。

| 版本 | 保留 / 停止 epoch | 验证 AUC | windowauclate_mean | 状态 |
|---|---:|---:|---:|---|
| 原始 DKT（复用） | 4 / 14 | 0.830839499 | 0.750830206 | 复用 |
| 此前验证最优单方案（复用） | 6 / 16 | 0.833164464 | 0.754213086 | 原单方案参照 |
| R1 完整组合／原权重 | 6 / 16 | 0.833349311 | 0.755671309 | 未选定 |
| R2 完整组合／附加权重减半 | 7 / 17 | 0.833523172 | 0.754849181 | 未选定 |
| R3 完整组合／再调权重 | 4 / 14 | 0.832531291 | 0.753860009 | 未选定 |
| R4 去掉首步行为 | 7 / 17 | 0.833414500 | 0.755106565 | 未选定 |
| R5 去掉迁移 | 6 / 16 | 0.833729052 | 0.754702007 | 验证选定联合版 |


- `r1_full`：All dataset-specific objectives combined at their previous validation-selected single-task coefficients.
  权重：`{"action": 0.1, "time": 0.03, "attempt": 0.1, "hint": 0.1, "distillation": 0.2, "transfer": 0.1}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/assist2009/r1_full)。
- `r2_half`：Halve additional objectives while preserving the strongest single-task anchor; test auxiliary overlap and gradient pressure.
  权重：`{"action": 0.05, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0.05}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/assist2009/r2_half)。
- `r3_scale`：Prewritten validation rule: if half extras beats full, reduce extras to .25 of original, otherwise increase to1.5; chosen extra scale=0.25.
  权重：`{"action": 0.025, "time": 0.0075, "attempt": 0.025, "hint": 0.025, "distillation": 0.2, "transfer": 0.025}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/assist2009/r3_scale)。
- `r4_ablation_a`：Matched ablation of action from validation-best complete combination r2_half; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"action": 0, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0.05}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/assist2009/r4_ablation_a)。
- `r5_ablation_b`：Matched ablation of transfer from validation-best complete combination r2_half; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"action": 0.05, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/assist2009/r5_ablation_b)。

- [完整双数据集报告](../DKT_同数据集联合监督_双数据集实验.md)
- [结果JSON](../../logs/joint_iterations_20261029/results.json)
- [逐轮协议](../../logs/joint_iterations_20261029/protocol.json)
- [冻结选模](../../logs/joint_iterations_20261029/selection_before_test.json)
