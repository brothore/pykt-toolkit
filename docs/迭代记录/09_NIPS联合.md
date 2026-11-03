# NIPS 联合监督：五轮迭代记录

独立模型 `dkt_joint_nips`，nips_task34 / fold0 / seed42 / 最多20 epoch，标准验证AUC选模，主指标 `windowauclate_mean`。

选定联合版 `r5_ablation_b`（有效任务 `confidence, option`），主指标 **0.767969306**。相对原 DKT +0.000526493，学生配对自举95%区间 [+0.000216960, +0.000860982]；相对此前验证最优单方案 -0.000253630，区间 [-0.000551941, +0.000060661]。区间包含0，尚未证实比此前单方案更有效。

完整任务组合中验证最高的是 `r3_scale`，其主指标 0.767922124。在原基线、此前单方案和所有联合轮次之间，标准验证 AUC 最高版本为 `best_single`；主测试值不用于改选。

| 版本 | 保留 / 停止 epoch | 验证 AUC | windowauclate_mean | 状态 |
|---|---:|---:|---:|---|
| 原始 DKT（复用） | 6 / 16 | 0.762496754 | 0.767442813 | 复用 |
| 此前验证最优单方案（复用） | 7 / 17 | 0.763264599 | 0.768222936 | 原单方案参照 |
| R1 完整组合／原权重 | 6 / 16 | 0.762267652 | 0.767448234 | 未选定 |
| R2 完整组合／附加权重减半 | 7 / 17 | 0.762841462 | 0.768088227 | 未选定 |
| R3 完整组合／再调权重 | 6 / 16 | 0.762853634 | 0.767922124 | 未选定 |
| R4 去掉信心 | 5 / 15 | 0.762773972 | 0.767695162 | 未选定 |
| R5 去掉测评画像 | 6 / 16 | 0.762908628 | 0.767969306 | 验证选定联合版 |


- `r1_full`：All dataset-specific objectives combined at their previous validation-selected single-task coefficients.
  权重：`{"confidence": 0.182, "option": 0.1, "profile": 0.03}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/nips_task34/r1_full)。
- `r2_half`：Halve additional objectives while preserving the strongest single-task anchor; test auxiliary overlap and gradient pressure.
  权重：`{"confidence": 0.091, "option": 0.1, "profile": 0.015}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/nips_task34/r2_half)。
- `r3_scale`：Prewritten validation rule: if half extras beats full, reduce extras to .25 of original, otherwise increase to1.5; chosen extra scale=0.25.
  权重：`{"confidence": 0.0455, "option": 0.1, "profile": 0.0075}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/nips_task34/r3_scale)。
- `r4_ablation_a`：Matched ablation of confidence from validation-best complete combination r3_scale; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"confidence": 0, "option": 0.1, "profile": 0.0075}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/nips_task34/r4_ablation_a)。
- `r5_ablation_b`：Matched ablation of profile from validation-best complete combination r3_scale; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"confidence": 0.0455, "option": 0.1, "profile": 0}`。模型与配置：[目录](../../examples/saved_model/joint_iterations_20261029/nips_task34/r5_ablation_b)。

- [完整双数据集报告](../DKT_同数据集联合监督_双数据集实验.md)
- [结果JSON](../../logs/joint_iterations_20261029/results.json)
- [逐轮协议](../../logs/joint_iterations_20261029/protocol.json)
- [冻结选模](../../logs/joint_iterations_20261029/selection_before_test.json)
