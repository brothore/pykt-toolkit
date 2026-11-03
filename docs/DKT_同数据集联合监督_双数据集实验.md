# 同数据集联合监督：ASSIST2009 与 NIPS task34

日期：2026-10-29。分支 `feat/student-state-simulator`。两个数据集分别完成五轮，共十轮新训练及主测试；旧七项共28轮的结果保持原样。

两个独立模型 `dkt_joint_assist`、`dkt_joint_nips`，预测骨干仍为原 DKT。联合监督不是模型集成，推理只运行一个 DKT。

## 1. 合并范围和训练约束

| 数据集 | 合并的原方案 | 训练参数 / 导出DKT参数 |
|---|---|---|
| ASSIST2009 | 首步行为（失败条件分类＋平衡）、作答过程（时间/尝试/提示）、丰富历史OOF蒸馏、同模板未来表现 | 476632 / 395523 |
| NIPS task34 | 有序信心、四选项CE、整块及25个父主题画像 | 504038 / 355857 |

每套模型的历史序列只经过一次DKT前向（LSTM、Dropout和答对概率），目标知识点特征在辅助头间共用；各辅助头共享同一状态，**主BCE只加一次**。使用原七项compute_loss的有效位置、标签验证和归一化实现，丢弃组件内重复算出的主损失，仅取辅助项。不同辅助项各自按原有效掩码/质量归一，再加权；不按覆盖率填补缺失标签，也不把某个任务的缺失当失败。原第一步/过程/选项/信心头为训练期模块，导出后推理不需要这些标签或头。

为保持单任务初始化可比，每个头复用原模型的同seed初始化，并在fork_rng中构造；不改变原骨干初始化后的随机数流。所有消融保留完整训练头集合，训练参数量不变，禁用任务无梯度。旧模型/旧Dataset未修改，通用路由只增加新名字。

当前LossView适配器限定于这七个现有组件，不承诺任意使用super()/property/classmethod的新组件可直接接入；组件修改后必须重新做loss/梯度等价和一次LSTM调用检查。

## 2. 固定预算、数据对齐与选模

两个数据集分别 fold0、seed42、每轮最多20 epoch、batch32、embedding/LSTM200、dropout0.2、Adam0.001，关闭ALSO/wandb/增强。保留标准验证AUC选模、`auc>max_auc+1e-3`更新和patience10。表中“保留epoch”不是所有epoch精确最大AUC。

ASSIST训练/验证3016/741序列、2465/617学生；NIPS为6043/1503序列、3148/787学生。JointDataset逐项核对uid、q/c/r/time、masks/smasks，拒绝字段覆盖与辅助字段冲突；首步/过程以及信心/选项的事件ID严格一致。迁移是当前已完成事件后的查询，首步/过程/信心/选项监督下一原始事件，测评画像查询块前固定状态，不混淆这些偏移。

过程归一化/类别计数、选项键/先验均仅训练折拟合，验证显式接收同一stats。ASSIST重用既有四折rich教师OOF缓存，没有重训或引入验证拟合；每个rich教师均排除自己的heldout折和外部fold0，验证teacher_mask全false。层级、QuizId块和未来配对仍遵循原实验的观测代理、完整块及删失约定，组合不会让这些信号变成新的心理真值。

与原始阶段的raw/processed训练文件、逐题窗口测试CSV哈希均匹配；真实cache中所有smasks均包含于有效masks，主BCE取交集未改变原任务位置。见 [数据审计](../logs/joint_iterations_20261029/data_audit.json)、[源码与数据沿用审计](../logs/joint_iterations_20261029/source_audit.json)、[掩码语义](../logs/joint_iterations_20261029/mask_semantics_audit.json)。

旧原DKT及此前**按标准验证AUC**最优单方案复用，不声称重新训练或测试它们。ASSIST单方案参照为丰富历史蒸馏（验证0.833164464），NIPS为具体选项（验证0.763264599；略高于画像0.763249718），不是按旧测试表挑参照。

## 3. 五轮闭环规则

完整组合沿用单方案选定系数。ASSIST保留蒸馏λ0.2，NIPS保留选项λ0.1作为锚定任务。第二轮其余系数减半；第三轮若第二轮标准验证AUC高于第一轮，进一步改为原系数0.25倍，否则改为1.5倍。此规则在训练前写入协议。随后从前三轮验证最高完整组合出发，分别去掉两个可能重叠目标，其余系数保持不变。两个数据集独立做决定，不能跨数据集直接比AUC。

ASSIST两项消融分别去掉首步行为和迁移；NIPS分别去掉信心和测评画像。所有五轮均为预先声明候选，按各自保留检查点验证AUC确定最佳联合版；同时保留完整组合最优版，避免把删掉任务的结果描述为全部信号同时有效。

两个数据集全部训练、全部选择于 `2026-10-29T09:53:44.033743+00:00` 同时冻结后才运行新测试。只运行原始逐题窗口late-fusion主指标：ASSIST batch128、NIPS batch512，均匹配旧基线；不宣称重跑全部四种测试。测试阶段最多两个只读评估进程共享A10，每进程CPU线程1，无训练并发。全部十个测试成功，无测试后追加调参。

## 4. ASSIST2009 五轮结果

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
  权重：`{"action": 0.1, "time": 0.03, "attempt": 0.1, "hint": 0.1, "distillation": 0.2, "transfer": 0.1}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/assist2009/r1_full)。
- `r2_half`：Halve additional objectives while preserving the strongest single-task anchor; test auxiliary overlap and gradient pressure.
  权重：`{"action": 0.05, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0.05}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/assist2009/r2_half)。
- `r3_scale`：Prewritten validation rule: if half extras beats full, reduce extras to .25 of original, otherwise increase to1.5; chosen extra scale=0.25.
  权重：`{"action": 0.025, "time": 0.0075, "attempt": 0.025, "hint": 0.025, "distillation": 0.2, "transfer": 0.025}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/assist2009/r3_scale)。
- `r4_ablation_a`：Matched ablation of action from validation-best complete combination r2_half; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"action": 0, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0.05}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/assist2009/r4_ablation_a)。
- `r5_ablation_b`：Matched ablation of transfer from validation-best complete combination r2_half; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"action": 0.05, "time": 0.015, "attempt": 0.05, "hint": 0.05, "distillation": 0.2, "transfer": 0}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/assist2009/r5_ablation_b)。

## 5. NIPS task34 五轮结果

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
  权重：`{"confidence": 0.182, "option": 0.1, "profile": 0.03}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/nips_task34/r1_full)。
- `r2_half`：Halve additional objectives while preserving the strongest single-task anchor; test auxiliary overlap and gradient pressure.
  权重：`{"confidence": 0.091, "option": 0.1, "profile": 0.015}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/nips_task34/r2_half)。
- `r3_scale`：Prewritten validation rule: if half extras beats full, reduce extras to .25 of original, otherwise increase to1.5; chosen extra scale=0.25.
  权重：`{"confidence": 0.0455, "option": 0.1, "profile": 0.0075}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/nips_task34/r3_scale)。
- `r4_ablation_a`：Matched ablation of confidence from validation-best complete combination r3_scale; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"confidence": 0, "option": 0.1, "profile": 0.0075}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/nips_task34/r4_ablation_a)。
- `r5_ablation_b`：Matched ablation of profile from validation-best complete combination r3_scale; all other coefficients fixed, training heads kept for parameter parity.
  权重：`{"confidence": 0.0455, "option": 0.1, "profile": 0}`。模型与配置：[目录](../examples/saved_model/joint_iterations_20261029/nips_task34/r5_ablation_b)。

## 6. 如何解释组合结果

判断联合是否有增益要同时看原DKT和此前最优单方案。超过原DKT不等于超过最强已有目标；完整组合与消融也分别报告。按学生配对自举固定1000次、seed20261029，匹配同一学生/题目/知识点/标签后精确重建原主AUC。bootstrap只针对固定模型和该测试群体，不能覆盖训练seed或fold波动，正差比例不是p值，也不参与选模。

每轮保留了固定验证loader的各项batch平均loss，以及前4个训练batch的骨干梯度余弦/加权范数。它们的掩码与分母不同，batch均值不是全数据按事件加权的loss；4个batch梯度也不能代表整个训练集或证明因果。它们只辅助识别梯度压力，不代替主指标和消融。

## 7. 验证、诊断修复与外包

**122项相关检查通过**（原七方案103项＋联合19项），包含所有七组件单独启用时与原loss/梯度一致、零辅助RNG/BCE/Adam一致、全任务一次LSTM、权重总和只包含一个主BCE、设备迁移和CPU标签/CUDA模型/eval对照、训练统计/验证复用、事件身份和变长块对齐、导出标准DKT。不是全仓测试通过声明。

首个ASSIST完整组合训练正常完成16 epoch并保存第6 epoch检查点，但后续梯度诊断在eval模式触发cuDNN反向限制。只将梯度诊断的forward改为原生LSTM，原训练/选模不改；保留原日志，核验检查点SHA后补齐诊断，**没有重训该轮**。其他轮次沿用同一训练/选模逻辑。见 [恢复记录](../logs/joint_iterations_20261029/diagnostic_fix/recovery.log)。

dsh交付联合Dataset草稿，但调用超时且未完成其环境内编译；主agent编译、检查并用两套真实数据验收。另一次只读审查提供了适配器未来兼容性和设备/mask测试建议；主agent检查原组件并增加两项CUDA/eval测试与真实掩码审计。旧组件都自行搬移所需标签，当前loss不读取view.training，因此未采纳仅凭假设改动实验中的核心代码。见 [Dataset验收](../logs/joint_iterations_20261029/dsh_dataset/acceptance.md)、[模型审查验收](../logs/joint_iterations_20261029/dsh_review/acceptance.md)。

- [完整结果JSON](../logs/joint_iterations_20261029/results.json)
- [ASSIST结果CSV](../logs/joint_iterations_20261029/assist2009_results.csv) / [NIPS结果CSV](../logs/joint_iterations_20261029/nips_task34_results.csv)
- [逐轮决策协议](../logs/joint_iterations_20261029/protocol.json)
- [测试前冻结选择](../logs/joint_iterations_20261029/selection_before_test.json)
- [学生配对自举](../logs/joint_iterations_20261029/paired_bootstrap.json)
- [模型导出/骨干/参数隔离检查](../logs/joint_iterations_20261029/artifact_checks.json)
- [源码快照](../logs/joint_iterations_20261029/source_manifest.json) / [数据缓存哈希](../logs/joint_iterations_20261029/data_manifest.json)
- [相关测试日志](../logs/joint_iterations_20261029/tests.txt)
- [模型、配置与逐题预测目录](../examples/saved_model/joint_iterations_20261029)

重现训练（使用新输出目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/bin/python examples/train_dkt_joint.py --dataset assist2009 --weights_json '{"action":0.1,"time":0.03,"attempt":0.1,"hint":0.1,"distillation":0.2,"transfer":0.1}' --save_dir examples/saved_model/joint_assist_reproduction
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/bin/python examples/train_dkt_joint.py --dataset nips_task34 --weights_json '{"confidence":0.182,"option":0.1,"profile":0.03}' --save_dir examples/saved_model/joint_nips_reproduction
```

当前结论限单fold/seed；没有以混合数据集、测试调参或模型集成替代同数据集联合监督。
