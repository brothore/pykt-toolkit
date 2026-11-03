# DKT 具体错误选项预测：三轮迭代与一轮消融

实验日期：2026-10-25；最终整理：2026-10-26。分支：`feat/student-state-simulator`。独立模型：`dkt_option`。

完成四轮实验：三轮错误选项方案迭代，加一轮仅正误监督的消融对照。按测试前保留检查点的验证 AUC，选定 **第一轮：四选项 CE**。`windowauclate_mean` 为 **0.768222936**，相对 DKT 为 **+0.000780123**。对照在训练前明确不参与方案选模。当前只有 NIPS task34 / fold 0 / seed 42，不据此认定普遍收益。

## 1. 新增监督是什么

原 DKT 只预测二元正误，本次辅助目标预测学生记录的选项 `AnswerValue`（1–4 映射为 0–3）。辅助头输入历史状态、下一题知识点与题目 ID 的嵌入；题目 ID 提供必要上下文，不能将跨题同一选项编号解释成同一错误原因。

DKT 交互嵌入、LSTM、主输出层及 `forward(c,r)` 继承不变。题目嵌入 64 维、辅助投影 100 维仅在训练/行为预测时使用，答题推理可全部移除。四个版本总参数均为 463433，原 DKT 与导出主干均为 355857。没有合并之前的信心损失，也没有将历史选项作为新的主干输入。

## 2. 原始对齐与答案键

复用已有 NIPS `AnswerId` 唯一事件、时间戳并列时原 CSV 行序、完整及分块 q/c/r/time 对齐。复用的是事件对齐器，不是置信度标签或其稀疏掩码，本次不局限于填写 Confidence 的记录。

选项与正确键均完整，`AnswerValue == CorrectAnswer` 与 `IsCorrect` 一致。但原始数据有 4 道题的正确键变化，版本/选项重排原因未确认。仅从训练折实际可监督原始事件拟合稳定键：944 题稳定、4 题不稳定、0 题完全未知。不稳定题以及验证记录与训练稳定键不符的情况，**只从辅助监督中排除**；主 BCE 和原测试集保留原样。

| 事件数 | 训练折 | 验证折 |
|---|---:|---:|
| 原始 next-step 目标事件 | 881747 | 220625 |
| 稳定键辅助事件 | 878510 | 219801 |
| 其中答错事件 | 408002 | 102890 |
| 排除不稳定/未知/键不匹配 | 3237 | 824 |

同题真实重试按 AnswerId 保留；多知识点展开仅首次位置监督，跨块副本不重复计入。训练折统计的稳定键、每题错误选项频次传给验证集；验证/测试不拟合这些统计。下一题实际 CorrectAnswer、AnswerValue、IsCorrect 不能作为预测输入。静态答案键是训练中已知题目属性，不是未来学生答案。

仅稳定正确键并不能保证不同版本的所有错误选项语义相同；本地缺少完整选项文本/语义对齐，当前建模的是记录编号，不能宣称识别了概念误解。对齐证据见 [alignment_audit.json](../logs/option_iterations_20261025/alignment_audit.json)。

## 3. 四轮目标与决策依据

主损失保持原 masked BCE，辅助只在去重原始事件上归一化：`L = BCE + 0.1 L_aux`。

| 轮次 | 辅助目标 | 诊断与下一步 |
|---|---|---|
| 第一轮 | 直接四选项 CE | 主验证 AUC 提升，但辅助正确选项概率与主概率 MAE 约 0.103；下一轮去掉重复正误并约束概率一致 |
| 第二轮 | 仅答错时的三个干扰项条件 CE | 正确键概率与原 DKT 一致；错误 NLL 微降但主验证 AUC 回落，下一轮加入题目静态先验 |
| 第三轮 | 条件 CE，logits 加训练题目错误频次先验 | 用残差学习超出题目偏好部分；主验证 AUC 未超过第一轮，需消融检查新信号收益来源 |
| 第四轮 | 同头同参数，仅监督辅助正确率 | 不提供具体错误选项身份；在训练前标为诊断对照，不参与错误选项方案选模 |

第二、三轮定义 `P(a=k)=p_DKT`，`P(a=j!=k)=(1-p_DKT)softmax_wrong(z)_j`，其中 k 为训练稳定键。真实未来正误仅用于选择哪些训练样本计算条件 loss，预测时不读取它。未知键标记不可用，主 DKT 预测仍可执行。

第三轮 `z = f(history,concept,question)+log π_train(a|question,wrong)`，先验为每个干扰项加 1 个伪计数，正确键概率为 0。这是固定 logit 偏置，不增加可训练参数。第一、二、三轮结构和初始化相同，仅目标/先验不同。

第四轮精确控制第一轮的正误分量。若四选项头分布为 s，则 `CE(choice)=BCE(s_correct,r)+(1-r)CE(choice|wrong)`。第四轮去掉后一项，保留相同题目上下文、头、参数、权重和事件归一化。该分解与错误身份变化不影响对照 loss 已通过测试。第二/三轮按错误子集单独平均，不能把其 0.1 权重与第一轮的错误分量强度直接等同。

全过程决定时间和证据见 [protocol.json](../logs/option_iterations_20261025/protocol.json)。

## 4. 公平设置、冻结与基线复用

NIPS task34，fold 0，seed 42，最多 20 epoch，batch 32，embedding/LSTM 200，dropout 0.2，Adam 0.001；关闭 ALSO/wandb/增强，训练/验证序列 6043 / 1503。每轮从相同原 DKT 初始化重新训练，不延续上一轮权重。保留 `auc > max_auc+1e-3` 检查点更新与 10 epoch patience。

复用上一轮同设置冻结 DKT 基线及已有测试结果，没有声称本次重新训练/评估基线。检查点 SHA 为 `cb9b88b69bc203ce2ec2f2121302e02840afdcd63daab4f888b2fc9c0f6a9d4f`，训练/验证原始来源哈希与前轮一致，当前逐题窗口 CSV/缓存早于原基线结果，详见 [baseline_data_checks.json](../logs/option_iterations_20261025/baseline_data_checks.json) 与 [baseline reference](../examples/saved_model/option_iterations_20261025/baseline/reference.json)。

冻结时间 `2026-10-25T10:04:30.921200+00:00`。候选仅含原 DKT、第一至三轮；第四轮消融排除决定先于其训练及所有新测试。见 [selection_before_test.json](../logs/option_iterations_20261025/selection_before_test.json)。

新模型只跑主指标所需的逐题窗口评估：原 KTDataset、`evaluate_question(...,['late_fusion'])`、batch 512，和旧基线对应评估一致。公式与融合不变；本次未重复普通测试/普通窗口/非窗口逐题三项，不把它们写成已完成。新结果 `all_results.json` 中直接读取 `windowauclate_mean`。

## 5. 主指标结果

| 方案 | 最佳 / 停止 epoch | 保留检查点验证 AUC | windowauclate_mean |
|---|---:|---:|---:|
| 原始 DKT（复用） | 6 / 16 | 0.762496754 | 0.767442813 |
| 第一轮：四选项 CE | 7 / 17 | 0.763264599 | 0.768222936 |
| 第二轮：仅错误选项条件 CE | 5 / 15 | 0.762421917 | 0.767202420 |
| 第三轮：题目先验加残差 | 6 / 16 | 0.762788057 | 0.767809196 |
| 第四轮：仅正误消融（不参选） | 6 / 16 | 0.763175734 | 0.768112883 |

选定方案相对 DKT 差值 **+0.000780123**，相对仅正误消融差值 **+0.000110053**。不根据测试结果改选。

以 983 名测试学生为单位，配对自举 1000 次；保存的 271833 条逐题预测能精确重建主指标。选定方案减 DKT 的 95% 百分位区间为 **[+0.000462273, +0.001086138]**；减仅正误消融的区间为 **[-0.000175653, +0.000401957]**。这仅覆盖固定模型和当前测试学生的抽样不确定性，不覆盖训练 seed/fold 变化；自举正比例不是 p 值，且不用于选模。见 [paired_bootstrap.json](../logs/option_iterations_20261025/paired_bootstrap.json)、[对照比较](../logs/option_iterations_20261025/paired_bootstrap_vs_control.json)。

相对原 DKT 的学生聚类区间在 0 以上，但相对仅正误消融的区间包含 0。因此对当前固定模型与测试学生可以观察到小幅收益，**尚无充分证据说明具体错误身份监督带来了超出二元辅助监督的主指标增益**；跨训练 seed/fold 的稳定性仍未知。

## 6. 选项行为诊断

下表使用相同 219801 个稳定键验证事件，其中 102890 个答错事件。错误选项指标条件于答错，正确键先排除再归一化；主指标仍是上一节的官方字段。

| 方案 | 错误选项 NLL | 错误选项准确率 | 全选项 NLL | 正确键概率与主概率 MAE |
|---|---:|---:|---:|---:|
| 第一轮：四选项 CE | 1.023275 | 0.463514 | 1.033809 | 0.102873 |
| 第二轮：仅错误选项条件 CE | 1.022352 | 0.463544 | 1.056970 | 0.000000 |
| 第三轮：题目先验加残差 | 1.026213 | 0.463893 | 1.059245 | 0.000000 |
| 第四轮：仅正误消融（不参选） | 1.642835 | 0.341608 | 1.322943 | 0.102516 |

仅题目错误选项先验 NLL 为 1.036328，准确率为 0.464204。前三轮的验证错误 NLL 略优于静态先验，但准确率未明显超过；第四轮没有错误身份监督，其错误选项行为指标不能当作经过训练的干扰项模型。第一轮主验证 AUC 与仅正误对照很接近，因此不能仅把相对原 DKT 的收益全归因于细粒度错误信息。

## 7. 实现与验收

模型：[dkt_option.py](../pykt/models/dkt_option.py)。数据：[option_dataset.py](../pykt/datasets/option_dataset.py)。训练：[train_dkt_option.py](../examples/train_dkt_option.py)。主指标评估：[evaluate_dkt_option.py](../examples/evaluate_dkt_option.py)。

69 项相关检查通过：原 DKT 零辅助权重的 RNG/Adam 等价、辅助梯度进入历史状态和题目上下文、未来标签隔离、条件概率一致、原始事件和折隔离、训练键/先验、未知与不稳定题、保存加载/导出，以及消融数学分解。见 [tests.txt](../logs/option_iterations_20261025/tests.txt)，不代表全仓测试均通过。

四份实际纯 DKT 导出经原类严格加载后，答题预测逐元素一致；原 DKT、标准加载器、ALSO、kt_config 及此前信心模型文件均未改，检查见 [artifact_checks.json](../logs/option_iterations_20261025/artifact_checks.json)。只增加新模型分发，评估公式不改。

dsh 只负责模型草稿，180 秒超时返回 124，不能算完整 worker 成功；主 agent 审查后接入，增加消融、原始对齐、测试与训练评估。见 [外包验收](../logs/option_iterations_20261025/dsh_model/acceptance.md)。

## 8. 复现

从仓库根目录执行，使用新的输出目录；默认即上述 fold/seed/epoch。先训练及验证，冻结配置后才测试：

```bash
python examples/train_dkt_option.py --save_dir examples/saved_model/new_option/r1
python examples/train_dkt_option.py --option_mode conditional --save_dir examples/saved_model/new_option/r2
python examples/train_dkt_option.py --option_mode prior --save_dir examples/saved_model/new_option/r3
python examples/train_dkt_option.py --option_mode binary_control --save_dir examples/saved_model/new_option/control
python examples/evaluate_dkt_option.py --save_dir examples/saved_model/new_option/r1
```

剩余局限：单 fold/seed、没有完整选项语义、训练期多了题目上下文而主 DKT 不直接接收题目 ID、消融与方案可能早停于不同 epoch。新增行为预测不等于完整学生模拟器。结果与完成情况见 [results.json](../logs/option_iterations_20261025/results.json)、[results.csv](../logs/option_iterations_20261025/results.csv)、[实验进度](学生模拟_实验进度.md)。
