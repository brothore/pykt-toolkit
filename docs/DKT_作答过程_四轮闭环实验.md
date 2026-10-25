# DKT 作答过程建模：四轮闭环实验

日期：2026-10-25。分支：`feat/student-state-simulator`。独立模型：`dkt_process`。

用户要求至少三轮迭代，本次实际完成四轮，每轮都依据前一轮验证结果与诊断调整，再从相同初始化重新训练。全部轮次结束后按验证 AUC 冻结第四轮，随后五个检查点各进行一次完整测试。当前选定版本的 `windowauclate_mean` 为 **0.752837372**，相对 DKT 为 **+0.002007166**。单 fold、单 seed 的探索结果不代表普遍提升。

## 1. 具体实现

DKT 的交互嵌入、LSTM、答题输出层和 `forward(c,r)` 保持原实现。训练时用历史状态 `h_t` 和下一题已知知识点的嵌入，经过一个训练期共享投影，再分别预测三个过程目标。目标题的真实回答、耗时、尝试和提示字段都不能进入预测输入。

| 目标 | 标签与损失 | 有效性规则 |
|---|---|---|
| 首反应耗时 | `log1p(ms_first_response/1000)`，按训练折均值/标准差标准化后计算 Smooth L1 | 只保留大于 0 且不超过 1 小时的记录；不把相邻时间戳差当作耗时 |
| 尝试次数 | `attempt_count` 分为 0、1、2、3+ 四档，交叉熵 | 0 是真实记录类别，不作缺失；负值、非整数和缺失掩码；极端次数落入 3+ |
| 提示使用 | `hint_count>0`，二元交叉熵 | 仅 `hint_total>0` 且首动作属于普通尝试或提示的记录；排除脚手架、异常和缺失 |

总损失：`L = BCE + λ_time × L_time + λ_attempt × L_attempt + λ_hint × L_hint`。

`joint` 在各自所有有效事件上监督；`conditional` 只在目标 `correct=0`（未首次答对，包括提示优先等情境）的事件上监督尝试和提示，耗时仍用所有有效事件。条件模型输出的尝试/提示概率是失败条件分布，不应当作所有题目的无条件分布。本次没有合并首步行为分类损失，也没有实现提示强度、底部提示或心理状态标签。

辅助头推理时可以移除。每个新模型均导出 `qid_dkt_backbone.ckpt`，可直接由原始 `DKT` 严格加载；四份实际导出权重与对应完整模型的答题预测逐元素一致。

## 2. 数据对齐与隔离

复用首步行为实验验证过的原始事件对齐器：使用唯一 `order_id`，逐一核对现有完整序列和分块序列的题目、知识点及正误。再通过原始事件 ID 关联过程字段。相同题目的真实重试保留；多知识点展开仅在原事件首次出现处监督，跨块复制也不重复计入。块首位置与原 DKT 一样不作为下一步目标，不能将已经进入历史的同事件概念副本用于辅助监督。

只有训练/验证学生的原始事件构造过程标签，测试评估不加载过程标签。归一化、类别统计及知识点先验仅从训练折拟合，验证集接收训练统计。原始 CSV、标准 KTDataset 和其他模型输入均未改写。检测到旧缓存与源序列不一致时直接报错。

| 监督 | 训练事件数 | 验证事件数 |
|---|---:|---:|
| 耗时 | 182351 | 43127 |
| 尝试次数（联合） | 182640 | 43191 |
| 提示使用（联合） | 123844 | 29160 |
| 尝试次数（失败条件） | 61259 | 14568 |
| 提示使用（失败条件） | 43581 | 10233 |

完整清洗计数见 [alignment_audit.json](../logs/process_iterations_20261025/alignment_audit.json)。过程字段完整不等于每个取值都有效。

## 3. 固定实验设置

ASSIST2009，fold 0，seed 42；最多 20 epoch，batch 32，embedding/LSTM 200，dropout 0.2，Adam，学习率 0.001；关闭 ALSO、wandb 和数据增强。训练/验证序列分别为 3016 / 741。

保持原训练器的检查点更新规则 `auc > max_auc + 1e-3` 与 10 epoch patience，未改成测试指标选模。跨轮次比较保留检查点的验证 AUC；最高者第四轮在测试前确定。冻结时间 `2026-10-25T07:59:16.577305+00:00`，检查点 SHA256 见 [selection_before_test.json](../logs/process_iterations_20261025/selection_before_test.json)。测试后检查点哈希保持一致。

本次 DKT 基线重新训练，检查点 SHA256 与上一轮独立 DKT 基线完全相同。基线与第一轮部分并行，记录的训练 wall time 不适合用于估算额外成本。原始 DKT 参数 395523，含过程头的训练模型参数 436229；移除头后参数量恢复。

## 4. 四轮闭环及决策依据

| 轮次 | 监督方式 | 耗时 / 尝试 / 提示权重 | 诊断与下一步 |
|---|---|---|---|
| 第一轮 | 联合 | 0.1 / 0.1 / 0.1 | 验证 AUC 高于 DKT，三项过程预测优于全局先验；联合尝试类别偏向 1 次，下一轮改用失败条件监督 |
| 第二轮 | 失败条件 | 0.1 / 0.1 / 0.1 | 在相同错误样本上尝试类别区分更好，也优于知识点先验；主任务 AUC 略低于第一轮，辅助梯度相对较强，下一轮尝试整体降权 |
| 第三轮 | 失败条件 | 0.03 / 0.03 / 0.03 | 主任务和过程预测均退步，不支持整体降权；下一轮回到第一轮，只降低梯度对齐较弱的耗时项 |
| 第四轮 | 联合 | 0.03 / 0.1 / 0.1 | 本次验证 AUC 最高，冻结为选定配置 |

第一轮共享主干的梯度余弦：耗时约 0.087，尝试 0.231，提示 0.251。第二轮原始辅助梯度范数/主任务范数约 2.59、4.00、2.97。该诊断只取前四个验证 batch，无参数更新，用作局部线索，不代表证明某任务存在因果收益。每轮的决定、时间和证据保存于 [protocol.json](../logs/process_iterations_20261025/protocol.json)。

## 5. 主指标结果

下列数值全部读取各实验 `all_results.json` 的 **`windowauclate_mean`**，没有使用 `window_testauc` 或 `oriauclate_mean` 代替。

| 方案 | 最佳 / 停止 epoch | 验证 AUC | windowauclate_mean |
|---|---:|---:|---:|
| 原始 DKT | 4 / 14 | 0.830839499 | 0.750830206 |
| 第一轮：三项联合监督 | 5 / 15 | 0.832695081 | 0.752888074 |
| 第二轮：失败条件监督 | 5 / 15 | 0.832539509 | 0.752245421 |
| 第三轮：整体降权 | 4 / 14 | 0.831332653 | 0.751129385 |
| 第四轮：仅降低耗时权重 | 6 / 16 | 0.832901894 | 0.752837372 |

第四轮由验证集选定，主指标相对基线变化 **+0.002007166**（+0.2007 个百分点）。第一轮测试值略高于第四轮，但第四轮在测试前已由验证集选定，因此不根据测试结果改选。此前首步行为模型最佳版本在相同数据集/fold/seed/epoch 预算下为 0.752309604，这里只作为既有实验参照，不参与本次选模。

## 6. 过程任务诊断

为公平比较联合与条件模型，下面的尝试 F1 和提示 AUC 都限定在**相同未首次答对事件（correct=0）**上，分别为 14568 / 10233 条。耗时 MAE 在相同有效耗时事件上计算，单位为 `log1p(seconds)`，不是秒。

| 方案 | 耗时 log MAE（低更好） | 错误子集尝试 macro F1 | 错误子集提示 AUC |
|---|---:|---:|---:|
| 第一轮：三项联合监督 | 0.780834 | 0.204965 | 0.761014 |
| 第二轮：失败条件监督 | 0.784474 | 0.437722 | 0.768559 |
| 第三轮：整体降权 | 0.801371 | 0.423920 | 0.761849 |
| 第四轮：仅降低耗时权重 | 0.788997 | 0.208021 | 0.760282 |

知识点对照只用训练折估计每个知识点的均值或类别频率，使用全局先验平滑。第四轮耗时 MAE 约 0.789，知识点均值约 0.806；尝试 NLL 约 0.630，知识点先验约 0.665；提示 NLL 约 0.411，知识点先验约 0.481。辅助头超出这些静态对照，但这不足以证明潜在心理状态已被识别。

验证诊断的补充复算使用 CPU，原训练后诊断使用 GPU，存在轻微浮点差异；保留两份原始输出。上表统一使用各版本 `validation_diagnostics_extra.json` 的 CPU 复算结果。

## 7. 实现与验收

- 模型：[dkt_process.py](../pykt/models/dkt_process.py)。数据：[process_dataset.py](../pykt/datasets/process_dataset.py)。训练：[train_dkt_process.py](../examples/train_dkt_process.py)。
- 只对新模型增加初始化、训练损失和现有 DKT 评估分发；未改变评估公式。测试通过原 KTDataset 和 `evaluate` / `evaluate_question`，四种测试文件均完成。
- 30 项相关检查通过，覆盖原 DKT 零辅助权重的初始化/RNG/Adam 更新等价、辅助梯度、未来目标隔离、跨事件和跨块对齐、训练/验证统计隔离、异常值与掩码、保存加载及纯 DKT 导出；见 [tests.txt](../logs/process_iterations_20261025/tests.txt)。这不是全仓测试通过声明。
- `dkt.py`、`data_loader.py`、`ALSO/also.py`、`configs/kt_config.json` 与 HEAD 字节一致。四份真实导出、检查点哈希与源码快照核验见 [artifact_checks.json](../logs/process_iterations_20261025/artifact_checks.json)。
- dsh 仅负责模型文件草稿，CLI 在 240 秒超时（124），已交付文件由主 agent 检查后接入，不能把该 worker 调用算作完整成功。主 agent 完成对齐、损失审查、测试、实验和最终验收；见 [外包验收记录](../logs/process_iterations_20261025/dsh_model/acceptance.md)。

未完成耗时/尝试/提示单独训练的完整消融，也未覆盖更多 fold/seed；因此不能将联合效果归因于某一个信号，或宣称显著、稳定优于已有方案。三个辅助头描述不同可观察过程目标，尚不构成一致的完整学生生成模型。

## 8. 复现

从仓库根目录执行，输出目录必须为新目录。先训练及验证，冻结配置后再测试：

```bash
python examples/train_dkt_process.py --model_name dkt --save_dir examples/saved_model/new_process/baseline
python examples/train_dkt_process.py --save_dir examples/saved_model/new_process/r1
python examples/train_dkt_process.py --process_mode conditional --save_dir examples/saved_model/new_process/r2
python examples/train_dkt_process.py --process_mode conditional --time_weight .03 --attempt_weight .03 --hint_weight .03 --save_dir examples/saved_model/new_process/r3
python examples/train_dkt_process.py --time_weight .03 --save_dir examples/saved_model/new_process/r4
python examples/evaluate_dkt_first_action.py --save_dir examples/saved_model/new_process/r4
```

默认即上述 fold 0 / seed 42 / 20 epoch 设置；共享评估脚本现在同时支持 `dkt`、`dkt_first_action` 和 `dkt_process`。结果汇总见 [results.json](../logs/process_iterations_20261025/results.json) 和 [results.csv](../logs/process_iterations_20261025/results.csv)。
