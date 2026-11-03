# DKT 信心与表现联合建模：四轮闭环实验

日期：2026-10-25。分支：`feat/student-state-simulator`。独立模型：`dkt_confidence`。

已完成四轮验证驱动迭代。测试前按保留检查点的验证 AUC 选定 **第二轮：有序监督**，最终 `windowauclate_mean` 为 **0.767748594**；相对 NIPS DKT 基线为 **+0.000305781**（提升约 0.0306 个百分点）。本次仅单 fold、单 seed，不能认定差异显著或推广到其他数据集。

## 1. 完成范围

此前首步行为三轮和作答过程四轮的完成情况已记录到 [实验进度](学生模拟_实验进度.md)。本次实现七方案中的第三项，NIPS 的自报信心与正误联合监督，不构造 ASSIST 的伪置信度标签。

DKT 的交互嵌入、LSTM、主输出层及 `forward(c,r)` 保持不变。训练辅助头使用历史状态和下一题已知知识点，推理答题时可移除。它没有读取目标真实正误/置信度，也没有把历史置信度作为新的主干输入。每轮从相同原始 DKT 初始化重新训练；不在上一轮权重上继续训练。

## 2. 原始监督与对齐

Confidence 为自报信心的 0、25、50、75、100 五档，映射标签 0–4。**0 是有效记录，不是缺失。** 缺失/非法值标记 -100 并掩码，不作为低信心，也不作为答对率软标签。

原始 `AnswerId` 唯一，答题元数据按一对一连接；同一分钟存在大量时间戳并列，严格复现原预处理的 `(answer_timestamp, primary CSV row index)` 排序。按题目层级提取原有 level-3 知识点，逐一核对完整及分块序列的题目、知识点、正误、时间戳和 `is_repeat`，然后切分辅助标签。多知识点同一事件仅首次位置监督，跨块副本不重复计入；真实重复作答保留。

只对训练/验证学生构造标签，测试链路不加载 Confidence。标签 sidecar 位于 `data/nips_task34/dkt_confidence/targets_v1.npz`，其原始来源哈希见 [data_alignment_manifest.json](../logs/confidence_iterations_20261025/data_alignment_manifest.json)。旧标准缓存若与 CSV 不一致，直接拒绝配对，不静默重写其他模型的数据。

| 范围 | 有效原始目标事件数 | 有置信度事件数 | 置信度覆盖 |
|---|---:|---:|---:|
| 训练折 | 881747 | 216991 | 24.6092% |
| 验证折 | 220625 | 56377 | 25.5533% |

训练档位计数 `[18975,11237,33601,54461,98717]`，验证 `[3473,2706,8312,15114,26772]`，顺序均为 0/25/50/75/100。原始训练/验证学生共有 1109911 条记录、276506 条非空置信度；分块和 next-step 掩码后监督事件数与原始行数不同。

置信度填写不是随机缺失：学生填写比例差异较大，不能将有置信度记录上的结论直接外推至全部学生。字段不能证明真实能力、粗心或心理状态，也未据本地数据确认其相对反馈的精确采集时机。

## 3. 四轮目标函数与闭环依据

主损失一直为原始 masked BCE；辅助损失在各自有效原始事件上单独归一化：`L = BCE + λ_conf × L_conf`。

| 轮次 | 辅助目标 | λ_conf | 类别平衡指数 | 决策依据 |
|---|---|---:|---:|---|
| 第一轮 | 五档置信度交叉熵 | 0.1 | 0 | 建立独立置信度边缘分布对照；主任务略低于基线，五档 argmax 强烈偏向 100 |
| 第二轮 | 五档 softmax 的累积有序 BCE | 0.182 | 0 | 让相邻档与远距离错误受到不同惩罚；训练初始四个 batch 梯度匹配 CE 的强度，主验证 AUC 本次最高 |
| 第三轮 | 正误条件置信度 CE | 0.1 | 0 | 预测两个条件分布，纠正独立边缘分解不能显式表达信心与正误关系的问题；联合 NLL 改善 |
| 第四轮 | 第二轮加逆平方根频次加权 | 0.182 | 0.5 | 针对少数档预测不足；macro F1 提升，但 NLL/有序误差变差，主验证 AUC 未超过第二轮 |

有序损失使用 `P(level>k), k=0..3` 与真实累计标签的 BCE，再对四个阈值取平均，五档仍是归一化 softmax 分布。为避免只因量纲变小而比较不公平，仅用训练折最初四个 batch、同一初始化和 logits 估计 CE/有序损失对共享主干的梯度范数比 1.817668，得到 λ=0.182；不使用验证或测试数据拟合此系数。它只匹配初始局部强度，不保证整个训练阶段梯度一致。见 [ordinal_calibration.json](../logs/confidence_iterations_20261025/ordinal_calibration.json)。

第三轮定义 `P(r,c|history,k)=P(r|history,k)P(c|r,history,k)`（k 为目标题知识点），输出两个五档条件头；真实正误仅用于选择训练标签对应的损失分支，预测时两个分支都从历史计算，再用 DKT 的正误概率混合，不能读取目标答案。其正误边缘仍是原 DKT 的预测。置信度缺失时仅计算原 BCE，不填充自报值。

第一、二、四轮的联合概率是 `P(r)P(c)` 的独立分解，只作为对照；不能将它们描述成已经学到了条件依赖。第四轮类别权重仅来自训练计数，按实际样本权重和归一化，不用验证/测试估计。

每轮的决定时间、诊断依据见 [protocol.json](../logs/confidence_iterations_20261025/protocol.json)。

## 4. 固定训练与选模设置

NIPS task34，fold 0，seed 42，最多 20 epoch；batch 32，embedding/LSTM 200，dropout 0.2，Adam，学习率 0.001。关闭 ALSO、wandb 和增强。训练/验证序列分别 6043 / 1503。原 DKT 参数 355857，五档/有序训练模型 396462，条件模型 396967；主干及导出模型参数量一致。

保留原训练器 `auc > max_auc + 1e-3` 的检查点更新规则和 10 epoch patience，没有修改为测试指标选模。全部模型训练完成后，按保留检查点验证 AUC、包含原始 DKT 基线一起选择。冻结时间 `2026-10-25T08:29:01.332272+00:00`，选定 `r2_ordinal`。随后五份检查点各运行一次完整测试，测试 batch 512、四种测试文件与融合公式相同。检查点 SHA256 保存于 [selection_before_test.json](../logs/confidence_iterations_20261025/selection_before_test.json)，评估后均不变。

## 5. 主指标

全部主指标直接读取各实验 `all_results.json` 中的 `windowauclate_mean`。

| 方案 | 最佳 / 停止 epoch | 验证 AUC | windowauclate_mean |
|---|---:|---:|---:|
| 原始 DKT | 6 / 16 | 0.762496754 | 0.767442813 |
| 第一轮：五档分类 | 6 / 16 | 0.762257276 | 0.767302366 |
| 第二轮：有序监督 | 8 / 18 | 0.762964677 | 0.767748594 |
| 第三轮：正误条件联合分布 | 7 / 17 | 0.762760404 | 0.767922841 |
| 第四轮：有序监督加类别平衡 | 8 / 18 | 0.762830428 | 0.767631763 |

保留测试前选定版本，不根据测试表改选或追加训练。第三轮测试主指标略高于第二轮，但验证 AUC 未超过第二轮，因此仍保留第二轮为选定配置。NIPS 与此前 ASSIST2009 的主指标分数不能直接比较。

对基线与已选第二轮的逐题窗口预测重建 AUC，与保存的主指标精确一致；按 983 名学生配对重采样 1000 次，AUC 差的 95% 百分位区间约为 **[-0.000117, +0.000738]**，包含 0。这只反映固定训练权重与当前测试学生的抽样不确定性，不涵盖训练 seed/fold 变化，也未用于选模。不能据此认定收益稳定成立。见 [paired_bootstrap.json](../logs/confidence_iterations_20261025/paired_bootstrap.json)。

## 6. 信心与联合行为诊断

以下均为相同 56377 条有自报置信度的验证事件，额外 CPU 复算用于统一诊断口径；训练后 GPU 诊断仍原样保留，存在轻微浮点差异。档位 MAE 单位为 0–4 的等级，乘 25 才是自报信心数值的误差，不是答对率误差。

| 方案 | 信心 NLL（低更好） | 档位 MAE（低更好） | 五档 macro F1 | 正误×信心联合 NLL（低更好） |
|---|---:|---:|---:|---:|
| 第一轮：五档分类 | 1.279481 | 0.878010 | 0.149124 | 1.836808 |
| 第二轮：有序监督 | 1.283085 | 0.877173 | 0.155594 | 1.838770 |
| 第三轮：正误条件联合分布 | 1.272276 | 0.878650 | 0.151222 | 1.812022 |
| 第四轮：有序监督加类别平衡 | 1.291363 | 0.945297 | 0.195063 | 1.847501 |

训练全局先验置信度 NLL 1.311836，知识点先验 1.302020，题目先验 1.266322。题目先验由训练折各题频次加 10 个全局先验伪计数平滑得到。当前模型虽优于全局/知识点先验，仍未超过这个题目先验，不能仅凭辅助 loss 下降宣称更准确识别了学生心理状态。

另用训练折构建 `P(Confidence|正误,题目)` 静态对照，保持第三轮的主正误概率不变。验证联合 NLL：第三轮模型约 1.812022，全局正误条件先验约 1.830375，题目×正误条件先验约 1.798215。模型超过全局条件先验，但仍未超过题目条件先验，因此联合分布改善也不能单独归因于更好的学生状态。该补充诊断未用于选模，见 [conditional_prior_diagnostic.json](../logs/confidence_iterations_20261025/conditional_prior_diagnostic.json)。

固定事件定义：高信心答错为 `r=0, Confidence>=75`，共 12207 条；低信心答对为 `r=1, Confidence<=25`，共 2144 条。第三轮联合 NLL 优于第一轮，高信心答错 AUC 约 0.686（第一轮约 0.682），但不代表所有分组指标都改善。第四轮从低频档学到更多分类信号，同时置信度概率校准和档位误差变差，必须同时报告。

主任务的置信度覆盖分组诊断使用原始事件首次位置，非官方主指标：

| 方案 | 有置信度事件答题 AUC（56377 条） | 无置信度事件答题 AUC（164248 条） |
|---|---:|---:|
| 原始 DKT | 0.751569 | 0.746983 |
| 第一轮：五档分类 | 0.751131 | 0.746996 |
| 第二轮：有序监督 | 0.753518 | 0.747139 |
| 第三轮：正误条件联合分布 | 0.752676 | 0.747177 |
| 第四轮：有序监督加类别平衡 | 0.753187 | 0.747040 |

完整混淆矩阵、预测档位计数、事件 Brier/AUC 和主干梯度诊断均保存在实验目录的 `validation.json` 与 `validation_diagnostics_extra.json`。没有将上述诊断当作 `windowauclate_mean`。

## 7. 代码与验收

- 模型：[dkt_confidence.py](../pykt/models/dkt_confidence.py)；原始对齐：[confidence_dataset.py](../pykt/datasets/confidence_dataset.py)；训练：[train_dkt_confidence.py](../examples/train_dkt_confidence.py)。
- 仅增加新模型的初始化/训练损失/评估分发，原 DKT 主干、标准加载器、ALSO 与 `kt_config.json` 未改；之前的作答过程模型、数据包装、runner、测试文件与其源码快照一致。
- 48 项相关测试通过，覆盖三个模式的零辅助权重 RNG/Adam 等价、辅助梯度、保存加载、推理不读取未来标签、联合归一化/正误边缘、有序距离惩罚、有效 0/缺失/非法值、原始时间并列/真实重试/跨块概念副本、缓存失效与折隔离。见 [tests.txt](../logs/confidence_iterations_20261025/tests.txt)，这不是全仓测试通过声明。
- 四份真实导出 `qid_dkt_backbone.ckpt` 由原 DKT 严格加载后，答题预测与完整模型逐元素一致；冻结权重、源码快照、旧模型隔离核验见 [artifact_checks.json](../logs/confidence_iterations_20261025/artifact_checks.json)。
- dsh 负责限定范围内模型文件草稿，180 秒超时设置下正常退出（0）；主 agent 审查并修正活动标签校验/掩码清理，完成其余代码、测试和实验。见 [外包验收](../logs/confidence_iterations_20261025/dsh_model/acceptance.md)。

## 8. 复现与剩余问题

从仓库根目录执行，输出目录必须是新目录。默认 fold 0、seed 42、20 epoch。先训练/验证，冻结选择后才测试：

```bash
python examples/train_dkt_confidence.py --model_name dkt --save_dir examples/saved_model/new_confidence/baseline
python examples/train_dkt_confidence.py --save_dir examples/saved_model/new_confidence/r1
python examples/train_dkt_confidence.py --confidence_mode ordinal --confidence_weight .182 --save_dir examples/saved_model/new_confidence/r2
python examples/train_dkt_confidence.py --confidence_mode conditional --save_dir examples/saved_model/new_confidence/r3
python examples/train_dkt_confidence.py --confidence_mode ordinal --confidence_weight .182 --confidence_balance .5 --save_dir examples/saved_model/new_confidence/r4
python examples/evaluate_dkt_first_action.py --batch_size 512 --save_dir examples/saved_model/new_confidence/r2
```

当前只覆盖单 fold/seed；没有证明缺失机制可忽略，没有在更多题目上下文下验证，也没有进行多 seed 显著性分析。训练辅助头不等于已完成完整学生生成器。七方案其余四项仍未实现。

结果汇总：[results.json](../logs/confidence_iterations_20261025/results.json)、[results.csv](../logs/confidence_iterations_20261025/results.csv)。
