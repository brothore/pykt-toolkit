# DKT 独立解决与迁移监督：四轮实验

日期：2026-10-26。分支：`feat/student-state-simulator`。方案 S05，独立模型 `dkt_transfer`。

已完成三轮未来迁移目标迭代与一轮下一事件对照。测试前按保留检查点的验证 AUC 选定 **第二轮：同模板全结果独立成功代理**，`windowauclate_mean` 为 **0.752619064**，相对原 DKT **+0.001788858**。但相对更简单的下一原始事件对照为 **-0.001188601**，当前结果不支持未来模板监督胜过该对照。

前五项共 19 轮已分别补充配置、退步版本、对照和证据，见 [逐轮索引](迭代记录/README.md)；这里记录新增 S05，不改写旧实验结果。

## 1. 严格只改训练目标

`DKTTransfer` 继承原 DKT 的 `forward(c,r)`，没有新模块、参数或注册 buffer，所有版本均 **395523** 参数。原 `qid_model.ckpt` 和导出均可直接由 DKT 严格加载，实际预测逐元素一致。推理不需要模板、提示、未来标签或未来题目字段。

给定当前已完成原始事件后的状态 `h_t`，DKT 已输出各知识点概率 `p_t(c)`。对未来目标题的去重知识点集合 C，取 `mean(p_t(c),c in C)`，用其监督未来记录的目标：`L = BCE_original + 0.1 BCE(future_mean_probability, future_label)`。未来知识点只在 loss 中查询已有输出，未来答案/行为不会回填到状态。

均值是整题表现的工程近似，不保证满足多知识点 AND 等认知模型。它不是闭环 rollout，也没有模拟中间学习步骤。原主 BCE 仍保留原 concept-expanded masks；额外 loss 按有效原始事件锚点归一化，零辅助权重与原 DKT 的 RNG/loss/Adam 更新一致。

## 2. 原始事件、模板与标签

复用 ASSIST `order_id` 唯一事件、完整/分块题目知识点正误核对。锚点是输入中**当前原始事件的最后一个知识点**，保证当前事件已经完成；目标只能来自同一学生的更晚原始事件。多知识点副本不是未来事件，跨块副本可以作为已完成事件末位置，不把它们当作新题。

未来窗口最多 **10 个原始事件**，不按知识点展开次数计数。第一至三轮都排除同一个题目 ID 的重试：目标必须为不同题目。没有可观察的合格未来练习则掩码，不标成失败。原始尾部事件即使被短 chunk 丢弃，也可作为同一训练学生的未来监督标签，不能成为当前输入。不同锚点可复用同一未来目标，配对数不是独立样本数。

观测代理定义：`first_action==0 and hint_count==0 and attempt_count>=1` 为普通首动作、无记录提示且有尝试；独立首次成功代理再要求 `correct==1`。它不是“没有任何外部帮助”的真值，也不是最终经过多次尝试后是否独立解决。229467 条原始训练/验证记录中，此成功代理只在 **474 条**上不同于原正误标签，不能把它说成大量全新心理标注；主要新增约束是**模板关系和监督时点**。

所有配对只在训练/验证学生内部构造，不使用测试学生标签；同学生未来仍属同一折。模板 ID 仅用于 loss 标签配对，未添加模板嵌入。模板相同不代表等难，知识点集合相同也不证明因果意义的迁移。

## 3. 四轮闭环与选择偏差

| 轮次 | 配对与标签 | 验证观察与下一步 |
|---|---|---|
| 第一轮 | 同模板不同题，只选择未来无记录提示的首次尝试，目标为其首答正误 | 目标成功率约81%，标准验证 AUC 回落；下一轮保留未来求助/失败，避免未来行为筛选 |
| 第二轮 | 同模板不同题，保留全结果，以独立首次成功代理为标签 | 成功率约68%，标准验证 AUC 与同模板未来诊断改善，但跨模板诊断未改善 |
| 第三轮 | 未来题知识点集合完全相同，题目与模板均不同；保留全结果 | 跨模板诊断提高，但标准验证 AUC 回落，暴露未来迁移与下一步目标的取舍 |
| 第四轮 | 只预测下一原始事件；相同整题概念平均、代理标签与权重，无模板限制 | 诊断对照，在训练前声明不参与 S05 方案选模；检查是否需要更远/模板约束 |

所有模型在**相同四个验证配对集合**上分别诊断，不混用不同成功率群体的 AUC。第一轮与第二轮不只是标签变化，也改变了未来目标选取位置；不能把全部差异因果归因于提示筛选。配对、标签与 UID/时序关系已独立核验，见 [pair_audit.json](../logs/transfer_iterations_20261026/pair_audit.json)。

| 配对集合 | 训练锚点 | 验证锚点 | 训练 / 验证成功率 | 平均未来距离（验证） |
|---|---:|---:|---|---:|
| 同模板、未来无提示子集 | 100794 | 23999 | 0.8081 / 0.8105 | 1.8355 |
| 同模板全结果 | 110376 | 26251 | 0.6853 / 0.6815 | 1.6920 |
| 同知识点集合跨模板 | 93385 | 21672 | 0.6150 / 0.6239 | 1.9999 |
| 下一原始事件对照 | 182640 | 43191 | 0.6626 / 0.6602 | 1.0000 |

下一事件对照不是严格重复原 BCE：它在当前同一前缀上平均未来整题全部知识点概率，目标也是上述成功代理；标准 BCE 按展开位置分别预测。因此不能据此认定先前 S04 缺失的“仅硬标签加权”消融已完全补齐。

## 4. 固定设置、冻结与基线复用

ASSIST2009 / fold 0 / seed 42 / 最多20 epoch，batch32，embedding/LSTM200，dropout0.2，Adam0.001。关闭 ALSO、wandb、增强。每轮重新从相同 DKT 初始化训练，保留 `auc>max_auc+1e-3` 更新和10 epoch patience。训练/验证序列3016/741，学生2465/617。

复用同设置原 DKT 基线，没有声称本次重新训练/评估基线；检查点 SHA 为 `8094b98fa8047f51b2478d4e5794a1c417b8ad8eab0fc527b122e058e944475a`，原始来源和逐题窗口 CSV 哈希匹配旧实验。见 [baseline_data_checks.json](../logs/transfer_iterations_20261026/baseline_data_checks.json)。

冻结时间 `2026-10-26T13:57:51.089071+00:00`。第四轮对照在训练前排除选模候选；候选包含基线和三种未来方案。第二轮按标准验证 AUC 选定，不按测试表改选。权重、配对缓存与源码哈希见 [selection_before_test.json](../logs/transfer_iterations_20261026/selection_before_test.json)。

新版本只评估主指标所需的原逐题窗口 late fusion，batch128 与基线相同；测试不构造未来模板标签、不加载原始提示字段，也不改指标公式。其他三类测试未重复运行，不写成已完成。

## 5. 主指标结果

各行均直接读取各自 `all_results.json` 的 **`windowauclate_mean`**。

| 方案 | 最佳 / 停止 epoch | 保留检查点验证 AUC | windowauclate_mean |
|---|---:|---:|---:|
| 原始 DKT（复用） | 4 / 14 | 0.830839499 | 0.750830206 |
| 第一轮：未来同模板无提示子集 | 4 / 14 | 0.830093235 | 0.751330708 |
| 第二轮：同模板全结果独立成功代理 | 7 / 17 | 0.831555282 | 0.752619064 |
| 第三轮：同知识点集合跨模板 | 4 / 14 | 0.830184703 | 0.751537305 |
| 第四轮：下一原始事件对照（不参选） | 6 / 16 | 0.831427782 | 0.753807665 |

第二轮相对原 DKT **+0.001788858**，相对下一事件对照 **-0.001188601**。对照测试值更高，但不追溯改选配置。

770 名学生、52378 条逐题窗口预测能精确重建保存的主指标。1000 次学生配对自举，选定方案减 DKT 的95%区间 **[+0.000378360, +0.003223632]**；减下一事件对照的区间 **[-0.002208523, -0.000222575]**。当前固定模型与测试学生中，未来模板方案优于原 DKT，但劣于更简单的下一事件对照；**尚不支持模板/更远未来监督带来额外主指标收益**。

区间只条件于当前固定权重，不覆盖训练 seed/fold 变化，自举正比例不是 p 值。分析未参与选模。见 [baseline 比较](../logs/transfer_iterations_20261026/paired_bootstrap.json)、[下一事件比较](../logs/transfer_iterations_20261026/paired_bootstrap_vs_control.json)。

## 6. 同一验证配对集合上的未来预测

每一列内部使用固定目标集合，只能在同一列内比较；这些是辅助验证诊断，不是官方主指标，也不是模拟完整未来状态轨迹。

| 模型 | 无提示子集 AUC（23999） | 同模板全结果 AUC（26251） | 跨模板 AUC（21672） | 下一事件 AUC（43191） |
|---|---:|---:|---:|---:|
| 原始 DKT（复用） | 0.651439 | 0.748607 | 0.698856 | 0.737171 |
| 第一轮：未来同模板无提示子集 | 0.657247 | 0.745962 | 0.697505 | 0.735903 |
| 第二轮：同模板全结果独立成功代理 | 0.654450 | 0.752151 | 0.697104 | 0.739665 |
| 第三轮：同知识点集合跨模板 | 0.650331 | 0.747721 | 0.705507 | 0.737375 |
| 第四轮：下一原始事件对照（不参选） | 0.655003 | 0.750504 | 0.699420 | 0.741147 |

第三轮跨模板预测从基线0.698856升至0.705507，但下一步主指标没有相应优势。用同一个当前概率同时约束不同未来时间尺度、题目难度和学习过程可能产生冲突，本实验没有识别因果原因，也不强制知识状态单调上升。行为预测改善不自动等于主指标改善。

## 7. 实现、补记与验收

- 模型：[dkt_transfer.py](../pykt/models/dkt_transfer.py)；配对：[transfer_dataset.py](../pykt/datasets/transfer_dataset.py)。
- 训练：[train_dkt_transfer.py](../examples/train_dkt_transfer.py)；主指标评估：[evaluate_dkt_transfer.py](../examples/evaluate_dkt_transfer.py)。
- 92 项相关检查通过：原 DKT state/参数/RNG/Adam 等价，当前概率查询/多知识点均值、未来标签不进入推理、空掩码/非法值、跨块末概念锚点、不同题/模板/知识点集合约束、未来行为仅改标签、训练验证隔离、缓存失效和框架分发。见 [tests.txt](../logs/transfer_iterations_20261026/tests.txt)，不代表全仓测试通过。
- 四份实际检查点与导出由原 DKT 严格加载后预测逐元素一致；旧 DKT、标准加载器、ALSO、kt_config 及先前蒸馏实现均未改。见 [artifact_checks.json](../logs/transfer_iterations_20261026/artifact_checks.json)。
- 前五项19轮已逐项补记并对照 JSON 核验数值和链接，见 [previous_document_audit.json](../logs/transfer_iterations_20261026/previous_document_audit.json)。这些是旧实验补记，不是新训练结果。
- dsh 在120秒内正常交付模型草稿，主 agent 修正设备迁移、原双精度 BCE 等价和 future gather 维度后，完成其余实现和验收；见 [外包记录](../logs/transfer_iterations_20261026/dsh_model/acceptance.md)。

## 8. 复现与剩余限制

从仓库根目录执行，输出目录必须为新目录，默认 fold0 / seed42 /20 epoch /horizon10 /weight0.1：

```bash
python examples/train_dkt_transfer.py --save_dir examples/saved_model/new_transfer/r1
python examples/train_dkt_transfer.py --pair_mode same_all --save_dir examples/saved_model/new_transfer/r2
python examples/train_dkt_transfer.py --pair_mode cross_all --save_dir examples/saved_model/new_transfer/r3
python examples/train_dkt_transfer.py --pair_mode next_event --save_dir examples/saved_model/new_transfer/control
python examples/evaluate_dkt_transfer.py --save_dir examples/saved_model/new_transfer/r2
```

先完成训练/验证并冻结选择，再测试。实现支持 Brier 辅助损失且经过单元检查，但本次四轮都用 BCE，未将 Brier 写成已做实验。

尚无多 fold/seed、不同 horizon/权重的系统验证，未控制未来练习选择与中间学习，代理标签不能等同真实独立能力。当前结果没有超过前一丰富历史蒸馏方案，更简单下一事件对照也优于选定 S05；不将退步轮次隐藏。汇总：[results.json](../logs/transfer_iterations_20261026/results.json)、[results.csv](../logs/transfer_iterations_20261026/results.csv)。
