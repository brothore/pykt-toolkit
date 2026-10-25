# DKT 首步行为结果：三轮闭环实验

日期：2026-10-25。独立分支：`feat/student-state-simulator`。

## 范围与协议

- 独立模型 `dkt_first_action`，继承原 DKT；`DKT.forward(c,r)`、主预测参数名、输入和推理结构保持原样。
- 训练期增加行为头。目标为首次正确、首次尝试错误、先请求提示、先转入脚手架；这些是日志行为，不称为心理学“粗心”标注。
- 本轮不改变原有主任务 BCE 和多知识点评测口径。仅辅助目标按原始 `order_id` 事件选取一次，避免展开重复计数和读入同事件已知回答。
- 数据 sidecar 放在 `data/assist2009/dkt_first_action/`，不重写现有 CSV 或旧模型缓存。
- ASSIST2009 / fold 0 / seed 42，所有对照 batch 32、embedding 200、dropout 0.2、Adam lr=0.001，预算 20 epochs，沿用原验证集选模和 10 epoch 早停。
- 首要指标 `windowauclate_mean`。各轮只依据验证集诊断做下一轮决策，全部训练和选型冻结后才集中评测测试集。
- 旧的 3 epoch 链路检查不作为正式预算对照；重新跑同预算 DKT。

## 数据与安全检查

原始记录首先按当前预处理相同的核心字段非空条件过滤，只为现有训练/验证学生生成标签。按原始事件排序，映射题目、知识点、回答，逐学生核对完整序列，再逐 chunk 校验三个序列完全一致。

同题重做用不同 `order_id` 区分；跨 chunk 的多知识点复制保留事件边界。行为头读取历史状态和目标题知识点嵌入，不读取目标题真实正误或行为。真实行为只在 loss 和离线诊断中使用。

实际用于辅助训练的四类事件数：`[121381, 44051, 13339, 3869]`；验证集为 `[28623, 10401, 3163, 1004]`。这些数量已排除无法作为下一步目标的 chunk 起始位置以及同事件复制。

## 对照

原始 DKT 已完成，最佳 epoch 4，在 epoch 14 早停；验证 AUC `0.8308394991180901`。此时没有评测测试集。

## 第一轮：四分类辅助任务

假设：历史状态能区分四种首步结果，辅助 CE 能提供额外过程监督。

目标：`L = BCE + 0.1 * CE_four_actions`。四分类头仅训练期使用，没有类别重加权。

实现与检查：新增独立模型、数据 wrapper、训练与冻结评测脚本。新增 11 项测试已通过，覆盖参数/RNG 与 DKT 对齐、零辅助权重一致、未来标签不可见、掩码、缺失标签、事件对齐、跨 chunk、多折隔离和 checkpoint 重载。

结果：最佳 epoch 5，在 epoch 15 早停。验证 AUC `0.8319555451`，较对照提高 `0.0011160460`；行为四类 macro-F1 `0.40708`，在真实失败样本内的三类 macro-F1 `0.52260`。脚手架类四分类召回约 `206/1004=20.5%`，三类条件诊断中约 `209/1004=20.8%`。

闭环判断：有小幅验证改善，但四分类会重复主任务正误监督；大量失败样本仍被预测为首次正确，失败类型也偏向“尝试错误”。决定第二轮只监督失败类型，保留主 BCE 不变。该决定发生在任何测试评测之前。

## 第二轮：去除重复的正误辅助监督

候选调整：保留主 BCE，只对首次未正确的样本学习三种失败行为。四类联合概率可以由 `[p_correct, (1-p_correct) * p_failure_type]` 组成。

启动条件与依据：第一轮验证与混淆矩阵已经支持执行。辅助损失改为 `0.1 * CE_three_failure_types`，仅真实失败样本参与，三类各自仍为未加权 CE；不读取第一轮测试指标。

结果：最佳 epoch 4，在 epoch 14 早停；验证 AUC `0.8314261658`，低于第一轮 `0.0005293794`，仍比原始 DKT 高 `0.0005866666`。四类 macro-F1 `0.43473`，失败三类 macro-F1 `0.53537`，比第一轮提高 `0.01277`。失败条件下脚手架召回 `251/1004=25.0%`，仍然偏低。

闭环判断：条件任务改善了行为区分，但未同时改善主验证 AUC。第三轮只引入训练折的平方根逆频率权重，保留第二轮其他设置及 `action_weight=0.1`，检验类别不平衡是否限制了辅助监督。

## 第三轮：针对失败类别不平衡调整

候选调整：依据第二轮验证的失败类别表现，在条件失败任务上采用训练折类别数的平方根逆频率权重。权重只由训练折的实际可监督事件计算。

启动条件与依据：第二轮失败条件混淆矩阵中脚手架召回仍低，且训练事件三类计数为 `[44051,13339,3869]`，因此执行 `action_balance=0.5`。类别权重归一化为均值 1，不读取测试指标。

结果：最佳 epoch 5，在 epoch 15 早停；验证 AUC `0.8323387199`，高于前两轮及对照。失败类型 macro-F1 `0.57195`，条件脚手架召回 `467/1004=46.5%`，但尝试错误类召回有所下降，属于类别权衡。

闭环判断：按验证 AUC 选定第三轮。测试开始前保存 `selection_before_test.json`，记录选择时间及四个 checkpoint 的 SHA-256；之后不改变超参数、选型或已训练 checkpoint。三轮都是“目标调整 → 检查 → 训练/验证 → 分析与决策”，不是把三个 epoch 称为三轮迭代。

## 冻结后的测试结果

| 模型/轮次 | 最佳 epoch | 验证 AUC | 测试 `windowauclate_mean` | 验证失败类型 macro-F1 |
|---|---:|---:|---:|---:|
| 原始 DKT 对照 | 4 | 0.830839 | 0.750830 | 不适用 |
| 第一轮：四分类 CE | 5 | 0.831956 | 0.752021 | 0.522601 |
| 第二轮：条件失败 CE | 4 | 0.831426 | 0.751414 | 0.535373 |
| 第三轮：条件失败 CE＋类别平衡 | 5 | 0.832339 | **0.752310** | 0.571954 |

第三轮在测试前已被验证集选中；其主指标比原始 DKT 提高 `0.0014793979`（约 0.148 个百分点）。这只是单 fold、单 seed 的探索，不构成稳定优越性的统计证据。第二轮没有超过第一轮，不能将迭代描述为每轮所有指标都单调改善。第三轮题目窗口准确率也没有同步超过基线，完整指标保存在各模型 `all_results.json`。

测试评测四组均正常结束，均使用原有 `evaluate_question` / late fusion，未改动主指标公式。各 checkpoint 的哈希在测试后与冻结记录一致。

## 测试说明

首次运行新增测试及既有 ALSO 两组测试：13 passed，1 failed。失败为 `test_also_step_aligns_pi_to_model_device`，期望 `float` 而原 ALSO 返回 Tensor；从 `git show HEAD:ALSO/also.py` 独立加载未修改代码后复现。没有修改 ALSO 或其测试以掩盖此问题。

扩展零辅助训练更新一致性、类别权重、数据初始化路由、纯 DKT 导出及旧缓存不一致拦截检查后，最终检查为 `18 passed`，输出见 `logs/first_action_iterations_20261025/tests.txt`。新增的数据缓存一致性检查只拒绝旧 cache 与当前 CSV 不一致的情况，不重写其他模型的 cache，也未改变本次训练数值。

`dsh` 编写了独立模型初稿；主 agent 完成数据、接入、测试、审阅和迭代。已修复初稿在外部 mask 为真时未显式忽略 `-100` 行为标签的问题，并加回归测试。另一次大范围只读复核请求超时，没有把它算作已通过审查；最终验收由主 agent 完成。

## 复现与推理

从仓库根目录运行。每次使用新的保存目录；脚本拒绝覆盖已有实验，评测拒绝向旧预测文件追加。

```bash
# 原始 DKT 对照
python examples/train_dkt_first_action.py --model_name dkt --num_epochs 20 --save_dir examples/saved_model/new_first_action/baseline
# 第一轮
python examples/train_dkt_first_action.py --action_mode joint --action_weight 0.1 --num_epochs 20 --save_dir examples/saved_model/new_first_action/r1
# 第二轮
python examples/train_dkt_first_action.py --action_mode conditional --action_weight 0.1 --num_epochs 20 --save_dir examples/saved_model/new_first_action/r2
# 第三轮（本次由验证集选定）
python examples/train_dkt_first_action.py --action_mode conditional --action_weight 0.1 --action_balance 0.5 --num_epochs 20 --save_dir examples/saved_model/new_first_action/r3
# 所有选择冻结后，再评测对应 checkpoint
python examples/evaluate_dkt_first_action.py --save_dir examples/saved_model/new_first_action/r3
```

默认数据为 ASSIST2009 / fold 0 / seed 42。当前辅助标签仅支持该数据集，其他模型或数据集不自动接入这一任务。

每个新模型保存 `qid_model.ckpt`（含训练头）以及 `qid_dkt_backbone.ckpt`（仅原始 DKT 参数）。三个模型的导出均由未修改的 `DKT` 类严格加载，并在同设备上检查逐值预测相等；推理不需要原始行为字段。

预算没有改变模型的验证选模和早停：20 是上限，本次实际在 14 或 15 epoch 早停。BCE 输入和普通 DKT 相同，主任务仍以概念 token 计数；辅助任务按原始事件只监督一次。

## 产物

- 训练：`examples/train_dkt_first_action.py`。
- 冻结测试：`examples/evaluate_dkt_first_action.py`，使用原有 `evaluate` / `evaluate_question` 和 late fusion。
- 日志与协议：`logs/first_action_iterations_20261025/`。
- 结果汇总：`logs/first_action_iterations_20261025/results.json` 和 `results.csv`。
- 测试前冻结记录：`logs/first_action_iterations_20261025/selection_before_test.json`。
- 模型、验证诊断、测试指标：`examples/saved_model/first_action_iterations_20261025/`。
