# DKT 丰富历史教师蒸馏：四轮闭环实验

日期：2026-10-26。分支：`feat/student-state-simulator`。方案 S04。独立学生模型：`dkt_distill`。

已完成四轮：丰富历史教师均匀 KL、普通教师蒸馏对照、教师置信度加权、温度 2。按测试前保留检查点的验证 AUC 选定 **第一轮：丰富历史教师，均匀 KL**，`windowauclate_mean` 为 **0.754213086**，比原 DKT **+0.003382880**。相对普通教师蒸馏的额外增益为 **+0.000643811**，其学生配对区间包含 0，暂不能认定丰富历史的独立收益稳定。

## 1. 学生严格只改目标函数

学生直接继承原 DKT 的 `forward(c,r)`，没有新模块、参数或注册 buffer；状态键、张量形状和总参数 **395523** 与原 DKT 完全相同。没有训练期辅助头，测试不需要教师或过程字段。连原始 `qid_model.ckpt` 本身都能由原 DKT 严格加载，另导出相同主干权重；四份真实检查点与导出均验证预测逐元素一致。

教师是另一个模型，在原交互嵌入上加 11 维已完成过程特征的线性投影，再经过 LSTM。教师参数 397723；普通教师对照使用相同结构、种子和训练预算，但过程特征全部置零。额外教师权重不部署到学生。

损失：`L = BCE_original + λ × T² KL(Bernoulli(q_T) || Bernoulli(p_T))`，其中 `p_T=sigmoid(logit(p)/T)`、`q_T=sigmoid(logit(q)/T)`。教师概率 detach；温度稳定夹断保护 logit；硬 BCE 沿用原 concept-expanded masks，KD 仅在原始事件首次位置计算并独立归一化。空掩码返回可微零，λ=0 时原 DKT 的 RNG、loss 和 Adam 更新一致。

## 2. 过程特征和因果位置

复用 ASSIST 原始 `order_id`、完整/分块 q/c/r 核对。每个教师输入位置 t 的特征只来自**同一已完成历史事件 t**，预测 t+1；不读取目标 t+1 的实际耗时、提示、尝试或正误。

11 维为：标准化 log 首反应耗时、耗时有效标志、尝试次数 0/1/2/3+ 的四个 one-hot、尝试有效标志、是否使用提示、标准化 log 提示次数、提示有效标志、是否存在提示机会。时间零/负/超过 1 小时等按既有清洗规则掩码，0 次尝试仍为有效类别，脚手架/无提示机会不能直接当作不求助。

仅对训练/验证学生构造历史特征，存于 `data/assist2009/dkt_distill/history_v1.npz`。标准化只使用教师拟合折中去重的**输入历史事件**，不使用留出学生或外层验证学生的过程字段拟合。教师可读取预测学生已经发生的过程历史，但不能用该学生拟合参数或归一化。

多知识点展开的同一事件可以作为已完成历史输入，但**教师 loss 和 KD 标签跳过同事件概念副本**，防止把已进入历史的目标过程用作监督。真正同题重试仍按不同 order_id 保留。原始 CSV、标准加载器和其他模型输入不改。

## 3. 四折交叉拟合，防止教师记忆训练学生

外层验证 fold 0 全程排除出教师拟合及软标签训练。学生训练折为 1/2/3/4；每次留出其中一折，教师在另三折拟合，给留出学生生成软标签。每折同时训练丰富历史和普通教师，总计 8 个折外教师。

教师固定训练 **6 epoch**、保存最后一轮，事先写入协议。没有用留出学生或外层验证 AUC 选教师检查点。每对教师使用相同种子 `42+heldout_fold`、batch 32、Adam 0.001；历史教师投影的初始化使用 fork RNG，原 DKT 初始化不变。

| 留出折 | 拟合学生数 | 留出学生数 | 丰富教师折外事件 AUC | 普通教师折外事件 AUC |
|---|---:|---:|---:|---:|
| 1 | 1848 | 617 | 0.745577 | 0.736306 |
| 2 | 1849 | 616 | 0.751945 | 0.743552 |
| 3 | 1849 | 616 | 0.751678 | 0.742001 |
| 4 | 1849 | 616 | 0.756937 | 0.745681 |

训练软标签共 **182640** 个去重 next-step 原始事件。归一化、拟合 UID、留出 UID 哈希、概率范围、完整掩码已独立审计：拟合与目标 UID 不相交，外层 fold 0 没有软标签。证据：[crossfit_audit.json](../logs/distill_iterations_20261026/crossfit_audit.json)。

另训练 2 个只使用外层训练学生的固定 6 epoch 教师，**仅作外层验证诊断，不生成学生训练软标签**。同 43191 个验证原始事件上：丰富教师 AUC 0.755994178，普通教师 0.744608675，原 DKT 基线 0.742799343。教师历史信息有预测价值，但学生输入缺少这些信息，不可能保证完整吸收。

教师 AUC 只在原事件首次位置评估；标准学生验证 AUC 包含知识点展开位置，数值口径不同。不能把教师约 0.756 与学生标准约 0.833 直接比较。

## 4. 四轮学生闭环

统一 ASSIST2009 / fold 0 / seed 42 / 最多 20 epoch；embedding/LSTM 200，dropout 0.2，batch 32，Adam 0.001；关闭 ALSO、wandb 和增强。每轮从相同原 DKT 初始化重新训练。保持 `auc > max_auc+1e-3` 的检查点更新和 10 epoch patience。训练/验证序列 3016 / 741。

| 轮次 | 教师 | λ | 温度 | KD 权重 | 决策与证据 |
|---|---|---:|---:|---|---|
| 第一轮 | 丰富历史 OOF | 0.2 | 1 | 均匀 | 验证 AUC 与原事件 AUC 都高于 DKT，下一轮检查普通蒸馏对照 |
| 第二轮 | 仅答题历史 OOF | 0.2 | 1 | 均匀 | 诊断对照，训练前声明不参与丰富历史方案选模；其收益说明普通蒸馏也起作用 |
| 第三轮 | 丰富历史 OOF | 0.2 | 1 | `2 abs(q-0.5)`，按实际权重和归一化 | 测试排除近 0.5 的无区分度软标签；验证下降，BCE/KL 梯度更相斥，不支持更信任确定预测 |
| 第四轮 | 丰富历史 OOF | 0.2 | 2 | 均匀 | 回到验证最好的第一轮，只改温度，保留 T²；验证仍未超过第一轮 |

第三轮的 confidence 是**教师预测的确定性**，不是 NIPS 自报 Confidence，也没有使用真实目标答案来筛选“正确教师”。T² 温度试验不等同于简单减小梯度强度。

各轮决定及时间见 [protocol.json](../logs/distill_iterations_20261026/protocol.json)。外层验证仅用于学生检查点/方案选择及诊断，不用于教师拟合或教师检查点选择。

## 5. 冻结与基线

测试前冻结时间 `2026-10-26T13:18:22.329224+00:00`；候选包括原 DKT 和丰富历史学生，普通教师对照按预先声明排除。选定 `r1_rich`。检查点、OOF 数组和元数据 SHA256 见 [selection_before_test.json](../logs/distill_iterations_20261026/selection_before_test.json)。评估后均保持不变。

复用同设置原 DKT 基线及已有 `all_results.json`，未声称本次重训/重测基线。基线检查点 SHA 为 `8094b98fa8047f51b2478d4e5794a1c417b8ad8eab0fc527b122e058e944475a`。训练原始来源和逐题窗口 CSV 哈希均与之前一致，证据见 [baseline_data_checks.json](../logs/distill_iterations_20261026/baseline_data_checks.json)。

新学生只跑主指标所需的原逐题窗口 late fusion，batch **128** 与旧基线一致，不加载任何教师/过程输入；原融合公式不改。本次未重复普通测试等其他三项，不把它们写成已完成。

## 6. 结果

主指标逐一取自各实验 `all_results.json` 的 **`windowauclate_mean`**。原事件验证 AUC 是另一诊断口径，不替代主指标。

| 方案 | 最佳 / 停止 epoch | 标准验证 AUC | 原事件验证 AUC | windowauclate_mean |
|---|---:|---:|---:|---:|
| 原始 DKT（复用） | 4 / 14 | 0.830839499 | 0.742799343 | 0.750830206 |
| 第一轮：丰富历史教师，均匀 KL | 6 / 16 | 0.833164464 | 0.746238270 | 0.754213086 |
| 第二轮：普通教师对照（不参选） | 5 / 15 | 0.832471826 | 0.745257463 | 0.753569275 |
| 第三轮：教师置信度加权 | 4 / 14 | 0.831789718 | 0.744293238 | 0.752578511 |
| 第四轮：温度 2 | 4 / 14 | 0.832349310 | 0.745144232 | 0.753133564 |

选定第一轮相对原 DKT **+0.003382880**（+0.3383 个百分点），相对普通教师学生 **+0.000643811**。不根据测试结果改选。

在 770 名测试学生、52378 条逐题窗口预测上，文件重建 AUC 与官方字段精确一致。1000 次学生配对自举的 AUC 差 95% 百分位区间：相对 DKT **[+0.002239533, +0.004541938]**；相对普通教师对照 **[-0.000044235, +0.001389173]**。前者在 0 以上，后者包含 0。

因此当前固定模型/测试学生上能观察到相对原 DKT 的小幅收益；**丰富过程历史较普通教师的独立额外收益尚未得到稳健支持**。自举不包含训练 seed/fold 随机性，不能推广成普遍提升；自举正比例不是 p 值，且未用于选模。见 [baseline 比较](../logs/distill_iterations_20261026/paired_bootstrap.json)、[普通教师比较](../logs/distill_iterations_20261026/paired_bootstrap_vs_control.json)。

## 7. 实现与验收

- 模型：[dkt_distill.py](../pykt/models/dkt_distill.py)；数据：[distill_dataset.py](../pykt/datasets/distill_dataset.py)。
- 教师交叉拟合：[build_dkt_distill_teachers.py](../examples/build_dkt_distill_teachers.py)；学生训练：[train_dkt_distill.py](../examples/train_dkt_distill.py)；主指标评估：[evaluate_dkt_distill.py](../examples/evaluate_dkt_distill.py)。
- 80 项相关检查通过，包括原 DKT state/RNG/Adam 等价、教师 detach、温度 KL、空掩码与非法概率、输入历史而非目标过程、未来输入因果性、拟合折归一化、OOF 标签来源、严格导出和框架分发。见 [tests.txt](../logs/distill_iterations_20261026/tests.txt)，不是全仓测试通过声明。
- 四份实际学生原检查点和导出均严格加载为原 DKT、参数相同、预测逐元素一致。原 DKT、标准加载器、ALSO、kt_config 与此前错误选项模型相关文件均未修改。见 [artifact_checks.json](../logs/distill_iterations_20261026/artifact_checks.json)。
- dsh 本次 180 秒超时且未交付文件；主 agent 完成全部实现与验收，没有以 worker 完成声明替代测试。见 [外包记录](../logs/distill_iterations_20261026/dsh_model/acceptance.md)。

额外训练成本：8 个交叉拟合教师加 2 个只作诊断的全外层训练教师，各 6 epoch；教师拟合、诊断和软标签生成计时约 110.3 秒（不含首次原始侧车构建），另有四次学生训练。推理时不运行教师。固定预算教师并非经过充分调优的最优教师；结论仍受单数据集/fold/seed 限制。

未额外运行“仅对原始事件加权硬 BCE”的消融，因此普通教师对照相对基线的收益也可能包含软目标平滑与原始事件加权效果，不能全解释为知识迁移。

## 8. 复现

从仓库根目录执行，输出目录须为新目录。先构建教师并完成学生训练/验证，冻结选择后再测试：

```bash
python examples/build_dkt_distill_teachers.py --save_dir examples/saved_model/new_distill/teachers
python examples/train_dkt_distill.py --soft_targets examples/saved_model/new_distill/teachers/soft_targets.npz --save_dir examples/saved_model/new_distill/r1
python examples/train_dkt_distill.py --soft_targets examples/saved_model/new_distill/teachers/soft_targets.npz --teacher_source thin --save_dir examples/saved_model/new_distill/control
python examples/train_dkt_distill.py --soft_targets examples/saved_model/new_distill/teachers/soft_targets.npz --kd_mode confidence --save_dir examples/saved_model/new_distill/r3
python examples/train_dkt_distill.py --soft_targets examples/saved_model/new_distill/teachers/soft_targets.npz --kd_temperature 2 --save_dir examples/saved_model/new_distill/r4
python examples/evaluate_dkt_distill.py --save_dir examples/saved_model/new_distill/r1
```

默认即 fold 0 / seed 42 / 学生 20 epoch；当前教师构建器固定外层 fold 0，不能将其软标签用于其他外层 fold。数据哈希变化会拒绝陈旧软标签。完成情况：[实验进度](学生模拟_实验进度.md)；汇总：[results.json](../logs/distill_iterations_20261026/results.json)、[results.csv](../logs/distill_iterations_20261026/results.csv)。
