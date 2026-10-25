## 0. 执行限制（请先读）

本 worker 的 shell 在本机不可用：所有 `bash` 调用都返回 `sandbox mode "workspace-write" is requested but no sandbox backend is usable`；改用 `danger-full-access` 又因没有审批通道而被拒绝。因此**我无法运行 pandas**。下文的交互数、人数、题数取自项目已有的预处理日志与代码，可以复核；但**字段非空率、分位数、order_id 唯一性，以及 correct 与行为字段的交叉表均未计算**，我不编造。本次没有修改任何文件，没有联网，也没有调用子代理。

## 1. 行数与过滤链（已核验）

| 阶段 | 交互 | 学生 | 题目 | 技能 |
|---|---|---|---|---|
| 原始 CSV | 346,860 | 4,217 | 26,688 | 123 |
| dropna 后 | 283,105 | 4,163 | 17,751 | 123 |
| 删短序列学生后 | 282,619 | 3,852 | — | — |
| 多知识点展开后 | 337,424 | 3,852 | 17,737 | 123 |

依据：`logs/preprocess_20261025_112142/assist2009.log` L4–L8、L13；`pykt/preprocess/assist2009_preprocess.py` L17；`assist2009.config.json` 的 `min_seq_len=3`、`max_concepts=4`。文件实际为 346,861 行（含表头），与 346,860 条一致。

- 当前过滤条件是：用 `dropna(subset=["user_id","problem_id","skill_id","correct","order_id"])` 丢掉 **63,755 条（18.38%）**，学生减少 54 人。
- 题目数 26,688→17,751，恰好减少 **8,937**，与日志中“无技能题数 na=8,937”完全相等，因此丢弃几乎全部来自**缺 skill 映射的题**，而不是 correct 缺失。
- 再删除 311 名短序列学生（len<3）及其 **486** 条 → **282,619 条 / 3,852 人**，人均 73.4 条；这是“当前有效作答”的基准。
- `split_datasets.py` 的 `extend_multi_concepts` 按 `_` 拆分多技能串（如 2–4 个技能），再增加 **54,805 个 token（+19.39%）**；train+valid 274,614 + test 62,810 = 337,424，随后丢掉 7+2 条 → 337,415。训练实际使用的是这个概念级 token 表。
- 下游只保留 `uid/questions/concepts/responses`；**`ms_first_response` 被丢弃**（preprocess L32–33 把时间戳与用时写成 `'NA'`）。`order_id` 只用于排序，之后也丢弃。

## 2. 字段可得性（按表结构和代码核验，不是统计）

**(a) 目前保留**：`user_id`、`problem_id`、`skill_id`、`correct`（→responses）。

**(b) 作答前可知、但当前管线丢弃**：`template_id`（题模板）、`answer_type`、`tutor_mode`、`original`（原题/脚手架）、`hint_total`（该题可用提示总数，属题面静态属性）、`opportunity` / `opportunity_original`（该技能第几次练习）、assignment/sequence/class/teacher/school id。

**(c) 只能作答后产生，或只能作训练目标**：`correct`、`attempt_count`、`ms_first_response`、`hint_count`、`bottom_hint`、`first_action`、`answer_id`、`answer_text`。

非空率我没有计算。仅从 5 行样本可确认 `answer_id`、`bottom_hint` 会出现字面 `"NA"`；`pd.read_csv(dtype=str)` 默认把 `"NA"` 解析为 NaN，这正是 dropna 能生效的机制。`answer_text` 是自由文本，只应统计长度/类型，不能展示内容。

## 3. order_id 与冲突

- 原始 `order_id` 是否唯一：**未统计，不能断言**。
- 可以确定的是：展开后同一 `order_id` 会出现 2–4 次（`extend_multi_concepts` 对非 concepts 字段用 `[value]*len(ids)` 复制），所以展开表内的“冲突”只是复制，不是真冲突。
- 真正的行为冲突只可能来自原始重复行，需用 `df.duplicated('order_id').sum()` 及组内 `nunique()` 核验。**若按 order_id 去重，样本量与技能分布都会变，必须与主 agent 的评测口径对齐。**

## 4. correct 与行为字段（结构性判断，仍待数值核验）

- `attempt_count`：首答即对时 `attempt_count=1`；但“答错一次后放弃”可产生 `attempt_count=1 & correct=0`。因此**不能假定近似恒等**，必须看交叉表。
- `first_action`：样本中全为 0，语义应为“首次动作是尝试还是求助”，属作答过程中、判定前的信息，与 correct 并非恒等。
- `hint_count`/`bottom_hint`：求助行为，预期与 correct 负相关；`bottom_hint` 表示看到底提示，接近“答案已揭示”，是最接近标签泄漏的字段。
- `correct` 的官方语义（对错/首答对/掌握）请主 agent 另行核对；本文只把它当作该次作答的对错标记。

## 5. 可构造的监督目标（7 个）

| # | 目标 | 范围 | 输入时点 | 泄漏/混淆风险 |
|---|---|---|---|---|
| 1 | correct 的 BCE 重加权/校准正则（按 opportunity 分层、按技能频率） | A 只改 loss | 仅训练期权重 | opportunity 与序列位置强相关，可能放大“位置捷径”并改变 base rate |
| 2 | 同学生同技能相邻两次作答的 pair ranking（后次更可能对） | A 只改 loss | 仅训练期成对 | 多技能展开会产生同一 order_id 的重复 token，造成“自己比自己” |
| 3 | attempt_count 分档（1/2/≥3）辅助头 | B 辅助头 | 作答后标签 | 与 correct 强耦合，头可能退化为 correct 副本；预测值不得回灌推理 |
| 4 | hint 使用（hint_count>0 或 bottom_hint）辅助头 | B 辅助头 | 作答后标签 | hint_count 是作答后行为；若同时当输入即泄漏，且会把“求助倾向”误当能力 |
| 5 | first_action（0 尝试/1 求助）辅助头 | B 辅助头 | 开始作答后、判定前 | 该字段推理时不可见，只能作目标；与 hint_count 共享方差 |
| 6 | log(1+ms_first_response) 分桶（快/中/慢）辅助头 | B 辅助头 | 作答后标签 | 重尾 + 题目/模板难度混淆；跨学生同题时延统计若含当前记录即泄漏 |
| 7 | 推理输入加入作答前静态特征：template_id/answer_type/tutor_mode/original/opportunity/hint_total，以及严格 t'<t 的历史正确率、历史 hint 率、历史中位时延 | C 改推理输入 | 全部作答前可得 | 历史窗口若含当前题或同 batch 未来题即时间泄漏；template_id 可能编码答案键 |

补充：方案 3–6 的辅助头默认只在训练期使用，推理端不接，评估仍走现有 `windowauclate_mean` 流程；凡使用当前行 `attempt_count/hint_count/first_action/ms_first_response` 作为输入，都属于直接泄漏。

## 6. 还缺的数值（建议主 agent 一条命令补齐）

对 282,619 基准表计算：`order_id` 的 `nunique/duplicated` 与组内 `nunique`；各字段非空率；`attempt_count/hint_count/ms_first_response` 的 P50/P90/P99；`first_action/bottom_hint/original/template_id` 频数；`correct × first_action`、`correct × (attempt_count==1)`、`correct × (hint_count>0)` 的交叉表；以及多技能行的 `skill_id.str.count('_')` 分布（预期与 54,805 增量吻合）。这些数字才能最终判定“近似恒等标签”和重复 order_id 冲突。