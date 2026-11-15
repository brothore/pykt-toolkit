# UKT如何加入双GRU：已做方案、完整结果与新增实验

截至2026-11-14，本轮复盘确认：**已完成8种结合方式、24轮新训练；尚无同时满足均衡和性能保护的联合候选。现在新增4个方向（1个对照＋3个优化方向），最多10轮。** 以下所有baseline都是两个独立GRU，不复用旧Mamba权重。

本文可以独立阅读：先解释结合方式和依据，再给出V1/V2全部轮次的绝对数值、同表baseline、退化与不确定性，最后记录V3的设计、队列与执行证据。每张结果表的加粗仅表示该列最佳，不表示整个方案有效。

## 指标与比较条件

数据集nips_task34，fold0、seed42；B0训练27 epoch、选中17 epoch。标准差ddof=0。窗口预测共221348条交互；序列表现主子集为满足有效交互≥20且少数类≥5的773条orirow。pooled衡量所有交互的排序，macro是每条orirow AUC等权平均；std/range衡量序列间差异。自身最差10/20%与baseline固定弱10/20%都报告；原强25%固定由baseline定义，防止通过压低强项得到“均衡”。退化数是这773条中相对B0下降的数量。

**本文件是开发验证结果；正式测试字段`all_results.json.windowauclate_mean`缺失，不能用这里的验证pooled替代。** 原UKT的概念展开验证、用户报告的967条正式测试序列与这里773条开发序列均不能直接相减。每轮实际/选中epoch见表。

普通联合候选要求std/range同时下降且pooled/macro、固定弱组、自身尾组和原强组均值保护通过；实质开发信号还要求std/range各相对下降≥1%、配对std差值95%区间上界<0。这些均不等于跨seed稳定或未来逐序列绝不退化。

## 八种已有结合究竟在改什么

R0–R3是在原双GRU预测旁边增加历史读出：根据当前题/KC检索过去交互，再有界修正最终logit；保留双GRU并一起训练。R0用普通均值距离，R1才加入正协方差与Wasserstein距离，R2加入不确定性收缩，R3加入原UKT的乘性距离调制。它们不是直接把完整UKT换成GRU。

S0/S1把UKT式Rasch表示迁到原输入和当前解码：题目难度因子与共享知识点方向相乘，让同一KC遇到不同题目时拥有不同表示；S0的难度为标量，S1为向量。S2把它改成预测端的因子修正并冻结原GRU；S3冻结原GRU，只训练R1式W2历史读出，检验端到端干扰是否是主要问题。八种方案各完成3轮；具体超参数与结果在下面原始完整表中同行列出。

## 复盘判断：观察、可能解释和未证明的部分

1. **历史检索还没有表现出可迁移的联合收益。** 普通距离到W2、加入协方差收缩/距离调制的V1都未达标。冻结GRU的S3也没有解决保护问题，因此不能仅归因于主干被训练破坏。可能存在读出冗余、目标不对齐或融合不当；现有实验不能证明W2在任何位置都无效。
2. **向量Rasch出现较一致的补弱线索，但幅度小且仍伤害强项。** S1第3轮std从.0854331310412降到.084944477406，range从.579344858963降到.577040711497，固定弱10%提高.001028940601；但固定强25%下降.000447717014，364/773条序列下降。不能称有效。S0/S1容量不同，不能将差异纯归因于向量化。
3. **最小std/range不代表达标。** S3第3轮在V2的这两列最优，却使pooled从.796528100282降到.793939178249，原强组也下降。冻结参数只能保证权重不动，附加分支仍会改变排序。
4. **原UKT的推理敏感性不能直接当迁移收益或训练因果。** 当前UKT配置use_CL=0；去Rasch还同时去掉其题目ID入口。原模型W2/协方差/alpha²的敏感性只说明其现有权重依赖这些计算，不能证明在已经含题目embedding的QIKT里直接叠加也有效。
5. **下一步应拆开位置、交互形式和保护约束。** 不再无界扫描已结束的八个方向。新增加性控制检验乘法的必要性；只改当前解码检验历史状态干扰；从B0继续训练并约束原概率/已有排序检验性能保护。新的组合仍可能失败，结果出来后按相同指标判断。

## V1：12轮完整结果


本批已终止等待：共取得13条完成结果，优化点候选0条，实质开发信号0条。任务失败/停止会明确保留，不能算完成实验。

nips_task34／QIKT双独立GRU／fold0／seed42。全新B0，不复用Mamba权重。原生开发窗口、主子集至少20位置/少数类5，std ddof0；正式windowauclate_mean未评估，实际训练/选中epoch逐行给出。

R0均值距离检索对照，R1学习正协方差的Wasserstein历史读出，R2按检索协方差收缩融合，R3检验UKT原有乘性距离调制。保留两GRU，严格过去响应，零初始化有界补充；scale .1/.25/.5、各最多3轮，普通候选即停。

测试序列报告只作已披露事后分析，不能证明UKT模块因果有效。协方差学习增加有效参数；端到端训练、早停和单seed混杂仍在。14项CPU检查和真实CUDA预检/新负载资源实测为前置，不能把预检或并发推荐当效力。

### 完整主指标

每列最佳数值加粗；某列最优不等于联合目标通过。退化数相对原baseline计算。

| 方案 | 改动 | 实际/选中epoch | pooled AUC ↑ | macro AUC ↑ | std ↓ | range ↓ | 退化序列数 ↓ | 点候选 | 实质开发信号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 27/17 | **0.796528100282** | 0.725051126341 | 0.0854331310412 | 0.579344858963 | **0** | 参照 | 参照 |
| B0/round1 | 全新双独立GRU共同参照，未加载Mamba权重；version=gru | 27/17 | **0.796528100282** | 0.725051126341 | 0.0854331310412 | 0.579344858963 | **0** | 控制，非优化候选 | False |
| R0/round1 | 均值距离相关历史检索，固定单位协方差，附加修正；version=gru, ukt_transfer_mode=mean, ukt_transfer_scale=0.1, ukt_transfer_fusion=additive | 27/17 | 0.796311798496 | 0.724950023711 | 0.0853903614117 | 0.576119052511 | 391 | False | False |
| R0/round2 | 均值距离相关历史检索，固定单位协方差，附加修正；version=gru, ukt_transfer_mode=mean, ukt_transfer_scale=0.25, ukt_transfer_fusion=additive | 27/17 | 0.795695459186 | 0.724225031387 | 0.0858404726575 | 0.577046581937 | 447 | False | False |
| R0/round3 | 均值距离相关历史检索，固定单位协方差，附加修正；version=gru, ukt_transfer_mode=mean, ukt_transfer_scale=0.5, ukt_transfer_fusion=additive | 28/18 | 0.794591359091 | 0.723296160594 | 0.0859913138366 | 0.573371686871 | 459 | False | False |
| R1/round1 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.1, ukt_transfer_fusion=additive | 28/18 | 0.796428779619 | **0.725096414464** | 0.0855987103698 | 0.57014000998 | 367 | False | False |
| R1/round2 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.25, ukt_transfer_fusion=additive | 23/13 | 0.795124549276 | 0.723448807332 | 0.0862808149127 | 0.575664093457 | 485 | False | False |
| R1/round3 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.5, ukt_transfer_fusion=additive | 21/11 | 0.793643012494 | 0.722363340325 | 0.0865929734393 | **0.560929290557** | 489 | False | False |
| R2/round1 | R1＋按检索协方差收缩融合修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.1, ukt_transfer_fusion=uncertainty | 27/17 | 0.796345997512 | 0.724959267182 | 0.08539349429 | 0.576585752444 | 393 | False | False |
| R2/round2 | R1＋按检索协方差收缩融合修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.25, ukt_transfer_fusion=uncertainty | 23/13 | 0.795317587825 | 0.723898505188 | 0.0861240520549 | 0.571971587073 | 458 | False | False |
| R2/round3 | R1＋按检索协方差收缩融合修正；version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.5, ukt_transfer_fusion=uncertainty | 25/15 | 0.79448534612 | 0.723364189785 | 0.0858135369326 | 0.569212480554 | 450 | False | False |
| R3/round1 | R1＋UKT原有乘性距离调制；version=gru, ukt_transfer_mode=w2_legacy, ukt_transfer_scale=0.1, ukt_transfer_fusion=additive | 27/17 | 0.796370411301 | 0.725020178359 | 0.0853998445379 | 0.577046581937 | 389 | False | False |
| R3/round2 | R1＋UKT原有乘性距离调制；version=gru, ukt_transfer_mode=w2_legacy, ukt_transfer_scale=0.25, ukt_transfer_fusion=additive | 27/17 | 0.795923830707 | 0.724720412773 | **0.0853832631184** | 0.570606709912 | 430 | False | False |
| R3/round3 | R1＋UKT原有乘性距离调制；version=gru, ukt_transfer_mode=w2_legacy, ukt_transfer_scale=0.5, ukt_transfer_fusion=additive | 27/17 | 0.795108219515 | 0.724097964501 | 0.0853989392794 | 0.564615926502 | 467 | False | False |

### 尾部、原有强项与分布宽度

固定弱/强组由原baseline主子集AUC稳定排序确定；自身尾部允许成员变化，两者不能混用。

| 方案 | 改动 | 自身尾10% ↑ | 自身尾20% ↑ | 固定弱10% ↑ | 固定弱20% ↑ | 固定强25% ↑ | P95−P05 ↓ | 去掉两端极差 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 0.553662388924 | 0.59529908901 | 0.553662388924 | 0.59529908901 | 0.823195670194 | 0.287966917197 | 0.544364202284 |
| B0/round1 | 全新双独立GRU共同参照，未加载Mamba权重 | 0.553662388924 | 0.59529908901 | 0.553662388924 | 0.59529908901 | 0.823195670194 | 0.287966917197 | 0.544364202284 |
| R0/round1 | 均值距离相关历史检索，固定单位协方差，附加修正 | 0.55348098792 | 0.595070992432 | 0.55348098792 | 0.595124981888 | 0.82292681773 | 0.289048104381 | 0.540208675004 |
| R0/round2 | 均值距离相关历史检索，固定单位协方差，附加修正 | 0.552508442065 | 0.59360096915 | 0.552745201451 | 0.593745531881 | 0.822480716438 | 0.291973171837 | 0.537410978516 |
| R0/round3 | 均值距离相关历史检索，固定单位协方差，附加修正 | 0.552022241706 | 0.592259155388 | 0.552329515531 | 0.592844194829 | 0.821763198966 | 0.292977897213 | 0.546577446517 |
| R1/round1 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.553387544254 | 0.594766144905 | 0.553396478547 | 0.594860457911 | **0.823229096438** | 0.289268129884 | **0.534063205679** |
| R1/round2 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.550481445927 | 0.592082191567 | 0.551000145316 | 0.592360085117 | 0.822276994366 | **0.283399915302** | 0.537223761937 |
| R1/round3 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.54983311797 | 0.59056252016 | 0.550419944545 | 0.591357280584 | 0.82171638124 | 0.286006558559 | 0.54557114473 |
| R2/round1 | R1＋按检索协方差收缩融合修正 | 0.55324666071 | 0.595182449587 | 0.553310817981 | 0.595222712839 | 0.822992380624 | 0.289543388769 | 0.543099979781 |
| R2/round2 | R1＋按检索协方差收缩融合修正 | 0.551220420115 | 0.592831054742 | 0.551447121433 | 0.59306419459 | 0.822683752059 | 0.287860189133 | 0.541981813286 |
| R2/round3 | R1＋按检索协方差收缩融合修正 | 0.551749628753 | 0.592735438148 | 0.552307536287 | 0.593203160235 | 0.821913376073 | 0.289174104 | 0.536030743624 |
| R3/round1 | R1＋UKT原有乘性距离调制 | **0.55369701181** | **0.595402783154** | 0.553733887079 | **0.595424049556** | 0.823105411345 | 0.287511014677 | 0.549826822343 |
| R3/round2 | R1＋UKT原有乘性距离调制 | 0.553522725494 | 0.594902229639 | 0.553734529957 | 0.595004978868 | 0.822688520554 | 0.28782003112 | 0.547932882949 |
| R3/round3 | R1＋UKT原有乘性距离调制 | 0.553545796883 | 0.594502303661 | **0.554045260367** | 0.594805051144 | 0.822137622631 | 0.284752229518 | 0.562007392177 |

### 不确定性与退化

下列区间是相对原baseline的orirow配对重采样95%区间，不能代表独立学生或跨seed稳定性；极差受两个端点影响。

| 方案 | 改动 | Δstd ↓ | Δrange ↓ | Δstd区间 | 下降>.001数 ↓ | 下降>.005数 ↓ | 下降>.01数 ↓ | 最差序列增益 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 0 | 0 | [0, 0] | **0** | **0** | **0** | **0** |
| B0/round1 | 全新双独立GRU共同参照，未加载Mamba权重 | 0 | 0 | [0.0, 0.0] | **0** | **0** | **0** | **0** |
| R0/round1 | 均值距离相关历史检索，固定单位协方差，附加修正 | -4.27696294524e-05 | -0.00322580645161 | [-0.00024767269882114174, 0.000213937162472495] | 218 | 25 | 3 | -0.0185185185185 |
| R0/round2 | 均值距离相关历史检索，固定单位协方差，附加修正 | 0.000407341616354 | -0.00229827702604 | [-9.118763391695733e-05, 0.0010039195199301701] | 354 | 114 | 22 | -0.0525451559934 |
| R0/round3 | 均值距离相关历史检索，固定单位协方差，附加修正 | 0.000558182795382 | -0.00597317209193 | [-0.00012923115122112706, 0.0014440606507179535] | 405 | 213 | 77 | -0.0583333333333 |
| R1/round1 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.000165579328625 | -0.00920484898295 | [-0.00016365799117928418, 0.0005232865114438649] | 257 | 52 | 11 | -0.0218855218855 |
| R1/round2 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.000847683871504 | -0.0036807655053 | [0.00021373211635506846, 0.0015210947652833074] | 422 | 176 | 57 | -0.0467980295567 |
| R1/round3 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | 0.00115984239815 | **-0.0184155684053** | [0.0003586756125460413, 0.00208865135560005] | 450 | 261 | 111 | -0.0730706075534 |
| R2/round1 | R1＋按检索协方差收缩融合修正 | -3.96367511949e-05 | -0.00275910651912 | [-0.00026226119306282795, 0.0002048649602765823] | 215 | 22 | 1 | -0.025 |
| R2/round2 | R1＋按检索协方差收缩融合修正 | 0.000690921013713 | -0.0073732718894 | [0.00014993593632635257, 0.0012194799887027242] | 375 | 146 | 41 | -0.0361111111111 |
| R2/round3 | R1＋按检索协方差收缩融合修正 | 0.000380405891413 | -0.0101323784085 | [-0.00016196719272711205, 0.000990768102556108] | 390 | 203 | 71 | -0.0458074534161 |
| R3/round1 | R1＋UKT原有乘性距离调制 | -3.32865032597e-05 | -0.00229827702604 | [-0.00026097823746145526, 0.00026175067147878115] | 204 | 16 | 3 | -0.023569023569 |
| R3/round2 | R1＋UKT原有乘性距离调制 | **-4.98679227898e-05** | -0.00873814905046 | [-0.00038721081058096067, 0.0003643940813510229] | 322 | 58 | 13 | -0.030303030303 |
| R3/round3 | R1＋UKT原有乘性距离调制 | -3.41917618132e-05 | -0.0147289324606 | [-0.0005065637860471459, 0.000522839689549322] | 403 | 134 | 43 | -0.0319865319865 |

### 如何判断本批的证据

- 标准差下降需要弱序列增益与原AUC的负协方差足够大，抵消增益自身方差；逐轮两项分解保存在audit.json，恒等式已核对。极差还要求最差端点获益，不能用平均尾部改善替代。
先比较R0/R1，区分普通历史检索与学习分布的增量，再比较R1/R2/R3判断融合/调制。UKT实际use_CL=0，不能归功于未用的对比学习。
- 实质门槛为std与range各相对下降至少1%、std区间上界<0，并保住pooled/macro、固定弱10%/20%、自身尾10%/20%及固定强25%均值。即使通过，逐序列无退化和正式测试仍需另外验证。
候选记录暂停，失效方向最多3轮；基于证据再决定UKT训练消融、Rasch或CL迁移，不自动追加seed/Peiyou。

### 任务、参数和证据入口

| 方向 | 改动 | 任务状态 |
| --- | --- | --- |
| B0 | 全新双独立GRU共同参照，未加载Mamba权重 | completed |
| R0 | 均值距离相关历史检索，固定单位协方差，附加修正 | completed |
| R1 | UKT对角Wasserstein检索，alpha²协方差传播，附加修正 | completed |
| R2 | R1＋按检索协方差收缩融合修正 | completed |
| R3 | R1＋UKT原有乘性距离调制 | completed |

- B0/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru"}}

- R0/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "mean", "ukt_transfer_scale": 0.1, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R0", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "d74e19317a32edd8fb3359a2a2064311c36b6d70643b75dc2baee2c1d813e6d3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R0/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "mean", "ukt_transfer_scale": 0.25, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R0", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "d74e19317a32edd8fb3359a2a2064311c36b6d70643b75dc2baee2c1d813e6d3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R0/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R0/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "mean", "ukt_transfer_scale": 0.5, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R0", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "d74e19317a32edd8fb3359a2a2064311c36b6d70643b75dc2baee2c1d813e6d3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R1/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.1, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R1", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ccd7520641f5bf9337e6a4ac746e714fccb86018231d29bdd9f5e414049f29eb", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R1/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.25, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R1", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ccd7520641f5bf9337e6a4ac746e714fccb86018231d29bdd9f5e414049f29eb", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R1/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R1/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.5, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R1", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ccd7520641f5bf9337e6a4ac746e714fccb86018231d29bdd9f5e414049f29eb", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R2/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.1, "ukt_transfer_fusion": "uncertainty"}, "family": "GRU-UKT-R2", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "e8cb78ed99a0f415b3be62050c8b1de1e8995bbcd703ab8b138aa0edc040e34c", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R2/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.25, "ukt_transfer_fusion": "uncertainty"}, "family": "GRU-UKT-R2", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "e8cb78ed99a0f415b3be62050c8b1de1e8995bbcd703ab8b138aa0edc040e34c", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R2/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R2/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2", "ukt_transfer_scale": 0.5, "ukt_transfer_fusion": "uncertainty"}, "family": "GRU-UKT-R2", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "e8cb78ed99a0f415b3be62050c8b1de1e8995bbcd703ab8b138aa0edc040e34c", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R3/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2_legacy", "ukt_transfer_scale": 0.1, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R3", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ead3df10812fabf1e6ab7720bff4d71751a109c521485b7593134553129c9eb0", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R3/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2_legacy", "ukt_transfer_scale": 0.25, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R3", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ead3df10812fabf1e6ab7720bff4d71751a109c521485b7593134553129c9eb0", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- R3/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/R3/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "ukt_transfer_mode": "w2_legacy", "ukt_transfer_scale": 0.5, "ukt_transfer_fusion": "additive"}, "family": "GRU-UKT-R3", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "ead3df10812fabf1e6ab7720bff4d71751a109c521485b7593134553129c9eb0", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

核对2529个独立冻结来源/输入/产物，逐轮数值与退化数、方差恒等式均核验；[审计及分解](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/completed_review/audit.json)。仅分析已有预测，不新增训练、推理或测试标签开发。

## V2：12轮完整结果


本批已终止等待：共取得13条完成结果，优化点候选0条，实质开发信号0条。任务失败/停止会明确保留，不能算完成实验。

nips_task34／QIKT两个独立GRU／fold0／seed42，primary773，std ddof0。复用B0训练27/选中17，不复用Mamba权重。正式windowauclate_mean缺失，均为开发窗口结果。

S0标量/S1向量题目difficulty调制共享KC方向，进入两GRU的历史与当前解码上下文；S2改为冻结主干的有界输出Rasch因子；S3冻结主干，仅训练W2读出。初始新增修正为0，各最多3轮、普通候选即停。

UKT推理干预敏感性不等于训练消融。输入/输出放置还包含冻结和lr变化，标量/向量有容量差异，不能做单因素因果归因。真实CUDA预检和匹配原生flags的资源探针是前置，不冒称稳定有效。

### 完整主指标

每列最佳数值加粗；某列最优不等于联合目标通过。退化数相对原baseline计算。

| 方案 | 改动 | 实际/选中epoch | pooled AUC ↑ | macro AUC ↑ | std ↓ | range ↓ | 退化序列数 ↓ | 点候选 | 实质开发信号 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 27/17 | 0.796528100282 | 0.725051126341 | 0.0854331310412 | 0.579344858963 | **0** | 参照 | 参照 |
| B0/round1 | 复用V1双独立GRU基线27/17 epoch；version=gru | 27/17 | 0.796528100282 | 0.725051126341 | 0.0854331310412 | 0.579344858963 | **0** | 控制，非优化候选 | False |
| S0/round1 | UKT标量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_scalar, ukt_rasch_scale=1.0 | 27/17 | 0.796551558665 | 0.72508072609 | 0.0853583348146 | 0.578423199977 | 345 | False | False |
| S0/round2 | UKT标量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_scalar, ukt_rasch_scale=0.5 | 27/17 | 0.796531055508 | 0.725061577097 | 0.0853685852426 | 0.580727347442 | 347 | False | False |
| S0/round3 | UKT标量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_scalar, ukt_rasch_scale=2.0 | 27/17 | **0.796581075683** | 0.725144455082 | 0.0853664632244 | 0.580727347442 | 345 | False | False |
| S1/round1 | UKT向量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_vector, ukt_rasch_scale=1.0 | 27/17 | 0.796559655068 | 0.725186850298 | 0.0852390029698 | 0.579344858963 | 338 | False | False |
| S1/round2 | UKT向量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_vector, ukt_rasch_scale=0.5 | 27/17 | 0.796547240487 | 0.725125414119 | 0.0853278988967 | 0.57888402947 | 313 | False | False |
| S1/round3 | UKT向量题目×KC方向调制，加入两GRU输入，端到端；learning_rate=0.0001, version=gru, ukt_rasch_mode=input_vector, ukt_rasch_scale=2.0 | 27/17 | 0.796575605315 | 0.725260195416 | 0.084944477406 | 0.577040711497 | 364 | False | False |
| S2/round1 | 向量Rasch有界输出修正，冻结GRU，仅训练因子；learning_rate=0.001, version=gru, ukt_rasch_mode=output_vector, ukt_rasch_scale=0.1, ukt_rasch_freeze=1 | 11/1 | 0.796464813921 | 0.724751813287 | 0.0853549669883 | 0.585808212745 | 445 | False | False |
| S2/round2 | 向量Rasch有界输出修正，冻结GRU，仅训练因子；learning_rate=0.001, version=gru, ukt_rasch_mode=output_vector, ukt_rasch_scale=0.25, ukt_rasch_freeze=1 | 11/1 | 0.796178427165 | 0.724130572088 | 0.0851721051054 | 0.590877337169 | 484 | False | False |
| S2/round3 | 向量Rasch有界输出修正，冻结GRU，仅训练因子；learning_rate=0.001, version=gru, ukt_rasch_mode=output_vector, ukt_rasch_scale=0.5, ukt_rasch_freeze=1 | 11/1 | 0.795476275282 | 0.722920456913 | 0.0851754913902 | 0.591798996155 | 548 | False | False |
| S3/round1 | 冻结GRU，重新训练W2检索补充分支，无KL；learning_rate=0.001, version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.1, ukt_transfer_freeze=1 | 11/1 | 0.796378227565 | 0.725308053351 | 0.0854972110165 | 0.576119052511 | 339 | False | False |
| S3/round2 | 冻结GRU，重新训练W2检索补充分支，无KL；learning_rate=0.001, version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.25, ukt_transfer_freeze=1 | 11/1 | 0.795528728067 | **0.725433271725** | 0.0855333389326 | 0.576119052511 | 349 | False | False |
| S3/round3 | 冻结GRU，重新训练W2检索补充分支，无KL；learning_rate=0.001, version=gru, ukt_transfer_mode=w2, ukt_transfer_scale=0.5, ukt_transfer_freeze=1 | 11/1 | 0.793939178249 | 0.725135580366 | **0.0849315254265** | **0.571055798527** | 379 | False | False |

### 尾部、原有强项与分布宽度

固定弱/强组由原baseline主子集AUC稳定排序确定；自身尾部允许成员变化，两者不能混用。

| 方案 | 改动 | 自身尾10% ↑ | 自身尾20% ↑ | 固定弱10% ↑ | 固定弱20% ↑ | 固定强25% ↑ | P95−P05 ↓ | 去掉两端极差 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 0.553662388924 | 0.59529908901 | 0.553662388924 | 0.59529908901 | 0.823195670194 | 0.287966917197 | 0.544364202284 |
| B0/round1 | 复用V1双独立GRU基线27/17 epoch | 0.553662388924 | 0.59529908901 | 0.553662388924 | 0.59529908901 | 0.823195670194 | 0.287966917197 | 0.544364202284 |
| S0/round1 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | 0.5536287219 | 0.595559495111 | 0.5536287219 | 0.595559495111 | 0.823134475016 | 0.28785786164 | 0.543884247052 |
| S0/round2 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | 0.553727515422 | 0.595419863797 | 0.553727515422 | 0.595420151323 | 0.823160227933 | 0.287760205584 | 0.54288927603 |
| S0/round3 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | 0.553546542888 | 0.595556135647 | 0.553621583563 | 0.595582677802 | 0.823161551716 | 0.288650071086 | 0.544574906034 |
| S1/round1 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | 0.553948161968 | 0.595706589882 | 0.553984993247 | 0.595725319384 | 0.823061824826 | 0.287216911992 | 0.538195245788 |
| S1/round2 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | 0.553908275897 | 0.595495357341 | 0.553916030571 | 0.595503127191 | 0.823181850954 | 0.287717295615 | 0.540512987044 |
| S1/round3 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | 0.554594219091 | **0.596221943452** | 0.554691329525 | 0.596356004856 | 0.822747953181 | 0.28763760671 | **0.534613282029** |
| S2/round1 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | 0.553486625902 | 0.595137549315 | 0.553486625902 | 0.595148842837 | 0.822785508807 | 0.286186816076 | 0.555239389044 |
| S2/round2 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | 0.553637980054 | 0.594922675592 | 0.553672916783 | 0.595007255973 | 0.8219669562 | 0.285870161229 | 0.564439834533 |
| S2/round3 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | 0.552807687416 | 0.593775076142 | 0.552943427054 | 0.594105955972 | 0.820713366925 | 0.283358034962 | 0.58284072551 |
| S3/round1 | 冻结GRU，重新训练W2检索补充分支，无KL | 0.553522884424 | 0.595119345605 | 0.553558377188 | 0.595170895094 | **0.823229508043** | 0.287090178493 | 0.545769692074 |
| S3/round2 | 冻结GRU，重新训练W2检索补充分支，无KL | 0.552991428065 | 0.595101721449 | 0.553306990193 | 0.595532061802 | 0.822733090145 | 0.29174452631 | 0.55424414246 |
| S3/round3 | 冻结GRU，重新训练W2检索补充分支，无KL | **0.555563256547** | 0.596001209185 | **0.55594549517** | **0.596702554434** | 0.822246087576 | **0.28175432516** | 0.546401005206 |

### 不确定性与退化

下列区间是相对原baseline的orirow配对重采样95%区间，不能代表独立学生或跨seed稳定性；极差受两个端点影响。

| 方案 | 改动 | Δstd ↓ | Δrange ↓ | Δstd区间 | 下降>.001数 ↓ | 下降>.005数 ↓ | 下降>.01数 ↓ | 最差序列增益 ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 0 | 0 | [0, 0] | **0** | **0** | **0** | **0** |
| B0/round1 | 复用V1双独立GRU基线27/17 epoch | 0 | 0 | [0.0, 0.0] | **0** | **0** | **0** | **0** |
| S0/round1 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | -7.47962265951e-05 | -0.000921658986175 | [-0.00019250405716331325, 3.38890128105014e-05] | 104 | 4 | **0** | -0.00961538461538 |
| S0/round2 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | -6.45457985823e-05 | 0.00138248847926 | [-0.0001436679314705796, 9.836670477377144e-06] | 80 | 1 | **0** | -0.00641025641026 |
| S0/round3 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | -6.66678167658e-05 | 0.00138248847926 | [-0.00022974730268144198, 8.728822618862743e-05] | 153 | 14 | 1 | -0.015873015873 |
| S1/round1 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | -0.000194128071395 | 0 | [-0.00036011544112688756, -4.5845221448962966e-05] | 150 | 11 | **0** | -0.00963855421687 |
| S1/round2 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | -0.000105232144522 | -0.000460829493088 | [-0.00020612485948681009, -7.952677187357911e-06] | 88 | 4 | **0** | -0.00961538461538 |
| S1/round3 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | -0.000488653635145 | -0.00230414746544 | [-0.0007159566521361263, -0.00026607744425104115] | 212 | 26 | 3 | -0.0112570356473 |
| S2/round1 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | -7.8164052899e-05 | 0.00646335378203 | [-0.0002886022831965259, 0.00014220262867932753] | 224 | 12 | 1 | -0.0132575757576 |
| S2/round2 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | -0.000261025935782 | 0.011532478206 | [-0.000662553401939842, 0.00018299548794290272] | 374 | 78 | 12 | -0.0227272727273 |
| S2/round3 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | -0.000257639651036 | 0.0124541371922 | [-0.000935446783890716, 0.0004856100675228474] | 487 | 182 | 61 | -0.0416666666667 |
| S3/round1 | 冻结GRU，重新训练W2检索补充分支，无KL | 6.40799753373e-05 | -0.00322580645161 | [-0.00016302235366336403, 0.00029104557121649536] | 188 | 23 | 2 | -0.0147783251232 |
| S3/round2 | 冻结GRU，重新训练W2检索补充分支，无KL | 0.00010020789141 | -0.00322580645161 | [-0.0005609938258521633, 0.0009603903365850423] | 266 | 89 | 34 | -0.05 |
| S3/round3 | 冻结GRU，重新训练W2检索补充分支，无KL | **-0.000501605614688** | **-0.00828906043617** | [-0.0013449362515367193, 0.00035471363913991167] | 308 | 130 | 56 | -0.0611111111111 |

### 如何判断本批的证据

- 标准差下降需要弱序列增益与原AUC的负协方差足够大，抵消增益自身方差；逐轮两项分解保存在audit.json，恒等式已核对。极差还要求最差端点获益，不能用平均尾部改善替代。
各方案都与同一GRU B0比较；均值保护不替代严格个体无退化，完整std/range及尾部/强组/退化数并列报告。
- 实质门槛为std与range各相对下降至少1%、std区间上界<0，并保住pooled/macro、固定弱10%/20%、自身尾10%/20%及固定强25%均值。即使通过，逐序列无退化和正式测试仍需另外验证。
有效方向记录暂停；无效最多3轮，稳定性与Peiyou后置。复盘后再决定新的独立机制，不无限扫描同一系数。

### 任务、参数和证据入口

| 方向 | 改动 | 任务状态 |
| --- | --- | --- |
| B0 | 复用V1双独立GRU基线27/17 epoch | completed |
| S0 | UKT标量题目×KC方向调制，加入两GRU输入，端到端 | completed |
| S1 | UKT向量题目×KC方向调制，加入两GRU输入，端到端 | completed |
| S2 | 向量Rasch有界输出修正，冻结GRU，仅训练因子 | completed |
| S3 | 冻结GRU，重新训练W2检索补充分支，无KL | completed |

- B0/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/B0/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/B0/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/B0/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru"}}

- S0/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_scalar", "ukt_rasch_scale": 1.0}, "family": "GRU-UKT-V2-S0", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bf8238930541278ee47bf08f167bff621c00ffb7ae77f32026aa714c021d58d2", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S0/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_scalar", "ukt_rasch_scale": 0.5}, "family": "GRU-UKT-V2-S0", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bf8238930541278ee47bf08f167bff621c00ffb7ae77f32026aa714c021d58d2", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S0/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S0/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_scalar", "ukt_rasch_scale": 2.0}, "family": "GRU-UKT-V2-S0", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bf8238930541278ee47bf08f167bff621c00ffb7ae77f32026aa714c021d58d2", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S1/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_vector", "ukt_rasch_scale": 1.0}, "family": "GRU-UKT-V2-S1", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "f19ac6ccda8dc97dbbdab7efc8f53ac9ee0782f66f9836e187e91140b4272b10", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S1/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_vector", "ukt_rasch_scale": 0.5}, "family": "GRU-UKT-V2-S1", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "f19ac6ccda8dc97dbbdab7efc8f53ac9ee0782f66f9836e187e91140b4272b10", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S1/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S1/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.0001, "ukt_rasch_mode": "input_vector", "ukt_rasch_scale": 2.0}, "family": "GRU-UKT-V2-S1", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "f19ac6ccda8dc97dbbdab7efc8f53ac9ee0782f66f9836e187e91140b4272b10", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S2/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_rasch_mode": "output_vector", "ukt_rasch_freeze": 1, "ukt_rasch_scale": 0.1}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S2", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "69c5093227648ea4ba1f5cffacf6afdaba7c907518c7aca02e81c6afdd0da10e", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S2/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_rasch_mode": "output_vector", "ukt_rasch_freeze": 1, "ukt_rasch_scale": 0.25}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S2", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "69c5093227648ea4ba1f5cffacf6afdaba7c907518c7aca02e81c6afdd0da10e", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S2/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S2/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_rasch_mode": "output_vector", "ukt_rasch_freeze": 1, "ukt_rasch_scale": 0.5}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S2", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "69c5093227648ea4ba1f5cffacf6afdaba7c907518c7aca02e81c6afdd0da10e", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S3/round1：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round1/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round1/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round1/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_transfer_mode": "w2", "ukt_transfer_freeze": 1, "ukt_transfer_scale": 0.1}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S3", "version": "discovery_v1", "round": 1, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bb38a87437e508a6c049195667b2df93362ea8c8d0e89a89ccc6e78e6b4dddf3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S3/round2：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round2/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round2/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round2/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_transfer_mode": "w2", "ukt_transfer_freeze": 1, "ukt_transfer_scale": 0.25}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S3", "version": "discovery_v1", "round": 2, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bb38a87437e508a6c049195667b2df93362ea8c8d0e89a89ccc6e78e6b4dddf3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

- S3/round3：[完整协议](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round3/protocol.json)；[完整指标](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round3/balance.json)；[配对序列](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/S3/round3/paired_orirow.csv)。参数：{"alpha": 0.0, "rank_weight": 0.0, "model_settings": {"version": "gru", "learning_rate": 0.001, "ukt_transfer_mode": "w2", "ukt_transfer_freeze": 1, "ukt_transfer_scale": 0.5}, "initial_checkpoint": "/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/B0/round1/baseline/saved_model/nips_task34_0_0.0001_42_32_200_0_0_/data/pykt-toolkit-archive/gru_ukt_runs_20261114/B0/round1/baseline/saved_model_qikt_mamba_attn_0.3_256_1_1.0_gru_0/attn_model.ckpt", "family": "GRU-UKT-V2-S3", "version": "discovery_v1", "round": 3, "dataset": "nips_task34", "fold": 0, "seed": 42, "parent_protocol_sha256": "bb38a87437e508a6c049195667b2df93362ea8c8d0e89a89ccc6e78e6b4dddf3", "baseline_predictions_sha256": "d79628f501884ffaa56da422090ce3a7f70d7b385a4f38c50a5cdb9cf2cb3311"}

核对1794个独立冻结来源/输入/产物，逐轮数值与退化数、方差恒等式均核验；[审计及分解](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/completed_review/audit.json)。仅分析已有预测，不新增训练、推理或测试标签开发。

## V3：先登记的新增实验


2026-11-14，先登记后实施。当前baseline为两个独立GRU，nips fold0 seed42；B0训练27/选中17，验证窗口pooled=.7965281002815745、macro=.7250511263413856、std(ddof0)=.08543313104118963、range=.5793448589626934。正式windowauclate_mean尚未评估。

### 已尝试的八种结合

|方向|结合位置/方式|主干训练|完成与结论|
|---|---|---|---|
|V1-R0|均值距离检索过去响应，在最终logit附加修正|端到端|3轮，无联合候选|
|V1-R1|学习均值/协方差的W2检索与alpha²协方差传播，附加logit|端到端|3轮，无联合候选|
|V1-R2|R1，按检索协方差收缩修正幅度|端到端|3轮，无联合候选|
|V1-R3|R1，加入UKT原乘性距离调制|端到端|3轮，无联合候选|
|V2-S0|标量题目因子×共享KC方向，改变两GRU历史输入与当前题解码上下文|端到端|3轮，均衡与性能保护未同时成立|
|V2-S1|向量题目因子×共享KC方向，同上|端到端|3轮，有补弱线索但伤害原强组，无联合候选|
|V2-S2|Rasch因子直接形成有界输出修正|冻结B0，仅训练因子|3轮，整体/宏均值退化，极差扩大|
|V2-S3|W2检索补充分支|冻结B0，仅训练检索|3轮，部分尾部/分散改善但整体或强组退化|

共24个新训练，两个批次审计均完成，未达到普通联合候选或实质信号。冻结保证参数未更新，不能保证AUC不下降。

最值得继续解释的是S1R3：pooled+.000047505、macro+.000209069、std−.000488654（约降.572%）、range−.002304147（约降.398%）、固定弱10%+.001028941，但固定强25%−.000447717，未通过保护。S1三轮已经结束，不再无限扫同一scale。

### 新问题

1. 题目ID本来就在QIKT中。Rasch线索来自乘性交互还是只是更多参数？需同容量、保留题目ID的加性对照。
2. S1同时修改历史状态和当前题条件。是否只需修当前题解码，而不应改变历史输入？
3. 冻结输出也失败，不能仅靠“少改参数”保护强项。可从B0开始有限调整，直接保留基线概率及训练内可靠排序；但这些只是训练约束，仍需验证。

### V3新增四个方向

|方向|实际改变|最多轮次/配置|
|---|---|---|
|C0 加性对照|与S1相同数量、同初始分布的题目/KC向量；将乘积改为b_q+(d_c−d_c_initial)，加入历史与当前解码；固定初始KC副本是buffer不训练，保持初始修正0|1轮，scale2、lr1e-4，与已存S1R3匹配；属于控制，不算优化候选|
|Q0 当前题调制|保留原GRU历史输入，仅在当前题解码上下文中加向量Rasch；不显式改变历史输入|3轮，从同seed初始化，scale1/.5/2，lr1e-4|
|A0 概率锚定|Q0结构从B0加载；主干可训练，使用独立冻结B0教师的Bernoulli KL保护|3轮，scale2，lr1e-4，KL权重1/3/10|
|A1 概率＋排序保护|A0固定KL1，并约束同训练片段内教师已正确且概率margin>.1的正负对不缩小margin|3轮，pair权重.01/.1/1，其他同A0|

共最多10个新训练，C0是一轮控制。每个优化方向首个普通候选就记录暂停，不追加seed/fold/Peiyou。Q0不加teacher；A0/A1从B0初始化并且学习率/训练路径不同，不能把它与Q0差异全归因给锚定。A1为训练内教师正确对，不是OOF可靠困难证据，教师可能过拟合。该排序保护曾在旧Mamba方向试过，本批是与新GRU、当前题调制结合，不能称全新未试机制。

固定教师在加载原B0后复制并去除新增调制，仅参与训练，不进入推理模型参数。恢复时先重建同一B0教师，再加载学生/优化器/RNG，避免把恢复后的学生误当教师。教师路径、SHA和源文件身份保存；不使用验证或测试标签定义保护对。

C0的加法与乘法梯度本来不同；即使容量/零初始/参数分布匹配，也不是证明唯一因果。KL约束概率、margin约束部分训练排序，二者都不能保证验证/未来每条序列的AUC不退化。

### 其他可考虑、暂不盲目叠加的结合

- UKT式响应视图/对比学习作为训练正则：原UKT配置use_CL=0，需独立对照，且序列级对比可能强化序列间差异。
- 将均值/协方差用于GRU状态或门控，而非再挂检索器：需正方差、状态更新及数值稳定消融。
- 对题目/KC两个分支分别加入小适配器并条件化融合：需区分分支容量与融合效应。
- 原UKT保留题目ID的加性替代重训消融：用于解释机制，不混用现有正式测试序列选参。

先执行上表四个有明确对照与来源的方向，不一次同时改变多个未验证模块。

### 评价与运行边界

原采样、选模/早停不改；主比较仍是开发窗口primary773、std ddof0，报告pooled/macro、std/range、固定/自身弱10/20%、强25%、退化数量/幅度/配对区间。实质门槛仍为std/range各降≥1%、std区间上界<0且均值保护；个体无退化另报。正式windowauclate_mean与稳定性尚未完成。

先做公式、零初始化、未来标签隔离、真实模型工厂/CLI、教师不更新、完整epoch恢复检查，再做32片真实训练CUDA预检。新增锚定/排序开销在匹配原生set_seed flags的代表性负载中重测，不复用不匹配的旧双并发结论。任务经tsp，独立冻结源码、协议、输入/权重SHA与状态，优化器/RNG/早停完整epoch恢复，窗口64批恢复；产物放/data，不上传Git。

## 已落地的执行状态

V3 GPU socket为`/tmp/pykt-qikt-gpu-20261104.sock`：110真实训练预检已完成；111新负载资源校准；116 C0、117 Q0、118 A0、119 A1按前置与共享槽位等待。CPU socket为`/tmp/pykt-qikt-cpu-20261104.sock`：116监测、117实时报告、118最终复盘。两条socket的ID可重复，不能相加当训练实例数；当前校准工作者不是正式训练轮次。

24项CPU测试通过；32片真实训练CUDA预检四配置全部通过、零初始化预测差均为0、两步梯度有限、教师不进入学生state。完整epoch中断恢复检查同时比较参数与随机状态。校准只覆盖代表性当前题调制和最重KL＋排序负载，采用原生set_seed的同步/确定性设置；校准后自动准入，未宣称所有未来负载最优。

首次排队时缺少方向输出目录，4个方向在训练前失败；已创建目录并恢复，失败记录与提前退出的观察者产物留在`setup_recovery.json`及`setup_attempt1_*`，没有把该故障算作科学轮次，也没有覆盖训练结果。

源码/协议/数据/原B0权重SHA、命令、依赖、jobID和生命周期均在`logs/qikt_gru_ukt_v3_20261114/`，实际产物位于`/data/pykt-toolkit-archive/gru_ukt_v3_runs_20261114/`。训练按完整epoch恢复模型/优化器/RNG/选模早停；窗口按64批恢复。找到普通候选即停止该方向并记录，稳定性/Peiyou后置。当前未提交或推送代码。

[新增实验实时结果](/root/pykt-toolkit/logs/qikt_gru_ukt_v3_20261114/GRU_UKT_V3_实时报告.md)会自动更新；本文件中V1/V2表为已完成证据的快照，V3尚无有效性结论。
