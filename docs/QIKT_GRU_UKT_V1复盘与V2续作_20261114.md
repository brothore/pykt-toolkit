# UKT首批复盘与双GRU V2续作

首批已经全部完成：1个新双GRU基线＋12个迁移训练，4个方向各3轮，无普通联合候选、无实质信号。以下含完整基线/逐轮指标、原UKT敏感性、资源修订和下一批依据；不是只展示差值。

## 主要发现

- 附加检索端到端训练没有解决目标：R0均值对照也未通过，因此不能把失败全部归因给W2；较大scale通常增加退化，冻结/改变加入位置值得分别验证。
- 原UKT依赖Rasch与协方差路径，但删模块造成分布偏移，推理干预不是重训因果消融。实际use_CL=0，不能以对比学习解释现有结果。
- 更关键的结构差异：UKT的`q_data`实际是知识点ID（57类），题目ID通过948类`difficult_param`进入；在此stoc_qid路径中，去掉Rasch也去掉了题目ID信息。QIKT本来就有题目embedding，因此“.059的降幅”不能解释为在QIKT上添加Rasch也会有同等收益。V2只把它作为共享题目×KC调制的探索；出现候选后，须用保留题目ID的加性对照区分信息与交互作用。
- 下一批不继续扫V1检索强度。S0/S1把UKT标量/向量题目因子作用于KC方向，进入GRU输入；S2放在冻结GRU后的有界输出；S3冻结GRU只训练W2补充。各方向最多3轮、首次普通候选即暂停。

## 原UKT的只读验证敏感性

nips fold0、seed3407，223341个概念展开计分位置；参考保存模型所在源码的预测最大差0。指标是原生验证片段AUC，不能与下文QIKT题目窗口AUC、正式windowauclate_mean相减。没有新优化步骤，原权重未写入。

|干预|原生片段验证AUC|相对完整模型变化|Brier↓|
|---|---|---|---|
|完整UKT|**0.794772634320**|+0.000000000000|**0.182640552521**|
|W2距离去掉协方差项，保留值与读出协方差|0.767735798534|-0.027036835786|0.195078909397|
|关闭原乘性距离调制|0.780274886298|-0.014497748022|0.189508557320|
|协方差聚合由alpha²改为alpha|0.754434593953|-0.040338040367|0.200592294335|
|将已学题目难度因子置零|0.735565940005|-0.059206694315|0.207315832376|

## 资源复测

原首轮漏配原生set_seed中的同步/确定性flags，已明确标记不匹配，不能沿用其双并发推荐。匹配原训练设置后结果如下；均为固定120batch、batch32、两次重复的中位数。

|代表性负载/阶段|1并发step/s|2并发step/s|3并发step/s|推荐|
|---|---|---|---|---|
|dual_gru/training|14.537598|5.027773|4.193741|1|
|dual_gru/inference|41.161061|13.909533|11.300362|1|
|gru_ukt_heavy/training|10.905023|4.312072|3.611723|1|
|gru_ukt_heavy/inference|34.156405|11.579482|10.015881|1|

原生同步设置下，多并发显著降低总吞吐。新V2的冻结/输入调制结构不同，已安排匹配flags的新探针；不是永远固定单并发。R0/R1当时已运行，按约定自然完成；R2/R3等待正确复测后使用单槽，科学配置未热补。

## V2方案与执行入口

- [V2复盘与预登记方案](/root/pykt-toolkit/docs/QIKT_GRU_UKT_V2_复盘与方案.md)。
- [V2实时报告](/root/pykt-toolkit/logs/qikt_gru_ukt_v2_20261114/GRU_UKT_V2_实时报告.md)：S0/S1全模型从同seed初始化；S2/S3从同B0初始化且只训练新因子/分支。学习率和参数集合差异已明确，不能把收益归因给单一因素。
- 23项CPU检查通过：零初始化、旧GRU别名、输入/输出因果、padding、真实模型工厂、冻结主干不更新、保存重建和恢复管线；四配置真实32片训练CUDA预检已通过：零输出差均0、两步梯度有限，冻结方案主干未变；资源复测正在执行，未成功不能开始正式新方案。

## 首批完整结果

以下为已完成审计的原报告快照；baseline为新GRU，未用旧Mamba作参照。
# 双GRU＋UKT V1完整结果与机制复盘

本批已终止等待：共取得13条完成结果，优化点候选0条，实质开发信号0条。任务失败/停止会明确保留，不能算完成实验。

nips_task34／QIKT双独立GRU／fold0／seed42。全新B0，不复用Mamba权重。原生开发窗口、主子集至少20位置/少数类5，std ddof0；正式windowauclate_mean未评估，实际训练/选中epoch逐行给出。

R0均值距离检索对照，R1学习正协方差的Wasserstein历史读出，R2按检索协方差收缩融合，R3检验UKT原有乘性距离调制。保留两GRU，严格过去响应，零初始化有界补充；scale .1/.25/.5、各最多3轮，普通候选即停。

测试序列报告只作已披露事后分析，不能证明UKT模块因果有效。协方差学习增加有效参数；端到端训练、早停和单seed混杂仍在。14项CPU检查和真实CUDA预检/新负载资源实测为前置，不能把预检或并发推荐当效力。

## 完整主指标

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

## 尾部、原有强项与分布宽度

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

## 不确定性与退化

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

## 如何判断本批的证据

- 标准差下降需要弱序列增益与原AUC的负协方差足够大，抵消增益自身方差；逐轮两项分解保存在audit.json，恒等式已核对。极差还要求最差端点获益，不能用平均尾部改善替代。
先比较R0/R1，区分普通历史检索与学习分布的增量，再比较R1/R2/R3判断融合/调制。UKT实际use_CL=0，不能归功于未用的对比学习。
- 实质门槛为std与range各相对下降至少1%、std区间上界<0，并保住pooled/macro、固定弱10%/20%、自身尾10%/20%及固定强25%均值。即使通过，逐序列无退化和正式测试仍需另外验证。
候选记录暂停，失效方向最多3轮；基于证据再决定UKT训练消融、Rasch或CL迁移，不自动追加seed/Peiyou。

## 任务、参数和证据入口

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


## 可追溯记录

[原UKT敏感性与源码/权重身份](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/ukt_sensitivity/summary.json)；[首批最终审计](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/completed_review/audit.json)；[正确flags的资源复测](/root/pykt-toolkit/logs/qikt_gru_ukt_20261114/resource_calibration_matched/summary.json)。全部均为本地实验产物，不上传Git。
