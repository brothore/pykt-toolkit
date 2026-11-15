# QIKT-Mamba全方位复盘与续作（2026-11-11）

当前结论：尚无同时满足幅度、保护和不确定性条件的实质开发信号。最值得保留的原生方案仍是V6-I4R2（多尺度历史＋KL1），但幅度有限且有个体退化。V14排序/均衡组合已完成9轮，无联合候选；V15原尝试因CLI漏接在优化前失败，已修复并重新启动，不把工程失败算机制失败。

本报告核查193条V2至V14及发现/固定融合的主开发结果；不同组件/CPU混合不是等量新训练。V1是封存历史，不重复计算；seed43/44历史复验见原完整报告，未用seed42表格替代稳定性结论。当前学习型C2仍缺合规OOF前置，不能称所有方案已经完成。

评估口径：nips_task34，QIKT-Mamba/public_mamba，fold0，seed42；773条orirow（位置≥20、少数类≥5），std ddof=0。原baseline训练14/选中4。pooled为验证窗口AUC，正式all_results.json.windowauclate_mean缺失，不能替代；未用测试集开发、未新增Peiyou或seed稳定性复验。

| 批次 | 已完成结果行数 | 非控制普通点候选 | 实质信号 |
|---|---|---|---|
| V2 | 12 | 1 | 0 |
| D07 | 27 | 2 | 0 |
| D08 | 3 | 3 | 0 |
| V6 | 17 | 1 | 0 |
| V7 | 9 | 0 | 0 |
| V8 | 11 | 0 | 0 |
| V9 | 9 | 1 | 0 |
| V10 | 15 | 0 | 0 |
| V11 | 12 | 0 | 0 |
| V12 | 15 | 0 | 0 |
| V13 | 2 | 1 | 0 |
| V14 | 9 | 0 | 0 |
| V3 | 21 | 0 | 0 |
| V4 | 13 | 0 | 0 |
| V5 | 9 | 1 | 0 |
| V6融合 | 9 | 0 | 0 |

实质开发门槛：std与range相对原baseline各下降至少1%、std配对95%区间上界<0，pooled/macro、固定弱10/20%、自身尾10/20%、固定强25%均值不降。普通点候选不要求幅度/区间；个体无退化是更强条件，仍未满足。反复验证选择的偏差不能由bootstrap自动消除。

下面每行同时给出方案、改动和参数；每列最优加粗只表示单项，不等于可选胜者。baseline为全部表格共同对照。epoch实际/选中，CPU行不新增训练，组件训练信息见协议。

| 方案 | 改动与参数 | epoch | pooled↑ | macro↑ | std↓ | range↓ | 退化条数↓ | 普通点候选 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始QIKT-Mamba | 14/4 | 0.788791532411 | 0.714835126776 | 0.0849285921818 | 0.553394581584 | **0** | 参照 |
| V2/A1/round1 | 序列等权BCE；{"alpha":0.25,"rank_weight":0.0} | 14/4 | 0.788533089003 | 0.715027710871 | 0.0848711618875 | 0.551580615809 | 405 | False |
| V2/A1/round2 | 序列等权BCE；{"alpha":0.1,"rank_weight":0.0} | 14/4 | 0.78872677524 | 0.715057295471 | 0.0849520402372 | 0.552023833984 | 375 | False |
| V2/A2/round1 | 平滑困难序列加权；{"alpha":0.25,"rank_weight":0.01} | 14/4 | 0.788547655768 | 0.715029084697 | 0.0846113239471 | 0.555722210807 | 400 | False |
| V2/A2/round2 | 平滑困难序列加权；{"alpha":0.25,"rank_weight":0.002} | 14/4 | 0.788569236041 | 0.715099119236 | 0.084725819579 | 0.552035574863 | 386 | False |
| V2/A3/round1 | 序列内排序；{"alpha":0.0,"rank_weight":0.0} | 14/4 | 0.788520724254 | 0.714939714509 | 0.0844204959911 | 0.552963104288 | 402 | False |
| V2/A3/round2 | 序列内排序；{"alpha":0.0,"rank_weight":0.0} | 14/4 | 0.788563953606 | 0.715026730372 | 0.0846259625683 | 0.547433150371 | 395 | False |
| V2/B1/round1 | 低频覆盖/固定融合，细节见参数；{"alpha":0.0,"rank_weight":0.0,"frequency_alpha":0.25,"frequency_cap":2.0} | 14/4 | 0.788807092037 | 0.715004513852 | 0.0847611687193 | 0.557547917462 | 348 | False |
| V2/B1/round2 | 低频覆盖/固定融合，细节见参数；{"alpha":0.0,"rank_weight":0.0,"frequency_alpha":0.1,"frequency_cap":1.5} | 14/4 | 0.788804080873 | 0.715025246568 | 0.084845656473 | 0.552472922598 | 343 | True |
| V2/B2/round1 | 冻结主干前缀小适配；{"alpha":0.0,"rank_weight":0.0} | 11/1 | 0.788790339035 | 0.714845148725 | 0.0849099572916 | 0.552472922598 | 233 | False |
| V2/B2/round2 | 冻结主干前缀小适配；{"alpha":0.0,"rank_weight":0.0} | 11/1 | 0.788790358705 | 0.714845616642 | 0.0849103280056 | 0.552472922598 | 230 | False |
| V2/C1/round1 | 基线保护组合；{"alpha":0.0,"rank_weight":0.0} | 11/1 | 0.78879153196 | 0.714835175927 | 0.0849285084225 | 0.553394581584 | 3 | False |
| V2/C1/round2 | 基线保护组合；{"alpha":0.0,"rank_weight":0.0} | 11/1 | 0.788791531919 | 0.714835165355 | 0.0849285094384 | 0.553394581584 | 4 | False |
| D07/D1/round1 | D1；{"alpha":0.1,"rank_weight":0.002,"model_settings":{"item_residual":1}} | 14/4 | 0.788696888378 | 0.715086298465 | 0.0847839307465 | 0.558469576448 | 366 | False |
| D07/D1/round2 | D1；{"alpha":0.05,"rank_weight":0.001,"model_settings":{"item_residual":1}} | 14/4 | 0.7887240068 | 0.714931985959 | 0.0848623133892 | 0.554777070064 | 377 | False |
| D07/D1/round3 | D1；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{"item_residual":1}} | 14/4 | 0.788792827745 | 0.714892998318 | 0.0848817341228 | 0.55108456368 | 379 | True |
| D07/D2/round1 | D2；{"alpha":0.1,"rank_weight":0.002,"model_settings":{"concept_memory_residual":1}} | 14/4 | 0.788678203828 | 0.715024383137 | 0.0848386723205 | 0.553412192903 | 375 | False |
| D07/D2/round2 | D2；{"alpha":0.05,"rank_weight":0.001,"model_settings":{"concept_memory_residual":1}} | 14/4 | 0.788706696225 | 0.714984275573 | 0.0848953558779 | 0.55569872905 | 369 | False |
| D07/D2/round3 | D2；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{"concept_memory_residual":1}} | 14/4 | 0.788770080364 | 0.714939030867 | 0.0850001933698 | 0.55295136341 | 357 | False |
| D07/D3/round1 | D3；{"alpha":0.1,"rank_weight":0.002,"model_settings":{"history_response_dropout":0.01}} | 14/4 | 0.788855416135 | 0.715112205202 | 0.085277986322 | 0.559361883237 | 362 | False |
| D07/D3/round2 | D3；{"alpha":0.05,"rank_weight":0.001,"model_settings":{"history_response_dropout":0.005}} | 14/4 | 0.788912218857 | 0.715104195823 | 0.0853438626722 | 0.563025037424 | 354 | False |
| D07/D3/round3 | D3；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{"history_response_dropout":0.002}} | 14/4 | 0.788951781489 | 0.715267418403 | 0.0853413942133 | 0.564862484957 | 342 | False |
| D07/D4/round1 | D4；{"alpha":0.1,"rank_weight":0.002,"model_settings":{"head_fusion_residual":1,"fusion_gate_scale":0.05}} | 14/4 | 0.788736473467 | 0.714976942232 | 0.0848351432799 | 0.550641345505 | 364 | False |
| D07/D4/round2 | D4；{"alpha":0.05,"rank_weight":0.001,"model_settings":{"head_fusion_residual":1,"fusion_gate_scale":0.02}} | 14/4 | 0.788784112586 | 0.714930147106 | 0.0848378662327 | 0.552939622531 | 364 | False |
| D07/D4/round3 | D4；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{"head_fusion_residual":1,"fusion_gate_scale":0.01}} | 14/4 | 0.788744075078 | 0.714879662611 | 0.0849592639662 | 0.554316240571 | 359 | False |
| D07/D5/round1 | D5；{"alpha":0.1,"rank_weight":0.002,"model_settings":{"training_overlap":64}} | 14/4 | 0.789332200449 | 0.716171840871 | 0.0853071647263 | 0.549796002231 | 295 | False |
| D07/D5/round2 | D5；{"alpha":0.05,"rank_weight":0.001,"model_settings":{"training_overlap":64}} | 14/4 | 0.789384848872 | 0.716359865788 | 0.0852518586424 | 0.544744489125 | 282 | False |
| D07/D5/round3 | D5；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{"training_overlap":64}} | 14/4 | 0.789405728675 | 0.716343459436 | 0.0852141167591 | 0.54244034166 | 290 | False |
| D07/D6/round1 | D6；{"alpha":0.0,"rank_weight":0.002,"model_settings":{},"frequency_alpha":0.1,"frequency_cap":1.5} | 14/4 | 0.788782892573 | 0.714957795203 | 0.0849721516398 | 0.552939622531 | 347 | False |
| D07/D6/round2 | D6；{"alpha":0.0,"rank_weight":0.001,"model_settings":{},"frequency_alpha":0.05,"frequency_cap":1.5} | 14/4 | 0.788771511178 | 0.714948483835 | 0.0848973950324 | 0.556153688104 | 357 | False |
| D07/D6/round3 | D6；{"alpha":0.0,"rank_weight":0.0005,"model_settings":{},"frequency_alpha":0.02,"frequency_cap":1.5} | 14/4 | 0.788797337504 | 0.714903141879 | 0.0849191973162 | 0.553388711145 | 373 | True |
| D07/E1/round1 | E1；{"candidate_weight":0.2} | CPU：无新训练 | 0.788783154803 | 0.714913305226 | 0.0848667651451 | 0.552478793038 | 360 | False |
| D07/E1/round2 | E1；{"candidate_weight":0.5} | CPU：无新训练 | 0.788729535361 | 0.715036327071 | 0.0848221082365 | 0.552484663477 | 366 | False |
| D07/E1/round3 | E1；{"candidate_weight":0.8} | CPU：无新训练 | 0.788627922446 | 0.715099753361 | 0.0848445913763 | 0.550653086384 | 380 | False |
| D07/E2/round1 | E2；{"candidate_weight":0.2} | CPU：无新训练 | 0.788790475498 | 0.71495206378 | 0.0848618821841 | 0.554327981449 | 343 | False |
| D07/E2/round2 | E2；{"candidate_weight":0.5} | CPU：无新训练 | 0.788748016036 | 0.715073351019 | 0.0848487051737 | 0.552023833984 | 365 | False |
| D07/E2/round3 | E2；{"candidate_weight":0.8} | CPU：无新训练 | 0.788656526637 | 0.715073625339 | 0.084832221644 | 0.55019225689 | 380 | False |
| D07/E3/round1 | E3；{"candidate_weight":0.2} | CPU：无新训练 | 0.788790755104 | 0.714954817928 | 0.0848129158653 | 0.55294549297 | 345 | False |
| D07/E3/round2 | E3；{"candidate_weight":0.5} | CPU：无新训练 | 0.788747060671 | 0.715089191241 | 0.0847833972945 | 0.551102174998 | 356 | False |
| D07/E3/round3 | E3；{"candidate_weight":0.8} | CPU：无新训练 | 0.788654100748 | 0.715087815645 | 0.0847023137171 | 0.548348938918 | 382 | False |
| D08/F1/round1 | F1；{"candidate_weight":0.05} | CPU：无新训练 | 0.788803186076 | 0.715021157803 | 0.08483910018 | 0.552012093105 | 351 | True |
| D08/F2/round1 | F2；{"candidate_weight":0.05} | CPU：无新训练 | 0.78880497276 | 0.715042772178 | 0.0848214679799 | 0.551551263612 | 344 | True |
| D08/F3/round1 | F3；{"candidate_weight":0.05} | CPU：无新训练 | 0.78880537424 | 0.715035178718 | 0.0848375550138 | 0.552012093105 | 347 | True |
| V6/I0/round1 | 当前题64维投影＋非线性残差；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"question","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788106444076 | 0.713153076717 | 0.0839802822879 | 0.561707123778 | 524 | False |
| V6/I0/round2 | 当前题64维投影＋非线性残差；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"question","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.7877347488 | 0.711697996117 | **0.0838034048521** | 0.564938800669 | 583 | False |
| V6/I0/round3 | 当前题64维投影＋非线性残差；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"question","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.788330732398 | 0.713362485894 | 0.0843715180223 | 0.557577269659 | 518 | False |
| V6/I1/round1 | 题目残差＋KC与全局前缀状态；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"prefix","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.787501522306 | 0.71440709942 | 0.0847740701402 | 0.54371716223 | 415 | False |
| V6/I1/round2 | 题目残差＋KC与全局前缀状态；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"prefix","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.786835996844 | 0.713615478741 | 0.0840413788118 | 0.550629604626 | 463 | False |
| V6/I1/round3 | 题目残差＋KC与全局前缀状态；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"prefix","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.787644460441 | 0.713821530855 | 0.084566799849 | 0.551108045437 | 464 | False |
| V6/I2/round1 | 前缀＋8/32步多尺度历史＋最近KC响应；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788100589684 | 0.714972793245 | 0.0845390873472 | 0.550168775133 | 347 | False |
| V6/I2/round2 | 前缀＋8/32步多尺度历史＋最近KC响应；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.787634247506 | 0.71439749614 | 0.0842303106643 | 0.545560480202 | 419 | False |
| V6/I2/round3 | 前缀＋8/32步多尺度历史＋最近KC响应；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 13/3 | 0.787675742342 | 0.715149101525 | 0.0839677679446 | 0.556620388036 | 376 | False |
| V6/I3/round1 | 多尺度历史＋序列等权BCE .25；{"alpha":0.25,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788058714883 | 0.71499433452 | 0.0846382458065 | 0.552012093105 | 338 | False |
| V6/I3/round2 | 多尺度历史＋序列等权BCE .25；{"alpha":0.25,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.787658907707 | 0.714516247445 | 0.0844238512034 | 0.544177991723 | 405 | False |
| V6/I3/round3 | 多尺度历史＋序列等权BCE .25；{"alpha":0.25,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 13/3 | 0.787622706248 | 0.715181374847 | 0.0839835548648 | 0.557081217529 | 366 | False |
| V6/I4/round1 | 多尺度历史＋冻结基线KL1；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.25,"causal_adapter_kl":1.0,"learning_rate":0.001}} | 11/1 | 0.78866594075 | 0.715261572546 | 0.0845544243332 | 0.55569872905 | 332 | False |
| V6/I4/round2 | 多尺度历史＋冻结基线KL1；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003}} | 17/7 | 0.789404884448 | 0.715822072871 | 0.0843097672828 | 0.550629604626 | 302 | True |
| V6/I5/round1 | 多尺度历史＋类别支持度门控；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"supported","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788373075313 | 0.714777555922 | 0.0846733999116 | 0.542346414629 | 379 | False |
| V6/I5/round2 | 多尺度历史＋类别支持度门控；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"supported","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.787954814756 | 0.714719981622 | 0.0846253343347 | 0.538204819631 | 385 | False |
| V6/I5/round3 | 多尺度历史＋类别支持度门控；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"supported","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.787346898295 | 0.71459757701 | 0.0846894075608 | 0.539126478617 | 413 | False |
| V7/U0/round1 | 按冻结baseline不确定性限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"uncertain","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788245491946 | 0.715012703774 | 0.0845340181839 | 0.552472922598 | 347 | False |
| V7/U0/round2 | 按冻结baseline不确定性限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"uncertain","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.78774468848 | 0.714113490652 | 0.0843079884582 | 0.539569696792 | 429 | False |
| V7/U0/round3 | 按冻结baseline不确定性限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"uncertain","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.787483444996 | 0.715068789279 | 0.0846516659199 | 0.552933752091 | 359 | False |
| V7/U1/round1 | 按冻结预测头分歧限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"disagreement","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788657634078 | 0.714878723023 | 0.084878846311 | 0.554777070064 | 369 | False |
| V7/U1/round2 | 按冻结预测头分歧限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"disagreement","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.788165163511 | 0.714406178911 | 0.0849431875248 | 0.546948839121 | 426 | False |
| V7/U1/round3 | 按冻结预测头分歧限制修正；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"disagreement","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.788251436862 | 0.714416205909 | 0.0849018148687 | 0.550641345505 | 432 | False |
| V7/U2/round1 | 优先低KC历史支持的位置；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"low_support","causal_adapter_scale":0.25,"causal_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.788449035329 | 0.714864270416 | 0.0846369385875 | 0.563072000939 | 398 | False |
| V7/U2/round2 | 优先低KC历史支持的位置；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"low_support","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.788128051437 | 0.714446811067 | 0.0842620278079 | 0.564448618979 | 420 | False |
| V7/U2/round3 | 优先低KC历史支持的位置；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"low_support","causal_adapter_scale":0.5,"causal_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.787801400513 | 0.714564527567 | 0.0846893708956 | 0.55569285861 | 421 | False |
| V8/K1/round1 | seed42＋43固定概率平均；{"candidate_weight":0.25} | CPU：无新训练 | 0.789234030626 | 0.715444745522 | 0.0849623862442 | 0.55294549297 | 284 | False |
| V8/K1/round2 | seed42＋43固定概率平均；{"candidate_weight":0.5} | CPU：无新训练 | 0.78934467232 | 0.715514036832 | 0.0851491692664 | 0.560271801344 | 325 | False |
| V8/K1/round3 | seed42＋43固定概率平均；{"candidate_weight":0.75} | CPU：无新训练 | 0.789124663818 | 0.715174832332 | 0.0852062929836 | 0.556083242831 | 356 | False |
| V8/K2/round1 | seed42＋44固定概率平均；{"candidate_weight":0.25} | CPU：无新训练 | 0.789406818086 | 0.715393511563 | 0.085145139711 | 0.556608647157 | 291 | False |
| V8/K2/round2 | seed42＋44固定概率平均；{"candidate_weight":0.5} | CPU：无新训练 | 0.789653928496 | 0.715603050489 | 0.0855840520525 | 0.552420088644 | 325 | False |
| V8/K2/round3 | seed42＋44固定概率平均；{"candidate_weight":0.75} | CPU：无新训练 | 0.789542005688 | 0.715259117707 | 0.0858588072277 | 0.555581320262 | 345 | False |
| V8/K3/round1 | seed43＋44固定概率平均，对原42比较；{"candidate_weight":0.25} | CPU：无新训练 | 0.789256725635 | 0.715257998931 | 0.0855151437295 | 0.555132231648 | 354 | False |
| V8/K3/round2 | seed43＋44固定概率平均，对原42比较；{"candidate_weight":0.5} | CPU：无新训练 | 0.789561903694 | 0.715441321781 | 0.0857668340179 | 0.557879597288 | 334 | False |
| V8/K3/round3 | seed43＋44固定概率平均，对原42比较；{"candidate_weight":0.75} | CPU：无新训练 | 0.789499794154 | 0.715159312849 | 0.0859325973303 | 0.561525140157 | 349 | False |
| V8/K4/round1 | seed42/43/44各1/3概率平均；{"candidate_weight":1.0} | CPU：无新训练 | 0.78975801123 | 0.715861121082 | 0.0854432235669 | 0.55883647891 | 302 | False |
| V8/K5/round1 | seed42/43/44各1/3 logit平均；{"candidate_weight":1.0} | CPU：无新训练 | **0.789769976056** | 0.715869228002 | 0.0854411047043 | 0.560195485632 | 302 | False |
| V9/L1/round1 | 三seed性能锚＋题目修正R1；{"candidate_weight":0.25} | CPU：无新训练 | 0.789643384812 | 0.715547744079 | 0.0851087104835 | 0.56207989668 | 309 | False |
| V9/L1/round2 | 三seed性能锚＋题目修正R1；{"candidate_weight":0.5} | CPU：无新训练 | 0.78933397279 | 0.714893981615 | 0.0847561208826 | 0.563503478235 | 373 | False |
| V9/L1/round3 | 三seed性能锚＋题目修正R1；{"candidate_weight":0.75} | CPU：无新训练 | 0.788826316948 | 0.714215707694 | 0.0843693070242 | 0.560301153541 | 447 | False |
| V9/L2/round1 | 三seed性能锚＋小lr题目修正R3；{"candidate_weight":0.25} | CPU：无新训练 | 0.789672242177 | 0.715566289665 | 0.0852225476308 | 0.560236578708 | 311 | False |
| V9/L2/round2 | 三seed性能锚＋小lr题目修正R3；{"candidate_weight":0.5} | CPU：无新训练 | 0.789406623185 | 0.715001038851 | 0.0849481561718 | 0.560756112595 | 366 | False |
| V9/L2/round3 | 三seed性能锚＋小lr题目修正R3；{"candidate_weight":0.75} | CPU：无新训练 | 0.7889640869 | 0.714283736788 | 0.084700383927 | 0.562622912325 | 439 | False |
| V9/L3/round1 | 三seed性能锚＋困难加权A3R2；{"candidate_weight":0.25} | CPU：无新训练 | 0.789669158192 | 0.715928209125 | 0.0851891148641 | 0.558854090229 | 293 | False |
| V9/L3/round2 | 三seed性能锚＋困难加权A3R2；{"candidate_weight":0.5} | CPU：无新训练 | 0.789436155121 | 0.715735975446 | 0.0849804884987 | 0.552927881652 | 291 | False |
| V9/L3/round3 | 三seed性能锚＋困难加权A3R2；{"candidate_weight":0.75} | CPU：无新训练 | 0.789066640304 | 0.715493149007 | 0.084798533175 | 0.548803897972 | 320 | True |
| V10/R0/round1 | 全局前缀预测误差对照；{"prior":3.0,"radius":0.5} | CPU：无新训练 | 0.786557704565 | 0.711914731619 | 0.0862505226725 | 0.577718747248 | 582 | False |
| V10/R0/round2 | 全局前缀预测误差对照；{"prior":1.0,"radius":1.0} | CPU：无新训练 | 0.784945864209 | 0.710107252702 | 0.0870998490974 | 0.586847280519 | 603 | False |
| V10/R0/round3 | 全局前缀预测误差对照；{"prior":10.0,"radius":1.0} | CPU：无新训练 | 0.787040018135 | 0.713023008502 | 0.0855548920422 | 0.558918665062 | 565 | False |
| V10/R1/round1 | 当前KC前缀预测误差；{"prior":3.0,"radius":0.5} | CPU：无新训练 | 0.787925719297 | 0.713622144962 | 0.0851626506571 | 0.574126038334 | 468 | False |
| V10/R1/round2 | 当前KC前缀预测误差；{"prior":1.0,"radius":1.0} | CPU：无新训练 | 0.783525916358 | 0.708983725739 | 0.0859106274369 | 0.59980334028 | 539 | False |
| V10/R1/round3 | 当前KC前缀预测误差；{"prior":10.0,"radius":1.0} | CPU：无新训练 | 0.788974475433 | 0.714835266108 | 0.0848975742444 | 0.558924535501 | 414 | False |
| V10/R2/round1 | 当前概率桶前缀预测误差；{"prior":3.0,"radius":0.5} | CPU：无新训练 | 0.786463589131 | 0.705685595995 | 0.0875516676568 | 0.569470779888 | 672 | False |
| V10/R2/round2 | 当前概率桶前缀预测误差；{"prior":1.0,"radius":1.0} | CPU：无新训练 | 0.78099134571 | 0.695924437543 | 0.0896362839096 | 0.596116704335 | 693 | False |
| V10/R2/round3 | 当前概率桶前缀预测误差；{"prior":10.0,"radius":1.0} | CPU：无新训练 | 0.787914842816 | 0.709820020022 | 0.0865098389682 | 0.561228682967 | 660 | False |
| V10/R3/round1 | KC误差与概率桶误差各一半；{"prior":3.0,"radius":0.5} | CPU：无新训练 | 0.788320395008 | 0.710387962022 | 0.0863291622929 | 0.58055416948 | 571 | False |
| V10/R3/round2 | KC误差与概率桶误差各一半；{"prior":1.0,"radius":1.0} | CPU：无新训练 | 0.785760052988 | 0.706208611979 | 0.0872941243232 | 0.602107487745 | 592 | False |
| V10/R3/round3 | KC误差与概率桶误差各一半；{"prior":10.0,"radius":1.0} | CPU：无新训练 | 0.788963128379 | 0.712727814048 | 0.0855902918562 | 0.562150341953 | 567 | False |
| V10/R4/round1 | KC×概率桶局部误差；{"prior":3.0,"radius":0.5} | CPU：无新训练 | 0.788494384532 | 0.712759781456 | 0.0859945486407 | 0.550162904693 | 467 | False |
| V10/R4/round2 | KC×概率桶局部误差；{"prior":1.0,"radius":1.0} | CPU：无新训练 | 0.784747564288 | 0.707555336283 | 0.0874182210565 | 0.57170448208 | 556 | False |
| V10/R4/round3 | KC×概率桶局部误差；{"prior":10.0,"radius":1.0} | CPU：无新训练 | 0.788993766281 | 0.71436254322 | 0.0852215920738 | 0.551545393173 | 416 | False |
| V11/T0/round1 | 冻结主干＋题目相似度因果检索；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"content","retrieval_adapter_scale":0.25,"retrieval_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.78788548816 | 0.715130100768 | 0.0844315679595 | 0.541873844257 | 332 | False |
| V11/T0/round2 | 冻结主干＋题目相似度因果检索；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"content","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.787354433438 | 0.713387007827 | 0.0847992957265 | 0.565842848337 | 444 | False |
| V11/T0/round3 | 冻结主干＋题目相似度因果检索；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"content","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.78732801864 | 0.715369365037 | 0.0842660590595 | 0.550647215944 | 368 | False |
| V11/T1/round1 | 相似度检索＋同KC偏置2；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"kc_bias","retrieval_adapter_scale":0.25,"retrieval_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.787721387829 | 0.714456643543 | 0.0849178042894 | 0.546021309695 | 410 | False |
| V11/T1/round2 | 相似度检索＋同KC偏置2；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"kc_bias","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.786973285937 | 0.713154870661 | 0.084924856349 | 0.563999530365 | 471 | False |
| V11/T1/round3 | 相似度检索＋同KC偏置2；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"kc_bias","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.787020934277 | 0.714639361705 | 0.0846432652625 | 0.54534942234 | 422 | False |
| V11/T2/round1 | 同KC检索＋全窗口/最近32两尺度；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.25,"retrieval_adapter_kl":0.0,"learning_rate":0.001}} | 11/1 | 0.787882353072 | 0.714726463819 | 0.0848140161325 | 0.545560480202 | 402 | False |
| V11/T2/round2 | 同KC检索＋全窗口/最近32两尺度；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.003}} | 11/1 | 0.787298747009 | 0.71351860775 | 0.084708673476 | 0.573198508908 | 451 | False |
| V11/T2/round3 | 同KC检索＋全窗口/最近32两尺度；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":0.0,"learning_rate":0.0003}} | 11/1 | 0.786920813031 | 0.714797258919 | 0.0847111537933 | 0.539126478617 | 402 | False |
| V11/T3/round1 | 两尺度检索＋冻结基线KL1；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.25,"retrieval_adapter_kl":1.0,"learning_rate":0.001}} | 11/1 | 0.788357091504 | 0.715172368434 | 0.0850307338211 | 0.552472922598 | 336 | False |
| V11/T3/round2 | 两尺度检索＋冻结基线KL1；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":1.0,"learning_rate":0.003}} | 11/1 | 0.788285083838 | 0.71461575424 | 0.084732704106 | 0.560342246617 | 384 | False |
| V11/T3/round3 | 两尺度检索＋冻结基线KL1；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"retrieval_adapter_mode":"multi_scale","retrieval_adapter_scale":0.5,"retrieval_adapter_kl":1.0,"learning_rate":0.0003}} | 11/1 | 0.788082818946 | 0.715232063995 | 0.084690002433 | 0.546499750506 | 381 | False |
| V12/M0/round1 | 原基线＋增强检索修正；{"strength":1.5,"transfer_mode":"raw"} | CPU：无新训练 | 0.786696464632 | 0.7146753205 | 0.0844581088526 | 0.540030526285 | 365 | False |
| V12/M0/round2 | 原基线＋增强检索修正；{"strength":2.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.784875636751 | 0.713887521182 | 0.0845946960888 | 0.53634389034 | 404 | False |
| V12/M0/round3 | 原基线＋增强检索修正；{"strength":3.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.778967284868 | 0.710715482784 | 0.0850299453585 | **0.531280636356** | 475 | False |
| V12/M1/round1 | 三seed性能锚＋检索logit修正；{"strength":1.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.789041157308 | 0.716001881168 | 0.0852371019277 | 0.554665531715 | 314 | False |
| V12/M1/round2 | 三seed性能锚＋检索logit修正；{"strength":2.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.786244368509 | 0.714897785626 | 0.0852784816493 | 0.546831430333 | 363 | False |
| V12/M1/round3 | 三seed性能锚＋检索logit修正；{"strength":3.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.780572965818 | 0.711694836461 | 0.0857830700842 | 0.547782763945 | 426 | False |
| V12/M2/round1 | 三seed锚＋置信门控检索修正；{"strength":1.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.789230485535 | 0.715995455255 | 0.0853405268336 | 0.553743872729 | 321 | False |
| V12/M2/round2 | 三seed锚＋置信门控检索修正；{"strength":2.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.787589760931 | 0.715500511891 | 0.0852403372909 | 0.554204702222 | 342 | False |
| V12/M2/round3 | 三seed锚＋置信门控检索修正；{"strength":3.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.784595866859 | 0.713903385772 | 0.0854425015329 | 0.547753089319 | 383 | False |
| V12/M3/round1 | 三seed锚＋严格前缀预测差中心化；{"strength":1.0,"transfer_mode":"prefix_center"} | CPU：无新训练 | 0.78949931596 | 0.715186489343 | 0.0853843883954 | 0.556981420059 | 343 | False |
| V12/M3/round2 | 三seed锚＋严格前缀预测差中心化；{"strength":2.0,"transfer_mode":"prefix_center"} | CPU：无新训练 | 0.787948268254 | 0.712957280688 | 0.0854997265003 | 0.552839825061 | 431 | False |
| V12/M3/round3 | 三seed锚＋严格前缀预测差中心化；{"strength":3.0,"transfer_mode":"prefix_center"} | CPU：无新训练 | 0.78483849759 | 0.709130499618 | 0.0858817461108 | 0.55687160504 | 504 | False |
| V12/M4/round1 | 低分散双seed锚＋置信门控修正；{"strength":1.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.788614861858 | 0.715541179574 | 0.0849006540779 | 0.554327981449 | 313 | False |
| V12/M4/round2 | 低分散双seed锚＋置信门控修正；{"strength":2.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.786841319521 | 0.714950322282 | 0.0848018455802 | 0.54695470956 | 351 | False |
| V12/M4/round3 | 低分散双seed锚＋置信门控修正；{"strength":3.0,"transfer_mode":"uncertain"} | CPU：无新训练 | 0.783672803154 | 0.713481228693 | 0.0847585146758 | 0.539120608178 | 399 | False |
| V13/E0/round1 | 已存多尺度KL修正的有界幅度筛选；{"strength":2.0,"transfer_mode":"raw"} | CPU：无新训练 | 0.788027568953 | 0.714927410645 | 0.0843073790487 | 0.534494701928 | 376 | False |
| V13/E0/round2 | 已存多尺度KL修正的有界幅度筛选；{"strength":1.5,"transfer_mode":"raw"} | CPU：无新训练 | 0.789079767157 | 0.715684022947 | 0.084177421779 | 0.541873844257 | 343 | True |
| V14/P0/round1 | 多尺度＋KL1＋序列等权BCE；{"alpha":0.25,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003}} | 11/1 | 0.788618699873 | 0.715211874088 | 0.0847879388606 | 0.553400452024 | 354 | False |
| V14/P0/round2 | 多尺度＋KL1＋序列等权BCE；{"alpha":0.5,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003}} | 11/1 | 0.788603652783 | 0.715210604874 | 0.084799873072 | 0.555243769996 | 352 | False |
| V14/P0/round3 | 多尺度＋KL1＋序列等权BCE；{"alpha":1.0,"rank_weight":0.0,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003}} | 11/1 | 0.788525336462 | 0.715229903081 | 0.084909359297 | 0.561695382899 | 353 | False |
| V14/P1/round1 | 多尺度＋KL1＋完整UID平滑排序；{"alpha":0.0,"rank_weight":0.03,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":0.0} | 11/1 | 0.78868436811 | 0.71515562935 | 0.0848506896134 | 0.553406322463 | 354 | False |
| V14/P1/round2 | 多尺度＋KL1＋完整UID平滑排序；{"alpha":0.0,"rank_weight":0.1,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":0.0} | 11/1 | 0.788723942298 | 0.715407054971 | 0.0848577332833 | 0.555710469929 | 334 | False |
| V14/P1/round3 | 多尺度＋KL1＋完整UID平滑排序；{"alpha":0.0,"rank_weight":0.3,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":0.0} | 11/1 | 0.788788709755 | 0.715968271896 | 0.0848813254796 | 0.556165428982 | 277 | False |
| V14/P2/round1 | 多尺度＋KL1＋排序与类别支持权重；{"alpha":0.0,"rank_weight":0.03,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":5.0} | 21/11 | 0.789445252295 | 0.716084287714 | 0.0842189513607 | 0.558463706008 | 298 | False |
| V14/P2/round2 | 多尺度＋KL1＋排序与类别支持权重；{"alpha":0.0,"rank_weight":0.1,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":5.0} | 11/1 | 0.788725545801 | 0.715406081625 | 0.084847342516 | 0.555249640436 | 331 | False |
| V14/P2/round3 | 多尺度＋KL1＋排序与类别支持权重；{"alpha":0.0,"rank_weight":0.3,"model_settings":{"causal_adapter_mode":"multiscale","causal_adapter_scale":0.5,"causal_adapter_kl":1.0,"learning_rate":0.003},"cached_rank_reliability":5.0} | 11/1 | 0.788791864307 | 0.715970647243 | 0.0848962818196 | 0.556165428982 | 279 | False |
| V3/C1/round1 | 基线保护组合；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"item_residual":1}} | 14/4 | 0.788804207132 | 0.714845158514 | 0.0848417580581 | 0.552933752091 | 378 | False |
| V3/C2/round1 | 固定融合；学习路由尚缺OOF前置；{"alpha":0.02,"rank_weight":0.0005,"model_settings":{}} | 14/4 | 0.788755868977 | 0.71495707412 | 0.0849520451946 | 0.555237899557 | 361 | False |
| V3/C3/round1 | V3-C3；{"alpha":0.0,"rank_weight":0.0,"model_settings":{"training_overlap":64}} | 14/4 | 0.789407666616 | 0.716242570491 | 0.0853663137858 | 0.547031025272 | 289 | False |
| V3/P1/round1 | V3-P1；{"alpha":0.0,"rank_weight":0.0,"frequency_alpha":0.1,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.788779160654 | 0.714931324383 | 0.0847584235703 | 0.554310370131 | 375 | False |
| V3/P1/round2 | V3-P1；{"alpha":0.0,"rank_weight":0.0,"frequency_alpha":0.05,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.78875774303 | 0.714869037225 | 0.084902710148 | 0.556159558543 | 378 | False |
| V3/P1/round3 | V3-P1；{"alpha":0.0,"rank_weight":0.0,"frequency_alpha":0.02,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.788825358223 | 0.714936446123 | 0.0847772411455 | 0.553849540638 | 365 | False |
| V3/P2/round1 | V3-P2；{"alpha":0.0,"rank_weight":0.0005,"frequency_alpha":0.1,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.788787664234 | 0.714959512874 | 0.0848251237519 | 0.556159558543 | 358 | False |
| V3/P2/round2 | V3-P2；{"alpha":0.0,"rank_weight":0.00025,"frequency_alpha":0.05,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.788772950065 | 0.714902690416 | 0.0848627048952 | 0.553849540638 | 368 | False |
| V3/P2/round3 | V3-P2；{"alpha":0.0,"rank_weight":0.0001,"frequency_alpha":0.02,"frequency_cap":1.5,"model_settings":{"item_residual":1}} | 14/4 | 0.78879801191 | 0.714864975228 | 0.0848976214578 | 0.555232029117 | 385 | False |
| V3/P3/round1 | V3-P3；{"alpha":0.0,"rank_weight":0.0005,"frequency_alpha":0.1,"frequency_cap":1.5,"model_settings":{"training_overlap":64}} | 14/4 | 0.789394928294 | 0.71625382614 | 0.0853280752642 | 0.541506941795 | 296 | False |
| V3/P3/round2 | V3-P3；{"alpha":0.0,"rank_weight":0.00025,"frequency_alpha":0.05,"frequency_cap":1.5,"model_settings":{"training_overlap":64}} | 14/4 | 0.789394437272 | 0.716276111092 | 0.0853231477017 | 0.54793507294 | 292 | False |
| V3/P3/round3 | V3-P3；{"alpha":0.0,"rank_weight":0.0001,"frequency_alpha":0.02,"frequency_cap":1.5,"model_settings":{"training_overlap":64}} | 14/4 | 0.789376340211 | 0.716187693101 | 0.0852743300207 | 0.546576066219 | 297 | False |
| V3/S1/round1 | V3-S1；{"candidate_weight":0.025} | CPU：无新训练 | 0.788780272194 | 0.714934585408 | 0.084756206698 | 0.553849540638 | 369 | False |
| V3/S1/round2 | V3-S1；{"candidate_weight":0.05} | CPU：无新训练 | 0.788781415124 | 0.714950685097 | 0.0847485876067 | 0.552467052159 | 370 | False |
| V3/S1/round3 | V3-S1；{"candidate_weight":0.1} | CPU：无新训练 | 0.7887825399 | 0.714974595162 | 0.0847453064851 | 0.552012093105 | 365 | False |
| V3/S2/round1 | V3-S2；{"candidate_weight":0.025} | CPU：无新训练 | 0.788788966658 | 0.714959686281 | 0.0848016138488 | 0.55569872905 | 367 | False |
| V3/S2/round2 | V3-S2；{"candidate_weight":0.05} | CPU：无新训练 | 0.788789850636 | 0.714955360755 | 0.0848080795378 | 0.555237899557 | 363 | False |
| V3/S2/round3 | V3-S2；{"candidate_weight":0.1} | CPU：无新训练 | 0.78879069683 | 0.714985950027 | 0.0848203231332 | 0.555232029117 | 356 | False |
| V3/S3/round1 | V3-S3；{"candidate_weight":0.025} | CPU：无新训练 | 0.789406208878 | 0.716251405035 | 0.0853445429993 | 0.541967771288 | 298 | False |
| V3/S3/round2 | V3-S3；{"candidate_weight":0.05} | CPU：无新训练 | 0.789415437721 | 0.716279437562 | 0.0853486322178 | 0.544709266489 | 290 | False |
| V3/S3/round3 | V3-S3；{"candidate_weight":0.1} | CPU：无新训练 | 0.789430653279 | 0.716268837396 | 0.0853773556073 | 0.544703396049 | 289 | False |
| V4/Q0/round1 | 完整UID分组、原损失对照；{"alpha":0.0,"rank_weight":0.0} | 14/4 | 0.78874488226 | 0.714898020891 | 0.0857720202282 | 0.566007062578 | 372 | False |
| V4/Q1/round1 | 完整UID跨片段logistic排序；{"alpha":0.0,"rank_weight":0.002} | 14/4 | 0.788769541027 | 0.714845203921 | 0.0857553516918 | 0.564008397668 | 371 | False |
| V4/Q1/round2 | 完整UID跨片段logistic排序；{"alpha":0.0,"rank_weight":0.01} | 14/4 | 0.788730392169 | 0.71483514815 | 0.0858280616135 | 0.564738478642 | 374 | False |
| V4/Q1/round3 | 完整UID跨片段logistic排序；{"alpha":0.0,"rank_weight":0.03} | 14/4 | 0.788724080277 | 0.714822704968 | 0.0858470320175 | 0.563469894707 | 375 | False |
| V4/Q2/round1 | 完整UID有界平滑AUC风险；{"alpha":0.0,"rank_weight":0.03} | 14/4 | 0.78875200916 | 0.714921476891 | 0.0857828715031 | 0.565427117222 | 364 | False |
| V4/Q2/round2 | 完整UID有界平滑AUC风险；{"alpha":0.0,"rank_weight":0.1} | 14/4 | 0.788721445842 | 0.715106362847 | 0.085669979523 | 0.561010400304 | 365 | False |
| V4/Q2/round3 | 完整UID有界平滑AUC风险；{"alpha":0.0,"rank_weight":0.3} | 14/4 | 0.788682315999 | 0.71587239052 | 0.0858251355744 | 0.555123098799 | 345 | False |
| V4/Q3/round1 | 完整UID平滑风险＋类别支持权重；{"alpha":0.0,"rank_weight":0.03} | 14/4 | 0.78874308484 | 0.71485481352 | 0.0857565167726 | 0.562470562252 | 363 | False |
| V4/Q3/round2 | 完整UID平滑风险＋类别支持权重；{"alpha":0.0,"rank_weight":0.1} | 14/4 | 0.788713633512 | 0.715093269094 | 0.0857043018603 | 0.561507460872 | 365 | False |
| V4/Q3/round3 | 完整UID平滑风险＋类别支持权重；{"alpha":0.0,"rank_weight":0.3} | 14/4 | 0.788672241618 | 0.715731807244 | 0.0857457536324 | 0.557313341721 | 341 | False |
| V4/Q4/round1 | 完整UID平滑风险＋类别支持＋温和在线关注；{"alpha":0.0,"rank_weight":0.03} | 14/4 | 0.788765748376 | 0.714902850393 | 0.0857137622978 | 0.564313880224 | 366 | False |
| V4/Q4/round2 | 完整UID平滑风险＋类别支持＋温和在线关注；{"alpha":0.0,"rank_weight":0.1} | 14/4 | 0.78870218032 | 0.714929743022 | 0.0857627903493 | 0.564541689311 | 368 | False |
| V4/Q4/round3 | 完整UID平滑风险＋类别支持＋温和在线关注；{"alpha":0.0,"rank_weight":0.3} | 14/4 | 0.788667935283 | 0.715543267332 | 0.0855777580218 | 0.558043422695 | 349 | False |
| V5/G0/round1 | 原sampler/generator＋索引包装、原损失对照；{"alpha":0.0,"rank_weight":0.0} | 14/4 | 0.788805100658 | 0.715013582279 | 0.0848136345456 | 0.552478793038 | 362 | 控制 |
| V5/G1/round1 | 原批次＋完整序列logistic梯度缓存；{"alpha":0.0,"rank_weight":0.002,"cached_rank_reliability":0.0} | 14/4 | 0.788804432727 | 0.714983710311 | 0.0849239241152 | 0.556614517597 | 355 | False |
| V5/G1/round2 | 原批次＋完整序列logistic梯度缓存；{"alpha":0.0,"rank_weight":0.01,"cached_rank_reliability":0.0} | 14/4 | 0.788822989952 | 0.715006189955 | 0.084808992149 | 0.552478793038 | 337 | True |
| V5/G2/round1 | 原批次＋完整序列平滑风险梯度缓存；{"alpha":0.0,"rank_weight":0.03,"cached_rank_reliability":0.0} | 14/4 | 0.788781821276 | 0.715079510348 | 0.0848528747882 | 0.552922011213 | 331 | False |
| V5/G2/round2 | 原批次＋完整序列平滑风险梯度缓存；{"alpha":0.0,"rank_weight":0.1,"cached_rank_reliability":0.0} | 14/4 | 0.788761063715 | 0.715362430931 | 0.0848883444518 | 0.549719686519 | 324 | False |
| V5/G2/round3 | 原批次＋完整序列平滑风险梯度缓存；{"alpha":0.0,"rank_weight":0.3,"cached_rank_reliability":0.0} | 15/5 | 0.788825343634 | **0.716401796811** | 0.0851307677686 | 0.570806304852 | 307 | False |
| V5/G3/round1 | 原批次＋平滑风险梯度缓存＋类别支持权重；{"alpha":0.0,"rank_weight":0.03,"cached_rank_reliability":5.0} | 14/4 | 0.788758913458 | 0.715010884596 | 0.0848391814695 | 0.55108456368 | 347 | False |
| V5/G3/round2 | 原批次＋平滑风险梯度缓存＋类别支持权重；{"alpha":0.0,"rank_weight":0.1,"cached_rank_reliability":5.0} | 14/4 | 0.78874113899 | 0.7151192525 | 0.0848846009681 | 0.555237899557 | 336 | False |
| V5/G3/round3 | 原批次＋平滑风险梯度缓存＋类别支持权重；{"alpha":0.0,"rank_weight":0.3,"cached_rank_reliability":5.0} | 15/5 | 0.788783784952 | 0.716245827806 | 0.0851158130245 | 0.568029587015 | 296 | False |
| V6融合/J1/round1 | 固定题目＋前缀历史融合；{"candidate_weight":0.25} | CPU：无新训练 | 0.788164806043 | 0.713607362526 | 0.0842650776305 | 0.558481317327 | 502 | False |
| V6融合/J1/round2 | 固定题目＋前缀历史融合；{"candidate_weight":0.5} | CPU：无新训练 | 0.788089146082 | 0.713987873789 | 0.0843923097562 | 0.550635475065 | 482 | False |
| V6融合/J1/round3 | 固定题目＋前缀历史融合；{"candidate_weight":0.75} | CPU：无新训练 | 0.787869984799 | 0.714248384282 | 0.0846048084546 | 0.550174645572 | 447 | False |
| V6融合/J2/round1 | 固定题目＋多尺度历史融合；{"candidate_weight":0.25} | CPU：无新训练 | 0.788343253199 | 0.713888796445 | 0.0841917208261 | 0.558020487834 | 489 | False |
| V6融合/J2/round2 | 固定题目＋多尺度历史融合；{"candidate_weight":0.5} | CPU：无新训练 | 0.78842616259 | 0.714394037746 | 0.0842715403848 | 0.552478793038 | 433 | False |
| V6融合/J2/round3 | 固定题目＋多尺度历史融合；{"candidate_weight":0.75} | CPU：无新训练 | 0.788345824443 | 0.71472787055 | 0.084421208053 | 0.549719686519 | 400 | False |
| V6融合/J3/round1 | 固定多尺度＋KL保护分支融合；{"candidate_weight":0.25} | CPU：无新训练 | 0.788288228719 | 0.715065073501 | 0.0845952847255 | 0.552472922598 | 338 | False |
| V6融合/J3/round2 | 固定多尺度＋KL保护分支融合；{"candidate_weight":0.5} | CPU：无新训练 | 0.788443692367 | 0.715133127693 | 0.0845506524564 | 0.554316240571 | 328 | False |
| V6融合/J3/round3 | 固定多尺度＋KL保护分支融合；{"candidate_weight":0.75} | CPU：无新训练 | 0.788569337794 | 0.715199812075 | 0.084503128614 | 0.554777070064 | 344 | False |

尾部使用两种口径：自身尾部随模型改变成员；固定弱/强组由原baseline定义。固定组增益不能替代原始std/range。

| 方案 | 改动 | 自身尾10%↑ | 自身尾20%↑ | 固定弱10%↑ | 固定弱20%↑ | 固定强25%↑ | q95-q05↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 原始模型 | 0.540628428066 | 0.583734493988 | 0.540628428066 | 0.583734493988 | 0.810171567152 | 0.284445431374 |
| V2/A1/round1 | 序列等权BCE | 0.540966372967 | 0.584172857569 | 0.540990277751 | 0.58429496374 | 0.810308456808 | 0.284271348307 |
| V2/A1/round2 | 序列等权BCE | 0.540759237499 | 0.584071021715 | 0.540759237499 | 0.584098022251 | 0.810423205461 | 0.290138471935 |
| V2/A2/round1 | 平滑困难序列加权 | 0.541551738555 | 0.584515400796 | 0.541551738555 | 0.584679724036 | 0.809907715581 | 0.284008891217 |
| V2/A2/round2 | 平滑困难序列加权 | 0.54133478141 | 0.584498958733 | 0.541367628134 | 0.584647411104 | 0.810224083907 | 0.284234958059 |
| V2/A3/round1 | 序列内排序 | 0.542027071702 | 0.584771055862 | 0.542068440927 | 0.584911967448 | 0.809906430519 | 0.28418495101 |
| V2/A3/round2 | 序列内排序 | 0.541530973761 | 0.584514594487 | 0.541548270434 | 0.584655423425 | 0.810043026854 | 0.284706755007 |
| V2/B1/round1 | 低频覆盖/固定融合，细节见参数 | 0.54146725673 | 0.584215269437 | 0.541485897629 | 0.584248311345 | 0.810251489202 | 0.282804206267 |
| V2/B1/round2 | 低频覆盖/固定融合，细节见参数 | 0.541155105436 | 0.584180215496 | 0.541195067732 | 0.584204752462 | 0.810349078613 | 0.285041087125 |
| V2/B2/round1 | 冻结主干前缀小适配 | 0.540655461249 | 0.583774432948 | 0.540655461249 | 0.583774432948 | 0.810143748471 | 0.284272947192 |
| V2/B2/round2 | 冻结主干前缀小适配 | 0.540654461828 | 0.583774123308 | 0.540654461828 | 0.583774123308 | 0.810144162645 | 0.284272947192 |
| V2/C1/round1 | 基线保护组合 | 0.54062883365 | 0.583734698089 | 0.54062883365 | 0.583734698089 | 0.810171545412 | 0.284445431374 |
| V2/C1/round2 | 基线保护组合 | 0.54062883365 | 0.583734698089 | 0.54062883365 | 0.583734698089 | 0.810171545412 | 0.284445431374 |
| D07/D1/round1 | D1 | 0.541366685655 | 0.584183441143 | 0.541395958525 | 0.584263680632 | 0.81022749732 | 0.285846400719 |
| D07/D1/round2 | D1 | 0.541004661575 | 0.584102573297 | 0.541024605946 | 0.584118608234 | 0.810259611668 | 0.285099504585 |
| D07/D1/round3 | D1 | 0.541096547431 | 0.583929471985 | 0.541156687407 | 0.584005008461 | 0.81029601039 | 0.286057808382 |
| D07/D2/round1 | D2 | 0.541020135024 | 0.58411770973 | 0.541035826729 | 0.584234276571 | 0.810195062962 | 0.285758288006 |
| D07/D2/round2 | D2 | 0.540681432122 | 0.583925087508 | 0.540741895089 | 0.583985370383 | 0.810221584774 | 0.285715862765 |
| D07/D2/round3 | D2 | 0.54061790496 | 0.583720267534 | 0.54064778832 | 0.583736257046 | 0.810241755291 | 0.285908324919 |
| D07/D3/round1 | D3 | 0.540849418766 | 0.583450823552 | 0.541243842129 | 0.583834273184 | 0.810750776317 | 0.286665204869 |
| D07/D3/round2 | D3 | 0.540657179224 | 0.583351986403 | 0.540938349192 | 0.58373599677 | 0.81078849595 | 0.288116236496 |
| D07/D3/round3 | D3 | 0.540891579017 | 0.583544654599 | 0.541265111361 | 0.583923214935 | 0.810995436245 | 0.287373175298 |
| D07/D4/round1 | D4 | 0.540803008487 | 0.584001035199 | 0.540852798162 | 0.584092986911 | 0.810167021386 | 0.285421978563 |
| D07/D4/round2 | D4 | 0.541126738998 | 0.584121008757 | 0.54120202555 | 0.584172899877 | 0.810228362154 | 0.287121174931 |
| D07/D4/round3 | D4 | 0.540873727641 | 0.583899909279 | 0.54091044769 | 0.583915434058 | 0.810342802712 | 0.286559562509 |
| D07/D5/round1 | D5 | 0.541666684972 | 0.584285064104 | 0.541918441009 | 0.58487293293 | 0.811794407057 | 0.283176370651 |
| D07/D5/round2 | D5 | 0.541811554717 | 0.584529049335 | 0.54213831732 | 0.585057792532 | 0.811945202084 | 0.28237762313 |
| D07/D5/round3 | D5 | 0.542185485662 | 0.584485968784 | 0.542472631357 | 0.585063637753 | 0.811917095497 | 0.284056207181 |
| D07/D6/round1 | D6 | 0.54083004325 | 0.583727151628 | 0.540884399065 | 0.583808485602 | 0.810309036823 | 0.285633099294 |
| D07/D6/round2 | D6 | 0.541212096882 | 0.58389930856 | 0.541212096882 | 0.583946715631 | 0.810344241824 | 0.286308073615 |
| D07/D6/round3 | D6 | 0.541023944733 | 0.583906324123 | 0.541067021332 | 0.583958001133 | 0.810276692071 | 0.286068385616 |
| D07/E1/round1 | E1 | 0.540820038488 | 0.583969859713 | 0.540820038488 | 0.583990137329 | 0.810177541176 | 0.285274075364 |
| D07/E1/round2 | E1 | 0.541086425629 | 0.584160293614 | 0.541100313679 | 0.584247703336 | 0.810272333106 | 0.286964331212 |
| D07/E1/round3 | E1 | 0.541061039114 | 0.584241431644 | 0.541078896021 | 0.584362405243 | 0.810352747474 | 0.284341768507 |
| D07/E2/round1 | E2 | 0.540919416957 | 0.584018925619 | 0.540919416957 | 0.584037389359 | 0.810251962734 | 0.285428281719 |
| D07/E2/round2 | E2 | 0.541164994738 | 0.584193217505 | 0.541164994738 | 0.584253629758 | 0.810378686598 | 0.287484521433 |
| D07/E2/round3 | E2 | 0.541143338221 | 0.584299040223 | 0.541155036751 | 0.584404485757 | 0.810306428931 | 0.285837854378 |
| D07/E3/round1 | E3 | 0.540960660835 | 0.584081292363 | 0.540960660835 | 0.584121493956 | 0.81015614574 | 0.284803093842 |
| D07/E3/round2 | E3 | 0.541128821859 | 0.584298077599 | 0.541128821859 | 0.584366274379 | 0.810239021572 | 0.286686277803 |
| D07/E3/round3 | E3 | 0.541340444537 | 0.584442070989 | 0.541341894821 | 0.584551723408 | 0.810227150979 | 0.285737617833 |
| D08/F1/round1 | F1 | 0.541154088227 | 0.584163669004 | 0.541196515939 | 0.584181757928 | 0.810307346796 | 0.284597245401 |
| D08/F2/round1 | F2 | 0.541222810967 | 0.584214862812 | 0.541269114783 | 0.584233040631 | 0.810334786481 | 0.284759583063 |
| D08/F3/round1 | F3 | 0.54119173739 | 0.584210008805 | 0.541235662902 | 0.584228008834 | 0.810329185001 | 0.284660930055 |
| V6/I0/round1 | 当前题64维投影＋非线性残差 | 0.54190709772 | 0.584130727875 | 0.542257098275 | 0.584251822552 | 0.807530480774 | 0.277731804971 |
| V6/I0/round2 | 当前题64维投影＋非线性残差 | 0.541116183581 | 0.583258914842 | 0.541797421093 | 0.583444797828 | 0.805840045724 | 0.280322674384 |
| V6/I0/round3 | 当前题64维投影＋非线性残差 | 0.54108353075 | 0.583635139891 | 0.54135099347 | 0.583835965336 | 0.808043214451 | 0.281948865278 |
| V6/I1/round1 | 题目残差＋KC与全局前缀状态 | 0.541027926682 | 0.583543711782 | 0.541454232407 | 0.5836930866 | 0.809834948511 | 0.285380049548 |
| V6/I1/round2 | 题目残差＋KC与全局前缀状态 | 0.542780259041 | 0.584710165964 | 0.543504489201 | 0.584948908491 | 0.808775644875 | 0.28318599595 |
| V6/I1/round3 | 题目残差＋KC与全局前缀状态 | 0.541122603345 | 0.583811014015 | 0.541554183075 | 0.584064053609 | 0.808789150374 | 0.281033822468 |
| V6/I2/round1 | 前缀＋8/32步多尺度历史＋最近KC响应 | 0.542057719893 | 0.584579895678 | 0.542222988186 | 0.584766995521 | 0.81017763891 | 0.284658606824 |
| V6/I2/round2 | 前缀＋8/32步多尺度历史＋最近KC响应 | 0.542926429562 | 0.585119269972 | 0.543227645025 | 0.585549996826 | 0.809700962345 | 0.282724057951 |
| V6/I2/round3 | 前缀＋8/32步多尺度历史＋最近KC响应 | 0.544817390864 | 0.585980859812 | 0.54519190918 | 0.586393489248 | 0.8100218782 | 0.279659449701 |
| V6/I3/round1 | 多尺度历史＋序列等权BCE .25 | 0.541637275488 | 0.584480145284 | 0.541789093774 | 0.58462355124 | 0.810252454055 | 0.284794461194 |
| V6/I3/round2 | 多尺度历史＋序列等权BCE .25 | 0.542432190581 | 0.584898277963 | 0.542555491583 | 0.585348597064 | 0.810016791858 | 0.284222086614 |
| V6/I3/round3 | 多尺度历史＋序列等权BCE .25 | 0.544647942194 | 0.585957594245 | 0.544934234566 | 0.586402086833 | 0.810092285233 | 0.280941810328 |
| V6/I4/round1 | 多尺度历史＋冻结基线KL1 | 0.542150788122 | 0.584896513441 | 0.542232973184 | 0.585041352983 | 0.810430520489 | 0.283410746305 |
| V6/I4/round2 | 多尺度历史＋冻结基线KL1 | 0.544799742675 | 0.586300375325 | 0.545271803803 | 0.586616188821 | 0.811125749408 | 0.281339341397 |
| V6/I5/round1 | 多尺度历史＋类别支持度门控 | 0.541662331964 | 0.584345578788 | 0.541850777675 | 0.584407588743 | 0.810161731882 | 0.282150272974 |
| V6/I5/round2 | 多尺度历史＋类别支持度门控 | 0.542311552471 | 0.584095955345 | 0.542743044135 | 0.584577403043 | 0.810118014154 | 0.283325719162 |
| V6/I5/round3 | 多尺度历史＋类别支持度门控 | 0.541371505405 | 0.58391827147 | 0.541574358852 | 0.584199774475 | 0.809870646223 | 0.27987439739 |
| V7/U0/round1 | 按冻结baseline不确定性限制修正 | 0.542056611091 | 0.584725209653 | 0.542126265162 | 0.584824063528 | 0.810199467941 | 0.284527717302 |
| V7/U0/round2 | 按冻结baseline不确定性限制修正 | 0.542706708268 | 0.584587303136 | 0.543316186247 | 0.584981477272 | 0.809704140179 | 0.281999776968 |
| V7/U0/round3 | 按冻结baseline不确定性限制修正 | 0.541404199691 | 0.584780061437 | 0.541460526227 | 0.584964483648 | 0.810245356685 | 0.284746215382 |
| V7/U1/round1 | 按冻结预测头分歧限制修正 | 0.541298057834 | 0.583997425606 | 0.541330026417 | 0.584074386616 | 0.810363399732 | 0.283870599347 |
| V7/U1/round2 | 按冻结预测头分歧限制修正 | 0.540838891336 | 0.583304780466 | 0.540899033881 | 0.58353999162 | 0.809939271801 | 0.28696354238 |
| V7/U1/round3 | 按冻结预测头分歧限制修正 | 0.541079507702 | 0.583493858931 | 0.541146636035 | 0.583605170877 | 0.809829019311 | 0.284100884317 |
| V7/U2/round1 | 优先低KC历史支持的位置 | 0.54139636816 | 0.584436002963 | 0.541458845083 | 0.584533312095 | 0.809917543196 | 0.283302923953 |
| V7/U2/round2 | 优先低KC历史支持的位置 | 0.542735034996 | 0.584882744017 | 0.542997618098 | 0.585253847505 | 0.809746931111 | 0.286481483884 |
| V7/U2/round3 | 优先低KC历史支持的位置 | 0.54055695004 | 0.584349144243 | 0.540657960536 | 0.584508106787 | 0.809451707843 | 0.28303880029 |
| V8/K1/round1 | seed42＋43固定概率平均 | 0.541188007176 | 0.584391565987 | 0.541253631016 | 0.584494985805 | 0.810922104806 | 0.283279873711 |
| V8/K1/round2 | seed42＋43固定概率平均 | 0.540537030954 | 0.584429169718 | 0.540911943508 | 0.584692247668 | 0.81104536748 | 0.286960404079 |
| V8/K1/round3 | seed42＋43固定概率平均 | 0.540005430796 | 0.584180081628 | 0.540669449886 | 0.584767207739 | 0.810833125874 | 0.284637796094 |
| V8/K2/round1 | seed42＋44固定概率平均 | 0.540610115682 | 0.584005139 | 0.540618187075 | 0.584117301005 | 0.810954617642 | 0.28589761585 |
| V8/K2/round2 | seed42＋44固定概率平均 | 0.539437487188 | 0.583653299169 | 0.539598463538 | 0.583950138938 | 0.811656115327 | 0.292402226899 |
| V8/K2/round3 | seed42＋44固定概率平均 | 0.538561048461 | 0.583024569404 | 0.539058867417 | 0.583489263068 | 0.811639841972 | 0.291944017308 |
| V8/K3/round1 | seed43＋44固定概率平均，对原42比较 | 0.539069957171 | 0.583672515565 | 0.539823400064 | 0.5845419445 | 0.811147687323 | 0.285288339746 |
| V8/K3/round2 | seed43＋44固定概率平均，对原42比较 | 0.538419069547 | 0.583362037953 | 0.539099399064 | 0.584199247494 | 0.811608339802 | 0.289379818209 |
| V8/K3/round3 | seed43＋44固定概率平均，对原42比较 | 0.538142500208 | 0.582853573878 | 0.53888593483 | 0.583593001534 | 0.811452237361 | 0.287963318088 |
| V8/K4/round1 | seed42/43/44各1/3概率平均 | 0.539913762141 | 0.584198536462 | 0.540339265442 | 0.584634111469 | 0.811782906702 | 0.288370014376 |
| V8/K5/round1 | seed42/43/44各1/3 logit平均 | 0.540065484845 | 0.584254597578 | 0.540471895095 | 0.584699322355 | 0.811806398941 | 0.288781427452 |
| V9/L1/round1 | 三seed性能锚＋题目修正R1 | 0.540787775059 | 0.584429679264 | 0.541044676641 | 0.58466442128 | 0.81116627118 | 0.287070688314 |
| V9/L1/round2 | 三seed性能锚＋题目修正R1 | 0.541012506552 | 0.584420225781 | 0.541207657517 | 0.584574584649 | 0.81009049816 | 0.281629451946 |
| V9/L1/round3 | 三seed性能锚＋题目修正R1 | 0.541569832556 | 0.584450704576 | 0.541814253634 | 0.584562534371 | 0.808982353608 | 0.278526030756 |
| V9/L2/round1 | 三seed性能锚＋小lr题目修正R3 | 0.540328819419 | 0.584251691374 | 0.540564529877 | 0.584547241611 | 0.811183078268 | 0.284053325853 |
| V9/L2/round2 | 三seed性能锚＋小lr题目修正R3 | 0.540545705311 | 0.584134474852 | 0.540698275222 | 0.584343625124 | 0.810310562025 | 0.282180213493 |
| V9/L2/round3 | 三seed性能锚＋小lr题目修正R3 | 0.540577343442 | 0.583899902073 | 0.540764772625 | 0.584107269141 | 0.809239249395 | 0.281215492393 |
| V9/L3/round1 | 三seed性能锚＋困难加权A3R2 | 0.540812095041 | 0.584768581406 | 0.54102164671 | 0.584951385306 | 0.811675248407 | 0.288491804392 |
| V9/L3/round2 | 三seed性能锚＋困难加权A3R2 | 0.541153559221 | 0.584758225117 | 0.541219077847 | 0.584894471142 | 0.811168161015 | 0.284563709147 |
| V9/L3/round3 | 三seed性能锚＋困难加权A3R2 | 0.541409985958 | 0.584751167232 | 0.541429527116 | 0.584820470898 | 0.81072004733 | 0.286423381105 |
| V10/R0/round1 | 全局前缀预测误差对照 | 0.537202344145 | 0.57918206907 | 0.537697939893 | 0.579412503112 | 0.808687527691 | 0.289631429534 |
| V10/R0/round2 | 全局前缀预测误差对照 | 0.534647227089 | 0.576305259746 | 0.535666464007 | 0.576617677718 | 0.80740272747 | 0.291682499794 |
| V10/R0/round3 | 全局前缀预测误差对照 | 0.53850340616 | 0.581106698205 | 0.538842475845 | 0.581247455718 | 0.80893291357 | 0.286889021831 |
| V10/R1/round1 | 当前KC前缀预测误差 | 0.542827209338 | 0.582478782989 | 0.543732728804 | 0.58309972385 | 0.809541606847 | 0.285395671006 |
| V10/R1/round2 | 当前KC前缀预测误差 | 0.540768695662 | 0.577778851708 | 0.544696097574 | 0.579120414552 | 0.805780541224 | 0.286960406532 |
| V10/R1/round3 | 当前KC前缀预测误差 | 0.542033837894 | 0.58412071746 | 0.542160310953 | 0.584306722327 | 0.810389898257 | 0.288126298889 |
| V10/R2/round1 | 当前概率桶前缀预测误差 | 0.529467068739 | 0.57106766743 | 0.530220157037 | 0.572282571856 | 0.804680577455 | 0.293216854488 |
| V10/R2/round2 | 当前概率桶前缀预测误差 | 0.520259440154 | 0.559970484615 | 0.523721345565 | 0.562265862692 | 0.797809180549 | 0.291033054785 |
| V10/R2/round3 | 当前概率桶前缀预测误差 | 0.533509973573 | 0.576488716475 | 0.533907866535 | 0.576693513242 | 0.807142275591 | 0.291824011607 |
| V10/R3/round1 | KC误差与概率桶误差各一半 | 0.536784690948 | 0.577627877982 | 0.537470401889 | 0.578229434211 | 0.80773889371 | 0.289479465149 |
| V10/R3/round2 | KC误差与概率桶误差各一半 | 0.534855865721 | 0.572701221862 | 0.537297911572 | 0.574197540585 | 0.804822031627 | 0.285877490176 |
| V10/R3/round3 | KC误差与概率桶误差各一半 | 0.538352874996 | 0.58095638821 | 0.538514689258 | 0.581095300502 | 0.809103652998 | 0.288005858133 |
| V10/R4/round1 | KC×概率桶局部误差 | 0.537508822955 | 0.579634076226 | 0.53804703683 | 0.580150582335 | 0.809407519381 | 0.287775787976 |
| V10/R4/round2 | KC×概率桶局部误差 | 0.532400465495 | 0.572878013997 | 0.535129518876 | 0.574582014094 | 0.80590989265 | 0.290259405938 |
| V10/R4/round3 | KC×概率桶局部误差 | 0.539868742266 | 0.582798550633 | 0.539916530061 | 0.582927213983 | 0.810147455978 | 0.285589795528 |
| V11/T0/round1 | 冻结主干＋题目相似度因果检索 | 0.543189462967 | 0.584872562434 | 0.543589268937 | 0.584983800355 | 0.8100866736 | 0.282828352118 |
| V11/T0/round2 | 冻结主干＋题目相似度因果检索 | 0.541823087614 | 0.582654641999 | 0.542774593126 | 0.583093128037 | 0.809023406379 | 0.287362659661 |
| V11/T0/round3 | 冻结主干＋题目相似度因果检索 | 0.543699343638 | 0.585694290256 | 0.544006911704 | 0.585755956597 | 0.809964036617 | 0.28139516982 |
| V11/T1/round1 | 相似度检索＋同KC偏置2 | 0.541216416537 | 0.583386232688 | 0.541629418519 | 0.583550307072 | 0.809641743504 | 0.285851026366 |
| V11/T1/round2 | 相似度检索＋同KC偏置2 | 0.539910365687 | 0.58290936398 | 0.540750092032 | 0.583236538273 | 0.808955552977 | 0.288372046142 |
| V11/T1/round3 | 相似度检索＋同KC偏置2 | 0.541382974749 | 0.584335247378 | 0.541600576859 | 0.584464315323 | 0.809132797793 | 0.281224806021 |
| V11/T2/round1 | 同KC检索＋全窗口/最近32两尺度 | 0.541510252181 | 0.583988754408 | 0.541831215021 | 0.584305743442 | 0.809900712224 | 0.28346036941 |
| V11/T2/round2 | 同KC检索＋全窗口/最近32两尺度 | 0.540713989976 | 0.582938788124 | 0.541351863572 | 0.583140114521 | 0.808490055322 | 0.279111504568 |
| V11/T2/round3 | 同KC检索＋全窗口/最近32两尺度 | 0.541243300016 | 0.584199917556 | 0.541481941288 | 0.584413496158 | 0.809365146873 | 0.282973823458 |
| V11/T3/round1 | 两尺度检索＋冻结基线KL1 | 0.540495802627 | 0.583990766438 | 0.540639962476 | 0.58405210469 | 0.81032182582 | 0.284272008649 |
| V11/T3/round2 | 两尺度检索＋冻结基线KL1 | 0.540774789711 | 0.584087757623 | 0.541035774403 | 0.584205105331 | 0.809377011279 | 0.2842456539 |
| V11/T3/round3 | 两尺度检索＋冻结基线KL1 | 0.541208383789 | 0.584530281922 | 0.541220467101 | 0.584577323064 | 0.809888323033 | 0.282638934888 |
| V12/M0/round1 | 原基线＋增强检索修正 | 0.543338653782 | 0.584361957324 | 0.544281202239 | 0.58454402288 | 0.809721979414 | 0.284508088148 |
| V12/M0/round2 | 原基线＋增强检索修正 | 0.543228823752 | 0.583524984504 | 0.545036772814 | 0.583847339865 | 0.80930473454 | 0.284697222731 |
| V12/M0/round3 | 原基线＋增强检索修正 | 0.54147721178 | 0.580596261629 | 0.546079777552 | 0.581573798881 | 0.807101792264 | 0.286151141551 |
| V12/M1/round1 | 三seed性能锚＋检索logit修正 | 0.541470153249 | 0.584634020003 | 0.542278592942 | 0.585096465568 | 0.811717614352 | 0.28528278863 |
| V12/M1/round2 | 三seed性能锚＋检索logit修正 | 0.541991503262 | 0.583628397596 | 0.544078830914 | 0.584344454135 | 0.810912975965 | 0.288524007711 |
| V12/M1/round3 | 三seed性能锚＋检索logit修正 | 0.540175234378 | 0.580462750313 | 0.544873410902 | 0.581834132064 | 0.808827159001 | 0.289374928769 |
| V12/M2/round1 | 三seed锚＋置信门控检索修正 | 0.540643445952 | 0.58442457843 | 0.541425490431 | 0.584896689211 | 0.811810759388 | 0.285014839692 |
| V12/M2/round2 | 三seed锚＋置信门控检索修正 | 0.541197213014 | 0.584263858338 | 0.542308281747 | 0.584750237149 | 0.811262280096 | 0.288713656365 |
| V12/M2/round3 | 三seed锚＋置信门控检索修正 | 0.540624334658 | 0.582557280491 | 0.542597757606 | 0.583335667027 | 0.810091962344 | 0.289919418484 |
| V12/M3/round1 | 三seed锚＋严格前缀预测差中心化 | 0.540137124145 | 0.583597818714 | 0.540927558148 | 0.584076864903 | 0.811119316309 | 0.285044428216 |
| V12/M3/round2 | 三seed锚＋严格前缀预测差中心化 | 0.539174767791 | 0.581445388786 | 0.541138990585 | 0.582256676173 | 0.809206300334 | 0.285120050513 |
| V12/M3/round3 | 三seed锚＋严格前缀预测差中心化 | 0.537083710664 | 0.577850115633 | 0.54107255073 | 0.579160445659 | 0.806407517694 | 0.290706837727 |
| V12/M4/round1 | 低分散双seed锚＋置信门控修正 | 0.541599589053 | 0.584462065836 | 0.54206616417 | 0.58462293674 | 0.810905843803 | 0.28490369755 |
| V12/M4/round2 | 低分散双seed锚＋置信门控修正 | 0.542097416921 | 0.584076060999 | 0.542975794403 | 0.584391955571 | 0.810316137171 | 0.283325111693 |
| V12/M4/round3 | 低分散双seed锚＋置信门控修正 | 0.542455154996 | 0.583061978639 | 0.544227544313 | 0.583680324993 | 0.809204967615 | 0.285076387439 |
| V13/E0/round1 | 已存多尺度KL修正的有界幅度筛选 | **0.546401338572** | 0.586270312895 | **0.548614868711** | 0.587020076285 | 0.810628392655 | 0.281442108887 |
| V13/E0/round2 | 已存多尺度KL修正的有界幅度筛选 | 0.545959788001 | 0.586733140728 | 0.547032076021 | **0.587161496448** | 0.810951251501 | 0.280019527475 |
| V14/P0/round1 | 多尺度＋KL1＋序列等权BCE | 0.542090197445 | 0.584768029644 | 0.542162169739 | 0.58491421653 | 0.810696100627 | 0.285952335188 |
| V14/P0/round2 | 多尺度＋KL1＋序列等权BCE | 0.542061106507 | 0.584714378584 | 0.542082134953 | 0.584850681247 | 0.810708816861 | 0.284267449767 |
| V14/P0/round3 | 多尺度＋KL1＋序列等权BCE | 0.541161770343 | 0.584444600147 | 0.54124885567 | 0.584594754083 | 0.810735618425 | 0.285520253147 |
| V14/P1/round1 | 多尺度＋KL1＋完整UID平滑排序 | 0.541970352542 | 0.584566991365 | 0.542045866888 | 0.584708248876 | 0.810695852199 | 0.283434295356 |
| V14/P1/round2 | 多尺度＋KL1＋完整UID平滑排序 | 0.542103961808 | 0.584706821349 | 0.542165172812 | 0.584863201351 | 0.810909903504 | 0.284085518387 |
| V14/P1/round3 | 多尺度＋KL1＋完整UID平滑排序 | 0.542145441081 | 0.585108349256 | 0.542180900577 | 0.585239081635 | 0.811408031643 | 0.284844711962 |
| V14/P2/round1 | 多尺度＋KL1＋排序与类别支持权重 | 0.545627403352 | **0.586889562344** | 0.546356739322 | 0.587153867379 | 0.811314533194 | 0.279132520846 |
| V14/P2/round2 | 多尺度＋KL1＋排序与类别支持权重 | 0.542128553834 | 0.584718521868 | 0.542196880107 | 0.584886251207 | 0.810894641884 | 0.28436814599 |
| V14/P2/round3 | 多尺度＋KL1＋排序与类别支持权重 | 0.542103812567 | 0.585089620381 | 0.542144028672 | 0.585241682975 | 0.811429820522 | 0.284852831152 |
| V3/C1/round1 | 基线保护组合 | 0.540916081241 | 0.58385893427 | 0.540992163069 | 0.583925070458 | 0.81008100421 | 0.284522072619 |
| V3/C2/round1 | 固定融合；学习路由尚缺OOF前置 | 0.541120547749 | 0.583981350359 | 0.541135935479 | 0.584070383168 | 0.810465730314 | 0.286216466462 |
| V3/C3/round1 | V3-C3 | 0.541604582362 | 0.584200244997 | 0.542005183308 | 0.584775485842 | 0.811993468489 | 0.286338949667 |
| V3/P1/round1 | V3-P1 | 0.541489931394 | 0.58408085009 | 0.541518574714 | 0.584128991925 | 0.810210755994 | 0.283909582625 |
| V3/P1/round2 | V3-P1 | 0.541201341422 | 0.583970382856 | 0.54122430526 | 0.583975909203 | 0.810288915604 | 0.285996571644 |
| V3/P1/round3 | V3-P1 | 0.541344077675 | 0.584187522332 | 0.541383426438 | 0.584231668124 | 0.81018645074 | 0.285549150244 |
| V3/P2/round1 | V3-P2 | 0.541079198014 | 0.584012617336 | 0.541101461209 | 0.584091636928 | 0.810172853286 | 0.286460611626 |
| V3/P2/round2 | V3-P2 | 0.541043903384 | 0.583995519988 | 0.541075058745 | 0.584061041855 | 0.810159453314 | 0.283321875318 |
| V3/P2/round3 | V3-P2 | 0.541040612969 | 0.583842186144 | 0.541067888464 | 0.583884324928 | 0.810216227624 | 0.28592474699 |
| V3/P3/round1 | V3-P3 | 0.541493600415 | 0.584190357231 | 0.541810124224 | 0.584798663932 | 0.811844601943 | 0.284962786914 |
| V3/P3/round2 | V3-P3 | 0.541776253659 | 0.584271280213 | 0.542028553315 | 0.584827009496 | 0.81194748203 | 0.286352941052 |
| V3/P3/round3 | V3-P3 | 0.54135867759 | 0.584254541311 | 0.541768264028 | 0.584755470993 | 0.811666674658 | 0.284042629447 |
| V3/S1/round1 | V3-S1 | 0.541490487744 | 0.584107382921 | 0.541515800996 | 0.584159189612 | 0.810222200423 | 0.284049369931 |
| V3/S1/round2 | V3-S1 | 0.541564528897 | 0.584133892937 | 0.541589656861 | 0.584184355517 | 0.810232611712 | 0.284644975021 |
| V3/S1/round3 | V3-S1 | 0.541628005515 | 0.584173319091 | 0.541653873352 | 0.584204623627 | 0.810248770474 | 0.284353455201 |
| V3/S2/round1 | V3-S2 | 0.541124008094 | 0.584037306635 | 0.541141162476 | 0.584112434745 | 0.810149480065 | 0.286451703535 |
| V3/S2/round2 | V3-S2 | 0.541110759158 | 0.584028521328 | 0.541131700654 | 0.584094292114 | 0.810177066352 | 0.28632768934 |
| V3/S2/round3 | V3-S2 | 0.541117025025 | 0.584060599781 | 0.541134707711 | 0.584131938622 | 0.810212060412 | 0.285985409131 |
| V3/S3/round1 | V3-S3 | 0.541376185492 | 0.584173887278 | 0.541685246736 | 0.584759730665 | 0.811825254899 | 0.28503314821 |
| V3/S3/round2 | V3-S3 | 0.541395079701 | 0.584218399696 | 0.541677073172 | 0.584758308712 | 0.811857949251 | 0.28467570344 |
| V3/S3/round3 | V3-S3 | 0.541301825176 | 0.584211676368 | 0.541567947879 | 0.584705640129 | 0.811903592236 | 0.283678125311 |
| V4/Q0/round1 | 完整UID分组、原损失对照 | 0.53909648044 | 0.583192094009 | 0.53991285109 | 0.584052863545 | 0.811209764706 | 0.29252561648 |
| V4/Q1/round1 | 完整UID跨片段logistic排序 | 0.538759067873 | 0.582975087183 | 0.539621254903 | 0.583942496411 | 0.811056461507 | 0.291743490582 |
| V4/Q1/round2 | 完整UID跨片段logistic排序 | 0.538873215139 | 0.582946417219 | 0.539588286222 | 0.583911275409 | 0.811107206096 | 0.292329486761 |
| V4/Q1/round3 | 完整UID跨片段logistic排序 | 0.538951553599 | 0.582934221885 | 0.539603106218 | 0.583905072404 | 0.811158087492 | 0.287680937554 |
| V4/Q2/round1 | 完整UID有界平滑AUC风险 | 0.538916800432 | 0.583207559421 | 0.539782426235 | 0.584074050545 | 0.811107580704 | 0.291773384208 |
| V4/Q2/round2 | 完整UID有界平滑AUC风险 | 0.53938629095 | 0.583344886188 | 0.54015688039 | 0.584196777033 | 0.811134597024 | 0.289293129165 |
| V4/Q2/round3 | 完整UID有界平滑AUC风险 | 0.540199390954 | 0.584133064532 | 0.540772250494 | 0.584925007614 | **0.812211952137** | 0.294802555885 |
| V4/Q3/round1 | 完整UID平滑风险＋类别支持权重 | 0.539004322968 | 0.583147105887 | 0.539803822013 | 0.584104780788 | 0.81109169414 | 0.291691608317 |
| V4/Q3/round2 | 完整UID平滑风险＋类别支持权重 | 0.539450946578 | 0.583565419656 | 0.54030868996 | 0.584372667671 | 0.811284527629 | 0.287813142857 |
| V4/Q3/round3 | 完整UID平滑风险＋类别支持权重 | 0.539977303569 | 0.583998776033 | 0.54077959879 | 0.58470645492 | 0.812032082423 | 0.294254449507 |
| V4/Q4/round1 | 完整UID平滑风险＋类别支持＋温和在线关注 | 0.539140343163 | 0.583212282637 | 0.54007995968 | 0.584186375156 | 0.811083889277 | 0.290296165229 |
| V4/Q4/round2 | 完整UID平滑风险＋类别支持＋温和在线关注 | 0.53900014429 | 0.583105620114 | 0.539707927521 | 0.583954242576 | 0.811040441282 | 0.293385953713 |
| V4/Q4/round3 | 完整UID平滑风险＋类别支持＋温和在线关注 | 0.540150574304 | 0.584105976374 | 0.540898238689 | 0.584754754172 | 0.811564291105 | 0.292498195123 |
| V5/G0/round1 | 原sampler/generator＋索引包装、原损失对照 | 0.541472947685 | 0.584160541215 | 0.541478557102 | 0.584186600587 | 0.810396597473 | 0.284978726758 |
| V5/G1/round1 | 原批次＋完整序列logistic梯度缓存 | 0.541291833166 | 0.584033247514 | 0.541300966486 | 0.584084806799 | 0.810384944256 | 0.285548026681 |
| V5/G1/round2 | 原批次＋完整序列logistic梯度缓存 | 0.541557204075 | 0.584055372139 | 0.541598980134 | 0.584102905353 | 0.810443985437 | 0.288277821001 |
| V5/G2/round1 | 原批次＋完整序列平滑风险梯度缓存 | 0.54146210317 | 0.584138607803 | 0.541505239047 | 0.584189875109 | 0.81039985656 | 0.287901930437 |
| V5/G2/round2 | 原批次＋完整序列平滑风险梯度缓存 | 0.541589152914 | 0.584397117717 | 0.541616025505 | 0.584470797027 | 0.810798687354 | 0.285118028559 |
| V5/G2/round3 | 原批次＋完整序列平滑风险梯度缓存 | 0.54302713711 | 0.585092011764 | 0.543433778063 | 0.585455070752 | 0.811876861825 | 0.280966487718 |
| V5/G3/round1 | 原批次＋平滑风险梯度缓存＋类别支持权重 | 0.541323300842 | 0.583999864837 | 0.541385795576 | 0.584052586575 | 0.810318898713 | 0.285853500393 |
| V5/G3/round2 | 原批次＋平滑风险梯度缓存＋类别支持权重 | 0.541390380414 | 0.584121852411 | 0.541398379419 | 0.58421070593 | 0.810563715909 | 0.284780125196 |
| V5/G3/round3 | 原批次＋平滑风险梯度缓存＋类别支持权重 | 0.542984398433 | 0.585191673518 | 0.543274293564 | 0.58555183366 | 0.811818668514 | 0.279871413752 |
| V6融合/J1/round1 | 固定题目＋前缀历史融合 | 0.54144585755 | 0.583989335797 | 0.541771180632 | 0.584080518551 | 0.808279862786 | 0.279082311742 |
| V6融合/J1/round2 | 固定题目＋前缀历史融合 | 0.54148768717 | 0.584107570121 | 0.541845921181 | 0.584219548733 | 0.808992214751 | 0.280470964047 |
| V6融合/J1/round3 | 固定题目＋前缀历史融合 | 0.541231665188 | 0.583887054526 | 0.541634376614 | 0.583973144766 | 0.809495149705 | 0.284000324247 |
| V6融合/J2/round1 | 固定题目＋多尺度历史融合 | 0.541844927681 | 0.584437955703 | 0.542089637678 | 0.584518403294 | 0.808470128992 | **0.277404697043** |
| V6融合/J2/round2 | 固定题目＋多尺度历史融合 | 0.542228943731 | 0.584806054878 | 0.54245136858 | 0.584877099321 | 0.809295858452 | 0.279339853873 |
| V6融合/J2/round3 | 固定题目＋多尺度历史融合 | 0.542082094867 | 0.584694315041 | 0.542278360908 | 0.584813265511 | 0.809794570835 | 0.281203237588 |
| V6融合/J3/round1 | 固定多尺度＋KL保护分支融合 | 0.541959300257 | 0.584615707781 | 0.542124247026 | 0.584789233992 | 0.810339993648 | 0.284319735263 |
| V6融合/J3/round2 | 固定多尺度＋KL保护分支融合 | 0.542096405619 | 0.584780540018 | 0.542227895336 | 0.584961581231 | 0.810337317962 | 0.28246123399 |
| V6融合/J3/round3 | 固定多尺度＋KL保护分支融合 | 0.542289123003 | 0.584916074928 | 0.542397078404 | 0.585070712287 | 0.810365000781 | 0.282369381994 |

| 方案 | 改动 | Δpooled | Δmacro | Δstd | Δrange | std变化95%区间 | 下降>0.01条数 | 最坏个体变化 |
|---|---|---|---|---|---|---|---|---|
| V2/A1/round1 | 序列等权BCE | -0.000258443408 | +0.000192584095 | -5.74302942e-05 | -0.00181396578 | [-0.000391408519,+0.000243613885] | 2 | -0.0179425837 |
| V2/A1/round2 | 序列等权BCE | -6.47571707e-05 | +0.000222168695 | +2.34480555e-05 | -0.0013707476 | [-0.000164050742,+0.000237310755] | 0 | -0.00796123226 |
| V2/A2/round1 | 平滑困难序列加权 | -0.000243876643 | +0.000193957921 | -0.000317268235 | +0.00232762922 | [-0.000632836046,-8.99653755e-06] | 7 | -0.0167464115 |
| V2/A2/round2 | 平滑困难序列加权 | -0.00022229637 | +0.00026399246 | -0.000202772603 | -0.00135900672 | [-0.000540800724,+0.000101370264] | 3 | -0.0152777778 |
| V2/A3/round1 | 序列内排序 | -0.000270808157 | +0.000104587732 | -0.000508096191 | -0.000431477296 | [-0.000889546998,-0.000169766302] | 8 | -0.0138888889 |
| V2/A3/round2 | 序列内排序 | -0.000227578805 | +0.000191603595 | -0.000302629613 | -0.00596143121 | [-0.000651020408,+7.55616945e-06] | 3 | -0.0111111111 |
| V2/B1/round1 | 低频覆盖/固定融合，细节见参数 | +1.55596259e-05 | +0.000169387076 | -0.000167423462 | +0.00415333588 | [-0.000378556056,+6.96191899e-06] | 0 | -0.00938086304 |
| V2/B1/round2 | 低频覆盖/固定融合，细节见参数 | +1.25484622e-05 | +0.000190119792 | -8.29357088e-05 | -0.000921658986 | [-0.000236243529,+6.63834894e-05] | 0 | -0.0061633282 |
| V2/B2/round1 | 冻结主干前缀小适配 | -1.19337633e-06 | +1.00219485e-05 | -1.86348902e-05 | -0.000921658986 | [-3.71022708e-05,-1.54578701e-06] | 0 | -0.00303030303 |
| V2/B2/round2 | 冻结主干前缀小适配 | -1.17370597e-06 | +1.0489866e-05 | -1.82641761e-05 | -0.000921658986 | [-3.64087029e-05,-1.19017552e-06] | 0 | -0.00303030303 |
| V2/C1/round1 | 基线保护组合 | -4.50779081e-10 | +4.91506206e-08 | -8.37592476e-08 | +0 | [-2.285168e-07,+6.53136868e-09] | 0 | -1.4859945e-05 |
| V2/C1/round2 | 基线保护组合 | -4.91758856e-10 | +3.85783848e-08 | -8.27433563e-08 | +0 | [-2.25893098e-07,+8.44458841e-09] | 0 | -1.4859945e-05 |
| D07/D1/round1 | D1 | -9.46440333e-05 | +0.000251171689 | -0.000144661435 | +0.00507499486 | [-0.000351034108,+4.42657188e-05] | 0 | -0.00902255639 |
| D07/D1/round2 | D1 | -6.75256104e-05 | +9.68591827e-05 | -6.62787926e-05 | +0.00138248848 | [-0.000224239045,+9.77101572e-05] | 0 | -0.00692281066 |
| D07/D1/round3 | D1 | +1.29533438e-06 | +5.78715421e-05 | -4.68580589e-05 | -0.0023100179 | [-0.000189666653,+0.000104751734] | 0 | -0.00773694391 |
| D07/D2/round1 | D2 | -0.000113328583 | +0.000189256361 | -8.99198613e-05 | +1.76113182e-05 | [-0.000298025451,+8.85071178e-05] | 1 | -0.0102941176 |
| D07/D2/round2 | D2 | -8.48361855e-05 | +0.000149148797 | -3.32363039e-05 | +0.00230414747 | [-0.000216150861,+0.000135434325] | 0 | -0.00882352941 |
| D07/D2/round3 | D2 | -2.14520471e-05 | +0.000103904091 | +7.1601188e-05 | -0.000443218175 | [-0.000106723017,+0.000232408006] | 1 | -0.0101010101 |
| D07/D3/round1 | D3 | +6.38837246e-05 | +0.000277078426 | +0.00034939414 | +0.00596730165 | [-0.000164464091,+0.00086268215] | 35 | -0.027972028 |
| D07/D3/round2 | D3 | +0.000120686446 | +0.000269069046 | +0.00041527049 | +0.00963045584 | [-0.000102017919,+0.000919600754] | 39 | -0.0337995338 |
| D07/D3/round3 | D3 | +0.000160249079 | +0.000432291626 | +0.000412802032 | +0.0114679034 | [-0.000115197556,+0.00092529904] | 34 | -0.0291375291 |
| D07/D4/round1 | D4 | -5.50589441e-05 | +0.000141815455 | -9.34489018e-05 | -0.00275323608 | [-0.000300199616,+0.000124448067] | 1 | -0.0113636364 |
| D07/D4/round2 | D4 | -7.41982483e-06 | +9.50203301e-05 | -9.07259491e-05 | -0.000454959054 | [-0.000262329591,+8.62771449e-05] | 1 | -0.0113636364 |
| D07/D4/round3 | D4 | -4.74573324e-05 | +4.45358344e-05 | +3.06717845e-05 | +0.000921658986 | [-0.000150412021,+0.000192634441] | 1 | -0.010483871 |
| D07/D5/round1 | D5 | +0.000540668038 | +0.00133671409 | +0.000378572545 | -0.00359857935 | [-0.000242447152,+0.00103945673] | 36 | -0.0348162476 |
| D07/D5/round2 | D5 | +0.000593316461 | +0.00152473901 | +0.000323266461 | -0.00865009246 | [-0.000271509514,+0.000936452106] | 30 | -0.0309477756 |
| D07/D5/round3 | D5 | +0.000614196265 | +0.00150833266 | +0.000285524577 | -0.0109542399 | [-0.000314392668,+0.000935823948] | 31 | -0.0312360054 |
| D07/D6/round1 | D6 | -8.63983811e-06 | +0.000122668426 | +4.35594581e-05 | -0.000454959054 | [-0.00015661041,+0.000207582664] | 0 | -0.00663780664 |
| D07/D6/round2 | D6 | -2.00212331e-05 | +0.000113357059 | -3.11971494e-05 | +0.00275910652 | [-0.000212820327,+0.000105452693] | 0 | -0.00692041522 |
| D07/D6/round3 | D6 | +5.80509293e-06 | +6.80151024e-05 | -9.39486552e-06 | -5.8704394e-06 | [-0.000161551348,+0.000121378805] | 1 | -0.0106060606 |
| D07/E1/round1 | E1 | -8.37760758e-06 | +7.817845e-05 | -6.18270366e-05 | -0.000915788547 | [-0.000153869368,+2.96247326e-05] | 0 | -0.00487012987 |
| D07/E1/round2 | E1 | -6.199705e-05 | +0.000201200295 | -0.000106483945 | -0.000909918107 | [-0.000287394472,+7.39396453e-05] | 0 | -0.00649350649 |
| D07/E1/round3 | E1 | -0.000163609965 | +0.000264626585 | -8.40008055e-05 | -0.0027414952 | [-0.000363017209,+0.000179582417] | 1 | -0.0119617225 |
| D07/E2/round1 | E2 | -1.05691319e-06 | +0.000116937003 | -6.67099976e-05 | +0.000933399865 | [-0.000164128918,+2.3013076e-05] | 0 | -0.004784689 |
| D07/E2/round2 | E2 | -4.35163751e-05 | +0.000238224243 | -7.98870081e-05 | -0.0013707476 | [-0.000275664559,+9.51873732e-05] | 0 | -0.00735294118 |
| D07/E2/round3 | E2 | -0.000135005774 | +0.000238498563 | -9.63705377e-05 | -0.00320232469 | [-0.000359573989,+0.000148650848] | 3 | -0.0152777778 |
| D07/E3/round1 | E3 | -7.77307176e-07 | +0.000119691152 | -0.000115676317 | -0.000449088614 | [-0.000212740531,-2.14630633e-05] | 0 | -0.00441176471 |
| D07/E3/round2 | E3 | -4.44717401e-05 | +0.000254064465 | -0.000145194887 | -0.00229240659 | [-0.00032485041,+4.87392651e-05] | 0 | -0.00717703349 |
| D07/E3/round3 | E3 | -0.000137431663 | +0.000252688869 | -0.000226278465 | -0.00504564267 | [-0.000524864785,+3.21859809e-05] | 0 | -0.00972222222 |
| D08/F1/round1 | F1 | +1.16536656e-05 | +0.000186031027 | -8.94920018e-05 | -0.00138248848 | [-0.00024718408,+5.71135853e-05] | 0 | -0.00694444444 |
| D08/F2/round1 | F2 | +1.34403492e-05 | +0.000207645402 | -0.000107124202 | -0.00184331797 | [-0.000262370696,+3.76986987e-05] | 0 | -0.0061633282 |
| D08/F3/round1 | F3 | +1.38418295e-05 | +0.000200051942 | -9.10371679e-05 | -0.00138248848 | [-0.000243918821,+5.69983265e-05] | 0 | -0.00694444444 |
| V6/I0/round1 | 当前题64维投影＋非线性残差 | -0.000685088335 | -0.00168205006 | -0.000948309894 | +0.00831254219 | [-0.00145197377,-0.000432604852] | 34 | -0.0230880231 |
| V6/I0/round2 | 当前题64维投影＋非线性残差 | -0.00105678361 | -0.00313713066 | -0.00112518733 | +0.0115442191 | [-0.00179474904,-0.000451135377] | 77 | -0.0263605442 |
| V6/I0/round3 | 当前题64维投影＋非线性残差 | -0.000460800013 | -0.00147264088 | -0.000557074159 | +0.00418268807 | [-0.000956964541,-0.00017231667] | 33 | -0.0200668896 |
| V6/I1/round1 | 题目残差＋KC与全局前缀状态 | -0.00129001011 | -0.000428027356 | -0.000154522042 | -0.00967741935 | [-0.000744321537,+0.000370769913] | 34 | -0.0305555556 |
| V6/I1/round2 | 题目残差＋KC与全局前缀状态 | -0.00195553557 | -0.00121964804 | -0.00088721337 | -0.00276497696 | [-0.00153859413,-0.000239773958] | 71 | -0.0263678312 |
| V6/I1/round3 | 题目残差＋KC与全局前缀状态 | -0.00114707197 | -0.00101359592 | -0.000361792333 | -0.00228653615 | [-0.000828808608,+0.000109828111] | 39 | -0.0194931774 |
| V6/I2/round1 | 前缀＋8/32步多尺度历史＋最近KC响应 | -0.000690942726 | +0.000137666469 | -0.000389504835 | -0.00322580645 | [-0.000800250108,+1.42702456e-05] | 23 | -0.0361111111 |
| V6/I2/round2 | 前缀＋8/32步多尺度历史＋最近KC响应 | -0.0011572849 | -0.000437630636 | -0.000698281517 | -0.00783410138 | [-0.00146510366,-3.44478935e-06] | 62 | -0.033438588 |
| V6/I2/round3 | 前缀＋8/32步多尺度历史＋最近KC响应 | -0.00111579007 | +0.000313974749 | -0.000960824237 | +0.00322580645 | [-0.00168235743,-0.000290447423] | 56 | -0.028013926 |
| V6/I3/round1 | 多尺度历史＋序列等权BCE .25 | -0.000732817528 | +0.000159207744 | -0.000290346375 | -0.00138248848 | [-0.000696585473,+9.40732039e-05] | 22 | -0.0305555556 |
| V6/I3/round2 | 多尺度历史＋序列等权BCE .25 | -0.0011326247 | -0.000318879332 | -0.000504740978 | -0.00921658986 | [-0.00124796064,+0.000197624398] | 55 | -0.0268804145 |
| V6/I3/round3 | 多尺度历史＋序列等权BCE .25 | -0.00116882616 | +0.000346248071 | -0.000945037317 | +0.00368663594 | [-0.00165021787,-0.0002605846] | 59 | -0.0287426119 |
| V6/I4/round1 | 多尺度历史＋冻结基线KL1 | -0.000125591661 | +0.000426445769 | -0.000374167849 | +0.00230414747 | [-0.000724435108,-6.57346466e-05] | 4 | -0.0222222222 |
| V6/I4/round2 | 多尺度历史＋冻结基线KL1 | +0.000613352037 | +0.000986946095 | -0.000618824899 | -0.00276497696 | [-0.00121510925,-4.40157588e-05] | 19 | -0.0294117647 |
| V6/I5/round1 | 多尺度历史＋类别支持度门控 | -0.000418457098 | -5.75708543e-05 | -0.00025519227 | -0.011048167 | [-0.000646584838,+0.000198182236] | 13 | -0.0229885057 |
| V6/I5/round2 | 多尺度历史＋类别支持度门控 | -0.000836717655 | -0.000115145154 | -0.000303257847 | -0.015189762 | [-0.000960461249,+0.000401983403] | 41 | -0.0254355401 |
| V6/I5/round3 | 多尺度历史＋类别支持度门控 | -0.00144463412 | -0.000237549766 | -0.000239184621 | -0.014268103 | [-0.000761029083,+0.000313671432] | 32 | -0.0279146141 |
| V7/U0/round1 | 按冻结baseline不确定性限制修正 | -0.000546040465 | +0.000177576998 | -0.000394573998 | -0.000921658986 | [-0.000772327716,-7.65570559e-05] | 16 | -0.0208333333 |
| V7/U0/round2 | 按冻结baseline不确定性限制修正 | -0.00104684393 | -0.000721636124 | -0.000620603724 | -0.0138248848 | [-0.00145553387,+0.000102044285] | 60 | -0.0402777778 |
| V7/U0/round3 | 按冻结baseline不确定性限制修正 | -0.00130808741 | +0.000233662502 | -0.000276926262 | -0.000460829493 | [-0.000692901101,+0.000105540564] | 20 | -0.0217864924 |
| V7/U1/round1 | 按冻结预测头分歧限制修正 | -0.000133898333 | +4.35962464e-05 | -4.97458707e-05 | +0.00138248848 | [-0.000265071909,+0.000136490682] | 0 | -0.00937950938 |
| V7/U1/round2 | 按冻结预测头分歧限制修正 | -0.0006263689 | -0.000428947866 | +1.45953431e-05 | -0.00644574246 | [-0.000381904094,+0.000385735323] | 16 | -0.0246212121 |
| V7/U1/round3 | 按冻结预测头分歧限制修正 | -0.000540095549 | -0.000418920868 | -2.67773131e-05 | -0.00275323608 | [-0.000367876952,+0.000314542327] | 10 | -0.0209235209 |
| V7/U2/round1 | 优先低KC历史支持的位置 | -0.000342497081 | +2.914364e-05 | -0.000291653594 | +0.00967741935 | [-0.000619342274,+1.14220902e-05] | 8 | -0.0175 |
| V7/U2/round2 | 优先低KC历史支持的位置 | -0.000663480974 | -0.00038831571 | -0.000666564374 | +0.0110540374 | [-0.00119429971,-8.95514632e-05] | 50 | -0.030350808 |
| V7/U2/round3 | 优先低KC历史支持的位置 | -0.000990131898 | -0.000270599209 | -0.000239221286 | +0.00229827703 | [-0.000665170206,+0.0001748659] | 24 | -0.0274193548 |
| V8/K1/round1 | seed42＋43固定概率平均 | +0.000442498215 | +0.000609618745 | +3.37940625e-05 | -0.000449088614 | [-0.000211555512,+0.000273249061] | 2 | -0.0130952381 |
| V8/K1/round2 | seed42＋43固定概率平均 | +0.000553139909 | +0.000678910056 | +0.000220577085 | +0.00687721976 | [-0.000224922358,+0.000673965013] | 18 | -0.0204081633 |
| V8/K1/round3 | seed42＋43固定概率平均 | +0.000333131407 | +0.000339705555 | +0.000277700802 | +0.00268866125 | [-0.000362280202,+0.000880223833] | 54 | -0.0295454545 |
| V8/K2/round1 | seed42＋44固定概率平均 | +0.000615285675 | +0.000558384787 | +0.000216547529 | +0.00321406557 | [-7.60382219e-05,+0.000538088441] | 4 | -0.013136289 |
| V8/K2/round2 | seed42＋44固定概率平均 | +0.000862396085 | +0.000767923712 | +0.000655459871 | -0.000974492941 | [+5.48459886e-05,+0.00126348582] | 29 | -0.0265151515 |
| V8/K2/round3 | seed42＋44固定概率平均 | +0.000750473277 | +0.000423990931 | +0.000930215046 | +0.00218673868 | [+4.88834315e-05,+0.0017685567] | 60 | -0.0297430646 |
| V8/K3/round1 | seed43＋44固定概率平均，对原42比较 | +0.000465193224 | +0.000422872155 | +0.000586551548 | +0.00173765006 | [-0.00027195377,+0.00136529437] | 73 | -0.05 |
| V8/K3/round2 | seed43＋44固定概率平均，对原42比较 | +0.000770371283 | +0.000606195004 | +0.000838241836 | +0.0044850157 | [-3.95235925e-05,+0.00168529229] | 64 | -0.0454545455 |
| V8/K3/round3 | seed43＋44固定概率平均，对原42比较 | +0.000708261743 | +0.000324186073 | +0.00100400515 | +0.00813055857 | [+1.84959571e-05,+0.00198755249] | 83 | -0.0431818182 |
| V8/K4/round1 | seed42/43/44各1/3概率平均 | +0.000966478819 | +0.00102599431 | +0.000514631385 | +0.00544189733 | [-4.69507849e-05,+0.0010951385] | 26 | -0.0295454545 |
| V8/K5/round1 | seed42/43/44各1/3 logit平均 | +0.000978443645 | +0.00103410123 | +0.000512512523 | +0.00680090405 | [-5.67926868e-05,+0.00109629814] | 25 | -0.0295454545 |
| V9/L1/round1 | 三seed性能锚＋题目修正R1 | +0.000851852401 | +0.000712617302 | +0.000180118302 | +0.0086853151 | [-0.000278553948,+0.000626500733] | 18 | -0.0204545455 |
| V9/L1/round2 | 三seed性能锚＋题目修正R1 | +0.000542440379 | +5.88548384e-05 | -0.000172471299 | +0.0101088967 | [-0.000550613403,+0.000224797222] | 13 | -0.0148148148 |
| V9/L1/round3 | 三seed性能锚＋题目修正R1 | +3.47845371e-05 | -0.000619419083 | -0.000559285158 | +0.00690657196 | [-0.00094621244,-0.000136046265] | 16 | -0.015015015 |
| V9/L2/round1 | 三seed性能锚＋小lr题目修正R3 | +0.000880709766 | +0.000731162888 | +0.000293955449 | +0.00684199712 | [-0.000158989577,+0.000733721076] | 18 | -0.0188046919 |
| V9/L2/round2 | 三seed性能锚＋小lr题目修正R3 | +0.000615090774 | +0.000165912075 | +1.95639901e-05 | +0.00736153101 | [-0.000324390653,+0.000418718342] | 13 | -0.0195286195 |
| V9/L2/round3 | 三seed性能锚＋小lr题目修正R3 | +0.000172554489 | -0.000551389989 | -0.000228208255 | +0.00922833074 | [-0.000570963806,+0.000127328888] | 19 | -0.0181818182 |
| V9/L3/round1 | 三seed性能锚＋困难加权A3R2 | +0.000877625781 | +0.00109308235 | +0.000260522682 | +0.00545950864 | [-0.000190222168,+0.000723047544] | 14 | -0.0166170173 |
| V9/L3/round2 | 三seed性能锚＋困难加权A3R2 | +0.00064462271 | +0.00090084867 | +5.18963169e-05 | -0.000466699932 | [-0.000312367138,+0.000390142693] | 5 | -0.0117845118 |
| V9/L3/round3 | 三seed性能锚＋困难加权A3R2 | +0.000275107893 | +0.000658022231 | -0.000130059007 | -0.00459068361 | [-0.000453412113,+0.000165448298] | 2 | -0.012244898 |
| V10/R0/round1 | 全局前缀预测误差对照 | -0.00223382785 | -0.00292039516 | +0.00132193049 | +0.0243241657 | [+0.000778009579,+0.00188262881] | 76 | -0.0246305419 |
| V10/R0/round2 | 全局前缀预测误差对照 | -0.0038456682 | -0.00472787407 | +0.00217125692 | +0.0334526989 | [+0.0013646273,+0.00303141447] | 156 | -0.0489130435 |
| V10/R0/round3 | 全局前缀预测误差对照 | -0.00175151428 | -0.00181211827 | +0.00062629986 | +0.00552408348 | [+0.000260474547,+0.000959309714] | 17 | -0.0161290323 |
| V10/R1/round1 | 当前KC前缀预测误差 | -0.000865813114 | -0.00121298181 | +0.000234058475 | +0.0207314567 | [-0.000709064908,+0.00117556944] | 112 | -0.0315789474 |
| V10/R1/round2 | 当前KC前缀预测误差 | -0.00526561605 | -0.00585140104 | +0.000982035255 | +0.0464087587 | [-0.000701791883,+0.00262186622] | 305 | -0.0667701863 |
| V10/R1/round3 | 当前KC前缀预测误差 | +0.000182943022 | +1.39331426e-07 | -3.10179374e-05 | +0.00552995392 | [-0.000495372405,+0.000431962252] | 10 | -0.0147058824 |
| V10/R2/round1 | 当前概率桶前缀预测误差 | -0.00232794328 | -0.00914953078 | +0.00262307548 | +0.0160761983 | [+0.00178355027,+0.0035242083] | 335 | -0.0571095571 |
| V10/R2/round2 | 当前概率桶前缀预测误差 | -0.0078001867 | -0.0189106892 | +0.00470769173 | +0.0427221228 | [+0.00315174311,+0.00618002074] | 571 | -0.0964285714 |
| V10/R2/round3 | 当前概率桶前缀预测误差 | -0.000876689595 | -0.00501510675 | +0.00158124679 | +0.00783410138 | [+0.0010154904,+0.0021336823] | 102 | -0.0412104623 |
| V10/R3/round1 | KC误差与概率桶误差各一半 | -0.000471137403 | -0.00444716475 | +0.00140057011 | +0.0271595879 | [+0.00058008801,+0.00222687551] | 177 | -0.0321910696 |
| V10/R3/round2 | KC误差与概率桶误差各一半 | -0.00303147942 | -0.0086265148 | +0.00236553214 | +0.0487129062 | [+0.00108542131,+0.00378185271] | 361 | -0.0559006211 |
| V10/R3/round3 | KC误差与概率桶误差各一半 | +0.000171595968 | -0.00210731273 | +0.000661699674 | +0.00875576037 | [+0.000245791916,+0.00107095227] | 23 | -0.0231143552 |
| V10/R4/round1 | KC×概率桶局部误差 | -0.000297147879 | -0.00207534532 | +0.00106595646 | -0.00323167689 | [+0.000226345312,+0.00177727869] | 88 | -0.0363636364 |
| V10/R4/round2 | KC×概率桶局部误差 | -0.00404396812 | -0.00727979049 | +0.00248962887 | +0.0183099005 | [+0.000968443357,+0.0040007301] | 303 | -0.0932330827 |
| V10/R4/round3 | KC×概率桶局部误差 | +0.00020223387 | -0.000472583556 | +0.000292999892 | -0.00184918841 | [-4.59051561e-05,+0.000598969242] | 9 | -0.0169608861 |
| V11/T0/round1 | 冻结主干＋题目相似度因果检索 | -0.000906044251 | +0.000294973992 | -0.000497024222 | -0.0115207373 | [-0.000957479528,-8.79783639e-05] | 14 | -0.0188834154 |
| V11/T0/round2 | 冻结主干＋题目相似度因果检索 | -0.00143709897 | -0.00144811895 | -0.000129296455 | +0.0124482668 | [-0.000859596871,+0.000701382121] | 99 | -0.0364342968 |
| V11/T0/round3 | 冻结主干＋题目相似度因果检索 | -0.00146351377 | +0.000534238261 | -0.000662533122 | -0.00274736564 | [-0.00109912082,-0.000226805924] | 14 | -0.0193548387 |
| V11/T1/round1 | 相似度检索＋同KC偏置2 | -0.00107014458 | -0.000378483233 | -1.07878924e-05 | -0.00737327189 | [-0.000606151254,+0.000548946773] | 30 | -0.0348484848 |
| V11/T1/round2 | 相似度检索＋同KC偏置2 | -0.00181824647 | -0.00168025612 | -3.73583279e-06 | +0.0106049488 | [-0.000826068669,+0.000833313209] | 92 | -0.0397727273 |
| V11/T1/round3 | 相似度检索＋同KC偏置2 | -0.00177059813 | -0.000195765071 | -0.000285326919 | -0.00804515924 | [-0.00091863706,+0.000337621649] | 32 | -0.0397727273 |
| V11/T2/round1 | 同KC检索＋全窗口/最近32两尺度 | -0.000909179338 | -0.000108662958 | -0.000114576049 | -0.00783410138 | [-0.000746724311,+0.000408008727] | 27 | -0.0213815789 |
| V11/T2/round2 | 同KC检索＋全窗口/最近32两尺度 | -0.0014927854 | -0.00131651903 | -0.000219918706 | +0.0198039273 | [-0.000974711016,+0.00060642487] | 85 | -0.0486111111 |
| V11/T2/round3 | 同KC检索＋全窗口/最近32两尺度 | -0.00187071938 | -3.78678573e-05 | -0.000217438388 | -0.014268103 | [-0.000856277487,+0.000395624545] | 39 | -0.0328407225 |
| V11/T3/round1 | 两尺度检索＋冻结基线KL1 | -0.000434440907 | +0.000337241658 | +0.000102141639 | -0.000921658986 | [-0.000232335306,+0.000474197152] | 6 | -0.0229885057 |
| V11/T3/round2 | 两尺度检索＋冻结基线KL1 | -0.000506448573 | -0.000219372537 | -0.000195888076 | +0.00694766503 | [-0.000765715024,+0.000356284608] | 29 | -0.0279146141 |
| V11/T3/round3 | 两尺度检索＋冻结基线KL1 | -0.000708713465 | +0.000396937219 | -0.000238589749 | -0.00689483108 | [-0.000578324584,+7.54325464e-05] | 6 | -0.0170807453 |
| V12/M0/round1 | 原基线＋增强检索修正 | -0.00209506778 | -0.000159806276 | -0.000470483329 | -0.0133640553 | [-0.00111671197,+0.000116567322] | 52 | -0.0370967742 |
| V12/M0/round2 | 原基线＋增强检索修正 | -0.00391589566 | -0.000947605594 | -0.000333896093 | -0.0170506912 | [-0.0011258904,+0.00045307472] | 100 | -0.0419354839 |
| V12/M0/round3 | 原基线＋增强检索修正 | -0.00982424754 | -0.00411964399 | +0.000101353177 | -0.0221139452 | [-0.0010908401,+0.00128193452] | 211 | -0.064516129 |
| V12/M1/round1 | 三seed性能锚＋检索logit修正 | +0.000249624897 | +0.00116675439 | +0.000308509746 | +0.00127095013 | [-0.000397235749,+0.00100744056] | 50 | -0.0334986226 |
| V12/M1/round2 | 三seed性能锚＋检索logit修正 | -0.0025471639 | +6.26588494e-05 | +0.000349889468 | -0.00656315125 | [-0.000608596243,+0.00129948895] | 109 | -0.0537741047 |
| V12/M1/round3 | 三seed性能锚＋检索logit修正 | -0.00821856659 | -0.00314029032 | +0.000854477902 | -0.00561181764 | [-0.000489359477,+0.00220405232] | 214 | -0.0747107438 |
| V12/M2/round1 | 三seed锚＋置信门控检索修正 | +0.000438953124 | +0.00116032848 | +0.000411934652 | +0.000349291144 | [-0.000234322343,+0.00105251288] | 36 | -0.0310743802 |
| V12/M2/round2 | 三seed锚＋置信门控检索修正 | -0.00120177148 | +0.000665385115 | +0.000311745109 | +0.000810120638 | [-0.000531724617,+0.00113812649] | 75 | -0.0450688705 |
| V12/M2/round3 | 三seed锚＋置信门控检索修正 | -0.00419566555 | -0.000931741005 | +0.000513909351 | -0.00564149227 | [-0.000518488057,+0.00154850346] | 146 | -0.0612672176 |
| V12/M3/round1 | 三seed锚＋严格前缀预测差中心化 | +0.000707783549 | +0.000351362567 | +0.000455796214 | +0.00358683847 | [-0.00024949085,+0.00115717718] | 57 | -0.0347107438 |
| V12/M3/round2 | 三seed锚＋严格前缀预测差中心化 | -0.000843264157 | -0.00187784609 | +0.000571134319 | -0.000554756524 | [-0.000371811283,+0.00153915706] | 131 | -0.0548387097 |
| V12/M3/round3 | 三seed锚＋严格前缀预测差中心化 | -0.00395303482 | -0.00570462716 | +0.000953153929 | +0.00347702346 | [-0.000331710992,+0.00227213909] | 257 | -0.085483871 |
| V12/M4/round1 | 低分散双seed锚＋置信门控修正 | -0.000176670553 | +0.000706052798 | -2.79381038e-05 | +0.000933399865 | [-0.000442465932,+0.000349501731] | 12 | -0.0208264463 |
| V12/M4/round2 | 低分散双seed锚＋置信门控修正 | -0.00195021289 | +0.000115195506 | -0.000126746602 | -0.00643987202 | [-0.000809304207,+0.000492856437] | 57 | -0.0387096774 |
| V12/M4/round3 | 低分散双seed锚＋置信门控修正 | -0.00511872926 | -0.00135389808 | -0.000170077506 | -0.0142739734 | [-0.00107582881,+0.000700770862] | 126 | -0.05 |
| V13/E0/round1 | 已存多尺度KL修正的有界幅度筛选 | -0.000763963458 | +9.22838685e-05 | -0.000621213133 | -0.0188998797 | [-0.00193473143,+0.00067417989] | 118 | -0.0485294118 |
| V13/E0/round2 | 已存多尺度KL修正的有界幅度筛选 | +0.000288234746 | +0.000848896171 | -0.000751170403 | -0.0115207373 | [-0.00168538883,+0.000183171459] | 63 | -0.0320621812 |
| V14/P0/round1 | 多尺度＋KL1＋序列等权BCE | -0.000172832538 | +0.000376747311 | -0.000140653321 | +5.8704394e-06 | [-0.000511676324,+0.000174786642] | 7 | -0.0208333333 |
| V14/P0/round2 | 多尺度＋KL1＋序列等权BCE | -0.000187879628 | +0.000375478098 | -0.00012871911 | +0.00184918841 | [-0.000490272862,+0.000196699416] | 9 | -0.0208333333 |
| V14/P0/round3 | 多尺度＋KL1＋序列等权BCE | -0.000266195948 | +0.000394776304 | -1.92328847e-05 | +0.00830080131 | [-0.000351973305,+0.000280929214] | 5 | -0.0170940171 |
| V14/P1/round1 | 多尺度＋KL1＋完整UID平滑排序 | -0.000107164301 | +0.000320502574 | -7.79025683e-05 | +1.17408788e-05 | [-0.000442722607,+0.000240251928] | 7 | -0.0191176471 |
| V14/P1/round2 | 多尺度＋KL1＋完整UID平滑排序 | -6.75901128e-05 | +0.000571928194 | -7.08588985e-05 | +0.00231588834 | [-0.000413132606,+0.000224722137] | 5 | -0.0170807453 |
| V14/P1/round3 | 多尺度＋KL1＋完整UID平滑排序 | -2.8226561e-06 | +0.00113314512 | -4.72667022e-05 | +0.0027708474 | [-0.000363553538,+0.00024468271] | 3 | -0.0186335404 |
| V14/P2/round1 | 多尺度＋KL1＋排序与类别支持权重 | +0.000653719884 | +0.00124916094 | -0.000709640821 | +0.00506912442 | [-0.00138729894,-0.000126290689] | 20 | -0.025 |
| V14/P2/round2 | 多尺度＋KL1＋排序与类别支持权重 | -6.59866094e-05 | +0.000570954848 | -8.12496658e-05 | +0.00185505885 | [-0.000423925157,+0.000225711673] | 5 | -0.0170807453 |
| V14/P2/round3 | 多尺度＋KL1＋排序与类别支持权重 | +3.31896395e-07 | +0.00113552047 | -3.23103621e-05 | +0.0027708474 | [-0.000355410998,+0.000265303977] | 3 | -0.0194099379 |
| V3/C1/round1 | 基线保护组合 | +1.26747213e-05 | +1.00317381e-05 | -8.68341237e-05 | -0.000460829493 | [-0.00025304337,+6.72150194e-05] | 0 | -0.00816326531 |
| V3/C2/round1 | 固定融合；学习路由尚缺OOF前置 | -3.56634335e-05 | +0.000121947343 | +2.34530128e-05 | +0.00184331797 | [-0.000135849676,+0.00015289565] | 0 | -0.00725806452 |
| V3/C3/round1 | V3-C3 | +0.000616134205 | +0.00140744372 | +0.000437721604 | -0.00636355631 | [-0.000172027908,+0.00108201113] | 34 | -0.0328820116 |
| V3/P1/round1 | V3-P1 | -1.23717567e-05 | +9.61976065e-05 | -0.000170168611 | +0.000915788547 | [-0.000345062237,-1.517902e-05] | 0 | -0.00668896321 |
| V3/P1/round2 | V3-P1 | -3.37893807e-05 | +3.39104488e-05 | -2.58820337e-05 | +0.00276497696 | [-0.000218982003,+0.000130165913] | 0 | -0.00952380952 |
| V3/P1/round3 | V3-P1 | +3.38258118e-05 | +0.000101319347 | -0.000151351036 | +0.000454959054 | [-0.000365226309,+3.31286089e-05] | 0 | -0.00971817298 |
| V3/P2/round1 | V3-P2 | -3.86817688e-06 | +0.000124386098 | -0.00010346843 | +0.00276497696 | [-0.000246289115,+3.72248236e-05] | 0 | -0.00606060606 |
| V3/P2/round2 | V3-P2 | -1.8582346e-05 | +6.75636399e-05 | -6.58872866e-05 | +0.000454959054 | [-0.0002375924,+7.90340207e-05] | 0 | -0.00692041522 |
| V3/P2/round3 | V3-P2 | +6.47949952e-06 | +2.98484518e-05 | -3.0970724e-05 | +0.00183744753 | [-0.000189675979,+9.48298065e-05] | 0 | -0.00580270793 |
| V3/P3/round1 | V3-P3 | +0.000603395883 | +0.00141869936 | +0.000399483082 | -0.0118876398 | [-0.000252496099,+0.0010685374] | 33 | -0.0336700337 |
| V3/P3/round2 | V3-P3 | +0.000602904862 | +0.00144098432 | +0.00039455552 | -0.00545950864 | [-0.000188299652,+0.00102920379] | 34 | -0.0328820116 |
| V3/P3/round3 | V3-P3 | +0.0005848078 | +0.00135256632 | +0.000345737839 | -0.00681851537 | [-0.000238994545,+0.000988452743] | 36 | -0.0309477756 |
| V3/S1/round1 | V3-S1 | -1.12602173e-05 | +9.94586313e-05 | -0.000172385484 | +0.000454959054 | [-0.000341336598,-2.5723708e-05] | 0 | -0.00668896321 |
| V3/S1/round2 | V3-S1 | -1.01172873e-05 | +0.00011555832 | -0.000180004575 | -0.000927529426 | [-0.000337950095,-3.06537599e-05] | 0 | -0.00668896321 |
| V3/S1/round3 | V3-S1 | -8.99251132e-06 | +0.000139468386 | -0.000183285697 | -0.00138248848 | [-0.000347469682,-2.76026704e-05] | 0 | -0.00588235294 |
| V3/S2/round1 | V3-S2 | -2.56575297e-06 | +0.000124559505 | -0.000126978333 | +0.00230414747 | [-0.000269460925,+9.49281062e-06] | 0 | -0.00606060606 |
| V3/S2/round2 | V3-S2 | -1.68177505e-06 | +0.000120233979 | -0.000120512644 | +0.00184331797 | [-0.000252571449,+8.22586656e-06] | 0 | -0.00606060606 |
| V3/S2/round3 | V3-S2 | -8.35580626e-07 | +0.000150823251 | -0.000108269049 | +0.00183744753 | [-0.000229395504,+1.83869229e-05] | 0 | -0.00606060606 |
| V3/S3/round1 | V3-S3 | +0.000614676467 | +0.00141627826 | +0.000415950818 | -0.0114268103 | [-0.000225475948,+0.00106216808] | 32 | -0.031986532 |
| V3/S3/round2 | V3-S3 | +0.00062390531 | +0.00144431079 | +0.000420040036 | -0.0086853151 | [-0.000212495839,+0.00105216071] | 30 | -0.0303030303 |
| V3/S3/round3 | V3-S3 | +0.000639120868 | +0.00143371062 | +0.000448763426 | -0.00869118554 | [-0.000155677517,+0.0010517894] | 29 | -0.0303030303 |
| V4/Q0/round1 | 完整UID分组、原损失对照 | -4.66501508e-05 | +6.28941148e-05 | +0.000843428046 | +0.012612481 | [-2.9293504e-05,+0.00176140249] | 106 | -0.0390977444 |
| V4/Q1/round1 | 完整UID跨片段logistic排序 | -2.19913838e-05 | +1.00771449e-05 | +0.00082675951 | +0.0106138161 | [-0.000145451943,+0.00177892269] | 108 | -0.0377272727 |
| V4/Q1/round2 | 完整UID跨片段logistic排序 | -6.11402417e-05 | +2.13735679e-08 | +0.000899469432 | +0.0113438971 | [-1.58937377e-05,+0.00181761794] | 102 | -0.0365853659 |
| V4/Q1/round3 | 完整UID跨片段logistic排序 | -6.74521334e-05 | -1.24218084e-05 | +0.000918439836 | +0.0100753131 | [+4.01110807e-05,+0.00187303232] | 108 | -0.0393996248 |
| V4/Q2/round1 | 完整UID有界平滑AUC风险 | -3.95232505e-05 | +8.63501142e-05 | +0.000854279321 | +0.0120325356 | [-0.000119858408,+0.00181392153] | 106 | -0.0393996248 |
| V4/Q2/round2 | 完整UID有界平滑AUC风险 | -7.00865687e-05 | +0.00027123607 | +0.000741387341 | +0.00761581872 | [-0.000188901012,+0.00167567871] | 104 | -0.0381818182 |
| V4/Q2/round3 | 完整UID有界平滑AUC风险 | -0.000109216412 | +0.00103726374 | +0.000896543393 | +0.00172851721 | [+2.56169686e-05,+0.00179139111] | 95 | -0.0436090226 |
| V4/Q3/round1 | 完整UID平滑风险＋类别支持权重 | -4.84475712e-05 | +1.96867432e-05 | +0.000827924591 | +0.00907598067 | [-1.18919384e-05,+0.00176793382] | 109 | -0.0412757974 |
| V4/Q3/round2 | 完整UID平滑风险＋类别支持权重 | -7.78988992e-05 | +0.000258142317 | +0.000775709679 | +0.00811287929 | [-0.000118615091,+0.00168416885] | 102 | -0.0390977444 |
| V4/Q3/round3 | 完整UID平滑风险＋类别支持权重 | -0.000119290793 | +0.000896680468 | +0.000817161451 | +0.00391876014 | [-4.19026008e-05,+0.00171996095] | 93 | -0.0406015038 |
| V4/Q4/round1 | 完整UID平滑风险＋类别支持＋温和在线关注 | -2.57840347e-05 | +6.77236164e-05 | +0.000785170116 | +0.0109192986 | [-9.48935655e-05,+0.00170760529] | 105 | -0.0390977444 |
| V4/Q4/round2 | 完整UID平滑风险＋类别支持＋温和在线关注 | -8.9352091e-05 | +9.46162455e-05 | +0.000834198168 | +0.0111471077 | [-4.61218253e-05,+0.00177363953] | 106 | -0.0393996248 |
| V4/Q4/round3 | 完整UID平滑风险＋类别支持＋温和在线关注 | -0.000123597127 | +0.000708140555 | +0.00064916584 | +0.00464884111 | [-0.000232787653,+0.00155555614] | 95 | -0.0436090226 |
| V5/G0/round1 | 原sampler/generator＋索引包装、原损失对照 | +1.35682475e-05 | +0.000178455502 | -0.000114957636 | -0.000915788547 | [-0.000307773055,+4.7965904e-05] | 0 | -0.00692041522 |
| V5/G1/round1 | 原批次＋完整序列logistic梯度缓存 | +1.29003158e-05 | +0.000148583534 | -4.66806658e-06 | +0.00321993601 | [-0.000131871569,+0.000129962545] | 0 | -0.00735294118 |
| V5/G1/round2 | 原批次＋完整序列logistic梯度缓存 | +3.14575411e-05 | +0.000171063178 | -0.000119600033 | -0.000915788547 | [-0.000383793926,+0.000101064652] | 0 | -0.00986842105 |
| V5/G2/round1 | 原批次＋完整序列平滑风险梯度缓存 | -9.71113525e-06 | +0.000244383572 | -7.57173936e-05 | -0.000472570372 | [-0.000253503549,+8.49335293e-05] | 1 | -0.0103806228 |
| V5/G2/round2 | 原批次＋完整序列平滑风险梯度缓存 | -3.04686956e-05 | +0.000527304155 | -4.02477299e-05 | -0.00367489507 | [-0.000346304755,+0.000224290426] | 0 | -0.00821018062 |
| V5/G2/round3 | 原批次＋完整序列平滑风险梯度缓存 | +3.3811223e-05 | +0.00156667003 | +0.000202175587 | +0.0174117233 | [-0.000387830746,+0.000753032984] | 27 | -0.033180778 |
| V5/G3/round1 | 原批次＋平滑风险梯度缓存＋类别支持权重 | -3.26189531e-05 | +0.000175757819 | -8.94107122e-05 | -0.0023100179 | [-0.000301413396,+0.000109048201] | 0 | -0.00932400932 |
| V5/G3/round2 | 原批次＋平滑风险梯度缓存＋类别支持权重 | -5.03934209e-05 | +0.000284125724 | -4.39912137e-05 | +0.00184331797 | [-0.000256510155,+0.000148798842] | 0 | -0.00865051903 |
| V5/G3/round3 | 原批次＋平滑风险梯度缓存＋类别支持权重 | -7.74745931e-06 | +0.00141070103 | +0.000187220843 | +0.0146350054 | [-0.000365117932,+0.000745632539] | 29 | -0.0354691076 |
| V6融合/J1/round1 | 固定题目＋前缀历史融合 | -0.000626726368 | -0.00122776425 | -0.000663514551 | +0.00508673574 | [-0.00115805535,-0.000182289632] | 30 | -0.018037518 |
| V6融合/J1/round2 | 固定题目＋前缀历史融合 | -0.000702386329 | -0.000847252987 | -0.000536282426 | -0.00275910652 | [-0.000996976985,-7.27961013e-05] | 21 | -0.0194444444 |
| V6融合/J1/round3 | 固定题目＋前缀历史融合 | -0.000921547612 | -0.000586742495 | -0.000323783727 | -0.00321993601 | [-0.000880467799,+0.000170385308] | 28 | -0.0236111111 |
| V6融合/J2/round1 | 固定题目＋多尺度历史融合 | -0.000448279212 | -0.000946330331 | -0.000736871356 | +0.00462590625 | [-0.00119219282,-0.000299712954] | 22 | -0.0155279503 |
| V6融合/J2/round2 | 固定题目＋多尺度历史融合 | -0.00036536982 | -0.00044108903 | -0.000657051797 | -0.000915788547 | [-0.00105943153,-0.000297986923] | 12 | -0.025 |
| V6融合/J2/round3 | 固定题目＋多尺度历史融合 | -0.000445707968 | -0.000107256226 | -0.000507384129 | -0.00367489507 | [-0.000949340358,-0.000131679054] | 16 | -0.0319444444 |
| V6融合/J3/round1 | 固定多尺度＋KL保护分支融合 | -0.000503303692 | +0.000229946725 | -0.000333307456 | -0.000921658986 | [-0.000749180239,+2.82725376e-05] | 18 | -0.0319444444 |
| V6融合/J3/round2 | 固定多尺度＋KL保护分支融合 | -0.000347840044 | +0.000298000917 | -0.000377939725 | +0.000921658986 | [-0.000777428118,-3.74191693e-05] | 13 | -0.0277777778 |
| V6融合/J3/round3 | 固定多尺度＋KL保护分支融合 | -0.000222194617 | +0.000364685299 | -0.000425463568 | +0.00138248848 | [-0.000798672986,-8.64510219e-05] | 10 | -0.0263888889 |

机制复盘与决策：

- 训练关注：等权、困难加权、完整序列排序和梯度缓存有局部收益，但更直接优化排序仍可能扩大增益差异，或损害pooled。V14-P2R1的pooled与std改善同时出现，但极差扩大；不能只挑两项宣布成功。
- 信息表示：题目、局部/KC统计、多尺度历史有小幅补弱线索；检索attention增加容量和历史选择并未产生联合候选。复杂度不等于补弱能力。
- 保护约束：I4R2支持温和多尺度修正＋KL作为后续基础；KL不能自动保AUC，检索＋KL失败也保留。组合结果不能将收益全部归给某个模块。
- 固定门控：置信度、预测头分歧、低支持门控九轮均未通过；只是少改并不足以知道该往哪个方向改。V15将基线状态作为修正输入，可学习方向，属于新机制而非继续扩大旧门控系数。
- 融合与放大：已有预测平均、残差迁移、幅度筛选往往在尾部、总体与极差之间取舍；普通点候选先记录，缺少正式集成链路/原生复验时不能称正式结果。
- 工程与资源：冻结KL训练1/2/3并发约10.255/10.174/10.149 step/s，推理11.055/10.350/10.267。双三并发忙碌率虽高，总吞吐更低；因此当前单槽有实测依据，不是默认限制显存。短热缓存固定工作量仍不代表所有新架构全流程全局最优。
- V15原CLI缺失使GPU空置，是入口检查不完整；已补CLI回归检查、真实--help并重新排GPU87/88/89。三方向最多9轮，首个普通候选暂停。独立冻结源码、从原B42新初始化，不冒称断点恢复，不改旧失败记录。

证据与历史边界：

- [全部逐轮数值、输入哈希与方差恒等式审计](/root/pykt-toolkit/logs/cross_version_review_20261111/audit.json)；所有193条的主子集、macro/std/range、个体增益与退化数逐一复算通过。
- [V2含说明的历史完整表](/root/pykt-toolkit/logs/qikt_v2_20261105/reports/方案全量指标汇总_20261107.md)。
- [V1封存档案](/data/pykt-toolkit-records/qikt_switch/V1_20261105/README.md)：早期元数据缺失保持标记，不补造。
- [V15方案](/root/pykt-toolkit/docs/QIKT_Mamba_V15_基线状态条件化修正.md)；[入口修复](/root/pykt-toolkit/docs/QIKT_Mamba_V15_训练入口修复01.md)；[当前实况报告](/root/pykt-toolkit/logs/qikt_v15_20261111_fix1/V15_入口修复后实时报告.md)。
