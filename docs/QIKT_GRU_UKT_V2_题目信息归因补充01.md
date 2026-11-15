# V2归因补充01：UKT去Rasch同时去掉了题目ID信息

V2方案登记后进一步核查：init_model用num_c构建UKT.n_question（这里57），num_q构建n_pid（948）。UKT.forward的q_data来自cseqs，pid_data来自qseqs。stoc_qid分支的均值/协方差base embedding以知识点和响应为索引，题目ID仅通过difficulty(pid)×KC_delta进入；并未启用aktrasch交互分支或CL。

因此，原UKT推理去Rasch后AUC下降约.0592，同时包含失去题目ID信息、已训练网络输入分布改变等效应，不能说是纯Rasch机制收益。QIKT双GRU基线已有独立题目embedding和题目输出头。V2探索的是在现有信息上加共享调制/改变加入位置，不预期复制这一下降幅度。

不回写已冻结的V2方案/源码或改运行中任务。当前S0/S1/S2/S3范围和判据保持；若有候选，再登记“保留题目ID、以加性题目向量替代乘性题目×KC因子”的匹配对照或原UKT重训消融，才能进一步讨论交互作用。推理干预继续只作为敏感性，不冒称重训因果消融。

证据：pykt/models/init_model.py UKT初始化、pykt/models/ukt.py base_emb与forward，以及保存UKT配置的n_question/num_c、n_pid/num_q映射。两个UKT源码在已有CUDA预检中预测差0，原checkpoint未修改。
