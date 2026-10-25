已创建 `pykt/models/dkt_first_action.py`，仅新增该文件，未修改其他文件。

实现要点：

- `DKTFirstAction(DKT)`：先调用父类构造，保留 DKT backbone、参数名与继承的 `forward(c, r) -> [B,T,C]`；`model_name='dkt_first_action'`。
- 构造参数包含 DKT 全部参数及 `action_weight=0.1`、`action_mode='joint'`、`action_balance=0.0`、`action_class_counts=None`、`action_hidden=100`；对 mode、隐藏层大小、weight/balance 范围、counts 长度与合法性、`emb_type='qid'` 做了显式校验。
- 辅助头为 `Linear(2*emb_size, action_hidden) -> Tanh -> Linear(action_hidden, 4/3)`，输入是 dropout 后 LSTM `h_t` 与目标知识点在 `interaction_emb` 两个回答通道嵌入均值的拼接；head 构造完全位于 `torch.random.fork_rng(devices=[])` 内，因此不推进父类初始化后的全局 RNG 状态，head 内无 dropout。
- 类别权重由 `counts.clamp(min=1) ** (-action_balance)` 得到，归一化为均值 1；conditional 使用 `counts[1:]`；以 buffer `action_class_weights` 保存；counts 为空时使用全 1。
- `forward_details(c, r, target_c)` 用与 DKT 相同的 backbone 计算返回 `{'predictions': [B,T,C], 'action_logits': [B,T,4/3]}`，只读取 `c/r` 与已知的 `target_c`。
- `compute_loss(...)`：
  - 主 loss 用 `sm = masks & smasks` 对按 `shft_cseqs` 取出的预测做 masked BCE(double)，per-sample 分支与现有 `cal_loss` 的归一化方式一致。
  - `action_weight=0` 时只走主 loss，不读取 action 字段。
  - 辅助 loss 仅在 `action_mask & sm & 合法标签` 位置计算；joint 为 4 类 CE，conditional 只用 `label>0` 的 3 类 CE（目标为 `label-1`）；未知标签直接抛 `ValueError`。
  - 加权 CE 为 `sum(loss*weight)/sum(weight)`；空 mask 返回与计算图相连的 0 且无 NaN。
  - 支持 `return_per_sample`、`return_components`，总 loss 为 `main + action_weight * aux`。
- `action_probabilities(data)` 返回 `[B,T,4]`：joint 为 4 类 softmax；conditional 为 `cat[p_selected, (1-p_selected)*softmax3]`，仅依赖可见的 `c/r` history 与 `shft_cseqs`。

未执行任何测试：本机 bash sandbox 后端不可用，尝试一次 `py_compile` 被环境拒绝，按约定未重试；因此未验证运行结果。