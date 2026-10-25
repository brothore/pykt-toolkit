只读审查结论（未修改、未运行训练）：

## 1. 最小安全接入点

- `init_model.py:275` 的 `else` 之前（即 274 行之后）加 `elif model_name=="simdkt"`，按 DKT 方式传 `num_c`、`emb_type`、`emb_path`（:13）；构造器需接受这些 kwargs。`load_model:280-285` 不用改，但 ckpt 名固定为 `emb_type+"_model.ckpt"`（`train_model.py:673,843`），重载时 emb_type 必须一致。
- `train_model.py:445-448` 前加独立 simdkt 分支，只传 `c.long(),r.long()`；`rshft/cshft` 绝不能进 forward。若只把名字塞进 :445 的 dkt 列表，辅助损失无处放；`cal_loss:249-253` 必须单独 `+preloss[0]`（参考 :278）。`return_per_sample:211` 未含 simdkt，先禁用 ALSO。
- 普通评估 `evaluate_model.py:545-548`、`evaluate_return_results:805-808`、题目 late fusion `evaluate_question:1414-1416` 并入 dkt 分支；forward 返回 `(B,L,num_c)`，用 `one_hot(cshft)` 聚合。fusion 无需改：`effective_fusion:968` 按 qidx 分组、`:954` 求均值、`:1017` cq 空时回退 cc；`group_fusion:1033-1043` 只有 hasearly 才读 hs，故 `hs=[]` 安全。
- `config.py:1,7,13` 不要登记 simdkt：这样 `init_dataset.py:316-321` 继续用 KTDataset，`evaluate_model.py:510` 不会走 `model.model.eval`，也不会触发无 simdkt 分支的 `early_fusion:915`。另 `evaluate_return_results:887` 只传 8 个位置参数，而 `save_cur_predict_result` 签名在 :384 要 12 个，当前必 TypeError，别走该路径。

## 2. masks/smasks/qseqs/cseqs

- `data_loader.py:329` 原始键；:338-347 读 q/c/r/t/ut/smasks/uid；:361-365 rseqs 为 Float，其余 Long；r∈{0,1}，pad=-1。
- `masks` :367 = `(cseqs[:,:-1]!=-1)&(cseqs[:,1:]!=-1)`，形状 (B,L-1)，只表示“当前与下一步概念都非 pad”，不是标签选择；:305-306 用它做 `raw[:-1]*m`、`raw[1:]*m`，pad 被乘成 0（伪概念 0），模型必须显式 mask，不能把 0 当真实概念。
- `smasks` :370 = `(selectmasks[:,1:]!=-1)`，才是 loss/eval 的标签选择；`cal_loss` 用 sm 不用 m（train_model.py:251）。sm=0 可能是“未选”而非 pad，辅助任务应 ignore，不要当负样本。
- `dcur["qseqs"]=q[:-1]`、`shft_qseqs=q[1:]`（:305-309）；y[t] 只用 ≤t 历史预测 r[t+1]，cshft 是下一步概念。qseqs 仅作题级融合；input_type 必须含 concepts（:367 依赖 cseqs），且现有 evaluate 假定 q 可切片（evaluate_model.py:516），端到端应保留 questions。
- 同题重复：KTDataset 每步只有一个 concept（:338；对比 evaluate_model.py:424-432 的“单 int/多 int list”分支），同题多概念只能在预处理展开为连续多行，qseqs 重复。qtest 只给 qidxs/rests/orirow（:357-359）；late fusion 按 qidx 把题内 k 个概念预测求均值（:968,:954），训练时同一 r 也重复 k 次。需按 (uid,qidx/row) 去重或 1/k 加权；辅助窗口不能把同题兄弟概念当独立未来标签。

## 3. 粗心/猜测的可辨识性

数据只有二元 r 和可选 timestamps/usetimes（:341-344），无尝试次数、提示、选项、置信度，DKT 只用 c/r。一次“错”可能是 slip，“对”可能是 guess，二者在正确率上对称，单次观测不可辨识；同题 k 概念复制同一 r 只是 k 条相关标签，不是 k 次独立证据。只能在总体先验/IRT 假设下估 slip/guess（正则化潜变量），不能声称个体级识别；未来窗口正确率也只是含噪聚合。

## 4. 五个关键测试

1. **形状/接口**：合成 (B,L) 的 c/r/rshft/sm，forward 返回 (B,L,num_c)，gather 后 (B,L)∈[0,1]；模型 forward 不依赖 q。
2. **因果性**：改 rshft/cshft 的 t 之后位置，y[:, :t] 不变；断言 forward 只接收 c/r（及 mask），不接收 shft 目标。
3. **padding**：短序列单独 vs 批内 pad 后有效位预测一致；全 pad 窗口 loss=0 无 NaN；sm=0 不计 loss；伪概念 0 不影响有效位。
4. **同题重复**：一题 k 概念，验证主 loss 不被 k 倍放大（或按 1/k），late_mean 等于 k 个预测均值，辅助目标不取同题兄弟的未来概念。
5. **重载/评估**：train_model 保存后 `load_model("simdkt")` 预测逐位一致；evaluate 与 evaluate_question 的 AUC 等于手工 masked AUC；model_name 仍为 simdkt；固定种子下 dkt 输出不变。