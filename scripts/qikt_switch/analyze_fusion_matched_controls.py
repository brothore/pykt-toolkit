"""Compare completed integrations to their matched structural controls on CPU."""
import argparse
from pathlib import Path
from analyze_strict_reference_ablations import evaluate_pair
from prediction_progress import atomic_json
from render_three_model_summary import table


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args(); repo=a.repo
    m=repo/'logs/qikt_mtkt_cic_v1_20261115'
    l=repo/'logs/qikt_lefokt_query_v1_20261115'
    root=repo/'logs/advanced_reference_20261115/fusion_matched_controls'
    root.mkdir(parents=True,exist_ok=True)
    pairs=[(f'M1{branch}_vs_M0{branch}',m/f'M0{branch}/round1',m/f'M1{branch}/round1',
        '同位置、scale=.1；只比较新CIC直接访问过去位置与同抽头当前状态控制。') for branch in ['Q','C','B']]
    pairs += [(f'L1B_round{r}_vs_L1H_beta05',l/'L1H/round2',l/f'L1B/round{r}',
        '历史显式题目通道同为β=.5；比较加入当前题标量条件化的增量。') for r in [1,2,3]]
    doc='''# 已完成融合的匹配对照复盘

nips_task34 / fold0 / seed42，开发题目窗口221348目标、主子集773 orirow、std ddof=0。正式windowauclate_mean缺失。只分析已有预测，不新增训练或GPU推理。

每个表首行是该比较的结构对照，最后一列按(当前−对照)/对照×100%给出带符号幅度；它与原始B0不同。固定弱／强组也按该表结构对照重新定义，不能直接与B0固定组混减。原B0是否通过仍以[全部融合汇总](13_三模型消融与全部GRU融合汇总.md)为准。

CIC当前状态控制有相同名义参数量和所有卷积抽头，但有效输入几何与时间卷积不同；当前GRU状态本来已含历史。它隔离的是新增模块的直接跨位置访问，不是证明已排除所有容量与优化因素。L1B与L1H都保留相同的间接题目信息路径；β=.5也并不删除完整题目身份。

'''
    results=[]
    for name,base,current,explanation in pairs:
        def prediction(folder):return folder/'baseline/validation_window/window_predictions.tsv'
        value=evaluate_pair(prediction(base),prediction(current),root/name)
        results.append({'comparison':name,'baseline':str(base),'candidate':str(current),'balance':value})
        doc+=f'## {name}\n\n{explanation}\n\n'
        rows=[]
        for label,c,pooled in [('结构对照',value['baseline'],value['pooled']['baseline']),
            ('加入目标机制',value['candidate'],value['pooled']['candidate'])]:
            rows.append([label,pooled,c['macro_auc'],c['std_ddof0'],c['range']])
        doc+=table(['条件','总体AUC ↑','序列平均AUC ↑','标准差 ↓','极差 ↓'],rows,
            {1:'max',2:'max',3:'min',4:'min'})
        d=value['deltas'];ci=value['paired_orirow_bootstrap_ci95']['std_delta']
        doc+=f"\n相对结构对照：固定弱10% Δ={d['fixed_baseline_bottom10_mean']:+.12f}，固定强25% Δ={d['fixed_baseline_top25_mean']:+.12f}；{value['worsened']}条退化，{value['decline_counts']['0.01']}条下降>.01，最差Δ={value['worst_delta']:+.12f}。Δstd配对95%区间{ci}；结构对照联合点通过={value['exploratory_candidate_pass']}。\n\n"
        doc+=f'[完整配对指标]({root/name/"balance.json"})；[逐序列增益]({root/name/"paired_orirow.csv"})。\n\n'
    doc+='''## 对后续实验的约束

相对结构对照的改善不等于相对原B0达标。首轮scale=.1之外的CIC没有匹配同scale控制，不对其作额外历史访问的单独归因。组合中的查询增量需要与相同β的历史结构对照比较，不能拿它相对B0的结果直接当查询模块贡献。

本报告是单seed事后机制分析；配对orirow区间不提供跨seed稳定性或独立学生保证。当前时间信息V2按已登记协议继续，不在训练中根据这些结果修改冻结参数或选模逻辑。
'''
    path=repo/'docs/先进模型消融与GRU融合/14_融合模块匹配对照复盘.md'
    path.write_text(doc)
    atomic_json(root/'summary.json',{'comparisons':results,'document':str(path),'new_training':False,'new_gpu_inference':False})
    print('Completed',len(results),'matched comparisons:',path)


if __name__=='__main__':main()
