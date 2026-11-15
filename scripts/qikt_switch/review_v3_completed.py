"""Audit completed V3 artifacts and explain matched controls, read-only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;names={'C1':'仅题目ID标量修正','C2':'仅轻量等序列/片内排序目标','C3':'仅历史重叠64',
        'P1':'题目修正＋低频覆盖','P2':'题目修正＋覆盖＋弱排序','P3':'重叠历史＋覆盖＋弱排序',
        'S1':'P1首轮＋少量A2预测','S2':'P2首轮＋少量A3预测','S3':'P3首轮＋少量A2预测'}
    identities={};records=[];decompositions=[]
    for family in names:
        job=root/'jobs'/family;state=json.loads((job/'state.json').read_text());plan=json.loads((job/'plan.json').read_text())
        if state.get('status')!='completed' or state.get('exit_code')!=0:raise ValueError(f'{family} not successfully completed')
        for category in ['input_sha256','source_sha256']:
            for path,sha in plan[category].items():
                if path in identities and identities[path]!=sha:raise ValueError('Conflicting identity')
                identities[path]=sha
        identities.update(state['output_sha256'])
        summary=json.loads((root/family/'summary.json').read_text())
        for i,value in enumerate(summary['rounds'],1):
            pp=root/family/f'round{i}'/'paired_orirow.csv';frame=pd.read_csv(pp);frame=frame[(frame.n>=20)&(frame.minority>=5)]
            if len(frame)!=773:raise ValueError('Primary cohort mismatch')
            x=frame.baseline_auc.to_numpy();g=frame.delta_auc.to_numpy();y=frame.auc.to_numpy()
            if np.max(np.abs(x+g-y))>1e-12:raise ValueError('Paired gains differ')
            for key,actual in [('macro_auc',float(y.mean())),('std_ddof0',float(y.std())),('range',float(np.ptp(y)))]:
                if abs(actual-value['candidate'][key])>1e-12:raise ValueError('Paired metrics disagree with summary')
            cov=2*np.mean((x-x.mean())*(g-g.mean()));variance=np.var(g)
            if abs(cov+variance-(np.var(y)-np.var(x)))>1e-15:raise ValueError('Variance identity mismatch')
            ends={}
            for name,at in [('baseline_min',np.argmin(x)),('baseline_max',np.argmax(x)),('candidate_min',np.argmin(y)),('candidate_max',np.argmax(y))]:
                row=frame.iloc[at];ends[name]={'orirow':int(row.orirow),'baseline_auc':float(row.baseline_auc),'candidate_auc':float(row.auc),'delta':float(row.delta_auc)}
            records.append({'family':family,'round':i,'value':value})
            decompositions.append({'family':family,'round':i,'twice_covariance':float(cov),'variance_gain':float(variance),
                'variance_delta':float(np.var(y)-np.var(x)),'endpoints':ends,'paired':str(pp),'paired_sha256':digest(pp)})
    for path,sha in identities.items():
        if digest(Path(path))!=sha:raise ValueError(f'Changed source/input/output: {path}')
    if len(records)!=21:raise ValueError('Expected 21 complete V3 results')
    out=root/'completed_review';out.mkdir(exist_ok=True)
    atomic_json(out/'audit.json',{'verified_unique_files':len(identities),'successful_families':9,'completed_comparisons':21,
        'candidate_count':sum(r['value']['exploratory_candidate_pass'] for r in records),
        'substantive_signal_count':sum(r['value'].get('substantive_development_signal',False) for r in records),
        'test_evaluated':False,'formal_windowauclate_mean':None,'variance_decompositions':decompositions})
    lines=['# V3完整完成复盘：21条结果与匹配对照','','本报告的C1/C2/C3是V3对照编号，不是V2同名方案。9个方向全部正常完成，12轮训练＋9次固定融合；没有联合点候选，也没有达到实质开发信号。', '',
        '全部nips_task34/QIKT-Mamba/public_mamba/fold0/seed42验证；primary773，ddof=0。正式windowauclate_mean缺失。原生核验匹配主子集macro/std/range，但不是通用late-fusion脚本输出；不得把验证pooled改名为正式字段。','']
    def table(headers,rows,directions=None):
        directions=directions or {};best={j:(min if dr=='min' else max)(r[j] for r in rows) for j,dr in directions.items()}
        lines.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        for row in rows:
            values=[]
            for j,x in enumerate(row):
                s=f'{x:.12g}' if isinstance(x,float) else str(x)
                if j in best and x==best[j]:s='**'+s+'**'
                values.append(s)
            lines.append('| '+' | '.join(values)+' |')
        lines.append('')
    v=records[0]['value'];b=v['baseline'];rows=[['Baseline','原始模型',v['comparison']['baseline_validation_window_auc'],b['macro_auc'],b['std_ddof0'],b['range'],0,'参照']]
    for r in records:
        v=r['value'];c=v['candidate'];rows.append([r['family']+f'R{r["round"]}',names[r['family']],v['comparison']['candidate_validation_window_auc'],c['macro_auc'],c['std_ddof0'],c['range'],v['worsened'],v['exploratory_candidate_pass']])
    table(['方案','改动','pooled AUC ↑','macro AUC ↑','std ↓','range ↓','退化数 ↓','初步联合条件'],rows,{2:'max',3:'max',4:'min',5:'min',6:'min'})
    lines.extend(['## 已有对照能区分什么','','以下均为“左侧−右侧”，参照不同，不作跨行最优排名；它们仍是单seed、随机轨迹耦合的描述，不等于普遍因果结论。',''])
    lookup={(r['family'],r['round']):r['value'] for r in records};data=[]
    for round in [1,2,3]:
        a=lookup['P2',round];b=lookup['P1',round]
        data.append([f'P2R{round}',f'P1R{round}','匹配覆盖/题目模块，增加弱排序',a['comparison']['candidate_validation_window_auc']-b['comparison']['candidate_validation_window_auc'],a['candidate']['std_ddof0']-b['candidate']['std_ddof0'],a['candidate']['range']-b['candidate']['range']])
    table(['方案','参照','区别','Δpooled','Δstd','Δrange'],data)
    lines.extend(['## 机制结论','','- 仅题目修正C1带来pooled和std/range的小改善，但强组均值下降；这证明它不是完全不起作用，也不能说已经无损补弱。',
        '- 仅轻量目标C2仍pooled下降且std/range扩大。结合旧D1的组合结果，模块与目标可能有交互，不能把旧收益归给排序目标一个部分。',
        '- 纯重叠C3本身已提高总体/宏均值但扩大std，说明历史覆盖是总体收益的重要线索，增益分散问题并非只由新增排序引入。',
        '- P1第三轮是本批较有价值的局部结果：pooled/macro/std、尾部和强组均值均保持，但range扩大，仍未通过。固定弱组改善不代表最低端点同步改善，端点在audit.json中逐轮保留。',
        '- P2相对匹配P1的增量在三轮不一致：排序没有稳定修复极差或标准差。不能继续假设只要降低系数就会自然得到联合收益。',
        '- P3较纯重叠C3减小了部分std扩大，但仍大于原baseline；覆盖/排序有调节线索，却没有解决增益分散。',
        '- S1有两轮双降，却损失pooled；S2仍range或总体保护失败；S3总体更好但std继续扩大。旧F2的少量互补不能直接迁移到新主体。',
        '- 因此V4换到更直接的目标对齐环节：完整训练UID内跨切片配对、对大错序风险有界化、类别支持与温和关注，并保留仅分组对照。不是继续同配方无限搜索。','',
        '## 训练身份与来源','','|方案|实际epoch/选中epoch|关键参数|协议|结果|','|---|---|---|---|---|'])
    for r in records:
        v=r['value'];t=v.get('training',{});name=r['family']+f'R{r["round"]}';source=root/r['family']/f'round{r["round"]}'
        epoch=f'{t.get("completed_epochs")}/{t.get("selected_epoch")}' if t else '固定融合无新增训练，组件见协议'
        params={k:v['protocol'][k] for k in ['alpha','rank_weight','frequency_alpha','frequency_cap','frequency_power','model_settings','candidate_weight'] if k in v['protocol']}
        lines.append(f'|{name}|{epoch}|{json.dumps(params,ensure_ascii=False)}|[参数]({source/"protocol.json"})|[完整结果]({source/"balance.json"})|')
    lines.extend(['',f'原始源码/输入/输出校验核对{len(identities)}个独立文件；21条方差恒等分解与端点、配对文件SHA见[audit.json]({out/"audit.json"})。没有新训练、没有用测试标签开发，也不修改已完成结果。'])
    (out/'V3_完成复盘.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
