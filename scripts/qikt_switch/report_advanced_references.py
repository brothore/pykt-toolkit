"""Read-only live reports; separate native intervention from retrained ablation."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
from prediction_progress import atomic_json

NAMES={'full':'完整参考模型','no_time':'去时间分支及时间直加路径，内容分支权重设1','equal_gate':'保留两分支，将学习门控改为固定.5','no_cic':'去CIC残差，保留attention/层归一化','no_alibi':'去ALiBi距离偏置，保留其他结构','no_rasch':'去全部Rasch，同时失去本配置题目ID入口','no_interaction_rasch':'仅去交互侧Rasch，保留查询侧题目ID'}

def read(p):return json.loads(p.read_text())
def table(headers,rows,best=None):
    bestvalues={i:(min if direction=='min' else max)(r[i] for r in rows) for i,direction in (best or {}).items()}
    text=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for row in rows:
        vals=[]
        for i,x in enumerate(row):
            value=f'{x:.12g}' if isinstance(x,float) else str(x)
            if i in bestvalues and x==bestvalues[i]:value='**'+value+'**'
            vals.append(value.replace('|','/'))
        text.append('| '+' | '.join(vals)+' |')
    return '\n'.join(text)+'\n'

def stats(folder):
    r=read(folder/'result.json');s=pd.read_csv(folder/'sequence_auc.csv');s=s[(s.n>=20)&(s.minority>=5)].sort_values('orirow').set_index('orirow');a=s.auc.to_numpy();n=len(a);order=np.argsort(a,kind='stable')
    return r,s,{'pooled':r['native_validation_fragment_auc'],'macro':float(a.mean()),'std':float(a.std(ddof=0)),'range':float(np.ptp(a)),'tail10':float(np.sort(a)[:math.ceil(n*.1)].mean()),'tail20':float(np.sort(a)[:math.ceil(n*.2)].mean()),'strong':float(np.sort(a)[-math.ceil(n*.25):].mean()),'N':n}

def render(root,docs):
    docs.mkdir(parents=True,exist_ok=True);raw=root/'runs';bases={};states={p.parent.name:read(p).get('status') for p in (root/'jobs').glob('*/state.json')};training_tasks=['mtkt_'+x for x in ['full','no_time','no_cic','no_alibi']]+['lefokt_akt_'+x for x in ['full','no_alibi','no_interaction_rasch']]
    scope='''本页为开发验证；nips/fold0，原参考seed3407。原生概念展开片段按原生smasks计分，以UID作为序列标识，std ddof=0。每题多KC和片段边界权重与GRU的题目窗口不同；**不能把本页native_validation_fragment_auc改名为windowauclate_mean，不能与GRU题目窗口直接相减。** 同题多KC标签隔离仍待严格窗口核验，不凭此表独自挑选最终融合模块。
'''
    for model in ['mtkt','lefokt_akt']:
        paths={p.parent.name:p.parent for p in (raw/model/'native_fragments').glob('*/result.json')}
        if model=='lefokt_akt':
            fix=raw/model/'native_interaction_fix1/no_interaction_rasch'
            if (fix/'result.json').exists():paths['no_interaction_rasch']=fix
        rows=[];tailrows=[]
        if 'full' in paths:
            br,bs,b=stats(paths['full']);bases[model]=(br,bs,b)
            for mode in ['full','no_time','equal_gate','no_cic','no_alibi','no_rasch','no_interaction_rasch']:
                if mode not in paths:continue
                v,s,z=stats(paths[mode]);delta=s.auc-bs.auc;balance=v.get('balance',{});order=bs.sort_values('auc',kind='stable');weak10=order.index[:math.ceil(len(order)*.1)];weak20=order.index[:math.ceil(len(order)*.2)];strong=order.index[-math.ceil(len(order)*.25):]
                rows.append([mode,NAMES[mode],f"{v['previous_training_epochs']} / 0",z['pooled'],z['macro'],z['std'],z['std']-b['std'],z['range'],z['range']-b['range'],int((delta<-1e-12).sum())])
                tailrows.append([mode,NAMES[mode],z['tail10'],z['tail20'],float(s.loc[weak10].auc.mean()),float(s.loc[weak20].auc.mean()),float(s.loc[strong].auc.mean()),int((delta<-.01).sum()),float(delta.min()),str(balance.get('paired_orirow_bootstrap_ci95',{}).get('std_delta','[0, 0]'))])
        text='# '+('MTKT' if model=='mtkt' else 'LEFOKT-AKT')+'：开发消融与序列公平性\n\n'+scope
        text+='\n## 已训练权重的推理干预\n\n“历史训练/新增训练epoch”中新增为0；这部分是固定权重敏感性，不是删除模块重训。完整参考模型在每表同行。\n\n'
        text+=table(['条件','实际改动','历史/新增epoch','原生pooled↑','macro↑','std↓','Δstd↓','range↓','Δrange↓','退化数↓'],rows,{3:'max',4:'max',5:'min',7:'min',9:'min'})
        text+='\n固定组均相对该模型自己的完整权重定义。区间为orirow配对500次重采样，不是跨seed稳定性保证。\n\n'+table(['条件','实际改动','自身尾10%↑','自身尾20%↑','固定弱10%↑','固定弱20%↑','固定强25%↑','下降>.01数↓','最差增益↑','Δstd95%区间'],tailrows,{2:'max',3:'max',4:'max',5:'max',6:'max',7:'min',8:'max'})
        text+='\n## 删除模块后从头重训：独立结果表\n\n使用原配置/seed3407，Adam、原生顺序片段、原BCE（LEFOKT加原Rasch正则）；原生开发AUC选模>1e-3、耐心10不变。选中权重再做同链路评估；完整训练也重新跑，和消融匹配，不能拿历史最优权重充当重训对照。支持完整epoch恢复模型/优化器/随机数/早停。\n\n'
        trainrows=[];finished=[]
        for task in training_tasks:
            if not task.startswith(model+'_'):continue
            folder=raw/'retraining'/task;progress=read(folder/'training_progress.json') if (folder/'training_progress.json').exists() else {};info=read(folder/'training_stats.json') if (folder/'training_stats.json').exists() else {}
            trainrows.append([task,NAMES[task[len(model)+1:]],states.get(task,'queued'),info.get('completed_epochs',0),info.get('selected_epoch','待完成'),progress.get('phase','待启动')])
            if (folder/'completion.json').exists():
                value,seq,z=stats(folder/'native_evaluation/full');finished.append((task,value,seq,z,info))
        text+=table(['任务','改动','状态','已完成epoch','当前选中epoch','物理阶段'],trainrows)
        if finished:
            text+='\n以下为已完成重训的绝对结果；后续补入匹配对照的配对差值与保护判断。\n\n'+table(['任务','改动','实际/选中epoch','原生pooled↑','macro↑','std↓','range↓','自身尾10%↑','自身尾20%↑'],[[task,NAMES[task[len(model)+1:]],f"{info['completed_epochs']}/{info['selected_epoch']}",z['pooled'],z['macro'],z['std'],z['range'],z['tail10'],z['tail20']] for task,v,s,z,info in finished],{3:'max',4:'max',5:'min',6:'min',7:'max',8:'max'})
        if model=='mtkt':text+='\n## 初步解释\n\n时间分支有门控及当前题时间直加两条路径，消融同时关闭；CIC是因果双卷积交互，不是普通FFN。当前固定权重去时间/CIC后平均表现与极差变差，说明该权重依赖它们，但重训与严格因果核验前不宣称它们是最佳公平性模块。绝对cosine位置参数requires_grad=False；源码创建Parameter不代表可学习。\n'
        else:text+='\n## 初步解释与失败记录\n\n本配置qid_alibi只启用ALiBi相对位置机制，AKT原gamma遗忘核、FIRE/T5等未走到。去ALiBi时std/range变小但平均表现也下降，不代表公平性有效改善。去全部Rasch也删除题目ID入口；仅去交互侧的对照保留查询侧题目身份。初次交互干预因源码缩进匹配失败，其他三条件保留；fix1另输出并通过CPU真实源测试，不覆盖失败源码/日志。\n'
        text+='\n## 证据\n\n'
        for mode,path in paths.items():text+=f'- {mode}：[完整指标]({path}/result.json)；[逐序列]({path}/sequence_auc.csv)。\n'
        text+=f'\n[实际队列与完整命令]({root}/queue_submission.json)，各任务spec/plan/state及冻结源码位于同一运行根。本页会随已落盘结果更新。\n'
        (docs/('01_MTKT消融与序列公平性.md' if model=='mtkt' else '02_LEFOKT消融与序列公平性.md')).write_text(text)
    upath=raw/'ukt/native_fragments/full'
    if (upath/'result.json').exists():bases['ukt']=stats(upath)
    text='# 三个参考模型：序列表现与公平性对比\n\n'+scope+'\n## 同一开发原生片段链路，完整已存权重\n\n'
    if len(bases)==3:
        ids=set.intersection(*(set(s.index) for v,s,z in bases.values()));weak=set.intersection(*(set(s.sort_values('auc',kind='stable').index[:math.ceil(len(s)*.2)]) for v,s,z in bases.values()))
        rows=[[m,v['seed'],v['previous_training_epochs'],z['N'],z['pooled'],z['macro'],z['std'],z['range'],z['tail10'],z['tail20'],float(s.loc[sorted(weak)].auc.mean())] for m,(v,s,z) in bases.items()]
        text+=f'共同主子集UID数{len(ids)}；“共同弱组”是三个模型各自最弱20%的交集，共{len(weak)}条，仅作验证事后描述，不用于训练路由。\n\n'+table(['模型','seed','历史epoch','主子集N','原生pooled↑','macro↑','std↓','range↓','自身尾10%↑','自身尾20%↑','共同弱组↑'],rows,{4:'max',5:'max',6:'min',7:'min',8:'max',9:'max',10:'max'})
    text+='\n## 用户推荐所采用的旧正式测试口径\n\n以下沿用原报告旧额外完整重叠窗口加权，单类别序列AUC=.5、std ddof0。与上表不同；只作已披露事后观察，不能混减。\n\n'
    original=pd.read_csv('/data/pykt-toolkit-latest/local_runs/sequence_auc_offline_20261106/summary.csv');selected=original[original.model.isin(['mtkt','ukt','lefokt_akt'])];rows=[[r.dataset,r.model,int(r.fold),int(r.seed),int(r.epochs),int(r.sequence_count),float(r.windowauclate_mean),float(r.legacy_std_ddof0),float(r.legacy_range)] for r in selected.itertuples()]
    text+=table(['数据集','模型','fold','seed','历史epoch','序列数','正式windowauclate_mean↑','旧口径std↓','旧口径range↓'],rows)
    text+='\n旧排名混合NIPS/Peiyou四项std/range的等权名次，不能将range较小直接解释为弱组获益。三模型模块选择仍需看重训消融及严格因果题目窗口；尚未据此宣布GRU融合成功。\n\n[研究协议](00_研究协议_20261115.md)；[MTKT单独报告](01_MTKT消融与序列公平性.md)；[LEFOKT单独报告](02_LEFOKT消融与序列公平性.md)。\n'
    (docs/'03_三个参考模型序列公平性比较.md').write_text(text)
    summary={'states':states,'training_completed':sum(states.get(t)=='completed' for t in training_tasks),'training_registered':len(training_tasks),'goal_complete':False,'next':'Audit causal question-window targets, complete matched retraining, select modules with matched evidence, register multiple GRU integration variants and bounded iterations'};atomic_json(root/'report_status.json',summary)
    return all(states.get(t) in ['completed','failed','stopped'] for t in training_tasks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=render(a.root,a.docs)
        if done or not a.watch:return
        time.sleep(20)


if __name__=='__main__':main()
