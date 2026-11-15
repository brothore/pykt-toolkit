"""Post-hoc sequence beneficiaries; explanatory groups never become routing labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prediction_progress import atomic_json,digest


def markdown(headers,rows):
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    pair = (2,3) if '对照平均AUC' in headers else (3,4) if '对照AUC' in headers else None
    for row in rows:
        cells=[]
        for index,x in enumerate(row):
            text=f'{x:.6f}' if isinstance(x,float) else str(x)
            if pair and index in pair and x==max(row[pair[0]],row[pair[1]]):text='**'+text+'**'
            cells.append(text)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    repo=a.repo;ref=repo/'logs/advanced_reference_20261115';rawpath=ref/'runs/data/raw_questions.csv';raw=pd.read_csv(rawpath);train=raw[raw.fold!=0];valid=raw[raw.fold==0];counts=Counter()
    for row in train.itertuples():counts.update(map(int,row.questions.split(',')))
    rare=float(np.quantile(list(counts.values()),.25));features=[]
    for row in raw.itertuples():
        q=list(map(int,row.questions.split(',')));cs=[set(map(int,x.split('_'))) for x in row.concepts.split(',')];y=np.fromstring(row.responses,sep=',');t=np.fromstring(row.timestamps,sep=',');gaps=np.maximum(np.diff(t),0)/60000
        support=Counter();supports=[]
        for i,kcs in enumerate(cs):
            if i:supports.append(np.mean([support[k] for k in kcs]))
            support.update(kcs)
        features.append({'orirow':int(row.uid),'fold':row.fold,'question_targets':len(q)-1,'correctness_posthoc':float(y[1:].mean()),'switch_rate_posthoc':float(np.mean(np.diff(y)!=0)),'rare_question_fraction':float(np.mean([counts[k]<=rare for k in q[1:]])),'distinct_kcs':len(set.union(*cs)),'mean_prefix_kc_support':float(np.mean(supports)),'mean_log_gap':float(np.log2(gaps+1).mean()),'large_gap_fraction':float((gaps>=60).mean()),'multikc_fraction':float(np.mean([len(c)>1 for c in cs[1:]]))})
    features=pd.DataFrame(features);thresholds={k:float(features[features.fold!=0][k].quantile(.75)) for k in ['mean_log_gap','rare_question_fraction','mean_prefix_kc_support','distinct_kcs','large_gap_fraction']};f=features[features.fold==0].set_index('orirow');assert f.index.is_unique
    f.to_csv(a.output/'validation_sequence_features.csv')
    native=ref/'ukt_sensitivity_balance';strict=ref/'runs/strict_questions'
    cases=[('ukt_rasch','ukt','UKT Rasch：完整权重−去Rasch','native',native/'no_rasch_sequence_auc.csv',native/'full_sequence_auc.csv'),('ukt_cov_distance','ukt','UKT协方差距离：完整权重−仅均值距离','native',native/'mean_distance_only_sequence_auc.csv',native/'full_sequence_auc.csv'),('ukt_cov_aggregation','ukt','UKT α²协方差聚合：完整权重−α聚合','native',native/'linear_covariance_aggregation_sequence_auc.csv',native/'full_sequence_auc.csv'),('mtkt_cic','mtkt','MTKT CIC：完整匹配重训−去CIC','strict',strict/'mtkt_no_cic_strict_trained/sequence_auc.csv',strict/'mtkt_full_strict_trained/sequence_auc.csv'),('mtkt_time','mtkt','MTKT时间路径：完整匹配重训−去时间','strict',strict/'mtkt_no_time_strict_trained/sequence_auc.csv',strict/'mtkt_full_strict_trained/sequence_auc.csv'),('lefokt_remove_history','lefokt_akt','LEFOKT去历史交互Rasch：删除后−完整匹配重训','strict',strict/'lefokt_akt_full_strict_trained/sequence_auc.csv',strict/'lefokt_akt_no_interaction_rasch_strict_trained/sequence_auc.csv'),('lefokt_alibi','lefokt_akt','LEFOKT ALiBi：完整匹配重训−去ALiBi','strict',strict/'lefokt_akt_no_alibi_strict_trained/sequence_auc.csv',strict/'lefokt_akt_full_strict_trained/sequence_auc.csv')]
    extended=repo/'logs/lefokt_extended_ablation_v2_fix01_20261115/strict_questions'
    for mode,label in [('no_query_rasch','LEFOKT查询侧乘法：完整−去查询侧'),('no_both_rasch','LEFOKT双侧乘法：完整−去双侧')]:
        path=extended/mode/'sequence_auc.csv'
        if path.exists():cases.append(('lefokt_'+mode,'lefokt_akt',label,'strict',path,strict/'lefokt_akt_full_strict_trained/sequence_auc.csv'))
    additive=extended/'additive_rasch/sequence_auc.csv'
    if additive.exists():
        cases.append(('lefokt_additive','lefokt_akt','LEFOKT保留题目ID的加性替代：加性−原乘法','strict',strict/'lefokt_akt_full_strict_trained/sequence_auc.csv',additive))
    base=repo/'logs/qikt_gru_ukt_20261114/B0/round1/baseline/validation_window/orirow_auc.csv'
    for family,round_ in [('T2I','round2'),('T2D','round1'),('T2G','round1')]:
        cases.append((family,'gru',family+'时间融合−双GRU B0','strict',base,repo/f'logs/qikt_mtkt_time_v2_20261115/{family}/{round_}/baseline/validation_window/orirow_auc.csv'))
    guided=repo/'logs/qikt_sequence_guided_v3_20261115'
    for family in ['G3J','G3S']:
        candidate=guided/family/'round1/baseline/validation_window/orirow_auc.csv'
        if candidate.exists():cases.append((family,'gru',family+'序列分析驱动结构−双GRU B0','strict',base,candidate))
    additive_root=repo/'logs/qikt_lefokt_additive_v3_20261115'
    for family in ['LAQ','LAI']:
        for trial in sorted((additive_root/family).glob('round*/balance.json')):
            path=trial.parent/'baseline/validation_window/orirow_auc.csv'
            if path.exists():cases.append((family+'_'+trial.parent.name,'gru',family+'/'+trial.parent.name+'标量加性条件−双GRU B0','strict',base,path))
    sections={k:'' for k in ['ukt','mtkt','lefokt_akt','gru']};records=[];identity={str(rawpath):digest(rawpath)}
    for name,model,title,scope,baseline,candidate in cases:
        b=pd.read_csv(baseline).set_index('orirow');c=pd.read_csv(candidate).set_index('orirow');eligible=(b.n>=20)&(b.minority>=5)
        z=b.loc[eligible,['auc','n','minority']].rename(columns={'auc':'baseline_auc'}).join(c.auc.rename('candidate_auc'),how='inner').join(f,how='left');assert z.question_targets.notna().all();z['delta']=z.candidate_auc-z.baseline_auc
        z.to_csv(a.output/(name+'_paired_features.csv'));identity.update({str(p):digest(p) for p in [baseline,candidate]})
        weak=z.baseline_auc.rank(method='first',pct=True)<=.25;strong=z.baseline_auc.rank(method='first',pct=True)>.75
        masks={'全部主序列':np.ones(len(z),dtype=bool),'该对照弱25%':weak,'该对照强25%':strong,'短序列<100题':z.question_targets<100,'长序列≥400题':z.question_targets>=400,'低正确率<.4（事后）':z.correctness_posthoc<.4,'高切换≥.5（事后）':z.switch_rate_posthoc>=.5,'低频题覆盖>训练Q75':z.rare_question_fraction>thresholds['rare_question_fraction'],'时间间隔≥训练Q75':z.mean_log_gap>=thresholds['mean_log_gap'],'长中断比例>训练Q75':z.large_gap_fraction>thresholds['large_gap_fraction'],'相关历史覆盖≥训练Q75':z.mean_prefix_kc_support>=thresholds['mean_prefix_kc_support'],'KC种类≥训练Q75':z.distinct_kcs>=thresholds['distinct_kcs']}
        rows=[];groups=[]
        for group,mask in masks.items():
            sub=z.loc[mask]
            if not len(sub):continue
            gain=float(sub.delta.mean());groups.append({'name':group,'N':len(sub),'baseline_mean':float(sub.baseline_auc.mean()),'candidate_mean':float(sub.candidate_auc.mean()),'delta_mean':gain,'improved':int((sub.delta>0).sum()),'declined':int((sub.delta<0).sum())})
            rows.append([group,len(sub),float(sub.baseline_auc.mean()),float(sub.candidate_auc.mean()),f'{gain:+.6f}',f'{gain/sub.baseline_auc.mean()*100:+.4f}%',int((sub.delta>0).sum()),int((sub.delta<0).sum())])
        s=f'\n### 哪些序列受益：{title}\n\n'
        s+=('固定权重原生概念片段敏感性；不是删除模块重训。去Rasch同时删除题目ID，协方差两个干预不能合称一种因果贡献。' if scope=='native' else '严格题目匹配比较；每表弱／强组由本项结构对照定义。')+' 下表是事后描述，分组重叠，不能用这些序列真实AUC或完整响应特征训练路由。\n\n'
        s+=markdown(['序列群体','N','对照平均AUC','当前平均AUC','平均ΔAUC','均值相对对照幅度','提升数','下降数'],rows)
        top=z.sort_values('delta',ascending=False).head(5);bottom=z.sort_values('delta').head(3);examples=[]
        for label,frame in [('提升最多（事后挑选）',top),('退化最多（事后挑选）',bottom)]:
            for uid,row in frame.iterrows():examples.append([label,int(uid),int(row.question_targets),float(row.baseline_auc),float(row.candidate_auc),f'{row.delta:+.6f}',f'{row.delta/row.baseline_auc*100:+.4f}%' if row.baseline_auc else '不适用（对照为0）',float(row.rare_question_fraction),float(row.mean_log_gap)])
        s+='\n具体序列示例：UID对应开发原始序列；题数为题目目标数，不是UKT概念展开计分数。极端示例不是代表性样本，更不能作为训练标签。\n\n'+markdown(['类型','UID','题目目标数','对照AUC','当前AUC','ΔAUC','相对对照幅度','低频题比例','平均log2分钟间隔'],examples)
        covariance=float(np.mean((z.baseline_auc-z.baseline_auc.mean())*(z.delta-z.delta.mean())));gainvar=float(np.var(z.delta));variance_delta=float(np.var(z.candidate_auc)-np.var(z.baseline_auc));assert abs(variance_delta-(2*covariance+gainvar))<1e-12
        s+=f'\n均衡解释：2Cov(对照AUC,增益)={2*covariance:+.9f}，Var(增益)={gainvar:.9f}，合计方差变化={variance_delta:+.9f}。负协方差表示较弱序列倾向获得更多增益；仍需足够抵消增益方差才会降低整体标准差。单次观察不能证明成因。\n'
        sections[model]+=s;records.append({'name':name,'scope':scope,'groups':groups,'covariance_twice':2*covariance,'gain_variance':gainvar,'variance_delta':variance_delta})
    intro='\n分组阈值来自训练fold（低频题频次数Q25，群体特征Q75），不是从测试或增益中搜索最佳切点；“弱25%”与极端UID示例是披露的事后诊断。序列身份是验证UID；各原生／严格评估口径分别解释。群体样本少于30时仅作个例线索，分组重叠、不作多重比较显著性声明。群体不同不跨行排名；同行对照／当前AUC的较好值加粗。\n'
    sections={k:intro+v for k,v in sections.items()}
    atomic_json(a.output/'profiles.json',{'sections':sections,'records':records,'training_thresholds':thresholds,'rare_question_train_q25':rare,'input_sha256':identity,'analyzer_sha256':digest(Path(__file__)),'features_for_explanation_only':True,'formal_windowauclate_mean':None})
    print('Completed',len(records),'module/candidate sequence profiles')


if __name__=='__main__':main()
