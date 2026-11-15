"""Compare each completed additive transfer with B0, its matched prior and T2I."""
import argparse
import json
from pathlib import Path
from analyze_strict_reference_ablations import evaluate_pair
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args();repo=a.repo;root=repo/'logs/qikt_lefokt_additive_v3_20261115';out=root/'matched_comparisons';out.mkdir(parents=True,exist_ok=True)
    states={f:json.loads((root/'jobs'/f/'state.json').read_text())['status'] for f in ['LAQ','LAI']}
    if not all(s in ['completed','failed','stopped'] for s in states.values()):raise ValueError('Transfer families not terminal')
    b0=repo/'logs/qikt_gru_ukt_20261114/B0/round1/baseline/validation_window/window_predictions.tsv';strong=repo/'logs/qikt_mtkt_time_v2_20261115/T2I/round2/baseline/validation_window/window_predictions.tsv';records=[];rows=[];identity={}
    for family in ['LAQ','LAI']:
        for path in sorted((root/family).glob('round*/balance.json')):
            trial=json.loads(path.read_text());scale=trial['protocol']['model_settings']['reference_fusion_scale'];current=path.parent/'baseline/validation_window/window_predictions.tsv'
            if family=='LAQ':
                round_=1 if scale==.1 else 2 if scale==.5 else None
                anchor=repo/f'logs/qikt_lefokt_query_v1_20261115/L1Q/round{round_}/baseline/validation_window/window_predictions.tsv'
                control='L1Q标量乘法查询'
            else:
                round_=2 if scale==.5 else 1 if scale==1 else None
                anchor=repo/f'logs/qikt_gru_ukt_v2_20261114/S0/round{round_}/baseline/validation_window/window_predictions.tsv'
                control='S0标量乘法输入'
            if round_ is None:raise ValueError('Unregistered scale, no matched control')
            identity[str(path)]=digest(path)
            for label,baseline,key in [('B0',b0,'B0'),(control,anchor,'matched_multiplicative'),('T2I最低std候选',strong,'T2I'),('G3J最高pooled候选',repo/'logs/qikt_sequence_guided_v3_20261115/G3J/round1/baseline/validation_window/window_predictions.tsv','G3J'),('T2D最低极差候选',repo/'logs/qikt_mtkt_time_v2_20261115/T2D/round1/baseline/validation_window/window_predictions.tsv','T2D')]:
                name=f'{family}_{path.parent.name}_vs_'+key
                value=evaluate_pair(baseline,current,out/name);d=value['deltas'];c=value['baseline'];records.append({'family':family,'round':path.parent.name,'scale':scale,'comparison':label,'baseline_predictions':str(baseline),'candidate_predictions':str(current),'balance':value})
                def arrow(delta,base):return ('↑' if delta<0 else '↓' if delta>0 else '→')+f'{abs(delta)/base*100:.4f}%'
                rows.append([family+'/'+path.parent.name,scale,label,f"{d['pooled_auc']:+.9f}",f"{d['macro_auc']:+.9f}",arrow(d['std_ddof0'],c['std_ddof0']),arrow(d['range'],c['range']),f"{d['fixed_baseline_bottom10_mean']:+.9f}",f"{d['fixed_baseline_top25_mean']:+.9f}",value['worsened'],value['exploratory_candidate_pass'],f"总体AUC {d['pooled_auc']/value['pooled']['baseline']*100:+.4f}%<br>序列均值 {d['macro_auc']/c['macro_auc']*100:+.4f}%<br>标准差 {arrow(d['std_ddof0'],c['std_ddof0'])}<br>极差 {arrow(d['range'],c['range'])}"])
    headers=['加性方案','scale','该行baseline','Δ总体AUC','Δ序列均值','std效果幅度','极差效果幅度','Δ固定弱10%','Δ固定强25%','退化数','联合点通过','相对本行baseline幅度']
    fragment='''\n### 加性迁移是否带来独立增益：三类对照\n\nnips/fold0/seed42、严格开发题目221348、primary773、std ddof=0。每行baseline分别为原B0、同位置／同scale／同标量参数量的既有乘法方案、或当前非支配候选T2I／G3J／T2D。相对既有乘法的比较跨冻结批次，源码版本不同；已有兼容性与初始化检查不能代替跨seed稳定性。弱／强组按该行baseline重新定义，不能混减。std/range箭头↑为缩小、↓为增大。正式windowauclate_mean仍缺失。\n\n'''
    fragment+='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    fragment+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
    fragment+='\n仅对B0通过不能声称超过最强候选；仅对乘法改善也不能称达到总体目标。源加性模型的收益是否迁移，应同时检查上述三类对照、尾部及个体退化，参数量相同也不证明优化几何相同。\n'
    fragment+='\n当前方向状态：'+', '.join(f'{f}={s}' for f,s in states.items())+'。未完成／失败轮次不算已完成实验。\n'
    destination=repo/'docs/先进模型消融与GRU融合/24_LEFOKT加性迁移的匹配比较.md';destination.write_text('# LEFOKT加性迁移：匹配比较\n'+fragment)
    atomic_json(out/'summary.json',{'states':states,'comparisons':records,'source_balance_sha256':identity,'report_fragment':fragment,'document':str(destination),'formal_windowauclate_mean':None})


if __name__=='__main__':main()
