"""Paired comparisons to a negative control from existing predictions only."""
import argparse
import json
from pathlib import Path
import signal
import threading
import pandas as pd
from compare_validation import compare
from sequence_balance_metrics import assess_balance
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--control',default='G0')
    args=ap.parse_args();root=args.root;reg=json.loads((root/'registration.json').read_text())
    families=[f for f in reg['families'] if f!=args.control];out=root/'control_comparisons';out.mkdir(exist_ok=True)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    folder=root/args.control/'round1';cb=json.loads((folder/'balance.json').read_text())
    source=folder/'baseline/validation_window/window_predictions.tsv';base=pd.read_csv(source,sep='\t',dtype={'questions':str,'concepts':str})
    base_sha=digest(source)
    while not stop.is_set():
        results=[];states={}
        for f in families:
            states[f]=json.loads((root/'jobs'/f/'state.json').read_text())['status']
            for p in sorted((root/f).glob('round*/balance.json')):
                prior=json.loads(p.read_text());pred=p.parent/'baseline/validation_window/window_predictions.tsv'
                dest=out/f/p.parent.name;dest.mkdir(parents=True,exist_ok=True)
                identity={'control':base_sha,'candidate':digest(pred),'original_result':digest(p),'source':digest(Path(__file__))}
                result=dest/'comparison.json';ip=dest/'identity.json'
                if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Control comparison identity differs')
                atomic_json(ip,identity)
                if result.exists():record=json.loads(result.read_text())
                else:
                    candidate=pd.read_csv(pred,sep='\t',dtype={'questions':str,'concepts':str})
                    comparison,paired=compare(base,candidate,500,metric_key='validation_window_auc')
                    assessment=assess_balance(paired,comparison['pooled_delta'])
                    record={'family':f,'round':p.parent.name,'comparison_to_control':comparison,'balance_to_control':assessment,
                            'original_baseline_result':str(p),'scope':'Validation only; no extra seeds/training/inference; control groups differ from original baseline groups'}
                    temporary=dest/'paired_orirow.tmp';paired.to_csv(temporary);temporary.replace(dest/'paired_orirow.csv')
                    atomic_json(result,record)
                results.append(record)
        lines=['# V5：原baseline、零排序控制与新方法','',
               'nips_task34/QIKT-Mamba/public_mamba/fold0/seed42；primary773，ddof=0。仅验证集，正式windowauclate_mean未取得。', '',
               'G0只加索引包装、不加入排序优化；其小幅点改善不能视为新方法有效。下表同时显示相对原baseline和相对G0的变化，防止只选有利参照。所有CI都是开发序列重采样，不代表跨seed稳定。','',
               '|方案|做了什么|pooled AUC ↑|Δpooled对原baseline|Δpooled对G0|macro AUC ↑|std ↓|Δstd对原baseline|Δstd对G0|range ↓|Δrange对原baseline|Δrange对G0|对G0的Δstd 95%区间|','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        b=cb['baseline'];c=cb['candidate'];old_auc=cb['comparison']['baseline_validation_window_auc'];control_auc=cb['comparison']['candidate_validation_window_auc']
        data=[['Baseline','原模型',old_auc,0.,old_auc-control_auc,b['macro_auc'],b['std_ddof0'],0.,b['std_ddof0']-c['std_ddof0'],b['range'],0.,b['range']-c['range'],'参照'],
              [args.control,'索引包装，零排序控制',control_auc,cb['deltas']['pooled_auc'],0.,c['macro_auc'],c['std_ddof0'],cb['deltas']['std_ddof0'],0.,c['range'],cb['deltas']['range'],0.,'[0,0]自身相减']]
        for record in results:
            prior=json.loads(Path(record['original_baseline_result']).read_text());a=prior['candidate'];d=prior['deltas'];rel=record['balance_to_control'];f=record['family'];pp=prior['protocol']
            description=reg.get('family_names',{}).get(f,f)+f"；rank={pp['rank_weight']}"
            data.append([f+'/'+record['round'],description,prior['comparison']['candidate_validation_window_auc'],d['pooled_auc'],record['comparison_to_control']['pooled_delta'],a['macro_auc'],a['std_ddof0'],d['std_ddof0'],rel['deltas']['std_ddof0'],a['range'],d['range'],rel['deltas']['range'],str(rel['paired_orirow_bootstrap_ci95']['std_delta'])])
        best={j:(min if j in (6,7,8,9,10,11) else max)(r[j] for r in data) for j in range(2,12)}
        for row in data:
            values=[]
            for j,value in enumerate(row):
                text=f'{value:.12g}' if isinstance(value,float) else str(value)
                if j in best and value==best[j]:text='**'+text+'**'
                values.append(text)
            lines.append('| '+' | '.join(values)+' |')
        lines.extend(['','与原baseline比较的保护/退化和训练epoch见各原始balance.json及实时报告；相对G0的配对结果和CI见control_comparisons各目录。相对G0改善不是目标完成，仍必须满足原baseline保护与实质幅度。'])
        path=root/'V5_原基线与控制的双重对照.md';temp=path.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(path)
        atomic_json(out/'progress.json',{'states':states,'completed_comparisons':len(results),'goal_achieved':False})
        if all(s in ('completed','failed','stopped') for s in states.values()):
            atomic_json(out/'summary.json',{'results':results,'states':states,'goal_achieved':False});return 0
        stop.wait(15)
    return 130


if __name__=='__main__':raise SystemExit(main())
