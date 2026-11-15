"""Continuously render persisted discovery records without touching training."""
import argparse
import json
from pathlib import Path
import signal
import threading
from prediction_progress import atomic_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args(); stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    names = {'C1':'纯题目标量修正对照','C2':'匹配轻量均衡/排序目标对照','C3':'纯历史重叠64对照',
             'P1':'题目标量＋低频覆盖','P2':'题目标量＋低频覆盖＋弱排序','P3':'历史重叠64＋低频覆盖＋弱排序'}
    registration=json.loads((args.root/'registration.json').read_text())
    names.update(registration.get('family_names',{}))
    while not stop.is_set():
        ledger_path = args.root/'ledger/candidates.json'
        if not ledger_path.exists(): stop.wait(5); continue
        ledger = json.loads(ledger_path.read_text()); results = []
        for item in ledger['rounds']:
            value = json.loads(Path(item['source']).read_text())
            results.append((item['family'], item['round'], value))
        lines = ['# '+registration.get('report_title','V3 实时研究进度'), '', '本文件自动更新；只读已有训练结果。', '',
                 '标准差、极差双降及总体保护是目标；微小点通过不称为显著结果。', '',
                 '|方向|具体改动|队列状态|GPU准入状态|', '|---|---|---|---|']
        for f, state in ledger['states'].items():
            admission=args.root/f/'gpu_admission.json'
            phase=json.loads(admission.read_text())['status'] if admission.exists() else '等待前置/按原阶段调度'
            lines.append(f"|{f}|{names.get(f,f)}|{state['status']}|{phase}|")
        gate = Path(registration.get('evaluation_gate_result',args.root/'native_audit_r2/audit_result.json'))
        evaluation_note=registration.get('evaluation_note','原生核验：'+('已完成，具体差值见审计JSON；不冒称通用late-fusion脚本输出' if gate.exists() else '尚未成功完成；窗口评估需等待'))
        lines.extend(['', evaluation_note, '',
                      registration.get('evaluation_header','nips/QIKT-Mamba/fold0/seed42；primary773，ddof=0。验证指标，不代替正式windowauclate_mean。'), '',
                      '|方向/轮次|改动|实际/选中epoch|pooled AUC|macro AUC|std|range|退化数|初步候选|实质开发信号|',
                      '|---|---|---|---|---|---|---|---|---|---|'])
        data = []
        for f, rd, v in results:
            c = v['comparison']; z = v['candidate']; t = v.get('training', {})
            data.append([f+'/'+rd, names.get(f,f),(str(t.get('completed_epochs'))+'/'+str(t.get('selected_epoch')) if t else '无新增训练'),
                         c['candidate_validation_window_auc'], z['macro_auc'], z['std_ddof0'], z['range'],
                         v['worsened'],'对照满足点条件，非优化候选' if f in registration.get('control_families',[]) and v['exploratory_candidate_pass'] else v['exploratory_candidate_pass'],v.get('substantive_development_signal',False)])
        if results:
            v = results[0][2]; b = v['baseline']; c = v['comparison']
            data.insert(0,['Baseline','原始模型','复用',c['baseline_validation_window_auc'], b['macro_auc'],
                           b['std_ddof0'],b['range'],0,'参照','参照'])
        best = {j: (min if j in (5,6,7) else max)(row[j] for row in data) for j in (3,4,5,6,7)} if data else {}
        for row in data:
            values = []
            for j, x in enumerate(row):
                text = f'{x:.12g}' if isinstance(x,float) else str(x)
                if j in best and x == best[j]: text = '**'+text+'**'
                values.append(text)
            lines.append('|'+ '|'.join(values)+'|')
        text = '\n'.join(lines)+'\n'; target = args.root/registration.get('report_filename','V3_实时研究报告.md')
        temporary = target.with_suffix('.tmp'); temporary.write_text(text); temporary.replace(target)
        terminal = all(s['status'] in ('completed','failed','stopped') for s in ledger['states'].values())
        if terminal:
            atomic_json(args.root/'report_complete.json', {'terminal':True,'report':str(target),
                'completed_comparisons':len(results),'substantive_signal_count':sum(bool(v.get('substantive_development_signal')) for _,_,v in results),
                'goal_achieved':False,'next':'Main scientific review required; no automatic stability expansion'})
            return 0
        stop.wait(15)
    return 130


if __name__ == '__main__': raise SystemExit(main())
