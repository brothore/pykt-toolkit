"""Wait for frozen component evaluations, then checkpoint CPU comparisons.

Resume repeats at most the in-flight trial comparison. No parameter fitting.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import pandas as pd
from compare_validation import compare
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set());signal.signal(signal.SIGINT,lambda *_:stopped.set())
    summaries=[]
    for job in manifest['jobs']:
        while True:
            path=Path(job['task_dir'])/'state.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') in {'failed','stopped'}:raise RuntimeError(f'Prerequisite requires review: {path}')
            if state.get('status')=='completed':break
            atomic_json(out/'progress.json',{'stage':'waiting','pending':job['task_id'],'completed':[r['task_id'] for r in summaries]})
            if stopped.wait(5):return 130
        if stopped.is_set():return 130
        source=Path(job['output_dir']);baseline=Path(job['baseline_dir']);result=json.loads((source/'all_results.json').read_text())
        identity={str(p):digest(p) for p in [source/'window_predictions.tsv',source/'all_results.json',baseline/'window_predictions.tsv',baseline/'all_results.json']}
        target=out/(job['dataset']+'_'+job['variant']+'.json')
        if target.exists():
            record=json.loads(target.read_text())
            if record['input_sha256']!=identity:raise ValueError('Comparison inputs changed')
        else:
            a=pd.read_csv(baseline/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
            b=pd.read_csv(source/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
            comparison,paired=compare(a,b,500,metric_key='windowauclate_mean')
            if abs(comparison['candidate_windowauclate_mean']-result['windowauclate_mean'])>1e-12:raise ValueError('Main metric mismatch')
            paired_path=target.with_suffix('.csv');temporary=paired_path.with_suffix('.tmp');paired.to_csv(temporary);os.replace(temporary,paired_path)
            record={'task_id':job['task_id'],'variant':job['variant'],'result':result,'comparison':comparison,'input_sha256':identity,'paired_rows':str(paired_path),'paired_sha256':digest(paired_path)}
            atomic_json(target,record)
        summaries.append(record)
        atomic_json(out/'progress.json',{'stage':'comparing','completed':[r['task_id'] for r in summaries]})
    atomic_json(out/'summary.json',{'trials':summaries,'scope':'Predeclared component confirmation; no test-driven fitting or automatic adoption','objective_achieved':False})
    atomic_json(out/'progress.json',{'stage':'completed','completed':[r['task_id'] for r in summaries]})
    return 0


if __name__=='__main__':raise SystemExit(main())
