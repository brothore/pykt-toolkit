"""Streaming CPU review of fixed-method seed replications; validation only."""
import argparse
import json
from pathlib import Path
import signal
import threading
import pandas as pd
from compare_validation import compare
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    records=[]
    for task in manifest['jobs']:
        while True:
            path=Path(task['task_dir'])/'state.json';state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') in {'failed','stopped'}:raise RuntimeError(f'Replication prerequisite requires review: {path}')
            if state.get('status')=='completed':break
            atomic_json(out/'progress.json',{'stage':'waiting','task':task['task_id'],'completed':len(records)})
            if stop.wait(5):return 130
        records.append(json.loads(Path(task['result']).read_text()))
        atomic_json(out/'trial_summaries.json',{'trials':records,'test_evaluated':False})
    mixtures={'mask_overlap':{'s2_history_drop005':.5,'s7_overlap64':.5},
              'rank_dominated':{'baseline':.15,'s2_history_drop005':.1,'s5_pairwise_005':.75}}
    combined=[]
    for seed in manifest['seeds']:
        neural=Path(manifest['roots'][str(seed)])
        paths={name:neural/name/'validation_window/window_predictions.tsv' for name in ['baseline','s2_history_drop005','s7_overlap64','s5_pairwise_005']}
        frames={name:pd.read_csv(path,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True) for name,path in paths.items()}
        base=frames['baseline'];keys=['orirow','qidx','late_trues','questions','concepts']
        for frame in frames.values():
            if not frame[keys].equals(base[keys]):raise ValueError('Component identity mismatch')
        for name,weights in mixtures.items():
            if stop.is_set():return 130
            target=out/f'seed{seed}_{name}.json';identity={str(p):digest(p) for p in paths.values()}
            if target.exists():
                result=json.loads(target.read_text())
                if result['input_sha256']!=identity:raise ValueError('Replication inputs changed')
            else:
                candidate=base.copy();candidate['late_mean']=sum(weights.get(n,0)*f.late_mean for n,f in frames.items())
                comparison,_=compare(base,candidate,500,metric_key='validation_window_auc')
                result={'dataset':'nips_task34','fold':0,'seed':seed,'model':'fixed_prediction_mixture','weights':weights,'comparison':comparison,'input_sha256':identity,'test_evaluated':False}
                atomic_json(target,result)
            combined.append(result)
    atomic_json(out/'summary.json',{'trials':records,'fixed_mixtures':combined,'scope':'Seeds 43/44 fixed before training. Held-out validation replication only; no test-based refitting or automatic Peiyou training.','objective_achieved':False})
    atomic_json(out/'progress.json',{'stage':'completed','trials':len(records),'mixtures':len(combined)})
    return 0


if __name__=='__main__':raise SystemExit(main())
