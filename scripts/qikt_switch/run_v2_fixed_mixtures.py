"""C2 fixed validation mixtures and a separate learned-gate prerequisite audit.

No model fitting or test inference. Each mixture is a resumable CPU partition.
Missing OOF evidence records a skipped learned-gate stage, not a success claim.
"""
import argparse
import json
from pathlib import Path
import signal
import threading
import numpy as np
import pandas as pd
from compare_validation import compare
from prediction_progress import atomic_json,digest
from sequence_balance_metrics import assess_balance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    protocol=json.loads(args.protocol.read_text());out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    for parent in protocol['dependencies']:
        path=Path(parent)
        while True:
            state=json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') in {'failed','stopped'}:raise RuntimeError(f'Dependency requires review: {path}')
            if state.get('status')=='completed':break
            atomic_json(out/'progress.json',{'stage':'waiting','dependency':str(path)})
            if stop.wait(5):return 130
    paths={name:Path(value) for name,value in protocol['predictions'].items()}
    identity={str(path):digest(path) for path in paths.values()};identity[str(args.protocol)]=digest(args.protocol)
    frames={name:pd.read_csv(path,sep='\t',dtype={'questions':str,'concepts':str}).sort_values(['orirow','qidx']).reset_index(drop=True) for name,path in paths.items()}
    baseline=frames['baseline'];keys=['orirow','qidx','late_trues','questions','concepts']
    for frame in frames.values():
        if not frame[keys].equals(baseline[keys]):raise ValueError('Mixture input identity mismatch')
    records=[]
    for specification in protocol['mixtures']:
        if stop.is_set():return 130
        target=out/(specification['id']+'.json')
        if target.exists():
            record=json.loads(target.read_text())
            if record['input_sha256']!=identity:raise ValueError('Completed mixture inputs changed')
        else:
            weights=specification['weights']
            if any(v<0 for v in weights.values()) or not np.isclose(sum(weights.values()),1.,atol=1e-12,rtol=0):raise ValueError('Invalid fixed weights')
            candidate=baseline.copy();candidate['late_mean']=sum(weight*frames[name].late_mean for name,weight in weights.items())
            comparison,paired=compare(baseline,candidate,500,metric_key='validation_window_auc');balance=assess_balance(paired,comparison['pooled_delta'])
            paired.to_csv(out/(specification['id']+'_paired.csv'))
            record={'id':specification['id'],'weights':weights,'comparison':comparison,'balance':balance,'input_sha256':identity,'test_evaluated':False,'model_training':'No new model; predeclared prediction average only'}
            atomic_json(target,record)
        records.append(record);atomic_json(out/'progress.json',{'stage':'fixed_mixtures','completed':[r['id'] for r in records]})
    oof=Path(protocol['oof_manifest'])
    audit={'learned_gate_trained':False,'status':'not_run_missing_oof' if not oof.exists() else 'not_run_requires_oof_source_and_fold_review',
        'oof_manifest':str(oof),'oof_manifest_sha256':digest(oof) if oof.exists() else None,
        'required':'Outer fold0 excluded; prediction UID not used for teacher training or selection; source/UID/fold provenance and a separately validated gate implementation required.',
        'note':'Completing this audit is not completing learned-gate training. No gate is fitted to validation/test true AUC or UID.'}
    atomic_json(out/'oof_prerequisite_audit.json',audit)
    families={}
    for name,path in protocol['family_summaries'].items():
        p=Path(path);families[name]={'source':str(p),'sha256':digest(p),'result':json.loads(p.read_text())}
    atomic_json(out/'summary.json',{'version':'V2','family':'C2','fixed_mixtures':records,'learned_gate_audit':audit,'family_review_index':families,'scope':'Validation only; no fit/test/Peiyou auto-adoption','objective_achieved':False})
    atomic_json(out/'progress.json',{'stage':'completed_fixed_mixtures_and_audit','learned_gate_trained':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
