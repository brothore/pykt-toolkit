"""Bounded CPU screening of a strictly prequential fixed state algorithm."""
import argparse
import json
from pathlib import Path
import signal
import threading
import pandas as pd
from compare_validation import compare
from prediction_progress import atomic_json,digest
from prefix_residual_memory import calibrate_frame
from sequence_balance_metrics import assess_balance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    design=json.loads(args.protocol.read_text());source=Path(design['baseline_predictions'])
    base=pd.read_csv(source,sep='\t',dtype={'questions':str,'concepts':str})
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    records=[];args.output_dir.mkdir(parents=True,exist_ok=True)
    for number,settings in enumerate(design['rounds'],1):
        if stop.is_set():return 130
        folder=args.output_dir/f'round{number}';folder.mkdir(exist_ok=True)
        protocol=dict(settings,family=design['family'],round=number,dataset='nips_task34',fold=0,seed=42,
                      parent_protocol_sha256=digest(args.protocol),baseline_predictions_sha256=digest(source),
                      algorithm_source_sha256=digest(Path(__file__).with_name('prefix_residual_memory.py')),
                      algorithm='Strictly past residual state; expanded streaming information, not native stateless window-only inference')
        pp=folder/'protocol.json'
        if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Existing round identity differs')
        atomic_json(pp,protocol);balance=folder/'balance.json'
        if balance.exists():
            record=json.loads(balance.read_text())
            if digest(folder/'predictions.tsv')!=record['prediction_sha256'] or digest(folder/'paired_orirow.csv')!=record['paired_orirow_sha256']:
                raise ValueError('Completed prefix residual artifacts changed')
        else:
            predictions,metadata=calibrate_frame(base,design['mode'],settings['prior'],settings['radius'],memory=200)
            target=folder/'predictions.tsv';temporary=target.with_suffix('.tmp')
            predictions.to_csv(temporary,sep='\t',index=False,float_format='%.17g');temporary.replace(target)
            comparison,paired=compare(base,predictions,500,metric_key='validation_window_auc')
            comparison['metric_scope']='Prequential residual state after native validation window scores; additional streaming memory'
            record=assess_balance(paired,comparison['pooled_delta'])
            record['checks'].update(fixed_weak20_preserved=record['deltas']['fixed_baseline_bottom20_mean']>=0,
                                    own_tail20_preserved=record['deltas']['own_bottom20_mean']>=0)
            record['exploratory_candidate_pass']=all(record['checks'].values())
            record['strict_goal_pass']=record['strict_goal_pass'] and record['exploratory_candidate_pass']
            record.update(protocol=protocol,comparison=comparison,prequential_metadata=metadata,
                          prediction_sha256=digest(target),no_new_training=True,test_evaluated=False,
                          evaluation_protocol='Prequential state adaptation, not the unmodified static window prediction script')
            record['substantive_development_signal']=bool(record['exploratory_candidate_pass']
                and record['candidate']['std_ddof0']<=record['baseline']['std_ddof0']*.99
                and record['candidate']['range']<=record['baseline']['range']*.99
                and record['paired_orirow_bootstrap_ci95']['std_delta'][1]<0)
            paired.to_csv(folder/'paired_orirow.csv')
            record['paired_orirow_sha256']=digest(folder/'paired_orirow.csv')
            atomic_json(balance,record)
        records.append(record)
        atomic_json(args.output_dir/'progress.json',{'completed_rounds':number,'candidate_found':record['exploratory_candidate_pass']})
        if record['exploratory_candidate_pass']:break
    atomic_json(args.output_dir/'summary.json',{'family':design['family'],'rounds':records,
        'new_model_training':False,'candidate_found':records[-1]['exploratory_candidate_pass'],
        'test_evaluated':False,'stability_verified':False,'original_static_window_protocol_matched':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
