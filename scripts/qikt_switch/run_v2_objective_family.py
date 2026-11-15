"""Two preregistered rounds of A1/A2, checkpointed at each durable phase.

R2 coefficients follow the published V2 rule, never test scores. Requeue the
same family --resume after stopping. Training/evaluation preserve own checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from advance_experiment import evaluate_finished
from compare_validation import compare
from prediction_progress import atomic_json,digest
from sequence_balance_metrics import assess_balance
from resource_admission import wait_for_capacity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['A1','A2','A3','B1'],required=True);ap.add_argument('--work-root',type=Path,required=True);ap.add_argument('--seed',type=int,choices=[42,43,44],default=42);ap.add_argument('--fixed-protocol',type=Path);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];script=Path(__file__).resolve().parent;family=args.work_root/args.family;family.mkdir(parents=True,exist_ok=True)
    baseline=root/'logs/qikt_switch_20261103/neural/baseline/validation_window'
    if args.seed!=42:baseline=root/f'logs/qikt_switch_20261103/seed_replication/neural_seed{args.seed}/baseline/validation_window'
    base_frame=pd.read_csv(baseline/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
    frozen=json.loads(args.fixed_protocol.read_text()) if args.fixed_protocol else None
    if frozen and (frozen['family']!=args.family or frozen['fold']!=0):raise ValueError('Fixed protocol identity mismatch')
    rounds=[int(frozen['round'])] if frozen else [1,2]
    previous=None
    for number in rounds:
        round_dir=family/f'round{number}';round_dir.mkdir(exist_ok=True);trial=round_dir/'baseline';trial.mkdir(exist_ok=True)
        if frozen:alpha=float(frozen['alpha']);rank=float(frozen['rank_weight'])
        elif args.family=='A1':
            alpha=.25 if number==1 else .5 if previous['exploratory_candidate_pass'] else .1
            rank=0.
        elif args.family=='A2':
            alpha=.25;rank=.01 if number==1 else .05 if previous['exploratory_candidate_pass'] else .002
        else:alpha=0.;rank=0.
        protocol={'version':'V2','family':args.family,'round':number,'dataset':'nips_task34','fold':0,'seed':args.seed,
                  'alpha':alpha,'rank_weight':rank,'baseline_reused':True,'baseline_predictions_sha256':digest(baseline/'window_predictions.tsv'),
                  'design_sha256':digest(root/'docs/QIKT_Mamba_V2_方案.md'),
                  'selection':'Published two-round coefficients; std/range + tail/strong/overall preservation; no test reads',
                  'decision_evidence':str(family/'round1/balance.json') if number==2 and not frozen else None}
        if args.family=='B1':
            good=number==2 and previous is not None and previous['exploratory_candidate_pass']
            protocol.update(frequency_alpha=.25 if number==1 else .5 if good else .1,
                            frequency_cap=2. if number==1 or good else 1.5,frequency_power=.25)
        if args.family=='A3':protocol.update(tilt_mix=.25,tilt_temperature=.5 if number==1 else .25,tilt_cap=2.,tilt_estimator='epoch-lagged complete-UID risk; bounded TERM-inspired approximation',round2_rule='Predeclared weaker-temperature ablation; Group DRO deferred pending supported training groups/OOF evidence')
        if frozen:
            protocol.update({k:frozen[k] for k in ['alpha','rank_weight','frequency_alpha','frequency_cap','frequency_power',
                                                   'tilt_mix','tilt_temperature','tilt_cap','tilt_estimator'] if k in frozen})
            protocol.update(fixed_coefficients_from=str(args.fixed_protocol),fixed_protocol_sha256=digest(args.fixed_protocol),selection='Fixed-coefficient seed replication; no parameter adaptation')
        protocol_path=round_dir/'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Round protocol changed')
        atomic_json(protocol_path,protocol)
        result_path=trial/'round_summary.json'
        exit_path=trial/'training_exit.json'
        if not exit_path.exists() or json.loads(exit_path.read_text())['returncode']!=0:
            wait_for_capacity(round_dir/'training_admission.json')
            command=[sys.executable,str(script/'run_neural.py'),'--variant','baseline','--seed',str(args.seed),'--output-root',str(round_dir),
                     '--resumable','--resume','--objective-protocol',str(protocol_path)]
            with (round_dir/'training_launcher.log').open('a') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(r.returncode)
        if not result_path.exists():wait_for_capacity(round_dir/'validation_admission.json')
        result=json.loads(result_path.read_text()) if result_path.exists() else evaluate_finished('baseline',round_dir)
        output=trial/'validation_window';target=output/'validation_window_results.json'
        if not target.exists():
            wait_for_capacity(round_dir/'window_admission.json')
            command=[sys.executable,str(script/'evaluate_window.py'),'--checkpoint-dir',result['checkpoint_dir'],'--output-dir',str(output),'--source','validation',
                     '--device','cuda','--batch-size','32','--checkpoint-batches','64','--resume']
            with (round_dir/'window_launcher.log').open('a') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(r.returncode)
        comparison_path=round_dir/'balance.json'
        if comparison_path.exists():previous=json.loads(comparison_path.read_text())
        else:
            candidate=pd.read_csv(output/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
            comparison,paired=compare(base_frame,candidate,500,metric_key='validation_window_auc')
            previous=assess_balance(paired,comparison['pooled_delta']);previous.update(protocol=protocol,training=result,validation_window=json.loads(target.read_text()),comparison=comparison)
            paired.to_csv(round_dir/'paired_orirow.csv');atomic_json(comparison_path,previous)
        d=previous['deltas']
        text=[f'# V2 {args.family} round {number}','',f"nips_task34/fold0/seed{args.seed}; completed {result['completed_epochs']} epochs, selected {result['selected_epoch']}; alpha={alpha}, rank={rank}.",
              f"Validation only: std delta {d['std_ddof0']:+.8f}, range delta {d['range']:+.8f}, pooled delta {d['pooled_auc']:+.8f}.",
              f"Macro delta {d['macro_auc']:+.8f}, fixed weak10 delta {d['fixed_baseline_bottom10_mean']:+.8f}, strong25 delta {d['fixed_baseline_top25_mean']:+.8f}.",
              f"Worsened {previous['worsened']}/{previous['N']}; worst delta {previous['worst_delta']:+.8f}. Strict goal pass {previous['strict_goal_pass']}.",
              'Frozen round protocol: '+json.dumps(protocol,ensure_ascii=False),
              'Checks: '+json.dumps(previous['checks']),
              'Next: '+('Use the published stronger coefficient only if all exploratory preservation checks passed; otherwise weaken it.' if number==1 else 'Family two-round review required; no automatic test inference or Peiyou training.')]
        (round_dir/'review.md').write_text('\n'.join(text)+'\n')
    atomic_json(family/'family_summary.json',{'family':args.family,'rounds':[json.loads((family/f'round{i}/balance.json').read_text()) for i in rounds],'objective_achieved':False,'test_evaluated':False})


if __name__=='__main__':main()
