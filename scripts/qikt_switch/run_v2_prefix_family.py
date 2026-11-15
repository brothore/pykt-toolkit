"""B2 two-round frozen-teacher prefix adapter in an isolated source runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from advance_experiment import evaluate_finished
from compare_validation import compare
from prediction_progress import atomic_json,digest
from resource_admission import wait_for_capacity
from sequence_balance_metrics import assess_balance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-root',type=Path,required=True);ap.add_argument('--family',choices=['B2','C1'],default='B2');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];script=Path(__file__).resolve().parent;family=args.work_root/args.family;family.mkdir(exist_ok=True)
    previous=None
    for number in [1,2]:
        directory=family/f'round{number}';directory.mkdir(exist_ok=True)
        seed=43 if args.family=='C1' and number==2 and previous['exploratory_candidate_pass'] else 42
        baseline=root/'logs/qikt_switch_20261103/neural/baseline' if seed==42 else root/f'logs/qikt_switch_20261103/seed_replication/neural_seed{seed}/baseline'
        base_frame=pd.read_csv(baseline/'validation_window/window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
        reference=directory/'baseline'
        if not reference.exists():reference.symlink_to(baseline.resolve(),target_is_directory=True)
        projection=bool(args.family=='B2' and number==2 and previous['checks']['strong_preserved'] and not previous['checks']['fixed_weak_preserved'])
        variant='v2_prefix_fixed' if number==1 else 'v2_prefix_projected' if projection else 'v2_prefix_anchored'
        kl=0. if number==1 else 1.
        if args.family=='C1':
            kl=1. if number==1 or seed==43 else 2. if not previous['checks']['strong_preserved'] else .1
            variant={1.:'v2_protected_kl1',2.:'v2_protected_kl2',.1:'v2_protected_kl01'}[kl]
        protocol={'version':'V2','family':args.family,'round':number,'dataset':'nips_task34','fold':0,'seed':seed,'alpha':0.,'rank_weight':0.,
            'adapter_scale':.05,'adapter_hidden':16,'projection_trainable':projection,'adapter_kl':kl,'learning_rate':.0001,
            'context':'Current question frozen embedding and current-KC posterior/support/event recency/coverage strictly before current response. Memory resets at each supplied window.',
            'baseline_reused':str(baseline.resolve()),'baseline_predictions_sha256':digest(baseline/'validation_window/window_predictions.tsv'),
            'design_sha256':digest(root/'docs/QIKT_Mamba_V2_方案.md'),'isolated_runtime':str(root),
            'decision_rule':'R2 opens only the small projection if strong-group mean is preserved but weak-group mean failed to improve; otherwise keeps it frozen. R2 adds KL=1; no test reads.'}
        if args.family=='C1':protocol.update(pair_weight=1.,teacher_probability_margin=.1,teacher_provenance='Frozen in-sample training baseline, not OOF. Confidence/order constraints are not generalization guarantees.',decision_rule='R1 KL1+pair1. If all exploratory checks pass, R2 repeats fixed coefficients at seed43. Otherwise increase KL to2 if strong group declined, else reduce KL to.1; pair coefficient remains1.')
        pp=directory/'protocol.json'
        if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('B2 protocol changed')
        atomic_json(pp,protocol);trial=directory/variant;trial.mkdir(exist_ok=True);exit_path=trial/'training_exit.json'
        if not exit_path.exists() or json.loads(exit_path.read_text())['returncode']!=0:
            wait_for_capacity(directory/'training_admission.json')
            command=[sys.executable,str(script/'run_neural.py'),'--variant',variant,'--seed',str(seed),'--output-root',str(directory),'--resumable','--resume','--objective-protocol',str(pp)]
            with (directory/'training_launcher.log').open('a') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(r.returncode)
        summary_path=trial/'round_summary.json'
        if not summary_path.exists():wait_for_capacity(directory/'validation_admission.json')
        result=json.loads(summary_path.read_text()) if summary_path.exists() else evaluate_finished(variant,directory)
        output=trial/'validation_window';target=output/'validation_window_results.json'
        if not target.exists():
            wait_for_capacity(directory/'window_admission.json')
            command=[sys.executable,str(script/'evaluate_window.py'),'--checkpoint-dir',result['checkpoint_dir'],'--output-dir',str(output),'--source','validation','--device','cuda','--batch-size','32','--checkpoint-batches','64','--resume']
            with (directory/'window_launcher.log').open('a') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:raise SystemExit(r.returncode)
        balance=directory/'balance.json'
        if balance.exists():previous=json.loads(balance.read_text())
        else:
            candidate=pd.read_csv(output/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
            comparison,paired=compare(base_frame,candidate,500,metric_key='validation_window_auc');previous=assess_balance(paired,comparison['pooled_delta'])
            previous.update(protocol=protocol,training=result,validation_window=json.loads(target.read_text()),comparison=comparison);paired.to_csv(directory/'paired_orirow.csv');atomic_json(balance,previous)
        (directory/'review.md').write_text(f"# {args.family} round {number}, seed {seed}\n\nCompleted {result['completed_epochs']} epochs, selected {result['selected_epoch']}.\n\n{json.dumps(previous['deltas'],indent=2)}\n\nChecks: {json.dumps(previous['checks'])}\nWorsened {previous['worsened']}/{previous['N']}. Strict goal pass {previous['strict_goal_pass']}.\n\nNo test evaluated; small/frozen adapter does not guarantee ranking protection.\n")
    atomic_json(family/'family_summary.json',{'family':args.family,'rounds':[json.loads((family/f'round{i}/balance.json').read_text()) for i in [1,2]],'objective_achieved':False,'test_evaluated':False})


if __name__=='__main__':main()
