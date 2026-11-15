"""One preregistered seed replication: resumable training and held-out validation.

Parameters and methods stay fixed; this entrypoint never reads model test scores.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from advance_experiment import evaluate_finished
from compare_validation import compare
from prediction_progress import atomic_json
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--neural-root',type=Path,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--variant',choices=['baseline','s2_history_drop005','s7_overlap64','s5_pairwise_005'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];directory=args.neural_root/args.variant;directory.mkdir(parents=True,exist_ok=True)
    if args.variant!='baseline' and not (args.neural_root/'baseline/round_summary.json').exists():raise ValueError('Same-seed baseline prerequisite missing')
    exit_path=directory/'training_exit.json'
    if not exit_path.exists() or json.loads(exit_path.read_text())['returncode']!=0:
        command=[sys.executable,str(Path(__file__).with_name('run_neural.py')),'--dataset','nips_task34','--fold','0',
                 '--seed',str(args.seed),'--variant',args.variant,'--output-root',str(args.neural_root),'--resumable','--resume']
        with (directory/'replication_training_launcher.log').open('a') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    summary_path=directory/'round_summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else evaluate_finished(args.variant,args.neural_root)
    if summary['seed']!=args.seed:raise ValueError('Seed identity mismatch')
    output=directory/'validation_window';target=output/'validation_window_results.json'
    if not target.exists():
        command=[sys.executable,str(Path(__file__).with_name('evaluate_window.py')),'--checkpoint-dir',summary['checkpoint_dir'],
                 '--output-dir',str(output),'--source','validation','--device','cuda','--batch-size','32','--checkpoint-batches','64','--resume']
        with (directory/'replication_window_launcher.log').open('a') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
    result=json.loads(target.read_text())
    final={**summary,'validation_window':result,'test_evaluated':False}
    if args.variant!='baseline':
        a=pd.read_csv(args.neural_root/'baseline/validation_window/window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
        b=pd.read_csv(output/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
        comparison,paired=compare(a,b,500,metric_key='validation_window_auc')
        final['window_comparison']=comparison;paired.to_csv(output/'paired_orirow.csv',index=False)
    atomic_json(directory/'replication_result.json',final)


if __name__=='__main__':main()
