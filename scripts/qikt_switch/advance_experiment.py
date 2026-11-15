"""Wait for one live training job, validate it, then optionally run one next trial.
Always serial on the GPU. No test inference or automatic hyperparameter search.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'logs/qikt_switch_20261103'


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def wait_finished(variant,neural_root=None):
    directory=(Path(neural_root) if neural_root else BASE/'neural')/variant
    marker=json.loads((directory/'training_started.json').read_text())
    while not (directory/'training_exit.json').exists():
        pid=marker['pid'];process=Path(f'/proc/{pid}/cmdline')
        if not process.exists() or 'run_neural.py' not in process.read_text().replace('\x00',' '):
            raise RuntimeError('Launcher handle missing without terminal result: inspect children/logs; no restart performed')
        time.sleep(10)
    status=json.loads((directory/'training_exit.json').read_text())
    if status['returncode']!=0:raise RuntimeError(f'{variant} training failed: {status}')
    return directory



def describe_saved_architecture(config):
    version=config.get('model_config',{}).get('version',config.get('params',{}).get('version','unknown'))
    return {'model':'qikt_mamba/'+str(version),'version':version,'architecture':'two independent GRUs' if version in ['gru','lstm'] else 'See saved version/source'}


def evaluate_finished(variant,neural_root=None):
    neural_root=Path(neural_root) if neural_root else BASE/'neural'
    directory=wait_finished(variant,neural_root)
    checkpoints=[p.parent for p in (directory/'saved_model').rglob('config.json') if (p.parent/'attn_model.ckpt').is_file()]
    if len(checkpoints)!=1:raise RuntimeError(f'Expected one selected checkpoint, found {checkpoints}')
    checkpoint=checkpoints[0];text=(directory/'training.log').read_text()
    epochs=list(re.finditer(r'Epoch: (\d+), validauc: ([\d.]+).*?best epoch: (\d+), best auc: ([\d.]+)',text))
    if not epochs:raise RuntimeError('No reliable completed epoch evidence')
    last=epochs[-1];completed=int(last.group(1));best=int(last.group(3));native_auc=float(last.group(4))
    dump(checkpoint/'training_stats.json',{'actual_epoch':best,'best_epoch':best,'completed_epochs':completed,
        'source':str(directory/'training.log'),'native_selected_validation_auc_rounded':native_auc})
    out=directory/'validation'
    if not (out/'validation_results.json').exists():
        with open(directory/'validation_export.log','a') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/qikt_switch/evaluate_validation.py'),
                '--checkpoint-dir',str(checkpoint),'--output-dir',str(out),'--batch-size','32','--device','cuda'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'validation_results.json').read_text())
    if abs(result['validation_auc']-native_auc)>1.1e-4:
        raise RuntimeError(f'Export AUC {result["validation_auc"]} disagrees with native selected AUC {native_auc}')
    config=json.loads((checkpoint/'config.json').read_text());params=config['params']
    summary={'variant':variant,'dataset':params['dataset_name'],**describe_saved_architecture(config),'fold':params['fold'],'seed':params['seed'],
        'completed_epochs':completed,'selected_epoch':best,'validation_auc':result['validation_auc'],'cohorts':result['cohorts'],
        'checkpoint_dir':str(checkpoint),'formal_test_windowauclate_mean':None,'test_evaluated':False}
    if variant!='baseline':
        comparison=directory/'comparison'
        with open(directory/'comparison.log','w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/qikt_switch/compare_validation.py'),
                '--baseline',str(neural_root/'baseline/validation/validation_predictions.tsv'),
                '--candidate',str(out/'validation_predictions.tsv'),'--output-dir',str(comparison)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary['comparison']=json.loads((comparison/'comparison.json').read_text())
    dump(directory/'round_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--current',required=True);ap.add_argument('--next-variant');args=ap.parse_args()
    from run_neural import VARIANTS
    if args.current not in VARIANTS or (args.next_variant and args.next_variant not in VARIANTS):raise ValueError('Unknown variant')
    marker=BASE/'neural'/args.current/'advance_controller.json'
    with open(marker,'x') as f:json.dump({'pid':os.getpid(),'current':args.current,'next':args.next_variant,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
    status=BASE/'neural'/args.current/'advance_status.json'
    try:
        dump(status,{'stage':'waiting_current','current':args.current})
        evaluate_finished(args.current)
        if args.next_variant:
            dump(status,{'stage':'training_next','current':args.current,'next':args.next_variant})
            nextdir=BASE/'neural'/args.next_variant;nextdir.mkdir(exist_ok=True)
            with open(nextdir/'launcher.log','x') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/qikt_switch/run_neural.py'),'--variant',args.next_variant],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            dump(status,{'stage':'evaluating_next','next':args.next_variant})
            evaluate_finished(args.next_variant)
        dump(status,{'stage':'complete','scope':'requested validation stages only; overall objective remains unfinished'})
    except BaseException as e:
        dump(status,{'stage':'failed','error':repr(e),'automatic_restart_performed':False});raise

if __name__=='__main__':main()
