"""After the serial initial trials, evaluate native windows on validation only.
Epoch selection/early stopping remain untouched. Test data are never read.
"""
import argparse,json,os,subprocess,sys,time,datetime
from pathlib import Path
import pandas as pd
from run_remaining_trials import BASE,ROOT,read,write,alive,verify_summary
from audit_inputs import sha
from compare_validation import compare


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    marker=BASE/'validation_windows_controller.json';status=BASE/'validation_windows_status.json'
    if marker.exists():
        old=read(marker)
        if alive(old['pid'],'run_validation_windows.py'):raise RuntimeError('Window-validation controller is already live')
        if not args.resume:raise RuntimeError('Prior controller record exists; inspect before --resume')
    write(marker,{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only held-out fold0 full-sequence windows; original training checkpoint selection unchanged; no test data or automatic promotion'})
    try:
        write(status,{'stage':'waiting_initial_training_trials'})
        while True:
            state=read(BASE/'remaining_trials_status.json')
            if state['stage']=='initial_mechanism_trials_complete':break
            if state['stage']=='failed':raise RuntimeError(f'Mechanism trial controller failed: {state}')
            if not alive(read(BASE/'remaining_trials_controller.json')['pid'],'run_remaining_trials.py'):
                raise RuntimeError('Initial controller handle missing without terminal success; no training restart performed')
            time.sleep(10)
        results=[p.parent.name for p in sorted((BASE/'neural').glob('*/round_summary.json'),key=lambda p:p.stat().st_mtime)]
        results=['baseline']+[r for r in results if r!='baseline']
        summaries=[]
        for name in results:
            selected=verify_summary(name);directory=BASE/'neural'/name;out=directory/'validation_window';target=out/'validation_window_results.json'
            write(status,{'stage':'validation_window_inference','variant':name,'finished':len(summaries),'total':len(results)})
            if not target.exists():
                with open(directory/'validation_window.log','x') as log:
                    subprocess.run([sys.executable,str(ROOT/'scripts/qikt_switch/evaluate_window.py'),
                        '--checkpoint-dir',selected['checkpoint_dir'],'--output-dir',str(out),'--source','validation','--device','cuda','--batch-size','32'],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            window=read(target)
            if window['test_evaluated'] or window['source']!='validation':raise RuntimeError('Unexpected evaluation population')
            if window['checkpoint_sha256']!=sha(Path(selected['checkpoint_dir'])/'attn_model.ckpt'):raise RuntimeError('Window checkpoint changed')
            item={'variant':name,'completed_epochs':selected['completed_epochs'],'selected_epoch':selected['selected_epoch'],
                  'validation_window_auc':window['validation_window_auc'],'cohorts':window['cohorts'],'test_evaluated':False}
            if name!='baseline':
                base=BASE/'neural/baseline/validation_window/window_predictions.tsv';candidate=out/'window_predictions.tsv'
                a=pd.read_csv(base,sep='\t',dtype={'questions':str,'concepts':str});b=pd.read_csv(candidate,sep='\t',dtype={'questions':str,'concepts':str})
                comparison,paired=compare(a,b,bootstrap=500,metric_key='validation_window_auc')
                if abs(comparison['candidate_validation_window_auc']-window['validation_window_auc'])>1e-12:raise RuntimeError('Window AUC export disagrees with recomputation')
                write(out/'comparison.json',comparison);paired.to_csv(out/'paired_orirow.csv');item['comparison']=comparison
            summaries.append(item);write(BASE/'validation_window_summaries.json',summaries)
            print(json.dumps(item,ensure_ascii=False),flush=True)
        write(status,{'stage':'validation_window_comparison_complete','variants':len(summaries),'test_evaluated':False,'objective_achieved':False,'next':'main-agent review and justified refinement/frozen-test selection, then peiyou confirmation'})
    except BaseException as e:
        write(status,{'stage':'failed','error':repr(e),'automatic_restart_performed':False});raise

if __name__=='__main__':main()
