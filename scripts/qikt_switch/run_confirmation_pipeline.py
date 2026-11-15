"""Bounded continuation: R2 review, frozen-head R3, frozen tests and Peiyou transfer.
No test-driven coefficient updates or automatic claim that the goal is solved.
"""
import argparse,json,os,subprocess,sys,time,datetime,shutil
from pathlib import Path
import pandas as pd
from run_remaining_trials import ROOT,BASE,read,write,alive
from advance_experiment import evaluate_finished
from compare_validation import compare
from audit_inputs import sha
from transfer_gate import assess_transfer

WORK=BASE/'confirmation_pipeline'
STATUS=WORK/'status.json'


def logstage(stage,**fields):write(STATUS,{'stage':stage,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields})


def run_command(command,log):
    with open(log,'x') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def run_or_attach_inference(command, target, log):
    matches=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            # Only Python can own this entrypoint. Avoid reading unrelated
            # process argv: procfs cmdline can block on another process's mm.
            if not (proc/'comm').read_text().strip().startswith('python'):continue
            args=[os.fsdecode(v) for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        except OSError:continue
        if args[1:]==command[1:]:matches.append(int(proc.name))
    if len(matches)>1:raise RuntimeError('Duplicate inference processes; inspect before proceeding')
    if matches:
        pid=matches[0]
        write(target.parent/'inference_attachment.json',{'pid':pid,'command':command,'reason':'Preserve live inference during controller handoff'})
        while alive(pid,'evaluate_window.py'):time.sleep(5)
        if not target.exists():raise RuntimeError('Attached inference exited without results; no automatic restart')
    elif not target.exists():run_command(command,log)


def run_model(name,dataset='nips_task34'):
    neural=BASE/('neural' if dataset=='nips_task34' else 'neural_peiyou_fold0');directory=neural/name
    logstage('training_or_attaching',dataset=dataset,variant=name)
    if (directory/'round_summary.json').exists():
        if read(directory/'training_exit.json')['returncode']!=0:raise RuntimeError('Failed training result')
        result=read(directory/'round_summary.json');exported=read(directory/'validation/validation_results.json')
        if sha(Path(exported['checkpoint']))!=exported['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        return result
    if not (directory/'training_started.json').exists():
        directory.mkdir(parents=True,exist_ok=True)
        run_command([sys.executable,str(ROOT/'scripts/qikt_switch/run_neural.py'),'--dataset',dataset,'--fold','0','--variant',name],directory/'confirmation_launcher.log')
    return evaluate_finished(name,neural_root=neural)


def clear_private_overlap_cache(dataset):
    removed=[]
    for folder in BASE.glob(f'overlap_{dataset}_fold0_*'):
        for p in folder.glob('*.pkl'):
            removed.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
    if removed:
        p=WORK/'cache_cleanup.json';old=read(p) if p.exists() else [];write(p,old+removed)


def predict_candidate(candidate,neural,out,source):
    out.mkdir(parents=True,exist_ok=True);components=[]
    for name,weight in sorted(candidate['components'].items()):
        if not weight:continue
        selected=read(neural/name/'round_summary.json');checkpoint=Path(selected['checkpoint_dir'])
        components.append({'variant':name,'weight':weight,'checkpoint_dir':str(checkpoint),'checkpoint_sha256':sha(checkpoint/'attn_model.ckpt')})
    primary=components[0]['checkpoint_dir'];mixture=len(components)>1
    spec=out/'mixture_spec.json'
    if mixture:write(spec,{'components':components,'selection':'frozen global coefficients from nips fold0 validation only; no per-UID rules or test-label fitting'})
    target=out/('all_results.json' if source=='test' else 'validation_window_results.json')
    if not target.exists():
        command=[sys.executable,str(ROOT/'scripts/qikt_switch/evaluate_window.py'),'--checkpoint-dir',primary,'--output-dir',str(out),'--source',source,'--device','cuda','--batch-size','32']
        if mixture:command+=['--mixture-json',str(spec)]
        run_or_attach_inference(command,target,out/'inference.log')
    result=read(target)
    if result['source']!=source or result['test_evaluated']!=(source=='test'):raise RuntimeError('Evaluation source mismatch')
    if mixture:
        if result['mixture_spec_sha256']!=sha(spec):raise RuntimeError('Frozen mixture specification changed')
    elif result['checkpoint_sha256']!=components[0]['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    return result


def compare_files(base_dir,candidate_dir,metric_key):
    a=pd.read_csv(base_dir/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str});b=pd.read_csv(candidate_dir/'window_predictions.tsv',sep='\t',dtype={'questions':str,'concepts':str})
    result,rows=compare(a,b,500,metric_key=metric_key);write(candidate_dir/'comparison.json',result);rows.to_csv(candidate_dir/'paired_orirow.csv');return result


def evaluate_new_windows(name):
    neural=BASE/'neural';selected=read(neural/name/'round_summary.json');directory=neural/name/'validation_window'
    result=predict_candidate({'components':{name:1.}},neural,directory,'validation')
    comparison=compare_files(neural/'baseline/validation_window',directory,'validation_window_auc')
    if abs(comparison['candidate_validation_window_auc']-result['validation_window_auc'])>1e-12:raise RuntimeError('Window AUC recomputation mismatch')
    return {'variant':name,'completed_epochs':selected['completed_epochs'],'selected_epoch':selected['selected_epoch'],'validation_window_auc':result['validation_window_auc'],'cohorts':result['cohorts'],'comparison':comparison,'test_evaluated':False}


def freeze_candidates():
    target=WORK/'frozen_nips_selection.json'
    if target.exists():return read(target)
    native=read(BASE/'validation_window_summaries.json');base=next(r for r in native if r['variant']=='baseline');b=base['cohorts']['primary_20_5'];candidates=[]
    for r in native:
        if r['variant']=='baseline':continue
        c=r['comparison'];p=c['primary_paired'];v=r['cohorts']['primary_20_5']
        candidates.append({'id':r['variant'],'components':{r['variant']:1.},'validation_window_auc':r['validation_window_auc'],'pooled_delta':c['pooled_delta'],
            'primary_std':v['std'],'primary_range':v['range'],'primary_worsened':p['worsened'],'strict_primary_no_regression_pass':c['strict_primary_no_regression_pass'],
            'baseline_low_mean_delta':c['baseline_performance_groups']['baseline_low']['mean_delta'],'baseline_high_mean_delta':c['baseline_performance_groups']['baseline_high']['mean_delta']})
    for stage in ['r2','r3']:
        for r in read(BASE/f'refinement_{stage}/mixtures/results.json'):
            candidates.append({**r,'id':f'mixture_{stage}_{r["id"]}'})
    eligible=[c for c in candidates if c['pooled_delta']>0 and c['baseline_low_mean_delta']>=0 and c['baseline_high_mean_delta']>=0]
    if not eligible:raise RuntimeError('No validation-supported candidate even at group level; main-agent refinement required')
    first=max(eligible,key=lambda c:c['pooled_delta'])
    joint=[c for c in eligible if c['pooled_delta']>=.0001 and c['primary_std']<=b['std'] and c['primary_range']<=b['range'] and c['components']!=first['components']]
    selected=[first]
    if joint:selected.append(max(joint,key=lambda c:c['pooled_delta']))
    for c in selected:
        c['frozen_components']={name:{'checkpoint_sha256':sha(Path(read(BASE/f'neural/{name}/round_summary.json')['checkpoint_dir'])/'attn_model.ckpt'),'weight':weight} for name,weight in c['components'].items() if weight}
    result={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':'nips_task34','fold':0,'seed':42,
        'criteria':'One maximum window-validation AUC candidate with nonnegative fixed low/high baseline-group mean deltas; optionally one distinct candidate with gain>=.0001 and nonincreasing primary std/range. Selected for evaluation, not automatic acceptance.',
        'selection_source':'validation only, before any test-model performance inference','candidates':selected,'test_labels_used_for_selection':False,'objective_achieved':False,
        'strict_constraint_note':'Individual no-regression flags are retained; group-level screening does not prove every sequence is protected.'}
    write(target,result);return result


def confirm_dataset(dataset,selection):
    neural=BASE/('neural' if dataset=='nips_task34' else 'neural_peiyou_fold0');folder=WORK/dataset;folder.mkdir(exist_ok=True)
    base_dir=folder/'baseline';logstage('test_reference_inference',dataset=dataset)
    baseline=predict_candidate({'components':{'baseline':1.}},neural,base_dir,'test')
    outcomes=[]
    for candidate in selection['candidates']:
        logstage('frozen_test_candidate_inference',dataset=dataset,candidate=candidate['id']);out=folder/candidate['id']
        result=predict_candidate(candidate,neural,out,'test');comparison=compare_files(base_dir,out,'windowauclate_mean')
        if abs(comparison['candidate_windowauclate_mean']-result['windowauclate_mean'])>1e-12 or abs(comparison['baseline_windowauclate_mean']-baseline['windowauclate_mean'])>1e-12:raise RuntimeError('all_results.json disagrees with independent mean-question AUC')
        outcomes.append({'candidate':candidate['id'],'all_results':str(out/'all_results.json'),'result':result,'comparison':comparison})
        write(folder/'test_summary.json',{'dataset':dataset,'baseline':baseline,'candidates':outcomes,'objective_achieved':False,'main_agent_review_required':True})
    return outcomes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();WORK.mkdir(exist_ok=True)
    marker=WORK/'controller.json'
    if marker.exists():
        old=read(marker)
        if alive(old['pid'],'run_confirmation_pipeline.py'):raise RuntimeError('Controller is already live')
        if not args.resume:raise RuntimeError('Prior controller exists; inspect before --resume')
    write(marker,{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Wait R2, fit frozen-head R3, validate then freeze candidate specifications; confirm nips test and transfer fixed methods/weights to Peiyou fold0/seed42. Serial GPU. No test-driven coefficient updates or model adoption.'})
    try:
        logstage('waiting_r2_training')
        status=BASE/'neural/r2_mask005_rank005/advance_status.json';parent=BASE/'neural/r2_mask005_rank005/advance_controller.json'
        while read(status)['stage']!='complete':
            current=read(status)
            if current['stage']=='failed':raise RuntimeError(f'R2 controller failed: {current}')
            if not alive(read(parent)['pid'],'advance_experiment.py'):raise RuntimeError('R2 handle missing without success; inspect child state, no restart')
            time.sleep(10)
        summaries=read(BASE/'validation_window_summaries.json')
        for name in ['r2_mask005_rank005','r2_overlap64_rank005']:
            logstage('r2_validation_window',variant=name);result=evaluate_new_windows(name);summaries=[r for r in summaries if r['variant']!=name]+[result]
            write(BASE/'validation_window_summaries.json',summaries)
        clear_private_overlap_cache('nips_task34')
        run_model('r3_frozen_memory_fusion_anchor1')
        logstage('r3_validation_window');result=evaluate_new_windows('r3_frozen_memory_fusion_anchor1');summaries=[r for r in summaries if r['variant']!=result['variant']]+[result];write(BASE/'validation_window_summaries.json',summaries)
        selection=freeze_candidates();outcomes=confirm_dataset('nips_task34',selection)
        # Fixed methods/coefficients stay frozen; Nips confirmation gates only transfer compute.
        checks=[assess_transfer(o) for o in outcomes]
        write(WORK/'peiyou_transfer_gate.json',{'screening':'Nips formal confirmation, no coefficient refitting','candidates':checks})
        eligible={c['candidate'] for c in checks if c['eligible_for_peiyou']}
        if not eligible:
            logstage('nips_confirmation_no_transfer_candidate',objective_achieved=False,peiyou_started=False,next='Review Nips evidence; do not launch expensive Peiyou training or tune on the observed test set.')
            return
        selection={**selection,'candidates':[c for c in selection['candidates'] if c['id'] in eligible]}
        write(WORK/'peiyou_transfer_selection.json',selection)
        needed={'baseline'}
        for candidate in selection['candidates']:needed.update(name for name,weight in candidate['components'].items() if weight)
        for name in ['baseline']+sorted(needed-{'baseline'}):
            run_model(name,'peiyou');clear_private_overlap_cache('peiyou')
        confirm_dataset('peiyou',selection)
        logstage('research_confirmation_complete',objective_achieved=False,main_agent_review_required=True,next='Inspect both dataset all_results.json, paired rows, std/range and all failed constraints; do not declare solved from aggregate AUC alone.')
    except BaseException as e:
        logstage('failed',error=repr(e),automatic_restart_performed=False);raise

if __name__=='__main__':main()
