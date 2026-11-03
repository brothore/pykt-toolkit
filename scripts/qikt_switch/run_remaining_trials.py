"""Serial initial mechanism trials with validation-only strength adaptation.
No test inference, no automatic model promotion, and no concurrent GPU jobs.
"""
import argparse,json,os,subprocess,sys,time,datetime,shutil
from pathlib import Path
from advance_experiment import BASE,ROOT,evaluate_finished
from audit_inputs import sha


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,path)


def read(path):
    for attempt in range(10):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:time.sleep(.2)
    raise RuntimeError(f'Incomplete JSON: {path}')


def alive(pid,needle):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and needle in p.read_text().replace('\x00',' ')


def verify_summary(name):
    directory=BASE/'neural'/name;summary=read(directory/'round_summary.json')
    if read(directory/'training_exit.json')['returncode']!=0:raise RuntimeError(f'{name} has a failed terminal result')
    exported=read(directory/'validation/validation_results.json')
    if sha(Path(exported['checkpoint']))!=exported['checkpoint_sha256']:raise RuntimeError(f'{name} checkpoint changed')
    if summary['variant']!=name:raise RuntimeError('Variant identity mismatch')
    return summary


def analyze(name,result,next_action):
    if name=='baseline':return
    c=result['comparison'];p=c['primary_paired'];a=c['baseline_cohorts']['primary_20_5'];b=c['candidate_cohorts']['primary_20_5'];ci=c['bootstrap']['pooled_delta_ci95']
    supported=c['pooled_delta']>0 and ci[0]>0
    lines=[f'# {name}: validation round analysis','',
      f"nips_task34, qikt_mamba/public_mamba, fold0, seed42, completed {result['completed_epochs']} epochs, selected epoch {result['selected_epoch']}.",
      f"Validation AUC {result['validation_auc']:.10f}; pooled delta {c['pooled_delta']:+.10f}, orirow-bootstrap CI95 [{ci[0]:+.10f}, {ci[1]:+.10f}]. Positive pooled effect supported by this resampling: {supported} (not independent training-seed evidence).",'',
      f"Primary orirows: {p['improved']} improve, {p['worsened']} worsen, {p['unchanged']} tie. Mean delta {p['mean_delta']:+.8f}; worst delta {p['min_delta']:+.8f}.",
      f"std {a['std']:.8f} -> {b['std']:.8f}; range {a['range']:.8f} -> {b['range']:.8f}. Strict primary no-regression pass: {c['strict_primary_no_regression_pass']}.",
      'Do not infer sequence protection from pooled AUC alone. A smaller std/range must be checked against the individual regressions and baseline-performance strata.','']
    for group,g in c['baseline_performance_groups'].items():lines.append(f"{group}: N={g['N']}, macro delta={g['mean_delta']:+.8f}, worsened={g['worsened']}.")
    lines+=['',f'Next action / refinement: {next_action}','',
      'Test data were not evaluated. windowauclate_mean is not supplied by these validation metrics. Overall objective is unfinished; this round is evidence, not automatic promotion.']
    (BASE/'neural'/name/'analysis.md').write_text('\n'.join(lines)+'\n')


def run_trial(name,status):
    d=BASE/'neural'/name;write(status,{'stage':'training_or_attaching','variant':name,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if (d/'round_summary.json').exists():return verify_summary(name)
    if (d/'training_started.json').exists():
        # Attach to authoritative existing state; never restart from elapsed time.
        return evaluate_finished(name)
    if shutil.disk_usage(ROOT).free<300*1024**2:raise RuntimeError('Insufficient free disk for another checkpoint')
    d.mkdir(exist_ok=True)
    with open(d/'launcher.log','x') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/qikt_switch/run_neural.py'),'--variant',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(status,{'stage':'validation_export','variant':name})
    return evaluate_finished(name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    marker=BASE/'remaining_trials_controller.json';status=BASE/'remaining_trials_status.json'
    if marker.exists():
        old=read(marker)
        if alive(old['pid'],'run_remaining_trials.py'):raise RuntimeError('Existing controller is live')
        if not args.resume:raise RuntimeError('Existing controller record: inspect it, then use --resume to attach/continue safely')
    write(marker,{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'protocol':'Wait S2 first round; use weaker response masking if pooled AUC or high-baseline macro worsens; otherwise test stronger masking. Then independently test S3, S4, small ranking; use weaker ranking if pooled AUC worsens, otherwise larger ranking. Finally bounded fusion at .1 then .05, and overlap64. Fixed nips fold0 seed42, original max200 epochs/earlystop, one GPU job at a time. Test data never read.'})
    try:
        write(status,{'stage':'waiting_existing_s2_round1'})
        parent=BASE/'neural/s1_item_residual/advance_controller.json'
        while not (BASE/'neural/s2_history_drop005/round_summary.json').exists():
            terminal=read(BASE/'neural/s1_item_residual/advance_status.json')
            if terminal['stage']=='failed':raise RuntimeError(f'Parent failed: {terminal}')
            if not alive(read(parent)['pid'],'advance_experiment.py'):raise RuntimeError('Parent handle missing; inspect existing training instead of restarting')
            time.sleep(10)
        first=verify_summary('s2_history_drop005');comp=first['comparison']
        weaker=comp['pooled_delta']<=0 or comp['baseline_performance_groups']['baseline_high']['mean_delta']<0
        follow='s2_history_drop001' if weaker else 's2_history_drop010'
        analyze('s2_history_drop005',first,f"Use {follow}: {'reduce historical masking to preserve continuity gains' if weaker else 'test whether stronger masking increases the supported input regularization effect'}; decision uses validation only.")
        second=run_trial(follow,status);analyze(follow,second,'Compare the two masking strengths and baseline before adopting either; continue with branch-sharing intervention S3.')
        for name,action in [
          ('s3_separate_branches','If gains reflect extra capacity, add a shared-backbone/adapter control; if regression dominates, retain parameter sharing. Next test explicit KC memory.'),
          ('s4_concept_memory','If memory helps weak rows but hurts strong rows, refine with a prefix-reliability gate or baseline distillation rather than an unconditional larger correction. Next test the ranking objective.')]:
            result=run_trial(name,status);analyze(name,result,action)
        rank=run_trial('s5_pairwise_001',status);follow_rank='s5_pairwise_0002' if rank['comparison']['pooled_delta']<0 else 's5_pairwise_005'
        analyze('s5_pairwise_001',rank,f'Test {follow_rank}: reduce ranking weight if pooled AUC fell; otherwise test a stronger within-fragment ranking signal. Preserve token BCE.')
        result=run_trial(follow_rank,status);analyze(follow_rank,result,'Retain only validation-supported objective changes; if per-row harm persists, try teacher-constrained residual learning in a later round. Next test fusion.')
        result=run_trial('s6_head_fusion_010',status);analyze('s6_head_fusion_010',result,'Halve the bounded fusion scale to .05 to test whether protecting baseline weights reduces regressions.')
        result=run_trial('s6_head_fusion_005',status);analyze('s6_head_fusion_005',result,'Compare the scale tradeoff using pooled, macro and individual deltas; do not use test labels. Next test overlap context.')
        result=run_trial('s7_overlap64',status);analyze('s7_overlap64',result,'The target set is unchanged but optimizer-step/context cost differs; require a compute-budget control if this is promising. Freeze/confirm candidates only after main-agent review of all rounds.')
        write(status,{'stage':'initial_mechanism_trials_complete','test_evaluated':False,'objective_achieved':False,'next':'main-agent review, justified refinements, frozen nips window test and peiyou confirmation'})
    except BaseException as e:
        write(status,{'stage':'failed','error':repr(e),'automatic_restart_performed':False});raise

if __name__=='__main__':main()
