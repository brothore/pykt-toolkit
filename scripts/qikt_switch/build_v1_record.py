"""Freeze the completed V1 evidence into a verified, local-only record.

Repeat with --resume after interruption. Existing sealed records are immutable.
Dataset files are referenced by hash rather than copied. No credentials collected.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prediction_progress import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=Path(__file__).resolve().parents[2];base=root/'logs/qikt_switch_20261103';out=args.output.resolve()
    if (out/'SEALED.json').exists():raise ValueError('Sealed V1 record cannot be overwritten')
    if out.exists() and not args.resume:raise ValueError('Unsealed record exists; inspect then --resume')
    out.mkdir(parents=True,exist_ok=True);inventory={}
    def put(relative,obj):
        p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);atomic_json(p,obj)
    def capture(source,relative):
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        before=source.stat();expected=digest(source)
        if not target.exists() or digest(target)!=expected:
            temporary=target.with_suffix(target.suffix+'.partial');shutil.copyfile(source,temporary)
            if digest(temporary)!=expected:raise ValueError(f'Copy mismatch: {source}')
            shutil.copystat(source,temporary);os.replace(temporary,target)
        after=source.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError(f'Source changed during snapshot: {source}')
        inventory[str(relative)]={'origin':str(source),'source_sha256':expected,'source_mtime_ns':before.st_mtime_ns}
    cutoff=datetime.datetime.now(datetime.timezone.utc).isoformat()
    evidence_files=sorted(p for p in base.rglob('*') if p.is_file() and 'records' not in p.relative_to(base).parts)
    # Capture only the source files present before new V2 implementation work.
    source_files=[]
    for folder in ['pykt','examples','scripts','tests']:
        source_files += [p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.sh'} and '__pycache__' not in p.parts]
    source_files += [p for p in (root/'configs').rglob('*') if p.is_file()]
    source_files += [root/p for p in ['AGENTS.md','setup.py','LICENSE','README.md']]
    source_files += list((root/'docs').glob('QIKT*.md'))
    put('metadata/snapshot_start.json',{'version':'V1','cutoff_utc':cutoff,'source_root':str(root),'evidence_root':str(base),'snapshot_type':'Retrospective evidence freeze, not a claim of complete historical telemetry','evidence_file_count':len(evidence_files)})
    for index,p in enumerate(evidence_files):
        capture(p,Path('evidence')/p.relative_to(base))
        if index%100==0:print('Evidence',index,'/',len(evidence_files),flush=True)
    for p in sorted(set(source_files)):capture(p,Path('source')/p.relative_to(root))
    for p in (root/'download').glob('*切换问题*zip'):capture(p,Path('inputs')/p.name)
    git=lambda *a:subprocess.check_output(['git',*a],cwd=root,text=True)
    put('metadata/git.json',{'HEAD':git('rev-parse','HEAD').strip(),'branch':git('branch','--show-current').strip(),'recent_history':git('log','-8','--format=%H %aI %s'),'status':git('status','--short'),'note':'Uncommitted scientific source is frozen under source; Git HEAD alone is insufficient to identify it.'})
    # Recover exact historical source blobs where Git still contains them.
    for revision in ['14c32de','8a15857']:
        for p in sorted(set(source_files)):
            relative=p.relative_to(root)
            result=subprocess.run(['git','show',f'{revision}:{relative}'],cwd=root,capture_output=True)
            if result.returncode==0:
                target=out/'history'/revision/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(result.stdout)
    commands=[]
    def command(name,cmd,env=None):
        result=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True)
        target=out/'audit'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(result.stdout+result.stderr)
        commands.append({'output':str(target.relative_to(out)),'command':cmd,'exit_code':result.returncode})
        return result.returncode
    python='/root/miniconda3/envs/pykt-qikt/bin/python'
    command('pip_freeze.txt',[python,'-m','pip','freeze'])
    command('conda_explicit.txt',['/root/miniconda3/bin/conda','list','-p','/root/miniconda3/envs/pykt-qikt','--explicit'])
    command('pip_check.txt',[python,'-m','pip','check'])
    command('gpu.txt',['nvidia-smi','-q'])
    command('disk.txt',['df','-h','/','/data'])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
    tests=sorted(str(p.relative_to(root)) for p in (root/'tests').glob('test_qikt_switch_*.py'))
    code=command('tests.txt',[python,'-m','pytest','-q',*tests],env)
    if code:raise RuntimeError('V1 verification suite failed; record not sealed')
    for label,sock in [('gpu','/tmp/pykt-qikt-gpu-20261104.sock'),('cpu','/tmp/pykt-qikt-cpu-20261104.sock')]:
        command(f'tsp_{label}.txt',['tsp','-l'],{**os.environ,'TS_SOCKET':sock})
    put('metadata/audit_commands.json',commands)
    data=[]
    for dataset in ['nips_task34','peiyou']:
        for name in ['train_valid_quelevel.csv','train_valid_sequences_quelevel.csv','test_quelevel.csv']:
            p=root/'data'/dataset/name;data.append({'dataset':dataset,'path':str(p),'resolved_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest(p),'copied':False})
    put('metadata/data_references.json',data)
    runs=[]
    for p in sorted((out/'evidence').rglob('round_summary.json')):
        d=json.loads(p.read_text());checkpoint=Path(d['checkpoint_dir'])
        d['record_source']=str(p.relative_to(out));d['config_record']='evidence/'+str((checkpoint/'config.json').relative_to(base))
        d['checkpoint_record']='evidence/'+str((checkpoint/'attn_model.ckpt').relative_to(base))
        d['checkpoint_present']=(out/d['checkpoint_record']).exists()
        runs.append(d)
    evaluations=[]
    for p in sorted((out/'evidence').rglob('all_results.json')):
        d=json.loads(p.read_text());d['record_source']=str(p.relative_to(out));evaluations.append(d)
    put('metadata/training_runs.json',runs);put('metadata/evaluations.json',evaluations)
    tasks=[]
    for p in sorted((out/'evidence').rglob('queue_submission.json')):
        d=json.loads(p.read_text());d['record_source']=str(p.relative_to(out));tasks.append(d)
    put('metadata/queue_tasks.json',tasks)
    # Audit whether each previously recorded source hash has any retrievable blob.
    known={}
    for folder in ['source','history']:
        for p in (out/folder).rglob('*'):
            if p.is_file():known.setdefault(digest(p),[]).append(str(p.relative_to(out)))
    for p in (out/'evidence').rglob('*.py'):known.setdefault(digest(p),[]).append(str(p.relative_to(out)))
    coverage=[]
    for p in sorted((out/'evidence').rglob('trial_source_manifest.json')):
        manifest=json.loads(p.read_text());missing={name:h for name,h in manifest.items() if h not in known}
        coverage.append({'manifest':str(p.relative_to(out)),'recorded_files':len(manifest),'unrecovered_source_blobs':missing})
    put('metadata/historical_source_coverage.json',coverage)
    gaps=[{'item':'Early optimizer/RNG/step checkpoints','status':'historically_missing','detail':'Initial nips seed42 and Peiyou runs saved selected model checkpoints, not full optimizer/RNG resume states. Eight seed43/44 runs have full epoch checkpoints.'},
          {'item':'Per-run historical environment and resource curves','status':'partial','detail':'Installation locks/CUDA checks and later monitors exist. Current pip/conda/hardware snapshots are labelled at V1 capture time; early complete CPU/GPU/IO histories were not recorded.'},
          {'item':'Exact intermediate source revisions','status':'partial','detail':'See historical_source_coverage.json. A source hash without a matching preserved blob is not sufficient for exact reconstruction.'},
          {'item':'Original handoff actual training epochs','status':'unknown','detail':'Historical predictions/configs are preserved, but configured max epochs do not establish actual completed epochs.'},
          {'item':'Early exact start/end times and optimizer steps','status':'partial','detail':'Native text logs contain some timestamps and durations; no values are invented for missing fields. Later queue states and resumable training_progress provide explicit values.'},
          {'item':'Historical narrative/runtime status freshness','status':'known_stale','detail':'Early progress docs/runtime status describe earlier phases. Retained as historical evidence; V1 README, training index and final result files are authoritative for this cutoff.'},
          {'item':'Unobserved test confirmation for seeds43/44','status':'not_performed_by_design','detail':'These are held-out validation replications, not missing formal test metrics.'},
          {'item':'Scientific goal','status':'not_achieved','detail':'No stable joint std/range improvement with protection of every sequence established. Snapshot completeness is not experimental success.'}]
    put('metadata/gaps.json',gaps)
    tables=out/'tables';tables.mkdir(exist_ok=True)
    for filename,records,fields in [('training_runs.csv',runs,['dataset','model','fold','seed','variant','completed_epochs','selected_epoch','validation_auc','record_source']),('evaluations.csv',evaluations,['dataset','model','fold','seed','completed_epochs','selected_epoch','windowauclate_mean','record_source'])]:
        with (tables/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    summary={'version':'V1','cutoff_utc':cutoff,'training_runs':len(runs),'formal_evaluations':sum(d.get('test_evaluated') is True for d in evaluations),'all_result_records_including_historical_reconstructions':len(evaluations),'queue_task_records':len(tasks),'historical_source_manifests':len(coverage),'manifests_with_unrecovered_blobs':sum(bool(r['unrecovered_source_blobs']) for r in coverage),'historical_metadata_fully_complete':False,'objective_achieved':False}
    put('metadata/summary.json',summary)
    narrative=root/'logs/qikt_switch_20261103/records/V1_narrative.md'
    if not narrative.exists():raise ValueError('V1 narrative must be authored before snapshot')
    shutil.copyfile(narrative,out/'README.md')
    # Seal every copied/generated file with a manifest; verify the destination bytes.
    entries=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name in {'INVENTORY.jsonl','SHA256SUMS','SEALED.json'}:continue
        rel=str(p.relative_to(out));entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p),**inventory.get(rel,{})})
    with (out/'INVENTORY.jsonl').open('w') as handle:
        for entry in entries:handle.write(json.dumps(entry,ensure_ascii=False)+'\n')
    entries.append({'path':'INVENTORY.jsonl','sha256':digest(out/'INVENTORY.jsonl')})
    (out/'SHA256SUMS').write_text(''.join(f"{e['sha256']}  {e['path']}\n" for e in entries))
    for entry in entries:
        if digest(out/entry['path'])!=entry['sha256']:raise ValueError('Final integrity verification failed')
    put('SEALED.json',{'version':'V1','sealed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':digest(out/'SHA256SUMS'),'verified_files':len(entries),'total_bytes':sum(e.get('bytes',0) for e in entries),'historical_gaps_disclosed':True})
    print(json.dumps(summary,indent=2));print('SEALED',out,flush=True)


if __name__=='__main__':main()
