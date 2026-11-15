"""Resumable CPU diagnostics, checkpointed per trial, suitable for a tsp job.

Stop with SIGTERM/SIGINT; in-flight trial computations finish before exit 130.
Resume uses --resume with identical inputs/code; completed checkpoints are reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys

import pandas as pd
from audit_inputs import sha
from validation_diagnostics import diagnose


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def analyze_one(record):
    result = diagnose(pd.read_csv(record['paired_path']))
    return {'variant': record['variant'], 'completed_epochs': record['completed_epochs'],
            'selected_epoch': record['selected_epoch'], 'diagnostics': result,
            'dataset': record['dataset'], 'fold': record['fold'], 'seed': record['seed'],
            'model': record['model'], 'input_sha256': record['input_sha256']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--neural-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.workers < 1: ap.error('workers must be positive')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = []
    for directory in sorted(args.neural_root.resolve().iterdir()):
        if directory.name == 'baseline' or not (directory / 'round_summary.json').exists(): continue
        paired = directory / 'validation_window/paired_orirow.csv'
        if not paired.exists(): raise ValueError(f'Window validation prerequisite missing: {paired}')
        summary = json.loads((directory / 'round_summary.json').read_text())
        record = {k: summary[k] for k in ['variant', 'dataset', 'fold', 'seed', 'model', 'completed_epochs', 'selected_epoch']}
        record.update(paired_path=str(paired), input_sha256=sha(paired))
        records.append(record)
    if not records: raise ValueError('No completed trial inputs')
    sources = {str(p): sha(p) for p in [Path(__file__).resolve(), Path(__file__).with_name('validation_diagnostics.py').resolve(), Path(__file__).with_name('audit_inputs.py').resolve()]}
    identity = hashlib.sha256(json.dumps({'inputs': records, 'sources': sources}, sort_keys=True).encode()).hexdigest()
    path = out / 'state.json'
    if path.exists():
        if not args.resume: raise ValueError('Existing state; inspect then use --resume')
        state = json.loads(path.read_text())
        if state['identity'] != identity: raise ValueError('Inputs or analysis code changed; use a new output directory')
    else:
        root = Path(__file__).resolve().parents[2]
        def git(*command): return subprocess.check_output(['git', *command], cwd=root).decode()
        state = {'identity': identity, 'created_utc': now(), 'inputs': records, 'source_sha256': sources,
                 'git_commit': git('rev-parse', 'HEAD').strip(),
                 'tracked_worktree_diff_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
                 'python': sys.version, 'dependencies': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas']},
                 'hostname': socket.gethostname(), 'platform': platform.platform(), 'history': [],
                 'scope': 'Independent validation diagnostics, no GPU or test data; no candidate selection',
                 'resume_granularity': 'Completed trial; stop finishes in-flight trials and preserves each atomic result.'}
    state.update(status='running', pid=os.getpid(), command=sys.argv, resources={'cpu_workers': args.workers, 'gpu': False},
                 queue_record=str(out / 'queue_submission.json'))
    state['history'].append({'event': 'resume' if args.resume else 'start', 'utc': now(), 'pid': os.getpid()})
    write_json(path, state)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    completed = []
    try:
        pending = []
        for record in records:
            checkpoint = out / (record['variant'] + '.json')
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text())
                if result['input_sha256'] != record['input_sha256']: raise ValueError('Checkpoint input changed')
                completed.append(result)
            else: pending.append(record)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(analyze_one, record): record for record in pending}
            for job in as_completed(jobs):
                if job.cancelled(): continue
                result = job.result(); write_json(out / (result['variant'] + '.json'), result); completed.append(result)
                state.update(completed_variants=sorted(r['variant'] for r in completed), updated_utc=now())
                write_json(path, state)
                if stopping:
                    for queued in jobs: queued.cancel()
        state.update(status='stopped' if stopping else 'completed', exit_code=130 if stopping else 0)
        write_json(out / 'summary.json', {'trials': sorted(completed, key=lambda r: r['variant']), 'scope': state['scope']})
        state['artifacts'] = {str(p): sha(p) for p in out.glob('*.json') if p.name not in ['state.json', 'queue_submission.json']}
    except BaseException as error:
        state.update(status='failed', exit_code=1, error=repr(error))
        raise
    finally:
        state.update(updated_utc=now()); state['history'].append({'event': state['status'], 'utc': now()})
        write_json(path, state)
    print(json.dumps({'status': state['status'], 'completed_trials': len(completed), 'output': str(out)}), flush=True)
    return state['exit_code']


if __name__ == '__main__':
    sys.exit(main())
