"""Bounded discovery: stop a direction at its first validation candidate.

Each completed training epoch and window batch is resumable through existing
drivers. A protocol describes at most three rounds; no seed replication here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from cooperative_process import run_cooperative

import pandas as pd

from advance_experiment import evaluate_finished
from compare_validation import compare
from prediction_progress import atomic_json, digest
from resource_admission import wait_for_capacity
from sequence_balance_metrics import assess_balance


def require_evaluation_gate(design):
    gate = design.get('evaluation_gate')
    if not gate:
        return
    while not Path(gate['result']).exists():
        state = Path(gate['state'])
        if state.exists() and json.loads(state.read_text()).get('status') in ('failed', 'stopped'):
            raise RuntimeError('Required native evaluation audit failed or stopped')
        time.sleep(5)
    if not json.loads(Path(gate['result']).read_text()).get('passed'):
        raise RuntimeError('Native evaluation audit did not pass')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not 1 <= len(design['rounds']) <= 3:
        raise ValueError('Discovery requires one to three bounded rounds')
    root = Path(__file__).resolve().parents[2]
    scripts = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = Path(design['baseline_predictions'])
    base = pd.read_csv(baseline, sep='\t', dtype={'questions': str, 'concepts': str})
    records = []
    for number, settings in enumerate(design['rounds'], 1):
        folder = args.output_dir / f'round{number}'
        folder.mkdir(exist_ok=True)
        trial = folder / 'baseline'
        protocol = dict(settings, family=design['family'], version='discovery_v1', round=number,
                        dataset='nips_task34', fold=0, seed=42,
                        parent_protocol_sha256=digest(args.protocol),
                        baseline_predictions_sha256=digest(baseline))
        pp = folder / 'protocol.json'
        if pp.exists() and json.loads(pp.read_text()) != protocol:
            raise ValueError('Existing round identity differs')
        atomic_json(pp, protocol)
        bp = folder / 'balance.json'
        if bp.exists():
            record = json.loads(bp.read_text())
        else:
            if design.get('mode', 'training') == 'training':
                exit_path = trial / 'training_exit.json'
                if not exit_path.exists() or json.loads(exit_path.read_text())['returncode'] != 0:
                    wait_for_capacity(folder / 'training_admission.json')
                    command = [sys.executable, str(scripts / 'run_neural.py'), '--variant', 'baseline',
                               '--output-root', str(folder), '--resumable', '--resume',
                               '--objective-protocol', str(pp)]
                    with (folder / 'launcher.log').open('a') as log:
                        run_cooperative(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                require_evaluation_gate(design)
                summary = trial / 'round_summary.json'
                if not summary.exists():
                    wait_for_capacity(folder / 'validation_admission.json')
                training = json.loads(summary.read_text()) if summary.exists() else evaluate_finished('baseline', folder)
                output = trial / 'validation_window'
                if not (output / 'validation_window_results.json').exists():
                    wait_for_capacity(folder / 'window_admission.json')
                    command = [sys.executable, str(scripts / 'evaluate_window.py'), '--checkpoint-dir',
                               training['checkpoint_dir'], '--output-dir', str(output), '--source',
                               'validation', '--device', 'cuda', '--batch-size', '32',
                               '--checkpoint-batches', '64', '--resume']
                    with (folder / 'window_launcher.log').open('a') as log:
                        run_cooperative(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                candidate = pd.read_csv(output / 'window_predictions.tsv', sep='\t',
                                        dtype={'questions': str, 'concepts': str})
                provenance = {'training': training}
            elif design.get('mode') == 'prediction_consensus':
                from consensus_prediction_delta import consensus
                first_source=Path(design['candidate_predictions']);second_source=Path(design['anchor_predictions'])
                first=pd.read_csv(first_source,sep='\t',dtype={'questions':str,'concepts':str})
                second=pd.read_csv(second_source,sep='\t',dtype={'questions':str,'concepts':str})
                candidate=consensus(base,first,second,settings['consensus_mode'],settings['consensus_amount'])
                provenance={'no_new_training':True,'candidate_predictions_sha256':digest(first_source),
                    'anchor_predictions_sha256':digest(second_source),'output_precision':'FP64 algebra then FP32 scores',
                    'evaluation_protocol':'Fixed current-score consensus; no additional history; formal native fusion integration pending'}
                temporary=folder/'predictions.tmp';candidate.to_csv(temporary,sep='\t',index=False,float_format='%.17g');temporary.replace(folder/'predictions.tsv')
                provenance['prediction_sha256']=digest(folder/'predictions.tsv')
            elif design.get('mode') == 'prediction_delta':
                from transfer_prediction_delta import transfer
                source=Path(design['candidate_predictions']);anchor_source=Path(design['anchor_predictions'])
                student=pd.read_csv(source,sep='\t',dtype={'questions':str,'concepts':str})
                anchor=pd.read_csv(anchor_source,sep='\t',dtype={'questions':str,'concepts':str})
                candidate=transfer(base,student,anchor,settings['transfer_mode'],settings['strength'])
                provenance={'candidate_predictions_sha256':digest(source),'anchor_predictions_sha256':digest(anchor_source),
                    'no_new_training':True,'output_precision':'FP64 algebra then explicit FP32 score quantization',
                    'evaluation_protocol':'Additional streaming prediction-difference memory' if settings['transfer_mode']=='prefix_center' else 'Current-score algebra; no extra temporal information; formal fusion integration pending'}
                temporary=folder/'predictions.tmp';candidate.to_csv(temporary,sep='\t',index=False,float_format='%.17g');temporary.replace(folder/'predictions.tsv')
                provenance['prediction_sha256']=digest(folder/'predictions.tsv')
            else:
                source = Path(design['candidate_predictions'])
                candidate = pd.read_csv(source, sep='\t', dtype={'questions': str, 'concepts': str})
                keys = ['orirow', 'qidx', 'late_trues', 'questions', 'concepts']
                candidate = candidate.sort_values(['orirow', 'qidx']).reset_index(drop=True)
                base = base.sort_values(['orirow', 'qidx']).reset_index(drop=True)
                if not candidate[keys].equals(base[keys]):
                    raise ValueError('Mixture alignment differs')
                anchor_source = Path(design['anchor_predictions']) if design.get('anchor_predictions') else baseline
                anchor = pd.read_csv(anchor_source, sep='\t', dtype={'questions': str, 'concepts': str})
                anchor = anchor.sort_values(['orirow', 'qidx']).reset_index(drop=True)
                if not anchor[keys].equals(base[keys]):
                    raise ValueError('Anchor alignment differs')
                weight = settings['candidate_weight']
                if not 0 <= weight <= 1:
                    raise ValueError('Invalid fixed mixture weight')
                candidate['late_mean'] = weight * candidate.late_mean + (1-weight) * anchor.late_mean
                provenance = {'candidate_predictions_sha256': digest(source),
                              'anchor_predictions_sha256': digest(anchor_source), 'no_new_training': True}
            comparison, paired = compare(base, candidate, 500, metric_key='validation_window_auc')
            record = assess_balance(paired, comparison['pooled_delta'])
            if design.get('require_tail20'):
                record['checks'].update(fixed_weak20_preserved=record['deltas']['fixed_baseline_bottom20_mean']>=0,
                                        own_tail20_preserved=record['deltas']['own_bottom20_mean']>=0)
                record['exploratory_candidate_pass']=all(record['checks'].values())
                record['strict_goal_pass']=record['strict_goal_pass'] and record['exploratory_candidate_pass']
            record.update(protocol=protocol, comparison=comparison, **provenance)
            if 'substantive_signal' in design:
                threshold = design['substantive_signal']
                record['substantive_development_signal'] = bool(record['exploratory_candidate_pass']
                    and record['candidate']['std_ddof0'] <= record['baseline']['std_ddof0'] * (1-threshold['std_fraction'])
                    and record['candidate']['range'] <= record['baseline']['range'] * (1-threshold['range_fraction'])
                    and record['paired_orirow_bootstrap_ci95']['std_delta'][1] < 0
                    and record['deltas']['fixed_baseline_bottom20_mean'] >= 0
                    and record['deltas']['own_bottom20_mean'] >= 0)
            paired.to_csv(folder / 'paired_orirow.csv')
            atomic_json(bp, record)
        records.append(record)
        candidate_found = record['exploratory_candidate_pass']
        atomic_json(args.output_dir / 'progress.json', {'completed_rounds': number,
                    'candidate_found': candidate_found, 'next': 'record_and_stop' if candidate_found else 'next_bounded_round'})
        if candidate_found:
            break
    atomic_json(args.output_dir / 'summary.json', {'family': design['family'], 'rounds': records,
                'candidate_found': records[-1]['exploratory_candidate_pass'],
                'stability_verified': False, 'test_evaluated': False,
                'formal_windowauclate_mean': None, 'automatic_replication': False})


if __name__ == '__main__':
    main()
