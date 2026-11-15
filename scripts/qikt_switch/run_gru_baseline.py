"""Build the new independent dual-GRU reference before any UKT transfer."""
import argparse
import json
from pathlib import Path
import sys
from cooperative_process import run_cooperative
from prediction_progress import atomic_json
from advance_experiment import evaluate_finished


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    folder=a.root/'B0/round1';folder.mkdir(parents=True,exist_ok=True)
    protocol=folder/'protocol.json'
    expected={'alpha':0.,'rank_weight':0.,'model_settings':{'version':'gru'}}
    if protocol.exists() and json.loads(protocol.read_text())!=expected:raise ValueError('Baseline identity changed')
    atomic_json(protocol,expected)
    scripts=Path(__file__).resolve().parent;repo=scripts.parents[1]
    run_cooperative([sys.executable,str(scripts/'run_neural.py'),'--variant','baseline','--output-root',str(folder),'--resumable','--resume','--objective-protocol',str(protocol)],cwd=repo,check=True)
    trial=folder/'baseline';summary=trial/'round_summary.json'
    value=json.loads(summary.read_text()) if summary.exists() else evaluate_finished('baseline',folder)
    output=trial/'validation_window'
    run_cooperative([sys.executable,str(scripts/'evaluate_window.py'),'--checkpoint-dir',value['checkpoint_dir'],'--output-dir',str(output),'--source','validation','--device','cuda','--batch-size','32','--checkpoint-batches','64','--resume'],cwd=repo,check=True)
    atomic_json(a.root/'baseline_ready.json',{'passed':True,'architecture':'two independent GRUs','checkpoint_dir':value['checkpoint_dir'],'predictions':str(output/'window_predictions.tsv'),'completed_epochs':value['completed_epochs'],'selected_epoch':value['selected_epoch'],'formal_windowauclate_mean':None})


if __name__=='__main__':main()
