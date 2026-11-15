"""Refresh sequence evidence and the user's single report after queued results."""
import argparse
from pathlib import Path
import sys
from cooperative_process import run_cooperative


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    scripts=Path(__file__).parent
    for command in [
        [sys.executable,str(scripts/'analyze_module_sequence_profiles.py'),'--repo',str(a.repo),'--output',str(a.repo/'logs/advanced_reference_20261115/module_sequence_profiles')],
        [sys.executable,str(scripts/'render_three_model_summary.py'),'--repo',str(a.repo),'--output',str(a.repo/'docs/先进模型消融与GRU融合/13_三模型消融与全部GRU融合汇总.md')]]:
        run_cooperative(command,cwd=a.repo,check=True)


if __name__=='__main__':main()
