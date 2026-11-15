"""Measure new GRU loads after the preflight CUDA process has exited."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from cooperative_process import run_cooperative
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output-root',type=Path);p.add_argument('--group-case');a=p.parse_args()
    destination=a.output_root or a.root/'resource_calibration'
    destination.mkdir(parents=True,exist_ok=True)
    probes=json.loads((a.root/'probe_ready.json').read_text());results={}
    if probes.get('passed') is False:
        atomic_json(destination/'summary.json',{'status':'skipped_source_selection','cases':{},'executed':False})
        return
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    if active:raise RuntimeError('Calibration requires a free GPU, observed PIDs: '+active)
    for name,case in probes['cases'].items():
        out=destination/name;out.mkdir(parents=True,exist_ok=True)
        protocol=json.loads(Path(case['protocol']).read_text())
        protocol.update(native_runtime_flags='pykt.utils.set_seed including deterministic cuDNN and CUDA_LAUNCH_BLOCKING=1',source_checkpoint_sha256=digest(Path(case['checkpoint'])/'attn_model.ckpt'))
        atomic_json(out/'protocol.json',protocol)
        run_cooperative([sys.executable,str(Path(__file__).with_name('calibrate_resources.py')),'--protocol',str(out/'protocol.json'),'--output-dir',str(out/'measurements')],check=True)
        results[name]=json.loads((out/'measurements/summary.json').read_text())
    if a.group_case:
        combined={}
        for phase in ['training','inference']:
            per_case={name:value['recommendations'][phase]['recommended_concurrency'] for name,value in results.items()}
            combined[phase]={'recommended_concurrency':min(per_case.values()),'case_recommendations':per_case,
                'criterion':'Conservative minimum across measured cases for a shared mixed-family slot group; not a measured mixed-load optimum'}
        results[a.group_case]={'recommendations':combined}
    atomic_json(destination/'summary.json',{'status':'completed','cases':results,'limitations':'New trainable GRU and representative heavy transfer with native runtime flags; not every variant global optimum'})


if __name__=='__main__':main()
