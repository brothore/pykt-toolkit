"""Verify completed frozen transfers against their exact GRU reference."""
import argparse
import json
from pathlib import Path
import signal
import threading
import torch
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    stop=threading.Event();signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    while not stop.is_set():
        ready=True
        for name in ['S0','S1','S2','S3','final_review']:
            path=root/'jobs'/name/'state.json';v=json.loads(path.read_text()) if path.exists() else {'status':'queued'}
            if v['status'] in {'failed','stopped'}:raise RuntimeError('Unsuccessful prerequisite: '+name)
            ready=ready and v['status']=='completed'
        if ready:break
        stop.wait(10)
    if stop.is_set():return 130
    origin=json.loads((root/'baseline_ready.json').read_text());checkpoint=Path(origin['checkpoint_dir'])/'attn_model.ckpt';base=torch.load(checkpoint,map_location='cpu',weights_only=True);records=[]
    for family,prefix in [('S2','model.rasch_transfer.'),('S3','model.ukt_transfer.')]:
        for path in sorted((root/family).glob('round*/balance.json')):
            if stop.is_set():return 130
            v=json.loads(path.read_text());selected=Path(v['training']['checkpoint_dir'])/'attn_model.ckpt'
            window=json.loads((path.parent/'baseline/validation_window/validation_window_results.json').read_text())
            if digest(selected)!=window['checkpoint_sha256']:raise ValueError('Selected inference identity differs')
            candidate=torch.load(selected,map_location='cpu',weights_only=True)
            if any(k not in candidate or not torch.equal(t,candidate[k]) for k,t in base.items()):raise ValueError('Frozen backbone differs: '+str(path))
            extra=set(candidate)-set(base)
            if not extra or any(not k.startswith(prefix) for k in extra):raise ValueError('Unexpected transferred parameter keys')
            if any(not torch.isfinite(candidate[k]).all() for k in extra):raise ValueError('Nonfinite transferred parameters')
            records.append({'family':family,'round':path.parent.name,'unchanged_backbone_tensors':len(base),'new_tensors':len(extra),'selected_sha256':digest(selected),'balance_sha256':digest(path)})
    calibration=json.loads((root/'resource_calibration/summary.json').read_text())
    for case in calibration['cases'].values():
        for cell in case.get('measurements',[]):
            for worker in cell['workers']:
                flags=worker['runtime_flags']
                if not flags['cudnn_deterministic'] or flags['cudnn_benchmark'] or flags['CUDA_LAUNCH_BLOCKING']!='1' or flags['CUBLAS_WORKSPACE_CONFIG']!=':16:8':raise ValueError('Resource probe native flags differ')
    atomic_json(root/'completed_review/freeze_audit.json',{'passed':True,'baseline_checkpoint_sha256':digest(checkpoint),'records':records,'native_resource_flags_verified':True,'scope':'Selected frozen-backbone tensors unchanged and transferred keys finite; not an AUC preservation or stability guarantee','goal_achieved':False})


if __name__=='__main__':raise SystemExit(main())
