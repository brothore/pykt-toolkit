"""Real batch checks, including activated-adapter label-isolation before training."""
import argparse
import copy
import json
from pathlib import Path
import torch
from pykt.utils import set_seed
from akt_experiment_io import build_model, loaders, predict
from reference_training import to_device
from prediction_progress import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.root/'baseline_config.json').read_text());torch.set_num_threads(2)
    set_seed(cfg['params']['seed']);base=build_model(cfg);base.eval()
    d,g=to_device(next(iter(loaders(cfg)[0])),'cuda')
    # Two real rows retain native tensor types/length and avoid excessive test cost.
    d={k:v[:2] for k,v in d.items()};g={k:v[:2] for k,v in g.items()}
    with torch.no_grad():expected=predict(base,d,g)[0]
    checks=[]
    for mode in ['ukt_cov','mtkt_cic','mtkt_time','lef_additive']:
        local=copy.deepcopy(cfg);local.update(mechanism=mode,scale=.1);set_seed(cfg['params']['seed'])
        model=build_model(local);model.eval()
        with torch.no_grad():error=float((predict(model,d,g)[0]-expected).abs().max())
        if error>1e-7:raise ValueError(f'Initial baseline mismatch: {mode}, {error}')
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(2):
            model.train();opt.zero_grad();out,reg=predict(model,d,g);mask=d['smasks'].bool()
            loss=torch.nn.functional.binary_cross_entropy(out[:,1:][mask].double(),d['shft_rseqs'][mask].double())+reg
            loss.backward()
            if not torch.isfinite(loss) or any(x.grad is not None and not torch.isfinite(x.grad).all() for x in model.parameters()):raise ValueError('Nonfinite real-data gradients')
            opt.step()
        model.eval()
        with torch.no_grad():
            original=predict(model,d,g)[0];changed={k:v.clone() for k,v in d.items()}
            # Flip response at full position100 and all later positions, keeping
            # the two overlapping native views aligned.
            changed['rseqs'][:,100:]=1-changed['rseqs'][:,100:]
            changed['shft_rseqs'][:,99:]=1-changed['shft_rseqs'][:,99:]
            alternate=predict(model,changed,g)[0];leak=float((original[:,:101]-alternate[:,:101]).abs().max())
        if leak>1e-7:raise ValueError(f'Current/future answer leak: {mode}: {leak}')
        checks.append(dict(mode=mode,initial_max_error=error,after_two_steps_prefix_error=leak,finite_gradients=True))
        del model,opt
    atomic_json(a.root/'preflight.json',dict(passed=True,real_container='DktForgetDataset tuple(data,gaps)',shape=[2,200],checks=checks,scientific_training=False))


if __name__=='__main__':main()
