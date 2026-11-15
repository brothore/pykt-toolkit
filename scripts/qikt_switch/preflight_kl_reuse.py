"""Native-data, trained-weight loss/gradient/RNG parity for KL reuse."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from evaluate_validation import load_run,load_saved_model
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.qikt_mamba import QIKT_MAMBA
from prediction_progress import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA preflight required')
    config,paths=load_run(a.checkpoint_dir);m=load_saved_model(config,paths,device='cuda');cfg=config['data_config']
    dataset=KTQueDataset(str(paths['fragment_path']),cfg['input_type'],set(cfg['folds'])-{paths['fold']},cfg['num_c'],cfg['max_concepts'],mode='train',aug_probs={})
    data=next(iter(DataLoader(dataset,batch_size=32)));m.model.train()
    initial_cpu=torch.get_rng_state().clone();initial_cuda=torch.cuda.get_rng_state_all()
    _,native=QIKT_MAMBA.train_one_step(m,data);outputs,current=m.predict_one_step(data,return_details=True)
    mask=current['sm'].bool();b=outputs['causal_baseline_logit'][mask];s=outputs['causal_student_logit'][mask];prob=torch.sigmoid(b)
    reference=native+m.causal_kl*((F.softplus(s)-prob*s)-(F.softplus(b)-prob*b)).mean();reference.backward()
    gradients={k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None}
    old_cpu=torch.get_rng_state().clone();old_cuda=torch.cuda.get_rng_state_all();m.zero_grad()
    torch.set_rng_state(initial_cpu);torch.cuda.set_rng_state_all(initial_cuda)
    calls=[];original=m.model.forward
    def count(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    m.model.forward=count
    _,optimized=m.train_one_step(data);optimized.backward()
    maximum=max(float((v.grad-gradients[k]).abs().max()) for k,v in m.named_parameters() if k in gradients)
    numerator=sum(float((v.grad-gradients[k]).double().square().sum()) for k,v in m.named_parameters() if k in gradients)
    denominator=sum(float(g.double().square().sum()) for g in gradients.values())
    relative=(numerator/max(denominator,1e-30))**.5;loss_error=abs(float(reference.detach())-float(optimized.detach()))
    rng_equal=torch.equal(old_cpu,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(old_cuda,torch.cuda.get_rng_state_all()))
    passed=len(calls)==1 and loss_error<=1e-7 and maximum<=1e-6 and relative<=1e-5 and rng_equal
    atomic_json(a.output,{'passed':passed,'checkpoint_sha256':digest(paths['checkpoint_path']),'training_fragments':32,
        'optimized_backbone_forward_calls':len(calls),'loss_error':loss_error,'max_gradient_error':maximum,
        'relative_gradient_l2_error':relative,'rng_equal':rng_equal,'optimizer_steps':0,
        'scope':'Implementation check on training data, no parameter update, no efficacy/stability result'})
    if not passed:raise ValueError('KL reuse tolerance failed')


if __name__=='__main__':main()
