"""Verify installed CUDA kernels and one full-size QIKT-Mamba optimizer step."""
import argparse
import json
from pathlib import Path
import time
from importlib.metadata import version
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    from mamba_ssm import Mamba
    from causal_conv1d import causal_conv1d_fn
    from pykt.models.qikt_mamba import QIKT_MAMBA
    assert torch.cuda.is_available()
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
    tiny=Mamba(d_model=64,d_state=16).cuda();x=torch.randn(2,10,64,device='cuda',requires_grad=True)
    tiny(x).square().mean().backward();assert torch.isfinite(x.grad).all()
    del tiny,x;torch.cuda.empty_cache()
    cfg=dict(output_mode='an_irt',loss_q_all_lambda=1.,loss_c_all_lambda=.5,loss_c_next_lambda=1.5,
             output_q_all_lambda=1.,output_c_all_lambda=1.,output_c_next_lambda=1.)
    model=QIKT_MAMBA(num_q=948,num_c=57,emb_size=256,dropout=.3,emb_type='attn',device='cuda',
                     num_attn_head=1,mlp_layer_num=1,version='public_mamba',other_config=cfg)
    model.model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    q=torch.randint(948,(32,200),device='cuda');c=torch.randint(57,(32,200,2),device='cuda');c[:,:,1]=-1
    r=torch.randint(2,(32,200),device='cuda')
    batch=dict(cq=q,cc=c,cr=r,qshft=q[:,1:],cshft=c[:,1:],rshft=r[:,1:].float(),sm=torch.ones(32,199,dtype=torch.bool,device='cuda'))
    torch.cuda.synchronize();start=time.perf_counter()
    p,loss=model.train_one_step(batch,process=False)
    assert p.shape==(32,199) and torch.isfinite(loss)
    loss.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    optimizer.step();torch.cuda.synchronize()
    result={'purpose':'CUDA installation and full-size optimizer-step check, not a trained-model experiment',
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'mamba_ssm':version('mamba-ssm'),'causal_conv1d':version('causal-conv1d'),
        'full_step_seconds':time.perf_counter()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'parameter_count':sum(v.numel() for v in model.parameters()),'synthetic_loss':loss.item(),
        'shape':list(p.shape),'batch_size':32,'sequence_length':200,'embedding_size':256,'passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
