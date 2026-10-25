"""Recompute validation diagnostics for frozen checkpoints, no optimizer/test data."""
import copy,json,sys,torch
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'examples'))
from pykt.datasets import init_dataset4train
from pykt.models.dkt_process import DKTProcess
from train_dkt_process import process_diagnostics,gradient_diagnostics
from train_dkt_first_action import response_diagnostics

torch.set_num_threads(1)
for variant in sys.argv[1:]:
 save=ROOT/'examples/saved_model/process_iterations_20261025'/variant
 snap=json.loads((save/'config.json').read_text()); cfg=copy.deepcopy(snap['model_config'])
 train,valid,*_=init_dataset4train('assist2009','dkt_process',{'assist2009':snap['data_config']},0,128,args=SimpleNamespace(process_mode=cfg['process_mode']),aug_probs={})
 model=DKTProcess(snap['data_config']['num_c'],**cfg)
 model.load_state_dict(torch.load(save/'qid_model.ckpt',map_location='cpu',weights_only=True))
 model.eval()
 out={'response':response_diagnostics(model,valid),'process':process_diagnostics(model,valid,train.dataset),'gradients':gradient_diagnostics(model,valid)}
 (save/'validation_diagnostics_extra.json').write_text(json.dumps(out,indent=2)+'\n')
 print(variant,json.dumps(out),flush=True)
