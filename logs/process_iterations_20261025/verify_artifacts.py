import copy,hashlib,json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pykt.models.dkt import DKT
from pykt.models.dkt_process import DKTProcess
LOG=Path(__file__).resolve().parent;BASE=ROOT/'examples/saved_model/process_iterations_20261025'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
freeze=json.loads((LOG/'selection_before_test.json').read_text())
checks={'checkpoint_sha_unchanged':{},'pure_dkt_export_identical':{},'source_snapshot_unchanged':{},'original_files_unchanged':{}}
for variant,expected in freeze['checkpoint_sha256'].items():
 checks['checkpoint_sha_unchanged'][variant]=sha(BASE/variant/'qid_model.ckpt')==expected
 if variant=='baseline':continue
 config=json.loads((BASE/variant/'config.json').read_text());n=config['data_config']['num_c'];cfg=config['model_config']
 model=DKTProcess(n,**copy.deepcopy(cfg)).eval();model.load_state_dict(torch.load(BASE/variant/'qid_model.ckpt',map_location='cpu',weights_only=True),strict=True)
 old=DKT(n,cfg['emb_size'],cfg['dropout']).eval();old.load_state_dict(torch.load(BASE/variant/'qid_dkt_backbone.ckpt',map_location='cpu',weights_only=True),strict=True)
 c=torch.arange(24).reshape(2,12)%n;r=torch.arange(24).reshape(2,12)%2
 with torch.no_grad():checks['pure_dkt_export_identical'][variant]=torch.equal(model(c,r),old(c,r))
for name,expected in json.loads((LOG/'source_manifest.json').read_text()).items():checks['source_snapshot_unchanged'][name]=sha(ROOT/name)==expected
for name in ['pykt/models/dkt.py','pykt/datasets/data_loader.py','ALSO/also.py','configs/kt_config.json']:
 checks['original_files_unchanged'][name]=(ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
checks['baseline_matches_previous_baseline']=sha(BASE/'baseline/qid_model.ckpt')==sha(ROOT/'examples/saved_model/first_action_iterations_20261025/baseline/qid_model.ckpt')
assert all(all(v.values()) if isinstance(v,dict) else v for v in checks.values()),checks
(LOG/'artifact_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
