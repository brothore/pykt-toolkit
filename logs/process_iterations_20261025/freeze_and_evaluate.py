import datetime,hashlib,json,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
LOG=Path(__file__).resolve().parent
BASE=ROOT/'examples/saved_model/process_iterations_20261025'
variants=['baseline','r1_joint','r2_conditional','r3_weaker','r4_time_reweight']
auc={v:json.loads((BASE/v/'validation.json').read_text())['validation_auc'] for v in variants}
freeze=LOG/'selection_before_test.json'
assert not freeze.exists(), 'Do not repeat frozen test evaluation'
assert all(not (BASE/v/'all_results.json').exists() for v in variants)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection={'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_by':'validation AUC of retained checkpoint only; no test metrics used','selected_variant':max(auc,key=auc.get),'validation_auc':auc,'checkpoint_sha256':{v:sha(BASE/v/'qid_model.ckpt') for v in variants},'test_evaluation_started':False}
freeze.write_text(json.dumps(selection,indent=2)+'\n')
protocol=json.loads((LOG/'protocol.json').read_text());v=json.loads((BASE/'r4_time_reweight/validation.json').read_text());protocol['rounds'][-1].update(validation_auc=v['validation_auc'],diagnostics=v['validation_diagnostics']);protocol['selection']=selection;(LOG/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
files=['pykt/models/dkt_process.py','pykt/datasets/process_dataset.py','examples/train_dkt_process.py','examples/evaluate_dkt_first_action.py','tests/test_dkt_process.py','pykt/models/init_model.py','pykt/models/train_model.py','pykt/models/evaluate_model.py','pykt/datasets/init_dataset.py']
source={}
for name in files:
 p=ROOT/name; source[name]=sha(p); target=LOG/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
(LOG/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
print('FROZEN_SELECTION',selection['selected_variant'],flush=True)
codes={}
for variant in variants:
 with (LOG/(variant+'_test.log')).open('w') as log:
  result=subprocess.run([sys.executable,'-u','examples/evaluate_dkt_first_action.py','--save_dir',str(BASE/variant)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 codes[variant]=result.returncode
 (LOG/'test_return_codes.json').write_text(json.dumps(codes,indent=2)+'\n')
 assert result.returncode==0, variant+' evaluation failed'
 assert sha(BASE/variant/'qid_model.ckpt')==selection['checkpoint_sha256'][variant]
 print('TEST_FINISHED',variant,flush=True)
protocol['test_status']='all_five_completed';(LOG/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
