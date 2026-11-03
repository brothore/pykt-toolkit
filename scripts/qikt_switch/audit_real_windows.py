"""Sample real preprocessed windows against the streaming native-question inputs."""
from pathlib import Path
import json
import pandas as pd
import torch
from evaluate_window import QuestionWindows
from pykt.datasets.que_data_loader import KTQueDataset
root=Path(__file__).resolve().parents[2];out=root/'logs/qikt_switch_20261103/real_window_audit';out.mkdir(exist_ok=True)
raw=pd.read_csv(root/'data/nips_task34/test_quelevel.csv').head(10);uids=set(raw.uid);need={int(r.uid):min(2,max(1,len(r.responses.split(','))-199)) for r in raw.itertuples()};seen={u:0 for u in uids};sample=[]
for chunk in pd.read_csv(root/'data/nips_task34/test_window_sequences_quelevel.csv',chunksize=500):
 for r in chunk.to_dict('records'):
  uid=int(r['uid'])
  if uid in uids and seen[uid]<need[uid]:sample.append(r);seen[uid]+=1
 if all(seen[u]==need[u] for u in uids):break
assert all(seen[u]==need[u] for u in uids)
file=out/'native_sample.csv';pd.DataFrame(sample).to_csv(file,index=False)
native=KTQueDataset(str(file),['questions','concepts'],{-1},57,2,mode='test');stream=QuestionWindows(raw,200,2)
lookup={};counts={u:0 for u in uids}
for i,(row_index,stop,first) in enumerate(stream.windows):
 uid=stream.rows[row_index]['uid']
 if counts[uid]<need[uid]:lookup[(uid,counts[uid])]=i;counts[uid]+=1
counts={u:0 for u in uids};checked=0
for index,row in enumerate(sample):
 uid=int(row['uid']);a,meta=stream[lookup[(uid,counts[uid])]];b=native[index]
 for key in ['qseqs','shft_qseqs','cseqs','shft_cseqs','rseqs','shft_rseqs','tseqs','shft_tseqs','masks','smasks']:
  assert torch.equal(a[key],b[key]),(uid,counts[uid],key)
 counts[uid]+=1;checked+=1
result={'scope':'real-data input provenance sampling, not model performance evaluation','uids':sorted(map(int,uids)),'windows_checked':checked,'exact_shifted_tensor_matches':checked,'fields_checked':10,'followup_windows':sum(v>1 for v in counts.values()),'sampling_limit':'first 10 raw UIDs, first two windows where available; not exhaustive legacy-file equivalence','test_predictions_computed':False}
(out/'evidence.json').write_text(json.dumps(result,indent=2));print(result)
