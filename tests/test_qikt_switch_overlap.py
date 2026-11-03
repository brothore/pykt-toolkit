import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pykt.datasets.que_data_loader import KTQueDataset
from pykt.models.qikt_mamba import QIKT_MAMBA

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qikt_switch'))
from prepare_overlap import build


def test_overlap_preserves_exact_baseline_targets_and_loader_masks(tmp_path):
    raw=pd.DataFrame([dict(uid=7,fold=1,questions='0,1,2,3,4,5,6,7',concepts='0,1,0,1,0,1,0,1',responses='1,0,1,0,1,0,1,0',timestamps='0,1,2,3,4,5,6,7'),
                      dict(uid=8,fold=0,questions='0,1,2',concepts='0,1,0',responses='0,1,0',timestamps='0,1,2')])
    pieces=[]
    for start in [0,4]:
        r=raw.iloc[0];p=dict(uid=7,fold=1,selectmasks='1,1,1,1')
        for k in ['questions','concepts','responses','timestamps']:p[k]=','.join(r[k].split(',')[start:start+4])
        pieces.append(p)
    windows,positions,info=build(raw,pd.DataFrame(pieces),length=5,overlap=2)
    assert info['target_sets_equal'] and info['baseline_scored_events']==6
    assert set(positions.qidx)=={1,2,3,5,6,7}
    assert windows.fold.unique().tolist()==[1]
    assert positions.duplicated(['uid','qidx']).sum()==0
    file=tmp_path/'overlap.csv';windows.to_csv(file,index=False)
    dataset=KTQueDataset(str(file),['questions','concepts'],{1},2,2,mode='train')
    for index in range(len(dataset)):
        batch=dataset[index];mapping=positions[positions.window==index]
        expected=torch.zeros(4,dtype=torch.bool)
        expected[torch.tensor(mapping.local_position.to_numpy()-1)]=True
        assert torch.equal(batch['smasks'],expected)
        assert batch['shft_qseqs'][expected].tolist()==mapping.qidx.tolist()
        assert torch.all(batch['masks'][expected])
    m=QIKT_MAMBA(num_q=8,num_c=2,emb_size=4,dropout=0,emb_type='attn',version='lstm',other_config={'output_mode':'an_irt'})
    m.model.train();_,loss=m.train_one_step(next(iter(DataLoader(dataset,batch_size=2))))
    assert torch.isfinite(loss);loss.backward()
