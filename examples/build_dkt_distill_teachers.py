"""Fixed-budget, student-disjoint teacher cross-fitting; no outer-val selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader
from pykt.datasets.distill_dataset import HistoryProcessDataset, build_history_sidecar
from pykt.models.dkt import DKT
from pykt.models.dkt_distill import RichHistoryTeacher
from pykt.utils import set_seed


def uid_hash(values):
    return hashlib.sha256(','.join(map(str, sorted(set(values)))).encode()).hexdigest()


def train_teacher(dataset, rich, seed, epochs, save, data):
    set_seed(seed)
    model = RichHistoryTeacher(data['num_c'], 200, .2, feature_dim=11).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    losses = []
    for epoch in range(epochs):
        model.train(); total = 0.; count = 0
        for batch in loader:
            c, r = batch['cseqs'].to(DEVICE).long(), batch['rseqs'].to(DEVICE).long()
            features = batch['history_process'].to(DEVICE)
            if not rich: features = torch.zeros_like(features)
            p = model(c, r, features).gather(-1, batch['shft_cseqs'].to(DEVICE).long().unsqueeze(-1)).squeeze(-1)
            mask = batch['teacher_event_mask'].to(DEVICE)
            if not mask.any(): continue
            loss = F.binary_cross_entropy(p[mask].double(), batch['shft_rseqs'].to(DEVICE)[mask].double())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())*int(mask.sum()); count += int(mask.sum())
        losses.append(total/max(count,1))
        print('TEACHER_EPOCH', save.name, epoch+1, losses[-1], flush=True)
    save.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save/'model.ckpt')
    (save/'config.json').write_text(json.dumps({'rich':rich,'seed':seed,'fixed_epochs':epochs,
        'normalizer':dataset.normalizer,'fit_uid_hash':uid_hash(dataset.dori['uid']),
        'fit_students':len(set(dataset.dori['uid'])),'epoch_losses':losses,
        'selection':'last fixed epoch; no held-out or outer-val checkpoint selection'},indent=2)+'\n')
    return model


def predict(model, dataset, rich=True, standard=False):
    probabilities, truths, masks, rows = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0):
            c,r=batch['cseqs'].to(DEVICE).long(),batch['rseqs'].to(DEVICE).long()
            if standard: output=model(c,r)
            else:
                f=batch['history_process'].to(DEVICE)
                output=model(c,r,f if rich else torch.zeros_like(f))
            p=output.gather(-1,batch['shft_cseqs'].to(DEVICE).long().unsqueeze(-1)).squeeze(-1).cpu().numpy()
            probabilities.append(p);truths.append(batch['shft_rseqs'].numpy());masks.append(batch['teacher_event_mask'].numpy())
            rows.extend(batch['row_index'].tolist())
    p,y,mask=np.concatenate(probabilities),np.concatenate(truths),np.concatenate(masks)
    metrics={'event_count':int(mask.sum()),'auc':float(roc_auc_score(y[mask],p[mask])),
             'nll':float(log_loss(y[mask],p[mask],labels=[0,1])),
             'brier':float(((p[mask]-y[mask])**2).mean())}
    return np.asarray(rows),p,mask,metrics


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--save_dir',type=Path,required=True)
    parser.add_argument('--teacher_epochs',type=int,default=6);args=parser.parse_args()
    save=args.save_dir.resolve();save.mkdir(parents=True,exist_ok=True)
    if (save/'soft_targets.npz').exists():raise FileExistsError('Refusing to regenerate completed crossfit')
    torch.set_num_threads(1)
    data=json.loads((ROOT/'configs/data_config.json').read_text())['assist2009']
    source=ROOT/'data/assist2009/train_valid_sequences.csv'
    history=build_history_sidecar(source.parent)
    with np.load(history,allow_pickle=False) as stored:
        folds,uids=stored['folds'].copy(),stored['uids'].copy();shape=stored['events'][:,1:].shape
    output={family+'_probs':np.full(shape,np.nan,dtype=np.float32) for family in ['rich','thin']}
    masks=np.zeros(shape,dtype=bool);records=[];started=time.monotonic()
    for heldout in [1,2,3,4]:
        fit_folds={1,2,3,4}-{heldout}
        fit=HistoryProcessDataset(source,data['input_type'],fit_folds)
        target=HistoryProcessDataset(source,data['input_type'],{heldout},mode='predict',normalizer=fit.normalizer)
        fit_uids,target_uids=set(fit.dori['uid']),set(target.dori['uid'])
        assert fit_uids.isdisjoint(target_uids)
        for family in ['rich','thin']:
            model=train_teacher(fit,family=='rich',42+heldout,args.teacher_epochs,save/f'{family}_heldout_{heldout}',data)
            rows,p,mask,metrics=predict(model,target,family=='rich')
            output[family+'_probs'][rows]=p;masks[rows]=mask
            records.append({'family':family,'fit_folds':sorted(fit_folds),'heldout_fold':heldout,
                'fit_students':len(fit_uids),'heldout_students':len(target_uids),'uid_disjoint':True,
                'fit_uid_hash':uid_hash(fit_uids),'heldout_uid_hash':uid_hash(target_uids),'metrics':metrics})
            print('OOF_FINISHED',json.dumps(records[-1]),flush=True)
            del model
    train=HistoryProcessDataset(source,data['input_type'],{1,2,3,4})
    valid=HistoryProcessDataset(source,data['input_type'],{0},mode='valid',normalizer=train.normalizer)
    assert set(train.dori['uid']).isdisjoint(set(valid.dori['uid']))
    valid_metrics={}
    for family in ['rich','thin']:
        model=train_teacher(train,family=='rich',42,args.teacher_epochs,save/(family+'_outer_train'),data)
        _,_,_,metrics=predict(model,valid,family=='rich');valid_metrics[family]=metrics
        del model
    baseline=DKT(data['num_c'],200,.2).to(DEVICE)
    baseline.load_state_dict(torch.load(ROOT/'examples/saved_model/process_iterations_20261025/baseline/qid_model.ckpt',map_location=DEVICE,weights_only=True))
    _,_,_,valid_metrics['standard_dkt_baseline']=predict(baseline,valid,standard=True)
    for family in ['rich','thin']:
        assert np.isfinite(output[family+'_probs'][masks]).all()
    assert not masks[folds==0].any()
    with (save/'soft_targets.npz').open('wb') as stream:
        np.savez_compressed(stream,**output,mask=masks,folds=folds,uids=uids)
    sources=json.loads(history.with_suffix('.json').read_text())['sources']
    metadata={'sources':sources,'fixed_teacher_epochs':args.teacher_epochs,'crossfit':records,
              'outer_validation_original_event_metrics':valid_metrics,'outer_fold_0_used_for_teacher_fit_or_selection':False,
              'train_soft_events':int(masks.sum()),'seconds':time.monotonic()-started}
    (save/'soft_targets.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('TEACHER_BUNDLE_FINISHED',json.dumps(metadata),flush=True)


if __name__=='__main__':main()
