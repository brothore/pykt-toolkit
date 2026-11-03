"""Fast validation question-window scoring for DKT probability vectors.

Uses each raw question's PRE-question state, exactly the original
expand_question window rule. This does not replace the public test evaluator.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,log_loss


def make_windows(frame,maxlen=200):
    early_c=[];early_r=[];owners=[];times=[];e_cs=[];e_y=[];e_q=[];e_row=[]
    late_c=[];late_r=[];l_cs=[];l_y=[];l_q=[];l_row=[];qid=-1;max_c=1
    for row_id,row in enumerate(frame.to_dict('records')):
        c=np.asarray([int(x) for x in row['concepts'].split(',')],dtype=np.int16)
        r=np.asarray([int(x) for x in row['responses'].split(',')],dtype=np.int8)
        repeat=np.asarray([int(x) for x in row['is_repeat'].split(',')])
        starts=np.flatnonzero(repeat==0);assert starts[0]==0
        ec=np.zeros(maxlen-1,dtype=np.int16);er=np.zeros(maxlen-1,dtype=np.int8);n=min(len(c),maxlen-1)
        ec[:n]=c[:n];er[:n]=r[:n];early_c.append(ec);early_r.append(er)
        for j,start in enumerate(starts):
            qid+=1;end=int(starts[j+1]) if j+1<len(starts) else len(c);cs=c[start:end]
            assert len(set(cs))==len(cs) and (r[start:end]==r[start]).all()
            if not start:continue
            max_c=max(max_c,len(cs))
            if start<=maxlen-1:
                owners.append(row_id);times.append(start-1);e_cs.append(cs);e_y.append(r[start]);e_q.append(qid);e_row.append(row_id)
            else:
                late_c.append(c[start-maxlen+1:start]);late_r.append(r[start-maxlen+1:start]);l_cs.append(cs);l_y.append(r[start]);l_q.append(qid);l_row.append(row_id)
    def pad(rows):
        result=np.full((len(rows),max_c),-1,dtype=np.int64)
        for i,row in enumerate(rows):result[i,:len(row)]=row
        return result
    return {'early_c':np.asarray(early_c),'early_r':np.asarray(early_r),'owners':np.asarray(owners,dtype=int),'times':np.asarray(times,dtype=int),
            'early_cs':pad(e_cs),'early_y':np.asarray(e_y,dtype=int),'early_q':np.asarray(e_q,dtype=int),'early_row':np.asarray(e_row,dtype=int),
            'late_c':np.asarray(late_c).reshape(-1,maxlen-1),'late_r':np.asarray(late_r).reshape(-1,maxlen-1),'late_cs':pad(l_cs),
            'late_y':np.asarray(l_y,dtype=int),'late_q':np.asarray(l_q,dtype=int),'late_row':np.asarray(l_row,dtype=int)}


def score_windows(model,windows,batch_size=512,return_frame=False):
    model.eval();device=next(model.parameters()).device;preds=[];ys=[];qs=[];rows=[]
    def means(output,concepts):
        cs=torch.as_tensor(concepts,device=device);mask=cs>=0
        chosen=output.gather(-1,cs.clamp_min(0)).double()
        values=(chosen*mask).sum(-1)/mask.sum(-1)
        # Public late_fusion rounds the group probability to four decimals.
        return np.asarray([round(float(x),4) for x in values.cpu().numpy()])
    with torch.no_grad():
        for start in range(0,len(windows['early_c']),batch_size):
            end=start+batch_size
            c=torch.as_tensor(windows['early_c'][start:end],device=device).long();r=torch.as_tensor(windows['early_r'][start:end],device=device).long()
            y=model(c,r);sel=(windows['owners']>=start)&(windows['owners']<end)
            output=y[torch.as_tensor(windows['owners'][sel]-start,device=device),torch.as_tensor(windows['times'][sel],device=device)]
            preds.extend(means(output,windows['early_cs'][sel]));ys.extend(windows['early_y'][sel]);qs.extend(windows['early_q'][sel]);rows.extend(windows['early_row'][sel])
        for start in range(0,len(windows['late_c']),batch_size):
            end=start+batch_size;c=torch.as_tensor(windows['late_c'][start:end],device=device).long();r=torch.as_tensor(windows['late_r'][start:end],device=device).long()
            output=model(c,r)[:,-1];preds.extend(means(output,windows['late_cs'][start:end]));ys.extend(windows['late_y'][start:end]);qs.extend(windows['late_q'][start:end]);rows.extend(windows['late_row'][start:end])
    y=np.asarray(ys);p=np.asarray(preds);frame=pd.DataFrame({'orirow':rows,'qidx':qs,'late_trues':y,'late_mean':p}).sort_values(['orirow','qidx']).reset_index(drop=True)
    result={'validation_windowauclate_mean':float(roc_auc_score(y,p)),'question_nll':float(log_loss(y,p,labels=[0,1])),'question_brier':float(np.mean((p-y)**2)),
            'questions':len(y),'students':len(windows['early_c']),'late_context_windows':len(windows['late_c'])}
    return (result,frame) if return_frame else result
