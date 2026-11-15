"""Strict per-question causal reference windows, independent target-KC queries.

Current question responses never enter the input, including through an earlier
KC of the same question. Native reference context is <=200 expanded concepts.
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def gap_categories(concepts,timestamps):
    last={};count={};previous=None;values=[]
    for c,t in zip(concepts,timestamps):
        if previous is not None and t<previous:raise ValueError('Nonmonotone timestamps')
        rg=round(math.log2((t-last[c])/60000+1))+1 if c in last else 0
        sg=round(math.log2((t-previous)/60000+1))+1 if previous is not None else 0
        pc=round(math.log2(count.get(c,0)+1));values.append((rg,sg,pc));last[c]=t;previous=t;count[c]=count.get(c,0)+1
    return np.asarray(values,dtype=np.int64)


class CausalReferenceWindows(Dataset):
    def __init__(self,raw,length=200):
        if length<2:raise ValueError('Context must include history and target')
        self.length=length;self.rows=[];self.targets=[]
        if not raw.uid.is_unique:raise ValueError('Expected unique UIDs')
        for row in raw.itertuples():
            q=np.fromstring(row.questions,sep=',',dtype=np.int64);r=np.fromstring(row.responses,sep=',',dtype=np.int64);t=np.fromstring(row.timestamps,sep=',',dtype=np.int64);concepts=[list(map(int,c.split('_'))) for c in row.concepts.split(',')]
            if len(q)!=len(r) or len(r)!=len(t) or len(t)!=len(concepts) or not np.isin(r,[0,1]).all():raise ValueError('Invalid raw alignment')
            expanded=[];starts=[]
            for i,cs in enumerate(concepts):
                if not cs or min(cs)<0:raise ValueError('Missing target concepts')
                starts.append(len(expanded))
                for c in cs:expanded.append((q[i],c,r[i],t[i]))
            index=len(self.rows);self.rows.append({'uid':int(row.uid),'q':q,'r':r,'t':t,'c':concepts,'expanded':np.asarray(expanded),'starts':starts})
            for question in range(1,len(q)):
                for kc in concepts[question]:self.targets.append((index,question,kc))

    def __len__(self):return len(self.targets)

    def __getitem__(self,index):
        rowindex,position,kc=self.targets[index];row=self.rows[rowindex];stop=row['starts'][position];history=row['expanded'][max(0,stop-self.length+1):stop]
        # Append exactly one target KC, with a constant response placeholder.
        current=np.asarray([[row['q'][position],kc,0,row['t'][position]]]);values=np.concatenate([history,current]);size=len(values);padded=np.zeros((self.length,4),dtype=np.int64);padded[:size]=values
        data={}
        for col,key in [(0,'qseqs'),(1,'cseqs'),(2,'rseqs'),(3,'tseqs')]:
            v=torch.from_numpy(padded[:,col].copy());data[key]=v[:-1];data['shft_'+key]=v[1:]
        selected=torch.zeros(self.length-1,dtype=torch.bool);selected[size-2]=True;data['smasks']=selected;data['masks']=torch.arange(self.length-1)<size-1
        gaps=gap_categories(values[:,1],values[:,3]);gap_pad=np.zeros((self.length,3),dtype=np.int64);gap_pad[:size]=gaps
        outgaps={}
        for col,key in enumerate(['rgaps','sgaps','pcounts']):
            v=torch.from_numpy(gap_pad[:,col].copy());outgaps[key]=v[:-1];outgaps['shft_'+key]=v[1:]
        meta={'orirow':row['uid'],'qidx':position,'kc':kc,'target_index':size-1,'label':int(row['r'][position]),'question':int(row['q'][position]),'questions':','.join([str(row['q'][position])]*len(row['c'][position])),'concepts':','.join(map(str,reversed(row['c'][position])))}
        return data,outgaps,meta
