"""Keep every supplied fragment of a UID in the same training batch."""
import math
import torch


class CompleteUIDBatchSampler:
    def __init__(self, dataset, uids_per_batch=16):
        if not isinstance(uids_per_batch,int) or uids_per_batch<1:
            raise ValueError('uids_per_batch must be positive')
        groups={}
        for index,uid in enumerate(dataset.dori['uid']): groups.setdefault(int(uid),[]).append(index)
        if not groups: raise ValueError('No training UIDs')
        self.groups=list(groups.values());self.uids_per_batch=uids_per_batch

    def __len__(self): return math.ceil(len(self.groups)/self.uids_per_batch)

    def __iter__(self):
        order=torch.randperm(len(self.groups)).tolist()
        for start in range(0,len(order),self.uids_per_batch):
            yield [i for g in order[start:start+self.uids_per_batch] for i in self.groups[g]]
