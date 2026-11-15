"""Attach stable fragment IDs without changing input tensors or sample order."""
from torch.utils.data import Dataset


class IndexedQueDataset(Dataset):
    def __init__(self,base):
        self.base=base;self.dori=base.dori;self.aug_probs=getattr(base,'aug_probs',{})
        self._qikt_fragment_indices=True

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        result=dict(self.base[index]);result['_fragment_index']=int(index)
        return result
