"""Aggregate future assessment outcomes without adding student parameters."""
import math
import torch
import torch.nn.functional as F
from .dkt import DKT


class DKTProfile(DKT):
    def __init__(self, num_c, emb_size, dropout=.1, emb_type='qid', emb_path='',
                 pretrain_dim=768, profile_weight=.1):
        weight = float(profile_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('profile_weight must be finite and nonnegative')
        if emb_type != 'qid':
            raise ValueError('DKTProfile requires qid')
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path, pretrain_dim)
        self.profile_weight = weight
        self.model_name = 'dkt_profile'

    def profile_predictions(self, output, data):
        device = output.device
        anchors = data['profile_anchors'].to(device)
        weights = data['profile_concepts'].to(device)
        mask = data['profile_mask'].to(device).bool()
        if anchors.ndim != 2 or mask.shape != anchors.shape or weights.shape != (*anchors.shape, self.num_c):
            raise ValueError('Invalid profile shapes')
        legal = torch.isfinite(anchors) & (anchors >= 0) & (anchors < output.shape[1])
        if anchors.is_floating_point():
            legal &= anchors == anchors.floor()
        valid_w = torch.isfinite(weights).all(-1) & (weights >= 0).all(-1)
        valid_w &= torch.isclose(weights.sum(-1), torch.ones_like(weights[...,0]), atol=1e-5)
        if (mask & ~(legal & valid_w)).any():
            raise ValueError('Invalid active anchor or exposure weights')
        safe_anchors = torch.where(mask, anchors, torch.zeros_like(anchors)).long()
        safe_weights = torch.where(mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        states = output.gather(1, safe_anchors.unsqueeze(-1).expand(-1,-1,self.num_c))
        predictions = (states * safe_weights).sum(-1)
        return torch.where(mask, predictions, torch.full_like(predictions,.5))

    def compute_loss(self, data, return_per_sample=False, return_components=False):
        device = next(self.parameters()).device
        output = self(data['cseqs'].to(device).long(), data['rseqs'].to(device).long())
        pred = output.gather(-1,data['shft_cseqs'].to(device).long().unsqueeze(-1)).squeeze(-1)
        target = data['shft_rseqs'].to(device)
        mask = data['masks'].to(device).bool() & data['smasks'].to(device).bool()
        def reduce(values, weights):
            if return_per_sample:
                return (values*weights).sum(-1)/weights.sum(-1).clamp_min(1)
            return (values*weights).sum()/weights.sum().clamp_min(1)
        if return_per_sample:
            main = reduce(F.binary_cross_entropy(pred.double(),target.double(),reduction='none'),mask)
        else:
            main = F.binary_cross_entropy(pred[mask].double(),target[mask].double()) if mask.any() else pred.sum().double()*0
        if self.profile_weight == 0:
            aux = torch.zeros_like(main)
        else:
            p = self.profile_predictions(output,data)
            m = data['profile_mask'].to(device).bool()
            labels = data['profile_labels'].to(device)
            mass = data['profile_mass'].to(device)
            if labels.shape != m.shape or mass.shape != m.shape:
                raise ValueError('Invalid profile labels/mass shape')
            if (m & (~torch.isfinite(labels) | (labels<0) | (labels>1) | ~torch.isfinite(mass) | (mass<=0))).any():
                raise ValueError('Invalid active profile target/mass')
            anchors = torch.where(m,data['profile_anchors'].to(device),0).long()
            if (m & ~mask.gather(1,anchors)).any():
                raise ValueError('Profile anchor must have a valid KT transition')
            labels = torch.where(m,labels,0)
            mass = torch.where(m,mass,0).double()
            aux = reduce(F.binary_cross_entropy(p.double(),labels.double(),reduction='none'),mass)
        loss = main + self.profile_weight*aux
        return (loss,{'response':main,'profile':aux}) if return_components else loss

    def export_dkt_state_dict(self):
        return self.state_dict()
