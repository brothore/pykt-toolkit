"""Privileged-history teachers and a parameter-identical distilled DKT."""
import math
import torch
import torch.nn.functional as F
from torch.nn import Linear
from .dkt import DKT


class RichHistoryTeacher(DKT):
    def __init__(self, num_c, emb_size, dropout=.1, feature_dim=11, emb_type='qid', emb_path='', pretrain_dim=768):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path, pretrain_dim)
        if emb_type != 'qid' or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError('Teacher requires qid and positive feature_dim')
        self.model_name = 'dkt_history_teacher'; self.feature_dim = feature_dim
        with torch.random.fork_rng(devices=[]):
            self.history_proj = Linear(feature_dim, emb_size, bias=False)

    def forward(self, c, r, history_process):
        if c.shape != r.shape or history_process.shape != (*c.shape, self.feature_dim):
            raise ValueError('Historical process features must align with current history c/r')
        x = self.interaction_emb(c + self.num_c*r) + self.history_proj(history_process.float())
        h, _ = self.lstm_layer(x)
        return torch.sigmoid(self.out_layer(self.dropout_layer(h)))


class DKTdistill(DKT):
    def __init__(self, num_c, emb_size, dropout=.1, emb_type='qid', emb_path='', pretrain_dim=768,
                 kd_weight=.2, kd_temperature=1., kd_mode='uniform'):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path, pretrain_dim)
        kd_weight, kd_temperature = float(kd_weight), float(kd_temperature)
        if not math.isfinite(kd_weight) or kd_weight < 0: raise ValueError('kd_weight must be finite >=0')
        if not math.isfinite(kd_temperature) or kd_temperature <= 0: raise ValueError('kd_temperature must be finite >0')
        if kd_mode not in ('uniform', 'confidence'): raise ValueError('Unknown kd_mode')
        if emb_type != 'qid': raise ValueError('DKTdistill requires qid')
        self.model_name = 'dkt_distill'
        self.kd_weight, self.kd_temperature, self.kd_mode = kd_weight, kd_temperature, kd_mode

    def compute_loss(self, data, return_per_sample=False, return_components=False):
        device = next(self.parameters()).device
        def get(key, dtype=None):
            value = data[key].to(device)
            return value if dtype is None else value.to(dtype)
        c, r = get('cseqs', torch.long), get('rseqs', torch.long)
        target_c = get('shft_cseqs', torch.long)
        target = get('shft_rseqs')
        mask = get('masks', torch.bool) & get('smasks', torch.bool)
        p = self(c, r).gather(-1, target_c.unsqueeze(-1)).squeeze(-1)
        if return_per_sample:
            bce = F.binary_cross_entropy(p.double(), target.double(), reduction='none')
            main = (bce*mask).sum(1)/mask.sum(1).clamp_min(1)
        elif mask.any():
            main = F.binary_cross_entropy(p[mask].double(), target[mask].double())
        else:
            main = p.sum().double()*0
        if self.kd_weight == 0:
            kd = torch.zeros_like(main)
        else:
            q = get('teacher_probs').detach()
            eligible = get('teacher_mask', torch.bool) & mask
            if q.shape != p.shape or eligible.shape != p.shape: raise ValueError('Teacher target shapes must match [B,T]')
            legal = torch.isfinite(q) & (q >= 0) & (q <= 1)
            if (eligible & ~legal).any(): raise ValueError('Active teacher_probs must be finite within [0,1]')
            q = torch.where(eligible, q, torch.full_like(q, .5)).double()
            pp = p.double().clamp(1e-6, 1-1e-6); qq = q.clamp(1e-6, 1-1e-6)
            temperature = self.kd_temperature
            pp = torch.sigmoid(torch.logit(pp)/temperature).clamp(1e-12,1-1e-12)
            qt = torch.sigmoid(torch.logit(qq)/temperature).clamp(1e-12,1-1e-12)
            losses = (qt*(qt.log()-pp.log()) + (1-qt)*((1-qt).log()-(1-pp).log()))*temperature**2
            weights = eligible.double()
            if self.kd_mode == 'confidence': weights = weights*(2*(q-.5).abs())
            if return_per_sample:
                denominator = weights.sum(1)
                kd = (losses*weights).sum(1)/denominator.clamp_min(1e-12)
            else:
                kd = (losses*weights).sum()/weights.sum().clamp_min(1e-12)
        total = main + self.kd_weight*kd
        return (total, {'response':main, 'distillation':kd}) if return_components else total

    def export_dkt_state_dict(self):
        return self.state_dict()
