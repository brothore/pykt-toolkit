import math

import torch
import torch.nn.functional as F

from .dkt import DKT


class DKTTransfer(DKT):
    """DKT with an optional future-question supervision loss."""

    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid",
                 emb_path="", pretrain_dim=768, transfer_weight=0.1,
                 transfer_loss="bce"):
        try:
            weight = float(transfer_weight)
        except (TypeError, ValueError):
            raise ValueError("transfer_weight must be a finite non-negative number")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("transfer_weight must be a finite non-negative number")
        if transfer_loss not in ("bce", "brier"):
            raise ValueError("transfer_loss must be 'bce' or 'brier'")
        if emb_type != "qid":
            raise ValueError("DKTTransfer only supports emb_type='qid'")

        super().__init__(num_c, emb_size, dropout=dropout, emb_type=emb_type,
                         emb_path=emb_path, pretrain_dim=pretrain_dim)
        self.transfer_weight = weight
        self.transfer_loss = transfer_loss
        self.model_name = "dkt_transfer"

    @staticmethod
    def _reduce(loss, mask, return_per_sample):
        mask = mask.to(dtype=loss.dtype)
        if return_per_sample:
            denom = mask.sum(dim=-1).clamp(min=1)
            return (loss * mask).sum(dim=-1) / denom
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def compute_loss(self, data, return_per_sample=False, return_components=False):
        device = next(self.parameters()).device
        def get(key, dtype=None):
            value = data[key].to(device)
            return value if dtype is None else value.to(dtype)
        c, r = get("cseqs", torch.long), get("rseqs", torch.long)
        y = self(c, r)
        shft_c = get("shft_cseqs", torch.long)
        pred = y.gather(-1, shft_c.unsqueeze(-1)).squeeze(-1)
        target = get("shft_rseqs")
        mask = get("masks", torch.bool) & get("smasks", torch.bool)
        if return_per_sample:
            per_pos = F.binary_cross_entropy(pred.double(), target.double(), reduction="none")
            main = self._reduce(per_pos, mask, True)
        elif mask.any():
            main = F.binary_cross_entropy(pred[mask].double(), target[mask].double())
        else:
            main = pred.sum().double() * 0
        if self.transfer_weight == 0:
            transfer = torch.zeros_like(main)
            return (main, {"response": main, "transfer": transfer}) if return_components else main
        future_c = get("future_concepts")
        future_l = get("future_labels")
        future_mask = get("future_mask", torch.bool)
        if future_c.ndim != 3 or future_c.shape[:2] != pred.shape or future_l.shape != pred.shape or future_mask.shape != pred.shape:
            raise ValueError("Future supervision shapes must be [B,T,K], [B,T], [B,T]")
        eligible = future_mask & mask
        finite = torch.isfinite(future_c)
        integral = future_c == future_c.floor() if future_c.is_floating_point() else torch.ones_like(finite)
        legal = finite & integral & ((future_c == -1) | ((future_c >= 0) & (future_c < self.num_c)))
        valid_concept = legal & (future_c != -1)
        if (eligible.unsqueeze(-1) & ~legal).any():
            raise ValueError("Invalid future concept at active position")
        if (eligible & ~valid_concept.any(-1)).any():
            raise ValueError("Active future target requires a concept")
        legal_label = torch.isfinite(future_l) & ((future_l == 0) | (future_l == 1))
        if (eligible & ~legal_label).any():
            raise ValueError("Active future_labels must be finite binary values")
        safe_c = torch.where(valid_concept, future_c, torch.zeros_like(future_c)).long()
        # Existing all-concept probabilities are queried at the CURRENT state;
        # no future interaction or label is inserted into the LSTM history.
        queried = y.gather(-1, safe_c)
        weights = valid_concept.to(queried.dtype)
        mean_p = (queried * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        mean_p = torch.where(eligible, mean_p, torch.full_like(mean_p, .5))
        safe_l = torch.where(eligible, future_l, torch.zeros_like(future_l))
        if self.transfer_loss == "bce":
            per_pos = F.binary_cross_entropy(mean_p.double(), safe_l.double(), reduction="none")
        else:
            per_pos = (mean_p.double() - safe_l.double()).square()
        transfer = self._reduce(per_pos, eligible, return_per_sample)
        total = main + self.transfer_weight * transfer
        return (total, {"response": main, "transfer": transfer}) if return_components else total

    def export_dkt_state_dict(self):
        return self.state_dict()
