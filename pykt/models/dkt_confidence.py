import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh

from .dkt import DKT


class DKTConfidence(DKT):
    """DKT with a training-only self-reported confidence head.

    ``forward(c, r)`` is inherited unchanged. Confidence labels are 0..4 (or
    -100); conditional mode models incorrect/correct branches separately.
    """

    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid",
                 emb_path="", pretrain_dim=768, confidence_weight=0.1,
                 confidence_mode="categorical", confidence_balance=0.0,
                 confidence_counts=None, confidence_hidden=100):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path,
                         pretrain_dim)
        self.model_name = "dkt_confidence"
        try:
            confidence_weight = float(confidence_weight)
            confidence_balance = float(confidence_balance)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_weight and confidence_balance must be numeric") from exc
        if not math.isfinite(confidence_weight) or confidence_weight < 0:
            raise ValueError(f"confidence_weight must be finite and >= 0, got {confidence_weight!r}")
        if not math.isfinite(confidence_balance) or not 0 <= confidence_balance <= 1:
            raise ValueError(f"confidence_balance must be finite and within [0, 1], got {confidence_balance!r}")
        if not isinstance(confidence_mode, str) or confidence_mode not in ("categorical", "ordinal", "conditional"):
            raise ValueError(f"confidence_mode must be 'categorical', 'ordinal' or 'conditional', got {confidence_mode!r}")
        if not isinstance(confidence_hidden, int) or isinstance(confidence_hidden, bool) or confidence_hidden <= 0:
            raise ValueError(f"confidence_hidden must be a positive int, got {confidence_hidden!r}")
        if self.emb_type != "qid":
            raise ValueError(f"DKTConfidence requires emb_type='qid', got {self.emb_type!r}")
        self.confidence_weight = confidence_weight
        self.confidence_mode = confidence_mode
        self.confidence_balance = confidence_balance
        self.confidence_hidden = confidence_hidden
        self.num_confidence_classes = 5

        if confidence_counts is None:
            counts = torch.ones(5)
        elif torch.is_tensor(confidence_counts):
            counts = confidence_counts.detach().cpu().float().flatten()
        else:
            counts = torch.as_tensor(confidence_counts, dtype=torch.float32).flatten()
        if counts.numel() != 5:
            raise ValueError(f"confidence_counts must have length 5, got {counts.numel()}")
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("confidence_counts must be finite and non-negative")
        weights = counts.clamp_min(1.0).pow(-confidence_balance)
        self.register_buffer("confidence_class_weights", weights / weights.mean())

        # Fork the RNG so the auxiliary head does not advance the global state.
        out_dim = 10 if confidence_mode == "conditional" else 5
        with torch.random.fork_rng(devices=[]):
            self.confidence_proj = Linear(2 * self.emb_size, confidence_hidden)
            self.confidence_act = Tanh()
            self.confidence_out = Linear(confidence_hidden, out_dim)

    def _selected_predictions(self, predictions, target_c):
        target_c = target_c.long().clamp(0, self.num_c - 1)
        return predictions.gather(-1, target_c.unsqueeze(-1)).squeeze(-1)

    def forward_details(self, c, r, target_c):
        """Return main predictions and confidence logits.

        Only the response history and the known target concept are used; no
        future response or confidence value is read.
        """
        xemb = self.interaction_emb(c + self.num_c * r)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        predictions = torch.sigmoid(self.out_layer(h))

        target_idx = target_c.long().clamp(0, self.num_c - 1)
        target_emb = (self.interaction_emb(target_idx) + self.interaction_emb(target_idx + self.num_c)) * 0.5
        confidence_hidden = self.confidence_act(self.confidence_proj(torch.cat((h, target_emb), dim=-1)))
        confidence_logits = self.confidence_out(confidence_hidden)
        if self.confidence_mode == "conditional":
            confidence_logits = confidence_logits.view(*confidence_logits.shape[:-1], 2, 5)
        return {"predictions": predictions, "confidence_logits": confidence_logits}

    def _get_tensor(self, data, key, dtype=None):
        if key not in data:
            raise KeyError(f"data is missing required field {key!r}")
        value = data[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(next(self.parameters()).device)
        return value if dtype is None else value.to(dtype)

    def _response_loss(self, predictions, target, sm, return_per_sample):
        if predictions.shape != target.shape:
            raise ValueError(f"predictions and response targets must have the same shape, got {tuple(predictions.shape)} and {tuple(target.shape)}")
        if return_per_sample:
            losses = F.binary_cross_entropy(predictions.double(), target.double(), reduction="none")
            losses = losses * sm.to(losses.dtype)
            return losses.sum(dim=1) / sm.sum(dim=1).clamp(min=1).to(losses.dtype)
        if int(sm.sum().item()) == 0:
            return predictions.sum().double() * 0.0
        return F.binary_cross_entropy(torch.masked_select(predictions, sm).double(), torch.masked_select(target, sm).double())

    def _confidence_loss(self, logits, labels, response, eligible, return_per_sample):
        batch, seq = labels.shape
        if labels.shape != eligible.shape:
            raise ValueError(f"confidence_mask must have the same shape as confidence_labels {tuple(labels.shape)}, got {tuple(eligible.shape)}")
        expected = (batch, seq, 2, 5) if self.confidence_mode == "conditional" else (batch, seq, 5)
        if tuple(logits.shape) != expected:
            raise RuntimeError(f"confidence_logits has shape {tuple(logits.shape)}, expected {expected} for confidence_mode={self.confidence_mode!r}")

        safe_labels = labels.clamp(0, 4)
        class_weights = self.confidence_class_weights.to(device=logits.device, dtype=logits.dtype)
        if self.confidence_mode == "conditional":
            branch = response.long().clamp(0, 1)
            index = branch.view(batch, seq, 1, 1).expand(-1, -1, 1, 5)
            branch_logits = logits.gather(2, index).squeeze(2)
            losses = F.cross_entropy(branch_logits.reshape(-1, 5), safe_labels.reshape(-1), reduction="none").view(batch, seq)
        elif self.confidence_mode == "categorical":
            losses = F.cross_entropy(logits.reshape(-1, 5), safe_labels.reshape(-1), reduction="none").view(batch, seq)
        else:
            probs = F.softmax(logits, dim=-1)
            survival = (1.0 - probs.cumsum(dim=-1)[..., :4]).clamp(1e-6, 1.0 - 1e-6)
            thresholds = torch.arange(4, device=labels.device)
            targets = (safe_labels.unsqueeze(-1) > thresholds).to(survival.dtype)
            losses = F.binary_cross_entropy(survival, targets, reduction="none").mean(dim=-1)

        per_position_weight = class_weights[safe_labels] * eligible.to(logits.dtype)
        weighted_losses = losses * per_position_weight
        if return_per_sample:
            denom = per_position_weight.sum(dim=1)
            return weighted_losses.sum(dim=1) / torch.where(denom > 0, denom, torch.ones_like(denom))
        denom = per_position_weight.sum()
        if float(denom.detach().item()) <= 0:
            return logits.sum() * 0.0
        return weighted_losses.sum() / denom

    def compute_loss(self, data, return_per_sample=False, return_components=False):
        """Compute the masked response loss plus the confidence loss."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        c = self._get_tensor(data, "cseqs", torch.long)
        r = self._get_tensor(data, "rseqs", torch.long)
        cshft = self._get_tensor(data, "shft_cseqs", torch.long)
        target = self._get_tensor(data, "shft_rseqs")
        masks = self._get_tensor(data, "masks", torch.bool)
        smasks = self._get_tensor(data, "smasks", torch.bool)
        sm = masks & smasks
        if target.dim() == c.dim() + 1 and target.size(-1) == 1:
            target = target.squeeze(-1)

        if self.confidence_weight == 0:
            predictions = self(c, r)
            selected = self._selected_predictions(predictions, cshft)
            response_loss = self._response_loss(selected, target, sm, return_per_sample)
            if return_components:
                return response_loss, {"response": response_loss, "confidence": torch.zeros_like(response_loss)}
            return response_loss

        labels = self._get_tensor(data, "confidence_labels")
        confidence_mask = self._get_tensor(data, "confidence_mask", torch.bool)
        if labels.shape != sm.shape or confidence_mask.shape != sm.shape:
            raise ValueError(f"confidence_labels and confidence_mask must both have shape [B, T] = {tuple(sm.shape)}, got {tuple(labels.shape)} and {tuple(confidence_mask.shape)}")
        legal = torch.isfinite(labels) & (labels >= 0) & (labels <= 4)
        if labels.is_floating_point():
            legal &= labels == labels.floor()
        active = confidence_mask & sm & (labels != -100)
        invalid = active & ~legal
        if bool(invalid.any().item()):
            found = torch.unique(labels[invalid]).detach().cpu().tolist()
            raise ValueError(f"confidence_labels may only contain 0..4 or -100, found {found}")
        eligible = active
        # Sanitize masked NaNs and sentinels before casting/indexing.
        labels = torch.where(eligible, labels, torch.zeros_like(labels)).long()

        details = self.forward_details(c, r, cshft)
        selected = self._selected_predictions(details["predictions"], cshft)
        response_loss = self._response_loss(selected, target, sm, return_per_sample)
        confidence_loss = self._confidence_loss(details["confidence_logits"], labels, target.long(), eligible, return_per_sample)
        total_loss = response_loss + self.confidence_weight * confidence_loss
        if return_components:
            return total_loss, {"response": response_loss, "confidence": confidence_loss}
        return total_loss

    def confidence_probabilities(self, data):
        """Return confidence and joint correctness/confidence probabilities.

        Only ``cseqs``, ``rseqs`` and ``shft_cseqs`` are read. joint_probs is
        [B, T, 2, 5] with the incorrect-response channel first.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        c = self._get_tensor(data, "cseqs", torch.long)
        r = self._get_tensor(data, "rseqs", torch.long)
        target_c = self._get_tensor(data, "shft_cseqs", torch.long)
        details = self.forward_details(c, r, target_c)
        p_correct = self._selected_predictions(details["predictions"], target_c)
        logits = details["confidence_logits"]
        if self.confidence_mode != "conditional":
            confidence_probs = F.softmax(logits, dim=-1)
            joint_probs = torch.stack(((1.0 - p_correct).unsqueeze(-1) * confidence_probs, p_correct.unsqueeze(-1) * confidence_probs), dim=2)
        else:
            branch_probs = F.softmax(logits, dim=-1)
            joint_probs = torch.stack(((1.0 - p_correct).unsqueeze(-1) * branch_probs[..., 0, :], p_correct.unsqueeze(-1) * branch_probs[..., 1, :]), dim=2)
            confidence_probs = joint_probs.sum(dim=2)
        return {"confidence_probs": confidence_probs, "joint_probs": joint_probs}

    def export_dkt_state_dict(self):
        """Drop the training-only confidence head for plain DKT loading."""
        prefixes = ("interaction_emb.", "lstm_layer.", "out_layer.")
        return {name: value for name, value in self.state_dict().items() if name.startswith(prefixes)}
