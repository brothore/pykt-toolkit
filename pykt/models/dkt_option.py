import math

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, Tanh

from .dkt import DKT


class DKTOption(DKT):
    """DKT with a training-only option (distractor) head.

    Option labels are 0..3, with -100 as the missing sentinel.  Unknown or
    unstable questions use key -1 and are excluded from the auxiliary loss.
    """

    def __init__(self, num_c, emb_size, num_q, dropout=0.1, emb_type="qid",
                 emb_path="", pretrain_dim=768, option_weight=0.1,
                 option_mode="direct", option_keys=None, option_prior=None,
                 option_qemb=64, option_hidden=100):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path,
                         pretrain_dim)
        self.model_name = "dkt_option"
        if not isinstance(num_q, int) or isinstance(num_q, bool) or num_q <= 0:
            raise ValueError(f"num_q must be a positive int, got {num_q!r}")
        if not isinstance(option_qemb, int) or isinstance(option_qemb, bool) or option_qemb <= 0:
            raise ValueError(f"option_qemb must be a positive int, got {option_qemb!r}")
        if not isinstance(option_hidden, int) or isinstance(option_hidden, bool) or option_hidden <= 0:
            raise ValueError(f"option_hidden must be a positive int, got {option_hidden!r}")
        try:
            option_weight = float(option_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("option_weight must be numeric") from exc
        if not math.isfinite(option_weight) or option_weight < 0:
            raise ValueError(f"option_weight must be finite and >= 0, got {option_weight!r}")
        if not isinstance(option_mode, str) or option_mode not in ("direct", "conditional", "prior", "binary_control"):
            raise ValueError(f"Unknown option_mode {option_mode!r}")
        if self.emb_type != "qid":
            raise ValueError(f"DKTOption requires emb_type='qid', got {self.emb_type!r}")

        self.num_q = num_q
        self.option_weight = option_weight
        self.option_mode = option_mode
        self.option_qemb = option_qemb
        self.option_hidden = option_hidden

        if option_keys is None:
            keys = torch.full((num_q,), -1, dtype=torch.long)
        else:
            keys = option_keys.detach().cpu() if torch.is_tensor(option_keys) else torch.as_tensor(option_keys)
            keys = keys.flatten()
            if keys.numel() != num_q:
                raise ValueError(f"option_keys must have length num_q={num_q}, got {keys.numel()}")
            if keys.is_floating_point():
                if not torch.isfinite(keys).all() or bool((keys != keys.floor()).any().item()):
                    raise ValueError("option_keys entries must be integral")
                keys = keys.long()
            elif keys.dtype == torch.bool:
                raise ValueError("option_keys entries must be integers in -1..3")
            else:
                keys = keys.long()
            if bool(((keys < -1) | (keys > 3)).any().item()):
                raise ValueError("option_keys entries must be -1 or in 0..3")
        self.register_buffer("option_keys", keys)

        if option_prior is None:
            prior = torch.ones(num_q, 4)
        else:
            prior = option_prior.detach().cpu().float() if torch.is_tensor(option_prior) else torch.as_tensor(option_prior, dtype=torch.float32)
            if tuple(prior.shape) != (num_q, 4):
                raise ValueError(f"option_prior must have shape [{num_q}, 4], got {tuple(prior.shape)}")
            if not torch.isfinite(prior).all() or bool((prior < 0).any().item()):
                raise ValueError("option_prior must be finite and non-negative")
        prior = prior.clamp_min(1e-6)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        self.register_buffer("option_log_prior", prior.log())

        # The auxiliary head must not advance the global RNG.
        with torch.random.fork_rng(devices=[]):
            self.option_question_emb = Embedding(num_q, option_qemb)
            self.option_proj = Linear(2 * self.emb_size + option_qemb, option_hidden)
            self.option_act = Tanh()
            self.option_out = Linear(option_hidden, 4)

    def _tensor(self, data, key, dtype=None):
        if key not in data:
            raise KeyError(f"data is missing required field {key!r}")
        value = data[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(next(self.parameters()).device)
        return value if dtype is None else value.to(dtype)

    def _squeeze(self, value):
        return value.squeeze(-1) if value.dim() > 2 and value.size(-1) == 1 else value

    def _selected(self, predictions, target_c):
        target_c = target_c.long().clamp(0, self.num_c - 1)
        return predictions.gather(-1, target_c.unsqueeze(-1)).squeeze(-1)

    def _lookup_keys(self, target_q):
        q = target_q.long()
        valid = (q >= 0) & (q < self.num_q)
        q = q.clamp(0, self.num_q - 1)
        keys = self.option_keys[q]
        return keys, valid & (keys >= 0), q

    def forward_details(self, c, r, target_c, target_q):
        """Return main predictions [B,T,C] and option logits [B,T,4]."""
        if c.shape != r.shape:
            raise ValueError(f"c and r must have the same shape, got {tuple(c.shape)} and {tuple(r.shape)}")
        xemb = self.interaction_emb(c + self.num_c * r)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        predictions = torch.sigmoid(self.out_layer(h))

        target_c = self._squeeze(target_c)
        target_q = self._squeeze(target_q)
        if target_c.shape != c.shape or target_q.shape != c.shape:
            raise ValueError("target_c and target_q must both have shape [B, T]")
        tc = target_c.long()
        tq = target_q.long()
        q_valid = (tq >= 0) & (tq < self.num_q)
        tc = tc.clamp(0, self.num_c - 1)
        tq = tq.clamp(0, self.num_q - 1)
        concept = (self.interaction_emb(tc) + self.interaction_emb(tc + self.num_c)) * 0.5
        question = self.option_question_emb(tq)
        hidden = self.option_act(self.option_proj(torch.cat((h, concept, question), dim=-1)))
        option_logits = self.option_out(hidden)
        if self.option_mode == "prior":
            prior = self.option_log_prior.to(option_logits.dtype)[tq]
            prior = torch.where(q_valid.unsqueeze(-1), prior, torch.zeros_like(prior))
            option_logits = option_logits + prior
        return predictions, option_logits

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

    def _option_loss(self, logits, labels, keys, known, eligible, return_per_sample):
        if tuple(logits.shape[:-1]) != tuple(labels.shape):
            raise ValueError(f"option_logits and option labels disagree: {tuple(logits.shape)} vs {tuple(labels.shape)}")
        safe_labels = labels
        if self.option_mode in ("conditional", "prior"):
            correct = F.one_hot(keys.clamp(0, 3), 4).bool() & known.unsqueeze(-1)
            # Inactive positions must not hit the -1e9 masked correct class.
            bad = (~eligible) & known & (safe_labels == keys)
            safe_labels = torch.where(bad, (keys + 1) % 4, safe_labels)
            logits = logits.masked_fill(correct, -1e9)
        losses = F.cross_entropy(logits.reshape(-1, 4), safe_labels.reshape(-1), reduction="none").view_as(safe_labels)
        weights = eligible.to(losses.dtype)
        if return_per_sample:
            denom = weights.sum(dim=1)
            return (losses * weights).sum(dim=1) / torch.where(denom > 0, denom, torch.ones_like(denom))
        denom = weights.sum()
        if float(denom.detach().item()) <= 0:
            return logits.sum() * 0.0
        return (losses * weights).sum() / denom

    def compute_loss(self, data, return_per_sample=False, return_components=False):
        """Masked response BCE plus the optional option-classification loss."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        c = self._tensor(data, "cseqs", torch.long)
        r = self._tensor(data, "rseqs", torch.long)
        cshft = self._squeeze(self._tensor(data, "shft_cseqs", torch.long))
        target = self._squeeze(self._tensor(data, "shft_rseqs"))
        masks = self._tensor(data, "masks", torch.bool)
        smasks = self._tensor(data, "smasks", torch.bool)
        sm = masks & smasks

        if self.option_weight == 0:
            predictions = self(c, r)
            response_loss = self._response_loss(self._selected(predictions, cshft), target, sm, return_per_sample)
            if return_components:
                return response_loss, {"response": response_loss, "option": torch.zeros_like(response_loss)}
            return response_loss

        target_q = self._squeeze(self._tensor(data, "shft_qseqs", torch.long))
        labels = self._squeeze(self._tensor(data, "option_labels"))
        option_mask = self._squeeze(self._tensor(data, "option_mask", torch.bool))
        if target_q.shape != sm.shape or labels.shape != sm.shape or option_mask.shape != sm.shape:
            raise ValueError(f"shft_qseqs, option_labels and option_mask must all have shape [B, T] = {tuple(sm.shape)}")
        keys, known, _ = self._lookup_keys(target_q)
        legal = (labels >= 0) & (labels <= 3)
        if labels.is_floating_point():
            legal = legal & torch.isfinite(labels) & (labels == labels.floor())
        active = option_mask & sm & (labels != -100)
        invalid = active & ~legal
        if bool(invalid.any().item()):
            found = torch.unique(labels[invalid].detach().cpu()).tolist()
            raise ValueError(f"option_labels may only contain 0..3 or -100, found {found}")
        safe_labels = torch.where(active, labels, torch.zeros_like(labels)).long()

        eligible = active & known
        if self.option_mode in ("conditional", "prior"):
            eligible = eligible & (target.long() == 0)
            if bool((eligible & (safe_labels == keys)).any().item()):
                raise ValueError("conditional/prior option labels must not equal the known correct option for wrong responses")

        predictions, option_logits = self.forward_details(c, r, cshft, target_q)
        response_loss = self._response_loss(self._selected(predictions, cshft), target, sm, return_per_sample)
        if self.option_mode == "binary_control":
            # Same head/metadata/parameters as direct, but no distractor identity
            # is supervised. This ablates the new fine-grained information.
            correct_probability = F.softmax(option_logits, dim=-1).gather(
                -1, keys.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            losses = F.binary_cross_entropy(correct_probability, target.float(), reduction="none")
            weights = eligible.to(losses.dtype)
            if return_per_sample:
                option_loss = (losses * weights).sum(1) / weights.sum(1).clamp_min(1)
            else:
                option_loss = (losses * weights).sum() / weights.sum().clamp_min(1)
        else:
            option_loss = self._option_loss(option_logits, safe_labels, keys, known, eligible, return_per_sample)
        total_loss = response_loss + self.option_weight * option_loss
        if return_components:
            return total_loss, {"response": response_loss, "option": option_loss}
        return total_loss

    def option_probabilities(self, data):
        """Return [B,T,4] option probabilities using only history and metadata.

        ``eligible_question_mask`` marks questions with a static known key.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        c = self._tensor(data, "cseqs", torch.long)
        r = self._tensor(data, "rseqs", torch.long)
        target_c = self._squeeze(self._tensor(data, "shft_cseqs", torch.long))
        target_q = self._squeeze(self._tensor(data, "shft_qseqs", torch.long))
        predictions, logits = self.forward_details(c, r, target_c, target_q)
        p_correct = self._selected(predictions, target_c)
        keys, known, _ = self._lookup_keys(target_q)

        if self.option_mode in ("direct", "binary_control"):
            option_probs = F.softmax(logits, dim=-1)
        else:
            correct = F.one_hot(keys.clamp(0, 3), 4).bool() & known.unsqueeze(-1)
            wrong_probs = F.softmax(logits.masked_fill(correct, -1e9), dim=-1)
            one_hot = F.one_hot(keys.clamp(0, 3), 4).to(wrong_probs.dtype)
            p = p_correct.unsqueeze(-1)
            blended = (1.0 - p) * wrong_probs + p * one_hot
            option_probs = torch.where(known.unsqueeze(-1), blended, wrong_probs)
        return {"option_probs": option_probs, "eligible_question_mask": known}

    def export_dkt_state_dict(self):
        """Keep only the original DKT interaction/LSTM/output parameters."""
        prefixes = ("interaction_emb.", "lstm_layer.", "out_layer.")
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(prefixes)}
