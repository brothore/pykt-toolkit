import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh

from .dkt import DKT


class DKTFirstAction(DKT):
    """DKT with an auxiliary first-action classification head.

    The response backbone, its parameter names and ``forward(c, r)`` are
    inherited from :class:`~pykt.models.dkt.DKT` unchanged.  The auxiliary
    head is only used by :meth:`forward_details`, :meth:`compute_loss` and
    :meth:`action_probabilities`; plain inference never touches it.

    Auxiliary classes:

    * 0: first attempt correct
    * 1: first attempt incorrect
    * 2: first action is a hint
    * 3: first action is a scaffold
    """

    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid",
                 emb_path="", pretrain_dim=768, action_weight=0.1,
                 action_mode="joint", action_balance=0.0,
                 action_class_counts=None, action_hidden=100):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path,
                         pretrain_dim)
        self.model_name = "dkt_first_action"

        if not isinstance(action_mode, str) or action_mode not in (
                "joint", "conditional"):
            raise ValueError(
                "action_mode must be 'joint' or 'conditional', got "
                f"{action_mode!r}")
        if not isinstance(action_hidden, int) or action_hidden <= 0:
            raise ValueError(
                f"action_hidden must be a positive int, got {action_hidden!r}")
        try:
            action_weight = float(action_weight)
            action_balance = float(action_balance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "action_weight and action_balance must be numeric") from exc
        if not math.isfinite(action_weight) or action_weight < 0:
            raise ValueError(
                f"action_weight must be finite and >= 0, got {action_weight!r}")
        if not math.isfinite(action_balance) or not 0 <= action_balance <= 1:
            raise ValueError(
                "action_balance must be finite and within [0, 1], got "
                f"{action_balance!r}")
        if self.emb_type != "qid":
            # The parent class only builds ``interaction_emb`` for qid and its
            # forward only implements the qid branch.
            raise ValueError(
                "DKTFirstAction requires emb_type='qid', got "
                f"{self.emb_type!r}")

        self.action_weight = action_weight
        self.action_mode = action_mode
        self.action_balance = action_balance
        self.action_hidden = action_hidden
        self.num_action_classes = 4 if action_mode == "joint" else 3

        # Initialise the auxiliary head under a forked RNG so that building it
        # does not advance the global random state left by the parent init.
        with torch.random.fork_rng(devices=[]):
            self.action_proj = Linear(2 * self.emb_size, self.action_hidden)
            self.action_act = Tanh()
            self.action_out = Linear(self.action_hidden,
                                     self.num_action_classes)

        action_weights = self._build_action_class_weights(action_class_counts)
        self.register_buffer("action_class_weights", action_weights)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_action_class_weights(self, action_class_counts):
        """Class weights from training-fold counts, normalised to mean 1.

        ``counts.clamp(min=1) ** (-action_balance)`` is computed on the classes
        actually used by the head (``counts[1:]`` for conditional mode) and then
        normalised to mean 1.  Missing/empty counts give uniform weights.
        """
        if action_class_counts is None:
            counts = torch.empty(0, dtype=torch.float32)
        elif torch.is_tensor(action_class_counts):
            counts = action_class_counts.detach().to(
                device="cpu", dtype=torch.float32).flatten()
        else:
            counts = torch.as_tensor(
                action_class_counts, dtype=torch.float32).flatten()

        if counts.numel() == 0:
            counts = torch.ones(4, dtype=torch.float32)
        elif counts.numel() != 4:
            raise ValueError(
                "action_class_counts must have length 4 (or be empty), got "
                f"{counts.numel()}")
        if not torch.isfinite(counts).all():
            raise ValueError("action_class_counts must be finite")
        if (counts < 0).any():
            raise ValueError("action_class_counts must be non-negative")

        weights = counts.clamp(min=1.0).pow(-self.action_balance)
        if self.action_mode == "conditional":
            weights = weights[1:]
        weights = weights / weights.mean()
        return weights

    # ------------------------------------------------------------------
    # forward / auxiliary head
    # ------------------------------------------------------------------
    def _selected_predictions(self, predictions, target_c):
        """Gather the probability of the target knowledge concept."""
        target_c = target_c.long().clamp(0, self.num_c - 1)
        return predictions.gather(-1, target_c.unsqueeze(-1)).squeeze(-1)

    def forward_details(self, c, r, target_c):
        """Return all-concept predictions and auxiliary action logits.

        The backbone computation is identical to ``DKT.forward``.  ``target_c``
        is the already-known knowledge concept of the next event; no unknown
        response or action information is read.
        """
        if self.emb_type != "qid":
            raise RuntimeError(
                "DKTFirstAction only implements the qid embedding branch")
        x = c + self.num_c * r
        xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        predictions = torch.sigmoid(y)

        # The two response channels of the target concept in the original
        # interaction embedding, averaged.
        target_idx = target_c.long().clamp(0, self.num_c - 1)
        target_emb = self.interaction_emb(target_idx)
        target_emb = target_emb + self.interaction_emb(target_idx + self.num_c)
        target_emb = target_emb * 0.5

        # No dropout inside the auxiliary head.
        action_input = torch.cat((h, target_emb), dim=-1)
        action_hidden = self.action_act(self.action_proj(action_input))
        action_logits = self.action_out(action_hidden)

        return {
            "predictions": predictions,
            "action_logits": action_logits,
        }

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def _response_loss(self, predictions, target, sm, return_per_sample):
        if predictions.shape != target.shape:
            raise ValueError(
                "predictions and response targets must have the same shape, "
                f"got {tuple(predictions.shape)} and {tuple(target.shape)}")
        if return_per_sample:
            losses = F.binary_cross_entropy(
                predictions.double(), target.double(), reduction="none")
            losses = losses * sm.to(losses.dtype)
            denom = sm.sum(dim=1).clamp(min=1).to(losses.dtype)
            return losses.sum(dim=1) / denom

        if int(sm.sum().item()) == 0:
            # Stay connected to the graph and avoid NaN on an empty mask.
            return predictions.sum().double() * 0.0
        y = torch.masked_select(predictions, sm)
        t = torch.masked_select(target, sm)
        return F.binary_cross_entropy(y.double(), t.double())

    def _action_loss(self, action_logits, action_labels, eligible,
                     return_per_sample):
        batch_size, seq_len, num_classes = action_logits.shape
        expected_classes = 4 if self.action_mode == "joint" else 3
        if num_classes != expected_classes:
            raise RuntimeError(
                f"action_logits has {num_classes} classes, expected "
                f"{expected_classes} for action_mode={self.action_mode!r}")
        if action_labels.shape != (batch_size, seq_len):
            raise ValueError(
                "action_labels must have shape [B, T] = "
                f"{(batch_size, seq_len)}, got {tuple(action_labels.shape)}")
        if eligible.shape != (batch_size, seq_len):
            raise ValueError(
                "action_mask must have shape [B, T] = "
                f"{(batch_size, seq_len)}, got {tuple(eligible.shape)}")

        legal = ((action_labels == -100) |
                 ((action_labels >= 0) & (action_labels <= 3)))
        if not bool(legal.all().item()):
            illegal = torch.unique(action_labels[~legal]).detach().cpu()
            raise ValueError(
                "action_labels may only contain 0..3 or -100, found "
                f"{illegal.tolist()}")

        eligible = eligible & (action_labels != -100)
        if self.action_mode == "joint":
            target = action_labels
            mask = eligible
        else:
            # Conditional mode only predicts classes 1..3; class 0 (first
            # attempt correct) is already covered by the main task.
            target = action_labels - 1
            mask = eligible & (action_labels > 0)

        target = target.clamp(0, num_classes - 1).long()
        losses = F.cross_entropy(
            action_logits.reshape(-1, num_classes),
            target.reshape(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        weight = self.action_class_weights.to(
            device=losses.device, dtype=losses.dtype)
        per_position_weight = weight[target] * mask.to(losses.dtype)
        weighted_losses = losses * per_position_weight

        if return_per_sample:
            denom = per_position_weight.sum(dim=1)
            safe_denom = torch.where(
                denom > 0, denom, torch.ones_like(denom))
            return weighted_losses.sum(dim=1) / safe_denom

        denom = per_position_weight.sum()
        if float(denom.item()) <= 0:
            # Stay connected to the graph and avoid NaN on an empty mask.
            return action_logits.sum() * 0.0
        return weighted_losses.sum() / denom

    def compute_loss(self, data, return_per_sample=False,
                     return_components=False):
        """Compute the response loss plus the auxiliary action loss.

        ``data`` is a normal KTDataset batch with ``cseqs``, ``rseqs``,
        ``shft_cseqs``, ``shft_rseqs``, ``masks`` and ``smasks``.  Action
        fields (``action_labels`` and ``action_mask``) are only required when
        ``action_weight > 0``.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        device = next(self.parameters()).device

        def get_tensor(key, dtype=None):
            if key not in data:
                raise KeyError(f"data is missing required field {key!r}")
            value = data[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            value = value.to(device)
            if dtype is not None:
                value = value.to(dtype)
            return value

        c = get_tensor("cseqs", torch.long)
        r = get_tensor("rseqs", torch.long)
        cshft = get_tensor("shft_cseqs", torch.long)
        target = get_tensor("shft_rseqs")
        masks = get_tensor("masks", torch.bool)
        smasks = get_tensor("smasks", torch.bool)
        sm = masks & smasks

        if target.dim() == c.dim() + 1 and target.size(-1) == 1:
            target = target.squeeze(-1)

        if self.action_weight == 0:
            predictions = self(c, r)
            selected = self._selected_predictions(predictions, cshft)
            main_loss = self._response_loss(
                selected, target, sm, return_per_sample)
            if return_components:
                aux_loss = torch.zeros_like(main_loss)
                return main_loss, {"response": main_loss, "action": aux_loss}
            return main_loss

        action_labels = get_tensor("action_labels", torch.long)
        action_mask = get_tensor("action_mask", torch.bool)
        if action_labels.shape != sm.shape:
            raise ValueError(
                "action_labels must have shape [B, T] = "
                f"{tuple(sm.shape)}, got {tuple(action_labels.shape)}")
        if action_mask.shape != sm.shape:
            raise ValueError(
                "action_mask must have shape [B, T] = "
                f"{tuple(sm.shape)}, got {tuple(action_mask.shape)}")

        details = self.forward_details(c, r, cshft)
        selected = self._selected_predictions(details["predictions"], cshft)
        main_loss = self._response_loss(
            selected, target, sm, return_per_sample)
        aux_loss = self._action_loss(
            details["action_logits"], action_labels, action_mask & sm,
            return_per_sample)

        total_loss = main_loss + self.action_weight * aux_loss
        if return_components:
            return total_loss, {
                "response": main_loss,
                "action": aux_loss,
            }
        return total_loss

    # ------------------------------------------------------------------
    # auxiliary probabilities (for validation)
    # ------------------------------------------------------------------
    def action_probabilities(self, data):
        """Return ``[B, T, 4]`` class probabilities for the first action.

        Only the observable ``cseqs``/``rseqs`` history and the known
        ``shft_cseqs`` are used.  In conditional mode the probability of class
        0 is the main-task correctness probability and the remaining mass is
        split by the 3-class conditional head.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict-like KTDataset batch")
        device = next(self.parameters()).device

        def get_tensor(key, dtype=torch.long):
            if key not in data:
                raise KeyError(f"data is missing required field {key!r}")
            value = data[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return value.to(device).to(dtype)

        c = get_tensor("cseqs")
        r = get_tensor("rseqs")
        target_c = get_tensor("shft_cseqs")

        details = self.forward_details(c, r, target_c)
        action_logits = details["action_logits"]
        if self.action_mode == "joint":
            return F.softmax(action_logits, dim=-1)

        p_correct = self._selected_predictions(
            details["predictions"], target_c)
        p_rest = F.softmax(action_logits, dim=-1)
        return torch.cat((
            p_correct.unsqueeze(-1),
            (1.0 - p_correct).unsqueeze(-1) * p_rest,
        ), dim=-1)

    def export_dkt_state_dict(self):
        """Drop the training-only head for loading with the original DKT class."""
        prefixes = ('interaction_emb.', 'lstm_layer.', 'out_layer.')
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(prefixes)}
