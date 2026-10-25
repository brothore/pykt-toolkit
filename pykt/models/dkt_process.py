import math

import torch
import torch.nn.functional as F
from torch.nn import Linear, Tanh

from .dkt import DKT


class DKTProcess(DKT):
    """DKT with training-only time / attempt / hint prediction heads.

    The response backbone, its parameter names and ``forward(c, r)`` are
    inherited from :class:`~pykt.models.dkt.DKT` unchanged.  The auxiliary
    heads are only used by :meth:`forward_details`, :meth:`compute_loss` and
    :meth:`process_predictions`; plain response inference never touches them.

    Auxiliary targets (all ``[B, T]`` at already de-duplicated event positions):

    * ``process_time``: raw (unstandardized) ``log1p`` response seconds.
    * ``process_attempt``: integer attempt category ``0..3``.
    * ``process_hint``: binary hint use indicator ``0/1``.

    In ``conditional`` mode attempts and hints are only supervised for target
    responses that are incorrect (``shft_rseqs == 0``); time is still
    supervised everywhere.  ``process_predictions`` therefore returns the
    conditional attempt/hint distributions when the model was trained in
    conditional mode.
    """

    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid",
                 emb_path="", pretrain_dim=768, time_weight=0.1,
                 attempt_weight=0.1, hint_weight=0.1, process_mode="joint",
                 process_balance=0.0, process_stats=None, process_hidden=100):
        super().__init__(num_c, emb_size, dropout, emb_type, emb_path,
                         pretrain_dim)
        self.model_name = "dkt_process"

        if not isinstance(process_mode, str) or process_mode not in (
                "joint", "conditional"):
            raise ValueError(
                "process_mode must be 'joint' or 'conditional', got "
                f"{process_mode!r}")
        if (not isinstance(process_hidden, int) or
                isinstance(process_hidden, bool) or process_hidden <= 0):
            raise ValueError(
                "process_hidden must be a positive int, got "
                f"{process_hidden!r}")

        self.time_weight = self._validate_weight("time_weight", time_weight)
        self.attempt_weight = self._validate_weight(
            "attempt_weight", attempt_weight)
        self.hint_weight = self._validate_weight("hint_weight", hint_weight)
        self.process_mode = process_mode
        self.process_balance = self._validate_balance(process_balance)
        self.process_hidden = process_hidden
        self.num_attempt_classes = 4
        self.num_hint_classes = 2

        if self.emb_type != "qid":
            # The parent only builds ``interaction_emb`` for qid, and the
            # target-concept embedding used here relies on that table.
            raise ValueError(
                "DKTProcess requires emb_type='qid', got "
                f"{self.emb_type!r}")

        stats = self._parse_process_stats(process_stats)
        # Keep the normalized training-fold statistics available to experiment
        # scripts (buffers below hold the tensors used by the losses).
        self.process_stats = {
            "time_center": stats["time_center"],
            "time_scale": stats["time_scale"],
            "attempt_counts": stats["attempt_counts"].tolist(),
            "hint_counts": stats["hint_counts"].tolist(),
        }
        self.register_buffer(
            "process_time_center",
            torch.tensor(stats["time_center"], dtype=torch.float32))
        self.register_buffer(
            "process_time_scale",
            torch.tensor(stats["time_scale"], dtype=torch.float32))
        self.register_buffer(
            "attempt_class_weights",
            self._build_class_weights(stats["attempt_counts"]))
        self.register_buffer(
            "hint_class_weights",
            self._build_class_weights(stats["hint_counts"]))

        # Initialise the auxiliary head under a forked RNG so that building it
        # does not advance the global random state left by the parent init.
        with torch.random.fork_rng(devices=[]):
            self.process_proj = Linear(2 * self.emb_size, self.process_hidden)
            self.process_act = Tanh()
            self.time_out = Linear(self.process_hidden, 1)
            self.attempt_out = Linear(self.process_hidden, 4)
            self.hint_out = Linear(self.process_hidden, 1)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_weight(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be finite and >= 0, got {value!r}")
        return value

    @staticmethod
    def _validate_balance(value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("process_balance must be numeric") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                "process_balance must be finite and within [0, 1], got "
                f"{value!r}")
        return value

    @staticmethod
    def _parse_process_stats(process_stats):
        if process_stats is None:
            stats = {}
        elif isinstance(process_stats, dict):
            stats = process_stats
        else:
            raise ValueError(
                "process_stats must be a dict or None, got "
                f"{type(process_stats).__name__}")

        try:
            time_center = float(stats.get("time_center", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("process_stats['time_center'] must be numeric") \
                from exc
        if not math.isfinite(time_center):
            raise ValueError(
                "process_stats['time_center'] must be finite, got "
                f"{time_center!r}")

        try:
            time_scale = float(stats.get("time_scale", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("process_stats['time_scale'] must be numeric") \
                from exc
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError(
                "process_stats['time_scale'] must be finite and > 0, got "
                f"{time_scale!r}")

        def counts(name, length):
            value = stats.get(name, [1] * length)
            try:
                tensor = torch.as_tensor(
                    value, dtype=torch.float32).flatten()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"process_stats[{name!r}] must be a length-{length} "
                    "sequence of numbers") from exc
            if tensor.numel() != length:
                raise ValueError(
                    f"process_stats[{name!r}] must have length {length}, got "
                    f"{tensor.numel()}")
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"process_stats[{name!r}] must be finite")
            if (tensor < 0).any():
                raise ValueError(
                    f"process_stats[{name!r}] must be non-negative")
            return tensor

        return {
            "time_center": time_center,
            "time_scale": time_scale,
            "attempt_counts": counts("attempt_counts", 4),
            "hint_counts": counts("hint_counts", 2),
        }

    def _build_class_weights(self, counts):
        """Inverse-frequency class weights normalised to mean one."""
        weights = counts.clamp(min=1.0).pow(-self.process_balance)
        return weights / weights.mean()

    # ------------------------------------------------------------------
    # forward / auxiliary head
    # ------------------------------------------------------------------
    def _selected_predictions(self, predictions, target_c):
        """Gather the probability of the target knowledge concept."""
        target_c = target_c.long().clamp(0, self.num_c - 1)
        return predictions.gather(-1, target_c.unsqueeze(-1)).squeeze(-1)

    def forward_details(self, c, r, target_c):
        """Return response predictions and auxiliary process outputs.

        The backbone computation is identical to ``DKT.forward``.  ``target_c``
        is the already-known knowledge concept of the next event; no unknown
        response or process target is read here.

        Returns a dict with ``predictions`` ``[B, T, C]``,
        ``time_prediction`` ``[B, T]`` in standardized log1p-seconds space,
        ``attempt_logits`` ``[B, T, 4]`` and ``hint_logits`` ``[B, T]``.
        """
        if self.emb_type != "qid":
            raise RuntimeError(
                "DKTProcess only implements the qid embedding branch")

        target_c = target_c.long()
        if target_c.dim() == c.dim() + 1 and target_c.size(-1) == 1:
            target_c = target_c.squeeze(-1)

        x = c + self.num_c * r
        xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        predictions = torch.sigmoid(y)

        # Mean of the two answer channels of the known target concept in the
        # original interaction embedding.
        target_idx = target_c.clamp(0, self.num_c - 1)
        target_emb = self.interaction_emb(target_idx)
        target_emb = target_emb + self.interaction_emb(
            target_idx + self.num_c)
        target_emb = target_emb * 0.5

        # No dropout inside the auxiliary head: h already has backbone dropout.
        shared = self.process_act(
            self.process_proj(torch.cat((h, target_emb), dim=-1)))
        time_prediction = self.time_out(shared).squeeze(-1)
        attempt_logits = self.attempt_out(shared)
        hint_logits = self.hint_out(shared).squeeze(-1)

        return {
            "predictions": predictions,
            "time_prediction": time_prediction,
            "attempt_logits": attempt_logits,
            "hint_logits": hint_logits,
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

    @staticmethod
    def _reduce_weighted(losses, weights, reference, return_per_sample):
        """Masked weighted mean with differentiable, NaN-free empty cases."""
        if return_per_sample:
            denom = weights.sum(dim=1)
            safe_denom = torch.where(
                denom > 0, denom, torch.ones_like(denom))
            return (losses * weights).sum(dim=1) / safe_denom

        denom = weights.sum()
        if float(denom.item()) <= 0:
            return reference.sum() * 0.0
        return (losses * weights).sum() / denom

    def _time_loss(self, time_prediction, labels, eligible,
                   return_per_sample):
        if time_prediction.shape != labels.shape:
            raise ValueError(
                "process_time must have shape [B, T] = "
                f"{tuple(time_prediction.shape)}, got {tuple(labels.shape)}")
        if eligible.shape != time_prediction.shape:
            raise ValueError(
                "process_time_mask must have shape [B, T] = "
                f"{tuple(time_prediction.shape)}, got {tuple(eligible.shape)}")

        eligible = eligible.bool()
        labels = labels.to(time_prediction.dtype)
        valid = torch.isfinite(labels) & (labels >= 0)
        invalid = eligible & ~valid
        if bool(invalid.any().item()):
            illegal = labels[invalid].detach().cpu().unique()
            raise ValueError(
                "process_time may only contain finite non-negative values at "
                f"active positions, found {illegal.tolist()}")

        # Sanitize *before* the loss so masked NaN / -100 sentinels cannot
        # contaminate the computation graph.
        safe_labels = torch.where(
            eligible, labels, torch.zeros_like(labels))
        center = self.process_time_center.to(
            device=time_prediction.device, dtype=time_prediction.dtype)
        scale = self.process_time_scale.to(
            device=time_prediction.device, dtype=time_prediction.dtype)
        standardized = (safe_labels - center) / scale
        losses = F.smooth_l1_loss(
            time_prediction, standardized, reduction="none")
        weights = eligible.to(losses.dtype)
        return self._reduce_weighted(
            losses, weights, time_prediction, return_per_sample)

    def _attempt_loss(self, attempt_logits, labels, eligible,
                      return_per_sample):
        if attempt_logits.dim() != 3 or attempt_logits.size(-1) != 4:
            raise RuntimeError(
                "attempt_logits must have shape [B, T, 4], got "
                f"{tuple(attempt_logits.shape)}")
        batch_size, seq_len, _ = attempt_logits.shape
        expected = (batch_size, seq_len)
        if labels.shape != expected:
            raise ValueError(
                "process_attempt must have shape [B, T] = "
                f"{expected}, got {tuple(labels.shape)}")
        if eligible.shape != expected:
            raise ValueError(
                "process_attempt_mask must have shape [B, T] = "
                f"{expected}, got {tuple(eligible.shape)}")

        eligible = eligible.bool()
        valid = (torch.isfinite(labels) & (labels >= 0) & (labels <= 3))
        if labels.is_floating_point():
            valid = valid & (labels == torch.floor(labels))
        invalid = eligible & ~valid
        if bool(invalid.any().item()):
            illegal = labels[invalid].detach().cpu().unique()
            raise ValueError(
                "process_attempt may only contain integers 0..3 at active "
                f"positions, found {illegal.tolist()}")

        safe_labels = torch.where(
            eligible, labels, torch.zeros_like(labels))
        target = safe_labels.long()
        losses = F.cross_entropy(
            attempt_logits.reshape(-1, 4),
            target.reshape(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        weight = self.attempt_class_weights.to(
            device=losses.device, dtype=losses.dtype)
        per_position_weight = weight[target] * eligible.to(losses.dtype)
        return self._reduce_weighted(
            losses, per_position_weight, attempt_logits, return_per_sample)

    def _hint_loss(self, hint_logits, labels, eligible, return_per_sample):
        if hint_logits.shape != labels.shape:
            raise ValueError(
                "process_hint must have shape [B, T] = "
                f"{tuple(hint_logits.shape)}, got {tuple(labels.shape)}")
        if eligible.shape != hint_logits.shape:
            raise ValueError(
                "process_hint_mask must have shape [B, T] = "
                f"{tuple(hint_logits.shape)}, got {tuple(eligible.shape)}")

        eligible = eligible.bool()
        labels = labels.to(hint_logits.dtype)
        valid = torch.isfinite(labels) & ((labels == 0) | (labels == 1))
        invalid = eligible & ~valid
        if bool(invalid.any().item()):
            illegal = labels[invalid].detach().cpu().unique()
            raise ValueError(
                "process_hint may only contain 0 or 1 at active positions, "
                f"found {illegal.tolist()}")

        safe_labels = torch.where(
            eligible, labels, torch.zeros_like(labels))
        losses = F.binary_cross_entropy_with_logits(
            hint_logits, safe_labels, reduction="none")
        weight = self.hint_class_weights.to(
            device=losses.device, dtype=losses.dtype)
        per_position_weight = weight[safe_labels.long()] * eligible.to(
            losses.dtype)
        return self._reduce_weighted(
            losses, per_position_weight, hint_logits, return_per_sample)

    def compute_loss(self, data, return_per_sample=False,
                     return_components=False):
        """Compute the response loss plus active auxiliary process losses.

        ``data`` is a normal KTDataset batch with ``cseqs``, ``rseqs``,
        ``shft_cseqs``, ``shft_rseqs``, ``masks`` and ``smasks``.  Process
        fields are only required for heads whose weight is greater than zero.
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

        if target.dim() == c.dim() + 1 and target.size(-1) == 1:
            target = target.squeeze(-1)
        sm = masks & smasks
        if sm.shape != c.shape:
            raise ValueError(
                "masks and smasks must have shape [B, T] = "
                f"{tuple(c.shape)}, got {tuple(sm.shape)}")

        # Fast path: no auxiliary supervision, so behave exactly like DKT.
        if (self.time_weight == 0 and self.attempt_weight == 0 and
                self.hint_weight == 0):
            predictions = self(c, r)
            selected = self._selected_predictions(predictions, cshft)
            response_loss = self._response_loss(
                selected, target, sm, return_per_sample)
            if return_components:
                zero = torch.zeros_like(response_loss)
                return response_loss, {
                    "response": response_loss,
                    "time": zero,
                    "attempt": zero,
                    "hint": zero,
                }
            return response_loss

        details = self.forward_details(c, r, cshft)
        selected = self._selected_predictions(details["predictions"], cshft)
        response_loss = self._response_loss(
            selected, target, sm, return_per_sample)

        zero = torch.zeros_like(response_loss)
        time_loss = zero
        attempt_loss = zero
        hint_loss = zero

        if self.time_weight > 0:
            time_labels = get_tensor("process_time", torch.float32)
            time_mask = get_tensor("process_time_mask", torch.bool)
            time_loss = self._time_loss(
                details["time_prediction"], time_labels, time_mask & sm,
                return_per_sample)

        if self.attempt_weight > 0:
            attempt_labels = get_tensor("process_attempt")
            attempt_mask = get_tensor("process_attempt_mask", torch.bool)
            attempt_eligible = attempt_mask & sm
            if self.process_mode == "conditional":
                attempt_eligible = attempt_eligible & (target == 0)
            attempt_loss = self._attempt_loss(
                details["attempt_logits"], attempt_labels, attempt_eligible,
                return_per_sample)

        if self.hint_weight > 0:
            hint_labels = get_tensor("process_hint", torch.float32)
            hint_mask = get_tensor("process_hint_mask", torch.bool)
            hint_eligible = hint_mask & sm
            if self.process_mode == "conditional":
                hint_eligible = hint_eligible & (target == 0)
            hint_loss = self._hint_loss(
                details["hint_logits"], hint_labels, hint_eligible,
                return_per_sample)

        total_loss = (response_loss +
                      self.time_weight * time_loss +
                      self.attempt_weight * attempt_loss +
                      self.hint_weight * hint_loss)
        if return_components:
            return total_loss, {
                "response": response_loss,
                "time": time_loss,
                "attempt": attempt_loss,
                "hint": hint_loss,
            }
        return total_loss

    # ------------------------------------------------------------------
    # auxiliary predictions (for validation)
    # ------------------------------------------------------------------
    def process_predictions(self, data):
        """Return process predictions from observable history only.

        Reads only ``cseqs``, ``rseqs`` and ``shft_cseqs``; no true response
        or process target is consulted.  ``time_prediction`` is unstandardized
        to raw log1p-seconds space.  In conditional mode the attempt and hint
        outputs are the conditional distributions trained on incorrect target
        responses.
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
        center = self.process_time_center.to(
            device=device, dtype=details["time_prediction"].dtype)
        scale = self.process_time_scale.to(
            device=device, dtype=details["time_prediction"].dtype)
        time_prediction = details["time_prediction"] * scale + center

        return {
            "time_prediction": time_prediction,
            "attempt_probs": F.softmax(details["attempt_logits"], dim=-1),
            "hint_prob": torch.sigmoid(details["hint_logits"]),
        }

    def export_dkt_state_dict(self):
        """Drop the training-only process heads for the original DKT class."""
        prefixes = ("interaction_emb.", "lstm_layer.", "out_layer.")
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(prefixes)}
