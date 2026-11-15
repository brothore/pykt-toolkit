"""Composable reference-mechanism transfers for AKT.

The default ``mechanism="baseline"`` delegates directly to
:class:`pykt.models.akt.AKT` and therefore reproduces the original model
exactly (including its RNG use and return structure).  Every other mechanism
keeps the AKT parameterisation and adds either an input-side reparameterisation
or a zero-initialised residual on the shared ``d_output``.  Zero-initialised
mechanisms are prediction-identical to AKT at initialisation for the same AKT
weights.

Source implementation attribution
---------------------------------
- AKT, original multiplicative Rasch/PID path and architecture:
  ``pykt/models/akt.py`` (Ghosh et al., "Context-Aware Attentive Knowledge
  Tracing for Predicting Student Performance", KDD 2020).
- UKT, positive covariance / Wasserstein attention and the ``scores ** 2``
  covariance aggregation: ``pykt/models/ukt.py`` (``wasserstein_distance_matmul``
  and ``uattention``).  The implementation here is a deliberately simplified
  transfer, not a reimplementation of the full UKT model: it has no
  contrastive loss, no stochastic reparameterisation, no ELU chain over the
  whole network and no UKT prediction head.  Only the covariance-distance
  history matching and alpha-squared covariance aggregation are reused.
- MTKT, the real Causal Interaction Convolution (``CausalConv1d`` / ``CIC``)
  and the ``timeGap`` bucket semantics: ``pykt/models/mtkt.py`` (the copy used
  for the reference study lives at
  ``/data/pykt-toolkit-latest/pykt/models/mtkt.py``).  ``mtkt_cic`` applies
  the genuine dual causal-convolution gated interaction, not an MLP.
- LEFOKT-AKT additive transfer and zero-output adapter conventions:
  ``pykt/models/qikt_reference_fusion.py`` and
  ``pykt/models/qikt_ukt_rasch.py`` (``RaschFactors`` mode
  ``input_scalar_additive``).  In particular, the additive form replaces a
  multiplicative difficulty modulation with ``pid + (delta - delta0)`` so the
  initial modulation is exactly zero, mirroring the earlier LEF additive
  transfer on the GRU branch.
- MTKT time: ``timeGap``/``timeGap2`` in the MTKT source operate on log bucket
  ids.  ``mtkt_time`` consumes pre-computed full-length ``B x L`` integer
  ``rgaps``/``sgaps``/``pcounts``, applies ``log1p`` and a zero-initialised
  MLP correction to ``d_output``.  It never reads the current response.

Design notes
------------
- ``lef_additive``: keeps every AKT PID/KC/response parameter.  The original
  multiplication ``pid * diff`` is replaced by
  ``scale * (pid + (diff - diff_initial))`` for both the question and the
  interaction encoding.  ``diff_initial`` is stored in non-trainable buffers.
  Because AKT initialises the difficulty embedding to zero and the buffers are
  captured at construction, the initial modulation is zero, exactly as in the
  original model.  Removing the original multiplication changes behaviour once
  the difficulty/change embeddings move away from their initial values; that
  is an expected property of the mechanism, not an initialisation mismatch.
- ``ukt_cov``: projects interaction states to a diagonal Gaussian mean and a
  strictly positive covariance (``ELU + 1``), scores strictly previous
  positions by the negative squared 2-Wasserstein distance, and aggregates
  covariance values with ``alpha ** 2``.  The result goes through a
  zero-initialised projection into ``d_output``.  Covariance is clamped before
  ``sqrt`` and the masked row is renormalised with a positive epsilon, so the
  residual and its gradients stay finite even at the first position.
- ``mtkt_cic``: the MTKT ``CIC`` equations are reproduced with left-padded
  causal convolutions.  The final 1x1 convolution is zero-initialised, so the
  residual is exactly zero at construction.  A one-tap branch sees only the
  current position; the longer branch sees only the current and past positions.
- ``mtkt_time``: reads only ``gaps`` (never ``target`` or a current response),
  validates the full-length ``B x L`` integer shapes, applies ``log1p`` and a
  zero-initialised MLP decode correction.  No current response is used.
- ``scale`` is applied once per mechanism; ``baseline`` ignores it.

Only this module and its test are new.  Nothing in the original AKT file is
modified.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .akt import AKT


MECHANISMS = ("baseline", "lef_additive", "ukt_cov", "mtkt_cic", "mtkt_time")
_TIME_KEYS = ("rgaps", "sgaps", "pcounts")


def _zero_output(layer):
    """Initialise a final projection to exact zero output."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _CausalConv1d(nn.Conv1d):
    """Left-padded convolution: output position ``t`` cannot see ``t + 1``."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        if kernel_size < 1:
            raise ValueError("Causal convolution needs a positive kernel")
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            **kwargs,
        )
        self._causal_padding = padding

    def forward(self, x):
        y = super().forward(x)
        if self._causal_padding:
            y = y[:, :, : -self._causal_padding]
        return y


class MTKTCausalInteraction(nn.Module):
    """The genuine MTKT CIC residual on a shared hidden sequence.

    Mirrors ``CIC`` in the MTKT source: a one-tap and a longer causal
    convolution, GELU/dropout, the two cross-gated products, and a 1x1 output
    convolution.  The output convolution starts at zero so adding this module
    never changes the initial AKT prediction.
    """

    def __init__(self, d_model, hidden, kernel1=1, kernel2=3, dropout=0.1, scale=0.1):
        super().__init__()
        if kernel1 < 1 or kernel2 < 1:
            raise ValueError("MTKT CIC kernels must be positive")
        if hidden < 1:
            raise ValueError("MTKT CIC hidden width must be positive")
        self.conv1 = _CausalConv1d(d_model, hidden, kernel1)
        self.conv2 = _CausalConv1d(d_model, hidden, kernel2)
        self.conv3 = _zero_output(_CausalConv1d(hidden, d_model, 1))
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.scale = float(scale)

    def forward(self, hidden):
        x = hidden.transpose(1, 2)
        raw1 = self.conv1(x)
        raw2 = self.conv2(x)
        act1 = self.drop(self.act(raw1))
        act2 = self.drop(self.act(raw2))
        out = self.conv3(raw1 * act2 + raw2 * act1)
        return self.scale * torch.tanh(out.transpose(1, 2))


class UKTCovarianceResidual(nn.Module):
    """Simplified UKT covariance history matching as a zero-init residual.

    This is not full UKT.  It keeps the two ingredients that are relevant to
    this comparison: a diagonal-Gaussian 2-Wasserstein distance between the
    current interaction state and strictly previous states, and UKT's
    ``alpha ** 2`` covariance aggregation.  The aggregated covariance is
    projected back to ``d_model`` by a zero-initialised linear layer.
    """

    def __init__(self, d_model, scale=0.1, eps=1e-4):
        super().__init__()
        if d_model < 1:
            raise ValueError("UKT covariance residual needs a positive width")
        self.eps = float(eps)
        self.scale = float(scale)
        self.q_mean = nn.Linear(d_model, d_model)
        self.k_mean = nn.Linear(d_model, d_model)
        self.q_cov = nn.Linear(d_model, d_model)
        self.k_cov = nn.Linear(d_model, d_model)
        self.v_cov = nn.Linear(d_model, d_model)
        self.out = _zero_output(nn.Linear(d_model, d_model))

    def forward(self, query, interaction=None):
        # ``interaction`` already contains the AKT question/KC and response
        # embeddings.  Position t may only read positions < t.
        if interaction is None:
            interaction = query
        mean_q = self.q_mean(query)
        mean_k = self.k_mean(interaction)
        # ELU + 1 is the positive covariance parameterisation used by UKT.
        cov_q = F.elu(self.q_cov(query)) + 1.0
        cov_k = F.elu(self.k_cov(interaction)) + 1.0
        cov_v = F.elu(self.v_cov(interaction)) + 1.0

        eps = self.eps
        mean_sq = (mean_q * mean_q).sum(dim=-1, keepdim=True)
        # Diagonal-Gaussian squared 2-Wasserstein distance.
        w2 = (
            mean_sq
            + (mean_k * mean_k).sum(dim=-1, keepdim=True).transpose(-1, -2)
            - 2.0 * torch.matmul(mean_q, mean_k.transpose(-1, -2))
        )
        sqrt_q = torch.sqrt(cov_q.clamp_min(eps))
        sqrt_k = torch.sqrt(cov_k.clamp_min(eps))
        w2 = (
            w2
            + cov_q.sum(dim=-1, keepdim=True)
            + cov_k.sum(dim=-1, keepdim=True).transpose(-1, -2)
            - 2.0 * torch.matmul(sqrt_q, sqrt_k.transpose(-1, -2))
        )
        w2 = w2.clamp_min(0.0)

        length = interaction.size(1)
        strict_prefix = torch.ones(
            length, length, dtype=torch.bool, device=interaction.device
        ).tril(-1)
        logits = -w2 / math.sqrt(interaction.size(-1))
        logits = logits.masked_fill(~strict_prefix.unsqueeze(0), -1e9)
        alpha = torch.softmax(logits, dim=-1)
        alpha = alpha * strict_prefix.unsqueeze(0).to(alpha.dtype)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # UKT's uattention aggregates covariance with softmax scores squared.
        aggregated_cov = torch.matmul(alpha.square(), cov_v)
        return self.scale * torch.tanh(self.out(aggregated_cov))


class MTKTLogGapResidual(nn.Module):
    """MTKT-inspired log-gap decode correction, zero-initialised.

    Consumes the full-length integer ``rgaps``/``sgaps``/``pcounts`` only.
    The current response is deliberately not an input.
    """

    def __init__(self, d_model, hidden, scale=0.1):
        super().__init__()
        if hidden < 1:
            raise ValueError("MTKT time hidden width must be positive")
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(len(_TIME_KEYS), hidden),
            nn.GELU(),
            _zero_output(nn.Linear(hidden, d_model)),
        )

    def forward(self, gaps):
        dtype = self.net[0].weight.dtype
        features = torch.stack(
            [
                torch.log1p(gaps[key].to(dtype=dtype).clamp_min(0))
                for key in _TIME_KEYS
            ],
            dim=-1,
        )
        return self.scale * torch.tanh(self.net(features))


class AKTFairness(AKT):
    """AKT with one composable reference-mechanism transfer at a time.

    Parameters mirror :class:`pykt.models.akt.AKT`.  Two additional keyword
    arguments select the mechanism and its residual scale:

    ``mechanism``
        ``baseline`` (default, exact original AKT), ``lef_additive``,
        ``ukt_cov``, ``mtkt_cic`` or ``mtkt_time``.
    ``scale``
        Positive multiplier applied by the selected mechanism.  ``baseline``
        ignores it.

    ``forward(q_data, target, pid_data=None, qtest=False, gaps=None)`` keeps
    the original AKT return contract: ``(preds, c_reg_loss)`` normally and
    ``(preds, c_reg_loss, concat_q)`` when ``qtest=True``.  ``gaps`` is only
    read by ``mtkt_time``.
    """

    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        dropout,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        mechanism="baseline",
        scale=0.1,
        cic_kernel1=1,
        cic_kernel2=3,
        cic_hidden=None,
        time_hidden=None,
        **kwargs,
    ):
        if mechanism not in MECHANISMS:
            raise ValueError(
                "Unknown AKT fairness mechanism {!r}; expected one of {}".format(
                    mechanism, MECHANISMS
                )
            )
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("AKT fairness scale must be finite and positive")
        if mechanism == "lef_additive" and int(n_pid) <= 0:
            raise ValueError("lef_additive requires AKT problem ids (n_pid > 0)")

        super().__init__(
            n_question=n_question,
            n_pid=n_pid,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            **kwargs,
        )
        self.mechanism = mechanism
        self.scale = float(scale)

        # Build extras under a forked RNG so the inherited AKT weights are
        # bit-identical to a plain AKT constructed from the same seed.
        if mechanism == "lef_additive":
            self.register_buffer(
                "q_diff_origin", self.q_embed_diff.weight.detach().clone()
            )
            self.register_buffer(
                "qa_diff_origin", self.qa_embed_diff.weight.detach().clone()
            )
        elif mechanism == "ukt_cov":
            with torch.random.fork_rng(devices=[]):
                self.ukt_cov = UKTCovarianceResidual(d_model, self.scale)
        elif mechanism == "mtkt_cic":
            hidden = int(cic_hidden) if cic_hidden is not None else int(d_ff)
            with torch.random.fork_rng(devices=[]):
                self.cic = MTKTCausalInteraction(
                    d_model,
                    hidden,
                    kernel1=int(cic_kernel1),
                    kernel2=int(cic_kernel2),
                    dropout=dropout,
                    scale=self.scale,
                )
        elif mechanism == "mtkt_time":
            hidden = (
                int(time_hidden)
                if time_hidden is not None
                else max(8, 2 * int(d_model))
            )
            with torch.random.fork_rng(devices=[]):
                self.time_residual = MTKTLogGapResidual(d_model, hidden, self.scale)

    @torch.no_grad()
    def reset_initial_buffers(self):
        """Recapture the LEF additive origin buffers from current weights.

        Use this after loading an AKT checkpoint into ``lef_additive`` so the
        additive modulation starts at exactly zero for those weights.
        """
        if self.mechanism != "lef_additive":
            raise RuntimeError("reset_initial_buffers is only valid for lef_additive")
        self.q_diff_origin.copy_(self.q_embed_diff.weight)
        self.qa_diff_origin.copy_(self.qa_embed_diff.weight)
        return self

    def _base_components(self, q_data, target, pid_data):
        """AKT's original embedding path, returned before the architecture."""
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        pid_embed_data = None
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(target)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        else:
            c_reg_loss = 0.0
        return q_embed_data, qa_embed_data, pid_embed_data, c_reg_loss

    def _lef_additive_components(self, q_data, target, pid_data):
        """LEF additive replacement of AKT's multiplicative PID modulation."""
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        q_embed_diff_data = self.q_embed_diff(q_data)
        pid_embed_data = self.difficult_param(pid_data)
        qa_embed_diff_data = self.qa_embed_diff(target)

        q_embed_data = q_embed_data + self.scale * (
            pid_embed_data + (q_embed_diff_data - F.embedding(q_data, self.q_diff_origin))
        )
        if self.separate_qa:
            qa_embed_data = qa_embed_data + self.scale * (
                pid_embed_data + (qa_embed_diff_data - F.embedding(target, self.qa_diff_origin))
            )
        else:
            qa_embed_data = qa_embed_data + self.scale * (
                pid_embed_data
                + (
                    (qa_embed_diff_data + q_embed_diff_data)
                    - (F.embedding(target, self.qa_diff_origin) + F.embedding(q_data, self.q_diff_origin))
                )
            )
        c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2
        return q_embed_data, qa_embed_data, pid_embed_data, c_reg_loss

    @staticmethod
    def _validate_gaps(gaps, q_data):
        if not isinstance(gaps, dict):
            raise ValueError(
                "mtkt_time requires gaps with integer B x L 'rgaps', 'sgaps' "
                "and 'pcounts'"
            )
        batch, length = q_data.shape[0], q_data.shape[1]
        checked = {}
        for key in _TIME_KEYS:
            if key not in gaps:
                raise ValueError("mtkt_time gaps is missing {!r}".format(key))
            value = gaps[key]
            if not torch.is_tensor(value):
                raise ValueError("mtkt_time gaps[{!r}] must be a tensor".format(key))
            if value.ndim != 2 or tuple(value.shape) != (batch, length):
                raise ValueError(
                    "mtkt_time gaps[{!r}] must be full-length B x L; got {} for "
                    "B x L = {}".format(key, tuple(value.shape), (batch, length))
                )
            if value.is_floating_point() or value.is_complex():
                raise ValueError(
                    "mtkt_time gaps[{!r}] must contain integer bucket ids".format(key)
                )
            if bool((value < 0).any()):
                raise ValueError(
                    "mtkt_time gaps[{!r}] must be non-negative".format(key)
                )
            checked[key] = value
        return checked

    def forward(self, q_data, target, pid_data=None, qtest=False, gaps=None):
        # Strict reuse: baseline is the original AKT forward, untouched.
        if self.mechanism == "baseline":
            return super().forward(q_data, target, pid_data, qtest)

        if self.n_pid > 0 and pid_data is None:
            raise ValueError("pid_data is required when n_pid > 0")

        if self.mechanism == "lef_additive":
            q_embed_data, qa_embed_data, pid_embed_data, c_reg_loss = (
                self._lef_additive_components(q_data, target, pid_data)
            )
        else:
            q_embed_data, qa_embed_data, pid_embed_data, c_reg_loss = (
                self._base_components(q_data, target, pid_data)
            )

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        if self.mechanism == "ukt_cov":
            # Query must exclude the current response even after the zero-init
            # residual learns nonzero weights. Only keys/values carry responses.
            d_output = d_output + self.ukt_cov(q_embed_data, qa_embed_data)
        elif self.mechanism == "mtkt_cic":
            d_output = d_output + self.cic(d_output)
        elif self.mechanism == "mtkt_time":
            checked_gaps = self._validate_gaps(gaps, q_data)
            d_output = d_output + self.time_residual(checked_gaps)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)
        if not qtest:
            return preds, c_reg_loss
        return preds, c_reg_loss, concat_q
