"""Frozen QIKT with a current-question conditioned causal history correction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .qikt_mamba import QIKT_MAMBA
from .qikt_prefix_adapter import prefix_features


def causal_history_features(concepts, responses, num_concepts):
    """16 features; position t only consumes response positions strictly < t."""
    if concepts.shape[:2] != responses.shape or concepts.ndim != 3:
        raise ValueError('Concept/response shape mismatch')
    valid = (concepts >= 0) & (concepts < num_concepts)
    query = torch.zeros((*responses.shape, num_concepts), device=responses.device)
    query.scatter_add_(-1, concepts.clamp(0, num_concepts - 1).long(), valid.float())
    query.clamp_max_(1)
    known = (responses == 0) | (responses == 1)
    incidence = query * known.unsqueeze(-1)
    positive = incidence * torch.where(known, responses, 0).float().unsqueeze(-1)
    total = torch.cat([torch.zeros_like(incidence[:, :1]), incidence.cumsum(1)], 1)
    success = torch.cat([torch.zeros_like(positive[:, :1]), positive.cumsum(1)], 1)
    global_n = torch.cat([torch.zeros_like(responses[:, :1]), known.float().cumsum(1)], 1)
    global_p = torch.cat([torch.zeros_like(responses[:, :1]), torch.where(known, responses, 0).float().cumsum(1)], 1)
    length = responses.shape[1]
    time = torch.arange(length, device=responses.device)
    divisor = query.sum(-1).clamp_min(1)
    prefix = prefix_features(concepts, responses, num_concepts)
    parts = [prefix, torch.stack([torch.logit(((global_p[:, :length] + 5) /
                    (global_n[:, :length] + 10)).clamp(1e-6, 1 - 1e-6)),
                    torch.log1p(global_n[:, :length])], -1)]
    for window in (8, 32):
        start = (time - window).clamp_min(0)
        count = total[:, time] - total[:, start]
        correct = success[:, time] - success[:, start]
        rate = (((correct + 1) / (count + 2)) * query).sum(-1) / divisor
        support = (count * query).sum(-1) / divisor
        gn = global_n[:, time] - global_n[:, start]
        gp = global_p[:, time] - global_p[:, start]
        parts.append(torch.stack([torch.logit(rate.clamp(1e-6, 1 - 1e-6)), torch.log1p(support),
                                  torch.logit(((gp + 1) / (gn + 2)).clamp(1e-6, 1 - 1e-6)),
                                  torch.log1p(gn)], -1))
    touched = torch.where(incidence.bool(), time.view(1, -1, 1), -1)
    latest = touched.cummax(1).values
    previous = torch.cat([torch.full_like(latest[:, :1], -1), latest[:, :-1]], 1)
    history = responses.gather(1, previous.clamp_min(0).reshape(responses.shape[0], -1)).reshape_as(previous)
    observed = (previous >= 0) & query.bool()
    last_response = (torch.where(observed, history.float() * 2 - 1, 0) * query).sum(-1) / divisor
    last_coverage = observed.float().sum(-1) / divisor
    parts.append(torch.stack([last_response, last_coverage], -1))
    result = torch.cat(parts, -1)
    return torch.where(query.sum(-1, keepdim=True) > 0, result, torch.zeros_like(result))


class CausalCorrection(nn.Module):
    def __init__(self, embedding_dim, feature_dim=16, interaction=False):
        super().__init__()
        self.project = nn.Linear(embedding_dim, 64, bias=False)
        self.evidence_project = nn.Linear(feature_dim, 64) if interaction else None
        self.hidden = nn.Sequential(nn.Linear(64 + feature_dim + (64 if interaction else 0), 64),
                                    nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, question, features):
        projected = self.project(question)
        parts = [projected, features]
        if self.evidence_project is not None:
            parts.append(projected * torch.tanh(self.evidence_project(features)))
        return self.output(self.hidden(torch.cat(parts, -1))).squeeze(-1)


def baseline_state_features(outputs, baseline):
    """Detached, current-query state; correctness labels are never inputs."""
    heads = torch.stack([outputs['y_question_all'], outputs['y_concept_all'],
                         outputs['y_concept_next']], -1).detach()
    probability = torch.sigmoid(baseline.detach())
    return torch.cat([heads, probability.unsqueeze(-1),
                      (heads.max(-1).values - heads.min(-1).values).unsqueeze(-1),
                      (4 * probability * (1 - probability)).unsqueeze(-1)], -1)


class QIKTCausalAdapter(QIKT_MAMBA):
    def __init__(self, *args, other_config=None, **kwargs):
        config = dict(other_config or {})
        self.causal_mode = config.get('causal_adapter_mode', 'off')
        state_modes = {'state_only', 'state_multiscale', 'state_interaction'}
        if self.causal_mode not in {'question', 'prefix', 'multiscale', 'supported', 'uncertain', 'disagreement', 'low_support'} | state_modes:
            raise ValueError('Unknown causal adapter mode')
        if config.get('output_mode') != 'an_irt' or any(config.get(k, 0) for k in
                ['item_residual', 'concept_memory_residual', 'head_fusion_residual',
                 'history_response_dropout', 'pairwise_loss_weight', 'anchor_kl_weight']):
            raise ValueError('Causal adapter isolates the unchanged an_irt backbone')
        config['freeze_backbone'] = False
        super().__init__(*args, other_config=config, **kwargs)
        self.causal_scale = float(config.get('causal_adapter_scale', .25))
        self.causal_kl = float(config.get('causal_adapter_kl', 0.))
        if not math.isfinite(self.causal_scale) or not 0 < self.causal_scale <= 1 or not math.isfinite(self.causal_kl) or self.causal_kl < 0:
            raise ValueError('Invalid causal adapter bounds')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.model.causal_adapter = CausalCorrection(
                self.model.emb_size, 22 if self.causal_mode in state_modes else 16,
                interaction=self.causal_mode == 'state_interaction').to(self.device)
        self.model.freeze_backbone = True
        self.model.train(self.model.training)
        self._collect_causal_anchor = False
        self._causal_anchor = None

    def predict_one_step(self, data, return_details=False, process=True, return_raw=False):
        outputs, current = super().predict_one_step(data, return_details=True, process=process)
        def inverse(value):
            return torch.log(value / (1 - value + 1e-8) + 1e-8)
        config = self.model.other_config
        baseline = (inverse(outputs['y_question_all']) * config.get('output_q_all_lambda', 1)
                    + inverse(outputs['y_concept_all']) * config.get('output_c_all_lambda', 1)
                    + inverse(outputs['y_concept_next']) * config.get('output_c_next_lambda', 1)).detach()
        question = self.model.que_emb.que_emb(current['qshft'].long()).detach()
        if self.causal_mode in {'question', 'state_only'}:
            features = question.new_zeros((*question.shape[:2], 16))
        elif self.causal_mode == 'prefix':
            features = causal_history_features(current['cc'], current['cr'], self.model.num_c)[:, 1:]
            features = torch.cat([features[..., :6], torch.zeros_like(features[..., 6:])], -1)
        else:
            features = causal_history_features(current['cc'], current['cr'], self.model.num_c)[:, 1:]
        if self.causal_mode in {'state_only', 'state_multiscale', 'state_interaction'}:
            features = torch.cat([features, baseline_state_features(outputs, baseline)], -1)
        delta = self.causal_scale * torch.tanh(self.model.causal_adapter(question, features))
        if self.causal_mode == 'supported':
            support = torch.expm1(features[..., 1]).clamp_min(0)
            delta = delta * support / (support + 5)
        elif self.causal_mode == 'low_support':
            support = torch.expm1(features[..., 1]).clamp_min(0)
            delta = delta * 5 / (support + 5)
        elif self.causal_mode == 'uncertain':
            probability = torch.sigmoid(baseline)
            delta = delta * (4 * probability * (1 - probability))
        elif self.causal_mode == 'disagreement':
            heads = torch.stack([outputs['y_question_all'], outputs['y_concept_all'],
                                 outputs['y_concept_next']], -1).detach()
            gate = ((heads.max(-1).values - heads.min(-1).values) / .5).clamp(0, 1)
            delta = delta * gate
        outputs['causal_baseline_logit'] = baseline
        outputs['causal_student_logit'] = baseline + delta
        outputs['causal_delta'] = delta
        outputs['y'] = torch.sigmoid(baseline + delta)
        if self._collect_causal_anchor:
            self._causal_anchor = (baseline, baseline + delta, current['sm'].bool())
        return (outputs, current) if return_details else outputs['y']

    def train_one_step(self, data, process=True, return_all=False):
        self._collect_causal_anchor = bool(self.causal_kl)
        try:
            prediction, loss = super().train_one_step(data, process=process, return_all=return_all)
            if self.causal_kl:
                baseline, corrected, mask = self._causal_anchor
                teacher = baseline[mask];student = corrected[mask]
                probability = torch.sigmoid(teacher)
                loss = loss + self.causal_kl * ((F.softplus(student) - probability * student)
                                               - (F.softplus(teacher) - probability * teacher)).mean()
            return prediction, loss
        finally:
            self._collect_causal_anchor = False
            self._causal_anchor = None
