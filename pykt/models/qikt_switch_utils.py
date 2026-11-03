"""Opt-in QIKT interventions; no CUDA-specific dependency."""
import math
import torch
import torch.nn.functional as F


def causal_concept_features(concepts, responses, num_concepts, strength=10.0):
    """Current-KC prefix log odds and support, excluding the current answer.

    A multi-KC event contributes once to each distinct KC. All-padding events
    contribute nothing. Memory resets at the start of each supplied sequence.
    """
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError('memory strength must be finite and positive')
    valid = (concepts >= 0) & (concepts < num_concepts)
    valid = valid & ((responses == 0) | (responses == 1)).unsqueeze(-1)
    safe = concepts.clamp(0, num_concepts - 1).long()
    incidence = responses.new_zeros((*responses.shape, num_concepts), dtype=torch.float32)
    incidence.scatter_add_(-1, safe, valid.float())
    incidence = incidence.clamp_max(1)
    safe_responses = torch.where((responses == 0) | (responses == 1), responses, 0)
    correct = incidence * safe_responses.float().unsqueeze(-1)
    counts = incidence.cumsum(1) - incidence
    sums = correct.cumsum(1) - correct
    # Current response validity must not determine whether current query exists.
    query_valid = (concepts >= 0) & (concepts < num_concepts)
    query = torch.zeros_like(incidence).scatter_add_(-1, safe, query_valid.float()).clamp_max(1)
    denom = query.sum(-1).clamp_min(1)
    rates = (sums + strength * .5) / (counts + strength)
    mean = (rates * query).sum(-1) / denom
    mean = torch.where(query.sum(-1) > 0, mean, torch.full_like(mean, .5))
    support = (counts * query).sum(-1) / denom
    return torch.stack((torch.logit(mean.clamp(1e-6, 1 - 1e-6)), torch.log1p(support)), dim=-1)


def fragment_pairwise_loss(probabilities, targets, mask):
    """Mean logistic ranking loss, equal weight per two-class fragment.

    This is a fragment surrogate, not complete-orirow AUC optimization.
    Single-class fragments are excluded. Padding never enters positive/negative
    selection, including when padded probabilities or targets are nonfinite.
    """
    losses = []
    for p, y, active in zip(probabilities, targets, mask.bool()):
        p, y = p[active], y[active]
        if not torch.isfinite(p).all() or not ((y == 0) | (y == 1)).all():
            raise ValueError('active probabilities must be finite and labels binary')
        logits = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        positive, negative = logits[y == 1], logits[y == 0]
        if positive.numel() and negative.numel():
            losses.append(F.softplus(negative.unsqueeze(0) - positive.unsqueeze(1)).mean())
    return torch.stack(losses).mean() if losses else probabilities[mask.bool()].sum() * 0
