"""Signed encoder-spline KAN-SAE with a hard per-row activation budget.

This project variant applies magnitude TopK AFTER the learned cubic splines.
The original L1 KANSAE remains unchanged.
"""
from typing import Optional

import torch
from torch import Tensor

from .kan_sae import KANSAE, _positive_integer


def magnitude_topk(values: Tensor, k: int, allowed: Optional[Tensor] = None) -> Tensor:
    """Retain at most k signed values per row, ranking by absolute magnitude.

    ``allowed`` restricts the candidate dictionary for the auxiliary loss.
    Gradients flow through retained values, not through the discrete indices.
    Equal magnitudes use PyTorch's topk tie rule; tied indices need not agree
    across devices, but a tie never causes more than k entries to survive.
    """
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("Expected floating-point (batch, features) values.")
    _positive_integer("k", k)
    if k > values.shape[1]:
        raise ValueError("k must not exceed the dictionary size.")
    scores = values.abs()
    if allowed is not None:
        if allowed.shape != (values.shape[1],) or allowed.dtype != torch.bool:
            raise ValueError("allowed must be a boolean mask over features.")
        scores = scores.masked_fill(~allowed, -torch.inf)
    indices = scores.topk(k, dim=1, sorted=False).indices
    kept = values.gather(1, indices)
    if allowed is not None:
        # When fewer than k are allowed, topk fills its indices from forbidden
        # entries. They must have neither a value nor a gradient in the result.
        kept = kept.masked_fill(~allowed[indices], 0)
    return torch.zeros_like(values).scatter(1, indices, kept)


class KANTopKSAE(KANSAE):
    """Cubic encoder splines followed by signed magnitude TopK.

    Normalization, grids and decoder are inherited from KANSAE. Public encode
    and forward ALWAYS apply k_active, even while the trainer performs its
    explicit dense warm-up. Loaded checkpoints therefore cannot accidentally
    expose dense codes to probes or plots.
    """

    def __init__(self, d_in: int, latent: int, *, k_active: int = 32, **kwargs):
        _positive_integer("k_active", k_active)
        if k_active > latent:
            raise ValueError("k_active must not exceed latent.")
        super().__init__(d_in, latent, **kwargs)
        self.k_active = k_active

    def config(self):
        return dict(super().config(), k_active=self.k_active)

    def encode_dense(self, x: Tensor) -> Tensor:
        """Pre-gate spline values, for warm-up and dead-feature recovery only."""
        return super().encode(x)

    def encode(self, x: Tensor) -> Tensor:
        return magnitude_topk(self.encode_dense(x), self.k_active)
