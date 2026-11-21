"""Per-feature B-spline SAE for the AMS691 GraphCast experiments.

Implements Cheon (2026), arXiv:2605.17493v1, equations (4)-(6): a linear
encoder, one learned univariate spline per latent, and a linear decoder.
Sparsity comes from an L1 penalty on signed codes, with no ReLU or TopK.
See docs/kan_sae.md for the GraphCast adaptation and unspecified paper details.
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class PerFeatureBSpline(nn.Module):
    """Independent, signed B-splines evaluated using the local de Boor recursion.

    Inputs have shape (batch, features). There are ``n_basis`` trainable
    coefficients per feature. Knots are fixed buffers, not parameters.
    Endpoints repeat degree+1 times; values outside the grid are clamped to
    the endpoint value (constant extension). Defaults are cubic, with nine
    coefficients, six intervals, and thirteen knots per feature.

    The initial [-1, 1] grids are placeholders for calibration by KANSAE.
    Evaluation gathers only degree+1 coefficients per input/feature pair.
    """

    def __init__(self, features: int, n_basis: int = 9, degree: int = 3):
        super().__init__()
        for name, value in (("features", features), ("n_basis", n_basis), ("degree", degree)):
            _positive_integer(name, value)
        if n_basis <= degree:
            raise ValueError("n_basis must be greater than degree.")
        self.features = features
        self.n_basis = n_basis
        self.degree = degree
        self.coefficients = nn.Parameter(torch.zeros(features, n_basis))
        self.register_buffer("knots", torch.empty(features, n_basis + degree + 1))
        self.set_grid_(-torch.ones(features), torch.ones(features))

    @torch.no_grad()
    def set_grid_(self, lower: Tensor, upper: Tensor) -> None:
        """Set uniform clamped grids; do this before fitting coefficients."""
        lower = torch.as_tensor(lower, device=self.knots.device, dtype=self.knots.dtype)
        upper = torch.as_tensor(upper, device=self.knots.device, dtype=self.knots.dtype)
        if lower.shape != (self.features,) or upper.shape != (self.features,):
            raise ValueError("Grid bounds must have shape (features,).")
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("Grid bounds must be finite.")
        if not (upper > lower).all():
            raise ValueError("Every upper grid bound must exceed its lower bound.")
        fractions = torch.linspace(
            0, 1, self.n_basis - self.degree + 1,
            device=self.knots.device, dtype=self.knots.dtype,
        )
        breaks = torch.lerp(lower[:, None], upper[:, None], fractions)
        if not (torch.diff(breaks, dim=1) > 0).all():
            raise ValueError("Grid intervals are too narrow for the model's floating-point dtype.")
        self.knots.copy_(torch.cat((
            lower[:, None].expand(-1, self.degree + 1),
            breaks[:, 1:-1],
            upper[:, None].expand(-1, self.degree + 1),
        ), dim=1))

    def forward(self, h: Tensor) -> Tensor:
        if h.ndim != 2 or h.shape[1] != self.features or not h.is_floating_point():
            raise ValueError("Spline inputs must be floating-point (batch, features) tensors.")
        t = self.knots
        x = h.clamp(min=t[:, self.degree], max=t[:, self.n_basis])
        # Work by feature to search each feature's own knot vector. The last
        # endpoint uses the final nonempty interval, so phi(upper) == c[-1].
        span = torch.searchsorted(t, x.T.contiguous(), right=True).T - 1
        span = span.clamp(min=self.degree, max=self.n_basis - 1)
        feature = torch.arange(self.features, device=h.device)
        indices = span[..., None] - self.degree + torch.arange(self.degree + 1, device=h.device)
        values = list(self.coefficients[feature[None, :, None], indices].unbind(dim=-1))
        for level in range(1, self.degree + 1):
            for j in range(self.degree, level - 1, -1):
                left_index = span - self.degree + j
                left = t[feature, left_index]
                right = t[feature, left_index + self.degree - level + 1]
                # Bounds validated in set_grid_ guarantee a positive denominator
                # even though the endpoint knots themselves are repeated.
                alpha = (x - left) / (right - left)
                values[j] = (1 - alpha) * values[j - 1] + alpha * values[j]
        return values[self.degree]


class KANSAE(nn.Module):
    """KAN-SAE with a shared pre/post bias and a learned encoder bias.

    ``forward(x)`` returns ``(reconstruction, signed_code)``. By default the
    reconstruction target is centered and L2-normalized per row, matching
    this project's TopK baseline. Set normalize_inputs=False for raw inputs.

    Before training, call initialize_from_data() with a training-only sample.
    After each optimizer step, call renormalize_decoder_(). Use config() plus
    state_dict() for checkpoints; the latter includes the calibrated knots.
    """

    def __init__(
        self, d_in: int, latent: int, *, n_basis: int = 9, degree: int = 3,
        normalize_inputs: bool = True, normalization_eps: float = 1e-6,
    ):
        super().__init__()
        _positive_integer("d_in", d_in)
        _positive_integer("latent", latent)
        if not isinstance(normalize_inputs, bool):
            raise ValueError("normalize_inputs must be True or False.")
        if not math.isfinite(normalization_eps) or normalization_eps <= 0:
            raise ValueError("normalization_eps must be finite and positive.")
        self.d_in = d_in
        self.latent = latent
        self.normalize_inputs = normalize_inputs
        self.normalization_eps = normalization_eps
        self.enc = nn.Linear(d_in, latent, bias=False)
        self.dec = nn.Linear(latent, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(latent))
        self.spline = PerFeatureBSpline(latent, n_basis=n_basis, degree=degree)
        self.register_buffer("grid_initialized", torch.tensor(False))
        with torch.no_grad():
            nn.init.normal_(self.dec.weight, std=1 / math.sqrt(d_in))
            self.renormalize_decoder_()
            # The paper does not specify W_enc initialization. Match our TopK
            # baseline's initialization; these remain independent parameters.
            self.enc.weight.copy_(self.dec.weight.T)

    def config(self) -> Dict:
        """JSON-compatible constructor settings; save alongside state_dict()."""
        return dict(
            d_in=self.d_in, latent=self.latent, n_basis=self.spline.n_basis,
            degree=self.spline.degree, normalize_inputs=self.normalize_inputs,
            normalization_eps=self.normalization_eps,
        )

    def reconstruction_target(self, x: Tensor) -> Tensor:
        """Apply exactly the input transform used by the encoder and loss."""
        if x.ndim != 2 or x.shape[1] != self.d_in or not x.is_floating_point():
            raise ValueError(f"Expected floating-point inputs with shape (batch, {self.d_in}).")
        if not self.normalize_inputs:
            return x
        centered = x - x.mean(dim=1, keepdim=True)
        return centered / centered.norm(dim=1, keepdim=True).clamp_min(self.normalization_eps)

    @torch.no_grad()
    def initialize_from_data(
        self, calibration_data: Tensor, *, training_mean: Optional[Tensor] = None,
        quantiles: Tuple[float, float] = (0.01, 0.99),
        feature_chunk_size: int = 128, min_grid_width: float = 1e-3,
    ) -> None:
        """Initialize bias and grids once, before the first optimizer step.

        Pass a representative sample drawn only from SAE training dates (the
        paper uses 50,000 tokens). Samples are not chosen inside this method.
        ``training_mean``, if supplied, must be the mean of reconstruction
        targets over the full training set, not the mean of raw inputs when
        normalization is enabled. Otherwise use this training sample's mean.

        Quantiles are computed independently for each feature's pre-activations.
        Feature chunks avoid a full (50,000, latent) calibration matrix. Narrow
        or constant ranges are widened to keep all intervals representable.
        """
        if bool(self.grid_initialized) or torch.count_nonzero(self.spline.coefficients):
            raise RuntimeError("Initialize a fresh model; recalibrating would change learned spline functions.")
        _positive_integer("feature_chunk_size", feature_chunk_size)
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1.")
        if not math.isfinite(min_grid_width) or min_grid_width <= 0:
            raise ValueError("min_grid_width must be finite and positive.")
        if self.enc.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("Calibrate the model in float32 or float64.")
        # Validate before converting so integer inputs are not silently accepted.
        self.reconstruction_target(calibration_data)
        if not len(calibration_data) or not torch.isfinite(calibration_data).all():
            raise ValueError("Calibration requires a nonempty, finite training sample.")
        targets = self.reconstruction_target(calibration_data.to(self.enc.weight))
        mean = targets.mean(dim=0) if training_mean is None else torch.as_tensor(
            training_mean, dtype=targets.dtype, device=targets.device,
        )
        if mean.shape != (self.d_in,) or not torch.isfinite(mean).all():
            raise ValueError("training_mean must be a finite vector of length d_in.")
        self.b_pre.copy_(mean)
        centered = targets - self.b_pre
        q = targets.new_tensor(quantiles)
        bounds = targets.new_empty((2, self.latent))
        for start in range(0, self.latent, feature_chunk_size):
            end = min(start + feature_chunk_size, self.latent)
            h = F.linear(centered, self.enc.weight[start:end], self.b_enc[start:end])
            bounds[:, start:end] = torch.quantile(h, q, dim=0)
        if not torch.isfinite(bounds).all():
            raise ValueError("Calibration produced nonfinite pre-activations.")
        lower, upper = bounds.unbind()
        # Keep roundoff from collapsing interior intervals when values are large.
        resolution = torch.finfo(targets.dtype).eps * 32 * bounds.abs().amax(dim=0).clamp_min(1)
        width = torch.maximum(upper - lower, resolution.clamp_min(min_grid_width))
        midpoint = lower / 2 + upper / 2
        self.spline.set_grid_(midpoint - width / 2, midpoint + width / 2)
        self.grid_initialized.fill_(True)

    def encode(self, x: Tensor) -> Tensor:
        """Return signed feature activations for each input row."""
        if not bool(self.grid_initialized):
            raise RuntimeError("Call initialize_from_data() with training data before encoding.")
        h = self.enc(self.reconstruction_target(x) - self.b_pre) + self.b_enc
        return self.spline(h)

    def decode(self, code: Tensor) -> Tensor:
        if code.ndim != 2 or code.shape[1] != self.latent:
            raise ValueError(f"Expected codes with shape (batch, {self.latent}).")
        return self.dec(code) + self.b_pre

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        code = self.encode(x)
        return self.decode(code), code

    def loss(
        self, x: Tensor, reconstruction: Tensor, code: Tensor, *, l1_weight: float,
    ) -> Dict[str, Tensor]:
        """Mean per-row squared L2 error plus lambda times mean per-row L1.

        Sum over input/features first, then average over rows. Averaging over
        all elements instead would silently change lambda's relative scale.
        The sparsity term uses absolute values because codes can be negative.
        """
        if not math.isfinite(l1_weight) or l1_weight < 0:
            raise ValueError("l1_weight must be finite and nonnegative.")
        target = self.reconstruction_target(x)
        if not len(x) or reconstruction.shape != target.shape or code.shape != (len(x), self.latent):
            raise ValueError("Loss requires a nonempty batch and matching reconstruction/code shapes.")
        reconstruction_loss = (target - reconstruction).square().sum(dim=1).mean()
        l1_loss = code.abs().sum(dim=1).mean()
        return dict(loss=reconstruction_loss + l1_weight * l1_loss,
                    reconstruction_loss=reconstruction_loss, l1_loss=l1_loss)

    @torch.no_grad()
    def renormalize_decoder_(self) -> None:
        """Enforce the paper's unit-column constraint after an optimizer step."""
        norms = self.dec.weight.norm(dim=0, keepdim=True)
        if not torch.isfinite(norms).all() or not (norms > 0).all():
            raise ValueError("Cannot normalize a zero or nonfinite decoder column.")
        self.dec.weight.div_(norms)

    @torch.no_grad()
    def coefficient_alive_mask(self, threshold: float) -> Tensor:
        """Paper's coefficient-based criterion; threshold must be chosen explicitly.

        The paper's 1.4021 threshold was fitted to its SONNY experiment and
        should not be assumed valid for normalized GraphCast activations.
        """
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive.")
        return self.spline.coefficients.abs().amax(dim=1) >= threshold
