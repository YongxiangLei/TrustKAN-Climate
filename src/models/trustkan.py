"""Core TrustKAN-Climate model components.

This initial implementation provides a dependency-light KAN-style temporal model.
It is intentionally modular so uncertainty, drift and selective-prediction modules
can be evaluated independently in ablations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GaussianRBFKANLayer(nn.Module):
    """KAN-style layer using learnable univariate Gaussian RBF expansions."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        grid = torch.linspace(-2.5, 2.5, grid_size)
        self.register_buffer("grid", grid)
        self.log_scale = nn.Parameter(torch.zeros(in_features, grid_size))
        self.coeff = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.base = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.coeff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        scale = self.log_scale.exp().clamp_min(1e-4)
        basis = torch.exp(
            -0.5 * ((x.unsqueeze(-1) - self.grid) / scale) ** 2
        )
        nonlinear = torch.einsum("...ig,oig->...o", basis, self.coeff)
        return self.base(x) + nonlinear


def _local_stem(n_features: int, hidden_dim: int) -> nn.Sequential:
    """Two symmetrically padded kernel-3 convolutions.

    Retained because every published result was produced with it. Its receptive
    field is five steps in the interior of the sequence but only three at the
    final position, where half of each kernel falls on right padding, so a
    last-step readout sees three input steps no matter how long the history is.
    """
    return nn.Sequential(
        nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.GELU(),
    )


def dilated_depth(history: int) -> int:
    """Layers of doubling dilation needed to span `history` steps.

    Stacking kernel-3 convolutions with dilations 1, 2, ..., 2^(L-1) gives a
    receptive field of 1 + 2(2^L - 1), so inverting that bound gives the depth.
    """
    return max(1, math.ceil(math.log2(max(1.0, (history + 1) / 2))))


class CausalDilatedStem(nn.Module):
    """Temporal stem whose receptive field spans the whole history window.

    Three properties matter, and the local stem lacks all three. Padding is
    causal, so the final position depends on its entire past instead of losing
    half of each kernel to the right boundary. Dilations double with depth, so
    the field grows exponentially and reaches a year of daily observations in
    eight layers rather than the hundreds a dense stack would need. The
    convolutions are depthwise with a single pointwise mixer, so spanning the
    window costs roughly half the parameters of the two dense layers it
    replaces, which keeps the comparison against a plain KAN a comparison of
    inductive bias rather than of capacity.
    """

    def __init__(self, n_features: int, hidden_dim: int, history: int):
        super().__init__()
        self.embed = nn.Conv1d(n_features, hidden_dim, kernel_size=1)
        self.depthwise = nn.ModuleList()
        pads = []
        for index in range(dilated_depth(history)):
            dilation = 2**index
            self.depthwise.append(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    groups=hidden_dim,
                )
            )
            pads.append(2 * dilation)
        self.pads = tuple(pads)
        self.mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()

    @property
    def receptive_field(self) -> int:
        return 1 + sum(self.pads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, features, time]
        z = self.activation(self.embed(x))
        for conv, pad in zip(self.depthwise, self.pads):
            # Left-only padding keeps the sequence length fixed while making
            # every output position a function of its own past exclusively.
            z = z + self.activation(conv(F.pad(z, (pad, 0))))
        return self.activation(self.mix(z))


STEMS = ("local", "dilated")


def _temporal_stem(
    n_features: int, hidden_dim: int, stem: str = "local", history: int | None = None
) -> nn.Module:
    if stem not in STEMS:
        raise ValueError(f"stem must be one of {sorted(STEMS)}")
    if stem == "local":
        return _local_stem(n_features, hidden_dim)
    if history is None:
        raise ValueError("the dilated stem needs the history length to size itself")
    return CausalDilatedStem(n_features, hidden_dim, history)


def kan_layer_parameters(in_features: int, out_features: int, grid_size: int) -> int:
    """Parameter count of one GaussianRBFKANLayer."""
    coefficients = out_features * in_features * grid_size
    base = in_features * out_features + out_features
    scales = in_features * grid_size
    return coefficients + base + scales


def budget_matched_width(hidden_dim: int, grid_size: int) -> int:
    """Hidden width making a two-layer MLP match the KAN layer's budget.

    A1 only isolates the KAN mapping if the replacement has a comparable
    parameter count; otherwise the comparison silently measures capacity.
    """
    target = kan_layer_parameters(hidden_dim, hidden_dim, grid_size)
    width = round((target - hidden_dim) / (2 * hidden_dim + 1))
    return max(1, int(width))


class TemporalKANBlock(nn.Module):
    """Temporal mixing followed by an interpretable KAN mapping."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        grid_size: int = 8,
        stem: str = "local",
        history: int | None = None,
    ):
        super().__init__()
        self.temporal = _temporal_stem(n_features, hidden_dim, stem, history)
        self.kan = GaussianRBFKANLayer(hidden_dim, hidden_dim, grid_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        z = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(z + self.kan(z))


class TemporalMLPBlock(nn.Module):
    """Budget-matched non-KAN encoder used by the A1 ablation."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        grid_size: int = 8,
        stem: str = "local",
        history: int | None = None,
    ):
        super().__init__()
        width = budget_matched_width(hidden_dim, grid_size)
        self.temporal = _temporal_stem(n_features, hidden_dim, stem, history)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(z + self.mlp(z))


ENCODERS = {"kan": TemporalKANBlock, "mlp": TemporalMLPBlock}

# Readout convention. Averaging the encoder states over the whole history
# discards recency, which dominates short-horizon temperature forecasting: with
# a 365-step history it scored well behind a persistence baseline. "last"
# matches the convention used by every recurrent and attention baseline here,
# so the choice is consistency rather than measured accuracy. "mean" is kept so
# the defect remains reproducible as an ablation. "attention" adds a learned
# global aggregation alongside the final state: even when the stem's receptive
# field spans the window, a single end state has to route a year of context
# through one channel vector, whereas the flattened-lag KAN baseline reads every
# lag through its own weight. The attention weights are also a distribution over
# time, so the aggregation itself can be inspected.
READOUTS = ("last", "mean", "attention")


class TrustKAN(nn.Module):
    """Temporal KAN forecaster with point and quantile outputs.

    Quantile heads enable subsequent conformal calibration. Reliability and OOD
    scoring are deliberately kept outside this class to avoid conflating model
    capacity with post-hoc trust mechanisms in ablation studies.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int = 1,
        hidden_dim: int = 64,
        grid_size: int = 8,
        quantiles=(0.05, 0.5, 0.95),
        encoder: str = "kan",
        quantile_head: bool = True,
        readout: str = "last",
        stem: str = "local",
        history: int | None = None,
    ):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {sorted(ENCODERS)}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {sorted(READOUTS)}")
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        self.encoder_name = encoder
        self.readout_name = readout
        self.stem_name = stem
        self.has_quantile_head = bool(quantile_head)
        self.encoder = ENCODERS[encoder](
            n_features, hidden_dim, grid_size, stem=stem, history=history
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        if readout == "attention":
            self.query = nn.Parameter(torch.empty(hidden_dim))
            nn.init.normal_(self.query, std=hidden_dim**-0.5)
            self.readout_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.point_head = nn.Linear(hidden_dim, horizon)
        self.quantile_head = (
            nn.Linear(hidden_dim, horizon * len(self.quantiles))
            if self.has_quantile_head
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        if self.readout_name == "mean":
            pooled = self.pool(z.transpose(1, 2)).squeeze(-1)
        elif self.readout_name == "attention":
            weights = torch.softmax(
                z @ self.query / math.sqrt(z.shape[-1]), dim=1
            )
            context = torch.einsum("bt,bth->bh", weights, z)
            pooled = self.readout_projection(torch.cat([z[:, -1], context], dim=-1))
        else:
            pooled = z[:, -1]
        point = self.point_head(pooled)
        if self.quantile_head is None:
            # A2 keeps the downstream contract but carries no quantile
            # information, so split conformal degenerates to symmetric
            # absolute-residual intervals around the point forecast.
            q = point.unsqueeze(-1).expand(-1, -1, len(self.quantiles))
            return {"point": point, "quantiles": q, "embedding": pooled}
        q = self.quantile_head(pooled).view(
            x.shape[0], self.horizon, len(self.quantiles)
        )
        # Sorting guarantees non-crossing outputs; later work can compare a
        # differentiable monotonic parameterization as an ablation.
        q, _ = torch.sort(q, dim=-1)
        return {"point": point, "quantiles": q, "embedding": pooled}
