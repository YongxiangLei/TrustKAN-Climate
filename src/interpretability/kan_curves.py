"""Extract univariate TrustKAN RBF curves for explanation-stability analysis."""
from __future__ import annotations

import numpy as np
import torch

from src.models.trustkan import GaussianRBFKANLayer, TrustKAN


def default_evaluation_grid(n_points: int = 51, low: float = -2.5, high: float = 2.5):
    if n_points < 2:
        raise ValueError("evaluation grid must contain at least two points")
    return np.linspace(low, high, n_points)


def extract_rbf_kan_curves(layer: GaussianRBFKANLayer, x_grid=None):
    """Return combined linear+RBF maps for every input→output pair.

    The shared output bias is omitted so each curve remains a univariate map.
    """
    if not isinstance(layer, GaussianRBFKANLayer):
        raise TypeError("layer must be a GaussianRBFKANLayer")
    grid = default_evaluation_grid() if x_grid is None else np.asarray(x_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("x_grid must be one-dimensional")
    x = torch.as_tensor(grid, dtype=layer.coeff.dtype, device=layer.coeff.device)
    scale = layer.log_scale.exp().clamp_min(1e-4)
    basis = torch.exp(-0.5 * ((x[:, None, None] - layer.grid) / scale) ** 2)
    rbf = torch.einsum("xig,oig->oix", basis, layer.coeff)
    linear = layer.base.weight[:, :, None] * x[None, None, :]
    curves = (linear + rbf).detach().cpu().numpy()
    return {
        "x_grid": grid,
        "curves": curves.reshape(layer.out_features * layer.in_features, -1),
        "n_out": int(layer.out_features),
        "n_in": int(layer.in_features),
    }


def extract_trustkan_curves(model: TrustKAN, x_grid=None):
    if not isinstance(model, TrustKAN):
        raise TypeError("model must be a TrustKAN")
    return extract_rbf_kan_curves(model.encoder.kan, x_grid=x_grid)
