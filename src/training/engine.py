"""Minimal reproducible PyTorch training engine."""
from __future__ import annotations

import copy
import os
import random
import time
import numpy as np
import torch
from torch import nn


def resolve_device(device=None) -> torch.device:
    """Resolve and validate an explicit or automatic PyTorch device."""
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {resolved} was requested but this PyTorch build has no usable CUDA"
            )
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is outside the available range "
                f"[0, {torch.cuda.device_count() - 1}]"
            )
        resolved = torch.device("cuda", index)
    return resolved


def set_seed(seed: int, *, deterministic=True, warn_only=False):
    """Seed all RNGs and configure deterministic CUDA kernels when requested."""
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(deterministic), warn_only=bool(warn_only)
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)


def train_regressor(
    model,
    train_loader,
    val_loader,
    *,
    epochs=100,
    lr=1e-3,
    patience=10,
    optimizer_name="adamw",
    weight_decay=None,
    device=None,
):
    device = resolve_device(device)
    model = model.to(device)
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        decay = 0.01 if weight_decay is None else weight_decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    elif optimizer_name == "adam":
        decay = 0.0 if weight_decay is None else weight_decay
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    loss_fn = nn.MSELoss()
    best = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); train_sum = 0.0; n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            if isinstance(out, dict): out = out["point"]
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_sum += loss.item() * len(xb); n_train += len(xb)
        model.eval(); val_sum = 0.0; n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                if isinstance(out, dict): out = out["point"]
                val_sum += loss_fn(out, yb).item() * len(xb); n_val += len(xb)
        train_loss = train_sum / max(1, n_train)
        val_loss = val_sum / max(1, n_val)
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_val:
            best_val = val_loss; best = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best)
    return model, history, time.perf_counter() - start


def predict(model, loader, device=None):
    device = device or next(model.parameters()).device
    model.eval(); preds=[]; truth=[]
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            if isinstance(out, dict): out = out["point"]
            preds.append(out.cpu().numpy()); truth.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(truth)
