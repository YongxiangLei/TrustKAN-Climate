"""Minimal reproducible PyTorch training engine."""
from __future__ import annotations

import copy
import random
import time
import numpy as np
import torch
from torch import nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
