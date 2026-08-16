"""Training utilities for TrustKAN point + quantile outputs."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch
from torch import nn


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """Mean quantile (pinball) loss.

    pred: [B, H, Q], target: [B, H].
    """
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def train_trustkan(
    model,
    train_loader,
    val_loader,
    *,
    epochs=100,
    lr=1e-3,
    patience=10,
    point_weight=1.0,
    quantile_weight=1.0,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()

    def total_loss(out, y):
        lp = mse(out["point"], y)
        lq = pinball_loss(out["quantiles"], y, model.quantiles)
        return point_weight * lp + quantile_weight * lq, lp, lq

    for epoch in range(1, epochs + 1):
        model.train(); tr_total=tr_point=tr_q=0.0; n=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss, lp, lq = total_loss(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            b=len(xb); n+=b; tr_total+=loss.item()*b; tr_point+=lp.item()*b; tr_q+=lq.item()*b

        model.eval(); va_total=va_point=va_q=0.0; nv=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss, lp, lq = total_loss(model(xb), yb)
                b=len(xb); nv+=b; va_total+=loss.item()*b; va_point+=lp.item()*b; va_q+=lq.item()*b
        row={
            "epoch":epoch,
            "train_loss":tr_total/max(1,n),"train_mse":tr_point/max(1,n),"train_pinball":tr_q/max(1,n),
            "val_loss":va_total/max(1,nv),"val_mse":va_point/max(1,nv),"val_pinball":va_q/max(1,nv),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val=row["val_loss"]; best=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best)
    return model, history, time.perf_counter()-start


def predict_trustkan(model, loader, device=None):
    device = device or next(model.parameters()).device
    model.eval(); point=[]; quant=[]; embed=[]; truth=[]
    with torch.no_grad():
        for xb, yb in loader:
            out=model(xb.to(device))
            point.append(out["point"].cpu().numpy())
            quant.append(out["quantiles"].cpu().numpy())
            embed.append(out["embedding"].cpu().numpy())
            truth.append(yb.numpy())
    return {
        "point":np.concatenate(point),
        "quantiles":np.concatenate(quant),
        "embedding":np.concatenate(embed),
        "target":np.concatenate(truth),
    }
