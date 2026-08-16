"""Preflight a CUDA environment before running publication experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import environment
from src.training.engine import resolve_device, set_seed


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "src" / "training" / "engine.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def nvidia_smi_metadata():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "rows": []}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "rows": rows,
        "stderr": result.stderr.strip() or None,
    }


def deterministic_training_probe(device, seed=20260816):
    device = resolve_device(device)

    def train_once():
        set_seed(seed, deterministic=True, warn_only=False)
        model = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(8 * 16, 3),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        x = torch.randn(12, 2, 16, generator=generator).to(device)
        y = torch.randn(12, 3, generator=generator).to(device)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        state = torch.cat(
            [parameter.detach().cpu().reshape(-1) for parameter in model.parameters()]
        ).numpy()
        prediction = model(x).detach().cpu().numpy()
        return state, prediction, float(loss.detach().cpu())

    state_a, prediction_a, loss_a = train_once()
    state_b, prediction_b, loss_b = train_once()
    exact = np.array_equal(state_a, state_b) and np.array_equal(
        prediction_a, prediction_b
    )
    return {
        "seed": seed,
        "exact_parameter_replay": bool(np.array_equal(state_a, state_b)),
        "exact_prediction_replay": bool(np.array_equal(prediction_a, prediction_b)),
        "exact_replay": bool(exact),
        "loss_first": loss_a,
        "loss_second": loss_b,
        "max_parameter_difference": float(np.max(np.abs(state_a - state_b))),
        "max_prediction_difference": float(
            np.max(np.abs(prediction_a - prediction_b))
        ),
    }


def check_environment(device="cuda:0", *, allow_cpu=False):
    resolved = resolve_device(device)
    if resolved.type != "cuda" and not allow_cpu:
        raise RuntimeError("A CUDA device is required for the GPU publication preflight")
    probe = deterministic_training_probe(resolved)
    if not probe["exact_replay"]:
        raise RuntimeError("The deterministic training replay check failed")
    details = environment(resolved)
    if resolved.type == "cuda":
        details.update(
            cuda_memory_allocated_bytes=int(torch.cuda.memory_allocated(resolved)),
            cuda_memory_reserved_bytes=int(torch.cuda.memory_reserved(resolved)),
        )
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_code_sha256": code_sha256(),
        "requested_device": str(device),
        "resolved_device": str(resolved),
        "environment": details,
        "nvidia_smi": nvidia_smi_metadata(),
        "deterministic_training_probe": probe,
        "eligible_for_publication_gpu_run": resolved.type == "cuda"
        and probe["exact_replay"],
    }


def main(args):
    payload = check_environment(args.device, allow_cpu=args.allow_cpu)
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    print(f"Preflight: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out", default="results/campaigns/ghcn_publication/gpu_environment.json"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Exercise the checker on CPU without marking it GPU-eligible.",
    )
    main(parser.parse_args())
