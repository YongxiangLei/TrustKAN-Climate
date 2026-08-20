"""Apply the frozen robustness grid to saved test histories.

The default predictor is persistence, which uses only the last uncorrupted or
corrupted history step. Learned models should be scored by passing their
predictions through the same library after a separate inference pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from src.models.baselines import PersistenceForecaster
from src.robustness.evaluation import evaluate_corruption_grid


def main(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    packed = np.load(args.split_file, allow_pickle=False)
    history = packed["x_test"]
    target = packed["y_test"]
    if args.predictor == "persistence":
        horizon = target.shape[1] if target.ndim == 2 else 1
        model = PersistenceForecaster(horizon)
        predict_fn = lambda item: model.predict(item).numpy()
    else:
        raise ValueError("Only --predictor persistence is built in; score other models via the library")
    rows, arrays = evaluate_corruption_grid(
        history,
        target,
        predict_fn,
        cfg,
        frequency=args.frequency,
        seed=cfg["corruption"]["seed"],
        fill=cfg["corruption"]["fill_value"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez_compressed(out.with_suffix(".npz"), **arrays, y_test=target)
    payload = {"rows": rows, "config": args.config, "predictor": args.predictor}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True, help="NPZ with x_test and y_test")
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--frequency", default="daily")
    parser.add_argument("--predictor", default="persistence")
    parser.add_argument("--out", default="results/robustness/persistence.csv")
    main(parser.parse_args())
