"""Evaluate pre-registered extreme subsets from saved forecast artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from src.extremes.subsets import evaluate_extremes


def load_array(path, key=None):
    packed = np.load(path, allow_pickle=False)
    if key is None:
        if "target" in packed.files:
            key = "target"
        elif "train_target" in packed.files:
            key = "train_target"
        else:
            raise ValueError(f"{path} must contain target or train_target")
    return packed[key]


def main(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)["extreme"]
    predictions = np.load(args.predictions, allow_pickle=False)
    train_target = load_array(args.train_target)
    result = evaluate_extremes(
        train_target,
        predictions["target"],
        predictions["prediction"],
        lower_quantile=policy["lower_quantile"],
        upper_quantile=policy["upper_quantile"],
        definition=policy["definition"],
        min_origins=policy["min_origins"],
    )
    result["predictions"] = Path(args.predictions).as_posix()
    result["train_target"] = Path(args.train_target).as_posix()
    result["thresholds_from"] = policy["thresholds_from"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Raw forecast NPZ with target and prediction")
    parser.add_argument("--train-target", required=True, help="NPZ containing training targets only")
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--out", default="results/extremes/summary.json")
    main(parser.parse_args())
