# Result Production Workflow

This project treats manuscript tables and figures as deterministic products of saved experiment outputs.

## 1. Train TrustKAN and create trust artifacts

```bash
python scripts/run_trustkan_reliability.py \
  --split-file results/splits/<dataset>_h<horizon>.npz \
  --out results/reliability/<dataset>_h<horizon>.json
```

This creates:

- `<stem>_reliability.npz`: target, point prediction, reliability components and selected mask;
- `<stem>_conformal_input.npz`: calibration/test quantile bounds for static/rolling/adaptive conformal analysis;
- `<stem>_risk_coverage.npz`: selective prediction risk-coverage arrays.

## 2. Run rolling/adaptive conformal drift analysis

```bash
python scripts/run_rolling_drift.py \
  --input results/reliability/<dataset>_h<horizon>_conformal_input.npz \
  --out results/drift/<dataset>_h<horizon>_adaptive.json
```

Primary outcomes:

- global empirical coverage;
- mean interval width;
- rolling coverage deviation from nominal coverage;
- adaptive alpha trajectory.

The sequential implementation is causal: current labels are incorporated only after the current interval is issued.

## 3. Reliability-error calibration analysis

```bash
python scripts/analyze_reliability.py \
  --input results/reliability/<dataset>_h<horizon>_reliability.npz \
  --out results/reliability/<dataset>_h<horizon>_reliability_error.json
```

Primary outcomes:

- Spearman/Pearson reliability-error association;
- error by reliability bin;
- monotonicity of binned error;
- AUROC/AUPRC for identifying top-tail forecast errors using low reliability.

A useful reliability score should normally have a negative reliability-error association, lower error in higher-reliability bins, and useful discrimination of large-error cases. These are empirical criteria, not assumptions.

## 4. Generate paper artifacts

```bash
python scripts/generate_paper_artifacts.py \
  --benchmark results/aggregated/all_runs.csv \
  --drift-npz results/drift/<dataset>_h<horizon>_adaptive.npz \
  --reliability-samples results/reliability/<dataset>_h<horizon>_reliability_error_samples.npz \
  --reliability-bins results/reliability/<dataset>_h<horizon>_reliability_error_bins.csv
```

Outputs are written to `paper/tables/` and `paper/figures/`.

## Scientific safeguards

1. Never tune adaptive-conformal hyperparameters on final test performance.
2. Report static conformal as a mandatory comparator.
3. Rolling coverage must preserve chronological order.
4. For overlapping multi-horizon forecasts, explicitly define the ordering/evaluation unit; one horizon per adaptive experiment is preferred initially.
5. Reliability-error analysis must use the same samples that generated the reliability score.
6. Do not claim adaptive conformal is superior unless it improves coverage stability without an unacceptable width penalty.
7. Do not claim the reliability score is meaningful unless it predicts realized error or improves selective-risk metrics.
