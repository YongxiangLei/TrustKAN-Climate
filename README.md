# TrustKAN-Climate

**TrustKAN-Climate: Interpretable and Reliability-Aware Kolmogorov–Arnold Networks for Trustworthy Climate Forecasting under Distribution Shift**

This repository is a research-grade project for developing, evaluating, and documenting a KAN-based trustworthy climate forecasting framework suitable for a top-tier journal submission.

## Scientific goal

Move beyond point-forecast accuracy by jointly studying:

- interpretable-by-design temporal KAN forecasting,
- calibrated predictive uncertainty,
- temporal drift and out-of-distribution detection,
- explanation stability,
- selective forecasting / abstention,
- robustness and computational efficiency.

## Evidence chain

Climate data → Temporal KAN → intrinsic interpretability → quantile forecasts → conformal calibration → drift/OOD awareness → reliability fusion → selective forecasting → trustworthy decision support.

## Current model families

Persistence, SVR, Random Forest, XGBoost, MLP, LSTM, GRU, TCN, Transformer, optional Mamba, standard KAN, Tem2-KAN reference, and TrustKAN.

## Current dataset strategy

- CET: long historical temperature continuity benchmark.
- GHCN-Daily: independent station/network generalization.
- Jena: high-frequency multivariate station weather.
- ERA5: multivariate reanalysis / shift experiments.

Dataset inclusion, provenance, licensing and anti-cherry-picking rules are defined in `docs/DATASET_PROTOCOL.md`.

## TrustKAN modules

- `src/models/trustkan.py`: temporal KAN point + quantile model.
- `src/training/trust_engine.py`: joint MSE + pinball training.
- `src/uncertainty/conformal.py`: split-conformal calibration.
- `src/drift/scores.py`: representation/residual shift scores.
- `src/interpretability/stability.py`: explanation stability metrics.
- `src/reliability/fusion.py`: reliability fusion and calibration-only threshold selection.
- `scripts/run_trustkan_reliability.py`: end-to-end trust experiment on prepared splits.

## Quick checks

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_cet_benchmark.py --config configs/cet_smoke.yaml
python scripts/aggregate_results.py \
  --input results/aggregated/cet_smoke_runs.csv \
  --outdir results/tables/cet_smoke
```

`configs/cet_smoke.yaml` intentionally uses only the latest 2,000 valid
observations and is for engineering validation only. It must not be used for
paper claims or benchmark tables; use `configs/cet.yaml` for full experiments.
Smoke and full runs use separate output namespaces, so validation runs cannot
overwrite publication candidates. Every successful run records a configuration
hash, target timestamps, training history, per-sample inference latency and the
path to its raw prediction artifact.

Before producing paper tables, require at least five unique seeds for every
stochastic model and validate every referenced raw artifact:

```bash
python scripts/run_cet_benchmark.py --config configs/cet.yaml --resume
python scripts/aggregate_results.py \
  --input results/aggregated/cet_full_runs.csv \
  --outdir results/tables/cet_full \
  --min-seeds 5
```

Paired model comparisons follow the dependence-aware protocol in
`docs/STATISTICAL_PROTOCOL.md`; overlapping forecast origins are evaluated with
moving-block bootstrap rather than IID resampling.

The paper-grade GHCN station panel is selected and audited without consulting
model performance:

```bash
python scripts/select_ghcn_candidates.py
python scripts/audit_ghcn_candidates.py --require-complete
python scripts/prepare_ghcn_panel.py
python scripts/audit_ghcn_windows.py --config configs/ghcn.yaml
```

The five frozen stations, raw archive hashes, fixed period, quality controls,
and observed eligibility evidence are tracked in
`configs/datasets/ghcn_frozen.yaml`. The separate `ghcn_example.yaml` remains a
negative engineering example and is not a paper dataset.
The preparation command creates one checksum-locked `.npz` artifact per region
and a manifest under `results/dataset_audits/`; both are generated evidence and
are intentionally excluded from Git.

Validate both GHCN evaluation paths with the engineering-only smoke run:

```bash
python scripts/run_ghcn_benchmark.py --config configs/ghcn_smoke.yaml
python scripts/aggregate_results.py \
  --input results/aggregated/ghcn_smoke_runs.csv \
  --outdir results/tables/ghcn_smoke
```

The paper protocol uses all five regions, 1/7/30-day horizons and five neural
seeds:

```bash
python scripts/run_ghcn_benchmark.py --config configs/ghcn.yaml --resume
python scripts/aggregate_results.py \
  --input results/aggregated/ghcn_full_runs.csv \
  --outdir results/tables/ghcn_full \
  --min-seeds 5 \
  --min-regions 5
```

It includes separate within-station and leave-one-region-out parameter-transfer
tasks. Exact leakage controls and reporting rules are defined in
`docs/GHCN_BENCHMARK_PROTOCOL.md`.

Run the engineering-only uncertainty and reliability check with:

```bash
python scripts/run_ghcn_reliability.py \
  --config configs/ghcn_reliability_smoke.yaml
python scripts/aggregate_reliability.py \
  --input results/reliability/aggregated/ghcn_reliability_smoke_runs.csv \
  --outdir results/tables/ghcn_reliability_smoke \
  --min-regions 2
```

The publication run uses all regions and seeds and fails closed on incomplete
panels:

```bash
python scripts/run_ghcn_reliability.py \
  --config configs/ghcn_reliability.yaml --resume
python scripts/aggregate_reliability.py \
  --input results/reliability/aggregated/ghcn_reliability_full_runs.csv \
  --outdir results/tables/ghcn_reliability_full \
  --min-seeds 5 \
  --min-regions 5
```

Marginal versus simultaneous coverage, target calibration in geographic
transfer, and selective-risk rules are frozen in
`docs/GHCN_RELIABILITY_PROTOCOL.md`.

For reliability experiments, prepare an `.npz` split containing `x_train`, `y_train`, `x_val`, `y_val`, `x_cal`, `y_cal`, `x_test`, and `y_test`, then run:

```bash
python scripts/run_trustkan_reliability.py \
  --split-file path/to/splits.npz \
  --out results/reliability/trustkan.json
```

## Reproducibility principles

All reported values must be generated from saved experiment outputs. No manually invented numbers, hidden failed runs, post-hoc dataset removal, or test-set leakage are allowed. Main neural comparisons should use multiple random seeds and fair tuning effort. Thresholds, fusion weights and calibration choices must be determined without inspecting final test outcomes.
