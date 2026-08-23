# Result Production Workflow

This project treats manuscript tables and figures as deterministic products of saved experiment outputs.

## 0. Use the authoritative environment

Every published run was produced by one interpreter, and it is the only one that
can reproduce them:

```
C:\Users\79441\Documents\Codex\.venvs\trustkan\Scripts\python.exe
Python 3.12.13, torch 2.13.0+cu126, CUDA available, NVIDIA GeForce RTX 2060
```

This is not a preference. Runners that retrain a reported model
(`run_robustness_campaign.py`, `run_kan_curves.py`, `run_receptive_field.py`)
refuse their results unless the retrained test RMSE reproduces the benchmark
ledger to within `1e-6`. Deterministic training reproduces bit-exactly on the
same torch build and device class and does not reproduce across a different one,
so running these under any other interpreter does not produce a wrong number: it
produces a hard failure. That is the intended behaviour, and it is why the gate
exists.

**Known hazard.** A second virtual environment exists inside the repository at
`.venv`. It carries a CPU-only torch build (`2.13.0+cpu`) and, as of 2026-08, was
missing `pandas`, `scikit-learn`, `matplotlib`, `tqdm` and `statsmodels`
entirely. It has never produced a published run. Two environments in one project
is a reproducibility hazard: `python` on `PATH` resolves to neither of them but
to an unrelated Anaconda installation that has no torch at all. Always invoke the
authoritative interpreter by absolute path, and never by a bare `python`.

Analysis-only scripts that read artifacts (`analyze_robustness.py`,
`make_paper_tables.py`, `make_paper_figures.py`, `make_paper_bundle.py`) do not
retrain and so are not gated, but they should still be run under the
authoritative interpreter so that a single environment accounts for everything
in `paper/`.

To check what produced any existing result, read its run record: every record
under `results/**/runs/` stores `python`, `torch`, `torch_cuda`, `device` and
`cuda_device_name` alongside the `code_sha256`, `config_sha256` and
`dataset_sha256` fingerprints. `results/runs/cet_full/` shows `cuda:0` with
`2.13.0+cu126` for all neural runs; classical baselines record `cpu`, which is
expected because they never touch the GPU.

## 0b. Produce and validate deterministic benchmark runs

For the 1,410-run GHCN publication campaign, first generate the immutable shard
manifest described in `docs/GHCN_CAMPAIGN.md`. Final collection must pass both
the five-seed/five-region aggregators and the campaign audit with
`--require-complete` before any values enter the manuscript.
Neural publication shards must additionally pass the CUDA environment gate in
`docs/GPU_EXECUTION.md`; CPU fallback results cannot fill GPU campaign cells.

Full CET experiments write raw artifacts under `results/raw/cet_full/` and a
run ledger to `results/aggregated/cet_full_runs.csv`. Smoke outputs are isolated
under the `cet_smoke` namespace and are never publication candidates. The
188-run CET publication campaign is planned by `scripts/plan_cet_campaign.py`
as described in `docs/CET_CAMPAIGN.md`. The 141-run Jena publication campaign
is planned by `scripts/plan_jena_campaign.py` as described in
`docs/JENA_CAMPAIGN.md`.

```bash
python scripts/run_cet_benchmark.py --config configs/cet.yaml --resume
python scripts/aggregate_results.py \
  --input results/aggregated/cet_full_runs.csv \
  --outdir results/tables/cet_full \
  --min-seeds 5
```

Aggregation fails if a run key is duplicated, a successful row has no raw
artifact, artifact metadata disagrees with the ledger, arrays have inconsistent
shapes or non-finite values, a non-test row is supplied, or a stochastic model
has fewer than the required number of seeds. `inference_ms` is wall-clock
latency per test sample over the complete prediction pass; GPU timing is
synchronized before and after prediction.

Each model/horizon/seed run is written immediately as an atomic JSON record
under `results/runs/<experiment>/`. `--resume` reuses a successful record only
when its configuration hash, source-code hash and raw artifact match the active
experiment; failed or stale runs are executed again.

Classical candidate sets are declared under `model_search` in the experiment
configuration. Each candidate is trained on the training region and ranked by
validation RMSE. Candidate metrics, fit times and selected hyperparameters are
stored in both the run ledger and raw artifact; final-test targets never enter
selection.

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

## 4. Score pre-registered robustness and extremes

Extreme labels and corruption levels come from `configs/robustness.yaml`.
Training-quantile thresholds and test-history corruptions are never chosen from
final rankings.

```bash
python scripts/evaluate_extremes.py \
  --predictions results/raw/cet_full/cet_trustkan_h1_s11.npz \
  --train-target results/splits/cet_train_target.npz \
  --out results/extremes/cet_trustkan_h1.json

python scripts/evaluate_robustness.py \
  --split-file results/splits/cet_h1.npz \
  --frequency daily \
  --out results/robustness/cet_persistence.csv
```

The CET corruption sweep and the receptive-field measurement both need weights,
which the benchmark did not retain, so both retrain under the frozen protocol and
accept a run only when its test RMSE reproduces the ledger. Run them under the
authoritative interpreter of section 0; under any other they fail the gate by
design.

```bash
python scripts/run_robustness_campaign.py --resume
python scripts/run_receptive_field.py
python scripts/analyze_robustness.py --outdir paper/tables
```

`analyze_robustness.py` reads both artifacts and refuses to report a receptive
field that varies with horizon, since a number that did would describe a trained
instance rather than the architecture.

Jena preparation is independent of model performance. After the official
archives pass eligibility, plan the 141-run shard campaign rather than
editing the frozen configs:

```bash
python scripts/audit_jena.py --manifest data/manifests/jena_sources.csv --require-eligible
python scripts/prepare_jena.py --require-eligible
python scripts/audit_jena_windows.py --config configs/jena.yaml
python scripts/plan_jena_campaign.py \
  --outdir results/campaigns/jena_publication
python scripts/audit_jena_campaign.py \
  --campaign results/campaigns/jena_publication/campaign.json \
  --out results/campaigns/jena_publication/progress.json
```

## 5. Generate paper artifacts

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
