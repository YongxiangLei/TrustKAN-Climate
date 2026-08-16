# Codex execution tasks

Use these prompts sequentially. Read `AGENTS.md` first for every task.

## Task 1 — Audit Phase 1

Read the repository and inspect the existing CET benchmark code. Do not change scientific claims. Run the test suite, inspect temporal splitting and scaling, and verify that no forecast target crosses split boundaries. Fix only reproducibility/leakage bugs. Write `docs/PHASE1_AUDIT.md` summarizing findings.

## Task 2 — Reproduce the CET baseline

Run `scripts/run_cet_benchmark.py` with the CET configuration. If full training is expensive, first run a smoke configuration with one horizon, one seed and 2 epochs, then run the full configuration. Preserve all raw predictions. Do not alter test results manually.

## Task 3 — Port Tem2-KAN faithfully

Use the prior implementation only as a source implementation. Refactor it into `src/models/tem2kan.py` while preserving the documented architecture. Do not silently relabel the proposed TrustKAN as Tem2-KAN. Add a shape test and a config entry. Record any architectural ambiguity in `docs/PHASE1_AUDIT.md`.

## Task 4 — Add classical baselines

Add Persistence, seasonal persistence where justified by sampling frequency, ARIMA/SARIMA, SVR and XGBoost. Keep hyperparameter selection confined to train/validation. Save the search space and selected configuration.

## Task 5 — Add modern sequence baselines

Add TCN and Mamba if the environment supports a stable implementation. If Mamba dependencies are unavailable, document the blocker rather than fabricating a result.

## Task 6 — Benchmark integrity

Produce a single tidy result schema with columns:
`dataset, model, horizon, seed, split, rmse, mae, parameters, train_seconds, inference_ms`.
Generate `results/aggregated/cet_summary.csv` only from per-run outputs.

## Task 7 — Reviewer audit

Act as a skeptical top-journal reviewer. Check whether the CET-only evidence can support any broad climate claim. Identify missing datasets, unfair comparisons, weak baselines, insufficient tuning, missing statistics, and any unsupported novelty statements. Write `docs/REVIEWER_AUDIT.md`.

## Stop condition for Phase 1

Do not begin manuscript result-writing until:
- leakage tests pass,
- CET benchmark runs end-to-end,
- plain KAN and Tem2-KAN are distinct from TrustKAN,
- at least five seeds are available for main neural models,
- raw predictions are saved,
- result aggregation is reproducible.
