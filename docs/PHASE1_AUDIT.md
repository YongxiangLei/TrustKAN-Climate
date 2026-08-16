# Phase 1 reproducibility and leakage audit

Date: 2026-08-16

## Scope

This audit covers the CET benchmark's chronological splitting, preprocessing,
window construction, script entrypoint, test suite, and smoke execution. It does
not validate scientific superiority or support any manuscript result claim.

## Verification performed

- Installed `requirements.txt` into an isolated Python 3.12 environment.
- Ran the original suite: 23 tests passed.
- Inspected the CET preprocessing and model loop in
  `scripts/run_cet_benchmark.py`.
- Verified that the standardizer is fitted on `values[split.train]` only.
- Verified contiguous train/validation/calibration/test slices and no shuffling
  during window construction.
- Ran the strengthened suite after result-integrity and Tem2-KAN fixes: 39 tests passed.
- Ran the documented smoke benchmark end-to-end and verified that seven model
  runs produced raw prediction archives plus aggregated CSV outputs.

## Findings and fixes

### P1 — Multi-step split safety depended on a caller-side filter (fixed)

`assign_windows_by_target_origin` originally assigned a window from only its
first target timestamp. A multi-step target near a boundary could therefore be
assigned to one region while ending in the next region unless every caller
remembered a second filter. The CET runner did apply that filter, so no leakage
was observed in that path, but the shared API was unsafe for reuse.

The function now accepts the forecast horizon and selects a window only when its
complete target lies within one split. The redundant CET-specific filter was
removed. A regression test covers all four boundaries for horizon 3.

### P1 — Documented script command failed to import `src` (fixed)

Running `python scripts/run_cet_benchmark.py ...` from a clean checkout failed
with `ModuleNotFoundError: src` because Python placed `scripts/`, rather than the
repository root, first on the module path. A small shared bootstrap now adds the
repository root for all scripts that import project modules. A subprocess test
verifies the file-path entrypoint.

### P2 — Small datasets could produce empty rounded splits (fixed)

Positive split fractions did not guarantee non-empty segments after integer
rounding. `chronological_split` now rejects configurations that produce any empty
segment, with a regression test.

### P2 — Smoke configuration was not operationally small (fixed)

The previous smoke configuration trained seven models on the complete CET
series and was too slow for a routine CPU check. The runner now supports an
optional `dataset.max_observations`, and the smoke configuration explicitly uses
the latest 2,000 valid observations. The README marks this subset as
engineering-only and prohibits its use for paper claims.

### P1 — Smoke and full experiments shared output names (fixed)

Both configurations previously wrote the same raw filenames and CET CSV paths,
allowing an engineering run to overwrite publication candidates. Configurations
now declare separate `cet_smoke` and `cet_full` experiment namespaces.

### P1 — Successful result rows were not linked to auditable artifacts (fixed)

The run ledger now includes the test split, per-sample inference latency,
configuration and source-code SHA-256 values, and raw artifact path. Raw artifacts include target
timestamps, origins, run metadata and neural training history. The strict
aggregator checks run-key uniqueness, artifact presence, metadata agreement,
array shapes, finite values and a configurable minimum seed count.

### P2 — Downloaded CET data had no integrity check (fixed)

All CET configurations use an immutable source-commit URL and pin the observed
SHA-256 of the raw CSV. Cached and newly downloaded data are rejected when the
checksum differs.

### P1 — Observation steps did not always equal calendar days (fixed)

The valid Pershore series contains 14 non-daily intervals, including a maximum
gap of 235 days. Dropping missing targets and then constructing array-indexed
windows could therefore mislabel observation steps as day horizons. Window
construction now accepts timestamps and excludes any history/target span whose
increments differ from the configured daily frequency. Regression tests cover
cross-gap exclusion and timestamp alignment.

### P1 — Long runs were not interruption-resilient (fixed)

Per-run metrics are now written immediately as atomic JSON records. `--resume`
reuses only successful records with the active configuration and source-code
hashes and an existing raw artifact. A changed config or implementation
invalidates stale records; failed runs are retried.

### P1 — Tem2-KAN identity and dependency were ambiguous (fixed in code)

The public source was audited at commit
`d4458e8fb11883b2eec0994aaaa6b7914e7f9e60`. Strict mode now enforces the verified
300→20 univariate architecture, passes the experiment seed to pykan, disables
automatic checkpoint side effects and pins `pykan==0.2.8`. A dedicated fair
protocol is provided in `configs/cet_tem2kan.yaml`; full five-seed execution is
still pending.

### P1 — Classical defaults lacked a validation selection trail (fixed in code)

The formal CET config now pre-declares SVR and random-forest candidate sets.
Candidates are fit only on training windows, selected by validation RMSE, and
their complete selection trace is saved in the raw artifact. Test targets are
not used for selection. Full executions remain pending.

## Leakage assessment

- Scaling: pass; training observations only.
- Temporal ordering: pass; contiguous splits and ordered windows.
- Target-boundary isolation: pass after the shared API hardening.
- Calibration/test separation: structurally present in the CET split.
- Hyperparameter leakage: not established by this audit. Classical models are
  currently fit directly without a recorded validation search protocol.

## Smoke-run interpretation

The smoke run is only an integration test: one seed, two epochs, one horizon,
and a 2,000-observation tail subset. Its metrics are not scientifically
meaningful, must not enter manuscript tables, and do not support comparative
claims. The run confirmed raw `.npz` prediction saving and CSV aggregation.

## Phase 1 stop-condition status

- Leakage tests pass: **yes**.
- CET benchmark runs end-to-end: **yes, smoke configuration only**.
- Plain KAN, Tem2-KAN, and TrustKAN are distinct: **yes in code; strict Tem2-KAN
  source mapping and shape are verified, but full reproduction results remain pending**.
- At least five seeds for main neural models: **no**.
- Raw predictions saved: **yes for smoke; full runs pending**.
- Reproducible aggregation: **smoke path works; schema still needs integrity
  review before paper production**.

Do not begin manuscript result writing. The repository has not yet met all Phase
1 stop conditions.
