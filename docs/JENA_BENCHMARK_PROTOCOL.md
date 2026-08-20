# Jena hourly multivariate benchmark protocol

## Scientific role

Jena tests a sampling regime that CET and GHCN cannot: high-frequency
multivariate station meteorology. The target remains air temperature so the
three observational datasets stay comparable, but the inputs include lagged
temperature plus physically distinct pressure, humidity and wind observations.

## Frozen station and period

The series is the official MPI-BGC Beutenberg station
(`50.91°N`, `11.56°E`). The publication window is 2010-01-01 00:00 through
2020-12-31 23:00 local naive timestamps. The start is after the 2003 station
opening and avoids the earliest years; the end is a complete calendar year.
This window was chosen before any model comparison.

Official 10-minute ZIP/CSV archives must be downloaded from
https://weather.bgc-jena.mpg.de/weather_data.html and listed with SHA-256
values in `data/manifests/jena_sources.csv`. The repository does not vendor the
raw archives.

## Variables and exclusions

Required columns, in this order:

1. `T (degC)` — target and lagged input
2. `p (mbar)` — station pressure
3. `rh (%)` — relative humidity
4. `wv (m/s)` — wind speed

Derived thermodynamic columns (`Tpot`, `Tdew`, vapour-pressure terms, specific
humidity and air density) are excluded because they are near-deterministic
transforms of temperature and humidity. Precipitation and radiation are not in
the required set because they are not present in every official 10-minute
schema year inside the frozen window.

## Quality control and aggregation

10-minute values equal to `-9999`, non-finite, or outside the physical ranges in
`configs/datasets/jena_frozen.yaml` are set to missing. Hours are retained only
when every required variable has at least four valid 10-minute slots. The hourly
value is the mean of those valid slots. Missing hours are not interpolated.
Duplicate local hours (autumn DST) keep the first row; spring-forward gaps
remain gaps and are later excluded by the window filter.

## Eligibility

Against the frozen 2010--2020 hourly window the prepared series must have at
least 10 years of span, 95% completeness, and a maximum gap of 48 hours. If the
official archives fail this gate, disclose the failed checks and extend the
period only by a pre-registered calendar rule. Do not drop or replace the
station because a model performs poorly.

Observed completeness, gap statistics and archive hashes are written by
`scripts/audit_jena.py` and `scripts/prepare_jena.py`. Those observed numbers
must not be typed into this file until an eligibility audit exists.

## Temporal construction

Chronological 60/15/10/15 splitting is applied to the hourly series. History is
168 hours. Horizons are 1, 6 and 24 hours. Windows whose history or target
crosses a non-hourly step are excluded. Standardization uses training hours
only. The target inverse uses the training moments of `T (degC)`.

## Reporting

Report MAE and RMSE in °C per horizon, with five neural seeds. Smoke
configurations may use a tail subset and are not publication candidates. Do not
begin Jena manuscript tables until the eligibility audit, window audit and
five-seed/three-horizon matrix exist.

The executable runner is `scripts/run_jena_benchmark.py`. It refuses prepared
artifacts whose feature order, frozen-config hash or preparation-code hash
differs from the tracked protocol. Persistence uses the most recent hourly
temperature only. Neural models receive all four standardized features.
`--resume` reuses a record only when config, code, dataset and artifact
checksums match. Split exports under `results/splits/` are for later
robustness and extreme-subset scoring; they are not manuscript tables.

Target-calibrated TrustKAN reliability uses `scripts/run_jena_reliability.py`
with the same conformal policy as GHCN/CET (`alpha=0.10`, quantiles
`0.05/0.5/0.95`, fusion weights `[0.5, 0.5]`). Target inversion uses the
training-only moments of `T (degC)`, not a univariate scaler fit on all four
features.

The 141-run publication campaign is planned by `scripts/plan_jena_campaign.py`
and audited by `scripts/audit_jena_campaign.py`. See `docs/JENA_CAMPAIGN.md`.

```bash
python scripts/run_jena_benchmark.py --config configs/jena_smoke.yaml --resume
python scripts/run_jena_reliability.py --config configs/jena_reliability_smoke.yaml --resume
python scripts/aggregate_results.py \
  --input results/aggregated/jena_smoke_runs.csv \
  --outdir results/tables/jena_smoke

python scripts/plan_jena_campaign.py \
  --benchmark-config configs/jena.yaml \
  --reliability-config configs/jena_reliability.yaml \
  --outdir results/campaigns/jena_publication
```
