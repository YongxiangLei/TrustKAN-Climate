# Statistical comparison protocol

## Scope and estimands

All model comparisons must be declared before final-test inspection. The primary
paired estimands are differences in MAE and RMSE on identical forecast origins:

`metric(model A) - metric(model B)`.

Negative values favor model A. Report the point difference, a dependence-aware
confidence interval, and the raw metric values; do not replace effect magnitude
with a binary significance label.

## Resampling unit

The experimental pipeline produces overlapping chronological forecast windows.
Individual horizon elements and adjacent origins are therefore not IID. The
primary interval uses a circular moving-block bootstrap over forecast origins.
All horizon elements belonging to one origin remain together in every resample.

Unless a block length is pre-specified from domain knowledge or an
autocorrelation analysis that does not use model rankings, the implementation
uses:

`max(forecast horizon width, ceil(number of origins^(1/3)))`,

capped at the available number of origins. The chosen block length, rule,
bootstrap seed and number of resamples are stored in every comparison artifact.
Use at least 5,000 resamples for final tables; smaller values are smoke checks.

## Seed handling

Compare stochastic models using matching experiment seeds and identical target
timestamps. Deterministic models use sentinel seed `-1` and may be paired with
each stochastic seed. Report both within-seed paired intervals and the
across-seed distribution; do not pool predictions from different seeds as if
they were independent test cases.

## Sensitivity analyses

The origin-level Wilcoxon signed-rank test and paired standardized mean
difference are descriptive sensitivity analyses. Wilcoxon still assumes
independent paired origins and is not the primary inferential result for these
overlapping time-series forecasts.

## Multiple comparisons

Define the primary comparison family in advance, for example TrustKAN versus
each baseline for one dataset/metric/horizon family. If p-values are reported,
apply Holm family-wise correction to the pre-specified primary family.
Benjamini–Hochberg may be reported separately for clearly labeled exploratory
families. Never select the correction family after viewing unadjusted results.
The repository provides `adjust_pvalues(..., method="holm" | "bh")`.

## Artifact integrity

`scripts/compare_models.py` refuses a paired comparison unless target values,
timestamps, origins, dataset, horizon, split, configuration fingerprint and
source-code fingerprint agree. For two stochastic models, seeds must also
match. Each output records the two input artifact paths and all relevant
fingerprints.

Example:

```bash
python scripts/compare_models.py \
  --a results/raw/cet_full/cet_trustkan_h7_s11.npz \
  --b results/raw/cet_full/cet_transformer_h7_s11.npz \
  --block-length 30 \
  --n-boot 5000 \
  --seed 2026 \
  --out results/statistical_tests/cet_h7_s11_trustkan_vs_transformer.json
```

## Interpretation guardrails

- A confidence interval excluding zero is not proof of practical importance.
- A small average gain can hide failures during extremes or shifted periods.
- Dependence-aware uncertainty does not repair post-hoc dataset, metric or
  baseline selection.
- Statistical evidence must be consistent with raw predictions and the
  claim–evidence matrix before entering the abstract or conclusion.
