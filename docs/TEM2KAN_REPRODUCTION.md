# Tem2-KAN reproduction note

## Source identity

The reference audit used `Temp_IKAN4.py` from
`YongxiangLei/kan_proj`, commit
`d4458e8fb11883b2eec0994aaaa6b7914e7f9e60`. The optional dependency is pinned
as `pykan==0.2.8` in `requirements-tem2kan.txt`, matching the public repository's
documented environment.

## Verified source specification

The source implementation uses:

- CET `Radcliffe` temperature;
- history length 300 and forecast horizon 20;
- `KAN(width=[300, 32, 64, 32, 20], k=10, grid=10)` for its final model;
- MSE loss, Adam with learning rate `1e-4`, and 200 full-batch epochs;
- the final 20% of windows as test data;
- global min–max normalization fitted before the split;
- no validation, calibration or random-seed study.

The same source also trains nine `k × grid` combinations and selects/displays
`k=10, grid=10` without a separately documented validation selection rule.

## Repository implementation

`Tem2KANReference` now defaults to strict mode and rejects any input/output shape
other than `(history=300, features=1, horizon=20)`. It passes the experiment seed
to pykan and disables pykan's automatic checkpoint-directory side effect.

`configs/cet_tem2kan.yaml` preserves the verified model width, basis order, grid,
target, optimizer, learning rate and epoch budget while applying this project's
fair, leakage-safe protocol. Run it with:

```bash
pip install -r requirements-tem2kan.txt
python scripts/run_cet_benchmark.py \
  --config configs/cet_tem2kan.yaml \
  --resume
```

## Deliberate protocol differences

The fair configuration does **not** reproduce known validity problems from the
source script:

- scaling is fitted on training observations only;
- `-99.90` missing-value sentinels are excluded;
- windows crossing missing calendar dates are excluded;
- train, validation, calibration and final-test regions remain separate;
- five initialization seeds are required;
- early-model selection is validation-based rather than test-based;
- raw predictions, timestamps, configuration hashes and per-run metrics are
  retained.

Consequently, results from `configs/cet_tem2kan.yaml` are an architectural
reproduction under a stricter experimental protocol, not a numerical
reproduction of the original reported values.

## Comparator naming policy

- `kan`: clean in-repository KAN baseline.
- `tem2kan`: strict verified Tem2-KAN architecture, only for 300→20 univariate
  experiments.
- `trustkan`: proposed reliability-aware model.

Do not resize `tem2kan` to arbitrary histories or horizons and continue calling
it a strict reproduction. A resized variant must be explicitly named
“Tem2-KAN-style” and documented as an adaptation.
