# Claim–Evidence Matrix

This file prevents manuscript claims from outrunning experimental evidence.

| Candidate claim | Required evidence | Status |
|---|---|---|
| TrustKAN improves deterministic climate forecasting | Multi-dataset, multi-horizon comparison against strong baselines; >=5 seeds | **Refuted on CET.** Across 4 horizons x 5 seeds it never leads; the transformer is best at every horizon and the gap widens with lead time (see below) |
| Temporal KAN component adds value beyond a plain KAN | Controlled KAN vs TrustKAN ablation under identical budgets | **Refuted on CET.** Origin-blocked paired bootstrap over 20 matched pairs: h1 inconclusive 5/5, and plain KAN significantly better 5/5 at h7, h30 and h90 (mean RMSE gap +0.15, +0.52, +1.40) |
| TrustKAN provides calibrated uncertainty | Calibration/test separation; marginal and simultaneous coverage, width and interval score vs UQ baselines | **Partially supported, but not a contribution.** Marginal coverage lands on nominal (0.894/0.889/0.901/0.904) — this is the split-conformal guarantee and holds for any backbone. Simultaneous bands under-cover (0.870/0.850/0.862 at h7/h30/h90) and per-origin joint coverage collapses (0.609/0.256/0.103) |
| Adaptive calibration is robust to temporal shift | Rolling coverage and shift experiments vs static conformal | **Supported on CET.** A8 cuts rolling coverage deviation against static split conformal by 13.7%, 14.9%, 21.1% and 32.9% at h1/h7/h30/h90 for a width cost of only 2.5-4.6%, and the gain grows with lead time |
| KAN component is necessary at all | Budget-matched replacement of the KAN mapping by an MLP (A1) | **Refuted on CET.** A1 differs from the full model by at most 0.008 degC, inside the across-seed spread, at every horizon |
| KAN explanations are stable | Across-seed and perturbation stability metrics, not only visual examples | **Refuted on CET.** Index-matched curve correlation is +0.001/+0.001/+0.001/+0.000 at h1/h7/h30/h90, i.e. chance. Both charitable readings fail their controls: matched \|r\| is 0.780 against a shuffled-pair control of 0.780, and best-match 0.996 against a within-seed control of 0.996 |
| Representation changes help reveal climate shift | Quantitative drift protocol or carefully defined temporal diagnostics | Planned |
| Reliability score predicts forecast failure | Error-vs-reliability association and top-error AUROC/AUPRC against width-only and shift-only components | **Refuted on CET.** Top-error AUROC is 0.516/0.492/0.503/0.540 across h1/h7/h30/h90, i.e. chance. AUPRC matches the base rate and the error-reliability Spearman correlation is |rho| <= 0.12 |
| Selective forecasting reduces risk | Origin-wise risk–coverage curves, AURC, retained-set error and width-only/shift-only ablations | **Refuted on CET.** The fused score loses to its own width-only component at every horizon, and discarding half the origins cuts RMSE by only 1-3% |
| Method is robust to corrupted inputs | Noise, random missingness and block missingness experiments | **Refuted on CET.** TrustKAN is the most fragile of the three models at all 9 pre-registered levels: noise 3.5% vs 1.6% (plain KAN) and 2.3% (transformer), 40% random missingness 33.0% vs 17.4%/26.9%, one-day block 51.1% vs 25.6%/40.6%. Block cost saturates at 3 days (80.8% for both 3-day and 7-day, bit-identical in 100% of runs) because the readout only sees 3 trailing timesteps |
| Method is computationally practical | Parameter count, memory, training time and CPU/GPU inference latency | **Supported, but it explains nothing.** On one RTX 2060 TrustKAN uses 58,496 parameters, 58.4 s to train and 0.097 ms per origin, mid-pack on all three; the transformer that beats it at every horizon is the largest and slowest. Cost does not account for the accuracy ordering |
| Method is useful on climate extremes | Pre-defined extreme subsets and event-tail metrics with uncertainty coverage | **Refuted on CET.** On the frozen 5th/95th-percentile subsets TrustKAN is best at no horizon, trailing the transformer by 0.10, 0.19, 0.28 and 0.75 degC at h1/h7/h30/h90 |

## Rule
Do not mark a row Supported until raw experiment outputs, configuration, seeds and an analysis artifact all exist. A supported claim must link to its generated result files before inclusion in the abstract or conclusion.

## CET accuracy outcome, 2026-08

The first complete CET campaign refutes the accuracy claims. Test RMSE, mean of
five seeds, current code fingerprint `6426a567`:

| model | h1 | h7 | h30 | h90 |
|---|---|---|---|---|
| transformer | 1.989 | 2.679 | 2.908 | 2.960 |
| gru | 2.011 | 2.713 | 2.991 | 3.141 |
| lstm | 2.016 | 2.722 | 3.022 | 3.183 |
| kan (plain) | 2.075 | 2.822 | 3.000 | 3.026 |
| mlp | 2.121 | 2.827 | 2.967 | 3.019 |
| trustkan | 2.057 | 2.969 | 3.524 | 4.429 |
| persistence | 2.144 | 3.316 | 4.037 | 5.458 |

TrustKAN is mid-pack at one day and degrades faster than every learned baseline
as the lead time grows, ending worst of them at 90 days. Its 90-day error
(4.429) exceeds what persistence achieves at 30 days (4.037), though it stays
ahead of persistence at a matched horizon. These numbers already reflect the
corrected last-state readout; the A9 reproduction of the mean-pooled readout
falls behind persistence at 3 of the 4 horizons, by up to 1.80 degC, and edges
ahead only at h90 where persistence itself collapses.

The architecture therefore cannot be presented as an accuracy contribution on
this dataset. Any surviving accuracy statement must be scoped to one-day
forecasts and reported as a tie rather than a win.

## CET reliability outcome, 2026-08

Twenty reliability runs (four horizons, five seeds, fingerprint `f1b28739`)
also fail to support the trust claims.

| horizon | marginal cov. | joint cov. | simultaneous cov. | fused AURC | width-only AURC | shift-only AURC | error AUROC |
|---|---|---|---|---|---|---|---|
| 1 | 0.894 | 0.894 | 0.894 | 2.024 | **2.012** | 2.025 | 0.516 |
| 7 | 0.889 | 0.609 | 0.870 | 2.932 | **2.760** | 3.035 | 0.492 |
| 30 | 0.901 | 0.256 | 0.850 | 3.470 | **3.190** | 3.643 | 0.503 |
| 90 | 0.904 | 0.103 | 0.862 | 4.314 | **3.989** | 4.551 | 0.540 |

Three things stand out. Marginal coverage is on target, but split conformal
guarantees that by construction for any backbone, so it evidences the wrapper
rather than the method. The fused reliability score is beaten by its own
width-only ablation at every horizon, so the shift component and the frozen
0.5/0.5 fusion actively degrade a signal that interval width already carries.
And top-error AUROC sits at chance, so the score does not identify failing
forecasts; consistently, selecting the most reliable half of the origins moves
RMSE by only one to three percent.

## CET ablation outcome, 2026-08

The eighty trained ablation runs (A0, A1, A2, A9) and the A3-A8 recomputations
complete the picture.

A1 is the most damaging result in the study. Replacing the KAN mapping with a
budget-matched MLP, holding the temporal stem and every protocol element fixed,
moves RMSE by at most 0.008 degC at any horizon, which is inside the across-seed
spread. The KAN layer supplies the architecture's name and its interpretability
claim and contributes nothing measurable to accuracy.

A9 confirms the readout defect was real: it is 1.82, 2.15, 1.63 and 0.86 degC
worse than A0, with an across-seed standard deviation at h1 of 1.086 against the
corrected model's 0.004.

The evaluation-side gains, positive meaning the component earns its place:

| component | h1 | h7 | h30 | h90 |
|---|---|---|---|---|
| A3 conformal correction | +0.019 | +0.009 | +0.004 | +0.003 |
| A4 embedding shift | -0.012 | -0.171 | -0.280 | -0.325 |
| A5 interval width | +0.001 | +0.103 | +0.173 | +0.238 |
| A6 fusion over best component | -0.015 | -0.171 | -0.280 | -0.325 |
| A7 abstention | +0.039 | +0.036 | +0.040 | +0.130 |
| A8 adaptive conformal | +0.005 | +0.011 | +0.016 | +0.024 |

## CET extreme-subset outcome, 2026-08

Scored on the frozen subsets from the stored predictions, with no retraining.
Union of cold and warm origins, mean over five seeds, test RMSE in degC:

| model | h1 | h7 | h30 | h90 |
|---|---|---|---|---|
| transformer | 2.450 | 2.761 | 2.725 | 2.839 |
| kan (plain) | 2.560 | 2.943 | 2.765 | 2.853 |
| trustkan | 2.547 | 2.953 | 3.001 | 3.587 |

Two cautions for anyone reusing this table. The union is dominated by warm
origins (246 against about 55 cold), so it largely measures warm extremes; cold
origins are about a degree harder for every model. And at h30/h90 most models
score *better* on the extreme subset than on its complement, because warm
extremes here fall in summer when English temperature variance is lowest. Only
the between-model comparison on a fixed subset is meaningful.

Notably, TrustKAN's h90 extreme-subset error (3.587) is well below its
complement error (4.512), so its degradation is concentrated on ordinary days
rather than on the rare events the framework targeted.

## CET interpretability outcome, 2026-08

The benchmark runner kept predictions, not weights, so all 20 TrustKAN runs were
retrained under the frozen protocol and accepted only when their test RMSE
equalled the ledger value. All 20 reproduced exactly, which is what licenses
reading these curves as the reported models' curves.

Each run exposes 4,096 univariate curves. Cross-seed agreement, averaged over the
10 seed pairs per horizon:

| statistic | h1 | h7 | h30 | h90 | control |
|---|---|---|---|---|---|
| index-matched r | +0.001 | +0.001 | +0.001 | +0.000 | 0 by construction |
| matched \|r\| | 0.778 | 0.775 | 0.780 | 0.789 | shuffled pairs: 0.780 |
| best-match r | 0.9965 | 0.9964 | 0.9965 | 0.9964 | within-seed: 0.9963 |

Both charitable readings die against their controls. Matched curves correlate in
magnitude no better than curves paired at random, and a near-perfect best match
is equally available inside a single run, because 3 principal components span 95%
of the shape variance across all 4,096 curves. The curves are also barely curved:
median linear R^2 is 0.94 and 60% of curves exceed 0.9.

One limitation is structural, not statistical. The KAN layer consumes the output
of a convolutional stem, so its inputs are latent channels rather than observed
variables; even a stable curve would describe a coordinate of the model's own
making. No physical reading is drawn from these curves.

## Provenance of the retrained sweeps, audited 2026-08

The benchmark kept predictions, not weights, so three analyses retrain: the
corruption sweep, the KAN curve extraction, and the receptive-field measurement.
Retraining raises a fair objection — that these might be different models sharing
a protocol — so the binding was audited rather than assumed.

| check | benchmark | robustness | KAN curves | receptive field |
|---|---|---|---|---|
| records | 162 ok | 60 ok | 20 ok | 12 ok |
| `config_sha256` | `ebd6ad64e219` | same | same | same |
| `dataset_sha256` | `c54a9da8a66e` | same | same | same |
| device / torch | `cuda:0`, `2.13.0+cu126` | same | same | same |
| RMSE vs ledger | — | max gap 3.3e-07 | max gap 4.4e-16 | exact, gated |
| clean predictions vs ledger artifact | — | **bit-identical, all 60** | — | — |

The decisive line is the last. An RMSE match is a scalar agreement that two
similar models could produce by luck; an elementwise match of the whole test
prediction vector, in all 60 runs, cannot be produced by different weights. The
3.3e-07 RMSE gap is not a weight difference but a summation-order difference: the
sweep recomputes RMSE through the corruption-grid path while the ledger value
came from the benchmark path, and the underlying predictions are identical.

`code_sha256` differs across these campaigns by construction and is not
comparable: each runner hashes its own file list, so the robustness runner's
`b7702556671f` covers `run_robustness_campaign.py` and the benchmark's
`6426a567bf54` does not. The curve and receptive-field runners additionally store
`benchmark_code_sha256`, which does match the ledger's `6426a567bf54`.

**Environment.** All published runs come from one interpreter, `.venvs/trustkan`
(Python 3.12.13, torch 2.13.0+cu126, RTX 2060). A second environment exists at
`.venv` inside the repository with CPU-only torch and missing dependencies; it
has produced nothing and cannot, because the RMSE gate fails across torch builds
and device classes. `docs/RESULT_PRODUCTION_WORKFLOW.md` section 0 records this.
The hazard is real and worth keeping documented: bare `python` on `PATH`
resolves to a third installation with no torch at all.

The audit is encoded in `tests/test_robustness_provenance.py` so it cannot drift.

## CET robustness outcome, 2026-08

Sixty runs (3 models x 4 horizons x 5 seeds) retrained under the frozen protocol.
Each sweep begins with the clean history and is accepted only when that RMSE
equals the benchmark ledger value, so the corrupted entries are anchored to the
reported models. Percentage increase in test RMSE over the clean forecast,
averaged over horizons and seeds:

| corruption | level | trustkan | kan (plain) | transformer |
|---|---|---|---|---|
| noise | strongest pre-registered | 3.5% | 1.6% | 2.3% |
| random missing | 40% | 33.0% | 17.4% | 26.9% |
| block missing | 1 day | 51.1% | 25.6% | 40.6% |
| block missing | 3 days | 80.8% | 29.1% | 40.7% |
| block missing | 7 days | 80.8% | 32.2% | 41.9% |

TrustKAN is the most fragile model at every corruption and every level, so the
robustness claim fails in the same direction as the accuracy and reliability
claims. The block column also behaves impossibly: extending the mask from 3 to
7 days does not change TrustKAN's error at all, and the two are bit-identical in
100% of runs against 0% for both comparators.

Identical errors mean the extra masked days were never read. Perturbing one
timestep at a time and checking which perturbations move the forecast gives the
effective receptive field directly: 3 timesteps for TrustKAN against the full 365
for the plain KAN and the transformer. The configured history is a year and the
model sees three days of it.

That measurement is made on trained weights, not on the architecture in the
abstract. `scripts/run_receptive_field.py` retrains each model under the frozen
protocol, admits it only after its test RMSE reproduces the ledger, and repeats
this at all four horizons; the reach does not vary with horizon, and a randomly
initialized control of the same architecture returns the same number. Agreement
between the trained measurement and the control is what makes this a structural
property of the design rather than an artifact of one optimization run — and the
control is also why the earlier, untrained-only version of this measurement
happened to give the right answer. This is an architectural property of the dilated
convolutional stem, independent of the readout defect corrected earlier, and it
explains both the 90-day accuracy collapse and the block fragility. It also
qualifies the A1 and plain-KAN comparisons: they are sound as paired tests but do
not isolate the KAN mapping, since the models being compared see different
amounts of history.

## Summary

Of the five intended contributions, four are withdrawn: the temporal KAN module,
the embedding-shift term, the reliability fusion, and the abstention rule each
fail the ablation that was pre-specified to test them. The extreme-event and
robustness claims fail on the same evidence, and the interpretability claim fails
against its own controls. What survives belongs to the conformal wrapper
rather than to the proposed method: conformal correction improves coverage,
interval width alone drives useful selective forecasting, and adaptive conformal
materially improves coverage stability under drift.

The pre-registration is doing exactly the job it was written for. Reporting these
as negative results, rather than retuning the frozen fusion weights until a claim
passes, is the only option consistent with the project rules.
