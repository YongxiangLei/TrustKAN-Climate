# Claim–Evidence Matrix

This file prevents manuscript claims from outrunning experimental evidence.

| Candidate claim | Required evidence | Status |
|---|---|---|
| TrustKAN improves deterministic climate forecasting | Multi-dataset, multi-horizon comparison against strong baselines; >=5 seeds | **Refuted on CET.** Across 4 horizons x 5 seeds it never leads; the transformer is best at every horizon and the gap widens with lead time (see below) |
| Temporal KAN component adds value beyond a plain KAN | Controlled KAN vs TrustKAN ablation under identical budgets | **Refuted on CET.** Origin-blocked paired bootstrap over 20 matched pairs: h1 inconclusive 5/5, and plain KAN significantly better 5/5 at h7, h30 and h90 (mean RMSE gap +0.15, +0.52, +1.40) |
| TrustKAN provides calibrated uncertainty | Calibration/test separation; marginal and simultaneous coverage, width and interval score vs UQ baselines | **Partially supported, but not a contribution.** Marginal coverage lands on nominal (0.894/0.889/0.901/0.904) — this is the split-conformal guarantee and holds for any backbone. Simultaneous bands under-cover (0.870/0.850/0.862 at h7/h30/h90) and per-origin joint coverage collapses (0.609/0.256/0.103) |
| Adaptive calibration is robust to temporal shift | Rolling coverage and shift experiments vs static conformal | **Supported on CET.** A8 cuts rolling coverage deviation against static split conformal by 13.7%, 14.9%, 21.1% and 32.9% at h1/h7/h30/h90 for a width cost of only 2.5-4.6%, and the gain grows with lead time |
| KAN component is necessary at all | Budget-matched replacement of the KAN mapping by an MLP (A1) | **Refuted on CET.** A1 differs from the full model by at most 0.008 degC, inside the across-seed spread, at every horizon |
| KAN explanations are stable | Across-seed and perturbation stability metrics, not only visual examples | Planned. Note any such analysis now describes a mapping that A1 shows is functionally interchangeable with an MLP |
| Representation changes help reveal climate shift | Quantitative drift protocol or carefully defined temporal diagnostics | Planned |
| Reliability score predicts forecast failure | Error-vs-reliability association and top-error AUROC/AUPRC against width-only and shift-only components | **Refuted on CET.** Top-error AUROC is 0.516/0.492/0.503/0.540 across h1/h7/h30/h90, i.e. chance. AUPRC matches the base rate and the error-reliability Spearman correlation is |rho| <= 0.12 |
| Selective forecasting reduces risk | Origin-wise risk–coverage curves, AURC, retained-set error and width-only/shift-only ablations | **Refuted on CET.** The fused score loses to its own width-only component at every horizon, and discarding half the origins cuts RMSE by only 1-3% |
| Method is robust to corrupted inputs | Noise, random missingness and block missingness experiments | Protocol frozen; full evidence pending |
| Method is computationally practical | Parameter count, memory, training time and CPU/GPU inference latency | Timing pipeline validated; full evidence pending |
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
| persistence | 2.144 | 3.316 | 4.037 | - |

TrustKAN is mid-pack at one day and degrades faster than every baseline as the
lead time grows, ending worst of all models at 90 days. These numbers already
reflect the corrected last-state readout; the earlier mean-pooled readout was
worse still and lost to persistence everywhere.

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

## Summary

Of the five intended contributions, four are withdrawn: the temporal KAN module,
the embedding-shift term, the reliability fusion, and the abstention rule each
fail the ablation that was pre-specified to test them. The extreme-event claim
fails on the same evidence. What survives belongs to the conformal wrapper
rather than to the proposed method: conformal correction improves coverage,
interval width alone drives useful selective forecasting, and adaptive conformal
materially improves coverage stability under drift.

The pre-registration is doing exactly the job it was written for. Reporting these
as negative results, rather than retuning the frozen fusion weights until a claim
passes, is the only option consistent with the project rules.
