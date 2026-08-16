# TrustKAN Falsifiable Ablation Plan

The proposed method should survive reviewer scrutiny by tying every claimed contribution to a removable module and a predefined metric. Modules that do not add credible evidence should be removed from the headline method.

| ID | Variant | Hypothesis | Primary metric | Failure criterion |
|---|---|---|---|---|
| A0 | Full TrustKAN | Integrated system improves reliability without unacceptable accuracy loss | RMSE + AURC + coverage | reference |
| A1 | Replace temporal KAN with MLP/temporal encoder | KAN mapping adds useful nonlinear structure | RMSE, params, explanation stability | no practical/statistical gain |
| A2 | No quantile head | predictive intervals add uncertainty information | interval score / coverage | calibrated intervals no better than baseline |
| A3 | No conformal calibration | conformal correction improves empirical coverage | coverage deviation, interval width | coverage not improved or width cost excessive |
| A4 | No embedding shift score | representation shift helps identify unreliable samples | reliability-error association / AURC | no AURC improvement |
| A5 | No interval-width reliability | interval sharpness contributes useful trust evidence | AURC / selective RMSE | no gain over shift-only reliability |
| A6 | No reliability fusion | combining independent signals outperforms individual signals | AURC | fusion not better than best component |
| A7 | No abstention | selective prediction reduces retained-set risk | risk–coverage curve | no meaningful risk reduction |
| A8 | Static vs adaptive conformal | adaptation improves coverage under temporal shift | rolling coverage deviation | adaptive method does not improve shifted periods |
| A9 | Explanation stability disabled from reliability | explanation change adds information beyond embedding drift | AURC / error correlation | redundant with existing signals |

## Experimental discipline

1. All thresholds and fusion weights are chosen from validation/calibration data only.
2. Test data may be evaluated only after hyperparameters and inclusion criteria are frozen.
3. Report negative ablations; do not hide modules that fail.
4. Use identical seeds and paired test samples when comparing variants.
5. Distinguish statistical significance from practical significance.
6. If A9 fails, explanation stability can remain an interpretability analysis but must not be claimed as a reliability mechanism.

## Headline novelty gate

The integrated method should not be called `TrustKAN` in the final paper merely because it includes KAN plus post-hoc tools. The paper should demonstrate at least:

- a credible intrinsic KAN interpretability result;
- calibrated uncertainty under IID and shift;
- a reliability score associated with actual forecast error;
- selective forecasting that improves the risk–coverage trade-off;
- evidence across multiple datasets/horizons.
