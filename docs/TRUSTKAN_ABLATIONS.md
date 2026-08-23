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

## Executable path

`scripts/run_ablations.py` with `configs/ablations.yaml` produces every row
above from one frozen CET protocol. Only A0--A2 and A9 require training;
A3--A7 are evaluation-side removals recomputed from the A0 artifact, and A8
replays the A0 predictions through sequential adaptive conformal. The full
matrix is therefore 4 trained variants x 4 horizons x 5 seeds = 80 GPU runs.

```bash
python scripts/run_ablations.py --config configs/ablations.yaml --resume --device cuda:0
```

A1 replaces the KAN mapping with a two-layer MLP whose width is chosen so the
parameter budget matches the KAN layer to within one percent. Without that
constraint the comparison would measure capacity rather than the KAN mapping
itself.

A2 removes the quantile head and emits degenerate quantiles equal to the point
forecast, which makes split conformal reduce exactly to symmetric
absolute-residual intervals. Its pinball weight is set to zero so no quantile
gradient reaches the point head through the degenerate outputs.

A consequence worth stating in advance: under A2 the interval width carries no
sample-specific information, so the width-only reliability score is constant
and its rank correlation with error is undefined. That is the expected
signature of the removal, not a defect, and it is what A5 tests.

A9 restores the mean-pooled temporal readout that the model originally used.
That readout averages encoder states across the whole history window, which on
a 365-step daily history sits near the local climatological mean and discards
recency. In the first CET campaign it left the model behind a persistence
baseline at every horizon (for example 4.95 versus 2.14 RMSE at one day under a
matched budget and seed), so the default readout is now the final encoder
state, matching every recurrent and attention baseline here. A9 keeps the
superseded behaviour measurable instead of only described.

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
