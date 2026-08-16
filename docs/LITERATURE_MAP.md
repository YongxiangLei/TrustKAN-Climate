# Verified Literature Map

This file records primary references that have been checked before inclusion in `paper/references.bib`. It is a manuscript-development aid, not a substitute for reading the full papers.

## KAN foundations
- Liu et al., **KAN: Kolmogorov–Arnold Networks**, ICLR 2025 Oral. Core architectural reference: learnable univariate edge functions and inspectable functional representations.
- Vaca-Rubio et al., **Kolmogorov-Arnold Networks (KANs) for Time Series Analysis**, arXiv:2405.08790 (2024). Early KAN time-series forecasting application.

## Modern sequence baselines
- Nie et al., **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**, arXiv:2211.14730. PatchTST reference.
- Gu and Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, arXiv:2312.00752. Selective state-space baseline reference.

## Conformal prediction and shift
- Romano, Patterson, and Candès, **Conformalized Quantile Regression**, NeurIPS 2019.
- Tibshirani et al., **Conformal Prediction Under Covariate Shift**, NeurIPS 2019.
- Stankeviciute, Alaa, and van der Schaar, **Conformal Time-series Forecasting**, NeurIPS 2021.
- Gibbs and Candès, **Adaptive Conformal Inference Under Distribution Shift**, NeurIPS 2021.
- Zaffran et al., **Adaptive Conformal Predictions for Time Series**, ICML 2022.
- Angelopoulos, Candès, and Tibshirani, **Conformal PID Control for Time Series Prediction**, NeurIPS 2023.

## Selective prediction and shift monitoring
- Geifman and El-Yaniv, **Selective Classification for Deep Neural Networks**, NeurIPS 2017.
- Geifman and El-Yaniv, **SelectiveNet**, ICML 2019.
- Bar-Shalom, Geifman, and El-Yaniv, **Window-Based Distribution Shift Detection for Deep Neural Networks**, NeurIPS 2023.

## Manuscript gap statement to test
The proposed paper should not claim novelty from KAN, conformal prediction, adaptive conformal inference, or abstention individually. The candidate contribution is their experimentally validated integration around an interpretable temporal KAN representation, with reliability evidence tested against realized forecast error, temporal shift, and risk–coverage behavior.

## Still to verify before final submission
- The user's own prior Tem2-KAN and climate-KAN publications and their exact bibliographic metadata.
- Dataset-specific benchmark papers for the final ERA5, GHCN, Jena and CET protocols.
- Any additional 2025–2026 trustworthy time-series/KAN papers included in the final Related Work section.
