# Tem2-KAN Reproduction Note

## Legacy evidence used
The previous `kan_proj/Temp_IKAN4.py` implementation imports `KAN` from the external `kan` package, constructs chronological CET windows, and evaluates architectures of the form `KAN(width=[300, 32, 64, 32, 20], k=k, grid=grid)`. Its final example loads `k=10, grid=10`.

## Important reproducibility caveat
The legacy script computes global min/max normalization before the train/test split. That leaks information from the future test period into preprocessing. TrustKAN-Climate must **not** reproduce this leakage. We preserve the model architecture as the historical comparator while using the new repository's training-only scaler and four-way chronological protocol.

## Comparator policy
- `standard KAN`: current clean baseline implemented in this repository.
- `Tem2KANReference`: adapter matching the legacy pykan-style width/k/grid formulation as closely as practical.
- `TrustKAN`: proposed new reliability-aware temporal model.

The manuscript must clearly distinguish architectural reproduction from experimental-protocol reproduction. Any performance difference from the historical paper/code may partly reflect the stricter leakage-safe protocol, changed horizons, seeds, or dependency versions.
