"""Dataset metadata registry.

Only CET is enabled initially. Candidate datasets remain disabled until their
provenance, license, preparation procedure and scientific role are verified.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    temporal_resolution: str
    task_role: str
    enabled: bool
    provenance: str
    notes: str = ""

DATASETS = {
    "cet": DatasetSpec(
        key="cet", name="Central England Temperature / Pershore series",
        temporal_resolution="daily", task_role="long historical univariate temperature benchmark",
        enabled=True, provenance="Repository-configured CET source; verify against authoritative upstream before publication.",
        notes="Current development benchmark."
    ),
    "era5": DatasetSpec(
        key="era5", name="ERA5 candidate", temporal_resolution="hourly/daily derived",
        task_role="multivariate reanalysis and geographic generalization", enabled=False,
        provenance="Candidate only; authoritative download/API and license documentation required before enabling."
    ),
    "noaa": DatasetSpec(
        key="noaa", name="NOAA/NCEI candidate", temporal_resolution="daily/hourly depending product",
        task_role="independent station/network generalization", enabled=False,
        provenance="Candidate only; exact NOAA/NCEI product must be selected and documented."
    ),
    "jena": DatasetSpec(
        key="jena", name="Jena climate candidate", temporal_resolution="10-minute",
        task_role="high-frequency multivariate weather forecasting", enabled=False,
        provenance="Candidate only; verify original Max Planck Institute source and redistribution terms."
    ),
}

def get_dataset_spec(key: str) -> DatasetSpec:
    if key not in DATASETS: raise KeyError(f"Unknown dataset {key!r}")
    return DATASETS[key]
