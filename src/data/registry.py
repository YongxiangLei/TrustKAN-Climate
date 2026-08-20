"""Dataset metadata registry with publication-oriented provenance."""
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
    license: str
    citation_hint: str
    notes: str = ""

DATASETS = {
    "cet": DatasetSpec(
        key="cet",
        name="Central England Temperature / Pershore series",
        temporal_resolution="daily",
        task_role="long historical univariate temperature benchmark",
        enabled=True,
        provenance="Current development copy originates from the prior project; verify the exact UK Met Office upstream landing page and station definition before final publication freeze.",
        license="To verify before publication freeze",
        citation_hint="Use the authoritative UK Met Office dataset citation once provenance is frozen.",
        notes="Development benchmark and continuity with prior KAN climate work."
    ),
    "era5": DatasetSpec(
        key="era5",
        name="ERA5 hourly data on single levels from 1940 to present",
        temporal_resolution="hourly; daily aggregation permitted by protocol",
        task_role="multivariate reanalysis, geographic generalization, and temporal-shift evaluation",
        enabled=True,
        provenance="Copernicus Climate Data Store / ECMWF: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
        license="CC-BY",
        citation_hint="Dataset DOI: 10.24381/cds.adbb2d47",
        notes="Exact region, variables, aggregation and period must be frozen in config before headline experiments."
    ),
    "ghcnd": DatasetSpec(
        key="ghcnd",
        name="NOAA Global Historical Climatology Network - Daily (GHCN-Daily)",
        temporal_resolution="daily",
        task_role="independent station-based generalization outside the UK benchmark",
        enabled=True,
        provenance="NOAA/NCEI: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily",
        license="NOAA/NCEI public data; check product-specific use/citation guidance in final data card",
        citation_hint="Use NOAA/NCEI GHCN-Daily Version 3 landing-page citation and access date.",
        notes="Station selection must be fixed by objective completeness/record-length criteria, not model performance."
    ),
    "jena": DatasetSpec(
        key="jena",
        name="MPI-BGC Jena Beutenberg weather station",
        temporal_resolution="official 10-minute archives aggregated to hourly means",
        task_role="high-frequency multivariate weather forecasting under a different sampling regime",
        enabled=True,
        provenance="Max Planck Institute for Biogeochemistry weather data: https://weather.bgc-jena.mpg.de/weather_data.html",
        license="Creative Commons CC-BY-4.0",
        citation_hint="Acknowledge MPI for Biogeochemistry Jena Beutenberg weather station and record access date.",
        notes="Frozen v1 uses 2010--2020 hourly T, p, rh and wv. Observed completeness is recorded by the Jena audit, not invented here."
    ),
}

def get_dataset_spec(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset {key!r}")
    return DATASETS[key]
