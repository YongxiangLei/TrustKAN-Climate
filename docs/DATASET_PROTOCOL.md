# Dataset Selection and Provenance Protocol

## Principle
A top-journal claim should not rely on CET alone. Additional datasets must test distinct failure/generalization modes rather than merely increase dataset count.

## Current benchmark
### CET / Pershore series
Role: long-history temperature forecasting and continuity with earlier KAN climate work.
Status: enabled for development.
Publication action: verify the exact authoritative UK Met Office upstream source, coverage, station definition, missing-data semantics and citation before the final experiment freeze.

## Verified expansion datasets
### ERA5 hourly data on single levels
Scientific role: multivariate reanalysis, geographic generalization, temporal shift, and uncertainty/reliability experiments.
Authoritative source: Copernicus Climate Data Store / ECMWF.
Landing page: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
Coverage: 1940 to present according to the CDS dataset page.
License: CC-BY.
Dataset DOI: 10.24381/cds.adbb2d47.
Protocol still to freeze: exact region(s), variables, aggregation, years, history lengths and forecast horizons.

### NOAA/NCEI Global Historical Climatology Network - Daily (GHCN-Daily)
Scientific role: independent station-based temperature/weather generalization outside the UK benchmark.
Authoritative source: NOAA National Centers for Environmental Information.
Landing page: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
Characteristics: daily station climate summaries with maximum/minimum temperature, precipitation and additional elements; NOAA reports more than 100,000 stations across 180 countries and territories on the current product page.
Frozen protocol: use five fixed geographic strata and derive daily midrange
temperature as `(TMAX + TMIN) / 2` after NOAA quality-flag filtering and an
inner date join. Candidates must advertise paired inventory coverage over
1950--2024. Within each stratum they are ranked by earliest paired start,
latest paired end, and station ID. Candidates are audited in that order, and
the first station passing 70 years of paired coverage, 95% fixed-window
completeness, and a maximum 31-day observation interval is frozen. Model
performance is never consulted.

The frozen v1 stations are `USW00014734` (North America), `UK000056225`
(Europe), `RSM00031915` (East Asia), `ASN00014015` (tropics), and
`ASN00094029` (southern midlatitudes). Their coordinates, candidate ranks,
quality statistics, and raw archive SHA-256 values are recorded in
`configs/datasets/ghcn_frozen.yaml`.

The initial five-candidate audit found no eligible tropical station. Before
running any model comparison, the same deterministic ranking was extended to a
maximum of 50 candidates; the first eligible tropical station occurred at rank
9. This eligibility-only extension is disclosed so the selection history is
auditable.

The repository's engineering example (`USW00094728`, `TAVG`, requested period
1950--2024) was audited on 2026-08-16. The retained element spans only
1998-04-01 through 2005-07-31 (2,643 observations) and therefore fails the
pre-registered 30-year minimum despite 98.66% within-span completeness. It must
not be used for headline results. This negative eligibility result is retained
to demonstrate that dataset gates are enforced rather than bypassed.

`scripts/audit_ghcn.py` preserves the official headerless gzip bytes, records a
SHA-256 and access timestamp, parses the documented eight-column station format,
rejects `-9999` sentinels and non-empty NOAA quality flags, quantifies calendar
gaps, and can fail closed with `--require-eligible`.

The multi-region selection can be reproduced with:

```bash
python scripts/select_ghcn_candidates.py
python scripts/audit_ghcn_candidates.py --require-complete
python scripts/prepare_ghcn_panel.py
```

The inventory snapshot used for v1 has SHA-256
`adecaf2efa1ca024835b293607ecc69139931e74d3b80364f8e5f01afdd32e44`.
Audit JSON files and raw station archives are generated locally and ignored by
Git; the compact frozen configuration is tracked. Prepared artifacts retain
dates, TMAX, TMIN, the derived target, raw/config/code hashes, and are verified
against the tracked continuity evidence before they are written.

### MPI-BGC Jena Beutenberg weather station
Scientific role: high-frequency multivariate forecasting under a very different sampling regime.
Authoritative source: Max Planck Institute for Biogeochemistry, Jena.
Station page: https://www.bgc-jena.mpg.de/en/servicegroups/fieldexperiements/locations/beutenberg
Data download: https://weather.bgc-jena.mpg.de/weather_data.html
Coverage: station operation from 2003 to present according to the institute station page.
Variables include air temperature/humidity, pressure, wind, radiation and precipitation.
License: CC-BY-4.0 according to the official weather-data download page.
Protocol still to freeze: raw variable schema, aggregation interval, quality control, and continuous period used for experiments.

## Required metadata before final experiment freeze
1. authoritative source and permanent landing page/DOI where available;
2. license/terms and required citation;
3. exact variables and units;
4. spatial/station selection rule;
5. temporal resolution and coverage dates;
6. missing-value and quality-control policy;
7. chronological train/validation/calibration/test boundaries;
8. forecast history and horizons;
9. deterministic preparation script/checksum;
10. scientific reason for inclusion.
11. machine-readable eligibility criteria and pass/fail audit.

## Anti-cherry-picking rule
Dataset inclusion/exclusion criteria must be fixed before headline model comparison. A dataset may not be removed because TrustKAN performs poorly; failures must be reported and analysed.

## Planned benchmark roles
- CET: very long univariate temperature history and continuity with prior KAN work.
- GHCN-Daily: independent observational station generalization.
- Jena: high-frequency multivariate station forecasting.
- ERA5: multivariate reanalysis, wider geographic/temporal shift and stress testing.

This combination is intended to test different scientific regimes rather than provide four near-duplicate datasets.
