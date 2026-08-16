# Dataset Selection and Provenance Protocol

## Principle
A top-journal claim should not rely on CET alone. Additional datasets must test distinct failure/generalization modes rather than merely increase dataset count.

## Current benchmark
### CET / Pershore series
Role: long-history temperature forecasting and continuity with earlier KAN climate work.
Status: enabled for development.
Publication action: verify the exact authoritative upstream source, coverage, station definition, missing-data semantics and citation before final experiments.

## Candidate expansion
### ERA5
Scientific role: multivariate reanalysis, multiple atmospheric variables, spatial/geographic generalization and controlled temporal shift.
Status: disabled pending exact variable/region/resolution definition and authoritative provenance documentation.

### NOAA/NCEI
Scientific role: independent observational station/network benchmark outside the UK series.
Status: disabled pending selection of the exact NCEI product and station/network protocol.

### Jena climate/weather
Scientific role: high-frequency multivariate forecasting to test whether conclusions survive a radically different sampling interval.
Status: disabled pending verification of original source and redistribution/citation terms.

## Required metadata before enabling any dataset
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

## Anti-cherry-picking rule
Dataset inclusion/exclusion criteria must be fixed before headline model comparison. A dataset may not be removed because TrustKAN performs poorly; failures must be reported and analysed.
