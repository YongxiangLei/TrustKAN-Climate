# Manuscript / Overleaf Workflow

This directory is the manuscript source for TrustKAN-Climate.

## Main file
Compile `main.tex`.

## Overleaf project
The working Overleaf project supplied by the author is:
`https://www.overleaf.com/project/6a8168aadaa6410e9715335e`

The current environment cannot write directly to Overleaf, so GitHub is treated as the controlled manuscript source. Upload/sync the contents of `paper/` to the Overleaf project.

## Evidence rule
Do not manually type experimental numbers into the final manuscript. Tables and figures should be copied/generated from `results/` through the project scripts.

## Section status
- Introduction: first evidence-safe draft
- Related work: structure only; verified citations still required
- Methodology: first technical draft matching the implemented pipeline
- Experimental setup: protocol draft
- Results: artifact-wired placeholders; no invented findings
- Discussion: evidence-safe draft
- Conclusion: placeholder pending final experiments

## Before submission
1. Freeze datasets and splits.
2. Run all headline experiments and ablations.
3. Generate paper tables/figures programmatically.
4. Replace all `TODO` markers only with verified evidence.
5. Verify every citation against the original publication.
6. Run reviewer and reproducibility audits.
7. Adapt `IEEEtran` to the final target journal template if needed.
