# Finish the three v2 campaigns that still have to run, in order, on one GPU.
# Each inner loop is the same record-count resume used elsewhere: Windows
# cuDNN teardown aborts after a record is written, so exit codes are ignored.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$resume = Join-Path $PSScriptRoot "resume_until_complete.ps1"

Write-Output "=== robustness v2 (60) ==="
& $resume -Config configs/cet_v2_neural.yaml `
    -RecordDir results/robustness/runs/cet_robustness_v2 `
    -Expected 60 `
    -Runner scripts/run_robustness_campaign.py `
    -ExtraArgs "--study v2"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "=== KAN curves v2 (20) ==="
& $resume -Config configs/cet_v2_neural.yaml `
    -RecordDir results/interpretability/runs/cet_kan_curves_v2 `
    -Expected 20 `
    -Runner scripts/run_kan_curves.py `
    -ExtraArgs "--model trustkan_v2 --ledger results/aggregated/cet_v2_neural_runs.csv --run-name cet_kan_curves_v2"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "=== receptive field v2 (20) ==="
& $resume -Config configs/cet_v2_neural.yaml `
    -RecordDir results/robustness/runs/cet_receptive_field_v2 `
    -Expected 20 `
    -Runner scripts/run_receptive_field.py `
    -ExtraArgs "--study v2"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "ALL remaining v2 campaigns complete"
