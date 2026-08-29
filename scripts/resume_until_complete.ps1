# Re-invoke a runner's --resume until it stops making progress.
#
# PyTorch's cuDNN RNN teardown aborts the process on Windows after a run has
# already written its record, so a campaign judged by exit code stops early with
# its work intact. Progress is therefore measured by records on disk, and the
# loop only gives up when an attempt adds nothing.
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$RecordDir,
    [Parameter(Mandatory = $true)][int]$Expected,
    [string]$Runner = "scripts/run_cet_benchmark.py",
    [int]$MaxAttempts = 40,
    # Runner-specific switches as one string, e.g. -ExtraArgs "--study v2".
    # A string[] Extra whose items start with dashes is rebound by powershell
    # -File as parameters of this script, which is how a previous invocation
    # sent "v2" to -MaxAttempts and never started the campaign.
    [string]$ExtraArgs = ""
)

$python = "C:\Users\79441\Documents\Codex\.venvs\trustkan\Scripts\python.exe"
# Python writes warnings to stderr, and on Windows a cuDNN teardown often
# exits non-zero after the record is already on disk. Neither is a reason to
# abort this loop; a parent $ErrorActionPreference=Stop would otherwise treat
# the transformer UserWarning as a terminating error.
$ErrorActionPreference = "Continue"

function Get-RecordCount {
    if (-not (Test-Path $RecordDir)) { return 0 }
    return (Get-ChildItem $RecordDir -Filter *.json -ErrorAction SilentlyContinue).Count
}

$attempt = 0
while ($attempt -lt $MaxAttempts) {
    $attempt++
    $before = Get-RecordCount
    if ($before -ge $Expected) {
        Write-Output "COMPLETE: $before/$Expected records present"
        break
    }
    Write-Output "attempt $attempt : $before/$Expected records, resuming"
    New-Item -ItemType Directory -Force -Path $RecordDir | Out-Null
    $log = Join-Path $RecordDir "_resume.log"
    Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue
    $argList = @($Runner, "--config", $Config, "--device", "cuda", "--resume")
    if ($ExtraArgs) {
        $argList += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    }
    # Launch as a child we can kill. A cuDNN teardown sometimes hangs instead of
    # exiting, which left the previous loop waiting overnight with the GPU idle.
    $proc = Start-Process -FilePath $python -ArgumentList $argList -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $lastCount = $before
    $lastProgress = Get-Date
    $idleMinutes = 40
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 30
        $now = Get-RecordCount
        if ($now -gt $lastCount) {
            $lastCount = $now
            $lastProgress = Get-Date
            Write-Output ("  progress $now/$Expected")
            if ($now -ge $Expected) { break }
        }
        elseif (((Get-Date) - $lastProgress).TotalMinutes -ge $idleMinutes) {
            Write-Output "WATCHDOG: no new record for $idleMinutes min; killing hung runner pid=$($proc.Id)"
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            break
        }
    }
    $after = Get-RecordCount
    Write-Output "attempt $attempt : $before -> $after"
    if ($after -le $before) {
        Write-Output "STALLED: attempt added no records; stopping so the failure is visible"
        break
    }
}

$final = Get-RecordCount
Write-Output "FINAL $final/$Expected"
if ($final -lt $Expected) { exit 1 }
