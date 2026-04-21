param(
    [string]$EnvName = "gaze_dash",
    [switch]$SkipConda,
    [switch]$SkipInstall,
    [string[]]$StreamlitArgs
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not $SkipConda) {
    Write-Step "Checking conda availability"
    $null = Get-Command conda -ErrorAction Stop

    Write-Step "Creating/updating conda env from environment.yml ($EnvName)"
    conda env update --name $EnvName --file "$RepoRoot\environment.yml" --prune

    Write-Step "Using conda env ($EnvName) via conda run"
}

if (-not $SkipInstall) {
    Write-Step "Installing editable package (pip install -e .)"
    if (-not $SkipConda) {
        conda run -n $EnvName python -m pip install -e .
    } else {
        python -m pip install -e .
    }
}

Write-Step "Starting dashboard"
if ($StreamlitArgs -and $StreamlitArgs.Count -gt 0) {
    if (-not $SkipConda) {
        conda run -n $EnvName python -m eyegaze run -- @StreamlitArgs
    } else {
        python -m eyegaze run -- @StreamlitArgs
    }
} else {
    if (-not $SkipConda) {
        conda run -n $EnvName python -m eyegaze run
    } else {
        python -m eyegaze run
    }
}
