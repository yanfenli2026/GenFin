# Regenerate every output of the paper in dependency order (Windows PowerShell 5.1 or PowerShell 7).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Python C:\envs\genfin\Scripts\python.exe -SkipSec
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -DownloadYahoo
#
# Paths can be redirected with GENFIN_OUT, GENFIN_FIG_DIR, GENFIN_YAHOO_DIR, GENFIN_SEC_FILE (code\config.py).
param(
    [string]$Python = "python",
    [switch]$SkipSec,
    [switch]$DownloadYahoo
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:MPLBACKEND = "Agg"
$env:PYTHONIOENCODING = "utf-8"
$LogDir = if ($env:GENFIN_LOG_DIR) { $env:GENFIN_LOG_DIR } else { "logs" }
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Invoke-Step([string]$Name, [string[]]$ScriptArgs) {
    Write-Host ("=== [{0:HH:mm:ss}] {1}: {2}" -f (Get-Date), $Name, ($ScriptArgs -join " "))
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $ErrorActionPreference = "Continue"   # Python warnings on stderr must not abort the run
    & $Python @ScriptArgs *>&1 | Tee-Object -FilePath (Join-Path $LogDir "$Name.log")
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    Write-Host ("=== {0} finished in {1:N0} s" -f $Name, $sw.Elapsed.TotalSeconds)
}

function Test-Data([string]$Fn) {
    & $Python -c "import sys; sys.path.insert(0,'code'); import config; sys.exit(0 if config.$Fn() else 1)"
    return ($LASTEXITCODE -eq 0)
}

if ($DownloadYahoo) { Invoke-Step "yahoo_download" @("data/download_recent_data.py") }
if (-not (Test-Data "have_yahoo_data")) {
    throw "Yahoo Finance statement CSVs not found (see README, 'Data'). Use -DownloadYahoo or set GENFIN_YAHOO_DIR."
}

$env:GENFIN_REQUIRE_REAL = "1"
Invoke-Step "foundation"        @("code/test_common.py")
Invoke-Step "e1_benchmark"      @("code/e1_benchmark.py")
Invoke-Step "e2_deep_tabular"   @("code/e2_deep_tabular.py")
Invoke-Step "e3_shap"           @("code/e3_shap.py")
Invoke-Step "e4_generative"     @("code/e4_generative.py")
Invoke-Step "e5_optimization"   @("code/e5_optimization.py")
Invoke-Step "e6_econometrics"   @("code/e6_econometrics.py")
Invoke-Step "e7_real_data"      @("code/e7_real_data.py")

if ($SkipSec) {
    Write-Host "=== E8 skipped (-SkipSec); build_master.py will fail on the missing e8_sec tables"
} elseif (Test-Data "have_sec_data") {
    Invoke-Step "e8_sec_validation" @("code_sec/e8_sec_validation.py")
} else {
    throw "Datavidence SEC sample not found (see README, 'Data'). Set GENFIN_SEC_FILE or use -SkipSec."
}

Invoke-Step "build_master" @("code/build_master.py")
Invoke-Step "final_figures_main" @("code/final_figures_main.py")
Invoke-Step "final_figures_si_a" @("code/final_figures_si_a.py")
Invoke-Step "final_figures_si_b" @("code/final_figures_si_b.py")
Write-Host "=== all done. Outputs: output\ (or GENFIN_OUT); figures: figures\ (or GENFIN_FIG_DIR)"
