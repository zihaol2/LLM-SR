<#  Launch one semi-anonymous physics-guided PDESR run.

    The spec keeps the structural prior ("a spatial bottleneck", "anticipation
    driven by the field's relative gradient") and withholds only the subject-area
    nouns. The prior is global: the library design and the search see exactly the
    same text. After the search the field is named from the statistics-based prior
    reading plus the best equation -- with its fitted coefficients -- and the
    classical parent PDEs it matches.

    pwsh run.ps1 -MaxSamples 100
    pwsh run.ps1 -DesignOnly 1              # design the library, stop, print it
    pwsh run.ps1 -DesignFrom ..\results\designed_library.json   # reuse a library
#>
param(
    [string]$Problem = "Traffic_Flow_Bottleneck",
    [string]$Spec = 'spec_traffic_flow_bottleneck.txt',
    [string]$Data = '',
    [int]$MaxSamples = 100,
    [string]$Model = "deepseek-v4-flash",
    [string]$LogDir = "",
    [string]$Design = 'true',
    [int]$DesignMechanisms = 10,
    [int]$StatsOps = 14,
    [string]$DesignStats = 'false',
    [string]$DesignFrom = '',
    [string]$DesignOnly = 'false',
    [string]$RoleIdentify = 'true',
    [int]$NParents = 6,
    [int]$MaxOrder = 2,
    [switch]$Foreground
)
$ErrorActionPreference = 'Stop'

$work   = $PSScriptRoot                       # .../带少量先验的LLMPDESR/code
$folder = Split-Path -Parent $work            # .../带少量先验的LLMPDESR
$root   = Split-Path -Parent $folder          # .../LLM-PDESR
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
if (-not $LogDir) { $LogDir = Join-Path $folder 'results' }
$specPath = Join-Path $work $Spec
if (-not $Data) { $Data = "data\traffic_flow_bottleneck.npz" }
$dataPath = Join-Path $folder $Data
$outLog = Join-Path $folder "$($Problem.ToLower())_driver.log"
$errLog = Join-Path $folder "$($Problem.ToLower())_driver.err.log"

$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
        }
    }
}
if (-not $env:TEAMOROUTER_API_KEY) { throw "TEAMOROUTER_API_KEY is not set (looked in $envFile)" }
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# `pwsh -File` hands string arguments through as strings, so the switches are parsed
# here instead of being declared [bool] (which rejects the literal "0").
function Get-Flag([string]$value) {
    if ($value -match '^(1|true|yes|y|on)$') { return '1' }
    return '0'
}
$designFlag = Get-Flag $Design
$designStatsFlag = Get-Flag $DesignStats
$designOnlyFlag = Get-Flag $DesignOnly
$roleFlag = Get-Flag $RoleIdentify

$runArgs = @(
    'main.py',
    '--problem_name', $Problem,
    '--spec_path', $specPath,
    '--data_path', $dataPath,
    '--log_path', $LogDir,
    '--use_api', 'True',
    '--api_model', $Model,
    '--api_base_url', 'https://api.teamorouter.cn/v1',
    '--api_key_env', 'TEAMOROUTER_API_KEY',
    '--thinking', 'disabled',
    '--max_samples', "$MaxSamples",
    '--max_order', "$MaxOrder",
    '--design_stats', $designStatsFlag,
    '--role_identify', $roleFlag,
    '--n_parents', "$NParents"
)
if ($designOnlyFlag -eq '1') {
    $runArgs += @('--design_only', '1')
}
if ($DesignFrom) {
    $runArgs += @('--design_from', $DesignFrom)
}
if ($designFlag -eq '1') {
    $runArgs += @('--design_library', '1', '--design_mechanisms', "$DesignMechanisms",
                  '--stats_ops', "$StatsOps")
}

Write-Output "problem : $Problem   model: $Model   samples: $MaxSamples   design: $designFlag   designStats: $designStatsFlag   role: $roleFlag   designOnly: $designOnlyFlag"
Write-Output "log dir : $LogDir"
Write-Output "spec    : $specPath"
Write-Output "data    : $dataPath"

if ($Foreground) {
    & $py @runArgs 2>&1 | Tee-Object -FilePath $outLog
    exit $LASTEXITCODE
}
$argString = ($runArgs | ForEach-Object {
    if ($_ -match '\s' -or $_ -eq '') { '"' + $_ + '"' } else { $_ }
}) -join ' '
$proc = Start-Process -FilePath $py -ArgumentList $argString `
    -WorkingDirectory $work -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
Write-Output "started pid : $($proc.Id)"
