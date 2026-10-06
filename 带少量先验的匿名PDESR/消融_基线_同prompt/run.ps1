<#  ABLATION: the baseline search, given the SAME semi-anonymous prior prompt.

    The only thing this arm does not get is the designed mechanism library. The
    spec is byte-identical to ../code/spec_traffic_flow_bottleneck.txt (same
    structural prior in the docstring, same free-numpy search rules), so the prompt
    the search sees is the same text minus the "LIBRARY GUIDANCE" block.

    No library is designed and no variable role is revealed to the search.

    pwsh run.ps1 -MaxSamples 50
#>
param(
    [string]$Problem = "Traffic_Flow_Bottleneck",
    [int]$MaxSamples = 100,
    [string]$Model = "deepseek-v4-flash",
    [string]$LogDir = "",
    [int]$MaxOrder = 2,
    [string]$RoleIdentify = 'false',
    [switch]$Foreground
)
$ErrorActionPreference = 'Stop'

$here   = $PSScriptRoot                       # .../带少量先验的LLMPDESR/消融_基线_同prompt
$folder = Split-Path -Parent $here            # .../带少量先验的LLMPDESR
$root   = Split-Path -Parent $folder          # .../LLM-PDESR
$work   = Join-Path $folder 'code'
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
if (-not $LogDir) { $LogDir = Join-Path $here 'results' }
$specPath = Join-Path $here 'spec_baseline_prior.txt'
$dataPath = Join-Path $folder 'data\traffic_flow_bottleneck.npz'
$outLog = Join-Path $here "$($Problem.ToLower())_driver.log"
$errLog = Join-Path $here "$($Problem.ToLower())_driver.err.log"

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

function Get-Flag([string]$value) {
    if ($value -match '^(1|true|yes|y|on)$') { return '1' }
    return '0'
}
$roleFlag = Get-Flag $RoleIdentify

# No --design_library and no --design_from: the DB of mechanisms stays empty and
# the search prompt carries no library guidance.
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
    '--role_identify', $roleFlag
)

Write-Output "ABLATION (no library, same prior prompt)"
Write-Output "problem : $Problem   model: $Model   samples: $MaxSamples   role: $roleFlag"
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
