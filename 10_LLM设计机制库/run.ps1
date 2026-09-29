<#  Launch one named (physics-disclosed) physics-guided PDESR run.

    pwsh run.ps1 -MaxSamples 100 -Grow

Everything in folder 09 is flat: the code, the spec and the data all sit next to
this script. The API key is read from the repository `.env`.
#>
param(
    [string]$Problem = "Traffic_Flow_Bottleneck",
    [int]$MaxSamples = 100,
    [string]$Model = "deepseek-v4-flash",
    [string]$LogDir = "",
    [switch]$Grow,
    [int]$GrowRounds = 1,
    [int]$GrowProposals = 3,
    [double]$PromoteThreshold = 1.5,
    [int]$MaxOrder = 2,
    [switch]$Design,
    [int]$DesignMechanisms = 10,
    [switch]$Foreground
)
$ErrorActionPreference = 'Stop'

$work = $PSScriptRoot
$root = Split-Path -Parent $work
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
if (-not $LogDir) { $LogDir = Join-Path $work 'results' }
$outLog = Join-Path $work "$($Problem.ToLower())_driver.log"
$errLog = Join-Path $work "$($Problem.ToLower())_driver.err.log"

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

$runArgs = @(
    'main.py',
    '--problem_name', $Problem,
    '--spec_path', (Join-Path $work 'spec_traffic_flow_bottleneck.txt'),
    '--data_path', (Join-Path $work 'traffic_flow_bottleneck.npz'),
    '--log_path', $LogDir,
    '--use_api', 'True',
    '--api_model', $Model,
    '--api_base_url', 'https://api.teamorouter.cn/v1',
    '--api_key_env', 'TEAMOROUTER_API_KEY',
    '--thinking', 'disabled',
    '--max_samples', "$MaxSamples",
    '--max_order', "$MaxOrder"
)
if ($Grow) {
    $runArgs += @('--grow_mechanisms', '1', '--grow_rounds', "$GrowRounds",
                  '--grow_proposals', "$GrowProposals",
                  '--promote_threshold', "$PromoteThreshold")
}
if ($Design) {
    $runArgs += @('--design_library', '1', '--design_mechanisms', "$DesignMechanisms")
}

Write-Output "problem : $Problem   model: $Model   samples: $MaxSamples   grow: $Grow   design: $Design"
Write-Output "log dir : $LogDir"

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
