#Requires -RunAsAdministrator
<#
Capture warmed batch-8 and batch-32 CPU traces for the existing RAG workload.
The Python harness refuses an existing WPR session and stops its own recording
in finally if generation fails. No security policies are changed.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$taskRepoRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRepoRoot 'docs/experiments/windows_cpu_sampling/etw'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
Set-Location -LiteralPath $taskRepoRoot
$env:PYTHONPATH = $taskRepoRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_CACHE = Join-Path $taskRepoRoot 'models/datasets-cache'
$env:TIKTOKEN_CACHE_DIR = Join-Path $taskRepoRoot 'models/tiktoken-cache'
$taskResult = @{ status = 'running'; controller_pid = $PID; started_utc = [DateTime]::UtcNow.ToString('o') }
$taskStatusPath = Join-Path $taskOutput 'controller_status.json'
$taskResult | ConvertTo-Json | Set-Content -LiteralPath $taskStatusPath -Encoding UTF8
try {
    $ErrorActionPreference = 'Continue'
    & (Join-Path $taskRepoRoot '.venv/Scripts/python.exe') `
        scripts/profile_generation.py `
        --source-json docs/experiments/nq500_batch8/adaptive_topk_adaptive_gap_20261113_160953_930358.json `
        --sample-num 500 --repeats 1 --warmup-sample-num 8 --cpu-threads 8 `
        --wpr-study --output-dir $taskOutput *> (Join-Path $taskOutput 'run.log')
    $taskExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $taskResult.exit_code = $taskExit
    $taskResult.status = if ($taskExit -eq 0) { 'complete' } else { 'failed' }
} catch {
    $taskResult.status = 'failed'
    $taskResult.error = $_.Exception.Message
} finally {
    $taskResult.finished_utc = [DateTime]::UtcNow.ToString('o')
    $taskResult | ConvertTo-Json | Set-Content -LiteralPath $taskStatusPath -Encoding UTF8
}
if ($taskResult.status -ne 'complete') { exit 1 }
