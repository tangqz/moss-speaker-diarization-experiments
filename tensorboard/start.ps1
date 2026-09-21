param([int]$Port = 6006)
$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$pythonPath = Join-Path $taskDirectory '.venv\Scripts\python.exe'
$statePath = Join-Path $taskDirectory 'processes.json'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'TensorBoard environment is missing.' }
if (Test-Path -LiteralPath $statePath) {
    $prior = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $live = @(Get-Process -Id $prior.tensorboard_pid,$prior.bridge_pid -ErrorAction SilentlyContinue)
    if ($live.Count -eq 2) { Write-Output "TensorBoard is already running: http://127.0.0.1:$($prior.port)"; exit 0 }
    if ($live.Count -gt 0) { throw 'One monitor process remains; inspect processes.json before restarting.' }
}
$server = Start-Process -FilePath $pythonPath -ArgumentList @('-m','tensorboard.main','--logdir=events','--host=127.0.0.1',"--port=$Port",'--reload_interval=5','--reload_multifile=true') -WorkingDirectory $taskDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDirectory 'tensorboard.out.log') -RedirectStandardError (Join-Path $taskDirectory 'tensorboard.err.log') -PassThru
$bridge = Start-Process -FilePath $pythonPath -ArgumentList @('bridge.py') -WorkingDirectory $taskDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDirectory 'bridge.out.log') -RedirectStandardError (Join-Path $taskDirectory 'bridge.err.log') -PassThru
@{tensorboard_pid=$server.Id;bridge_pid=$bridge.Id;port=$Port;started=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Output "TensorBoard: http://127.0.0.1:$Port"
