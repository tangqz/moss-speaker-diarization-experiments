$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$pythonPath = Join-Path $taskDirectory '.venv\Scripts\python.exe'
$server = Start-Process -FilePath $pythonPath -ArgumentList @(
    '-m', 'tensorboard.main',
    '--logdir=events',
    '--host=127.0.0.1',
    '--port=6006',
    '--reload_interval=5',
    '--reload_multifile=true'
) -WorkingDirectory $taskDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDirectory 'tensorboard.out.log') -RedirectStandardError (Join-Path $taskDirectory 'tensorboard.err.log') -PassThru
Set-Content -LiteralPath (Join-Path $taskDirectory 'tensorboard.pid') -Value $server.Id -Encoding ascii
Write-Output "TensorBoard PID=$($server.Id)"
