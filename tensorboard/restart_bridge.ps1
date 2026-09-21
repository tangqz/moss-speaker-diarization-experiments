$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$statePath = Join-Path $taskDirectory 'processes.json'
$prior = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$pythonPath = Join-Path $taskDirectory '.venv\Scripts\python.exe'
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $($prior.bridge_pid)"
if ($process) {
    if ($process.ExecutablePath -ne $pythonPath -or $process.CommandLine -notmatch 'bridge\.py') {
        throw 'The saved PID belongs to a different process. Refusing to stop it.'
    }
    Stop-Process -Id $process.ProcessId
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$bridge = Start-Process -FilePath $pythonPath -ArgumentList @('bridge.py') -WorkingDirectory $taskDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDirectory "bridge-$stamp.out.log") -RedirectStandardError (Join-Path $taskDirectory "bridge-$stamp.err.log") -PassThru
$prior.bridge_pid = $bridge.Id
$prior | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
Write-Output "Log sync restarted (PID $($bridge.Id)); TensorBoard remains at http://127.0.0.1:$($prior.port)."
