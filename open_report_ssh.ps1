$ErrorActionPreference = 'Continue'
$sshExecutable = 'C:\Program Files\Git\usr\bin\ssh.exe'
$controlPath = 'C:/Users/qizhi/AppData/Local/Temp/cm-moss-report-914d'
$connectionFile = Join-Path $PSScriptRoot 'ssh_connection.json'
if (Test-Path -LiteralPath $connectionFile) {
  $controlPath = (Get-Content -LiteralPath $connectionFile -Raw | ConvertFrom-Json).control_path
}
$Host.UI.RawUI.WindowTitle = 'DKUCC report - SSH proxy session'
& $sshExecutable -S $controlPath -O check qt28@dkucc-login-01.rc.duke.edu 2>$null
if ($LASTEXITCODE -eq 0) {
  Write-Host 'Reusing the authenticated connection (proxy mode).'
  & $sshExecutable -tt -O proxy -S $controlPath -o BatchMode=yes qt28@dkucc-login-01.rc.duke.edu "date -Is; squeue -u qt28; exec bash -l"
  Read-Host 'Session ended. Press Enter to close'
  exit
}
$controlPath = 'C:/Users/qizhi/AppData/Local/Temp/cm-moss-report-' + [guid]::NewGuid().ToString('N').Substring(0,8)
Write-Host "New SSH control path: $controlPath"
@{control_path=$controlPath} | ConvertTo-Json | Set-Content -LiteralPath $connectionFile -Encoding UTF8

Write-Host 'DKUCC report session'
Write-Host 'Please enter your DKUCC password and complete Duo when prompted.'
Write-Host 'After Duo succeeds this window will remain at a blank cursor. Keep it open.'

& $sshExecutable -N -o ControlMaster=yes -S $controlPath `
  -o ControlPersist=no `
  -o PubkeyAuthentication=no `
  -o IdentitiesOnly=yes `
  -o PreferredAuthentications=keyboard-interactive,password `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  qt28@dkucc-login-01.rc.duke.edu

Read-Host 'SSH session ended. Press Enter to close'
