$ErrorActionPreference = 'Continue'
$sshExecutable = 'C:\Program Files\Git\usr\bin\ssh.exe'
$controlPath = 'C:/Users/qizhi/AppData/Local/Temp/cm-moss-rswa-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$connectionFile = Join-Path $PSScriptRoot 'ssh_connection.json'
$logDirectory = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ('rswa-visible-ssh-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
@{control_path=$controlPath;purpose='MOSS RSWA 20260920';local_log=$logPath} |
    ConvertTo-Json | Set-Content -LiteralPath $connectionFile -Encoding UTF8
$Host.UI.RawUI.WindowTitle = 'DKUCC - MOSS R-SWA - login and live progress'
Write-Host 'Please enter your DKUCC password and complete Duo in this window.'
Write-Host 'After login, keep this window open. Live experiment output will appear here.'
Write-Host "Local output log: $logPath"
$remoteCommand = @'
date -Is
hostname
id -un
squeue -u qt28
mkdir -p /work/qt28/moss/logs
touch /work/qt28/moss/logs/rswa-20260920-console.log
echo RSWA_SSH_READY
tail -n 15 -F /work/qt28/moss/logs/rswa-20260920-console.log &
exec bash -l
'@
$remoteCommand = $remoteCommand.Replace("`r`n", "`n")
& $sshExecutable -tt -M -S $controlPath `
    -o ControlPersist=no -o PubkeyAuthentication=no -o IdentitiesOnly=yes `
    -o PreferredAuthentications=keyboard-interactive,password `
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    qt28@dkucc-login-01.rc.duke.edu $remoteCommand |
    Tee-Object -FilePath $logPath
Write-Host "SSH session ended. Output log: $logPath"
Read-Host 'Press Enter to close'
