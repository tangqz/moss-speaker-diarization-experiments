$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $workspaceRoot 'dkucc\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logDirectory "interactive-$timestamp.log"

$remoteCommand = @'
printf '\n===== DKUCC initial inventory =====\n'
date -Is
hostname
id
pwd
printf '\n===== Slurm partitions =====\n'
sinfo -o '%P %a %l %D %G'
printf '\n===== Current jobs =====\n'
squeue -u "$USER"
printf '\n===== Storage =====\n'
df -h /dkucc/home /work
printf '\n===== Available modules (first 80 lines) =====\n'
module avail 2>&1 | head -n 80
printf '\n===== Inventory complete; interactive shell follows =====\n'
exec bash -l
'@

Write-Host "Visible session output is also recorded at: $logPath"
Write-Host 'Authenticate with your DKUCC password and Duo when prompted.'

# Quote the remote command for bash (single-quote wrapping, escape embedded single quotes).
# Avoid [Management.Automation.Language.CodeGeneration]::QuoteArgument: not available in Windows PowerShell 5.1.
$escapedRemoteCommand = $remoteCommand.Replace("'", "'\''")
$remoteArgument = "bash -lc '" + $escapedRemoteCommand + "'"

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=3 qt28@dkucc-login-01.rc.duke.edu $remoteArgument |
    Tee-Object -FilePath $logPath
$ErrorActionPreference = $previousErrorActionPreference

Write-Host "SSH session ended. Log retained at: $logPath"
Read-Host 'Press Enter to close this window'
