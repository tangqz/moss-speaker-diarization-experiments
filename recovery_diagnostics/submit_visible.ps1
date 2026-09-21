$ErrorActionPreference = 'Stop'

$sourceDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $sourceDirectory)
$logDirectory = Join-Path $workspaceRoot 'dkucc\logs'
$packageDirectory = Join-Path $workspaceRoot 'output\recovery-diagnostics'
New-Item -ItemType Directory -Force -Path $logDirectory, $packageDirectory | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$archive = Join-Path $packageDirectory "recovery-diagnostics-$timestamp.tar.gz"
$submitLog = Join-Path $logDirectory "recovery-diagnostics-submit-$timestamp.log"
$remoteArchive = "/work/qt28/moss/recovery-diagnostics-$timestamp.tar.gz"
$remoteSource = '/work/qt28/moss/dkucc/recovery_diagnostics_20260915'

& tar.exe -czf $archive -C $sourceDirectory .
if ($LASTEXITCODE -ne 0) {
    throw "Failed to package the diagnostic source."
}

Write-Host 'D0/D1 diagnostic package is ready.'
Write-Host 'DKUCC will request your password and Duo for upload, then again for submission.'
Write-Host "Visible submission output will also be saved to: $submitLog"

& scp -o ServerAliveInterval=30 -o ServerAliveCountMax=3 $archive "qt28@dkucc-login-01.rc.duke.edu:$remoteArchive"
if ($LASTEXITCODE -ne 0) {
    throw "Upload failed; no Slurm job was submitted."
}

$remoteCommand = @'
set -euo pipefail
mkdir -p /work/qt28/moss/dkucc/recovery_diagnostics_20260915
tar -xzf __REMOTE_ARCHIVE__ -C /work/qt28/moss/dkucc/recovery_diagnostics_20260915
cd /work/qt28/moss/dkucc/recovery_diagnostics_20260915
python3 -m py_compile prepare_cases.py hf_replay.py vllm_replay.py summarize.py
job_id=$(sbatch --parsable d0_d1.slurm)
printf '%s\n' "$job_id" > /work/qt28/moss/results/ms-swift-63643/recovery_diagnostics_job_id.txt
printf '\nD0/D1 diagnostic submitted as Slurm job %s\n' "$job_id"
squeue -j "$job_id" -o '%i %j %T %M %l %R'
printf '\nLogs: /work/qt28/moss/logs/moss-recovery-d0d1-%s.{out,err}\n' "$job_id"
'@.Replace('__REMOTE_ARCHIVE__', $remoteArchive)

$escapedRemoteCommand = $remoteCommand.Replace("'", "'\''")
$remoteArgument = "bash -lc '" + $escapedRemoteCommand + "'"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=3 qt28@dkucc-login-01.rc.duke.edu $remoteArgument |
    Tee-Object -FilePath $submitLog
$sshExit = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($sshExit -ne 0) {
    throw "Remote validation or submission failed. Review $submitLog"
}

Write-Host 'Submission finished. This window can now be closed.'
Read-Host 'Press Enter to close'
