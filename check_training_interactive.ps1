$ErrorActionPreference = 'Continue'
$sshExecutable = 'C:\Program Files\Git\usr\bin\ssh.exe'
$logDirectory = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ('ssh-status-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$controlPath = 'C:/Users/qizhi/.ssh/cm-moss-check-20260912'
$runInfoPath = Join-Path $PSScriptRoot 'evaluation\current_run.json'
$evalJob = '63363'
if (Test-Path -LiteralPath $runInfoPath) {
    $runInfo = Get-Content -LiteralPath $runInfoPath -Raw | ConvertFrom-Json
    $evalJob = [string]$runInfo.evaluation_job
    if ($evalJob -notmatch '^\d+$') { throw 'Invalid evaluation job ID.' }
}

$remoteCommand = @'
date -Is
squeue -u qt28
sacct -j 62994,__EVAL_JOB__ --format=JobID,State,ExitCode,Elapsed,Start,End -P
tail -5 /work/qt28/moss/logs/moss-eval-__EVAL_JOB__.out
tail -8 /work/qt28/moss/logs/moss-eval-__EVAL_JOB__.err
for f in /work/qt28/moss/results/eval-62994-__EVAL_JOB__/worker-*-status.json; do
    if [ -f "$f" ]; then cat "$f"; fi
done
if [ -f /work/qt28/moss/results/eval-62994-__EVAL_JOB__/job_exit.json ]; then
    cat /work/qt28/moss/results/eval-62994-__EVAL_JOB__/job_exit.json
fi
echo SSH_STATUS_CAPTURE_COMPLETE
exec bash -l
'@
$remoteCommand = $remoteCommand.Replace("`r`n", "`n").Replace('__EVAL_JOB__', $evalJob)

Write-Host "DKUCC status: training 62994 and evaluation $evalJob"
Write-Host 'Enter your DKUCC password and complete Duo in this window.'
Write-Host 'After login, the status will be collected automatically. Keep this window open.'
Write-Host "Output log: $logPath"
& $sshExecutable -tt -M -S $controlPath -o ServerAliveInterval=30 -o ServerAliveCountMax=3 qt28@dkucc-login-01.rc.duke.edu $remoteCommand |
    Tee-Object -FilePath $logPath
Write-Host "SSH ended. Saved output: $logPath"
Read-Host 'Press Enter to close'
