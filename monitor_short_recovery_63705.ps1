$ErrorActionPreference = 'Continue'
$workspace = 'C:\Users\qizhi\Documents\ChatGPT\说话人日志'
$sshHelper = Join-Path $workspace 'dkucc\ssh_moss.py'
Set-Location -LiteralPath $workspace
while ($true) {
    Clear-Host
    Write-Host ('MOSS short-recovery 150 | job 63705 | ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    python $sshHelper --command "squeue -j 63705 -o '%.18i %.24j %.2t %.10M %R'"
    python $sshHelper --command "tail -n 25 /work/qt28/moss/logs/short-recovery-sub150-r3-63705.out"
    python $sshHelper --command "tail -n 15 /work/qt28/moss/logs/short-recovery-sub150-r3-63705.err"
    $state = python $sshHelper --command "squeue -h -j 63705 -o '%T'"
    if (-not $state) {
        python $sshHelper --command "sacct -j 63705 --format=JobID,State,ExitCode,Elapsed"
        Read-Host 'Job left queue. Press Enter to close'
        break
    }
    Start-Sleep -Seconds 20
}
