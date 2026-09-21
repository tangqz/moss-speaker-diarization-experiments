$ErrorActionPreference = 'Continue'
$workspace = 'C:\Users\qizhi\Documents\ChatGPT\说话人日志'
$sshHelper = Join-Path $workspace 'dkucc\ssh_moss.py'
$remoteLog = '/work/qt28/moss/logs/short-recovery-sub150-r2-63701.out'
$remoteErr = '/work/qt28/moss/logs/short-recovery-sub150-r2-63701.err'

Set-Location -LiteralPath $workspace
while ($true) {
    Clear-Host
    Write-Host ('MOSS short-recovery sub 150 retry | job 63701 | ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    python $sshHelper --command "squeue -j 63701 -o '%.18i %.24j %.2t %.10M %R'"
    Write-Host "`n--- stdout (last 45 lines) ---"
    python $sshHelper --command "test -f '$remoteLog' && tail -n 45 '$remoteLog' || true"
    Write-Host "`n--- stderr (last 25 lines) ---"
    python $sshHelper --command "test -f '$remoteErr' && tail -n 25 '$remoteErr' || true"
    $state = python $sshHelper --command "squeue -h -j 63701 -o '%T'"
    if (-not $state) {
        Write-Host "`nJob has left the queue. Press Enter to close."
        Read-Host
        break
    }
    Start-Sleep -Seconds 20
}
