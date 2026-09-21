$ErrorActionPreference = 'Continue'
$workspace = 'C:\Users\qizhi\Documents\ChatGPT\说话人日志'
Set-Location -LiteralPath $workspace
$remoteRun = '/work/qt28/moss/results/short-recovery-sr-v2-sub-150-20260916-r3/sub/seed-0/evaluations/test-150-20260916'
$Host.UI.RawUI.WindowTitle = 'MOSS generation evaluation | 63719'
while ($true) {
    Clear-Host
    Write-Host ('MOSS generation evaluation | job 63719 | ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    python dkucc\ssh_moss.py --command "squeue -j 63719 -o '%i %T %M %N'; cat '$remoteRun/status.json' '$remoteRun/test_progress.json' 2>/dev/null; tail -n 8 /work/qt28/moss/logs/short-recovery-gen150-63719.out; tail -n 8 /work/qt28/moss/logs/short-recovery-gen150-63719.err"
    Start-Sleep -Seconds 20
}
