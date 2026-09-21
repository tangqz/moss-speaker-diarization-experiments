$ErrorActionPreference = 'Continue'
$workspace = 'C:\Users\qizhi\Documents\ChatGPT\说话人日志'
Set-Location -LiteralPath $workspace
while ($true) {
    $run = Get-Content dkucc\short_recovery_generation\current_run.json -Raw | ConvertFrom-Json
    $job = $run.job_id
    $Host.UI.RawUI.WindowTitle = "MOSS generation evaluation | $job"
    Clear-Host
    Write-Host ("MOSS generation evaluation | job $job | " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host 'temperature = 0; repetition_penalty = 1.0'
    $command = "sacct -j $job --format=JobID,State,Elapsed -n; cat '$($run.remote_run)/status.json' 2>/dev/null; tail -n 6 '$($run.remote_log)'; tail -n 6 '$($run.remote_error_log)'; tail -c 1800 '$($run.remote_run)/logs/generation.log' 2>/dev/null"
    python dkucc\ssh_moss.py --command $command
    Start-Sleep -Seconds 20
}
