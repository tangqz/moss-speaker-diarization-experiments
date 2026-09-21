$ErrorActionPreference = 'Continue'
$workspace = 'C:\Users\qizhi\Documents\ChatGPT\说话人日志'
Set-Location -LiteralPath $workspace
$run = Get-Content dkucc\repeated_error_recovery\current_run.json -Raw | ConvertFrom-Json
$job = $run.job_id
$Host.UI.RawUI.WindowTitle = "MOSS error-repeat seed1 | $job"
while ($true) {
    Clear-Host
    Write-Host ("MOSS error-repeat seed1 | job $job | " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host 'Milestone gates: 30 / 60 / 90 / 120 / 150; repetition_penalty = 1.0'
    $command = "squeue -j $job -o '%i %T %M %N'; cat '$($run.controller_root)/status.json' 2>/dev/null; find '$($run.training_root)' -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f`n' 2>/dev/null | sort -V; find '$($run.training_root)/evaluations' -name gate_decision.json -print -exec cat {} `; 2>/dev/null; tail -n 12 '$($run.stdout)'; tail -n 8 '$($run.stderr)'"
    python dkucc\ssh_moss.py --command $command
    $state = python dkucc\ssh_moss.py --command "squeue -h -j $job -o '%T'"
    if (-not $state) {
        python dkucc\ssh_moss.py --command "sacct -j $job --format=JobID,State,ExitCode,Elapsed"
        Read-Host 'Job left queue. Press Enter to close'
        break
    }
    Start-Sleep -Seconds 20
}
