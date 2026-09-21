$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'MOSS MS-Swift - SSH login and status'
$sshExecutable = 'C:\Program Files\Git\usr\bin\ssh.exe'
$controlPath = 'C:/Users/qizhi/.ssh/cm-moss-check-20260912'

Write-Host '请登录 DKUCC 并完成 Duo；登录后保持此窗口开启。'
$remoteCommand = @'
date -Is
squeue -j 63532 -o '%i|%T|%M|%L|%R'
sacct -j 63532 --format=JobID,State,ExitCode,Elapsed,Start,End -P
cat /work/qt28/moss/results/ms-swift-63521/status.json 2>/dev/null || true
exec bash -l
'@
$remoteCommand = $remoteCommand.Replace("`r`n", "`n")

& $sshExecutable -tt -M -S $controlPath -o ServerAliveInterval=30 -o ServerAliveCountMax=3 qt28@dkucc-login-01.rc.duke.edu $remoteCommand
Read-Host 'SSH 已结束；按 Enter 关闭窗口'
