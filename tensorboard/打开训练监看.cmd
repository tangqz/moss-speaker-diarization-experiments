@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 exit /b 1
start "" "http://127.0.0.1:6006"
