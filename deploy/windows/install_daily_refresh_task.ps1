[CmdletBinding()]
param(
    [string]$TaskName = "Retail Forecast Daily Refresh",
    [string]$RunAsUser = $env:USERNAME
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python.exe" }
$JobScript = Join-Path $ProjectRoot "scripts\04_daily_refresh_job.py"

if (-not (Test-Path -LiteralPath $JobScript)) {
    throw "Daily refresh script not found: $JobScript"
}

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument ('"{0}"' -f $JobScript) `
    -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At "00:00"
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15)
$Principal = New-ScheduledTaskPrincipal `
    -UserId $RunAsUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Refresh retail forecast and Power BI CSV tables daily at 12:00 AM local time." `
    -Force

Write-Host "Installed '$TaskName' at 12:00 AM daily (local machine time)."
Write-Host "Job: $Python $JobScript"
