$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

& py -3.14 "$PSScriptRoot\setup_python.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
