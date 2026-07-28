$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

& py -3.14 "$PSScriptRoot\build_release.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
