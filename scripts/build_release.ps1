param(
  [string]$Version = "2.0.0",
  [switch]$WithModelExport
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Arguments = @("$PSScriptRoot\build_release.py", "--version", $Version)
if ($WithModelExport) { $Arguments += "--with-model-export" }
& py -3.14 @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
