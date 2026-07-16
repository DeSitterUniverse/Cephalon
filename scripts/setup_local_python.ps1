param(
  [switch]$WithExportTools
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Arguments = @("$PSScriptRoot\setup_python.py")
if ($WithExportTools) { $Arguments += "--with-export-tools" }
& py -3.14 @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
