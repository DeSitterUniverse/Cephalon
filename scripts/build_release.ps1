param(
  [string]$Version = "1.4.0",
  [switch]$SkipModelExport
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  py -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not $SkipModelExport) {
  if (-not (Test-Path ".venv-export\Scripts\python.exe")) {
    py -m venv .venv-export
  }
  .\.venv-export\Scripts\python.exe -m pip install -r requirements-export.txt
  .\.venv-export\Scripts\python.exe export_onnx.py
}
.\.venv\Scripts\python.exe scripts\validate_onnx_models.py --mark

.\.venv\Scripts\python.exe -m py_compile python\main.py python\test_ingest_query.py python\test_query_only.py
.\.venv\Scripts\python.exe -m pytest
npm.cmd ci
npm.cmd run build
Push-Location src-tauri
cargo check
Pop-Location
.\.venv\Scripts\python.exe build_backend.py
npm.cmd run tauri build

Write-Host "Cephalon $Version release build completed."
