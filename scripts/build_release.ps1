param(
  [string]$Version = "1.5.0",
  [switch]$WithModelExport
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Resolve-LocalPython {
  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) {
    return [pscustomobject]@{ Exe = $python.Source; Args = @() }
  }
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    return [pscustomobject]@{ Exe = $py.Source; Args = @("-3") }
  }
  throw "Python was not found on PATH. Enable the Windows Python app execution alias or add python.exe/py.exe to PATH."
}

$ResolvedPython = Resolve-LocalPython
$script:PythonExe = $ResolvedPython.Exe
$script:PythonArgs = @($ResolvedPython.Args)
$env:PYTHONNOUSERSITE = "1"

function Invoke-LocalPython {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
  & $script:PythonExe @script:PythonArgs @Args
}

Invoke-LocalPython -m pip install --upgrade -r requirements.txt
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
$env:FORCE_CMAKE = "1"
Invoke-LocalPython -m pip install --upgrade --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
if ($WithModelExport) {
  Invoke-LocalPython -m pip install --upgrade -r requirements-export.txt
  Invoke-LocalPython export_onnx.py
  Invoke-LocalPython scripts\validate_onnx_models.py --mark
}
Invoke-LocalPython scripts\preflight_runtime.py --skip-onnx

Invoke-LocalPython -m py_compile python\main.py python\test_ingest_query.py python\test_query_only.py
Invoke-LocalPython -m pytest
npm.cmd ci
npm.cmd run build
Push-Location src-tauri
cargo check
Pop-Location
Invoke-LocalPython build_backend.py
npm.cmd run tauri build

Write-Host "Cephalon $Version release build completed."
