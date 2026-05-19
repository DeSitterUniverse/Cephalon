param(
  [Parameter(Mandatory = $true)]
  [string]$Namespace
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ProjectsRoot = Split-Path -Parent $RepoRoot
$EmbedderDir = Join-Path $ProjectsRoot "jina-embeddings-v5-small-onnx"
$RerankerDir = Join-Path $ProjectsRoot "jina-reranker-v3-onnx"

function Resolve-HfCli {
  $hf = Get-Command hf -ErrorAction SilentlyContinue
  if ($hf) { return $hf.Source }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) { throw "Python is required to locate the Hugging Face CLI." }
  $scriptsDir = & $python.Source -c "import sysconfig; print(sysconfig.get_path('scripts'))"
  $hfExe = Join-Path $scriptsDir "hf.exe"
  if (Test-Path $hfExe) { return $hfExe }
  throw "Hugging Face CLI was not found. Install huggingface_hub or add hf.exe to PATH."
}

$Hf = Resolve-HfCli
& $Hf auth whoami

& $Hf repos create "$Namespace/jina-embeddings-v5-small-onnx" --type model --exist-ok
& $Hf repos create "$Namespace/jina-reranker-v3-onnx" --type model --exist-ok
& $Hf upload-large-folder "$Namespace/jina-embeddings-v5-small-onnx" "$EmbedderDir" --type model
& $Hf upload-large-folder "$Namespace/jina-reranker-v3-onnx" "$RerankerDir" --type model

Write-Host "Uploaded ONNX model repos:"
Write-Host "https://huggingface.co/$Namespace/jina-embeddings-v5-small-onnx"
Write-Host "https://huggingface.co/$Namespace/jina-reranker-v3-onnx"
