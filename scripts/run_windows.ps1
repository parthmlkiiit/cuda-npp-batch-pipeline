# Windows version of run.sh: build, run on the whole dataset, pack the proof.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 [extra flags]
#   e.g.  ... run_windows.ps1 --ops=equalize,sobel --max_images=20

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$Tools = "tools\cuda"
& "$PSScriptRoot\build_windows.ps1" -Tools $Tools

if (-not (Test-Path data\sipi) -or -not (Get-ChildItem data\sipi -Filter *.png)) {
  Write-Output "data\sipi is empty, downloading the dataset (needs Git Bash)"
  $env:PYTHON = "python"
  bash scripts/download_data.sh
  if ($LASTEXITCODE -ne 0) { throw "dataset download failed" }
}

$env:PATH = "$((Resolve-Path "$Tools\cuda_runtime\nvidia\cuda_runtime\bin").Path);" +
            "$((Resolve-Path "$Tools\npp\nvidia\npp\bin").Path);$env:PATH"

New-Item -ItemType Directory -Force output | Out-Null
$flags = @("--input_dir=data/sipi", "--output_dir=output") + $args

# cmd keeps the log plain ASCII (PowerShell 5 Tee-Object writes UTF-16).
cmd /c "bin\npp_batch.exe $($flags -join ' ') > output\run.log 2>&1"
$code = $LASTEXITCODE
Get-Content output\run.log
if ($code -ne 0) { throw "npp_batch exited with code $code" }

tar -czf proof_of_execution.tar.gz output
Write-Output "packed output\ into proof_of_execution.tar.gz"
