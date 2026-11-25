# Windows build without Visual Studio / nvcc.
# NPP and the CUDA runtime come from NVIDIA's pip wheels (headers + DLLs),
# the host code is compiled with MinGW g++ and linked straight to the DLLs.
# Needs: an NVIDIA driver, Python + pip, MinGW-w64 g++ (e.g. MSYS2 ucrt64).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1

param(
  [string]$Tools = "tools\cuda"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$packages = @{
  "cuda_runtime" = "nvidia-cuda-runtime-cu12"
  "cuda_nvcc"    = "nvidia-cuda-nvcc-cu12"   # only for the crt/ headers
  "npp"          = "nvidia-npp-cu12"
}

foreach ($name in $packages.Keys) {
  $dir = Join-Path $Tools $name
  if (-not (Test-Path "$dir\nvidia\$name")) {
    Write-Output "installing $($packages[$name]) into $dir"
    python -m pip install --quiet --no-deps --target $dir $packages[$name]
    if ($LASTEXITCODE -ne 0) { throw "pip install $($packages[$name]) failed" }
  }
}

$runtime = "$Tools\cuda_runtime\nvidia\cuda_runtime"
$nvcc = "$Tools\cuda_nvcc\nvidia\cuda_nvcc"
$npp = "$Tools\npp\nvidia\npp"

New-Item -ItemType Directory -Force bin | Out-Null

$sources = Get-ChildItem src\*.cc | ForEach-Object { $_.FullName }
$dlls = @("$runtime\bin\cudart64_12.dll") +
        (Get-ChildItem "$npp\bin\npp*.dll" | ForEach-Object { $_.FullName })

Write-Output "compiling $($sources.Count) files with g++"
g++ -std=c++17 -O2 -Wall -Wextra -I. `
  -isystem third_party -isystem "$runtime\include" `
  -isystem "$nvcc\include" -isystem "$npp\include" `
  $sources $dlls -o bin\npp_batch.exe
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Output "built bin\npp_batch.exe"
Write-Output "before running, put the DLL folders on PATH:"
Write-Output "  `$env:PATH = `"$((Resolve-Path "$runtime\bin").Path);$((Resolve-Path "$npp\bin").Path);`$env:PATH`""
