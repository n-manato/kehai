<#
.SYNOPSIS
  Build the embedded frontend source (frontend-src/) and deploy it to frontend/.

.DESCRIPTION
  The web UI lives as source under frontend-src/ (a copy of KEHAI-Web
  with this project's customizations). The server serves the *built* output from
  frontend/. This script rebuilds from source and swaps the result in, removing
  the previous hashed bundle so only the new one remains.

  Run it after editing anything under frontend-src/. Requires Node.js + npm.

.EXAMPLE
  pwsh scripts/build-frontend.ps1
#>
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $root 'frontend-src'
$distDir = Join-Path $srcDir 'dist\web'
$deployDir = Join-Path $root 'frontend'

# Electron binary isn't needed for the web build; skip its large download.
$env:ELECTRON_SKIP_BINARY_DOWNLOAD = '1'

if (-not (Test-Path (Join-Path $srcDir 'node_modules'))) {
  Write-Host '==> Installing frontend dependencies (first run)...' -ForegroundColor Cyan
  Push-Location $srcDir
  npm install --no-audit --no-fund
  Pop-Location
}

Write-Host '==> Building web frontend...' -ForegroundColor Cyan
Push-Location $srcDir
npm run build:web
Pop-Location

if (-not (Test-Path (Join-Path $distDir 'index.html'))) {
  throw "Build did not produce $distDir\index.html"
}

Write-Host '==> Deploying to frontend/ ...' -ForegroundColor Cyan
$newJs = (Get-ChildItem (Join-Path $distDir 'assets') -Filter 'main-*.js' | Select-Object -First 1).Name
# Every file the new build produced. Anything in frontend/assets NOT in this set
# is a leftover from an older build: vite emits content-hashed names, so a stale
# file is never referenced again, just dead weight.
$wanted = @(Get-ChildItem (Join-Path $distDir 'assets') | ForEach-Object Name)

Copy-Item (Join-Path $distDir 'index.html') (Join-Path $deployDir 'index.html') -Force
Copy-Item (Join-Path $distDir 'assets\*') (Join-Path $deployDir 'assets') -Force
Copy-Item (Join-Path $distDir 'libs\*') (Join-Path $deployDir 'libs') -Recurse -Force
if (Test-Path (Join-Path $distDir 'physics')) {
  # Baked MMD hair-physics data (rigid bodies/joints from the source PMX) for the VRM viewer.
  New-Item -ItemType Directory -Force (Join-Path $deployDir 'physics') | Out-Null
  Copy-Item (Join-Path $distDir 'physics\*') (Join-Path $deployDir 'physics') -Recurse -Force
}
if (Test-Path (Join-Path $distDir 'favicon.ico')) {
  Copy-Item (Join-Path $distDir 'favicon.ico') (Join-Path $deployDir 'favicon.ico') -Force
}

# Mirror assets/: drop anything the new build did not produce. Previously only
# stale 'main-*.js' was cleaned, so every other code-split chunk accumulated
# forever — 22 dead copies of ammo-*.js (1.9 MB each, ~43 MB) had built up.
$stale = Get-ChildItem (Join-Path $deployDir 'assets') -File -ErrorAction SilentlyContinue |
  Where-Object { $wanted -notcontains $_.Name }
foreach ($f in $stale) {
  Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
}
if ($stale) {
  Write-Host "    removed $($stale.Count) stale asset(s)" -ForegroundColor DarkGray
}

# Physics engine assets are load-bearing: if they are missing the viewer only
# logs a warning and silently runs without physics (stiff hair/skirt).
foreach ($f in @('libs\ammo\ammo.wasm.js', 'libs\ammo\ammo.wasm.wasm')) {
  if (-not (Test-Path (Join-Path $deployDir $f))) {
    throw "Deploy is missing $f — physics would be silently disabled."
  }
}

Write-Host "==> Done. Deployed $newJs (replaced: $($oldJs -join ', '))" -ForegroundColor Green
Write-Host '    Hard-reload the browser (Ctrl+Shift+R) to pick up the new build.'
