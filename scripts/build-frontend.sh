#!/usr/bin/env bash
# Build the embedded frontend source (frontend-src/) and deploy it to frontend/.
#
# The web UI lives as source under frontend-src/ (a copy of KEHAI-Web with this
# project's customizations). The server serves the *built* output from frontend/.
# This script rebuilds from source and swaps the result in, removing the previous
# hashed bundle so only the new one remains.
#
# Run it after editing anything under frontend-src/. Requires Node.js + npm.
#
#   bash scripts/build-frontend.sh
#
# Linux port of build-frontend.ps1 — PC-B has node/npm but no pwsh. Keep the two
# in step: everything below that is marked load-bearing is load-bearing in both.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
srcDir="$root/frontend-src"
distDir="$srcDir/dist/web"
deployDir="$root/frontend"

# Electron binary isn't needed for the web build; skip its large download.
export ELECTRON_SKIP_BINARY_DOWNLOAD=1

if [ ! -d "$srcDir/node_modules" ]; then
  echo "==> Installing frontend dependencies (first run)..."
  (cd "$srcDir" && npm install --no-audit --no-fund)
fi

echo "==> Building web frontend..."
(cd "$srcDir" && npm run build:web)

[ -f "$distDir/index.html" ] || { echo "Build did not produce $distDir/index.html" >&2; exit 1; }

echo "==> Deploying to frontend/ ..."
newJs="$(cd "$distDir/assets" && ls main-*.js 2>/dev/null | head -1)"

mkdir -p "$deployDir/assets" "$deployDir/libs"
cp -f "$distDir/index.html" "$deployDir/index.html"
cp -f "$distDir"/assets/* "$deployDir/assets/"
cp -rf "$distDir"/libs/* "$deployDir/libs/"
if [ -d "$distDir/physics" ]; then
  # Baked MMD hair-physics data (rigid bodies/joints from the source PMX) for the VRM viewer.
  mkdir -p "$deployDir/physics"
  cp -rf "$distDir"/physics/* "$deployDir/physics/"
fi
[ -f "$distDir/favicon.ico" ] && cp -f "$distDir/favicon.ico" "$deployDir/favicon.ico"

# Mirror assets/: drop anything the new build did not produce. Vite emits
# content-hashed names, so a file the new build did not emit is never referenced
# again, just dead weight. Pruning only 'main-*.js' once let every other
# code-split chunk accumulate forever — 22 dead copies of ammo-*.js (1.9 MB
# each, ~43 MB) had built up.
stale=0
for f in "$deployDir"/assets/*; do
  [ -f "$f" ] || continue
  if [ ! -f "$distDir/assets/$(basename "$f")" ]; then
    rm -f "$f"
    stale=$((stale + 1))
  fi
done
[ "$stale" -gt 0 ] && echo "    removed $stale stale asset(s)"

# Physics engine assets are load-bearing: if they are missing the viewer only
# logs a warning and silently runs without physics (stiff hair/skirt).
for f in libs/ammo/ammo.wasm.js libs/ammo/ammo.wasm.wasm; do
  [ -f "$deployDir/$f" ] || { echo "Deploy is missing $f — physics would be silently disabled." >&2; exit 1; }
done

echo "==> Done. Deployed $newJs"
echo "    Hard-reload the browser (Ctrl+Shift+R) to pick up the new build."
