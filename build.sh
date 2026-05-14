#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "╔══════════════════════════════════════╗"
echo "║     Building SUBSTRATE  v0.1.0       ║"
echo "╚══════════════════════════════════════╝"

# ── 1. Python engine deps ────────────────────────────────────────────────────
echo ""
echo "── Python engine ──────────────────────"
pip install -e . --quiet
echo "   substrate-engine installed"

# ── 2. CUDA kernels (optional) ───────────────────────────────────────────────
echo ""
echo "── CUDA kernels ────────────────────────"
if command -v nvcc &>/dev/null; then
    cmake -B build/cuda cuda/ -DCMAKE_BUILD_TYPE=Release
    cmake --build build/cuda --parallel
    echo "   CUDA kernels built → build/cuda/"
else
    echo "   nvcc not found — skipping CUDA build"
fi

# ── 3. Rust workspace ────────────────────────────────────────────────────────
echo ""
echo "── Rust workspace ──────────────────────"
cargo build --release 2>&1
echo "   substrate binary → target/release/substrate"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  SUBSTRATE ready                     ║"
echo "║  Run: ./target/release/substrate run ║"
echo "╚══════════════════════════════════════╝"
