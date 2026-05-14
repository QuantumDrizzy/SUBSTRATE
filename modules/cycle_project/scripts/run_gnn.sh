#!/usr/bin/env bash
# cycle_project — GNN anomaly detection
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== cycle_project :: gnn_prototype ==="
python src/cycle_detect/gnn_prototype.py \
    --window 5000 \
    --stride 500  \
    --corr-thresh 0.4 \
    --epochs 150  \
    "$@"
