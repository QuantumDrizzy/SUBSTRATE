#!/usr/bin/env bash
# cycle_project — fetch + preprocess all proxies
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== cycle_project :: fetch_data ==="
python src/cycle_detect/fetch_data.py "$@"
echo ""
echo "Output: data/processed/"
ls -lh data/processed/ 2>/dev/null || true
