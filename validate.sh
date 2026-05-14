#!/usr/bin/env bash
# SUBSTRATE — pre-switch validation checklist
set -uo pipefail

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $*"; ((FAIL++)); }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}SUBSTRATE — Validation Checklist${NC}"
echo "────────────────────────────────────"

# Rust tests
echo -e "\n${BOLD}[1] Rust workspace tests${NC}"
if cargo test --workspace --quiet 2>&1 | tail -3; then
    ok "cargo test passed"
else
    fail "cargo test failed"
fi

# Binary exists
echo -e "\n${BOLD}[2] Release binary${NC}"
if [ -f "target/release/substrate" ] || [ -f "target/release/substrate.exe" ]; then
    ok "binary found"
else
    fail "binary missing — run install.sh first"
fi

# Python layer smoke tests
echo -e "\n${BOLD}[3] Layer smoke tests${NC}"
for layer in quantum geomagnetic magnon quantum_lab solar cosmological eeg; do
    if python -c "
import sys
sys.path.insert(0, 'engine')
sys.path.insert(0, '.')
mod = __import__('${layer}_layer')
r = mod.run()
assert 0 <= r['score'] <= 1, f'score out of range: {r[\"score\"]}'
" 2>/dev/null; then
        ok "${layer}"
    else
        fail "${layer}"
    fi
done

# GRAVITACHYON workspace member check (optional — requires zeromq)
echo -e "\n${BOLD}[3b] GRAVITACHYON Rust crate (optional)${NC}"
if grep -q '"modules/GRAVITACHYON/rust"' Cargo.toml 2>/dev/null; then
    if cargo check -p gravitachyon_rs --quiet 2>/dev/null; then
        ok "gravitachyon_rs compiles"
    else
        fail "gravitachyon_rs failed — check zmq/libzmq (pacman -S zeromq)"
    fi
else
    echo -e "  ${GREEN}–${NC} gravitachyon_rs commented out in Cargo.toml (zmq not yet available)"
fi

# Full report
echo -e "\n${BOLD}[4] substrate report${NC}"
if ./target/release/substrate report 2>/dev/null || \
   ./target/release/substrate.exe report 2>/dev/null; then
    ok "report executed"
else
    fail "report failed"
fi

# Summary
echo ""
echo "────────────────────────────────────"
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}ALL $TOTAL CHECKS PASSED — ready for iNFAMØUS${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}$FAIL/$TOTAL CHECKS FAILED — fix before switching${NC}"
    exit 1
fi
