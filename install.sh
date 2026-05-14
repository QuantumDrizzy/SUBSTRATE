#!/usr/bin/env bash
# SUBSTRATE — Arch Linux installer (iNFAMØUS)
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[SUBSTRATE]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()   { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}"
cat << 'EOF'
╔══════════════════════════════════════════════════╗
║   SUBSTRATE — Unified Field Analysis System      ║
║   Arch Linux / iNFAMØUS installer               ║
╚══════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 1. System packages
info "Checking system dependencies..."
PKGS=()
for pkg in base-devel cmake python python-pip git; do
    pacman -Q "$pkg" &>/dev/null || PKGS+=("$pkg")
done
if [ ${#PKGS[@]} -gt 0 ]; then
    info "Installing: ${PKGS[*]}"
    sudo pacman -S --noconfirm --needed "${PKGS[@]}"
fi

# CUDA (optional — skip if not available)
if ! pacman -Q cuda &>/dev/null; then
    warn "CUDA not found via pacman — trying cuda from extra/multilib..."
    sudo pacman -S --noconfirm --needed cuda cudnn 2>/dev/null || warn "CUDA not installed — GPU kernels disabled"
fi

# ZeroMQ (required by modules/GRAVITACHYON/rust)
if ! pacman -Q zeromq &>/dev/null; then
    info "Installing zeromq (required by GRAVITACHYON Rust crate)..."
    sudo pacman -S --noconfirm --needed zeromq || warn "zeromq not installed — GRAVITACHYON ZMQ transport disabled"
fi

# 2. Rust
if ! command -v cargo &>/dev/null; then
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
info "Rust $(rustc --version)"

# 3. Python dependencies
info "Installing Python packages..."
pip install --break-system-packages --upgrade pip
pip install --break-system-packages \
    numpy scipy pandas pyarrow scikit-learn \
    torch matplotlib healpy \
    quimb cotengra 2>/dev/null || true

# cryptotn_gpu — tensor network engine for radical pair dynamics (absorbed, no .git)
info "Installing cryptotn_gpu..."
pip install --break-system-packages -e modules/cryptotn_gpu || warn "cryptotn_gpu install failed — quantum layer will use numpy fallback"

# cryptotn_gpu GPU extras (cupy + cuquantum) — requires CUDA runtime
if command -v nvcc &>/dev/null; then
    info "Installing cryptotn_gpu GPU extras (cupy, cuquantum)..."
    pip install --break-system-packages -e "modules/cryptotn_gpu[gpu]" 2>/dev/null \
        || warn "cryptotn_gpu[gpu] install failed — CuPy/cuQuantum not available"
fi

# EEG layer — pylsl (LSL client) + muselsl (Muse 2 BLE bridge)
# Simulated mode works without either. Real mode needs both + Muse 2 paired.
pip install --break-system-packages pylsl muselsl 2>/dev/null \
    || warn "pylsl/muselsl not installed — EEG layer will run in simulated mode"
# muselsl usage: muselsl stream &  (starts LSL stream from Muse 2 over BLE)
#                muselsl view      (optional real-time plot)
# Then SUBSTRATE picks it up automatically via pylsl.resolve_byprop('type','EEG')

# Install engine as editable package
pip install --break-system-packages -e . 2>/dev/null || true

# 4. CUDA kernels (optional)
# sm_120 = Blackwell (RTX 5060 Ti); include legacy arches for portability
if command -v nvcc &>/dev/null; then
    info "Building CUDA kernels..."
    cmake -B build/cuda cuda/ -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90;120"
    cmake --build build/cuda --parallel "$(nproc)"
    info "CUDA kernels built"
else
    warn "nvcc not found — skipping CUDA kernel build (CPU fallback active)"
fi

# 5. Rust release build
info "Building SUBSTRATE (release)..."
cargo build --release
info "Binary: $(pwd)/target/release/substrate"

# 6. Smoke tests
info "Running layer smoke tests..."
LAYERS=(quantum geomagnetic magnon quantum_lab solar cosmological eeg)
N_LAYERS=${#LAYERS[@]}
PASS=0; FAIL=0
for layer in "${LAYERS[@]}"; do
    if python -c "
import sys, json
sys.path.insert(0, 'engine')
sys.path.insert(0, '.')
mod = __import__('${layer}_layer')
r = mod.run()
assert isinstance(r.get('score'), float) and 0.0 <= r['score'] <= 1.0, f'bad score: {r}'
print(f'  ${layer:<12} score={r[\"score\"]:.4f}  mode={r[\"data\"].get(\"mode\", r[\"data\"].get(\"blend_method\", \"?\"))}')
" 2>/dev/null; then
        ((PASS++))
    else
        warn "  ${layer} layer smoke test failed (check engine/${layer}_layer.py)"
        ((FAIL++))
    fi
done
info "Smoke tests: ${PASS}/${N_LAYERS} passed"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   SUBSTRATE ready                                ║${NC}"
echo -e "${GREEN}║   ./target/release/substrate run                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
