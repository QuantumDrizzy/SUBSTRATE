"""SUBSTRATE as the 2nd consumer of the Spectra spine.

Builds a real multi-spin Hamiltonian from cryptotn's own operators (embed + Pauli
matrices — a Heisenberg chain) and gets its extremal eigenvalues (ground-state
energy + top) via Spectra's **matrix-free** Lanczos, never densifying the 2^n
Hamiltonian. Validated against the dense NumPy reference on a size where dense is
still feasible.

This is the validation that lets Spectra freeze its API (1st consumer: AETHER;
2nd: SUBSTRATE). It does NOT touch the production engine — additive only.

Requires: spectra (pip install -e ../../../Spectra) and cryptotn. Skips if absent.

Run standalone:  python modules/cryptotn_gpu/tests/test_spectra_consumer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # cryptotn_gpu/

try:
    import scipy.sparse as sp

    from cryptotn.hamiltonian import Sx, Sy, Sz, embed
    from spectra import LinearOperator, lanczos

    HAVE_DEPS = True
except Exception:  # noqa: BLE001
    HAVE_DEPS = False


def _heisenberg_chain(n: int, j: float = 1.0):
    """Open Heisenberg chain H = J Σ_i (Sx_i Sx_{i+1} + Sy Sy + Sz Sz) — sparse, 2^n."""
    dim = 2 ** n
    h = sp.csr_matrix((dim, dim), dtype=complex)
    for i in range(n - 1):
        for s in (Sx, Sy, Sz):
            h = h + j * (embed(s, i, n) @ embed(s, i + 1, n))
    return h.tocsr()


def test_spectra_groundstate_matches_dense():
    if not HAVE_DEPS:
        print("  SKIP  (spectra/cryptotn not available)")
        return
    n = 8  # 256-dimensional Hilbert space; dense reference still feasible
    h = _heisenberg_chain(n)

    # Matrix-free: Spectra never forms the 2^n matrix, only applies H @ x.
    op = LinearOperator.from_matvec(lambda x: h @ x, n=h.shape[0], hermitian=True)
    ritz = lanczos(op, k=4, iters=80)

    ref = np.linalg.eigvalsh(h.toarray())
    assert abs(ritz[0] - ref[0]) < 1e-6, f"ground state {ritz[0]} vs {ref[0]}"
    assert abs(ritz[-1] - ref[-1]) < 1e-6, f"top state {ritz[-1]} vs {ref[-1]}"


def test_spectra_matrixfree_scales_past_dense():
    """Sanity: matrix-free Lanczos runs on a system too big to densify casually
    (2^12 = 4096) and returns a finite ground-state energy."""
    if not HAVE_DEPS:
        print("  SKIP  (spectra/cryptotn not available)")
        return
    n = 12  # 4096-dim; we never call .toarray() here
    h = _heisenberg_chain(n)
    op = LinearOperator.from_matvec(lambda x: h @ x, n=h.shape[0], hermitian=True)
    ritz = lanczos(op, k=2, iters=60)
    assert np.isfinite(ritz[0]) and ritz[0] < 0.0, "AFM Heisenberg ground state should be negative"


def _run_standalone() -> int:
    tests = [test_spectra_groundstate_matches_dense, test_spectra_matrixfree_scales_past_dense]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
