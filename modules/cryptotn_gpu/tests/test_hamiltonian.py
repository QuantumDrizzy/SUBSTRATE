"""tests/test_hamiltonian.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.hamiltonian import (
    embed, embed2, build_hyperfine, build_zeeman, build_exchange,
    build_recombination, build_liouvillian, build_fmo_hamiltonian,
    Sx, Sy, Sz, Q_SINGLET, Q_TRIPLET,
)


def test_pauli_commutators():
    """[Sx,Sy] = i Sz etc."""
    np.testing.assert_allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-12)
    np.testing.assert_allclose(Sy @ Sz - Sz @ Sy, 1j * Sx, atol=1e-12)
    np.testing.assert_allclose(Sz @ Sx - Sx @ Sz, 1j * Sy, atol=1e-12)


def test_singlet_triplet_projectors():
    """Q_S + Q_T = I, Q_S² = Q_S, Tr(Q_S) = 1."""
    I4 = np.eye(4)
    np.testing.assert_allclose(Q_SINGLET + Q_TRIPLET, I4, atol=1e-12)
    np.testing.assert_allclose(Q_SINGLET @ Q_SINGLET, Q_SINGLET, atol=1e-12)
    assert abs(np.trace(Q_SINGLET) - 1.0) < 1e-12


def test_embed_identity():
    """embedding identity gives global identity."""
    import numpy as np
    I2 = np.eye(2)
    for n in [2, 3, 4]:
        for k in range(n):
            op = embed(I2, k, n).toarray()
            np.testing.assert_allclose(op, np.eye(2**n), atol=1e-12)


def test_embed_spin_commutator():
    """[Sx_0, Sy_0] = i Sz_0 in 3-site chain."""
    n = 3
    Sx0 = embed(Sx, 0, n).toarray()
    Sy0 = embed(Sy, 0, n).toarray()
    Sz0 = embed(Sz, 0, n).toarray()
    np.testing.assert_allclose(Sx0 @ Sy0 - Sy0 @ Sx0, 1j * Sz0, atol=1e-12)


def test_zeeman_diagonal():
    """Zeeman along z: diagonal, eigenvalues ±g*γ*B/2."""
    n = 1
    H_z = build_zeeman([0], [2.0], B_mT=1.0, B_axis=np.array([0., 0., 1.]), n_sites=1)
    eigs = np.linalg.eigvalsh(H_z.toarray())
    # H_z = -g γ_e B Sz, g=2, γ_e=27.994, B=1 → eigenvalues ±g*γ*0.5 = ±27.994
    expected = np.sort([-27.994, 27.994])
    np.testing.assert_allclose(eigs, expected, rtol=1e-4)


def test_recombination_trace_decay():
    """
    evolving rho under K only should give exponential trace decay
    dTr(ρ)/dt = -Tr(K ρ) → Tr(ρ(t)) = exp(-k_S t) for pure singlet start.
    """
    from cryptotn.radical_pair import fad_w_model
    from cryptotn.tdvp import ExactSolver
    cfg = fad_w_model(n_nuc_fad=1, n_nuc_w=0)   # 3-site: e1, nuc, e2
    # set B=0 to test only recombination
    cfg.B_mT = 0.0
    solver = ExactSolver(cfg)
    t, P_S, trace = solver.run(t_max_us=5.0, n_steps=100)
    # trace should be monotonically decreasing
    assert np.all(np.diff(trace) <= 1e-6), "trace should decay monotonically"
    # initial trace ~1
    assert abs(trace[0] - 1.0) < 0.05


def test_liouvillian_trace_preservation_unitary():
    """for K=0, Liouvillian conserves trace (unitary evolution)."""
    from scipy import sparse
    n = 2
    H = embed(Sz, 0, n).toarray()  # simple Sz hamiltonian
    K = np.zeros_like(H)
    L = build_liouvillian(sparse.csr_matrix(H), sparse.csr_matrix(K)).toarray()
    dim = 2 ** n
    rho0 = np.eye(dim) / dim  # maximally mixed
    rho0_vec = rho0.reshape(-1)
    # dTr(ρ)/dt = Tr(L rho) should be ~0 for unitary
    rate = np.real(np.trace((L @ rho0_vec).reshape(dim, dim)))
    assert abs(rate) < 1e-10


def test_fmo_hamiltonian_hermitian():
    H = build_fmo_hamiltonian()
    np.testing.assert_allclose(H, H.T.conj(), atol=1e-10)
    # eigenvalues should be real
    eigs = np.linalg.eigvalsh(H)
    assert eigs.shape == (7,)
