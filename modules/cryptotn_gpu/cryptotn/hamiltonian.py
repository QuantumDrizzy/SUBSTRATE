"""
cryptotn/hamiltonian.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
spin Hamiltonians for radical pair dynamics.

builds the Liouvillian superoperator L such that dρ/dt = L(ρ):

  L(ρ) = -i[H_spin, ρ] - ½{K, ρ}  (Haberkorn recombination)

where:
  H_spin = H_HF + H_Zeeman + H_exchange + H_dipolar
  K      = k_S Q_S + k_T Q_T

all energies in MHz (natural units for EPR).
1 mT ≈ 27.994 MHz (g=2.0023, free electron).

reference: Haberkorn, Mol. Phys. 32, 1491 (1976).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import kron, eye
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Pauli matrices (spin-½)
# ─────────────────────────────────────────────────────────────
Sx = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex)
Sy = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex)
Sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# projectors onto singlet / triplet subspace (two-electron sector)
# |S⟩ = (|↑↓⟩ - |↓↑⟩)/√2
_S00 = np.array([[0, 0, 0, 0],
                 [0, 1, -1, 0],
                 [0, -1, 1, 0],
                 [0, 0, 0, 0]], dtype=complex) * 0.25

Q_SINGLET = _S00 / np.trace(_S00)   # normalized: Tr(Q_S) = 1
Q_TRIPLET = np.eye(4, dtype=complex) - Q_SINGLET


# ─────────────────────────────────────────────────────────────
# Utility: embed an operator on site k in a chain of L sites
# ─────────────────────────────────────────────────────────────

def embed(op: np.ndarray, site: int, n_sites: int) -> sparse.csr_matrix:
    """
    Embed a single-site operator `op` (2×2) at position `site`
    in a chain of `n_sites` spin-½ particles.
    Returns a (2^n_sites × 2^n_sites) sparse matrix.
    """
    result = sparse.eye(1, dtype=complex, format='csr')
    for k in range(n_sites):
        if k == site:
            result = kron(result, sparse.csr_matrix(op), format='csr')
        else:
            result = kron(result, sparse.eye(2, dtype=complex, format='csr'), format='csr')
    return result


def embed2(op_a: np.ndarray, site_a: int,
           op_b: np.ndarray, site_b: int,
           n_sites: int) -> sparse.csr_matrix:
    """
    Embed a two-site operator op_a ⊗ op_b at (site_a, site_b).
    """
    return embed(op_a, site_a, n_sites) @ embed(op_b, site_b, n_sites)


# ─────────────────────────────────────────────────────────────
# Hyperfine Hamiltonian
# ─────────────────────────────────────────────────────────────

def build_hyperfine(
    electron_site: int,
    nuclear_sites: List[int],
    hf_tensors: List[np.ndarray],   # list of 3×3 A tensors (MHz)
    n_sites: int,
) -> sparse.csr_matrix:
    """
    H_HF = Σ_n A_n · (Se · In)
         = Σ_n [Axx Sex Inx + Ayy Sey Iny + Azz Sze Inz
                + Axy Sex Iny + Axz Sex Inz + ...]  (anisotropic)

    for isotropic coupling A = a * I_3: H_HF = a Σ_n Se · In.

    args:
        electron_site  : site index of the electron spin
        nuclear_sites  : list of site indices for coupled nuclei
        hf_tensors     : list of 3×3 hyperfine tensors (MHz)
        n_sites        : total number of sites in chain
    """
    H = sparse.csr_matrix((2 ** n_sites, 2 ** n_sites), dtype=complex)
    ops_e = [Sx, Sy, Sz]

    for n_site, A in zip(nuclear_sites, hf_tensors):
        ops_n = [Sx, Sy, Sz]
        for i, Sei in enumerate(ops_e):
            for j, Inj in enumerate(ops_n):
                if abs(A[i, j]) < 1e-12:
                    continue
                H += A[i, j] * embed2(Sei, electron_site, Inj, n_site, n_sites)
    return H


def build_zeeman(
    electron_sites: List[int],
    g_factors: List[float],
    B_mT: float,
    B_axis: np.ndarray,
    n_sites: int,
) -> sparse.csr_matrix:
    """
    H_Z = -γ_e Σ_e g_e B · Se
        = -γ_e Σ_e g_e |B| (Bx/|B| Sex + By/|B| Sey + Bz/|B| Sez)

    B_axis : unit vector (3,), direction of field
    B_mT   : field magnitude in mT
    γ_e    : 27.994 MHz/mT (free electron, g=2.0023)

    positive B along z → Zeeman splitting ω_Z = g * γ_e * B
    """
    GAMMA_E = 27.994  # MHz/mT (free electron, rounded from 28.0245)
    B_axis = np.asarray(B_axis, dtype=float)
    B_axis = B_axis / np.linalg.norm(B_axis)

    H = sparse.csr_matrix((2 ** n_sites, 2 ** n_sites), dtype=complex)
    ops = [Sx, Sy, Sz]

    for e_site, g in zip(electron_sites, g_factors):
        for alpha, S_alpha in enumerate(ops):
            coeff = -g * GAMMA_E * B_mT * B_axis[alpha]
            if abs(coeff) < 1e-12:
                continue
            H += coeff * embed(S_alpha, e_site, n_sites)
    return H


def build_exchange(
    site_e1: int,
    site_e2: int,
    J_MHz: float,
    n_sites: int,
) -> sparse.csr_matrix:
    """
    H_ex = -J S1·S2  (J > 0: ferromagnetic, J < 0: antiferromagnetic)
    sign convention: negative J stabilizes singlet.
    """
    H = sparse.csr_matrix((2 ** n_sites, 2 ** n_sites), dtype=complex)
    for S in [Sx, Sy, Sz]:
        H += -J_MHz * embed2(S, site_e1, S, site_e2, n_sites)
    return H


# ─────────────────────────────────────────────────────────────
# Haberkorn recombination operator
# ─────────────────────────────────────────────────────────────

def build_recombination(
    site_e1: int,
    site_e2: int,
    n_sites: int,
    k_S: float,       # singlet recombination rate (μs⁻¹)
    k_T: float = 0.0, # triplet recombination rate (μs⁻¹)
) -> sparse.csr_matrix:
    """
    K = k_S Q_S + k_T Q_T   (acts on the two-electron subspace)
    embedded into the full Hilbert space.
    """
    dim = 2 ** n_sites
    K = sparse.csr_matrix((dim, dim), dtype=complex)

    # build Q_S and Q_T embedded at (site_e1, site_e2)
    # Q_S = ¼ I - S1·S2
    Q_S_full = (
        0.25 * sparse.eye(dim, dtype=complex)
        - embed2(Sx, site_e1, Sx, site_e2, n_sites)
        - embed2(Sy, site_e1, Sy, site_e2, n_sites)
        - embed2(Sz, site_e1, Sz, site_e2, n_sites)
    )
    Q_T_full = sparse.eye(dim, dtype=complex) - Q_S_full

    K += k_S * Q_S_full
    if abs(k_T) > 0:
        K += k_T * Q_T_full
    return K


# ─────────────────────────────────────────────────────────────
# Liouvillian superoperator (vectorized density matrix)
# ─────────────────────────────────────────────────────────────

def build_liouvillian(
    H: sparse.csr_matrix,
    K: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """
    L = -i(H ⊗ I - I ⊗ H*) - ½(K ⊗ I + I ⊗ K*)

    acts on vectorized density matrix |ρ⟩⟩ = vec(ρ).
    convention: vec stacks columns.

    dρ/dt = L(ρ)  ↔  d|ρ⟩⟩/dt = L_super |ρ⟩⟩
    """
    dim = H.shape[0]
    I = sparse.eye(dim, dtype=complex, format='csr')

    L_coherent = -1j * (kron(H, I, format='csr') - kron(I, H.conj(), format='csr'))
    L_kinetic  = -0.5 * (kron(K, I, format='csr') + kron(I, K.conj(), format='csr'))
    return L_coherent + L_kinetic


# ─────────────────────────────────────────────────────────────
# FMO Hamiltonian (open quantum system, cm⁻¹ units)
# ─────────────────────────────────────────────────────────────
# Adolphs & Renger, Biophysical Journal 91, 2778 (2006)
# 7-site BChl complex; energies in cm⁻¹

FMO_SITE_ENERGIES_CM1 = np.array([215.0, 220.0, 0.0, 125.0, 450.0, 330.0, 280.0])

FMO_COUPLINGS_CM1 = {
    (0, 1): -104.1, (0, 2):   5.1, (0, 3):  -4.3,
    (0, 4):    4.7, (0, 5): -15.1, (0, 6):  -7.8,
    (1, 2):   32.6, (1, 3):   7.1, (1, 4):  11.5,
    (1, 5):    8.3, (1, 6):   0.8,
    (2, 3):  -46.8, (2, 4):   1.0, (2, 5):  -8.1, (2, 6): 5.1,
    (3, 4):  -70.7, (3, 5): -14.7, (3, 6): -61.5,
    (4, 5):   89.7, (4, 6):  -2.5,
    (5, 6):   32.7,
}


def build_fmo_hamiltonian() -> np.ndarray:
    """
    returns the 7×7 FMO Hamiltonian in cm⁻¹ (dense, Hermitian).
    Site 3 is set to zero as energy reference.
    """
    n = 7
    H = np.diag(FMO_SITE_ENERGIES_CM1.copy())
    for (i, j), v in FMO_COUPLINGS_CM1.items():
        H[i, j] = v
        H[j, i] = v
    return H


def build_fmo_lindblad(
    T_K: float = 300.0,
    lambda_cm1: float = 35.0,
    gamma_cm1: float = 106.14,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    returns (H_FMO, [L_k]) for Redfield/Lindblad simulation.

    bath: Drude-Lorentz spectral density J(ω) = 2λγω/(ω²+γ²)
    λ = reorganisation energy (cm⁻¹)
    γ = Drude cutoff (cm⁻¹)
    T = temperature (K)

    lindblad operators: secular approximation, site-basis dephasing.
    L_k = sqrt(γ_k) |k⟩⟨k|
    γ_k ≈ 2λ k_B T / ħ γ  (high-T Markovian limit)

    note: full non-Markovian HEOM treatment is in the TENSO benchmark.
    """
    # k_B T in cm⁻¹ (k_B = 0.695 cm⁻¹/K)
    kBT = 0.695 * T_K
    n = 7
    H = build_fmo_hamiltonian()
    # dephasing rate per site in secular approx
    gamma_deph = 2.0 * lambda_cm1 * kBT / gamma_cm1
    lindblad_ops = [np.sqrt(gamma_deph) * np.diag(np.eye(n)[k]) for k in range(n)]
    return H, lindblad_ops
