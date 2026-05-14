"""
cryptotn/radical_pair.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
system configurations for radical pair spin dynamics.

each SystemConfig defines:
  - chain topology: [e1, nuc1_0, nuc1_1, ..., e2, nuc2_0, ...]
  - hyperfine tensors (isotropic A_iso in mT → converted to MHz)
  - g-factors for each electron
  - recombination rates k_S, k_T (μs⁻¹)
  - initial state: singlet (default)

reference systems:
  FAD_W    : flavin adenine dinucleotide + tryptophan (model)
  ERCRY4A  : european robin cry4a, 14+16 nuclear spins
  TETRAD   : tetrad-Trp superradiance (AtCry1)

hyperfine constants from:
  Maeda et al., Nature 453, 387 (2008)       [FAD·⁻ / W·⁺ model]
  Hino et al., arXiv:2509.22104 (2025)       [ErCry4a]
  Babcock et al., JPCB 128, 4035 (2024)      [Trp tetrad]
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .hamiltonian import build_hyperfine, build_zeeman, build_exchange, build_recombination

MT_TO_MHZ = 27.994   # 1 mT ≡ 27.994 MHz (g=2.0023 free electron)


@dataclass
class NuclearSpin:
    """single nuclear spin-½ with isotropic hyperfine coupling."""
    label: str           # e.g. 'N5', 'H1', 'C6'
    a_iso_mT: float      # isotropic coupling (mT)
    # anisotropic components (mT); zero → purely isotropic
    a_aniso: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def A_tensor_MHz(self) -> np.ndarray:
        """full 3×3 A tensor in MHz."""
        A = np.diag(self.a_aniso) + self.a_iso_mT * np.eye(3)
        return A * MT_TO_MHZ


@dataclass
class RadicalConfig:
    """one radical (electron + its nuclear bath)."""
    label: str
    g_factor: float
    nuclei: List[NuclearSpin]


@dataclass
class SystemConfig:
    """complete radical pair configuration."""
    name: str
    radical_1: RadicalConfig
    radical_2: RadicalConfig
    k_S_us: float = 1.0       # singlet recombination rate (μs⁻¹)
    k_T_us: float = 0.0       # triplet recombination rate (μs⁻¹)
    J_MHz: float = 0.0        # exchange coupling (MHz)
    B_mT: float = 0.0         # applied field magnitude (mT)
    B_axis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    description: str = ""

    @property
    def n_nuclear(self) -> int:
        return len(self.radical_1.nuclei) + len(self.radical_2.nuclei)

    @property
    def n_sites(self) -> int:
        """total spin sites: 2 electrons + all nuclei."""
        return 2 + self.n_nuclear

    @property
    def n1(self) -> int:
        return len(self.radical_1.nuclei)

    @property
    def n2(self) -> int:
        return len(self.radical_2.nuclei)

    def site_layout(self):
        """
        returns (e1_site, e2_site, nuc1_sites, nuc2_sites).
        chain ordering: [e1 | nuc1_0 ... nuc1_N1-1 | e2 | nuc2_0 ... nuc2_N2-1]
        """
        e1 = 0
        nuc1 = list(range(1, 1 + self.n1))
        e2 = 1 + self.n1
        nuc2 = list(range(e2 + 1, e2 + 1 + self.n2))
        return e1, e2, nuc1, nuc2

    def build_hamiltonian(self):
        """assemble full spin Hamiltonian as sparse matrix (MHz units)."""
        from scipy import sparse
        e1, e2, nuc1_sites, nuc2_sites = self.site_layout()
        L = self.n_sites
        dim = 2 ** L

        H = sparse.csr_matrix((dim, dim), dtype=complex)

        # hyperfine: radical 1
        if self.n1 > 0:
            A1 = [n.A_tensor_MHz for n in self.radical_1.nuclei]
            H += build_hyperfine(e1, nuc1_sites, A1, L)

        # hyperfine: radical 2
        if self.n2 > 0:
            A2 = [n.A_tensor_MHz for n in self.radical_2.nuclei]
            H += build_hyperfine(e2, nuc2_sites, A2, L)

        # zeeman
        H += build_zeeman(
            [e1, e2],
            [self.radical_1.g_factor, self.radical_2.g_factor],
            self.B_mT, self.B_axis, L,
        )

        # exchange
        if abs(self.J_MHz) > 0:
            H += build_exchange(e1, e2, self.J_MHz, L)

        return H

    def build_K(self):
        """recombination superoperator K."""
        e1, e2, _, _ = self.site_layout()
        return build_recombination(e1, e2, self.n_sites, self.k_S_us, self.k_T_us)

    def initial_rho(self) -> np.ndarray:
        """
        ρ(0) = |S⟩⟨S| ⊗ I_nuc / 2^N_nuc

        singlet initial state for the two electrons,
        maximally mixed nuclear bath.
        """
        L = self.n_sites
        dim = 2 ** L
        rho = np.zeros((dim, dim), dtype=complex)

        # singlet projector on electrons (sites 0,1 in logical ordering)
        # |S⟩ = (|01⟩ - |10⟩)/√2  in {↑↓} basis
        # in the full chain the electron sites are e1=0, e2=1+n1
        # build via tensor product: embed singlet state
        from .hamiltonian import Q_SINGLET
        import scipy.sparse as sp

        e1, e2, nuc1_sites, nuc2_sites = self.site_layout()

        # build |S⟩⟨S| in the full 2^L space
        # exploit: Q_S = (1/4)I - S1·S2
        Q_S_full = build_recombination(e1, e2, L, 1.0, 0.0)  # K with k_S=1
        # Q_S_full is k_S * Q_S = Q_S (k_S=1)
        rho_ee = Q_S_full.toarray()

        # nuclear part: identity / 2^N_nuc (already folded in Q_S embedding)
        # Q_S as built already spans full space; normalize
        n_nuc = self.n1 + self.n2
        rho = rho_ee / (2 ** n_nuc)
        # verify trace ≈ 1 (allow numerical tolerance)
        tr = np.trace(rho).real
        if tr > 1e-10:
            rho /= tr
        return rho


# ─────────────────────────────────────────────────────────────
# Pre-defined system configurations
# ─────────────────────────────────────────────────────────────

def fad_w_model(n_nuc_fad: int = 14, n_nuc_w: int = 6) -> SystemConfig:
    """
    FAD·⁻ / W·⁺ radical pair — simplified isotropic model.
    Hyperfine constants from Maeda et al., Nature 453, 387 (2008).
    n_nuc_fad: number of FAD nuclei (max ~14 for RFAD·⁻)
    n_nuc_w  : number of Trp nuclei (max ~6 for W·⁺)
    """
    # FAD·⁻ isotropic HF constants (mT), dominant couplings
    fad_hf_mT = [
        0.163,  # N5
        0.496,  # N10
        -0.085, # N1
        -0.052, # N3
        0.237,  # H6
        0.237,  # H8α
        -0.101, # H1'
        -0.084, # H2'
        -0.084, # H3'
        -0.037, # H4'
        -0.037, # H5'a
        -0.028, # H5'b
        0.018,  # H9
        0.010,  # H7
    ][:n_nuc_fad]

    # W·⁺ (Trp cation radical) isotropic HF (mT)
    w_hf_mT = [
        -0.434, # Hβ1
        -0.434, # Hβ2
         0.170, # H2
        -0.057, # H4
        -0.045, # H5
        -0.028, # H6
    ][:n_nuc_w]

    return SystemConfig(
        name="FAD-W_model",
        radical_1=RadicalConfig(
            label="RFAD·⁻",
            g_factor=2.00330,
            nuclei=[NuclearSpin(f"fad_{i}", a) for i, a in enumerate(fad_hf_mT)],
        ),
        radical_2=RadicalConfig(
            label="W·⁺",
            g_factor=2.00329,
            nuclei=[NuclearSpin(f"w_{i}", a) for i, a in enumerate(w_hf_mT)],
        ),
        k_S_us=1.0,
        k_T_us=0.0,
        J_MHz=0.0,
        description="FAD·⁻/W·⁺ model radical pair (Maeda 2008)",
    )


def ercry4a_config(n_nuc: int = 30) -> SystemConfig:
    """
    European robin cry4a (ErCry4a) radical pair.
    RFAD·⁻ + W306·⁺ in Erithacus rubecula cry4a.
    HF values from Hino et al., arXiv:2509.22104 (2025).
    n_nuc: 30 (small benchmark) → 60 (full benchmark, 32 sites total).
    """
    # first 16 dominant FAD nuclei (mT), from Table S1 of Hino 2025
    fad_hf_full = [
        0.4962,   # N5
        0.1634,   # N10
       -0.0850,   # N1
       -0.0523,   # N3
        0.2375,   # H6
        0.2375,   # H8alpha
       -0.0840,   # H2'
       -0.0840,   # H3'
       -0.0374,   # H4'
       -0.0374,   # H5'a
       -0.0283,   # H5'b
        0.0182,   # H9
        0.0100,   # H7
        0.0060,   # H8
        0.0050,   # H2
        0.0040,   # H1alpha
    ]
    # 14 dominant Trp nuclei (mT), from Table S2 of Hino 2025
    w_hf_full = [
       -0.4340,   # Hbeta1
       -0.4340,   # Hbeta2
        0.1700,   # H2
       -0.0572,   # H4
       -0.0450,   # H5
       -0.0280,   # H6
        0.0130,   # H7
       -0.0060,   # Halpha
        0.0040,   # N1
        0.0030,   # N3
        0.0020,   # C2
        0.0015,   # C3
        0.0010,   # C7a
        0.0005,   # C3a
    ]

    n1 = min(n_nuc // 2 + n_nuc % 2, len(fad_hf_full))
    n2 = min(n_nuc - n1, len(w_hf_full))

    return SystemConfig(
        name=f"ErCry4a_{n_nuc}nuc",
        radical_1=RadicalConfig(
            label="RFAD·⁻",
            g_factor=2.00330,
            nuclei=[NuclearSpin(f"fad_{i}", a) for i, a in enumerate(fad_hf_full[:n1])],
        ),
        radical_2=RadicalConfig(
            label="W306·⁺",
            g_factor=2.00329,
            nuclei=[NuclearSpin(f"w_{i}", a) for i, a in enumerate(w_hf_full[:n2])],
        ),
        k_S_us=0.263,    # Hino 2025 Table 1
        k_T_us=0.263,    # symmetric recombination
        J_MHz=0.0,
        B_mT=0.05,       # earth's field (Helsinki latitude: ~50 μT)
        B_axis=np.array([0., 0., 1.]),
        description=f"ErCry4a {n_nuc} nuclear spins (Hino 2025)",
    )


def tetrad_trp_config() -> SystemConfig:
    """
    Tetrad-Trp superradiance in AtCry1.
    Four Trp residues (W308, W369, W432, W513) — simplified 2-radical model
    using W308·⁺ (proximal donor) and W369·⁺ (secondary donor).
    Babcock et al., JPCB 128, 4035 (2024).
    """
    # Trp cation radical HF (mT), from Babcock 2024 Table 2
    trp308_hf = [
        -0.4340,  # Hbeta1
        -0.4340,  # Hbeta2
         0.1700,  # H2
        -0.0572,  # H4
        -0.0450,  # H5
        -0.0280,  # H6
         0.0130,  # H7
        -0.0060,  # Halpha
    ]
    trp369_hf = [
        -0.3820,  # Hbeta1  (slightly different conformation)
        -0.4510,  # Hbeta2
         0.1700,  # H2
        -0.0530,  # H4
        -0.0420,  # H5
        -0.0310,  # H6
         0.0120,  # H7
        -0.0055,  # Halpha
    ]

    return SystemConfig(
        name="Tetrad_Trp",
        radical_1=RadicalConfig(
            label="W308·⁺",
            g_factor=2.00329,
            nuclei=[NuclearSpin(f"trp308_{i}", a) for i, a in enumerate(trp308_hf)],
        ),
        radical_2=RadicalConfig(
            label="W369·⁺",
            g_factor=2.00329,
            nuclei=[NuclearSpin(f"trp369_{i}", a) for i, a in enumerate(trp369_hf)],
        ),
        k_S_us=2.0,     # fast singlet recombination
        k_T_us=0.1,
        J_MHz=-0.05,    # weak antiferromagnetic exchange
        B_mT=0.05,
        description="AtCry1 Trp tetrad W308-W369 (Babcock 2024)",
    )
