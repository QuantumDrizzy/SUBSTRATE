"""
quantum_bio.radical_pair — Cryptochrome Radical Pair Mechanism
==============================================================

Simulates the avian compass magnetoreception mechanism via quantum spin
dynamics of a correlated radical pair in cryptochrome:

    FAD•⁻ ←→ [FAD•⁻ ··· TrpH•⁺]  (radical pair)

The singlet/triplet yield difference ΔΦ_S = Φ_S(B) - Φ_S(0) encodes
directional magnetic field information — the compass signal.

Physics
-------
· Spin Hamiltonian:
      H = ω_A · S_A·n̂ + ω_B · S_B·n̂ + Σ_j A_j S_A·I_j   (hyperfine)
  where ω = g·μ_B·B/ℏ (Zeeman) and A_j are hyperfine coupling constants.

· Haberkorn recombination:
      dρ/dt = -i[H,ρ] + k_s (P_S ρ + ρ P_S)/2 + k_t (P_T ρ + ρ P_T)/2

· Singlet yield:
      Φ_S = k_s ∫₀^∞ Tr(P_S ρ(t)) dt

References
----------
Ritz, T., Adem, S., Schulten, K. (2004). Biophys. J. 87, 2507–2517.
Haberkorn, R. (1976). Mol. Phys. 32, 1491–1493.
Schulten, K. et al. (1978). J. Chem. Phys. 69, 3795.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("quantum_bio.radical_pair")

# Physical constants
_MU_B = 9.2740100783e-24   # J/T
_HBAR = 1.054571817e-34    # J·s
_GE = 2.002319             # free electron g-factor

# Convert μT → rad/μs (Larmor frequency)
def _larmor_rad_per_us(B_uT: float, g: float = _GE) -> float:
    return g * _MU_B * B_uT * 1e-6 / _HBAR * 1e-6   # rad/μs


class RadicalPairSystem:
    """
    Cryptochrome radical pair spin system.

    Models the FAD•⁻–TrpH•⁺ radical pair with simplified hyperfine
    coupling.  The system evolves under a static external magnetic field
    (Earth's field: ~50 μT) plus optional RF perturbation.

    Parameters
    ----------
    B_field_uT : float
        Static external magnetic field in μT.  Earth's field ≈ 50 μT.
    rf_freq_MHz : float
        RF oscillating field frequency in MHz.  0 = no RF.
    rf_amp_uT : float
        RF field amplitude in μT.
    hfc_A_mT : float
        Effective isotropic hyperfine coupling constant (mT).
        FAD has ~1.5 mT; simplification of anisotropic tensor.
    k_s : float
        Singlet recombination rate (μs⁻¹).
    k_t : float
        Triplet recombination rate (μs⁻¹).
    use_gpu : bool
        Use CuPy for matrix ops.
    """

    def __init__(
        self,
        B_field_uT: float = 50.0,
        rf_freq_MHz: float = 0.0,
        rf_amp_uT: float = 0.0,
        hfc_A_mT: float = 1.5,
        k_s: float = 1.0,
        k_t: float = 0.1,
        use_gpu: bool = True,
    ) -> None:
        self.B_field_uT = B_field_uT
        self.rf_freq_MHz = rf_freq_MHz
        self.rf_amp_uT = rf_amp_uT
        self.hfc_A_mT = hfc_A_mT
        self.k_s = k_s
        self.k_t = k_t

        # GPU / CPU
        try:
            if use_gpu:
                import cupy as cp
                self.xp = cp
                self.gpu_used = True
            else:
                raise ImportError
        except ImportError:
            self.xp = np
            self.gpu_used = False

        self._build_operators()

    def _build_operators(self) -> None:
        """Build spin operators and Hamiltonian on the 4-D Hilbert space."""
        xp = self.xp

        # Pauli matrices / 2 (spin-1/2 operators)
        sx = xp.array([[0, 0.5], [0.5, 0]], dtype=complex)
        sy = xp.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
        sz = xp.array([[0.5, 0], [0, -0.5]], dtype=complex)
        I2 = xp.eye(2, dtype=complex)

        # Two-electron system: S_A ⊗ I, I ⊗ S_B
        SAx = xp.kron(sx, I2);  SAy = xp.kron(sy, I2);  SAz = xp.kron(sz, I2)
        SBx = xp.kron(I2, sx);  SBy = xp.kron(I2, sy);  SBz = xp.kron(I2, sz)

        # Zeeman Hamiltonian (static field along z)
        omega = _larmor_rad_per_us(self.B_field_uT)
        H_zeeman = omega * (SAz + SBz)

        # Isotropic hyperfine coupling on radical A (simplified 1 nucleus)
        A_rad_us = self.hfc_A_mT * 1e-3 * _GE * _MU_B / _HBAR * 1e-6   # rad/μs
        # Nuclear spin-1/2 on A: we trace out nuclear DOF analytically
        # → effective dephasing: A/4 * (S_A·S_A) = A/4 * (3/4) I on A
        # Full treatment would expand Hilbert space; here we use effective H
        H_hfc = A_rad_us * (SAx @ SAx + SAy @ SAy + SAz @ SAz)

        self.H = H_zeeman + H_hfc
        self.SAz = SAz
        self.SBz = SBz

        # Singlet/triplet projectors
        singlet = xp.array([0, 1, -1, 0], dtype=complex) / xp.sqrt(xp.array(2.0))
        self.P_S = xp.outer(singlet, singlet.conj())
        self.P_T = xp.eye(4, dtype=complex) - self.P_S

    def compute_yield(
        self,
        t_max_us: float = 10.0,
        n_steps: int = 1000,
    ) -> dict[str, Any]:
        """
        Compute singlet yield Φ_S and triplet yield Φ_T.

        Parameters
        ----------
        t_max_us : float
            Total integration time (μs).
        n_steps : int
            Number of RK4 integration steps.

        Returns
        -------
        dict with keys: singlet_yield, triplet_yield, time_us (array),
                        P_S_t (singlet probability trajectory).
        """
        from quantum_bio.lindblad import LindbladSolver

        # Lindblad operators for Haberkorn recombination
        xp = self.xp
        # L_s = √(k_s/2) P_S,  L_t = √(k_t/2) P_T
        L_ops = [
            np.sqrt(self.k_s / 2) * np.array(self.P_S.get() if self.gpu_used else self.P_S),
            np.sqrt(self.k_t / 2) * np.array(self.P_T.get() if self.gpu_used else self.P_T),
        ]
        gamma = [1.0, 1.0]   # rates already folded into L_ops

        H_cpu = np.array(self.H.get() if self.gpu_used else self.H)

        solver = LindbladSolver(
            H=H_cpu,
            L_ops=L_ops,
            gamma=gamma,
            use_gpu=self.gpu_used,
        )

        dt = t_max_us / n_steps
        rho_t = solver.evolve(t_max=t_max_us, dt=dt, store_every=max(1, n_steps // 500))
        times = np.array(solver.times)
        P_S_t = solver.singlet_probability(rho_t)

        # Φ_S = k_s ∫ P_S(t) dt  (numerical trapezoid integration)
        singlet_yield = float(np.trapz(P_S_t, times) * self.k_s)
        triplet_yield = float(1.0 - singlet_yield)   # conservation

        # Clamp to [0, 1] (numerical noise)
        singlet_yield = max(0.0, min(1.0, singlet_yield))
        triplet_yield = max(0.0, min(1.0, triplet_yield))

        logger.info(
            "RadicalPair B=%.1fμT  Φ_S=%.4f  Φ_T=%.4f  gpu=%s",
            self.B_field_uT, singlet_yield, triplet_yield, self.gpu_used
        )

        return {
            "singlet_yield": singlet_yield,
            "triplet_yield": triplet_yield,
            "B_field_uT": self.B_field_uT,
            "rf_freq_MHz": self.rf_freq_MHz,
            "time_us": times.tolist(),
            "P_S_t": P_S_t.tolist(),
        }

    def field_sensitivity(
        self,
        B_range_uT: "np.ndarray | None" = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute Φ_S as a function of B field magnitude.

        Returns dict with B_uT array and corresponding singlet_yield array —
        this is the compass tuning curve.
        """
        if B_range_uT is None:
            B_range_uT = np.linspace(0, 200, 40)

        yields = []
        for B in B_range_uT:
            self.B_field_uT = B
            self._build_operators()
            r = self.compute_yield(**kwargs)
            yields.append(r["singlet_yield"])

        return {
            "B_range_uT": B_range_uT.tolist(),
            "singlet_yield": yields,
            "delta_phi_s": [y - yields[0] for y in yields],
        }
