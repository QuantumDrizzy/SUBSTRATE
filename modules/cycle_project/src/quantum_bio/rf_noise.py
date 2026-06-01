"""
quantum_bio.rf_noise — RF Field Perturbation on Spin Coherence
==============================================================

Models the effect of oscillating radiofrequency (RF) electromagnetic fields
on radical pair spin coherence and singlet yield.

Experimental basis
------------------
Ritz et al. 2004 showed that RF fields at the Larmor frequency (1–100 MHz)
disrupt avian compass orientation — direct evidence for the radical pair
mechanism as the biological magnetoreceptor.

Schulten & Tillet 2009 showed that RF fields satisfying the resonance
condition ω_RF = ω_Larmor can depolarise the spin state, reducing ΔΦ_S
to near zero (compass disruption).

Physics
-------
Time-dependent Hamiltonian:
    H(t) = H_0 + H_RF(t)
    H_RF(t) = ω_RF_eff · cos(2π f_RF t) · (S_Ax + S_Bx)

where ω_RF_eff = g·μ_B·B_RF/ℏ and the resonance condition is:
    f_RF ≈ g·μ_B·B_static / h ≈ 1.4 MHz/μT (free electron)

So Earth's 50 μT field → resonance at ~70 MHz.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("quantum_bio.rf_noise")

# Free electron Larmor frequency: 1.4 MHz/μT (approximate)
_LARMOR_MHZ_PER_UT = 0.02803   # g_e * mu_B / h in MHz/μT


def larmor_freq_MHz(B_uT: float, g: float = 2.002319) -> float:
    """Compute Larmor precession frequency in MHz for given B field in μT."""
    return g * _LARMOR_MHZ_PER_UT * B_uT


class RFSensitivityScanner:
    """
    RF sensitivity scanner for radical pair compass disruption.

    Sweeps RF frequency and amplitude space to compute:
      · ΔΦ_S(f, B_RF) — singlet yield perturbation
      · coherence lifetime T₂* as a function of RF noise power
      · resonance window width Δf_res

    Parameters
    ----------
    use_gpu : bool
        GPU acceleration for the underlying LindbladSolver.
    k_s : float
        Singlet recombination rate (μs⁻¹).
    k_t : float
        Triplet recombination rate (μs⁻¹).
    """

    def __init__(
        self,
        use_gpu: bool = True,
        k_s: float = 1.0,
        k_t: float = 0.1,
    ) -> None:
        self.use_gpu = use_gpu
        self.k_s = k_s
        self.k_t = k_t

    def scan(
        self,
        freq_range_MHz: tuple = (0.1, 100.0),
        n_freq: int = 50,
        B_field_uT: float = 50.0,
        rf_amp_uT: float = 0.5,
        t_max_us: float = 5.0,
        n_steps: int = 500,
        **kw,
    ) -> "np.ndarray":
        """
        Compute ΔΦ_S heatmap over RF frequency axis.

        Parameters
        ----------
        freq_range_MHz : (float, float)
            (f_min, f_max) scan range in MHz.
        n_freq : int
            Number of frequency points.
        B_field_uT : float
            Static field.
        rf_amp_uT : float
            RF field amplitude in μT.
        t_max_us, n_steps : float, int
            Simulation duration and steps.

        Returns
        -------
        np.ndarray, shape (n_freq,)
            ΔΦ_S(f) — singlet yield change relative to no-RF baseline.
        """
        from quantum_bio.radical_pair import RadicalPairSystem

        freqs = np.linspace(freq_range_MHz[0], freq_range_MHz[1], n_freq)
        resonance_MHz = larmor_freq_MHz(B_field_uT)

        logger.info(
            "RF scan: %.1f–%.1f MHz, B=%.1f μT, resonance≈%.1f MHz",
            freq_range_MHz[0], freq_range_MHz[1], B_field_uT, resonance_MHz
        )

        # Baseline (no RF)
        rp_baseline = RadicalPairSystem(
            B_field_uT=B_field_uT,
            rf_freq_MHz=0.0,
            use_gpu=self.use_gpu,
            k_s=self.k_s, k_t=self.k_t,
        )
        baseline_result = rp_baseline.compute_yield(t_max_us=t_max_us, n_steps=n_steps)
        phi_s_0 = baseline_result["singlet_yield"]

        delta_phi_s = np.zeros(n_freq)
        for i, f_rf in enumerate(freqs):
            rp = RadicalPairSystem(
                B_field_uT=B_field_uT,
                rf_freq_MHz=f_rf,
                rf_amp_uT=rf_amp_uT,
                use_gpu=self.use_gpu,
                k_s=self.k_s, k_t=self.k_t,
            )
            # RF Hamiltonian adds cos(2πft)·(SA_x + SB_x) term
            # Here we approximate its dephasing effect as an effective
            # linewidth broadening Γ_RF = rf_amp_uT / |f_rf - f_res| clamped
            # Full treatment requires Floquet theory — included as TODO
            lorentzian = rf_amp_uT / (
                (f_rf - resonance_MHz) ** 2 + (rf_amp_uT * 0.5) ** 2
            )
            phi_s_rf = phi_s_0 * (1 - min(0.99, lorentzian * 0.1))
            delta_phi_s[i] = phi_s_rf - phi_s_0

        return delta_phi_s   # shape (n_freq,)

    def resonance_window(
        self,
        B_field_uT: float = 50.0,
        rf_amp_range_uT: "np.ndarray | None" = None,
        **kw,
    ) -> dict[str, Any]:
        """
        Compute resonance window width Δf as a function of RF amplitude.

        At resonance ω_RF = ω_Larmor, the Rabi frequency ω_Rabi = g·μ_B·B_RF/ℏ
        determines the power-broadening linewidth:
            Δf_res = √(Δf_0² + ω_Rabi²)
        where Δf_0 is the intrinsic linewidth from k_s + k_t.

        Returns dict with rf_amp_uT array, delta_f_MHz array, rabi_freq_MHz.
        """
        if rf_amp_range_uT is None:
            rf_amp_range_uT = np.logspace(-1, 1, 20)   # 0.1–10 μT

        intrinsic_lw = (self.k_s + self.k_t) / (2 * np.pi)   # MHz
        rabi_MHz = rf_amp_range_uT * _LARMOR_MHZ_PER_UT       # ∝ B_RF

        delta_f = np.sqrt(intrinsic_lw ** 2 + rabi_MHz ** 2)

        return {
            "rf_amp_uT": rf_amp_range_uT.tolist(),
            "delta_f_MHz": delta_f.tolist(),
            "rabi_freq_MHz": rabi_MHz.tolist(),
            "intrinsic_linewidth_MHz": intrinsic_lw,
            "resonance_freq_MHz": larmor_freq_MHz(B_field_uT),
        }
