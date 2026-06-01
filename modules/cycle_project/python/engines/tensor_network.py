"""
quantum_bio.tensor_network — GPU Matrix Product State Engine
============================================================

Matrix Product State (MPS) representation for many-body quantum states.
Used to compress large Hilbert spaces (n > 12 qubits) where dense matrix
methods hit memory limits.

An MPS of n sites with bond dimension χ uses O(n·χ²·d) memory vs O(d^n)
for the full state vector — exponential compression for low-entanglement
states (which quantum biological systems typically are).

GPU acceleration via CuPy BLAS routines:
  · SVD truncation: cupy.linalg.svd (cuSOLVER backend)
  · Tensor contraction: cupy.tensordot / einsum
  · Target: χ ≤ 2500 on RTX 5060 Ti (24 GB VRAM)

References
----------
Vidal (2003). Phys. Rev. Lett. 91, 147902 — TEBD algorithm
Schollwöck (2011). Ann. Phys. 326, 96 — MPS review
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("quantum_bio.tensor_network")


def _get_xp(use_gpu: bool):
    if use_gpu:
        try:
            import cupy as cp
            return cp, True
        except ImportError:
            logger.warning("CuPy unavailable — NumPy fallback")
    return np, False


class MPSEngine:
    """
    Matrix Product State engine.

    Parameters
    ----------
    n_sites : int
        Number of physical sites (qubits / spin-1/2 particles).
    chi : int
        Maximum bond dimension (controls accuracy vs memory).
        RTX 5060 Ti target: chi ≈ 2500 for n ≈ 20 sites.
    phys_dim : int
        Physical dimension per site (2 for qubits).
    use_gpu : bool
        Use CuPy GPU arrays for SVD + contraction.

    Examples
    --------
    Compress a random state vector::

        eng = MPSEngine(n_sites=16, chi=128, use_gpu=True)
        psi = np.random.randn(2**16) + 1j * np.random.randn(2**16)
        psi /= np.linalg.norm(psi)
        result = eng.compress(state_vector=psi)
        print(result["truncation_error"])

    Compute entanglement entropy profile::

        entropy = eng.entanglement_entropy()
    """

    def __init__(
        self,
        n_sites: int = 20,
        chi: int = 64,
        phys_dim: int = 2,
        use_gpu: bool = True,
    ) -> None:
        self.n = n_sites
        self.chi = chi
        self.d = phys_dim
        self.xp, self.gpu_used = _get_xp(use_gpu)

        self.tensors: list[Any] = []   # MPS tensors: list of shape (chi, d, chi)
        self.singular_values: list[Any] = []
        self._truncation_error: float = 0.0

    # ------------------------------------------------------------------
    def compress(
        self,
        state_vector: "np.ndarray | None" = None,
        **kw,
    ) -> dict[str, Any]:
        """
        Compress a state vector |ψ⟩ ∈ ℂ^(d^n) into MPS form.

        If state_vector is None, creates a random low-entanglement state
        (GHZ-like) as a benchmark.

        Returns
        -------
        dict:
            mps_tensors     — list of chi×d×chi arrays (CPU numpy)
            singular_values — list of χ-vectors per bond
            truncation_error — accumulated Frobenius truncation error
            bond_entropies  — entanglement entropy S per bond
            memory_MB       — MPS memory footprint
        """
        xp = self.xp
        d, n, chi = self.d, self.n, self.chi

        if state_vector is None:
            logger.info("No state_vector provided — generating GHZ benchmark state")
            psi = self._ghz_state()
        else:
            psi = xp.array(state_vector, dtype=complex)

        psi = psi / xp.linalg.norm(psi)

        # Right-to-left SVD sweep (standard MPS compression)
        tensors = []
        svs = []
        err = 0.0

        psi_mat = psi.reshape(d, -1)   # d × d^(n-1)

        for site in range(n - 1):
            right_dim = psi_mat.shape[1]
            # SVD: psi_mat = U S V†
            try:
                U, S, Vh = xp.linalg.svd(psi_mat, full_matrices=False)
            except Exception as e:
                logger.error("SVD failed at site %d: %s", site, e)
                break

            # Truncate to chi
            chi_eff = min(chi, len(S))
            err += float(xp.sum(S[chi_eff:] ** 2).real) if len(S) > chi_eff else 0.0

            U_trunc = U[:, :chi_eff]
            S_trunc = S[:chi_eff]
            Vh_trunc = Vh[:chi_eff, :]

            # Store tensor: reshape U to (chi_left, d, chi_right)
            chi_left = U_trunc.shape[0] // d if site > 0 else 1
            A = U_trunc.reshape(chi_left, d, chi_eff)
            tensors.append(A)
            svs.append(S_trunc)

            # Next matrix: S V†
            psi_mat = (xp.diag(S_trunc) @ Vh_trunc).reshape(chi_eff * d, -1)

        # Last tensor
        tensors.append(psi_mat.reshape(-1, d, 1))
        self.tensors = tensors
        self.singular_values = svs
        self._truncation_error = err

        # Bond entropies: S_i = -Σ_α s_α² log(s_α²)
        entropies = []
        for sv in svs:
            sv_np = np.array(sv.get() if self.gpu_used else sv, dtype=float)
            sv_np = sv_np[sv_np > 1e-16]
            p = sv_np ** 2
            p /= p.sum()
            entropies.append(float(-np.sum(p * np.log(p))))

        # Memory footprint
        n_params = sum(t.size for t in tensors)
        memory_MB = n_params * 16 / 1e6   # complex128 = 16 bytes

        logger.info(
            "MPS compression: n=%d, chi=%d, err=%.2e, mem=%.1f MB, gpu=%s",
            n, chi, err, memory_MB, self.gpu_used
        )

        # Convert to CPU numpy for return
        cpu_tensors = [
            np.array(t.get() if self.gpu_used else t) for t in tensors
        ]
        cpu_svs = [
            np.array(s.get() if self.gpu_used else s) for s in svs
        ]

        return {
            "mps_tensors": cpu_tensors,
            "singular_values": cpu_svs,
            "truncation_error": err,
            "bond_entropies": entropies,
            "memory_MB": memory_MB,
            "n_sites": n,
            "chi": chi,
        }

    def entanglement_entropy(self) -> list[float]:
        """Return per-bond entanglement entropy S from last compress() call."""
        entropies = []
        for sv in self.singular_values:
            sv_np = np.array(sv.get() if self.gpu_used else sv, dtype=float)
            sv_np = sv_np[sv_np > 1e-16]
            p = sv_np ** 2 / (sv_np ** 2).sum()
            entropies.append(float(-np.sum(p * np.log(p))))
        return entropies

    def _ghz_state(self) -> "Any":
        """Generate |GHZ⟩ = (|00…0⟩ + |11…1⟩)/√2 as a benchmark."""
        xp = self.xp
        dim = self.d ** self.n
        psi = xp.zeros(dim, dtype=complex)
        psi[0] = 1.0 / xp.sqrt(xp.array(2.0))
        psi[-1] = 1.0 / xp.sqrt(xp.array(2.0))
        return psi

    def benchmark(self, chi_values: "list[int] | None" = None) -> dict[str, Any]:
        """
        Benchmark compression time and truncation error across bond dimensions.

        Returns dict mapping chi → {time_s, truncation_error, memory_MB}.
        """
        import time
        if chi_values is None:
            chi_values = [32, 64, 128, 256, 512]

        psi = self._ghz_state()
        results = {}
        for chi in chi_values:
            self.chi = chi
            t0 = time.perf_counter()
            r = self.compress(state_vector=np.array(psi.get() if self.gpu_used else psi))
            elapsed = time.perf_counter() - t0
            results[chi] = {
                "time_s": round(elapsed, 4),
                "truncation_error": r["truncation_error"],
                "memory_MB": r["memory_MB"],
            }
        return results
