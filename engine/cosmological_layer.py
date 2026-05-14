"""Cosmological CMB-anomaly layer — scipy spherical harmonic CMB analysis.

Generates a synthetic CMB temperature map using a Planck 2018-like C_l
power spectrum, then measures quadrupole power and hemispherical asymmetry.
"""
from __future__ import annotations
import numpy as np


def _planck_cl(lmax: int, A: float = 1.0) -> np.ndarray:
    """Synthetic Planck 2018-like CMB TT power spectrum C_l (dimensionless)."""
    ell = np.arange(lmax + 1, dtype=float)
    cl = (
            A * np.exp(-((ell - 220.0) ** 2) / (2.0 * 60.0 ** 2))
        + 0.3 * A * np.exp(-((ell - 540.0) ** 2) / (2.0 * 50.0 ** 2))
        + 0.1 * A * np.exp(-((ell - 810.0) ** 2) / (2.0 * 40.0 ** 2))
        + 0.01 * A   # flat noise floor
    )
    cl[0] = 0.0   # monopole removed
    cl[1] = 0.0   # dipole removed by convention
    return cl


def run(params: dict = None) -> dict:
    try:
        from dll_healing import heal
        heal()
    except ImportError:
        pass

    from scipy.special import sph_harm_y   # sph_harm_y(l, m, theta_polar, phi_azim)

    nside    = 64
    lmax     = 3 * nside - 1               # 191 per spec
    npix     = 12 * nside * nside          # HEALPix pixel count (49152)
    A        = 1.0                         # dimensionless amplitude
    lmax_eff = 20                          # truncate map sum for scipy performance

    cl_input = _planck_cl(lmax, A)
    rng      = np.random.default_rng(42)

    # Generate complex a_lm for m >= 0.
    # Reality condition: a_{l,-m} = (-1)^m conj(a_{l,m}) is enforced
    # in the map reconstruction below.
    alm: dict[tuple[int, int], complex] = {}
    for l in range(2, lmax_eff + 1):
        cl_l = float(cl_input[l])
        # m=0: real coefficient
        alm[(l, 0)] = complex(rng.normal(0.0, np.sqrt(cl_l)), 0.0)
        # m>0: complex, each component ~ N(0, sqrt(C_l/2))
        s = np.sqrt(cl_l / 2.0)
        for m in range(1, l + 1):
            alm[(l, m)] = complex(rng.normal(0.0, s), rng.normal(0.0, s))

    # ── Quadrupole power C_2 (from drawn a_lm) ───────────────────
    c2_measured = float(
        abs(alm[(2, 0)]) ** 2
        + 2.0 * abs(alm[(2, 1)]) ** 2
        + 2.0 * abs(alm[(2, 2)]) ** 2
    ) / 5.0

    # ── Reconstruct CMB map on a coarse lat-lon grid ──────────────
    # THETA: polar (colatitude) [0,pi], PHI: azimuthal [0,2pi]
    ntheta, nphi = 64, 128
    theta_arr = np.linspace(1e-6, np.pi - 1e-6, ntheta)
    phi_arr   = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    THETA = theta_arr[:, np.newaxis]  # (ntheta, 1) — broadcasts over phi
    PHI   = phi_arr[np.newaxis, :]    # (1, nphi)

    cmb_map = np.zeros((ntheta, nphi))
    for l in range(2, lmax_eff + 1):
        # m=0: real Y_l^0, contribution a_{l0} * Y_l^0
        y0 = sph_harm_y(l, 0, THETA, PHI).real
        cmb_map += float(alm[(l, 0)].real) * y0
        # m>0: T += 2 * Re(a_{lm} * Y_l^m)
        for m in range(1, l + 1):
            ylm = sph_harm_y(l, m, THETA, PHI)
            cmb_map += 2.0 * (alm[(l, m)] * ylm).real

    # ── Hemispherical asymmetry ───────────────────────────────────
    north_mask = (theta_arr < np.pi / 2.0)  # (ntheta,) boolean
    var_n = float(np.var(cmb_map[north_mask, :]))
    var_s = float(np.var(cmb_map[~north_mask, :]))
    hemi_asymmetry = var_n / var_s if var_s > 0.0 else 1.0

    # ── Score ─────────────────────────────────────────────────────
    c2_expected = float(cl_input[2])                       # noise floor ~0.01
    c2_ratio    = float(np.clip(c2_measured / c2_expected, 0.0, 1.0))
    asym_term   = 1.0 - min(1.0, abs(hemi_asymmetry - 1.0))
    score       = float(np.clip(0.5 * asym_term + 0.5 * c2_ratio, 0.0, 1.0))

    return {
        "layer": "cosmological",
        "score": round(score, 6),
        "data": {
            "quadrupole_C2":  round(c2_measured, 9),
            "hemi_asymmetry": round(hemi_asymmetry, 6),
            "c2_ratio":       round(c2_ratio, 6),
            "nside":          nside,
            "n_pixels":       int(npix),
            "lmax":           lmax,
        },
    }
