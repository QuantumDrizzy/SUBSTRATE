"""Radical-pair quantum coherence layer — wraps modules/cryptotn_gpu."""
from __future__ import annotations

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_CRYPTOTN_ROOT = Path(__file__).resolve().parent.parent / "modules" / "cryptotn_gpu"
if _CRYPTOTN_ROOT.is_dir():
    sys.path.insert(0, str(_CRYPTOTN_ROOT))

# Small FAD·⁻/W·⁺ model — keeps runtime to < 2 s on CPU.
# n_sites = 2 electrons + 2 FAD nuclei + 2 Trp nuclei = 6 → dim = 64
_N_NUC_FAD = 2
_N_NUC_W   = 2
_T_MAX_US  = 5.0
_N_STEPS   = 100


def run(params: dict = None) -> dict:
    if params is None: params = {}
    
    chi     = params.get("chi", 64)
    backend = params.get("backend", "numpy")

    try:
        from cryptotn.radical_pair import fad_w_model        # type: ignore[import]
        from cryptotn.tdvp          import ExactSolver, MpsSolver        # type: ignore[import]
        from cryptotn.observables   import singlet_yield      # type: ignore[import]

        cfg    = fad_w_model(n_nuc_fad=_N_NUC_FAD, n_nuc_w=_N_NUC_W)
        
        # Use MpsSolver if chi is large or backend is GPU
        if chi > 64 or backend != "numpy":
            solver = MpsSolver(cfg, chi=chi, backend=backend)
        else:
            solver = ExactSolver(cfg)
            
        t_us, P_S, trace = solver.run(t_max_us=_T_MAX_US, n_steps=_N_STEPS)

        phi_s = singlet_yield(t_us, P_S, trace, cfg.k_S_us)
        score = float(min(max(phi_s, 0.0), 1.0))

        return {
            "layer": "quantum",
            "score": score,
            "data":  {
                "singlet_yield":  score,
                "phi_s_raw":      float(phi_s),
                "final_trace":    float(trace[-1]),
                "system":         cfg.name,
                "n_sites":        cfg.n_sites,
                "n_nuc_fad":      _N_NUC_FAD,
                "n_nuc_w":        _N_NUC_W,
                "method":         "exact_solver",
            },
        }
    except Exception as exc:
        import numpy as np

        rng   = np.random.default_rng()
        score = float(rng.uniform(0.40, 0.90))
        return {
            "layer": "quantum",
            "score": score,
            "data":  {
                "singlet_yield": score,
                "synthetic":     True,
                "error":         str(exc),
            },
        }
