"""
substrate.instruments.simulation — Lithospheric Simulation Instrument
======================================================================

Wraps src/pole_shift_sim/ (Lattice Boltzmann D3Q19 lithosphere sim).

Tasks
-----
  run_lbm         Run a full Lattice Boltzmann simulation step sequence.
                  Returns vtk output path + scalar displacement metrics.

  visualize       Render a PyVista animation from an existing LBM output.
                  Returns path to .mp4 / .vtp animation file.

  benchmark       Benchmark the LBM kernel on the available GPU.
                  Returns MLUPS (mega lattice updates per second).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.simulation")


class SimulationInstrument(SubstrateInstrument):
    """
    Lithospheric Pole-Shift Simulation Instrument.

    Runs a 3-D Lattice Boltzmann (D3Q19 BGK) simulation of lithospheric
    slab displacement driven by asymmetric ice-load torque, with U(1)
    gauge-field coupling for asthenosphere viscosity.

    Grid : 512 × 256 × 32 spherical lat-lon cells
    GPU  : CuPy CUDA arrays (sm_120) — ~10× speedup over NumPy CPU
    Val  : Steinberger & O'Connell 1997 asthenosphere flow benchmark
    """

    def execute(
        self,
        task: str = "run_lbm",
        data_root: Path = Path("data/processed"),
        gpu: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": []}

        if task == "run_lbm":
            return self._run_lbm(data_root, gpu, meta, **kwargs)
        elif task == "visualize":
            return self._visualize(data_root, meta, **kwargs)
        elif task == "benchmark":
            return self._benchmark(gpu, meta, **kwargs)
        else:
            raise ValueError(f"SimulationInstrument: unknown task '{task}'")

    def _run_lbm(self, data_root: Path, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        n_steps: int = kw.pop("n_steps", 1000)
        save_every: int = kw.pop("save_every", 100)
        meta.update({"n_steps": n_steps, "save_every": save_every})

        try:
            import cupy  # noqa: F401
            backend = "cupy" if gpu else "numpy"
        except ImportError:
            backend = "numpy"
            self._warn(meta, "CuPy not found — using NumPy backend (slow)")
        meta["backend"] = backend

        try:
            from pole_shift_sim.lbm_lithosphere import run_lbm_simulation
            result = run_lbm_simulation(
                out_dir=data_root / "lbm",
                n_steps=n_steps,
                save_every=save_every,
                backend=backend,
                **kw,
            )
            return result, meta
        except ImportError:
            self._warn(meta, "pole_shift_sim not importable — returning stub")
            stub = {
                "vtp_path": str(data_root / "lbm" / "output.vtp"),
                "max_displacement_km": None,
                "note": "STUB — CuPy + pole_shift_sim required",
            }
            return stub, meta

    def _visualize(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        vtp_path = kw.pop("vtp_path", str(data_root / "lbm" / "output.vtp"))
        try:
            from pole_shift_sim.visualize import render_animation
            out = render_animation(vtp_path=vtp_path, **kw)
            return {"animation_path": str(out)}, meta
        except ImportError:
            self._warn(meta, "pole_shift_sim.visualize not importable — PyVista required")
            return {"animation_path": None, "note": "STUB"}, meta

    def _benchmark(self, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            from pole_shift_sim.lbm_lithosphere import benchmark_kernel
            mlups = benchmark_kernel(gpu=gpu, **kw)
            meta["mlups"] = mlups
            return {"mlups": mlups}, meta
        except ImportError:
            self._warn(meta, "pole_shift_sim not importable")
            return {"mlups": None, "note": "STUB"}, meta
