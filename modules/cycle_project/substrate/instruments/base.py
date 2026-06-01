"""
substrate.instruments.base — abstract base for all instruments
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.lab import SubstrateLab


class SubstrateInstrument(ABC):
    """
    Abstract base class for every SUBSTRATE instrument.

    Subclasses must implement execute().  They receive a back-reference
    to the parent SubstrateLab via self._lab (set by the lab on load).

    execute() contract
    ------------------
    Parameters
        task      : str        — which analysis to run
        data_root : Path       — shared processed-data directory
        gpu       : bool       — global GPU flag (respect it; fall back gracefully)
        **kwargs               — task-specific parameters

    Returns
        (data, meta) : (Any, dict)
            data  — primary payload (DataFrame, ndarray, dict, Path to output…)
            meta  — provenance dict (at minimum: {'warnings': []})
    """

    _lab: "SubstrateLab | None" = None   # set by SubstrateLab after instantiation

    @abstractmethod
    def execute(
        self,
        task: str,
        data_root: Path,
        gpu: bool,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__

    def _warn(self, meta: dict, msg: str) -> None:
        """Append a warning to meta['warnings'] (creates key if absent)."""
        meta.setdefault("warnings", []).append(msg)
