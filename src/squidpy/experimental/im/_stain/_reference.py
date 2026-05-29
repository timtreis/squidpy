"""Container for a fitted stain reference.

Holds either a 3x3 stain matrix (Macenko/Vahadane) or a pair of Ruderman Lab
channel statistics (Reinhard), plus optional cohort/provenance metadata and
on-disk persistence (:meth:`StainReference.save`/:meth:`StainReference.load`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from squidpy.experimental.im._stain._constants import STAIN_REFERENCE_SCHEMA_VERSION

StainMethod = Literal["macenko", "vahadane", "reinhard"]
_DECOMPOSITION_METHODS: frozenset[str] = frozenset({"macenko", "vahadane"})
_VALID_METHODS: frozenset[str] = _DECOMPOSITION_METHODS | {"reinhard"}


def _coerce_finite(arr: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


@dataclass(frozen=True, eq=False)
class StainReference:
    """Container for a fitted stain reference.

    Parameters
    ----------
    method
        Fitting method: ``"macenko"``, ``"vahadane"``, or ``"reinhard"``.
    stain_matrix
        Shape ``(3, 3)`` unit-norm matrix in canonical order
        ``(H, E, complement)``. Required for decomposition methods.
    mu
        Shape ``(3,)`` Ruderman Lab channel means. Reinhard only.
    sigma
        Shape ``(3,)`` Ruderman Lab channel standard deviations. Reinhard
        only.
    background_intensity
        Shape ``(3,)`` per-channel white-point estimate. Required for
        decomposition methods (apply consumes it). Forbidden for Reinhard
        because Reinhard's color transfer operates in Ruderman Lab and
        does not model absorbance. There is no universal default; pass an
        estimate from your data (see ``estimate_background_intensity``).
    max_concentrations
        Shape ``(2,)`` reference per-stain (H, E) maximum concentrations.
        Decomposition only. Stored because at apply time the reference image
        is gone, so the target concentration scale must travel with the
        reference. Optional (Reinhard references and externally-built
        decomposition references without it remain valid); forbidden for
        Reinhard.
    version
        On-disk schema version, stamped for :meth:`save`/:meth:`load`.
    cohort_members
        Image keys that contributed to a cohort fit (``None`` for a
        single-image reference).
    per_image_stats
        Per-slide stats and skip/outlier reasons from a cohort fit
        (``None`` for a single-image reference).
    fit_metadata
        Free-form provenance (method, aggregation, parameters, counts).
    """

    method: StainMethod
    stain_matrix: np.ndarray | None = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    background_intensity: np.ndarray | None = None
    max_concentrations: np.ndarray | None = None
    version: int = STAIN_REFERENCE_SCHEMA_VERSION
    cohort_members: tuple[str, ...] | None = None
    per_image_stats: dict[str, Any] | None = None
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        # The numpy-array fields make the dataclass-generated __eq__ raise
        # ("truth value of an array is ambiguous"), so compare explicitly.
        # Provenance metadata (per_image_stats/fit_metadata) is excluded:
        # two references are equal when their fitted content matches.
        if not isinstance(other, StainReference):
            return NotImplemented
        if (self.method, self.version, self.cohort_members) != (other.method, other.version, other.cohort_members):
            return False
        return all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("stain_matrix", "mu", "sigma", "background_intensity", "max_concentrations")
        )

    # eq=False keeps the default identity-based __hash__ (the array fields are
    # unhashable, so a value-based hash is impossible); references remain usable
    # as set members / dict keys by identity.
    __hash__ = object.__hash__

    def save(self, path: str | Path) -> None:
        """Serialise this reference to JSON at ``path``."""
        from squidpy.experimental.im._stain._persistence import save_reference

        save_reference(self, path)

    @classmethod
    def load(cls, path: str | Path) -> StainReference:
        """Load a reference previously written by :meth:`save`."""
        from squidpy.experimental.im._stain._persistence import load_reference

        return load_reference(path)

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected one of {sorted(_VALID_METHODS)}.")

        if self.method in _DECOMPOSITION_METHODS:
            if self.stain_matrix is None:
                raise ValueError(f"method={self.method!r} requires stain_matrix.")
            if self.mu is not None or self.sigma is not None:
                raise ValueError(f"method={self.method!r} forbids mu/sigma; pass them only for Reinhard.")
            if self.background_intensity is None:
                raise ValueError(f"method={self.method!r} requires background_intensity.")
            object.__setattr__(
                self,
                "stain_matrix",
                _coerce_finite(self.stain_matrix, shape=(3, 3), name="stain_matrix"),
            )
            bg = _coerce_finite(self.background_intensity, shape=(3,), name="background_intensity")
            if np.any(bg <= 0):
                raise ValueError("background_intensity must be strictly positive.")
            object.__setattr__(self, "background_intensity", bg)
            if self.max_concentrations is not None:
                maxc = _coerce_finite(self.max_concentrations, shape=(2,), name="max_concentrations")
                if np.any(maxc <= 0):
                    raise ValueError("max_concentrations must be strictly positive.")
                object.__setattr__(self, "max_concentrations", maxc)
        else:
            if self.mu is None or self.sigma is None:
                raise ValueError("method='reinhard' requires both mu and sigma.")
            if self.stain_matrix is not None:
                raise ValueError("method='reinhard' forbids stain_matrix.")
            if self.background_intensity is not None:
                raise ValueError(
                    "method='reinhard' forbids background_intensity; Reinhard's color "
                    "transfer is in Ruderman Lab and does not use a white point."
                )
            if self.max_concentrations is not None:
                raise ValueError("method='reinhard' forbids max_concentrations.")
            mu = _coerce_finite(self.mu, shape=(3,), name="mu")
            sigma = _coerce_finite(self.sigma, shape=(3,), name="sigma")
            if np.any(sigma <= 0):
                raise ValueError("sigma must be strictly positive.")
            object.__setattr__(self, "mu", mu)
            object.__setattr__(self, "sigma", sigma)
