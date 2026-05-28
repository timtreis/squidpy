"""Cohort stain-reference fitting.

Fits a per-slide reference for every requested image, drops slides that fail
to fit (or, optionally, that are outliers), and aggregates the survivors into
one robust reference. Stain matrices are not linearly averageable (sign/order
ambiguity, non-Euclidean), so decomposition members are canonical-aligned and
combined by a per-column angular (component-wise) median, then re-normalised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import spatialdata as sd

from squidpy.experimental.im._stain._reference import StainMethod, StainReference
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    complement_third_column,
    reorder_to_canonical,
    validate_stain_matrix,
)

_VALID_METHODS = ("reinhard", "macenko", "vahadane")
_DECOMPOSITION_METHODS = ("macenko", "vahadane")


def _unit_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _consensus_he(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-column angular (component-wise median) consensus H and E directions."""
    stack = np.stack(matrices)  # (M, 3, 3)
    h = np.median(stack[:, :, 0], axis=0)
    e = np.median(stack[:, :, 1], axis=0)
    return h / np.linalg.norm(h), e / np.linalg.norm(e)


def _reject_outliers(
    fits: dict[str, StainReference],
    method: StainMethod,
    *,
    max_angle_deg: float,
    z_threshold: float,
) -> tuple[dict[str, StainReference], dict[str, str]]:
    keys = list(fits)
    kept: dict[str, StainReference] = {}
    dropped: dict[str, str] = {}

    if method in _DECOMPOSITION_METHODS:
        h_c, e_c = _consensus_he([fits[k].stain_matrix for k in keys])
        for k in keys:
            w = fits[k].stain_matrix
            dev = max(_unit_angle_deg(w[:, 0], h_c), _unit_angle_deg(w[:, 1], e_c))
            if dev > max_angle_deg:
                dropped[k] = f"stain vectors deviate {dev:.1f} deg from cohort consensus (max {max_angle_deg})."
            else:
                kept[k] = fits[k]
    else:
        mus = np.stack([fits[k].mu for k in keys])
        center = np.median(mus, axis=0)
        dist = np.linalg.norm(mus - center, axis=1)
        mad = float(np.median(np.abs(dist - np.median(dist)))) or 1.0
        scores = 0.6745 * (dist - np.median(dist)) / mad  # modified z-score
        for k, score in zip(keys, scores, strict=True):
            if abs(score) > z_threshold:
                dropped[k] = f"mu is a statistical outlier (modified z={score:.1f} > {z_threshold})."
            else:
                kept[k] = fits[k]
    return kept, dropped


def _aggregate(fits: dict[str, StainReference], method: StainMethod) -> tuple[StainReference, str]:
    refs = list(fits.values())
    if method in _DECOMPOSITION_METHODS:
        h_c, e_c = _consensus_he([r.stain_matrix for r in refs])
        matrix = complement_third_column(reorder_to_canonical(np.stack([h_c, e_c], axis=1)))
        validate_stain_matrix(matrix)
        background = np.median(np.stack([r.background_intensity for r in refs]), axis=0)
        max_conc_members = [r.max_concentrations for r in refs if r.max_concentrations is not None]
        max_conc = np.median(np.stack(max_conc_members), axis=0) if len(max_conc_members) == len(refs) else None
        reference = StainReference(
            method=method,
            stain_matrix=matrix,
            background_intensity=background,
            max_concentrations=max_conc,
        )
        return reference, "angular_median"

    mu = np.mean(np.stack([r.mu for r in refs]), axis=0)
    sigma = np.sqrt(np.mean(np.stack([r.sigma**2 for r in refs]), axis=0))
    return StainReference(method="reinhard", mu=mu, sigma=sigma), "mean_mu_pooled_sigma"


def _member_stat(reference: StainReference) -> dict[str, Any]:
    if reference.method in _DECOMPOSITION_METHODS:
        return {"stain_matrix": reference.stain_matrix, "background_intensity": reference.background_intensity}
    return {"mu": reference.mu, "sigma": reference.sigma}


def fit_cohort_reference(
    sdata: sd.SpatialData,
    *,
    image_keys: Sequence[str] | None = None,
    method: StainMethod = "macenko",
    scale: str | Literal["auto"] = "auto",
    method_params: Any = None,
    reject_outliers: bool = False,
    outlier_max_angle_deg: float = 20.0,
    outlier_z_threshold: float = 3.5,
    min_members: int = 2,
) -> StainReference:
    """Fit one robust stain reference across several images in a SpatialData object.

    Parameters
    ----------
    sdata
        SpatialData object containing the cohort images.
    image_keys
        Keys in ``sdata.images`` to fit on. ``None`` (default) uses every
        image in the object.
    method
        Fitting method, applied to every slide (one method per cohort).
    scale, method_params
        Passed through to the per-slide fit (see
        :func:`~squidpy.experimental.im.fit_stain_reference`).
    reject_outliers
        If ``True``, drop members far from the cohort consensus before
        aggregating (decomposition: angular deviation; Reinhard: modified
        z-score on ``mu``). Off by default.
    outlier_max_angle_deg, outlier_z_threshold
        Thresholds for the respective outlier criteria.
    min_members
        Minimum surviving members; raise if fewer fit successfully.

    Returns
    -------
    A :class:`StainReference` aggregated across the surviving slides, with
    ``cohort_members``, ``per_image_stats`` (kept members, skipped/dropped
    reasons by key), and ``fit_metadata`` populated.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(_VALID_METHODS)}.")
    keys = list(sdata.images) if image_keys is None else list(image_keys)
    if not keys:
        raise ValueError("no images to fit a cohort reference from.")

    from squidpy.experimental.im._stain._normalize import fit_stain_reference

    fits: dict[str, StainReference] = {}
    skipped: dict[str, str] = {}
    for key in keys:
        try:
            fits[key] = fit_stain_reference(sdata, key, method=method, scale=scale, method_params=method_params)
        except StainFittingError as error:
            skipped[key] = error.reason

    dropped: dict[str, str] = {}
    if reject_outliers and len(fits) >= 3:
        fits, dropped = _reject_outliers(
            fits, method, max_angle_deg=outlier_max_angle_deg, z_threshold=outlier_z_threshold
        )

    if len(fits) < min_members:
        raise StainFittingError(
            f"cohort fit kept only {len(fits)} member(s) (min_members={min_members}); "
            f"skipped={skipped}, dropped_outliers={dropped}."
        )

    reference, aggregation = _aggregate(fits, method)
    object.__setattr__(reference, "cohort_members", tuple(sorted(fits)))
    object.__setattr__(
        reference,
        "per_image_stats",
        {"members": {k: _member_stat(r) for k, r in fits.items()}, "skipped": skipped, "dropped_outliers": dropped},
    )
    object.__setattr__(
        reference,
        "fit_metadata",
        {
            "method": method,
            "aggregation": aggregation,
            "scale": scale,
            "n_members": len(fits),
            "n_skipped": len(skipped),
            "n_dropped": len(dropped),
            "reject_outliers": reject_outliers,
        },
    )
    return reference
