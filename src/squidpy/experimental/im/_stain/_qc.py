"""Quality-control metrics for stain normalization.

Primitive layer: numeric diagnostics on images / references, no plotting and
no ``sdata``. Two questions a stain pipeline must be able to answer:

- did normalizing a set of slides actually align their colours?
  (:func:`normalization_consistency`)
- how cleanly did a decomposition separate H from E?
  (:func:`stain_separation_quality`)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from squidpy.experimental.im._stain._constants import DEFAULT_LUMINOSITY_THRESHOLD
from squidpy.experimental.im._stain._conversion import _check_channel_dim
from squidpy.experimental.im._stain._decomposition import decompose_to_concentrations
from squidpy.experimental.im._stain._mask import (
    absorbance_foreground_mask,
    luminosity_foreground_mask,
)
from squidpy.experimental.im._stain._reference import StainReference
from squidpy.experimental.im._stain._validation import angle_between_deg


def normalization_consistency(
    images: Sequence[xr.DataArray],
    *,
    luminosity_threshold: float = DEFAULT_LUMINOSITY_THRESHOLD,
) -> dict[str, Any]:
    """Cross-image colour consistency via the Normalized Median Intensity (NMI).

    For each image, NMI is the median over tissue pixels of the per-pixel
    mean channel intensity (Basavanhally & Madabhushi, 2013). A well-normalized
    cohort has near-identical NMI across slides, so the coefficient of
    variation of NMI (``cv = std / mean``) is the consistency score: lower is
    better. Computing it on the images before vs. after normalization shows
    whether normalization helped.

    Parameters
    ----------
    images
        Images to compare, each with a length-3 ``"c"`` dimension. Pass a
        manageable scale level; the median is evaluated eagerly per image.
    luminosity_threshold
        Tissue cutoff for :func:`luminosity_foreground_mask`.

    Returns
    -------
    ``{"nmi": (n,) array, "cv": float, "n_images": int}``.
    """
    if len(images) < 2:
        raise ValueError("normalization_consistency needs at least 2 images to compare.")
    nmi = np.empty(len(images), dtype=np.float64)
    for i, image in enumerate(images):
        _check_channel_dim(image)
        mean_intensity = image.mean(dim="c")
        mask = luminosity_foreground_mask(image, luminosity_threshold)
        values = np.asarray(mean_intensity.where(mask).values, dtype=np.float64)
        if not np.any(np.isfinite(values)):
            raise ValueError(
                f"image at index {i} has no tissue pixels under luminosity_threshold={luminosity_threshold}; "
                "raise the threshold or drop the blank image."
            )
        nmi[i] = float(np.nanmedian(values))
    mean = float(nmi.mean())
    cv = float(nmi.std() / mean) if mean != 0 else float("nan")
    return {"nmi": nmi, "cv": cv, "n_images": len(images)}


def stain_separation_quality(image: xr.DataArray, reference: StainReference) -> dict[str, Any]:
    """Diagnostics of how cleanly a decomposition separates H from E.

    Parameters
    ----------
    image
        RGB image with a length-3 ``"c"`` dimension.
    reference
        A decomposition (macenko/vahadane) :class:`StainReference`. Reinhard
        references have no stain matrix and are rejected.

    Returns
    -------
    ``{"residual_fraction": float, "he_angle_deg": float}`` where
    ``residual_fraction`` is the share of tissue absorbance assigned to the
    complement (residual) channel rather than H or E - lower means cleaner
    separation - and ``he_angle_deg`` is the angle between the H and E stain
    vectors (larger = more distinct stains).
    """
    _check_channel_dim(image)
    if reference.method == "reinhard" or reference.stain_matrix is None:
        raise ValueError("stain_separation_quality requires a decomposition reference with a stain matrix.")

    concentrations = decompose_to_concentrations(image, reference.stain_matrix, reference.background_intensity)
    mask = absorbance_foreground_mask(image, reference.background_intensity)
    magnitude = np.abs(concentrations)
    residual_sum, total_sum = (
        magnitude.isel(c=2).where(mask).sum(),
        magnitude.sum(dim="c").where(mask).sum(),
    )
    residual_value, total_value = (float(residual_sum.compute()), float(total_sum.compute()))
    if total_value == 0:
        raise ValueError(
            "no tissue absorbance under the foreground mask; the image may be blank or the white point wrong."
        )
    residual_fraction = residual_value / total_value

    he_angle_deg = angle_between_deg(reference.stain_matrix[:, 0], reference.stain_matrix[:, 1])
    return {"residual_fraction": residual_fraction, "he_angle_deg": he_angle_deg}
