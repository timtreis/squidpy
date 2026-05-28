"""Visualisation for stain normalization and decomposition."""

from __future__ import annotations

from typing import Literal

import numpy as np
import spatialdata as sd
import xarray as xr

from squidpy.experimental.im import apply_stain_normalization, decompose_stains
from squidpy.experimental.im._stain._reference import StainReference
from squidpy.experimental.im._utils import get_element_data

__all__ = ["stain_separation", "stain_comparison"]


def _display_scale(node: object, scale: str) -> str:
    """Resolve ``scale="auto"`` to the coarsest level for display.

    Plots are materialised for ``imshow``, so rendering the full-resolution
    pyramid level would be needlessly slow; the coarsest level is plenty.
    """
    if scale != "auto" or not hasattr(node, "keys"):
        return scale

    def _idx(k: str) -> int:
        num = "".join(ch for ch in k if ch.isdigit())
        return int(num) if num else -1

    return max(node.keys(), key=_idx)


def _as_rgb_uint8(image: xr.DataArray) -> np.ndarray:
    """``(c, y, x)`` float DataArray -> ``(y, x, c)`` uint8 for imshow."""
    arr = np.clip(np.asarray(image.transpose("y", "x", "c").data), 0, 255).astype(np.uint8)
    return arr


def stain_separation(
    sdata: sd.SpatialData,
    image_key: str,
    reference_or_method: StainReference | Literal["macenko", "vahadane"],
    *,
    scale: str | Literal["auto"] = "auto",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render the hematoxylin and eosin concentration channels side by side.

    Decomposes ``sdata.images[image_key]`` (via
    :func:`~squidpy.experimental.im.decompose_stains`) and shows the H and E
    channels as separate panels. Requires a decomposition reference or method;
    a Reinhard reference has no stain matrix and raises.

    Parameters
    ----------
    sdata, image_key
        SpatialData object and RGB image key.
    reference_or_method
        A decomposition :class:`StainReference` or a method name to fit first.
    scale
        Scale level to decompose.
    figsize
        Matplotlib figure size.
    """
    import matplotlib.pyplot as plt

    display_scale = _display_scale(sdata.images[image_key], scale)
    concentrations = decompose_stains(sdata, image_key, reference_or_method, scale=display_scale)
    concentrations = concentrations.transpose("c", "y", "x")
    hematoxylin = np.asarray(concentrations.isel(c=0).data)
    eosin = np.asarray(concentrations.isel(c=1).data)

    _, axes = plt.subplots(1, 2, figsize=figsize or (8, 4))
    axes[0].imshow(hematoxylin, cmap="Purples")
    axes[0].set_title("hematoxylin")
    axes[1].imshow(eosin, cmap="Reds")
    axes[1].set_title("eosin")
    for ax in axes:
        ax.axis("off")


def stain_comparison(
    sdata: sd.SpatialData,
    image_key: str,
    reference: StainReference,
    *,
    scale: str | Literal["auto"] = "auto",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render the source image and its normalized result side by side.

    Parameters
    ----------
    sdata, image_key
        SpatialData object and RGB image key.
    reference
        A fitted :class:`StainReference` to normalize towards.
    scale
        Scale level to normalize/display.
    figsize
        Matplotlib figure size.
    """
    import matplotlib.pyplot as plt

    display_scale = _display_scale(sdata.images[image_key], scale)
    source = get_element_data(sdata.images[image_key], display_scale, "image", image_key, prefer="coarsest")
    normalized = apply_stain_normalization(sdata, image_key, reference, scale=display_scale)

    _, axes = plt.subplots(1, 2, figsize=figsize or (8, 4))
    axes[0].imshow(_as_rgb_uint8(source))
    axes[0].set_title("before")
    axes[1].imshow(_as_rgb_uint8(normalized))
    axes[1].set_title("after")
    for ax in axes:
        ax.axis("off")
