from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel

from squidpy.experimental.im import fit_stain_reference
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import sda_to_rgb
from squidpy.experimental.im._stain._qc import normalization_consistency, stain_separation_quality
from squidpy.experimental.im._stain._validation import complement_third_column, reorder_to_canonical

_WHITE = np.array([255.0, 255.0, 255.0])
_TRUTH = complement_third_column(
    reorder_to_canonical(np.stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]], axis=1))
)


def _synthetic_rgb(stain_matrix: np.ndarray, *, seed: int, n_side: int = 48, scale: float = 1.0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    # floor keeps every pixel absorbing (tissue) so masks stay non-empty
    # across intensity scalings; anchors define the H/E angular extremes
    conc = rng.uniform(8.0, 70.0, size=(n, 2)) * scale
    third = n // 3
    conc[:third, 1] = 0.0
    conc[third : 2 * third, 0] = 0.0
    od = (conc @ stain_matrix[:, :2].T).T.reshape(3, n_side, n_side)
    return sda_to_rgb(xr.DataArray(od, dims=("c", "y", "x")), _WHITE)


class TestNormalizationConsistency:
    def test_improves_after_normalization(self) -> None:
        from squidpy.experimental.im import apply_stain_normalization

        # three slides with deliberately different staining intensities.
        # threshold=1.0 counts every pixel as tissue (the default 0.8 is tuned
        # for real H&E and reads these synthetic tiles as background).
        raw = [_synthetic_rgb(_TRUTH, seed=i, scale=s) for i, s in enumerate([0.6, 1.0, 1.5])]
        before = normalization_consistency(raw, luminosity_threshold=1.0)["cv"]

        params = {"mask_background": False}
        sdatas = [
            sd.SpatialData(images={"img": Image2DModel.parse(np.asarray(r.data), dims=("c", "y", "x"))}) for r in raw
        ]
        ref = fit_stain_reference(sdatas[1], "img", method="reinhard", method_params=params)
        normalized = [apply_stain_normalization(s, "img", ref, method_params=params) for s in sdatas]
        after = normalization_consistency(normalized, luminosity_threshold=1.0)["cv"]

        assert after < before

    def test_requires_two_images(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            normalization_consistency([_synthetic_rgb(_TRUTH, seed=0)])


class TestStainSeparationQuality:
    def test_clean_separation_low_residual(self) -> None:
        img = _synthetic_rgb(_TRUTH, seed=0)
        sdata = sd.SpatialData(images={"img": Image2DModel.parse(np.asarray(img.data), dims=("c", "y", "x"))})
        ref = fit_stain_reference(sdata, "img", method="macenko", background_intensity=_WHITE)
        out = stain_separation_quality(img, ref)
        assert 0.0 <= out["residual_fraction"] < 0.25
        assert 0.0 < out["he_angle_deg"] < 90.0

    def test_reinhard_reference_rejected(self) -> None:
        img = _synthetic_rgb(_TRUTH, seed=0)
        sdata = sd.SpatialData(images={"img": Image2DModel.parse(np.asarray(img.data), dims=("c", "y", "x"))})
        ref = fit_stain_reference(sdata, "img", method="reinhard")
        with pytest.raises(ValueError, match="decomposition reference"):
            stain_separation_quality(img, ref)
