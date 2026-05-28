from __future__ import annotations

import xarray as xr
from spatialdata.models import Image2DModel

import squidpy as sq
from squidpy.experimental.im._utils import get_element_data
from tests.conftest import PlotTester, PlotTesterMeta


def _restained_copy(sdata_hne, image_key: str, key_added: str, weights: list[float]) -> None:
    da = get_element_data(sdata_hne.images[image_key], "auto", "image", image_key).astype("float32")
    w = xr.DataArray(weights, dims="c", coords={"c": da.coords["c"]})
    sdata_hne.images[key_added] = Image2DModel.parse((da * w).clip(0, 255).data, dims=da.dims)


class TestStainPlotting(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_stain_separation(self, sdata_hne) -> None:
        """Visual: hematoxylin and eosin concentration channels."""
        image_key = next(iter(sdata_hne.images))
        ref = sq.experimental.im.fit_stain_reference(sdata_hne, image_key, method="macenko")
        sq.experimental.pl.stain_separation(sdata_hne, image_key, ref)

    def test_plot_stain_comparison(self, sdata_hne) -> None:
        """Visual: before/after normalization to a re-stained reference."""
        image_key = next(iter(sdata_hne.images))
        # fit the reference on a re-stained copy so before/after differ visibly
        _restained_copy(sdata_hne, image_key, "hne_target", [0.7, 1.0, 1.4])
        ref = sq.experimental.im.fit_stain_reference(sdata_hne, "hne_target", method="reinhard")
        sq.experimental.pl.stain_comparison(sdata_hne, image_key, ref)
