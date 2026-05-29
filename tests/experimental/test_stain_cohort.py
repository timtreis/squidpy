from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel

import squidpy as sq
from squidpy.experimental.im._stain._cohort import fit_cohort_reference
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import sda_to_rgb
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    complement_third_column,
    reorder_to_canonical,
)

_WHITE = np.array([255.0, 255.0, 255.0])


def _canonical(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return complement_third_column(reorder_to_canonical(np.stack([h, e], axis=1)))


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(u @ v)) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _synthetic_rgb(stain_matrix: np.ndarray, *, seed: int, n_side: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    conc = rng.uniform(0.0, 70.0, size=(n, 2))
    third = n // 3
    conc[:third, 1] = 0.0
    conc[third : 2 * third, 0] = 0.0
    od = (conc @ stain_matrix[:, :2].T).T.reshape(3, n_side, n_side)
    return np.asarray(sda_to_rgb(xr.DataArray(od, dims=("c", "y", "x")), _WHITE).data)


def _sdata(images: dict[str, np.ndarray]) -> sd.SpatialData:
    return sd.SpatialData(images={k: Image2DModel.parse(v, dims=("c", "y", "x")) for k, v in images.items()})


_TRUTH = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])


def _good_cohort() -> sd.SpatialData:
    return _sdata({f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(3)})


class TestCohortAggregation:
    def test_recovers_consensus_matrix(self) -> None:
        ref = fit_cohort_reference(_good_cohort(), method="macenko")
        assert ref.cohort_members == ("s0", "s1", "s2")
        assert _angle_deg(ref.stain_matrix[:, 0], _TRUTH[:, 0]) < 12.0
        assert _angle_deg(ref.stain_matrix[:, 1], _TRUTH[:, 1]) < 12.0
        assert ref.fit_metadata["aggregation"] == "componentwise_median"
        assert ref.fit_metadata["n_members"] == 3

    def test_image_keys_subset(self) -> None:
        ref = fit_cohort_reference(_good_cohort(), method="macenko", image_keys=["s0", "s1"])
        assert ref.cohort_members == ("s0", "s1")

    def test_reinhard_cohort(self) -> None:
        ref = fit_cohort_reference(_good_cohort(), method="reinhard")
        assert ref.method == "reinhard"
        assert ref.mu.shape == (3,)
        assert ref.fit_metadata["aggregation"] == "mean_mu_pooled_sigma"


class TestSkippedAndOutliers:
    def test_unfittable_slide_skipped(self) -> None:
        images = {f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(2)}
        images["blank"] = np.full((3, 32, 32), 255.0)  # empty tissue -> StainFittingError
        ref = fit_cohort_reference(_sdata(images), method="macenko")
        assert "blank" not in ref.cohort_members
        assert "blank" in ref.per_image_stats["skipped"]
        assert ref.fit_metadata["n_skipped"] == 1

    def test_outlier_dropped_when_enabled(self) -> None:
        e_bad = RUIFROK_HE["eosin"] + 0.7 * RUIFROK_HE["hematoxylin"]
        bad = _canonical(RUIFROK_HE["hematoxylin"], e_bad / np.linalg.norm(e_bad))
        images = {f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(3)}
        images["outlier"] = _synthetic_rgb(bad, seed=99)

        ref = fit_cohort_reference(_sdata(images), method="macenko", reject_outliers=True, outlier_max_angle_deg=10.0)
        assert "outlier" not in ref.cohort_members
        assert "outlier" in ref.per_image_stats["dropped_outliers"]

    def test_outlier_kept_by_default(self) -> None:
        e_bad = RUIFROK_HE["eosin"] + 0.7 * RUIFROK_HE["hematoxylin"]
        bad = _canonical(RUIFROK_HE["hematoxylin"], e_bad / np.linalg.norm(e_bad))
        images = {f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(3)}
        images["outlier"] = _synthetic_rgb(bad, seed=99)

        ref = fit_cohort_reference(_sdata(images), method="macenko")
        assert "outlier" in ref.cohort_members
        assert ref.per_image_stats["dropped_outliers"] == {}


class TestGuards:
    def test_too_few_members_raises(self) -> None:
        images = {"good": _synthetic_rgb(_TRUTH, seed=0), "blank": np.full((3, 32, 32), 255.0)}
        with pytest.raises(StainFittingError, match="kept only 1 member"):
            fit_cohort_reference(_sdata(images), method="macenko", min_members=2)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            fit_cohort_reference(_good_cohort(), method="bogus")

    def test_round_trips_through_save_load(self, tmp_path) -> None:
        from squidpy.experimental.im._stain._reference import StainReference

        ref = fit_cohort_reference(_good_cohort(), method="macenko")
        path = tmp_path / "cohort.json"
        ref.save(path)
        loaded = StainReference.load(path)
        np.testing.assert_array_equal(loaded.stain_matrix, ref.stain_matrix)
        assert loaded.cohort_members == ref.cohort_members
        assert loaded.fit_metadata["n_members"] == 3


class TestCohortOnHnE:
    def test_two_image_cohort_smoke(self, sdata_hne) -> None:
        image_key = next(iter(sdata_hne.images))
        # add a deterministically re-stained copy as a second cohort member
        from squidpy.experimental.im._utils import get_element_data

        da = get_element_data(sdata_hne.images[image_key], "auto", "image", image_key).astype("float32")
        weights = xr.DataArray([1.15, 1.0, 0.9], dims="c", coords={"c": da.coords["c"]})
        sdata_hne.images["hne_b"] = Image2DModel.parse((da * weights).clip(0, 255).data, dims=da.dims)

        # vahadane (not macenko): the Visium H&E is low-contrast and macenko's
        # angular fit is borderline on it (macenko correctness is covered by
        # the synthetic-recovery tests).
        ref = sq.experimental.im.fit_cohort_reference(sdata_hne, image_keys=[image_key, "hne_b"], method="vahadane")
        assert ref.method == "vahadane"
        assert len(ref.cohort_members) == 2


def test_malformed_slide_skipped_not_aborted() -> None:
    # a non-3-channel slide raises ValueError in the per-slide fit; it must be
    # recorded and skipped, not abort the whole cohort.
    images = {f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(2)}
    rgba = np.random.default_rng(7).integers(20, 120, (4, 32, 32)).astype("uint8")
    sdata = sd.SpatialData(
        images={
            **{k: Image2DModel.parse(v, dims=("c", "y", "x")) for k, v in images.items()},
            "rgba": Image2DModel.parse(rgba, dims=("c", "y", "x")),
        }
    )
    ref = fit_cohort_reference(sdata, method="macenko")
    assert "rgba" not in ref.cohort_members
    assert "rgba" in ref.per_image_stats["skipped"]


def test_reject_outliers_warns_below_three_members() -> None:
    images = {f"s{i}": _synthetic_rgb(_TRUTH, seed=i) for i in range(2)}
    with pytest.warns(UserWarning, match="outlier rejection needs >=3"):
        fit_cohort_reference(_sdata(images), method="macenko", reject_outliers=True)
