from __future__ import annotations

import json

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import RUIFROK_HE, STAIN_REFERENCE_SCHEMA_VERSION
from squidpy.experimental.im._stain._reference import StainReference
from squidpy.experimental.im._stain._validation import complement_third_column, reorder_to_canonical

_WHITE = np.array([255.0, 255.0, 255.0])


def _decomposition_ref(method: str = "macenko") -> StainReference:
    w = complement_third_column(
        reorder_to_canonical(np.stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]], axis=1))
    )
    return StainReference(
        method=method,
        stain_matrix=w,
        background_intensity=_WHITE,
        max_concentrations=np.array([33.0, 44.0]),
    )


def _reinhard_ref() -> StainReference:
    return StainReference(method="reinhard", mu=np.array([1.0, 2.0, 3.0]), sigma=np.array([0.5, 0.6, 0.7]))


def _assert_array_equal_or_none(a, b) -> None:
    if a is None:
        assert b is None
    else:
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape


@pytest.mark.parametrize("ref_factory", [_reinhard_ref, _decomposition_ref, lambda: _decomposition_ref("vahadane")])
def test_round_trip_array_exact(tmp_path, ref_factory) -> None:
    ref = ref_factory()
    path = tmp_path / "ref.json"
    ref.save(path)
    loaded = StainReference.load(path)

    assert loaded.method == ref.method
    assert loaded.version == ref.version
    for name in ("stain_matrix", "mu", "sigma", "background_intensity", "max_concentrations"):
        _assert_array_equal_or_none(getattr(ref, name), getattr(loaded, name))


def test_cohort_fields_round_trip(tmp_path) -> None:
    ref = _decomposition_ref()
    object.__setattr__(ref, "cohort_members", ("a", "b", "c"))
    object.__setattr__(ref, "per_image_stats", {"skipped": {"d": "empty mask"}, "a": {"matrix": ref.stain_matrix}})
    object.__setattr__(ref, "fit_metadata", {"aggregation": "angular_median", "n_members": 3})

    path = tmp_path / "cohort.json"
    ref.save(path)
    loaded = StainReference.load(path)

    assert loaded.cohort_members == ("a", "b", "c")
    assert loaded.fit_metadata["n_members"] == 3
    assert loaded.per_image_stats["skipped"] == {"d": "empty mask"}
    np.testing.assert_array_equal(loaded.per_image_stats["a"]["matrix"], ref.stain_matrix)


def test_version_written(tmp_path) -> None:
    path = tmp_path / "ref.json"
    _reinhard_ref().save(path)
    assert json.loads(path.read_text())["version"] == STAIN_REFERENCE_SCHEMA_VERSION


def test_future_version_raises(tmp_path) -> None:
    path = tmp_path / "ref.json"
    payload = {"version": STAIN_REFERENCE_SCHEMA_VERSION + 1, "method": "reinhard"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported StainReference schema version"):
        StainReference.load(path)


def test_corrupted_json_raises(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        StainReference.load(path)


def test_missing_keys_raises(tmp_path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"method": "reinhard"}))
    with pytest.raises(ValueError, match="missing 'method'/'version'"):
        StainReference.load(path)


def test_string_version_raises(tmp_path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"version": "1", "method": "reinhard"}))
    with pytest.raises(ValueError, match="unsupported StainReference schema version"):
        StainReference.load(path)
