"""JSON persistence for :class:`StainReference`.

Plain JSON (no binary blobs) so a saved reference is human-readable and
diffable. Numpy arrays anywhere in the structure are encoded as a tagged
``{"__ndarray__": ...}`` object and decoded back to exact dtype/shape. A
top-level ``version`` gates loading so future schema changes fail loudly
rather than silently mis-parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from squidpy.experimental.im._stain._constants import STAIN_REFERENCE_SCHEMA_VERSION
from squidpy.experimental.im._stain._reference import StainReference

_ARRAY_TAG = "__ndarray__"
# Fields persisted verbatim (after array encoding); method/version handled explicitly.
_PERSISTED_FIELDS = (
    "stain_matrix",
    "mu",
    "sigma",
    "background_intensity",
    "max_concentrations",
    "cohort_members",
    "per_image_stats",
    "fit_metadata",
)


def _encode(obj: Any) -> Any:
    """Recursively make ``obj`` JSON-serialisable, tagging numpy arrays."""
    if isinstance(obj, np.ndarray):
        return {_ARRAY_TAG: obj.tolist(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    """Inverse of :func:`_encode`: rebuild numpy arrays from tagged dicts."""
    if isinstance(obj, dict):
        if _ARRAY_TAG in obj:
            return np.asarray(obj[_ARRAY_TAG], dtype=obj["dtype"]).reshape(obj["shape"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def to_json(reference: StainReference) -> dict[str, Any]:
    """Return a JSON-serialisable dict describing ``reference``."""
    payload: dict[str, Any] = {"version": reference.version, "method": reference.method}
    for name in _PERSISTED_FIELDS:
        value = getattr(reference, name)
        if value is None or (isinstance(value, dict) and not value):
            continue
        payload[name] = _encode(value)
    return payload


def from_json(payload: dict[str, Any]) -> StainReference:
    """Reconstruct a :class:`StainReference` from a :func:`to_json` dict."""
    if not isinstance(payload, dict) or "method" not in payload or "version" not in payload:
        raise ValueError("not a valid StainReference document (missing 'method'/'version').")
    version = payload["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version > STAIN_REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported StainReference schema version {version!r}; "
            f"this squidpy supports integer versions up to {STAIN_REFERENCE_SCHEMA_VERSION}."
        )

    kwargs: dict[str, Any] = {"method": payload["method"], "version": version}
    for name in _PERSISTED_FIELDS:
        if name not in payload:
            continue
        decoded = _decode(payload[name])
        if name == "cohort_members":
            decoded = tuple(decoded)
        kwargs[name] = decoded
    return StainReference(**kwargs)


def save_reference(reference: StainReference, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_json(reference), indent=2))


def load_reference(path: str | Path) -> StainReference:
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    return from_json(payload)
