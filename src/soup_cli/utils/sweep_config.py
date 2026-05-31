"""v0.44.0 Part D — `soup sweep --config sweep.yaml` separate-file loader.

Schema for a standalone sweep YAML so it can be version-controlled
independently from the training config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import yaml

from soup_cli.utils.paths import is_under_cwd

_VALID_STRATEGIES = frozenset({"grid", "random"})
_MAX_PARAM_KEYS = 32
_MAX_VALUES_PER_KEY = 64
_MAX_FILE_BYTES = 256 * 1024


# Allowed value types for sweep params — strict scalar allowlist matching the
# project's "no untrusted YAML through to runtime" stance.
_VALID_VALUE_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class SweepSpec:
    """Parsed sweep config — fully immutable.

    `params` is a `MappingProxyType` of `tuple` values so callers cannot
    mutate the spec post-construction (matches the v0.43.0 `Tournament`
    pattern for shared-state safety).
    """

    strategy: str
    n_runs: int
    seed: int
    params: Mapping[str, Tuple[Any, ...]]


def _validate_param_key(key: str) -> str:
    if not isinstance(key, str):
        raise TypeError("sweep param key must be str")
    if not key or "\x00" in key:
        raise ValueError("sweep param key must be non-empty + NUL-free")
    if len(key) > 128:
        raise ValueError("sweep param key exceeds 128 chars")
    return key


def parse_sweep_yaml(text: str) -> SweepSpec:
    """Parse a sweep YAML payload."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if "\x00" in text:
        raise ValueError("sweep yaml contains NUL byte")
    if len(text.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError(f"sweep yaml exceeds {_MAX_FILE_BYTES} bytes")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("sweep yaml must be a mapping at the top level")
    strategy = data.get("strategy", "grid")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_VALID_STRATEGIES)}; "
            f"got {strategy!r}"
        )
    n_runs = data.get("n_runs", 0)
    if isinstance(n_runs, bool) or not isinstance(n_runs, int):
        raise TypeError("n_runs must be int")
    if not (0 <= n_runs <= 10000):
        raise ValueError("n_runs must be in [0, 10000]")
    seed = data.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    if not (0 <= seed < 2**31):
        raise ValueError("seed must be in [0, 2**31)")
    raw_params = data.get("params") or {}
    if not isinstance(raw_params, dict):
        raise ValueError("params must be a mapping of name -> list-of-values")
    if len(raw_params) > _MAX_PARAM_KEYS:
        raise ValueError(f"too many param keys (>{_MAX_PARAM_KEYS})")
    params: dict[str, Tuple[Any, ...]] = {}
    for raw_key, raw_values in raw_params.items():
        key = _validate_param_key(raw_key)
        if not isinstance(raw_values, list):
            raise ValueError(f"params[{key}] must be a list")
        if len(raw_values) > _MAX_VALUES_PER_KEY:
            raise ValueError(
                f"params[{key}] exceeds {_MAX_VALUES_PER_KEY} values"
            )
        if not raw_values:
            raise ValueError(f"params[{key}] must be non-empty")
        for value in raw_values:
            if not isinstance(value, _VALID_VALUE_TYPES):
                raise ValueError(
                    f"params[{key}] contains a non-scalar value "
                    f"(expected str/int/float/bool, got {type(value).__name__})"
                )
        params[key] = tuple(raw_values)
    return SweepSpec(
        strategy=strategy,
        n_runs=n_runs,
        seed=seed,
        params=MappingProxyType(params),
    )


def load_sweep_yaml(path: str) -> SweepSpec:
    """Read + parse a sweep YAML file under cwd containment."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be non-empty str")
    if "\x00" in path:
        raise ValueError("path contains NUL byte")
    if not is_under_cwd(path):
        raise ValueError(
            f"sweep config is outside cwd: {os.path.basename(path)}"
        )
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise FileNotFoundError(f"sweep config not found: {os.path.basename(real)}")
    with open(real, "rb") as fh:
        raw_bytes = fh.read(_MAX_FILE_BYTES + 1)
    if len(raw_bytes) > _MAX_FILE_BYTES:
        raise ValueError(f"sweep yaml exceeds {_MAX_FILE_BYTES} bytes")
    return parse_sweep_yaml(raw_bytes.decode("utf-8-sig"))
