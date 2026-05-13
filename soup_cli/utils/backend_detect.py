"""v0.53.9 #28 — Auto-detect inference backend from a model directory.

Reads `config.json` `architectures` field; falls back to `transformers`
on missing config, decode errors, or non-LLM arches. Returns a closed-set
backend identifier so callers can dispatch deterministically.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Optional

from soup_cli.utils.paths import is_under_cwd

# Closed allowlist — matches `soup serve --backend`.
SUPPORTED_BACKENDS = frozenset({"transformers", "vllm", "sglang", "mlx"})

# Architecture-name keyword -> preferred backend. First match wins; the
# scan walks `architectures` left-to-right.
_MLX_KEYWORDS = frozenset({"mlx", "mlx_lm"})


def _read_config_json(model_dir: Path) -> Optional[dict]:
    """Read `<model_dir>/config.json` with TOCTOU-safe symlink rejection.

    Returns None on any error (missing dir, symlinked config, decode
    failure). Never raises — caller treats None as "unknown arch".
    """
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        # Reject symlinked config.json (mirrors v0.53.1 #82 policy).
        st = os.lstat(config_path)
        if stat.S_ISLNK(st.st_mode):
            return None
    except OSError:
        return None
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _has_mlx_weights(model_dir: Path) -> bool:
    """Return True if the directory has MLX-format `weights.npz` (mlx-lm convention).

    Each entry is `os.lstat`-checked so a symlinked `weights.npz` does not
    accidentally trigger MLX dispatch (mirrors v0.33.0 #22 TOCTOU policy).
    """
    try:
        for path in model_dir.iterdir():
            if path.name.lower() != "weights.npz":
                continue
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode):
                continue
            if stat.S_ISREG(st.st_mode):
                return True
    except OSError:
        return False
    return False


def detect_backend(model_path: str, *, env: Optional[dict] = None) -> str:
    """Probe `model_path` and return a preferred backend identifier.

    Resolution order:
      1. Env hint `SOUP_BENCH_BACKEND` (validated against allowlist).
      2. MLX weight files in the model dir.
      3. `architectures` field in `config.json` (always implies transformers
         today; vllm / sglang need explicit selection because they wrap
         the same HF weights).
      4. Fallback: `transformers`.

    `model_path` must be a non-empty str; on traversal / NUL / missing dir
    the helper falls back to `transformers` silently.
    """
    if not isinstance(model_path, str) or not model_path:
        return "transformers"
    if "\x00" in model_path:
        return "transformers"

    env = env if env is not None else os.environ
    hint = env.get("SOUP_BENCH_BACKEND")
    if isinstance(hint, str) and hint:
        canonical = hint.strip().lower()
        if canonical in SUPPORTED_BACKENDS:
            return canonical

    # Containment guard — only probe paths inside cwd. Out-of-cwd paths
    # are still benched; we just skip the local-file probe.
    if not is_under_cwd(model_path):
        return "transformers"
    real = os.path.realpath(model_path)
    model_dir = Path(real)
    if not model_dir.is_dir():
        return "transformers"

    if _has_mlx_weights(model_dir):
        return "mlx"

    config = _read_config_json(model_dir)
    if config:
        # MLX configs sometimes carry `mlx`-prefixed model_type.
        model_type = str(config.get("model_type") or "").lower()
        if any(kw in model_type for kw in _MLX_KEYWORDS):
            return "mlx"

    return "transformers"
