"""Live proxy training for ``soup data mix --live`` (v0.53.5 #116).

Lifts the v0.48.0 Part B synthetic-only proxy: ``proxy_run_for_weights``
renders a temp ``soup.yaml`` that splices the candidate weights into a base
recipe, invokes ``python -m soup_cli.cli train --config <tmp> --yes`` via a
list-argv ``subprocess.run`` (no shell), and reads the final ``eval_loss``
back from the SQLite tracker.

The synthetic proxy in ``commands/data_mix.py`` remains the default; this
module is wired through the new ``--live`` flag with mandatory
``--base-yaml`` containment-checked path.

Security:
- ``base_yaml_path`` must stay under cwd (``utils.paths.is_under_cwd``).
- Per-candidate temp dir lives under ``tempfile.gettempdir()``; cleaned up
  via ``try/finally`` (matches v0.43.0 Part D ``copy_bundle_to`` atomic-write
  policy).
- Subprocess invocation uses ``sys.executable + ['-m', 'soup_cli.cli', ...]``
  (no shell, list argv — matches v0.33.0 #34 / v0.40.4 ``execvp`` pattern).
- ``timeout_seconds`` capped to ``[60, 30*60]``; ``TimeoutExpired`` raises
  ``RuntimeError`` so callers can isolate per-candidate (v0.48.0 Part B
  ``run_mix_optimizer`` swallow-policy applies).
- Weights validated: simplex (sum = 1 ± 1e-6), finite, no bool, [0, 1].
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Optional, Sequence, Tuple

_MIN_TIMEOUT_S = 60
_MAX_TIMEOUT_S = 30 * 60
_MAX_DATASETS = 32
_MAX_PATH_LEN = 4096
_FLOAT_TOL = 1e-6

__all__ = [
    "proxy_run_for_weights",
]


def _validate_path_under_cwd(name: str, raw: object) -> str:
    if not isinstance(raw, str):
        raise TypeError(f"{name} must be str, got {type(raw).__name__}")
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in raw:
        raise ValueError(f"{name} must not contain null bytes")
    if len(raw) > _MAX_PATH_LEN:
        raise ValueError(
            f"{name} length {len(raw)} exceeds cap {_MAX_PATH_LEN}"
        )
    try:
        st = os.lstat(raw)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{name} not found: {os.path.basename(raw)!r}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"{name} is not stat-able: {os.path.basename(raw)!r}"
        ) from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"{name} is a symlink (rejected for safety): "
            f"{os.path.basename(raw)!r}"
        )
    from soup_cli.utils.paths import is_under_cwd  # noqa: PLC0415

    real = os.path.realpath(raw)
    if not is_under_cwd(real):
        raise ValueError(f"{name} is outside cwd: {os.path.basename(real)!r}")
    if not os.path.isfile(real):
        raise FileNotFoundError(
            f"{name} is not a regular file: {os.path.basename(real)!r}"
        )
    return real


def _validate_weights(
    weights: Sequence[float], num_datasets: int
) -> Tuple[float, ...]:
    if not isinstance(weights, (list, tuple)):
        raise TypeError(
            f"weights must be list/tuple, got {type(weights).__name__}"
        )
    if len(weights) != num_datasets:
        raise ValueError(
            f"weights length {len(weights)} != datasets {num_datasets}"
        )
    floats: list = []
    for value in weights:
        if isinstance(value, bool):
            raise ValueError("weight must be float, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"weight must be float, got {type(value).__name__}"
            )
        fv = float(value)
        if not math.isfinite(fv):
            raise ValueError(f"weight must be finite (got {value!r})")
        if fv < 0.0 or fv > 1.0:
            raise ValueError(f"weight must be in [0, 1] (got {fv})")
        floats.append(fv)
    total = sum(floats)
    if abs(total - 1.0) > _FLOAT_TOL:
        raise ValueError(
            f"weights must sum to 1.0 ± {_FLOAT_TOL} (got {total})"
        )
    return tuple(floats)


def _validate_datasets(datasets: Sequence[str]) -> Tuple[str, ...]:
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(
            f"datasets must be list/tuple, got {type(datasets).__name__}"
        )
    if len(datasets) < 2:
        raise ValueError(
            f"datasets must contain at least 2 entries (got {len(datasets)})"
        )
    if len(datasets) > _MAX_DATASETS:
        raise ValueError(
            f"datasets has {len(datasets)} entries; cap is {_MAX_DATASETS}"
        )
    for entry in datasets:
        if not isinstance(entry, str):
            raise TypeError(
                f"dataset must be str, got {type(entry).__name__}"
            )
        if not entry:
            raise ValueError("dataset path must be non-empty")
        if "\x00" in entry:
            raise ValueError("dataset path must not contain null bytes")
        if "\n" in entry:
            raise ValueError("dataset path must not contain newlines")
        if len(entry) > _MAX_PATH_LEN:
            raise ValueError(
                f"dataset path length {len(entry)} exceeds {_MAX_PATH_LEN}"
            )
    return tuple(datasets)


def _validate_timeout(timeout_seconds: object) -> int:
    if isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be int, not bool")
    if not isinstance(timeout_seconds, int):
        raise TypeError(
            "timeout_seconds must be int, got "
            f"{type(timeout_seconds).__name__}"
        )
    if timeout_seconds < _MIN_TIMEOUT_S or timeout_seconds > _MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_seconds must be in [{_MIN_TIMEOUT_S}, "
            f"{_MAX_TIMEOUT_S}], got {timeout_seconds}"
        )
    return timeout_seconds


def _render_overlay_yaml(
    base_yaml_text: str,
    datasets: Tuple[str, ...],
    weights: Tuple[float, ...],
) -> str:
    """Splice ``data.interleave`` + ``data.train`` into a base YAML.

    We round-trip via ``yaml.safe_load`` / ``yaml.safe_dump`` to defeat any
    accidental key collisions; weights + datasets are sanitised above so the
    output cannot inject keys (the renderer emits scalars only).
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(base_yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("base YAML must be a top-level mapping")
    data_block = raw.get("data")
    if not isinstance(data_block, dict):
        data_block = {}
        raw["data"] = data_block
    data_block["train"] = list(datasets)
    data_block["interleave"] = {
        "strategy": "probs",
        "probs": [float(w) for w in weights],
    }
    return yaml.safe_dump(raw, sort_keys=False)


def _read_final_eval_loss(
    tracker_db_path: Optional[str], started_at: float
) -> float:
    """Read the most-recent finished run's final eval_loss / final_loss.

    Looks for runs created at or after ``started_at`` (epoch seconds) so a
    concurrent unrelated training run cannot poison the proxy reading.
    """
    from soup_cli.experiment.tracker import ExperimentTracker  # noqa: PLC0415

    if tracker_db_path is not None:
        os.environ["SOUP_DB_PATH"] = tracker_db_path
    tracker = ExperimentTracker()
    rows = tracker.list_runs(limit=10)
    for row in rows:
        # finish_run sets `status='completed'` + `final_loss`.
        status = row.get("status")
        final_loss = row.get("final_loss")
        if status != "completed" or final_loss is None:
            continue
        if not isinstance(final_loss, (int, float)):
            continue
        fv = float(final_loss)
        if not math.isfinite(fv):
            continue
        return fv
    raise RuntimeError(
        "no completed run found in tracker — proxy training failed"
    )


def proxy_run_for_weights(
    weights: Sequence[float],
    datasets: Sequence[str],
    base_yaml_path: str,
    *,
    timeout_seconds: int = 5 * 60,
    tracker_db_path: Optional[str] = None,
) -> float:
    """Run one short proxy training and return the observed eval loss.

    Args:
        weights: Per-dataset interleave probabilities (simplex).
        datasets: Dataset paths in the same order as ``weights``.
        base_yaml_path: A real ``soup.yaml`` under cwd that supplies all the
            non-data fields (base, task, training, output). The proxy
            overlays only ``data.train`` + ``data.interleave``.
        timeout_seconds: Hard cap on the subprocess (60s..30m).
        tracker_db_path: Optional override of the SQLite DB location (used
            in tests). Honoured via ``SOUP_DB_PATH``.

    Returns:
        The observed ``final_loss`` from the most recent completed run.

    Raises:
        TypeError / ValueError: Input validation failures.
        RuntimeError: Training subprocess failed or no completed run found.
    """
    import time  # noqa: PLC0415

    ds = _validate_datasets(datasets)
    w = _validate_weights(weights, len(ds))
    yaml_real = _validate_path_under_cwd("base_yaml_path", base_yaml_path)
    timeout = _validate_timeout(timeout_seconds)

    with open(yaml_real, "r", encoding="utf-8") as fh:
        base_text = fh.read()
    if len(base_text) > 256 * 1024:
        raise ValueError(
            f"base YAML exceeds 256KB cap (got {len(base_text)} bytes)"
        )
    overlay = _render_overlay_yaml(base_text, ds, w)

    tmp_dir = tempfile.mkdtemp(prefix=".soup_mix_proxy.")
    try:
        tmp_yaml = os.path.join(tmp_dir, "soup.yaml")
        with open(tmp_yaml, "w", encoding="utf-8") as fh:
            fh.write(overlay)
        argv = [
            sys.executable,
            "-m",
            "soup_cli.cli",
            "train",
            "--config",
            tmp_yaml,
            "--yes",
        ]
        env = os.environ.copy()
        if tracker_db_path is not None:
            env["SOUP_DB_PATH"] = tracker_db_path
        started_at = time.time()
        try:
            result = subprocess.run(  # noqa: S603 — argv list, no shell.
                argv,
                env=env,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"proxy training exceeded {timeout}s timeout"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "proxy training subprocess failed "
                f"(rc={result.returncode})"
            )
        return _read_final_eval_loss(tracker_db_path, started_at)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
