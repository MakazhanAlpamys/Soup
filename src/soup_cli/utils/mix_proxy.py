"""Live proxy training for ``soup data mix --live`` (v0.53.5 #116).

Lifts the v0.48.0 Part B synthetic-only proxy: ``proxy_run_for_weights``
renders a temp ``soup.yaml`` that splices the candidate weights into a base
recipe, invokes ``python -m soup_cli.cli train --config <tmp> --yes`` via a
list-argv ``subprocess.run`` (no shell), and reads that candidate's measured
final-step validation loss from the SQLite tracker. Training loss and other
runs' measurements are never substitutes for a missing evaluation.

The synthetic proxy in ``commands/data_mix.py`` remains the default; this
module is wired through the new ``--live`` flag with mandatory
``--base-yaml`` containment-checked path.

Security:
- ``base_yaml_path`` must stay under cwd (``utils.paths.is_under_cwd``).
- Each candidate has its own run ID, output directory and (by default) database
  under a temporary directory in cwd. Temporary artifacts are cleaned up after
  reading the score; the base recipe's output is never overwritten.
- Subprocess invocation uses ``sys.executable + ['-m', 'soup_cli.cli', ...]``
  (no shell, list argv — matches v0.33.0 #34 / v0.40.4 ``execvp`` pattern).
- ``timeout_seconds`` capped to ``[60, 30*60]``; ``TimeoutExpired`` raises
  ``RuntimeError`` so callers can isolate per-candidate (v0.48.0 Part B
  ``run_mix_optimizer`` swallow-policy applies).
- Weights validated: simplex (sum = 1 ± 1e-6), finite, no bool, [0, 1].
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Sequence, Tuple

_MIN_TIMEOUT_S = 60
_MAX_TIMEOUT_S = 30 * 60
_MAX_DATASETS = 32
_MAX_PATH_LEN = 4096
_FLOAT_TOL = 1e-6

__all__ = [
    "proxy_run_for_weights",
    "validate_proxy_config",
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
    *,
    output_dir: Optional[str] = None,
) -> str:
    """Splice ``data.interleave`` + ``data.train`` into a base YAML.

    We round-trip via ``load_yaml_limited`` (``yaml.safe_load`` under the shared
    size, nesting and expansion limits) and ``yaml.safe_dump`` to defeat any
    accidental key collisions; weights + datasets are sanitised above so the
    output cannot inject keys (the renderer emits scalars only).

    ``data.train`` renders as the full dataset list, index-aligned with
    ``data.interleave.probs`` — #443 wired ``data.interleave`` into
    ``load_dataset()``, so ``soup train`` can now consume the real
    N-dataset mixture this candidate searched, rather than the
    single-highest-weighted-dataset collapse #330/#442 used as a stopgap
    while there was no training-time reader.
    """
    import yaml  # noqa: PLC0415

    from soup_cli.utils.yaml_limits import (  # noqa: PLC0415
        MAX_CONFIG_BYTES,
        load_yaml_limited,
    )

    raw = load_yaml_limited(base_yaml_text, "base YAML", max_bytes=MAX_CONFIG_BYTES)
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
    if output_dir is not None:
        raw["output"] = output_dir
        # SFT appends experiment_name to output. A proxy must not inherit a
        # name that redirects its artifact outside the isolated candidate path.
        raw["experiment_name"] = None
    dumped = yaml.safe_dump(raw, sort_keys=False)

    # Insert an explanatory comment directly above the `train:` line inside
    # the `data:` block — a targeted line insert, not a value edit, so it
    # can't touch anything the schema will parse. The comment names the
    # exact list `data.train` renders as, so an anti-drift test can tie the
    # comment's claim to the actual emitted value rather than to a
    # separately-tracked variable (the #442 lesson).
    lines = dumped.split("\n")
    data_idx = next((i for i, ln in enumerate(lines) if ln == "data:"), -1)
    if data_idx >= 0:
        for i in range(data_idx + 1, len(lines)):
            ln = lines[i]
            if ln and not ln.startswith(" "):
                break  # left the data: block
            if ln.startswith("  train:"):
                rounded_probs = [round(float(w), 6) for w in weights]
                lines.insert(
                    i,
                    f"  # data.interleave=probs {rounded_probs} over "
                    f"{list(datasets)!r} — see #443",
                )
                break
        dumped = "\n".join(lines)
    return dumped


def _validate_proxy_text(base_text: str) -> None:
    """Require the loader/trainer path with a comparable validation objective."""
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(base_text)
    if cfg.task != "sft" or cfg.backend != "transformers" or cfg.modality != "text":
        raise ValueError("live mix requires task=sft, backend=transformers and modality=text")
    if cfg.data.streaming:
        raise ValueError("live mix requires data.streaming=false for fixed per-source validation")
    if cfg.data.format == "pre_tokenized":
        raise ValueError(
            "live mix requires raw datasets; pre_tokenized caches bypass mixture loading"
        )
    if not math.isfinite(cfg.data.val_split) or cfg.data.val_split <= 0:
        raise ValueError("live mix requires data.val_split > 0 to measure validation loss")


def _read_base_text(base_yaml_path: str) -> str:
    yaml_real = _validate_path_under_cwd("base_yaml_path", base_yaml_path)
    with open(yaml_real, "r", encoding="utf-8") as handle:
        base_text = handle.read(256 * 1024 + 1)
    if len(base_text) > 256 * 1024:
        raise ValueError("base YAML exceeds 256KB cap")
    return base_text


def validate_proxy_config(base_yaml_path: str) -> None:
    """Reject unsupported live objectives before any training is launched.

    Eager local probability mixtures hold validation rows out of each source
    before applying weights. Other tasks/backends and streaming mixtures need
    their own measured-objective contracts before live optimization can use them.
    """
    _validate_proxy_text(_read_base_text(base_yaml_path))


def _read_final_eval_loss(tracker_db_path: str, run_id: str, expected_output: str) -> float:
    """Read only a completed candidate's explicit final-step validation loss."""
    from soup_cli.experiment.tracker import ExperimentTracker  # noqa: PLC0415

    if not os.path.isfile(tracker_db_path):
        raise RuntimeError("proxy training did not create its candidate tracker")
    tracker = ExperimentTracker(db_path=Path(tracker_db_path))
    try:
        run = tracker.get_run(run_id)
        # get_run accepts unique prefixes for interactive use; evidence may not.
        if not run or run.get("run_id") != run_id or run.get("status") != "completed":
            raise RuntimeError("proxy candidate has no exact completed run in tracker")
        try:
            config = json.loads(run.get("config_json", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("proxy candidate has invalid recorded configuration") from exc
        output = os.path.realpath(expected_output)
        declared = config.get("output") if isinstance(config, dict) else None
        for path in (declared, run.get("output_dir")):
            if not isinstance(path, str) or not path or os.path.realpath(path) != output:
                raise RuntimeError("proxy candidate output does not match its recorded run")
        final_step = run.get("total_steps")
        if isinstance(final_step, bool) or not isinstance(final_step, int) or final_step <= 0:
            raise RuntimeError("proxy candidate has no completed optimizer steps")
        evaluations = [row for row in tracker.get_metrics(run_id) if row.get("kind") == "eval"]
        if not evaluations:
            raise RuntimeError("proxy candidate has no measured validation loss")
        # get_metrics sorts by step, not arrival. A later invalid evaluation must
        # not fall back to an older finite one, including one at a higher step.
        latest = max(evaluations, key=lambda row: row["id"])
        if latest.get("step") != final_step:
            raise RuntimeError(
                "proxy candidate has no evaluation at its final optimizer step; "
                "remove training.eval_steps to evaluate each epoch"
            )
        value = latest.get("val_loss")
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            raise RuntimeError("proxy candidate's final validation loss is not a finite loss")
        return float(value)
    finally:
        tracker.close()


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
            model and training settings. Text SFT on transformers with eager
            data loading and a positive validation split is supported. Output
            and experiment_name are isolated for each temporary candidate.
        timeout_seconds: Hard cap on the subprocess (60s..30m).
        tracker_db_path: Optional override of the SQLite DB location (used
            in tests). Passed only to the child; the default is a private DB.

    Returns:
        The candidate's measured ``val_loss`` at its final optimizer step.

    Raises:
        TypeError / ValueError: Input validation failures.
        RuntimeError: Training subprocess failed or no completed run found.
    """
    ds = _validate_datasets(datasets)
    w = _validate_weights(weights, len(ds))
    timeout = _validate_timeout(timeout_seconds)
    base_text = _read_base_text(base_yaml_path)
    _validate_proxy_text(base_text)

    tmp_dir = tempfile.mkdtemp(prefix=".soup_mix_proxy.", dir=os.getcwd())
    try:
        output_dir = os.path.join(tmp_dir, "output")
        run_id = f"mix_{uuid.uuid4().hex}"
        db_path = os.path.abspath(tracker_db_path) if tracker_db_path else os.path.join(
            tmp_dir, "experiments.db"
        )
        overlay = _render_overlay_yaml(base_text, ds, w, output_dir=output_dir)
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
        env["SOUP_DB_PATH"] = db_path
        # The train command already accepts an exact run identity through this
        # variable. Never inherit an outer MCP job's ID and overwrite its row.
        env["SOUP_MCP_RUN_ID"] = run_id
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
        return _read_final_eval_loss(db_path, run_id, output_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
