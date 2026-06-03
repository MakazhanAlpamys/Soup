"""v0.71.10 #200 — live RA-DIT two-stage orchestration + retriever auto-link.

Lifts the v0.62.0 Part B deferred note ("Live orchestration that chains the
two stages in a single call is deferred"). Two surfaces:

* ``autolink_generator_retriever(cfg)`` — when a ``soup train`` of a generator
  stage (``training.ra_dit_stage='generator'``) has no
  ``training.ra_dit_retriever_model`` set, discover the most-recent
  RA-DIT retriever run from the Registry and splice its output in. A manual
  value always wins (no overwrite); a yellow advisory string is returned so
  the caller can surface what happened.
* ``run_ra_dit(retriever_config, generator_config, ...)`` — the one-shot
  orchestrator behind ``soup ra-dit``: run the retriever stage, link its
  output into the generator config, run the generator stage. Subprocess
  invocation mirrors ``utils.mix_proxy`` (list argv, no shell, per-stage
  ``soup train --config <yaml> --yes``).

Security:
- Config paths are containment-checked via the shared
  :func:`enforce_under_cwd_and_no_symlink` helper (TOCTOU symlink rejection).
- The rewritten generator YAML is staged in a ``tempfile.mkdtemp`` dir and
  cleaned up in ``finally`` (matches v0.53.5 ``mix_proxy`` policy).
- ``timeout_seconds`` capped to ``[60, 6*3600]``; ``TimeoutExpired`` raises
  ``RuntimeError`` so the CLI can report a clean per-stage failure.
- No top-level torch / transformers import (CLI stays fast).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

_MIN_TIMEOUT_S = 60
_MAX_TIMEOUT_S = 6 * 60 * 60  # retriever + generator can each be long.
_MAX_PATH_LEN = 4096
_MAX_YAML_BYTES = 256 * 1024
_RETRIEVER_STAGE = "retriever"
_GENERATOR_STAGE = "generator"

__all__ = [
    "RaDitRunResult",
    "autolink_generator_retriever",
    "discover_latest_retriever",
    "resolve_retriever_for_generator",
    "run_ra_dit",
    "validate_ra_dit_config_path",
]


@dataclass(frozen=True)
class RaDitRunResult:
    """Outcome of a two-stage :func:`run_ra_dit`. Frozen post-construction."""

    retriever_output: str
    generator_output: str
    retriever_model_used: str
    autolinked: bool


def discover_latest_retriever(*, store: Any = None) -> Optional[str]:
    """Return the output dir of the most-recent RA-DIT retriever Registry run.

    Returns the ``output`` field of the first ``task='embedding'`` entry whose
    stored config declares ``training.ra_dit_stage == 'retriever'``.
    ``RegistryStore.list`` orders ``created_at DESC, id DESC`` (see
    ``registry/store.py``), so the first match IS the most-recent retriever
    run. Returns ``None`` when none exist. Registry/IO errors degrade to
    ``None`` (auto-link is advisory; a missing Registry must never crash a
    training run). The discovered ``output`` is run through
    ``validate_ra_dit_retriever_model`` so a corrupt Registry row (null byte /
    oversize) is skipped rather than flowing into the training config.
    """
    from soup_cli.utils.ra_dit import (  # noqa: PLC0415
        validate_ra_dit_retriever_model,
    )

    own_store = False
    if store is None:
        try:
            from soup_cli.registry.store import RegistryStore  # noqa: PLC0415

            store = RegistryStore()
            own_store = True
        except Exception:  # noqa: BLE001 — Registry is optional.
            return None
    try:
        entries = store.list(task="embedding", limit=100)
    except Exception:  # noqa: BLE001 — degrade to "no retriever".
        return None
    finally:
        if own_store:
            _close_quietly(store)

    for entry in entries:
        raw = entry.get("config_json")
        if not isinstance(raw, str):
            continue
        try:
            cfg = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(cfg, dict):
            continue
        training = cfg.get("training")
        if not isinstance(training, dict):
            continue
        if training.get("ra_dit_stage") != _RETRIEVER_STAGE:
            continue
        output = cfg.get("output")
        if not (isinstance(output, str) and output):
            continue
        try:
            return validate_ra_dit_retriever_model(output)
        except (TypeError, ValueError):
            # Corrupt Registry row — skip it, keep looking for a clean one.
            continue
    return None


def _close_quietly(store: Any) -> None:
    closer = getattr(store, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:  # noqa: BLE001 — best-effort cleanup.
            pass


def resolve_retriever_for_generator(
    retriever_model: Optional[str], *, store: Any = None
) -> Tuple[Optional[str], str]:
    """Resolve the retriever model for a generator stage.

    A manual ``retriever_model`` always wins. Otherwise the latest RA-DIT
    retriever run is discovered from the Registry. Returns
    ``(resolved_or_None, advisory_message)``.
    """
    if retriever_model is not None:
        return (
            retriever_model,
            f"Using manual --retriever-model override: {retriever_model}",
        )
    discovered = discover_latest_retriever(store=store)
    if discovered is not None:
        return (
            discovered,
            f"Auto-linked latest RA-DIT retriever from Registry: {discovered}",
        )
    return (
        None,
        "No RA-DIT retriever found in Registry to auto-link; train the "
        "retriever stage first or pass --retriever-model.",
    )


def autolink_generator_retriever(cfg: Any) -> Optional[str]:
    """Auto-link a generator-stage config to the latest retriever run.

    Mutates ``cfg.training.ra_dit_retriever_model`` in place when (a) the
    stage is ``generator`` and (b) no retriever model is already set. Returns
    an advisory string describing what happened, or ``None`` when no action
    was taken (non-generator stage, or a manual value already present).
    """
    training = getattr(cfg, "training", None)
    if training is None:
        return None
    if getattr(training, "ra_dit_stage", None) != _GENERATOR_STAGE:
        return None
    if getattr(training, "ra_dit_retriever_model", None) is not None:
        # Manual value present — never overwrite.
        return None
    resolved, advisory = resolve_retriever_for_generator(None)
    if resolved is not None:
        training.ra_dit_retriever_model = resolved
    return advisory


def validate_ra_dit_config_path(name: str, raw: str) -> str:
    """Containment-check an operator-supplied RA-DIT config path.

    Public so the ``soup ra-dit`` CLI can reuse it without importing a private
    symbol across modules (code-review M5).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink  # noqa: PLC0415

    if not isinstance(raw, str):
        raise TypeError(f"{name} must be str, got {type(raw).__name__}")
    if len(raw) > _MAX_PATH_LEN:
        raise ValueError(f"{name} length {len(raw)} exceeds {_MAX_PATH_LEN}")
    enforce_under_cwd_and_no_symlink(raw, name)
    if not os.path.isfile(os.path.realpath(raw)):
        raise FileNotFoundError(
            f"{name} not found: {os.path.basename(raw)!r}"
        )
    return raw


# Back-compat private alias (pre-v0.71.10-review callers).
_validate_config_path = validate_ra_dit_config_path


def _validate_timeout(timeout_seconds: object) -> int:
    if isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be int, not bool")
    if not isinstance(timeout_seconds, int):
        raise TypeError(
            f"timeout_seconds must be int, got {type(timeout_seconds).__name__}"
        )
    if timeout_seconds < _MIN_TIMEOUT_S or timeout_seconds > _MAX_TIMEOUT_S:
        raise ValueError(
            f"timeout_seconds must be in [{_MIN_TIMEOUT_S}, "
            f"{_MAX_TIMEOUT_S}], got {timeout_seconds}"
        )
    return timeout_seconds


def _load_yaml_config(path: str) -> dict:
    """Read + parse a config YAML, re-rejecting a symlink at read time.

    ``_validate_config_path`` already ran ``enforce_under_cwd_and_no_symlink``
    on this path, but the file is opened again here — so use ``O_NOFOLLOW``
    (POSIX) + ``fstat`` to close the TOCTOU window a symlink-swap-after-check
    would otherwise open (mirrors ``_eval_v07110._load_jsonl_rows`` policy).
    """
    import stat  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, os.O_RDONLY | no_follow)
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise ValueError("config YAML must be a regular file")
        if st.st_size > _MAX_YAML_BYTES:
            raise ValueError(
                f"config YAML exceeds {_MAX_YAML_BYTES // 1024}KB cap "
                f"(got {st.st_size} bytes)"
            )
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("config YAML must be a top-level mapping")
    return raw


def _run_train_subprocess(config_path: str, *, timeout_seconds: int) -> None:
    """Default stage runner: ``python -m soup_cli.cli train --config <p> --yes``.

    List argv, no shell (matches v0.53.5 ``mix_proxy`` / v0.40.4 ``execvp``
    policy). Raises ``RuntimeError`` on non-zero rc or timeout.
    """
    argv = [
        sys.executable,
        "-m",
        "soup_cli.cli",
        "train",
        "--config",
        config_path,
        "--yes",
    ]
    try:
        result = subprocess.run(  # noqa: S603 — argv list, no shell.
            argv,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"RA-DIT stage exceeded {timeout_seconds}s timeout"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"RA-DIT stage failed (rc={result.returncode})"
        )


def run_ra_dit(
    retriever_config: str,
    generator_config: str,
    *,
    retriever_model: Optional[str] = None,
    timeout_seconds: int = 6 * 60 * 60,
    _runner: Optional[Callable[[str], None]] = None,
) -> RaDitRunResult:
    """Chain the two RA-DIT stages: retriever → auto-link → generator.

    Args:
        retriever_config: Path (under cwd) to the stage-1 embedding recipe.
        generator_config: Path (under cwd) to the stage-2 RAFT-SFT recipe.
        retriever_model: Optional manual override; when set, skips Registry
            auto-link and uses this value as the generator's retriever model.
        timeout_seconds: Per-stage hard timeout (60s..6h).
        _runner: Test seam — a ``(config_path) -> None`` callable that stands
            in for the subprocess train call.

    Returns:
        :class:`RaDitRunResult` describing both stages' outputs + the link.

    Raises:
        TypeError / ValueError / FileNotFoundError: input validation.
        RuntimeError: a training stage failed.
    """
    retr_path = _validate_config_path("retriever_config", retriever_config)
    gen_path = _validate_config_path("generator_config", generator_config)
    timeout = _validate_timeout(timeout_seconds)
    runner = _runner if _runner is not None else (
        lambda p: _run_train_subprocess(p, timeout_seconds=timeout)
    )

    retr_cfg = _load_yaml_config(retr_path)
    gen_cfg = _load_yaml_config(gen_path)
    retriever_output = retr_cfg.get("output")
    if not isinstance(retriever_output, str) or not retriever_output:
        raise ValueError(
            "retriever config must declare a non-empty 'output' directory"
        )
    generator_output = gen_cfg.get("output")
    if not isinstance(generator_output, str) or not generator_output:
        raise ValueError(
            "generator config must declare a non-empty 'output' directory"
        )

    # Stage 1 — train the retriever.
    runner(retr_path)

    # Link: manual override wins; else the retriever's own output dir.
    if retriever_model is not None:
        model_used = retriever_model
        autolinked = False
    else:
        model_used = retriever_output
        autolinked = True

    # Rewrite the generator config to carry the resolved retriever model.
    training = gen_cfg.get("training")
    if not isinstance(training, dict):
        training = {}
        gen_cfg["training"] = training
    training["ra_dit_retriever_model"] = model_used

    tmp_dir = tempfile.mkdtemp(prefix=".soup_ra_dit.")
    try:
        import yaml  # noqa: PLC0415

        tmp_gen = os.path.join(tmp_dir, "generator.yaml")
        with open(tmp_gen, "w", encoding="utf-8") as fh:
            yaml.safe_dump(gen_cfg, fh, sort_keys=False)
        # Stage 2 — train the generator with the linked retriever.
        runner(tmp_gen)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return RaDitRunResult(
        retriever_output=retriever_output,
        generator_output=generator_output,
        retriever_model_used=model_used,
        autolinked=autolinked,
    )
