"""``soup local-rl`` — personal-LLM flywheel daemon (v0.68.0 Part E).

Wrap Ollama / MLX inference, capture thumbs into SQLite, harvest DPO pairs,
and (in v0.68.1) DPO-train nightly via systemd / launchd. Smaller-scope
cousin of v0.58 ``soup loop`` — runs locally on a single workstation,
trains the user's personal model from their own feedback.

Schema + thumbs recording + DPO-pair harvester are LIVE from v0.68.0; the
nightly DPO/KTO/ORPO train runner + systemd/launchd scheduler scaffold land
in v0.71.13 (#229).

Public surface:

- ``SUPPORTED_LOCAL_RL_BACKENDS`` (``ollama``/``mlx``)
- ``SUPPORTED_LOCAL_RL_TRAIN_METHODS`` (``dpo``/``kto``/``orpo``)
- ``validate_local_rl_backend`` / ``validate_local_rl_train_method``
- ``LocalRLConfig`` frozen dataclass
- ``init_local_rl_db(db_path)`` — atomic table creation (interactions/thumbs/state)
- ``record_thumb(...)`` — append a thumbs-up/down record
- ``harvest_dpo_pairs(db_path)`` — pair up/down responses to same prompt
- ``get_state`` / ``set_state`` / ``count_new_thumbs_since`` — state-table I/O
- ``pairs_to_rows(pairs, method)`` — DPO pairs -> dpo/orpo/kto JSONL rows
- ``run_nightly_train(config)`` — live DPO/KTO/ORPO train runner (v0.71.13 #229)
"""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from soup_cli.utils.paths import is_under_cwd

SUPPORTED_LOCAL_RL_BACKENDS: frozenset[str] = frozenset({"ollama", "mlx"})
SUPPORTED_LOCAL_RL_TRAIN_METHODS: frozenset[str] = frozenset({"dpo", "kto", "orpo"})
_VALID_THUMBS: frozenset[str] = frozenset({"up", "down"})

MAX_PROMPT_LEN = 16_384
MAX_RESPONSE_LEN = 16_384
_MAX_BACKEND_LEN = 32
_MAX_TRAIN_METHOD_LEN = 32
_MAX_MODEL_LEN = 512

# v0.71.13 #229 — nightly train scheduler defaults.
MIN_PAIRS_DEFAULT = 10
_MAX_MIN_PAIRS = 1_000_000
_LAST_TRAIN_KEY = "last_train_at"
# task -> data format the trainer consumes (dpo/orpo read chosen+rejected;
# kto reads prompt+completion+label).
_METHOD_FORMAT = {"dpo": "dpo", "orpo": "dpo", "kto": "kto"}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_local_rl_backend(name: object) -> str:
    if isinstance(name, bool):
        raise TypeError("backend must not be bool")
    if not isinstance(name, str):
        raise TypeError("backend must be str")
    if not name:
        raise ValueError("backend must be non-empty")
    if "\x00" in name:
        raise ValueError("backend must not contain null bytes")
    if len(name) > _MAX_BACKEND_LEN:
        raise ValueError(
            f"backend length {len(name)} > {_MAX_BACKEND_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_LOCAL_RL_BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_LOCAL_RL_BACKENDS))
        )
    return canonical


def validate_local_rl_train_method(name: object) -> str:
    if isinstance(name, bool):
        raise TypeError("train_method must not be bool")
    if not isinstance(name, str):
        raise TypeError("train_method must be str")
    if not name:
        raise ValueError("train_method must be non-empty")
    if "\x00" in name:
        raise ValueError("train_method must not contain null bytes")
    if len(name) > _MAX_TRAIN_METHOD_LEN:
        raise ValueError(
            f"train_method length {len(name)} > {_MAX_TRAIN_METHOD_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_LOCAL_RL_TRAIN_METHODS:
        raise ValueError(
            f"unknown train_method {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_LOCAL_RL_TRAIN_METHODS))
        )
    return canonical


def _validate_model(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("model must not be bool")
    if not isinstance(value, str):
        raise TypeError("model must be str")
    if not value:
        raise ValueError("model must be non-empty")
    if "\x00" in value:
        raise ValueError("model must not contain null bytes")
    # Reject newlines / CR (systemd unit-file injection defence — the model
    # flows into a scheduler ExecStart line via local_rl_scheduler).
    if "\n" in value or "\r" in value:
        raise ValueError("model must not contain newline / carriage return")
    if len(value) > _MAX_MODEL_LEN:
        raise ValueError(f"model length {len(value)} > {_MAX_MODEL_LEN}")
    return value


def _validate_thumb(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("thumb must not be bool")
    if not isinstance(value, str):
        raise TypeError("thumb must be str")
    if value not in _VALID_THUMBS:
        raise ValueError(
            f"thumb must be one of {sorted(_VALID_THUMBS)}, got {value!r}"
        )
    return value


def validate_db_path(path: object) -> str:
    if isinstance(path, bool):
        raise TypeError("db_path must not be bool")
    if not isinstance(path, str):
        raise TypeError("db_path must be str")
    if not path:
        raise ValueError("db_path must be non-empty")
    if "\x00" in path:
        raise ValueError("db_path must not contain null bytes")
    # Newline / CR rejection: db_path flows into a scheduler ExecStart line
    # via local_rl_scheduler (systemd unit-file injection defence).
    if "\n" in path or "\r" in path:
        raise ValueError("db_path must not contain newline / carriage return")
    if not is_under_cwd(path):
        raise ValueError(
            f"db_path {os.path.basename(path)!r} must stay under cwd"
        )
    if os.path.lexists(path):
        try:
            st = os.lstat(path)
        except OSError as exc:
            raise ValueError(
                f"db_path unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                "db_path must not be a symlink (TOCTOU defence)"
            )
    return path


def _validate_text(value: object, field: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} length {len(value)} > {max_len}")
    return value


# ---------------------------------------------------------------------------
# LocalRLConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalRLConfig:
    """Runtime config for the local-RL flywheel."""

    backend: str
    model: str
    db_path: str
    train_method: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend", validate_local_rl_backend(self.backend)
        )
        _validate_model(self.model)
        validate_db_path(self.db_path)
        object.__setattr__(
            self,
            "train_method",
            validate_local_rl_train_method(self.train_method),
        )


@dataclass(frozen=True)
class DpoPair:
    """A harvested DPO training pair (prompt + chosen + rejected)."""

    prompt: str
    chosen: str
    rejected: str


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------


_SCHEMA_INTERACTIONS = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL
)
"""

_SCHEMA_THUMBS = """
CREATE TABLE IF NOT EXISTS thumbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    thumb TEXT NOT NULL CHECK (thumb IN ('up', 'down'))
)
"""

# v0.71.13 #229 — key/value state table (tracks last_train_at across runs).
_SCHEMA_STATE = """
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""


def init_local_rl_db(db_path: str) -> None:
    """Atomically create the local-RL SQLite schema. Idempotent."""
    validate_db_path(db_path)
    real = os.path.abspath(db_path)
    parent = os.path.dirname(real) or "."
    os.makedirs(parent, exist_ok=True)
    with sqlite3.connect(real) as conn:
        conn.execute(_SCHEMA_INTERACTIONS)
        conn.execute(_SCHEMA_THUMBS)
        conn.execute(_SCHEMA_STATE)
        conn.commit()
    # Best-effort 0o600 perms (matches v0.26.0 registry.db policy on POSIX).
    if os.name == "posix":
        try:
            os.chmod(real, 0o600)
        except OSError:
            pass


def record_thumb(
    *,
    db_path: str,
    prompt: str,
    response: str,
    thumb: str,
) -> None:
    """Append a thumbs-up/down record. cwd-contained, symlink-rejected."""
    validate_db_path(db_path)
    _validate_text(prompt, field="prompt", max_len=MAX_PROMPT_LEN)
    _validate_text(response, field="response", max_len=MAX_RESPONSE_LEN)
    _validate_thumb(thumb)
    real = os.path.abspath(db_path)
    with sqlite3.connect(real) as conn:
        conn.execute(
            "INSERT INTO thumbs (ts, prompt, response, thumb) VALUES (?, ?, ?, ?)",
            (time.time(), prompt, response, thumb),
        )
        conn.commit()


def harvest_dpo_pairs(db_path: str) -> Tuple[DpoPair, ...]:
    """Pair up/down responses to the same prompt into DPO training rows.

    For each prompt that has at least one ``up`` AND at least one ``down``
    thumb, emit a single (chosen, rejected) tuple using the most recent
    up + most recent down. Subsequent up/down pairs for the same prompt
    are skipped (one row per prompt, last writes win) — keeps the harvest
    deterministic without exploding into a Cartesian product.
    """
    validate_db_path(db_path)
    real = os.path.abspath(db_path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"db_path not found: {db_path!r}")
    with sqlite3.connect(real) as conn:
        rows = conn.execute(
            "SELECT prompt, response, thumb, ts FROM thumbs ORDER BY ts ASC"
        ).fetchall()
    last_up: dict[str, str] = {}
    last_down: dict[str, str] = {}
    for prompt, response, thumb, _ts in rows:
        if thumb == "up":
            last_up[prompt] = response
        elif thumb == "down":
            last_down[prompt] = response
    pairs = []
    for prompt in sorted(set(last_up) & set(last_down)):
        pairs.append(
            DpoPair(
                prompt=prompt,
                chosen=last_up[prompt],
                rejected=last_down[prompt],
            )
        )
    return tuple(pairs)


# ---------------------------------------------------------------------------
# State table I/O (v0.71.13 #229)
# ---------------------------------------------------------------------------


def _ensure_state_table(conn: sqlite3.Connection) -> None:
    """Create the ``state`` table if a pre-v0.71.13 DB lacks it."""
    conn.execute(_SCHEMA_STATE)


def get_state(db_path: str, key: str) -> "str | None":
    """Return the stored value for ``key`` or ``None`` when absent."""
    validate_db_path(db_path)
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    real = os.path.abspath(db_path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"db_path not found: {db_path!r}")
    with sqlite3.connect(real) as conn:
        _ensure_state_table(conn)
        row = conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row is not None else None


def set_state(db_path: str, key: str, value: str) -> None:
    """Upsert ``key`` -> ``value`` in the ``state`` table."""
    validate_db_path(db_path)
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    real = os.path.abspath(db_path)
    with sqlite3.connect(real) as conn:
        _ensure_state_table(conn)
        conn.execute(
            "INSERT INTO state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def count_new_thumbs_since(db_path: str, since_ts: "float | None") -> int:
    """Count thumbs with ``ts > since_ts`` (all thumbs when ``since_ts`` is None)."""
    validate_db_path(db_path)
    if since_ts is not None and (
        isinstance(since_ts, bool) or not isinstance(since_ts, (int, float))
    ):
        raise TypeError("since_ts must be a number or None")
    real = os.path.abspath(db_path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"db_path not found: {db_path!r}")
    with sqlite3.connect(real) as conn:
        if since_ts is None:
            row = conn.execute("SELECT COUNT(*) FROM thumbs").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM thumbs WHERE ts > ?", (float(since_ts),)
            ).fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Pair -> training rows (v0.71.13 #229)
# ---------------------------------------------------------------------------


def pairs_to_rows(
    pairs: "Tuple[DpoPair, ...]", train_method: str
) -> "list[dict]":
    """Convert harvested DPO pairs to JSONL rows for ``train_method``.

    - ``dpo`` / ``orpo`` -> one ``{prompt, chosen, rejected}`` row per pair.
    - ``kto`` -> two unpaired ``{prompt, completion, label}`` rows per pair
      (chosen -> label True, rejected -> label False).
    """
    method = validate_local_rl_train_method(train_method)
    rows: "list[dict]" = []
    for pair in pairs:
        if not isinstance(pair, DpoPair):
            raise TypeError("pairs must contain DpoPair objects")
        if method == "kto":
            rows.append(
                {"prompt": pair.prompt, "completion": pair.chosen, "label": True}
            )
            rows.append(
                {"prompt": pair.prompt, "completion": pair.rejected, "label": False}
            )
        else:  # dpo / orpo
            rows.append(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Nightly train runner (v0.71.13 #229) — lifts the v0.68.1 stub
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NightlyTrainResult:
    """Outcome of a ``run_nightly_train`` invocation."""

    status: str  # "trained" / "skipped_no_new_thumbs" / "skipped_insufficient_pairs"
    num_pairs: int
    output_dir: "str | None"
    reason: str


def _default_train_fn(
    *,
    base_model: str,
    pairs_path: str,
    output_dir: str,
    train_method: str,
) -> None:
    """Train one round via a ``soup train`` subprocess (no shell).

    Mirrors v0.71.11 ``iterative_dpo._default_train_fn``: render a YAML via
    ``yaml.safe_dump`` (no value can inject extra keys) and invoke
    ``python -m soup_cli.cli train --config <tmp> --yes`` with an argv list.
    """
    import subprocess
    import sys
    import tempfile

    import yaml

    data_format = _METHOD_FORMAT[train_method]
    yaml_text = yaml.safe_dump(
        {
            "base": base_model,
            "task": train_method,
            "data": {"train": pairs_path, "format": data_format, "max_length": 256},
            "training": {"epochs": 1, "batch_size": 1},
            "output": output_dir,
        },
        default_flow_style=False,
        sort_keys=False,
    )
    fd, tmp_yaml = tempfile.mkstemp(
        suffix=".yaml", prefix=".soup_localrl_", dir=os.getcwd()
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        subprocess.run(  # noqa: S603 — argv list, no shell
            [
                sys.executable,
                "-m",
                "soup_cli.cli",
                "train",
                "--config",
                tmp_yaml,
                "--yes",
            ],
            check=True,
        )
    finally:
        try:
            os.remove(tmp_yaml)
        except OSError:
            pass


def run_nightly_train(
    config: LocalRLConfig,
    *,
    once: bool = False,
    min_pairs: int = MIN_PAIRS_DEFAULT,
    output_dir: str = "local_rl_adapter",
    train_fn: Optional[Callable[..., None]] = None,
) -> NightlyTrainResult:
    """Harvest the latest DPO pairs and train them (v0.71.13 #229).

    The flow: read ``last_train_at`` from the SQLite ``state`` table; if a
    prior train exists and no new thumbs landed since, skip. Otherwise
    harvest pairs and — when at least ``min_pairs`` are available — write a
    tmp JSONL in the right shape for ``config.train_method`` and invoke the
    DPO/KTO/ORPO trainer via ``soup train`` (or an injected ``train_fn`` for
    tests). ``last_train_at`` is stamped only after a real train.

    ``config.model`` is the **training base** here (an HF repo id or local
    path) — distinct from the Ollama tag used by the record/harvest loop.

    The ``once`` flag is accepted for symmetry with the CLI (the runner is
    the same whether invoked ad-hoc or by the scheduler); it is recorded but
    does not change the harvest/skip logic.
    """
    if not isinstance(config, LocalRLConfig):
        raise TypeError("config must be LocalRLConfig")
    if not isinstance(once, bool):
        raise TypeError("once must be bool")
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int):
        raise TypeError("min_pairs must be int")
    if min_pairs < 1 or min_pairs > _MAX_MIN_PAIRS:
        raise ValueError(f"min_pairs must be in [1, {_MAX_MIN_PAIRS}]")
    if not isinstance(output_dir, str) or not output_dir or "\x00" in output_dir:
        raise ValueError("output_dir must be a non-empty NUL-free string")

    runner = train_fn if train_fn is not None else _default_train_fn
    if not callable(runner):
        raise TypeError("train_fn must be callable")

    db = config.db_path
    last_raw = get_state(db, _LAST_TRAIN_KEY)
    last_ts: "float | None" = None
    if last_raw is not None:
        try:
            last_ts = float(last_raw)
        except (TypeError, ValueError):
            last_ts = None

    if last_ts is not None:
        new = count_new_thumbs_since(db, last_ts)
        if new == 0:
            return NightlyTrainResult(
                status="skipped_no_new_thumbs",
                num_pairs=0,
                output_dir=None,
                reason=f"no new thumbs since last train at {last_ts}",
            )

    # Stamp the run start (BEFORE harvest), not the post-train wall-clock, so
    # any thumbs recorded *during* the train window keep ``ts > last_train_at``
    # and are counted by the next run instead of being silently dropped
    # (code-review MEDIUM fix).
    run_started = time.time()
    pairs = harvest_dpo_pairs(db)
    if len(pairs) < min_pairs:
        return NightlyTrainResult(
            status="skipped_insufficient_pairs",
            num_pairs=len(pairs),
            output_dir=None,
            reason=f"{len(pairs)} pairs < min_pairs {min_pairs}",
        )

    import json
    import tempfile

    rows = pairs_to_rows(pairs, config.train_method)
    fd, tmp_jsonl = tempfile.mkstemp(
        suffix=".jsonl", prefix=".soup_localrl_pairs_", dir=os.getcwd()
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        runner(
            base_model=config.model,
            pairs_path=tmp_jsonl,
            output_dir=output_dir,
            train_method=config.train_method,
        )
    finally:
        try:
            os.remove(tmp_jsonl)
        except OSError:
            pass

    set_state(db, _LAST_TRAIN_KEY, repr(run_started))
    return NightlyTrainResult(
        status="trained",
        num_pairs=len(pairs),
        output_dir=output_dir,
        reason=f"trained {len(pairs)} pairs via {config.train_method}",
    )


__all__ = [
    "SUPPORTED_LOCAL_RL_BACKENDS",
    "SUPPORTED_LOCAL_RL_TRAIN_METHODS",
    "MAX_PROMPT_LEN",
    "MAX_RESPONSE_LEN",
    "MIN_PAIRS_DEFAULT",
    "validate_local_rl_backend",
    "validate_local_rl_train_method",
    "validate_db_path",
    "LocalRLConfig",
    "DpoPair",
    "NightlyTrainResult",
    "init_local_rl_db",
    "record_thumb",
    "harvest_dpo_pairs",
    "get_state",
    "set_state",
    "count_new_thumbs_since",
    "pairs_to_rows",
    "run_nightly_train",
]
