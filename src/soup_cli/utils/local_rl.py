"""``soup local-rl`` — personal-LLM flywheel daemon (v0.68.0 Part E).

Wrap Ollama / MLX inference, capture thumbs into SQLite, harvest DPO pairs,
and (in v0.68.1) DPO-train nightly via systemd / launchd. Smaller-scope
cousin of v0.58 ``soup loop`` — runs locally on a single workstation,
trains the user's personal model from their own feedback.

Schema + thumbs recording + DPO-pair harvester are LIVE in v0.68.0; the
nightly train scheduler is the deferred stub.

Public surface:

- ``SUPPORTED_LOCAL_RL_BACKENDS`` (``ollama``/``mlx``)
- ``SUPPORTED_LOCAL_RL_TRAIN_METHODS`` (``dpo``/``kto``/``orpo``)
- ``validate_local_rl_backend`` / ``validate_local_rl_train_method``
- ``LocalRLConfig`` frozen dataclass
- ``init_local_rl_db(db_path)`` — atomic table creation
- ``record_thumb(...)`` — append a thumbs-up/down record
- ``harvest_dpo_pairs(db_path)`` — pair up/down responses to same prompt
- ``run_nightly_train(config)`` — NotImplementedError stub w/ v0.68.1 marker
"""

from __future__ import annotations

import os
import sqlite3
import stat
import time
from dataclasses import dataclass
from typing import Tuple

from soup_cli.utils.paths import is_under_cwd

SUPPORTED_LOCAL_RL_BACKENDS: frozenset = frozenset({"ollama", "mlx"})
SUPPORTED_LOCAL_RL_TRAIN_METHODS: frozenset = frozenset({"dpo", "kto", "orpo"})
_VALID_THUMBS: frozenset = frozenset({"up", "down"})

MAX_PROMPT_LEN = 16_384
MAX_RESPONSE_LEN = 16_384
_MAX_BACKEND_LEN = 32
_MAX_TRAIN_METHOD_LEN = 32
_MAX_MODEL_LEN = 512


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


def init_local_rl_db(db_path: str) -> None:
    """Atomically create the local-RL SQLite schema. Idempotent."""
    validate_db_path(db_path)
    real = os.path.abspath(db_path)
    parent = os.path.dirname(real) or "."
    os.makedirs(parent, exist_ok=True)
    with sqlite3.connect(real) as conn:
        conn.execute(_SCHEMA_INTERACTIONS)
        conn.execute(_SCHEMA_THUMBS)
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
    last_up: dict = {}
    last_down: dict = {}
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
# Live train stub (v0.68.1)
# ---------------------------------------------------------------------------


def run_nightly_train(config: LocalRLConfig) -> None:
    """Run the nightly DPO/KTO/ORPO train. Deferred to v0.68.1."""
    if not isinstance(config, LocalRLConfig):
        raise TypeError("config must be LocalRLConfig")
    raise NotImplementedError(
        "local-rl nightly train is deferred to v0.68.1 — "
        "harvest DPO pairs today, train them once the runner lands"
    )


__all__ = [
    "SUPPORTED_LOCAL_RL_BACKENDS",
    "SUPPORTED_LOCAL_RL_TRAIN_METHODS",
    "MAX_PROMPT_LEN",
    "MAX_RESPONSE_LEN",
    "validate_local_rl_backend",
    "validate_local_rl_train_method",
    "validate_db_path",
    "LocalRLConfig",
    "DpoPair",
    "init_local_rl_db",
    "record_thumb",
    "harvest_dpo_pairs",
    "run_nightly_train",
]
