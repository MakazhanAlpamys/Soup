"""Loop state file (v0.58.0 Part A — control plane).

`soup loop` orchestrates the *production traces → preference pairs →
Eval-Gated DPO → canary deploy → rollback* cycle. State for the whole
loop lives in a single ``.soup/loop.yaml`` next to the project, with
atomic writes + cwd containment + symlink rejection — the same TOCTOU
policy as every other v0.5x persistence surface (v0.33.0 #22 /
v0.43.0 Part C / v0.53.7 #106).

Status (``running`` / ``paused`` / ``stopped``) and counters (traces /
pairs / runs / deploys) live here; per-iteration artifacts ship as
v0.26.0 Soup Cans under ``.soup-loops/``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.process_liveness import process_is_alive

# Status values are deliberately closed — every state machine transition
# below must remain auditable.
LOOP_STATUSES: frozenset = frozenset({"running", "paused", "stopped"})
# The check before rollout (`soup loop init --verify`).
VERIFY_METHODS: tuple = ("ship", "eval_gate")
# The dataset the pre-wired loop trains on (`soup loop init --train-method`).
TRAIN_METHODS: tuple = ("dpo", "kto", "sft")
# What a canary does when the ratings its quality verdict needs have not arrived by the
# maximum wait (`soup loop init --canary-on-timeout`): "errors-only" lets the request-error
# verdict decide, "rollback" withdraws the canary, "wait" never gives up.
CANARY_ON_TIMEOUT: tuple = ("errors-only", "rollback", "wait")
# How long after a rollout started the ratings may stay short (`soup loop init
# --canary-max-wait`): a day by default, at most 30 days.
DEFAULT_CANARY_MAX_WAIT_SEC = 86400.0
MAX_CANARY_MAX_WAIT_SEC = 30 * 86400.0

# The keys of the sft recipe the pre-wired train stage renders (`soup loop init --sft-*`).
SFT_RECIPE_KEYS: tuple = (
    "target_modules", "lora_r", "lora_alpha", "lr", "epochs",
    "batch_size", "gradient_accumulation_steps", "seed", "quantization",
)
# What `quantization` may say. "4bit" and "none" are written as they are. "auto" is measured
# when the train stage renders the soup.yaml: "none" if the base fits in free VRAM
# (parameters x 2 bytes x 5), else "4bit" (utils/loop_quantization.py).
SFT_QUANTIZATIONS: tuple = ("4bit", "none", "auto")
# The sft recipe of a loop whose loop.yaml names none (or only some) of the keys.
# target_modules None would write no line, so the config schema's default ("auto", peft decides)
# applies.
# Fit runs, 2026-11-06: candidate B in bf16.
# batch_size, gradient_accumulation_steps and seed are the config's own fields, pinned so a
# run inherits nothing: a live run on the same rows inherited batch 1 and no seed, made 152
# optimizer steps instead of 80 at lr 5e-4, and its quality fell (format_json 1.00 to 0.775).
DEFAULT_SFT_RECIPE: Mapping[str, object] = MappingProxyType({
    "target_modules": "all-linear",
    "lora_r": 64,
    "lora_alpha": 128,
    "lr": 5.0e-4,
    "epochs": 8,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "seed": 42,
    "quantization": "auto",
})
# What peft calls every linear layer; a plain string in the config, never a list entry.
ALL_LINEAR = "all-linear"
_SFT_LORA_R_MAX = 1024
_SFT_LORA_ALPHA_MAX = 4096
_SFT_EPOCHS_MAX = 100
_SFT_BATCH_MAX = 1024
_SFT_ACCUM_MAX = 1024
#: numpy.random.seed, which transformers.set_seed feeds, takes 0 to 2**32 - 1: the schema's bound.
_SFT_SEED_MAX = 2**32 - 1
_MAX_TARGET_MODULES = 64
_MAX_MODULE_NAME_LEN = 128
# A module name, or a dotted path to one: no pattern, no whitespace, no control characters.
_MODULE_NAME_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", re.ASCII)

_MAX_PATH_LEN = 4096
_MAX_STR_FIELD = 512
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB cap on the state file
# The data cursor is epoch seconds up to the year 9999 (datetime.max): `soup loop status`
# prints it as a date, and a larger value is a typo that would stop all learning.
MAX_CURSOR_TS = 253402300799.0
_DEFAULT_STATE_DIR = ".soup"
_DEFAULT_STATE_FILENAME = "loop.yaml"


@dataclass(frozen=True)
class LoopState:
    """Immutable snapshot of a `soup loop` configuration + counters.

    The persisted file is JSON-formatted (despite the ``.yaml`` extension)
    so we can use the stdlib parser without pulling pyyaml into the read
    path; YAML is a superset of JSON for objects and the file remains
    human-readable. Counters are absolute lifetime totals; per-iteration
    detail lives in the ``.soup-loops/`` Soup Can artifacts.
    """

    served_model: str
    eval_suite: str
    baseline: str
    status: str = "stopped"
    # v0.71.4 #176 — use the pre-wired harvest/train/gate/deploy stages
    # (utils/loop_stages.py) instead of the no-op default callbacks.
    pre_wired: bool = False
    traces_collected: int = 0
    pairs_distilled: int = 0
    runs_gated: int = 0
    adapters_shipped: int = 0
    canary_active: Optional[str] = None
    canary_traffic_pct: Optional[float] = None
    canary_autoroll_on_regress: bool = True
    canary_rollout_id: Optional[str] = None
    monthly_budget_usd: Optional[float] = None
    spent_this_month_usd: float = 0.0
    max_runs_per_day: Optional[int] = None
    runs_today: int = 0
    last_run_date: Optional[str] = None  # YYYY-MM-DD UTC
    last_iteration_id: Optional[str] = None
    iteration_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    # #1440 — local-RL SQLite that `soup serve --record-thumbs` writes; the
    # pre-wired harvest reads its thumbs as preference pairs. None = not used.
    # Last, so adding it shifted no existing positional field.
    thumbs_db: Optional[str] = None
    # The model the pre-wired train stage starts from: `soup loop init
    # --base-model`, or what init resolved from a registry:// entry. None =
    # resolve at watch time (served_model itself when it is not registry://).
    base_model: Optional[str] = None
    # The directory of conversation logs (`soup serve --trace-log` files,
    # *.jsonl) the pre-wired harvest reads: `soup loop init --trace-dir`.
    # None = .soup-loops/traces. The served model path is never read as one.
    trace_dir: Optional[str] = None
    # The check before rollout: "ship" (a soup ship verdict against the
    # production version) or "eval_gate". None = a loop.yaml written before
    # this field existed; those loops ran the eval gate and still do.
    verify: Optional[str] = None
    # Adapter of the version serving as production: the "before" side of the
    # ship verdict. None = the base model itself.
    production_adapter: Optional[str] = None
    # Share of keyed traffic a newly deployed canary gets (`soup loop init
    # --canary-traffic`, default 5%).
    deploy_traffic_pct: float = 5.0
    # Path of the adapter now in canary; it becomes production_adapter on promotion.
    canary_adapter: Optional[str] = None
    # Served adapter name of the production version: the canary router's stable
    # side. None = the served model itself (nothing promoted yet).
    production_name: Optional[str] = None
    # Production before the last promotion; `soup loop rollback` returns to it
    # (previous_production_name None = the served model itself).
    previous_production_name: Optional[str] = None
    previous_production_adapter: Optional[str] = None
    # The data cursor: the newest feedback timestamp (epoch seconds) that the version in
    # PRODUCTION was trained on (its lineage); None = nothing promoted yet, so every row is new.
    # It moves when the canary that used the rows is promoted, never on a training run alone, so
    # the rows of a rejected candidate are new again (``canary_cursor_ts`` and
    # ``previous_harvest_cursor_ts``, at the end, carry it through a canary and a rollback).
    # With ``attempt_cursor_ts`` it is also the trigger: training waits for `min_new_pairs` pairs
    # newer than the later of the two; the run still trains on all the pairs, as each iteration
    # trains a fresh LoRA from the base model. A loop.yaml from before the lineage cursor holds
    # the cursor of its last training run: that is taken as production's.
    harvest_cursor_ts: Optional[float] = None
    # Training waits until this many preference pairs are newer than the cursor
    # (`soup loop init --min-new-pairs`).
    min_new_pairs: int = 1
    # The feedback log (feedback.jsonl lines: thumbs, ratings, edits, comments on logged
    # answers) the pre-wired harvest reads: `soup loop init --feedback-log`. None =
    # .soup-loops/feedback.jsonl; SOUP_LOOP_FEEDBACK_LOG overrides both.
    feedback_log: Optional[str] = None
    # What the pre-wired loop trains: "dpo" (preference pairs), "kto" (rows labelled good or
    # bad) or "sft" (the answers people corrected and the ones they rated good).
    # `min_new_pairs` counts new rows of this method's dataset. "dpo" stays the default HERE,
    # so a loop.yaml without the field keeps the method it always had; `soup loop init`
    # (new loops) defaults to "sft".
    train_method: str = "dpo"
    # When the active rollout started (epoch seconds): the deploy stage, or `soup loop
    # canary`, writes it with canary_active. The quality verdict counts only answers given
    # from then on. None = no canary, or one started before this field existed (it is then
    # judged by request errors alone).
    canary_started_ts: Optional[float] = None
    # How long (seconds) after a rollout started its quality verdict may stay short of
    # ratings before `canary_on_timeout` applies (`soup loop init --canary-max-wait`).
    # Last, so adding it shifted no existing positional field.
    canary_max_wait_sec: float = DEFAULT_CANARY_MAX_WAIT_SEC
    # What happens then: "errors-only", "rollback" or "wait" (CANARY_ON_TIMEOUT).
    canary_on_timeout: str = "errors-only"
    # The version counter of the loop's version log (.soup-loops/versions.jsonl, see
    # docs/loop-formats.md): 1 is the served model, each promotion adds 1 and takes the next
    # number after `latest_version`, so a number is never reused after a rollback. Last, so
    # adding them shifted no existing positional field; a loop.yaml from before them is
    # numbered by `read_state` from the names it remembers.
    production_version: int = 1
    # Production's version before the last promotion: what `soup loop rollback` returns to
    # (None = nothing to go back to, or the served model when previous_production_name is None).
    previous_production_version: Optional[int] = None
    # The highest version ever promoted.
    latest_version: int = 1
    # The iteration that produced production_version, and the one before it (None = the
    # served model, an iteration this log does not know, or no log yet).
    production_iteration_id: Optional[str] = None
    previous_production_iteration_id: Optional[str] = None
    # The last line written to iterations.jsonl for the iteration whose canary is running (an
    # iteration line, outcome "canary"): the promotion or withdrawal appends the same snapshot
    # with its new outcome, so the later line is complete. None = no canary, or one that no
    # iteration started (`soup loop canary`, or a rollout from before this field).
    canary_iteration: Optional[dict] = None
    # True once the first version (the "initial" line) is in the version log.
    version_log_started: bool = False
    # When the loop trains sft and a feedback log exists, training starts once this many new
    # edits (real corrections newer than the data cursor) OR this many new rated answers
    # (any label) have arrived, whichever comes first (`soup loop init --min-new-edits` /
    # `--min-new-rated`). Without a feedback log, or for dpo / kto, `min_new_pairs` applies.
    # Last, so adding them shifted no existing positional field.
    min_new_edits: int = 8
    min_new_rated: int = 30
    # The sft recipe overrides (`soup loop init --sft-*`): None = DEFAULT_SFT_RECIPE; a dict
    # names only the keys it overrides (SFT_RECIPE_KEYS), the rest keep the constant's value.
    # Only the pre-wired train stage reads it, and only for train_method "sft". Last, so adding
    # it shifted no existing positional field; a loop.yaml from before it loads with None.
    sft_recipe: Optional[dict] = None
    # Where the pre-wired train stage asks for paraphrases of the prompts of the edits it
    # trains on (`soup loop init --paraphraser-endpoint` / `--paraphraser-model`): an
    # OpenAI-compatible base URL (loopback or LAN) and the model name sent to it. Both set or
    # both None; None = the loop's own serve endpoint (SOUP_LOOP_SERVE_ENDPOINT). Last, so
    # adding them shifted no existing positional field; a loop.yaml from before them loads
    # with None.
    paraphraser_endpoint: Optional[str] = None
    paraphraser_model: Optional[str] = None
    # The data cursor above (``harvest_cursor_ts``) follows the version in PRODUCTION: it moves
    # when the candidate that used the rows is promoted, and ``soup loop rollback`` returns to the
    # previous production's. A candidate that is thrown away (DON'T SHIP, a withdrawn or rolled
    # back canary) leaves its rows new. A loop.yaml from before takes its cursor as production's.
    # Three more cursors and a note complete it. All last, so adding them shifted no existing
    # positional field; a loop.yaml from before them loads with None.
    # The newest feedback a training run has read, promoted or not: the trigger counts what is new
    # since the later of this and the data cursor, so a rejected run is not retrained every poll.
    attempt_cursor_ts: Optional[float] = None
    # The cursor of the candidate now in canary: it becomes the data cursor when it is promoted
    # (None = no canary, or one no iteration started: its promotion leaves the cursor alone).
    canary_cursor_ts: Optional[float] = None
    # The data cursor before the last promotion; what ``soup loop rollback`` returns to.
    previous_harvest_cursor_ts: Optional[float] = None
    # The plan-time wait of the latest cycle that waited ("Waiting: 4/5 trainable corrections
    # ..."), for ``soup loop status``; None once the loop trains or has nothing to wait for.
    waiting_note: Optional[str] = None
    # A candidate that trained and passed the check but whose deploy was refused (serve
    # restarting, 401, 409 ...): the next cycles retry ONLY the deploy, up to
    # ``DEPLOY_RETRY_LIMIT`` times in all. A dict with "train" and "gate" (the stage outputs
    # the deploy needs), "attempts", "cursor_ts" and "iteration_id"; None = nothing pending.
    # Last, so adding it shifted no existing positional field; loads as None from before.
    pending_deploy: Optional[dict] = None
    # Why the latest cycle's training or deploy failed (exit code and stderr tail, or the
    # deploy's refusal), for ``soup loop status``; None once a cycle went through.
    last_failure: Optional[str] = None
    # True once ``--canary-on-timeout`` was given on the command line. A canary that got NO
    # answers by the time its wait is over is withdrawn (reason ``timeout_no_answers``) unless
    # this is set and the policy is ``errors-only``, which is then taken to mean "promote on
    # request errors alone, even with nothing to judge by". A loop.yaml from before it loads
    # as False. Last, so adding it shifted no existing positional field.
    canary_on_timeout_explicit: bool = False

    def __post_init__(self) -> None:  # noqa: D401 — validator hook
        # Closed-allowlist enforcement + bool-as-int rejection mirror the
        # project's v0.30/v0.34/v0.50 policy. `replace(...)` (immutable)
        # is the only sanctioned mutation path; this validator runs at
        # construction time so a hand-rolled instance still gets checks.
        _require_str("served_model", self.served_model)
        _require_str("eval_suite", self.eval_suite)
        _require_str("baseline", self.baseline)
        if self.status not in LOOP_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(LOOP_STATUSES)}, got {self.status!r}"
            )
        for fname in (
            "traces_collected",
            "pairs_distilled",
            "runs_gated",
            "adapters_shipped",
            "iteration_count",
            "runs_today",
        ):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{fname} must be a non-negative int, got {v!r}")
        if self.canary_traffic_pct is not None:
            v = self.canary_traffic_pct
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("canary_traffic_pct must be numeric or None")
            if not (0.0 <= float(v) <= 100.0):
                raise ValueError("canary_traffic_pct must be in [0, 100]")
        if self.monthly_budget_usd is not None:
            v = self.monthly_budget_usd
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError("monthly_budget_usd must be >= 0 or None")
        v = self.spent_this_month_usd
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError("spent_this_month_usd must be >= 0")
        if self.max_runs_per_day is not None:
            v = self.max_runs_per_day
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("max_runs_per_day must be a positive int or None")
        if not isinstance(self.canary_autoroll_on_regress, bool):
            raise ValueError("canary_autoroll_on_regress must be bool")
        if not isinstance(self.pre_wired, bool):
            raise ValueError("pre_wired must be bool")
        if self.canary_active is not None:
            _require_str("canary_active", self.canary_active, allow_empty=False)
        if self.thumbs_db is not None:
            _require_str("thumbs_db", self.thumbs_db, allow_empty=False)
        if self.base_model is not None:
            _require_str("base_model", self.base_model, allow_empty=False)
        if self.trace_dir is not None:
            _require_str("trace_dir", self.trace_dir, allow_empty=False)
        if self.verify is not None and self.verify not in VERIFY_METHODS:
            raise ValueError(
                f"verify must be one of {list(VERIFY_METHODS)}, got {self.verify!r}"
            )
        if self.production_adapter is not None:
            _require_str("production_adapter", self.production_adapter, allow_empty=False)
        v = self.deploy_traffic_pct
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not (0.0 < float(v) <= 100.0)
        ):
            raise ValueError("deploy_traffic_pct must be in (0, 100]")
        for fname in (
            "canary_adapter",
            "production_name",
            "previous_production_name",
            "previous_production_adapter",
        ):
            if getattr(self, fname) is not None:
                _require_str(fname, getattr(self, fname), allow_empty=False)
        if self.harvest_cursor_ts is not None:
            v = self.harvest_cursor_ts
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not (0 <= v <= MAX_CURSOR_TS)  # False for NaN too
            ):
                raise ValueError(
                    "harvest_cursor_ts must be epoch seconds in "
                    f"[0, {int(MAX_CURSOR_TS)}] or None, got {v!r}"
                )
        for fname in ("attempt_cursor_ts", "canary_cursor_ts", "previous_harvest_cursor_ts"):
            v = getattr(self, fname)
            if v is None:
                continue
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not (0 <= v <= MAX_CURSOR_TS)  # False for NaN too
            ):
                raise ValueError(
                    f"{fname} must be epoch seconds in [0, {int(MAX_CURSOR_TS)}] or None, "
                    f"got {v!r}"
                )
        if self.waiting_note is not None:
            _require_str("waiting_note", self.waiting_note, allow_empty=False)
            if any(ord(char) < 0x20 or ord(char) == 0x7F for char in self.waiting_note):
                raise ValueError("waiting_note must not contain control characters")
        v = self.min_new_pairs
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"min_new_pairs must be a positive int, got {v!r}")
        for fname in ("min_new_edits", "min_new_rated"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{fname} must be a positive int, got {v!r}")
        if self.feedback_log is not None:
            _require_str("feedback_log", self.feedback_log, allow_empty=False)
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(
                f"train_method must be one of {list(TRAIN_METHODS)}, got {self.train_method!r}"
            )
        if self.canary_started_ts is not None:
            v = self.canary_started_ts
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not (0 <= v <= MAX_CURSOR_TS)  # False for NaN too
            ):
                raise ValueError(
                    "canary_started_ts must be epoch seconds in "
                    f"[0, {int(MAX_CURSOR_TS)}] or None, got {v!r}"
                )
        v = self.canary_max_wait_sec
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not (0 < v <= MAX_CANARY_MAX_WAIT_SEC)  # False for NaN too
        ):
            raise ValueError(
                "canary_max_wait_sec must be seconds in "
                f"(0, {int(MAX_CANARY_MAX_WAIT_SEC)}], got {v!r}"
            )
        if self.canary_on_timeout not in CANARY_ON_TIMEOUT:
            raise ValueError(
                f"canary_on_timeout must be one of {list(CANARY_ON_TIMEOUT)}, "
                f"got {self.canary_on_timeout!r}"
            )
        if not isinstance(self.canary_on_timeout_explicit, bool):
            raise ValueError("canary_on_timeout_explicit must be bool")
        if self.canary_rollout_id is not None:
            _require_str("canary_rollout_id", self.canary_rollout_id, allow_empty=False)
        if self.last_iteration_id is not None:
            _require_str("last_iteration_id", self.last_iteration_id, allow_empty=False)
        if self.last_run_date is not None:
            _require_str("last_run_date", self.last_run_date, allow_empty=False)
        validate_sft_recipe(self.sft_recipe)
        for fname in ("paraphraser_endpoint", "paraphraser_model"):
            if getattr(self, fname) is not None:
                _require_str(fname, getattr(self, fname), allow_empty=False)
                if any(ord(char) < 0x20 or ord(char) == 0x7F for char in getattr(self, fname)):
                    raise ValueError(f"{fname} must not contain control characters")
        self._validate_versions()

    def _validate_versions(self) -> None:
        """The version counter, the iteration ids and the canary's iteration snapshot."""
        for fname in ("production_version", "latest_version"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{fname} must be a positive int, got {v!r}")
        v = self.previous_production_version
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise ValueError(
                f"previous_production_version must be a positive int or None, got {v!r}"
            )
        behind = [self.production_version]
        if self.previous_production_version is not None:
            behind.append(self.previous_production_version)
        if self.latest_version < max(behind):
            raise ValueError(
                f"latest_version must be at least {max(behind)} (the highest version in "
                f"use), got {self.latest_version}"
            )
        for fname in ("production_iteration_id", "previous_production_iteration_id"):
            if getattr(self, fname) is not None:
                _require_iteration_id(fname, getattr(self, fname))
        if not isinstance(self.version_log_started, bool):
            raise ValueError("version_log_started must be bool")
        if self.canary_iteration is not None:
            _check_canary_iteration(self.canary_iteration)
        if self.pending_deploy is not None:
            _check_pending_deploy(self.pending_deploy)
        if self.last_failure is not None:
            _require_str("last_failure", self.last_failure, allow_empty=False)

    def stable_name(self) -> str:
        """The canary router's stable side: the promoted adapter, else the served model."""
        return self.production_name or self.served_model

    def effective_sft_recipe(self) -> dict:
        """The sft recipe this loop trains with: ``DEFAULT_SFT_RECIPE`` under ``sft_recipe``."""
        return effective_sft_recipe(self.sft_recipe)

    def to_dict(self) -> Mapping[str, object]:
        """Stable, JSON-serialisable view (returned as ``MappingProxyType``)."""
        return MappingProxyType(asdict(self))

    def with_status(self, status: str) -> "LoopState":
        """Return a copy with ``status`` set + ``updated_at`` refreshed."""
        if status not in LOOP_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(LOOP_STATUSES)}, got {status!r}"
            )
        return replace(self, status=status, updated_at=_utc_now_iso())

    def bumped(self, **counters: int) -> "LoopState":
        """Return a copy with named counters incremented atomically."""
        updates = {}
        for k, v in counters.items():
            if k not in {
                "traces_collected",
                "pairs_distilled",
                "runs_gated",
                "adapters_shipped",
                "iteration_count",
                "runs_today",
            }:
                raise ValueError(f"unknown counter: {k}")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} delta must be a non-negative int")
            current = getattr(self, k)
            updates[k] = current + v
        updates["updated_at"] = _utc_now_iso()
        return replace(self, **updates)


def _require_str(field: str, value: object, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > _MAX_STR_FIELD:
        raise ValueError(f"{field} exceeds {_MAX_STR_FIELD} characters")


_ITERATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
#: The snapshot of a canary's iteration is one line of iterations.jsonl; far under this.
_MAX_CANARY_ITERATION_CHARS = 32 * 1024


def _require_iteration_id(field: str, value: object) -> None:
    """``value`` is an iteration id: safe as a directory name, as the formats require."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if _ITERATION_ID_RE.fullmatch(value) is None or ".." in value:
        raise ValueError(
            f"{field} must be 1-128 letters, digits, '.', '_' or '-' (not starting with a "
            f"symbol, no '..'), got {value[:40]!r}"
        )


def _check_canary_iteration(snapshot: object) -> None:
    """``canary_iteration`` is plain, small JSON: an iteration line whose outcome is canary."""
    if not isinstance(snapshot, dict):
        raise ValueError("canary_iteration must be a dict or None")
    try:
        body = json.dumps(snapshot, allow_nan=False)
        plain = json.loads(body) == snapshot  # an int key or a tuple is not plain JSON
    except (TypeError, ValueError) as exc:
        raise ValueError("canary_iteration must hold plain finite JSON values") from exc
    if not plain:
        raise ValueError("canary_iteration must hold plain finite JSON values with string keys")
    if len(body) > _MAX_CANARY_ITERATION_CHARS:
        raise ValueError(f"canary_iteration exceeds {_MAX_CANARY_ITERATION_CHARS} characters")
    if "iteration_id" not in snapshot:
        raise ValueError("canary_iteration must name its iteration_id")
    _require_iteration_id("canary_iteration iteration_id", snapshot["iteration_id"])
    if snapshot.get("outcome") != "canary":
        raise ValueError("canary_iteration must be an iteration whose outcome is canary")


_MAX_PENDING_DEPLOY_CHARS = 32 * 1024


def _check_pending_deploy(pending: object) -> None:
    """``pending_deploy`` is plain, small JSON: the stage outputs a retried deploy needs."""
    if not isinstance(pending, dict):
        raise ValueError("pending_deploy must be a dict or None")
    try:
        body = json.dumps(pending, allow_nan=False)
        plain = json.loads(body) == pending
    except (TypeError, ValueError) as exc:
        raise ValueError("pending_deploy must hold plain finite JSON values") from exc
    if not plain:
        raise ValueError("pending_deploy must hold plain finite JSON values with string keys")
    if len(body) > _MAX_PENDING_DEPLOY_CHARS:
        raise ValueError(f"pending_deploy exceeds {_MAX_PENDING_DEPLOY_CHARS} characters")
    for key in ("train", "gate"):
        if not isinstance(pending.get(key), dict):
            raise ValueError(f"pending_deploy must hold a {key} dict")
    attempts = pending.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 100:
        raise ValueError("pending_deploy attempts must be an int from 1 to 100")


def _brief(value: object) -> str:
    """``repr(value)`` cut to 60 characters, for a message that quotes untrusted input."""
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _check_target_modules(value: object) -> None:
    rule = (
        f'target_modules must be "{ALL_LINEAR}" or a non-empty list of up to '
        f"{_MAX_TARGET_MODULES} module names (letters, digits, '_' and '.')"
    )
    if value is None or (isinstance(value, str) and value == ALL_LINEAR):
        return
    if not isinstance(value, list) or not value or len(value) > _MAX_TARGET_MODULES:
        raise ValueError(f"{rule}, got {_brief(value)}")
    for name in value:
        if isinstance(name, str) and name == ALL_LINEAR:
            # peft only expands the string; as a list entry it is a module that does not exist.
            raise ValueError(f'{rule}; "{ALL_LINEAR}" is a string, not a list entry')
        if (
            not isinstance(name, str)
            or len(name) > _MAX_MODULE_NAME_LEN
            or _MODULE_NAME_RE.fullmatch(name) is None
        ):
            raise ValueError(f"{rule}, got the entry {_brief(name)}")


def _check_sft_quantization(value: object) -> None:
    if not isinstance(value, str) or value not in SFT_QUANTIZATIONS:
        options = ", ".join(f'"{name}"' for name in SFT_QUANTIZATIONS)
        raise ValueError(f"quantization must be one of {options}, got {_brief(value)}")


def _check_sft_int(key: str, value: object, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= high):
        raise ValueError(f"{key} must be an int in [1, {high}], got {_brief(value)}")


def _check_sft_seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= _SFT_SEED_MAX):
        raise ValueError(f"seed must be an int in [0, {_SFT_SEED_MAX}], got {_brief(value)}")


def _check_sft_lr(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (0 < value <= 1)  # False for NaN too
    ):
        raise ValueError(f"lr must be a number in (0, 1], got {_brief(value)}")


_SFT_RECIPE_CHECKS = {
    "target_modules": _check_target_modules,
    "lora_r": lambda value: _check_sft_int("lora_r", value, _SFT_LORA_R_MAX),
    "lora_alpha": lambda value: _check_sft_int("lora_alpha", value, _SFT_LORA_ALPHA_MAX),
    "lr": _check_sft_lr,
    "epochs": lambda value: _check_sft_int("epochs", value, _SFT_EPOCHS_MAX),
    "batch_size": lambda value: _check_sft_int("batch_size", value, _SFT_BATCH_MAX),
    "gradient_accumulation_steps": lambda value: _check_sft_int(
        "gradient_accumulation_steps", value, _SFT_ACCUM_MAX
    ),
    "seed": _check_sft_seed,
    "quantization": _check_sft_quantization,
}


def _unknown_recipe_keys(names: list) -> ValueError:
    noun = "key" if len(names) == 1 else "keys"
    return ValueError(
        f"unknown sft_recipe {noun} {', '.join(_brief(name) for name in names)}; "
        f"allowed: {', '.join(sorted(SFT_RECIPE_KEYS))}"
    )


def validate_sft_recipe_value(key: str, value: object) -> None:
    """Raise ``ValueError`` unless ``value`` is valid for the sft recipe key ``key``.

    ``target_modules``: ``None`` (the schema default), ``"all-linear"`` or a non-empty list of
    module names; ``lora_r`` an int in 1..1024; ``lora_alpha`` an int in 1..4096; ``lr`` a
    finite number above 0 and at most 1; ``epochs`` an int in 1..100; ``batch_size`` and
    ``gradient_accumulation_steps`` an int in 1..1024; ``seed`` an int in 0..2**32-1;
    ``quantization`` one of ``"4bit"``, ``"none"``, ``"auto"``. A bool is no int and no number.
    The message starts with the key (the command adds the flag).
    """
    check = _SFT_RECIPE_CHECKS.get(key) if isinstance(key, str) else None
    if check is None:
        raise _unknown_recipe_keys([key])
    check(value)


def validate_sft_recipe(recipe: object) -> Optional[dict]:
    """Return ``recipe`` if it is ``None`` or a dict of valid sft recipe keys, else raise.

    Unknown keys are refused (all of them named), then each value is checked by
    ``validate_sft_recipe_value``. A dict names only the keys it overrides.
    """
    if recipe is None:
        return None
    if not isinstance(recipe, dict):
        raise ValueError(f"sft_recipe must be a dict or None, got {type(recipe).__name__}")
    unknown = [key for key in recipe if key not in SFT_RECIPE_KEYS]
    if unknown:
        raise _unknown_recipe_keys(unknown)
    for key, value in recipe.items():
        validate_sft_recipe_value(key, value)
    return recipe


def effective_sft_recipe(recipe: Optional[Mapping[str, object]] = None) -> dict:
    """``DEFAULT_SFT_RECIPE`` with the keys ``recipe`` names replaced; a fresh dict.

    ``recipe`` is validated by ``LoopState`` already; a ``None`` there is the constant, a
    partial dict overrides only its own keys (an explicit ``target_modules: None`` too).
    """
    merged = {**DEFAULT_SFT_RECIPE, **(recipe or {})}
    if isinstance(merged["target_modules"], list):
        merged["target_modules"] = list(merged["target_modules"])
    return merged


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_WAIT_UNITS = {"m": 60, "h": 3600, "d": 86400}
# ASCII digits only: str.isdigit() and \d accept other scripts' digits (and superscripts).
_WAIT_RE = re.compile(r"([0-9]+)([mhd]?)", re.ASCII)
_MAX_WAIT_DIGITS = 12  # more cannot be within the 30-day cap; never build a huge int
_WAIT_FORMS = "use <int>m, <int>h or <int>d (for example 90m, 24h, 7d) or plain seconds"


def parse_wait_duration(raw: object) -> float:
    """Seconds from ``"90m"`` / ``"24h"`` / ``"7d"`` / ``"3600"`` (plain seconds).

    Whole numbers only, an optional surrounding space and any case; zero, a sign, a fraction,
    anything else, and more than 30 days raise ``ValueError`` (the message does not name the
    flag; the command adds it).
    """
    if not isinstance(raw, str):
        raise ValueError(f"{raw!r} is not a valid wait: {_WAIT_FORMS}")
    match = _WAIT_RE.fullmatch(raw.strip().lower())
    if match is None:
        raise ValueError(f"{raw!r} is not a valid wait: {_WAIT_FORMS}")
    digits, unit = match.groups()
    if len(digits.lstrip("0")) > _MAX_WAIT_DIGITS:
        seconds = float("inf")
    else:
        seconds = float(int(digits) * _WAIT_UNITS.get(unit, 1))
    if seconds == 0:
        raise ValueError(f"{raw.strip()!r} must be above 0: {_WAIT_FORMS}")
    if seconds > MAX_CANARY_MAX_WAIT_SEC:
        raise ValueError(f"{raw.strip()!r} must be at most 30d")
    return seconds


def format_wait_duration(seconds: float) -> str:
    """``86400`` -> ``"24h"``: the largest of d (two days or more), h, m that divides it."""
    whole = int(seconds)
    if whole != seconds:
        return f"{seconds:g}s"
    if whole % 86400 == 0 and whole >= 2 * 86400:
        return f"{whole // 86400}d"
    if whole % 3600 == 0:
        return f"{whole // 3600}h"
    if whole % 60 == 0:
        return f"{whole // 60}m"
    return f"{whole}s"


def default_state_path(cwd: Optional[str] = None) -> str:
    """Return the canonical state-file path under cwd."""
    base = cwd if cwd is not None else os.getcwd()
    return os.path.join(base, _DEFAULT_STATE_DIR, _DEFAULT_STATE_FILENAME)


def _check_path(path: str, *, allow_missing: bool) -> str:
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if not path:
        raise ValueError("path must not be empty")
    if "\x00" in path:
        raise ValueError("path must not contain NUL")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"path exceeds {_MAX_PATH_LEN} characters")
    if not is_under_cwd(path):
        raise ValueError(f"path {os.path.basename(path)!r} must stay under cwd")
    # Direct lstat — no lexists guard — closes the TOCTOU window where a
    # symlink could be planted between the existence check and the stat.
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if not allow_missing:
            raise FileNotFoundError(
                f"state file not found: {os.path.basename(path)}"
            ) from None
        return path
    except OSError as exc:
        raise ValueError(f"path unreadable: {type(exc).__name__}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("path must not be a symlink (TOCTOU defence)")
    return path


def _replace_file(tmp: str, target: str) -> None:
    """``os.replace``, retried briefly on Windows, where a reader that has the target open (a
    ``soup loop status`` at that instant, a virus scanner) makes the replace fail."""
    for attempt in range(40):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 39:
                raise
            time.sleep(0.025)


def write_state(state: LoopState, path: Optional[str] = None) -> str:
    """Persist ``state`` to ``path`` (default: ``./.soup/loop.yaml``) under the state lock.

    The write is atomic (below) and serialised with every other writer through
    :func:`state_lock`. It writes ``state`` AS IT IS: a caller that read the file earlier
    and may be stale uses :func:`update_state` (read-modify-write) or :func:`merge_state`.
    The target is validated before the lock is taken, so a bad path never leaves a lock file.
    """
    if not isinstance(state, LoopState):
        raise TypeError(f"state must be LoopState, got {type(state).__name__}")
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    with state_lock(target):
        return _write_state_unlocked(state, target)


def _write_state_unlocked(state: LoopState, path: Optional[str] = None) -> str:
    """Atomically persist ``state`` to ``path``; the caller holds :func:`state_lock`.

    Uses ``tempfile.mkstemp`` + ``os.replace`` so a SIGKILL mid-write
    cannot leave a torn file at the target — matches v0.43.0 Part D
    / v0.55.0 ``lock_suite`` / v0.57.0 atomic-write policy.
    """
    if not isinstance(state, LoopState):
        raise TypeError(f"state must be LoopState, got {type(state).__name__}")
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    payload = dict(state.to_dict())
    if not payload.get("created_at"):
        payload["created_at"] = _utc_now_iso()
    payload["updated_at"] = _utc_now_iso()
    body = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
    if len(body) > _MAX_FILE_BYTES:
        raise ValueError("state payload exceeds 1 MiB cap")
    fd, tmp = tempfile.mkstemp(prefix=".loop_state_", dir=parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        _replace_file(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        if os.name == "posix":
            os.chmod(target, 0o600)
    except OSError:
        pass
    return target


#: How long a writer waits for the state lock before it gives up (seconds). The lock is held
#: for one read and one small write, so a wait this long means a stuck holder.
STATE_LOCK_TIMEOUT_SEC = 30.0
#: A lock file nobody can read a pid from is taken for abandoned once it is this old (seconds).
_UNPARSABLE_LOCK_GRACE_SEC = 5.0
_LOCK_POLL_SEC = 0.02


def _lock_holder(lock_path: str) -> "tuple[Optional[int], bool]":
    """``(pid, abandoned)`` of a lock file: abandoned = dead pid, or none readable for long."""
    try:
        with open(lock_path, "rb") as handle:
            raw = handle.read(512)
        pid = json.loads(raw.decode("utf-8")).get("pid")
    except FileNotFoundError:
        return None, False
    except (OSError, ValueError, AttributeError):
        pid = None
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        return pid, not process_is_alive(pid)
    try:
        age = time.time() - os.stat(lock_path).st_mtime
    except OSError:
        return None, False
    return None, age >= _UNPARSABLE_LOCK_GRACE_SEC


def _try_kernel_lock(fd: int) -> bool:
    """Take a non-blocking exclusive OS lock on ``fd``; ``False`` if another handle holds it.

    A kernel lock dies with its holder (the process ends, the handle closes), so unlike a lock
    file it can never be left behind and never needs a staleness judgement.
    """
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _kernel_unlock(fd: int) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _take_over_stale(lock_path: str) -> bool:
    """Remove ``lock_path`` if it is STILL abandoned, one waiter at a time.

    Two waiters can both judge the same dead holder's lock abandoned. If each then unlinked it,
    the second would delete the first one's NEW lock and both would enter the critical section.
    The takeover is therefore serialised by a kernel lock on ``<lock>.takeover`` (never removed:
    deleting a file other processes have open would split the lock in two), and the holder is
    judged AGAIN once the guard is held: a lock the previous winner has already replaced is alive
    and is left alone. Returns ``False`` (back off and retry) when another waiter holds the
    guard, ``True`` once this waiter has done the check (whether or not it removed anything).
    """
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    try:
        guard = os.open(lock_path + ".takeover", flags, 0o600)
    except OSError:
        return False
    try:
        if not _try_kernel_lock(guard):
            return False
        try:
            _, still_abandoned = _lock_holder(lock_path)
            if still_abandoned:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
        finally:
            _kernel_unlock(guard)
        return True
    finally:
        os.close(guard)


@contextmanager
def state_lock(path: Optional[str] = None, *, timeout: float = STATE_LOCK_TIMEOUT_SEC):
    """Hold the exclusive lock on ``loop.yaml`` (``<path>.lock``, created with ``O_EXCL``).

    Every writer takes it (watch, rollback, pause, resume, canary, init), so two of them never
    interleave their read-modify-write. The lock file holds the owner's pid: one whose process is
    gone (or that nobody can read for ``_UNPARSABLE_LOCK_GRACE_SEC``) is removed and taken over.
    Raises ``TimeoutError`` naming the lock file after ``timeout`` seconds. Not re-entrant.
    """
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    lock_path = target + ".lock"
    deadline = time.monotonic() + timeout
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        try:
            fd = os.open(lock_path, flags, 0o600)
            break
        except (FileExistsError, PermissionError) as exc:
            # Windows answers "access denied" (not "exists") for a lock file whose deletion is
            # still pending because a waiter has it open; that is a lock in use too.
            if isinstance(exc, PermissionError) and os.name != "nt":
                raise
            holder, abandoned = _lock_holder(lock_path)
            if abandoned and _take_over_stale(lock_path):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"loop.yaml is locked by another process (pid {holder}); lock file "
                    f"{os.path.basename(lock_path)} - remove it if that process is gone"
                ) from None
            time.sleep(_LOCK_POLL_SEC)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps({"pid": os.getpid()}).encode("utf-8"))
        yield
    finally:
        for _attempt in range(100):
            try:
                os.unlink(lock_path)
                break
            except PermissionError:
                time.sleep(0.01)  # Windows: a waiter is reading the lock file this instant
            except OSError:
                break


def update_state(mutate: Callable[[LoopState], LoopState], path: Optional[str] = None) -> LoopState:
    """Read ``loop.yaml`` fresh under the lock, apply ``mutate`` to it, write it, return it.

    The read-modify-write of a command: what another writer changed since the command began is
    kept, and ``mutate`` sets only the fields the command owns (``replace(fresh, ...)``).
    """
    target = path if path is not None else default_state_path()
    with state_lock(target):
        updated = mutate(read_state(target))
        _write_state_unlocked(updated, target)
        return updated


#: Fields no iteration owns: ``write_state`` refreshes them on every persist.
_UNOWNED_FIELDS = frozenset({"created_at", "updated_at"})


class StateMergeError(ValueError):
    """``loop.yaml`` cannot be read, so an iteration's changes cannot be merged onto it."""


def merge_state(
    base: LoopState,
    new: LoopState,
    path: Optional[str] = None,
    *,
    skipped: Optional[list] = None,
) -> LoopState:
    """Persist what an iteration changed (``base`` -> ``new``) onto the file's CURRENT contents.

    An iteration can run for hours; ``soup loop rollback`` / ``pause`` / ``canary`` may change
    ``loop.yaml`` meanwhile, and writing the iteration's copy back would undo them. Under the
    lock this re-reads the file and applies ONLY the fields ``new`` changed relative to ``base``
    (the ones the iteration owns). A field both changed takes the iteration's value, except
    ``status``: a pause or stop made meanwhile wins over the iteration's copy.

    The iteration's copy is NEVER written as it is. If all the changes together do not make a
    valid ``LoopState`` next to what the file holds now, they are applied one field at a time
    and the fields that do not fit are left out (their names go into ``skipped``, if given).
    If the file is gone or unreadable, ``StateMergeError`` is raised and nothing is written.
    """
    target = path if path is not None else default_state_path()
    with state_lock(target):
        changes = {
            item.name: getattr(new, item.name)
            for item in dataclass_fields(LoopState)
            if item.name not in _UNOWNED_FIELDS
            and getattr(new, item.name) != getattr(base, item.name)
        }
        try:
            fresh = read_state(target)
        except (FileNotFoundError, ValueError) as exc:
            raise StateMergeError(
                f"loop.yaml cannot be read ({exc}); the iteration's changes were not written"
            ) from exc
        if fresh.status != base.status:
            changes.pop("status", None)
        try:
            merged = replace(fresh, **changes)
        except ValueError:
            merged, pending = fresh, dict(changes)
            progress = True
            while pending and progress:  # a field that needs another one applied first retries
                progress = False
                for name in list(pending):
                    try:
                        merged = replace(merged, **{name: pending[name]})
                    except ValueError:
                        continue
                    del pending[name]
                    progress = True
            if skipped is not None:
                skipped.extend(pending)
        _write_state_unlocked(merged, target)
        return merged


def read_state(path: Optional[str] = None) -> LoopState:
    """Load a ``LoopState`` from disk."""
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=False)
    try:
        size = os.path.getsize(target)
    except OSError as exc:
        raise ValueError(f"state file unreadable: {type(exc).__name__}") from exc
    if size > _MAX_FILE_BYTES:
        raise ValueError("state file exceeds 1 MiB cap")
    with open(target, "rb") as fh:
        raw = fh.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"state file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("state file root must be a JSON object")
    # Drop keys we don't recognise so a forward-compat dump can still load
    # backwards (the alternative is mass-rejecting on any unknown field).
    allowed = set(LoopState.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    if "production_version" not in data:
        _number_versions_of_old_state(filtered)
    try:
        return LoopState(**filtered)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state file has invalid contents: {exc}") from exc


def _number_versions_of_old_state(fields: dict) -> None:
    """Give a ``loop.yaml`` written before the version counter the numbers its names imply.

    The served model is version 1; the production adapter, if any, is at least version 2,
    and a remembered previous adapter means one more promotion before it. The file keeps no
    count, so this is the lowest number consistent with what it remembers.
    """
    version = 1
    if fields.get("production_name"):
        version += 1
        if fields.get("previous_production_name"):
            version += 1
        fields["previous_production_version"] = version - 1
    fields["production_version"] = version
    fields["latest_version"] = version


def init_state(
    served_model: str,
    eval_suite: str,
    baseline: str,
    *,
    monthly_budget_usd: Optional[float] = None,
    max_runs_per_day: Optional[int] = None,
    pre_wired: bool = False,
    thumbs_db: Optional[str] = None,
    base_model: Optional[str] = None,
    trace_dir: Optional[str] = None,
    verify: Optional[str] = None,
    production_adapter: Optional[str] = None,
    deploy_traffic_pct: float = 5.0,
    harvest_cursor_ts: Optional[float] = None,
    min_new_pairs: int = 1,
    feedback_log: Optional[str] = None,
    train_method: str = "dpo",
    canary_max_wait_sec: float = DEFAULT_CANARY_MAX_WAIT_SEC,
    canary_on_timeout: str = "errors-only",
    canary_on_timeout_explicit: bool = False,
    min_new_edits: int = 8,
    min_new_rated: int = 30,
    sft_recipe: Optional[dict] = None,
    paraphraser_endpoint: Optional[str] = None,
    paraphraser_model: Optional[str] = None,
    path: Optional[str] = None,
    force: bool = False,
) -> Tuple[LoopState, str]:
    """Create the loop.yaml state file. Refuses to clobber unless ``force``."""
    target = path if path is not None else default_state_path()
    _check_path(target, allow_missing=True)
    # Use lstat (not exists) — defends against a planted symlink between
    # the existence check and the write. `_check_path` already rejected
    # symlinks so any stat-success here is a real regular-file collision.
    try:
        os.lstat(target)
        present = True
    except FileNotFoundError:
        present = False
    except OSError as exc:
        raise ValueError(f"state path unreadable: {type(exc).__name__}") from exc
    if present and not force:
        raise FileExistsError(
            f"loop state already exists at {os.path.basename(target)} "
            "(re-run with --force to overwrite)"
        )
    now = _utc_now_iso()
    state = LoopState(
        served_model=served_model,
        eval_suite=eval_suite,
        baseline=baseline,
        status="stopped",
        pre_wired=pre_wired,
        thumbs_db=thumbs_db,
        base_model=base_model,
        trace_dir=trace_dir,
        verify=verify,
        production_adapter=production_adapter,
        deploy_traffic_pct=deploy_traffic_pct,
        harvest_cursor_ts=harvest_cursor_ts,
        min_new_pairs=min_new_pairs,
        feedback_log=feedback_log,
        train_method=train_method,
        canary_max_wait_sec=canary_max_wait_sec,
        canary_on_timeout=canary_on_timeout,
        canary_on_timeout_explicit=canary_on_timeout_explicit,
        min_new_edits=min_new_edits,
        min_new_rated=min_new_rated,
        sft_recipe=sft_recipe,
        paraphraser_endpoint=paraphraser_endpoint,
        paraphraser_model=paraphraser_model,
        monthly_budget_usd=monthly_budget_usd,
        max_runs_per_day=max_runs_per_day,
        created_at=now,
        updated_at=now,
    )
    write_state(state, target)
    return state, target
