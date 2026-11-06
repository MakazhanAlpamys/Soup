"""Pre-wired production stages for `soup loop watch` (v0.71.4 #176).

v0.58.0 shipped the watch daemon with no-op default stage callbacks; this
module supplies real harvest / train / gate / deploy callables that compose
the existing v0.26.0 trace-to-preference, eval-gate, and v0.30.0 multi-adapter
deploy surfaces. Operators opt in via ``soup loop init --pre-wired`` (or
``soup loop watch --pre-wired``).

Each callable matches the ``HarvestFn`` / ``TrainFn`` / ``GateFn`` /
``DeployFn`` protocols from ``loop_daemon``. Heavy deps (transformers / peft /
trl / httpx) are lazy-imported inside the callables so importing this module
stays cold-start cheap (project lazy-import policy).

Module-level escape hatches (mirrors v0.53.1 #109 deploy_measure) let tests
and offline smokes inject behaviour without a GPU or a network:

- ``_TRACE_DIR_RESOLVER(state) -> Optional[str]``
- ``_TRAIN_RUNNER(argv, ...)``               (a ``subprocess.run`` stand-in)
- ``_GATE_GENERATE_FACTORY(adapter_dir) -> generate_fn``
- ``_DEPLOY_POSTER(endpoint, name) -> bool``
- ``_SHIP_RUNNER(argv, ...)``                (a ``subprocess.run`` stand-in)
- ``_LOAD_POSTER(endpoint, name, path) -> (status, detail)``
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Tuple

from soup_cli.utils.loop_feedback import (
    count_new,
    good_answer_row,
    harvest_feedback,
    is_new,
    newest_rated_ts,
    newest_ts,
    prompt_key,
    read_quality_stats,
    unique_sft,
    without_edited,
)
from soup_cli.utils.loop_state import VERIFY_METHODS, LoopState
from soup_cli.utils.net_guard import LOOPBACK_HOSTS as _LOOPBACK_HOSTS
from soup_cli.utils.net_guard import parse_ip_literal
from soup_cli.utils.paths import atomic_write_text, quote_path

if TYPE_CHECKING:
    from soup_cli.data.traces.pair_builder import PreferencePair
    from soup_cli.data.traces.parsers import Trace
    from soup_cli.utils.canary_router import QualityStats
    from soup_cli.utils.loop_daemon import WatchConfig

_LOG = logging.getLogger(__name__)

_PAIRS_DIR = ".soup-loops/pairs"
_ADAPTERS_DIR = ".soup-loops/adapters"
#: The trace directory read when neither SOUP_LOOP_TRACE_DIR nor trace_dir is set.
DEFAULT_TRACE_DIR = ".soup-loops/traces"
#: The feedback log read when neither SOUP_LOOP_FEEDBACK_LOG nor feedback_log is set.
DEFAULT_FEEDBACK_LOG = ".soup-loops/feedback.jsonl"
# #1440 — `source` tag on pairs harvested from the `--record-thumbs` database.
_THUMBS_SOURCE = "soup-serve-thumbs"

# Test / smoke escape hatches (default None → real behaviour).
_TRACE_DIR_RESOLVER: Optional[Callable[[LoopState], Optional[str]]] = None
_TRAIN_RUNNER: Optional[Callable[..., object]] = None
_GATE_GENERATE_FACTORY: Optional[Callable[[str], Callable[[str], str]]] = None
_DEPLOY_POSTER: Optional[Callable[[str, str], bool]] = None
_SHIP_RUNNER: Optional[Callable[..., object]] = None
_LOAD_POSTER: Optional[Callable[[str, str, str], Tuple[int, str]]] = None
#: Bytes of a /v1/adapters/load or /deactivate reply read (post_capped): the
#: answer is a short status object, and a larger one counts as no answer.
_SERVE_REPLY_LIMIT_BYTES = 64 * 1024


# ---------------------------------------------------------------------------
# Harvest — production traces → preference pairs (v0.26.0 trace-to-pref)
# ---------------------------------------------------------------------------


def trace_dir_source(state: LoopState) -> Tuple[str, str]:
    """Return ``(trace directory, where it came from)``.

    ``SOUP_LOOP_TRACE_DIR`` overrides ``trace_dir`` from ``loop.yaml``
    (``soup loop init --trace-dir``), as for the thumbs database; with
    neither, ``DEFAULT_TRACE_DIR``. The source is the field name, or
    ``"default"``.
    """
    env = os.environ.get("SOUP_LOOP_TRACE_DIR")
    if env:
        return env, "SOUP_LOOP_TRACE_DIR"
    if state.trace_dir:
        return state.trace_dir, "trace_dir"
    return DEFAULT_TRACE_DIR, "default"


def _resolve_trace_dir(state: LoopState) -> Optional[str]:
    """Return the directory of ``*.jsonl`` serve logs to harvest, or ``None``.

    The directory is ``trace_dir_source``'s, never a guess: the served model
    path used to be tried too, so a model directory that happened to hold
    ``*.jsonl`` files (a dataset next to the weights) was harvested as serve
    logs. ``None`` when the directory does not exist (yet).

    Note: this is the one read surface in this module that intentionally does
    NOT enforce cwd-containment — operators legitimately keep serve logs
    outside the project dir (e.g. ``/var/log/soup-serve``). The directory is
    operator-supplied config (env / state), not untrusted input, and harvested
    content is only used to build DPO pairs (never exec'd).
    """
    if _TRACE_DIR_RESOLVER is not None:
        return _TRACE_DIR_RESOLVER(state)
    path, _ = trace_dir_source(state)
    return path if os.path.isdir(path) else None


def trace_dir_notice(state: LoopState) -> Optional[str]:
    """Check the trace directory before a pre-wired watch starts.

    Raises ``ValueError`` naming where the path came from when it exists but
    is not a directory. Returns a warning when a configured directory does
    not exist yet; the default directory may be absent silently. Otherwise
    ``None``.
    """
    path, field = trace_dir_source(state)
    if os.path.isdir(path):
        return None
    if os.path.exists(path):
        where = f"{DEFAULT_TRACE_DIR} (the default)" if field == "default" else field
        raise ValueError(f"{where}: {quote_path(path)} is not a directory")
    if field == "default":
        return None
    return (
        f"{field}: no conversation logs at {quote_path(path)} yet; "
        f"soup serve --trace-log {quote_path(os.path.join(path, 'serve.jsonl'))} writes them"
    )


def feedback_log_source(state: LoopState) -> Tuple[str, str]:
    """Return ``(feedback log path, where it came from)``.

    ``SOUP_LOOP_FEEDBACK_LOG`` overrides ``feedback_log`` from ``loop.yaml``
    (``soup loop init --feedback-log``), as for the trace directory; with neither,
    ``DEFAULT_FEEDBACK_LOG``. The source is the env variable, the field name, or
    ``"default"``.
    """
    env = os.environ.get("SOUP_LOOP_FEEDBACK_LOG")
    if env:
        return env, "SOUP_LOOP_FEEDBACK_LOG"
    if state.feedback_log:
        return state.feedback_log, "feedback_log"
    return DEFAULT_FEEDBACK_LOG, "default"


def _resolve_feedback_log(state: LoopState) -> Optional[str]:
    """The feedback log to read, or ``None`` when it does not exist (yet)."""
    path, _ = feedback_log_source(state)
    return path if os.path.isfile(path) else None


def feedback_log_notice(state: LoopState) -> Optional[str]:
    """Check the feedback log before a pre-wired watch starts.

    Raises ``ValueError`` naming where the path came from when it exists but is not a
    file (a directory). Returns a warning when a configured file does not exist yet; the
    default may be absent silently, since nothing has been given feedback then. Otherwise
    ``None``.
    """
    path, field = feedback_log_source(state)
    if os.path.isfile(path):
        return None
    if os.path.exists(path):
        where = f"{DEFAULT_FEEDBACK_LOG} (the default)" if field == "default" else field
        what = "a directory, not a file" if os.path.isdir(path) else "not a regular file"
        raise ValueError(f"{where}: {quote_path(path)} is {what}")
    if field == "default":
        return None
    return (
        f"{field}: no feedback log at {quote_path(path)} yet; "
        "the loop reads it once feedback lines are written there"
    )


def _resolve_thumbs_db(state: LoopState) -> Optional[Tuple[str, str]]:
    """Return ``(path, field)`` for the thumbs database, or ``None``.

    Same precedence as the trace directory: ``SOUP_LOOP_THUMBS_DB`` env, then
    ``thumbs_db`` from ``loop.yaml`` (``soup loop init --thumbs-db``). The
    field name travels with the path so a refusal names where it came from.
    """
    env = os.environ.get("SOUP_LOOP_THUMBS_DB")
    if env:
        return env, "SOUP_LOOP_THUMBS_DB"
    if state.thumbs_db:
        return state.thumbs_db, "thumbs_db"
    return None


def checked_thumbs_db(state: LoopState) -> Optional[str]:
    """Return the configured thumbs database, checked, or ``None`` (#1440).

    The harvest reads a database only when told to, like ``soup serve
    --record-thumbs`` itself. A configured path that leaves cwd or is a
    symlink (the rule ``--record-thumbs`` applies on the serve side), or that
    is not an existing, readable thumbs database, raises ``ValueError`` naming
    its field instead of harvesting a silent 0. A database that is merely
    locked (a running serve) passes. ``soup loop watch`` runs this before it
    starts.
    """
    resolved = _resolve_thumbs_db(state)
    if resolved is None:
        return None
    path, field = resolved
    from soup_cli.utils.local_rl import THUMBS_HARVEST_SELECT, validate_db_path

    try:
        validate_db_path(path)
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        if "under cwd" in msg or "symlink" in msg:
            msg += "; move the database under the project directory"
        raise ValueError(f"{field}: {msg}") from exc
    if not os.path.isfile(path):
        raise ValueError(
            f"{field}: no database file at {quote_path(path)}; "
            f"soup serve --record-thumbs {quote_path(path)} creates it"
        )
    try:
        # Short timeout: a lock is not a verdict, so do not wait out sqlite's 5 s.
        with contextlib.closing(sqlite3.connect(os.path.abspath(path), timeout=0.1)) as conn:
            conn.execute(f"{THUMBS_HARVEST_SELECT} LIMIT 1")
    except sqlite3.Error as exc:
        if "locked" in str(exc).lower():
            return path
        raise ValueError(
            f"{field}: {quote_path(path)} is not a readable thumbs database "
            f"({type(exc).__name__}); soup serve --record-thumbs creates one"
        ) from exc
    return path


def _pair_key(pair: "PreferencePair") -> Tuple[str, str, str]:
    return (pair.prompt, pair.chosen, pair.rejected)


def _newer(left: Optional[float], right: Optional[float]) -> Optional[float]:
    """The larger of two optional timestamps."""
    if left is None:
        return right
    return left if right is None or left >= right else right


def _thumbs_pairs(
    state: LoopState,
) -> "Tuple[list[PreferencePair], set[Tuple[str, str, str]], Optional[float]]":
    """``(pairs, keys of the new ones, newest ts)`` from the ``--record-thumbs`` database (#1440).

    New means newer than ``state.harvest_cursor_ts`` (``harvest_dpo_pairs_since``).
    """
    path = checked_thumbs_db(state)
    if path is None:
        return [], set(), None
    from soup_cli.data.traces.pair_builder import PreferencePair
    from soup_cli.utils.local_rl import harvest_dpo_pairs_since

    try:
        found, new, newest = harvest_dpo_pairs_since(path, state.harvest_cursor_ts)
    except sqlite3.Error as exc:
        # Passed the check, then failed while reading (lock, corruption).
        raise ValueError(
            f"{_resolve_thumbs_db(state)[1]}: {quote_path(path)} could not be read "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    pairs = [
        PreferencePair(
            prompt=p.prompt, chosen=p.chosen, rejected=p.rejected, source=_THUMBS_SOURCE
        )
        for p in found
    ]
    new_keys = {(p.prompt, p.chosen, p.rejected) for p in new}
    return pairs, new_keys, newest


def _trace_pairs(
    traces: "list[Trace]", cursor: Optional[float]
) -> "Tuple[list[PreferencePair], set[Tuple[str, str, str]], Optional[float]]":
    """``(pairs, keys of the new ones, newest ts)`` from serve-log traces.

    Every trace builds pairs. A trace is NEW when its ``ts`` is later than
    ``cursor``; a trace without a ``ts`` is new only while there is no cursor
    (else it would count as new forever). A pair is new when a trace it was built
    from is: the thumbs-up whose output is the chosen answer, or the thumbs-down
    whose output is the rejected one, of the same prompt. The newest ts covers
    those traces only, so a lone thumbs-up does not move the cursor.
    """
    from soup_cli.data.traces.pair_builder import build_pairs

    pairs = list(build_pairs(traces, signal="thumbs_up"))
    by_feedback: "dict[Tuple[str, str, str], list[Trace]]" = {}
    for trace in traces:
        if trace.signal in ("thumbs_up", "thumbs_down"):
            by_feedback.setdefault((trace.prompt, trace.signal, trace.output), []).append(trace)
    new_keys: "set[Tuple[str, str, str]]" = set()
    newest: Optional[float] = None
    for pair in pairs:
        used = by_feedback.get((pair.prompt, "thumbs_up", pair.chosen), []) + by_feedback.get(
            (pair.prompt, "thumbs_down", pair.rejected), []
        )
        for trace in used:
            if trace.ts is not None:
                newest = _newer(newest, trace.ts)
            if cursor is None or (trace.ts is not None and trace.ts > cursor):
                new_keys.add(_pair_key(pair))
    return pairs, new_keys, newest


def _trace_good_answers(
    traces: "list[Trace]",
) -> "list[Tuple[str, str, Optional[float]]]":
    """``(prompt, answer, ts)`` of the serve-trace answers whose newest label is a thumbs-up.

    Traces are labelled one line at a time: for one prompt and answer the newest
    ``thumbs_up`` / ``thumbs_down`` line in file order decides, so an up followed by a down is
    not good. A down alone gives nothing; lines with any other signal are not labels.
    """
    newest: "dict[Tuple[str, str], Tuple[str, Optional[float]]]" = {}
    for trace in traces:
        if trace.signal in ("thumbs_up", "thumbs_down"):
            newest[(trace.prompt, trace.output)] = (trace.signal, trace.ts)
    return [
        (prompt, output, ts)
        for (prompt, output), (signal, ts) in newest.items()
        if signal == "thumbs_up"
    ]


def _thumbs_good_answers(state: LoopState) -> "list[Tuple[str, str, Optional[float]]]":
    """``(prompt, answer, ts)`` of the answers whose newest thumb in the database is ``up``."""
    path = checked_thumbs_db(state)
    if path is None:
        return []
    from soup_cli.utils.local_rl import harvest_up_answers

    try:
        return list(harvest_up_answers(path))
    except sqlite3.Error as exc:
        raise ValueError(
            f"{_resolve_thumbs_db(state)[1]}: {quote_path(path)} could not be read "
            f"({type(exc).__name__}: {exc})"
        ) from exc


#: File name prefix of the harvested dataset, per training method.
_DATASET_PREFIX = {"dpo": "pairs", "kto": "kto", "sft": "sft"}
#: What ``harvest_from_traces`` reports as the rule that decides whether enough is new.
_RULE_MIN_NEW_PAIRS = "min_new_pairs"
_RULE_EDITS_OR_RATED = "edits_or_rated"


def harvest_from_traces(state: LoopState) -> Mapping[str, object]:
    """Serve traces, recorded thumbs and the feedback log → the dataset of ``train_method``.

    Three datasets come out of one harvest (see ``loop_feedback`` for the feedback rows):

    - ``dpo``: preference pairs from ``soup serve --trace-log`` directories
      (thumbs/regeneration signals) and, when one is configured, the ``soup serve
      --record-thumbs`` database (#1440), unioned without exact duplicates, plus a pair
      for every answer a person corrected in the feedback log (chosen = the edited text);
    - ``kto``: rows labelled good or bad from the feedback log's thumbs, ratings and edits;
    - ``sft``: the corrected answers of the feedback log and every unedited answer whose
      newest label is good: a thumb up or a rating of 4-5 in the feedback log, an ``up``
      thumb in the database, a ``thumbs_up`` serve trace. An answer that was only rated bad
      gives no row. The same prompt and answer count once, and an answer a person corrected
      in the feedback log is never trained on as it was (an edit wins over every thumb).

    The feedback log (``feedback_log_source``) joins the conversation records of the
    trace directory by id. The traces and the thumbs database feed ``dpo`` and ``sft``.

    Writes the dataset of ``state.train_method`` to ``.soup-loops/pairs/<uuid>.jsonl``
    (cwd-contained) and returns ``{pairs_harvested, new_pairs, cursor_ts, pairs_path,
    traces_collected, thumbs_pairs, train_method, dpo_pairs, kto_rows, sft_rows, new_edits,
    new_rated, new_data_rule, feedback_unknown_response, feedback_lines_skipped}``.
    ``thumbs_pairs`` counts the pairs READ from the database, before the union drops any that
    a trace already supplied; ``dpo_pairs`` / ``kto_rows`` / ``sft_rows`` are the sizes of the
    three datasets; ``feedback_unknown_response`` counts feedback lines about an answer no
    conversation record has (ignored); ``feedback_lines_skipped`` the log lines that could not
    be used. ``traces_collected`` counts the answers in the trace directory (version 0 and 1
    lines). Yields zero rows (no path) when the selected dataset is empty; a configured
    database that is refused or missing raises (see ``checked_thumbs_db``).

    The data cursor (``state.harvest_cursor_ts``) decides what is NEW, not what is
    trained on: ``pairs_harvested`` and the dataset file hold ALL rows of the method, since
    every iteration trains a fresh LoRA from the base model. ``new_pairs`` counts the rows
    that arrived after the cursor. A database pair is new when the newer of its up and down
    rows is later than the cursor; a trace pair, when a trace it is built from is (a trace
    without a ts counts as new only while there is no cursor, see ``_trace_pairs``); a
    feedback row, when the newest feedback line it came from is. ``cursor_ts`` is the
    newest ts among the rows of the dataset, which ``run_once`` stores once a training run
    used them, or ``None``.

    Whether enough is new depends on the loop. For ``dpo`` and ``kto``, and for ``sft``
    without a feedback log, it is ``new_pairs >= state.min_new_pairs``
    (``new_data_rule`` ``"min_new_pairs"``). For ``sft`` WITH a feedback log it is
    ``new_edits >= state.min_new_edits`` OR ``new_rated >= state.min_new_rated``
    (``new_data_rule`` ``"edits_or_rated"``): ``new_edits`` counts the answers with a real
    edit newer than the cursor, ``new_rated`` the answers with any thumb, rating or real edit
    newer than the cursor (``loop_feedback.count_new``; both are reported for every method).
    Training consumes every rated answer, so ``cursor_ts`` then also covers answers that gave
    no row (a thumb down), else they would count as new for ever. The dataset file is not
    written while not enough is new: that cycle trains nothing, and a file per poll would
    pile up.
    """
    from soup_cli.data.traces.parsers import parse_soup_serve

    cursor = state.harvest_cursor_ts
    trace_dir = _resolve_trace_dir(state)
    traces = list(parse_soup_serve(trace_dir)) if trace_dir else []
    pairs, new_keys, newest = _trace_pairs(traces, cursor)
    thumbs, thumbs_new_keys, thumbs_newest = _thumbs_pairs(state)
    new_keys |= thumbs_new_keys
    newest = _newer(newest, thumbs_newest)
    seen = {_pair_key(p) for p in pairs}
    for p in thumbs:
        if _pair_key(p) not in seen:  # harvest_dpo_pairs yields one pair per prompt: no self-dupes
            pairs.append(p)
    # Every row as (the JSONL object, is it new).
    dpo_rows = [(p.to_jsonl_dict(), _pair_key(p) in new_keys) for p in pairs]
    dpo_newest = newest
    taken = {_pair_key(p) for p in pairs}

    feedback_path = _resolve_feedback_log(state)
    feedback = harvest_feedback(trace_dir, feedback_path)
    found = feedback.datasets
    for item in found.dpo:
        key = (prompt_key(item.row["prompt"]), item.row["chosen"], item.row["rejected"])
        if key in taken:  # the same pair, already supplied by a trace or the database
            continue
        dpo_rows.append((item.row, is_new(item.ts, cursor)))
        dpo_newest = _newer(dpo_newest, item.ts)
    # sft: the answers rated good in the traces and the database, then the feedback log's rows;
    # an answer the feedback log corrected is not an sft row from those stores.
    rated_good = [
        *(good_answer_row(prompt, answer, ts, "thumbs_up")
          for prompt, answer, ts in _trace_good_answers(traces)),
        *(good_answer_row(prompt, answer, ts, _THUMBS_SOURCE)
          for prompt, answer, ts in _thumbs_good_answers(state)),
    ]
    sft_items = unique_sft([*without_edited(rated_good, found.edited_originals), *found.sft])
    new_edits, new_rated = count_new(found.rated, cursor)
    feedback_rule = state.train_method == "sft" and feedback_path is not None
    sft_newest = newest_ts(sft_items)
    if feedback_rule:
        sft_newest = _newer(sft_newest, newest_rated_ts(found.rated))
    by_method = {
        "dpo": (dpo_rows, dpo_newest),
        "kto": ([(r.row, is_new(r.ts, cursor)) for r in found.kto], newest_ts(found.kto)),
        "sft": ([(r.row, is_new(r.ts, cursor)) for r in sft_items], sft_newest),
    }
    rows, selected_newest = by_method[state.train_method]
    new_rows = sum(1 for _, new in rows if new)
    common = {
        # parse_soup_serve reads version 0 and version 1 lines: each answer once.
        "traces_collected": len(traces),
        "thumbs_pairs": len(thumbs),
        "train_method": state.train_method,
        "dpo_pairs": len(dpo_rows),
        "kto_rows": len(found.kto),
        "sft_rows": len(sft_items),
        "new_edits": new_edits,
        "new_rated": new_rated,
        "new_data_rule": _RULE_EDITS_OR_RATED if feedback_rule else _RULE_MIN_NEW_PAIRS,
        "feedback_unknown_response": found.unknown_response,
        "feedback_lines_skipped": feedback.skipped_lines,
    }
    if not rows:
        return {
            "pairs_harvested": 0,
            "new_pairs": 0,
            "cursor_ts": None,
            "pairs_path": None,
            **common,
        }

    if feedback_rule:
        enough = new_edits >= state.min_new_edits or new_rated >= state.min_new_rated
    else:
        enough = new_rows >= state.min_new_pairs
    pairs_path = None
    if enough:
        pairs_dir = Path(_PAIRS_DIR)
        pairs_dir.mkdir(parents=True, exist_ok=True)
        prefix = _DATASET_PREFIX[state.train_method]
        pairs_path = str(pairs_dir / f"{prefix}-{uuid.uuid4().hex[:12]}.jsonl")
        lines = "\n".join(json.dumps(row) for row, _ in rows) + "\n"
        # Atomic + cwd-contained + symlink-rejected (mirrors the rest of the
        # changeset's write policy).
        atomic_write_text(lines, pairs_path, field="pairs_path")
    return {
        "pairs_harvested": len(rows),
        "new_pairs": new_rows,
        "cursor_ts": selected_newest,
        "pairs_path": pairs_path,
        **common,
    }


# ---------------------------------------------------------------------------
# Train — generated DPO YAML → `soup train` subprocess
# ---------------------------------------------------------------------------


_REGISTRY_SCHEME = "registry://"
# What `pack_iteration_as_can` records when it does not know the base.
_UNKNOWN_BASE = "unknown"
_BASE_HINT = "pass soup loop init --base-model <model> to name the model to train"
_MAX_ADAPTER_CONFIG_BYTES = 1024 * 1024


def _adapter_dir_base(served_model: str) -> Optional[str]:
    """``base_model_name_or_path`` of a local LoRA adapter directory.

    ``None`` when ``served_model`` is not a directory holding an
    ``adapter_config.json``. Raises ``ValueError`` when it is one but names no
    readable base: training from the adapter directory itself would start
    from the wrong model or fail deep inside model loading.
    """
    config_path = os.path.join(served_model, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    problem = f"{served_model}: its adapter_config.json"
    try:
        if os.path.getsize(config_path) > _MAX_ADAPTER_CONFIG_BYTES:
            raise ValueError(f"{problem} exceeds 1 MiB; {_BASE_HINT}")
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{problem} could not be read ({type(exc).__name__}); {_BASE_HINT}"
        ) from exc
    base = data.get("base_model_name_or_path") if isinstance(data, dict) else None
    if not isinstance(base, str) or not base.strip():
        raise ValueError(f"{problem} names no base_model_name_or_path; {_BASE_HINT}")
    return base


def resolve_base_model_source(
    served_model: str, base_model: Optional[str] = None
) -> Tuple[str, str]:
    """Return ``(base model, where it came from)``; see ``resolve_base_model``."""
    if base_model:
        return base_model, "--base-model"
    if not served_model.startswith(_REGISTRY_SCHEME):
        adapter_base = _adapter_dir_base(served_model)
        if adapter_base is not None:
            return adapter_base, f"adapter_config.json in {served_model}"
        return served_model, "the served model"
    ref = served_model[len(_REGISTRY_SCHEME):]
    hint = _BASE_HINT
    from soup_cli.registry.store import RegistryStore, _default_db_path

    # Checked before RegistryStore(), whose constructor creates the database.
    db_path = _default_db_path()
    if not db_path.is_file():
        raise ValueError(
            f"{served_model}: no registry database at {quote_path(str(db_path))} to look "
            f"the base model up in; {hint}"
        )
    try:
        with RegistryStore(db_path) as store:
            entry_id = store.resolve(ref)
            entry = store.get(entry_id) if entry_id else None
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise ValueError(
            f"{served_model}: the registry could not be read ({type(exc).__name__}: {exc}); {hint}"
        ) from exc
    if entry is None:
        raise ValueError(
            f"{served_model}: no registry entry matches {ref!r} (soup registry list shows them); "
            f"{hint}"
        )
    base = entry.get("base_model")
    if not isinstance(base, str) or not base.strip() or base == _UNKNOWN_BASE:
        raise ValueError(f"{served_model}: registry entry {entry_id} records no base model; {hint}")
    return base, f"registry entry {entry_id}"


def resolve_base_model(served_model: str, base_model: Optional[str] = None) -> str:
    """Return the model the pre-wired train stage starts from.

    ``base_model`` (``soup loop init --base-model`` / ``loop.yaml``) wins. A
    ``registry://<ref>`` served model resolves to its registry entry's
    ``base_model``; a local LoRA adapter directory to the
    ``base_model_name_or_path`` of its ``adapter_config.json``; any other
    served model is trained from itself. Raises ``ValueError`` naming why the
    base is unknown: the stage never falls back to a stand-in model, which
    used to be the tiny random test model.
    """
    return resolve_base_model_source(served_model, base_model)[0]


#: The ``data.format`` of the soup train config, per training method.
_TRAIN_FORMAT = {"dpo": "dpo", "kto": "kto", "sft": "chatml"}
#: What a row of each method's dataset is called in a skip note.
_ROW_NAME = {"dpo": "preference pairs", "kto": "kto rows", "sft": "sft rows"}
#: The sft recipe the loop trains with (dpo and kto keep 1 epoch at lr 5e-5).
_SFT_EPOCHS = 5
_SFT_LR = 1.0e-4
_SFT_LORA_R = 64
_SFT_LORA_ALPHA = 16


def _render_train_yaml(
    state: LoopState, data_path: str, output_dir: str, method: Optional[str] = None
) -> str:
    """Render a minimal ``soup.yaml`` for ``method`` (default: ``state.train_method``).

    The task is the method, the data the harvested dataset in the format ``soup train``
    reads for it (``dpo``, ``kto``, ``chatml`` for ``sft``). Raises ``ValueError`` when the
    base model cannot be determined (``resolve_base_model``).
    """
    import yaml

    method = method or state.train_method
    base = resolve_base_model(state.served_model, state.base_model)
    if method == "sft":
        # The recipe measured for the loop's sft: a higher lr, five passes over a small
        # dataset, and LoRA rank 64 with alpha 16 (written out, not left to the schema).
        training: dict = {
            "epochs": _SFT_EPOCHS,
            "batch_size": 1,
            "lr": _SFT_LR,
            "lora": {"r": _SFT_LORA_R, "alpha": _SFT_LORA_ALPHA},
        }
    else:
        # KTO needs a per-device batch of at least 2: TRL's KL term is degenerate at 1
        # and the config is refused (#1420).
        training = {"epochs": 1, "batch_size": 2 if method == "kto" else 1, "lr": 5.0e-5}
    config = {
        "base": base,
        "task": method,
        # val_split 0: a one- or two-row harvest would otherwise lose its only
        # training rows to the schema default 0.1 (#1909).
        "data": {"train": data_path, "format": _TRAIN_FORMAT[method], "val_split": 0},
        "training": training,
        "output": output_dir,
    }
    return yaml.safe_dump(config, sort_keys=False)


def _render_dpo_yaml(state: LoopState, pairs_path: str, output_dir: str) -> str:
    """Render a minimal DPO ``soup.yaml`` referencing the harvested pairs.

    Raises ``ValueError`` when the base model cannot be determined
    (``resolve_base_model``).
    """
    return _render_train_yaml(state, pairs_path, output_dir, "dpo")


def train_dpo_from_pairs(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Train an adapter on the harvested dataset via a ``soup train`` subprocess.

    The method is ``state.train_method``: ``dpo`` (the name is from when it was the only
    one), ``kto`` or ``sft``; ``_render_train_yaml`` writes the matching config. Returns
    ``{run_id, skipped, adapter_path}``. Skips (no run) when the
    harvest produced no rows, or not enough is NEW since the data cursor (see
    ``harvest_from_traces``): fewer new rows than ``state.min_new_pairs``, or, for an sft loop
    whose harvest reports the ``edits_or_rated`` rule (it has a feedback log), fewer than
    ``state.min_new_edits`` new edits AND fewer than ``state.min_new_rated`` new rated
    answers. It adds a ``notes`` reason
    (``no_preference_pairs`` / ``no_kto_rows`` / ``no_sft_rows`` with the count,
    ``not_enough_new_data`` with the counts, or ``pairs_file_missing``) that the loop saves
    in the iteration record (#1909). A harvest that reports no ``new_pairs`` (a custom
    ``harvest_fn``) has every row counted as new. The subprocess is argv-list (no shell);
    ``_TRAIN_RUNNER`` injects a stand-in for tests.
    """
    pairs_path = ctx.get("pairs_path")
    pairs_n = int(ctx.get("pairs_harvested", 0) or 0)
    new_n = int(ctx.get("new_pairs", pairs_n) or 0)
    method = state.train_method
    skip = {"run_id": None, "skipped": True, "adapter_path": None}
    if pairs_n <= 0:
        if method == "dpo":
            note = "no_preference_pairs: 0 valid pairs harvested, training skipped"
        else:
            note = f"no_{method}_rows: 0 valid {method} rows harvested, training skipped"
        return {**skip, "notes": note}
    if method == "sft" and ctx.get("new_data_rule") == _RULE_EDITS_OR_RATED:
        # An sft loop with a feedback log: new edits OR new rated answers, not new rows.
        new_edits = int(ctx.get("new_edits", 0) or 0)
        new_rated = int(ctx.get("new_rated", 0) or 0)
        if new_edits < state.min_new_edits and new_rated < state.min_new_rated:
            note = (
                f"not_enough_new_data: {new_edits} new edits (needs {state.min_new_edits}) and "
                f"{new_rated} new rated answers (needs {state.min_new_rated}) "
                "since the last training run"
            )
            return {**skip, "notes": note}
    elif new_n < state.min_new_pairs:
        note = (
            f"not_enough_new_data: {new_n} new {_ROW_NAME[method]} since the last training run, "
            f"training needs {state.min_new_pairs} (soup loop init --min-new-pairs)"
        )
        return {**skip, "notes": note}
    if not pairs_path:
        return {**skip, "notes": "pairs_file_missing: no pairs path recorded, training skipped"}
    if not os.path.isfile(str(pairs_path)):
        note = f"pairs_file_missing: {quote_path(str(pairs_path))} not found, training skipped"
        return {**skip, "notes": note}

    run_id = f"loop-train-{uuid.uuid4().hex[:8]}"
    output_dir = os.path.join(_ADAPTERS_DIR, run_id)
    try:
        rendered = _render_train_yaml(state, str(pairs_path), output_dir)
    except ValueError as exc:
        return {**skip, "notes": f"base_model_unknown: {exc}; training skipped"}
    yaml_dir = Path(_ADAPTERS_DIR)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(yaml_dir / f"{run_id}.yaml")
    atomic_write_text(rendered, yaml_path, field="train yaml")

    argv = [
        sys.executable,
        "-m",
        "soup_cli.cli",
        "train",
        "--config",
        yaml_path,
        "--yes",
    ]
    runner = _TRAIN_RUNNER
    if runner is None:
        import subprocess  # noqa: S404 — argv list mode, no shell

        runner = subprocess.run
    try:
        result = runner(  # noqa: S603 — internal argv, no shell
            argv, capture_output=True, timeout=24 * 3600, check=False
        )
    except Exception as exc:  # noqa: BLE001 — train failure must not crash loop
        _LOG.warning("train subprocess failed: %s", type(exc).__name__)
        return {"run_id": None, "skipped": True, "adapter_path": None, "failed": True}

    rc = getattr(result, "returncode", 1)
    if rc != 0:
        _LOG.warning("soup train exited rc=%s", rc)
        return {"run_id": None, "skipped": True, "adapter_path": None, "failed": True}
    return {"run_id": run_id, "skipped": False, "adapter_path": output_dir}


# ---------------------------------------------------------------------------
# Gate — eval-gate against the registered baseline (v0.26.0 Part B)
# ---------------------------------------------------------------------------


def _build_gate_generator(adapter_dir: str) -> Callable[[str], str]:
    """Build a ``generate_fn`` for the trained adapter (live model load)."""
    if _GATE_GENERATE_FACTORY is not None:
        return _GATE_GENERATE_FACTORY(adapter_dir)
    # Live default — load the merged/trained adapter as a full model dir.
    from soup_cli.eval.quant_check import make_model_generator

    return make_model_generator(adapter_dir)


def gate_against_baseline(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Run the v0.26.0 eval gate for the trained adapter vs the baseline.

    Returns ``{gate_verdict}`` — ``SKIPPED`` when training was skipped or the
    adapter dir is missing, otherwise ``OK`` / ``MAJOR`` from the gate result.
    """
    if ctx.get("skipped"):
        return {"gate_verdict": "SKIPPED"}
    adapter_path = ctx.get("adapter_path")
    if not adapter_path or not os.path.isdir(str(adapter_path)):
        return {"gate_verdict": "SKIPPED"}

    from soup_cli.eval.gate import load_suite, resolve_baseline, run_gate

    try:
        suite = load_suite(state.eval_suite)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _LOG.warning("gate suite load failed: %s", type(exc).__name__)
        return _verdict_skipped(f"the eval-gate suite could not be loaded ({_short(exc)})")
    # An unresolvable baseline degrades to a threshold-only gate (no
    # regression check) rather than failing the whole stage — the baseline
    # may not exist yet on the first iteration.
    try:
        baseline = resolve_baseline(state.baseline)
    except (FileNotFoundError, ValueError, TypeError, OSError):
        baseline = {}
    try:
        generate_fn = _build_gate_generator(str(adapter_path))
        result = run_gate(suite, generate_fn=generate_fn, baseline=baseline)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        _LOG.warning("gate run failed: %s", type(exc).__name__)
        return _verdict_skipped(f"the eval gate failed to run ({_short(exc)})")
    verdict = "OK" if result.passed else "MAJOR"
    return {"gate_verdict": verdict, "verdict_method": "eval_gate"}


# ---------------------------------------------------------------------------
# Verify — the check before rollout: a soup ship verdict (default) or the gate
# ---------------------------------------------------------------------------

_VERDICTS_DIR = ".soup-loops/verdicts"
_MAX_VERDICT_BYTES = 1024 * 1024
_MAX_LOCK_PROBE_BYTES = 16 * 1024 * 1024
_SHIP_TIMEOUT_SEC = 6 * 3600
_SHIP_DECISIONS = {"SHIP": "OK", "DON'T SHIP": "MAJOR"}


def _printable(text: str, limit: int = 300) -> str:
    """``text`` on one line with control characters removed, capped, for a note."""
    return "".join(ch if ch.isprintable() else " " for ch in text)[:limit]


def _short(exc: BaseException) -> str:
    """One line naming an error, for a note."""
    return _printable(f"{type(exc).__name__}: {exc}")


def _verdict_skipped(reason: str) -> Mapping[str, object]:
    """A SKIPPED verdict carrying its reason: the iteration does not roll out."""
    return {"gate_verdict": "SKIPPED", "notes": f"verdict_skipped: {reason}"}


def verify_method(state: LoopState) -> str:
    """``ship`` or ``eval_gate``; a ``loop.yaml`` from before ``verify`` ran the gate."""
    return state.verify or "eval_gate"


def _lock_design_hint(path: str) -> Optional[str]:
    """A refusal message when ``path`` is a ``soup eval lock`` design file."""
    try:
        if os.path.getsize(path) > _MAX_LOCK_PROBE_BYTES:
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or "dimensions" not in data:
        return None
    return (
        f"--eval {quote_path(path)} is a soup eval lock design file (dimensions), not "
        "runnable tasks; give --verify ship a task-eval JSONL (one {\"prompt\", "
        "\"expected\"} object per line) or --verify eval-gate an eval-gate suite YAML "
        "(suite + tasks)"
    )


def check_verify_eval(method: str, path: str) -> None:
    """Refuse an ``--eval`` file the chosen check cannot run, naming why.

    ``ship`` needs a task-eval JSONL under cwd (what ``soup ship --task-eval``
    accepts); ``eval_gate`` needs an eval-gate suite (``suite`` + ``tasks``).
    A ``soup eval lock`` design file is neither and gets its own message:
    the docs used to pass one, and the gate then skipped every iteration.
    """
    if method not in VERIFY_METHODS:
        raise ValueError(f"--verify must be ship or eval-gate, got {method!r}")
    if os.path.isfile(path):
        hint = _lock_design_hint(path)
        if hint is not None:
            raise ValueError(hint)
    if method == "ship":
        from soup_cli.eval.custom import load_eval_tasks
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        try:
            enforce_under_cwd_and_no_symlink(path, "--eval")
            tasks = load_eval_tasks(path)
        except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
            raise ValueError(
                f"--eval {quote_path(path)} is not a soup ship task-eval JSONL ({_short(exc)})"
            ) from exc
        if not tasks:
            raise ValueError(f"--eval {quote_path(path)} has no tasks")
        return
    from soup_cli.eval.gate import load_suite

    try:
        load_suite(path)
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        raise ValueError(
            f"--eval {quote_path(path)} is not an eval-gate suite ({_short(exc)})"
        ) from exc


def _read_ship_decision(path: str) -> Optional[str]:
    """The ``decision`` of a ``soup ship --output`` file, or ``None``."""
    try:
        if os.path.getsize(path) > _MAX_VERDICT_BYTES:
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    decision = data.get("decision") if isinstance(data, dict) else None
    return decision if decision in _SHIP_DECISIONS else None


def verify_with_ship(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Run ``soup ship`` for the trained adapter against the production version.

    The "before" side is the base model plus ``production_adapter`` when one
    is serving (``soup ship --base-adapter``), else the base model alone.
    SHIP maps to ``OK``, DON'T SHIP to ``MAJOR``; when no verdict file is
    written the result is ``SKIPPED`` with the reason, and nothing rolls out.
    The decision is read from ``--output``, never from the exit code: a
    usage error exits 2 like DON'T SHIP does.
    """
    adapter_path = ctx.get("adapter_path")
    if ctx.get("skipped") or not adapter_path:
        return {"gate_verdict": "SKIPPED", "verdict_method": "ship"}
    if not os.path.isdir(str(adapter_path)):
        return _verdict_skipped(f"the trained adapter {quote_path(str(adapter_path))} is missing")
    try:
        base = resolve_base_model(state.served_model, state.base_model)
    except ValueError as exc:
        return _verdict_skipped(f"base model unknown ({_short(exc)})")
    run_id = str(ctx.get("run_id") or f"verify-{uuid.uuid4().hex[:8]}")
    Path(_VERDICTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(_VERDICTS_DIR, f"{run_id}.json")
    argv = [
        sys.executable, "-m", "soup_cli.cli", "ship",
        "--base", base,
        "--adapter", str(adapter_path),
        "--task-eval", state.eval_suite,
        "--output", out_path,
    ]
    if state.production_adapter:
        argv.extend(["--base-adapter", state.production_adapter])
    baseline = (
        {"kind": "version", "ref": state.production_adapter}
        if state.production_adapter
        else {"kind": "base", "ref": base}
    )
    runner = _SHIP_RUNNER
    if runner is None:
        import subprocess  # noqa: S404 — argv list mode, no shell

        runner = subprocess.run
    common = {"verdict_method": "ship", "verdict_path": out_path, "verdict_baseline": baseline}
    try:
        result = runner(  # noqa: S603 — internal argv, no shell
            argv, capture_output=True, timeout=_SHIP_TIMEOUT_SEC, check=False
        )
    except Exception as exc:  # noqa: BLE001 — a verdict failure must not crash the loop
        return {**common, **_verdict_skipped(f"soup ship could not run ({_short(exc)})")}
    decision = _read_ship_decision(out_path)
    if decision is None:
        rc = getattr(result, "returncode", None)
        stderr = getattr(result, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        lines = [line.strip() for line in str(stderr).splitlines() if line.strip()]
        tail = f": {_printable(lines[-1])}" if lines else ""
        reason = f"soup ship wrote no verdict (exit {rc}){tail}"
        return {**common, **_verdict_skipped(reason)}
    return {**common, "gate_verdict": _SHIP_DECISIONS[decision], "verdict_decision": decision}


def verify_candidate(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """The pre-wired check before rollout: ``soup ship`` or the eval gate."""
    if verify_method(state) == "ship":
        return verify_with_ship(state, ctx)
    return gate_against_baseline(state, ctx)


# ---------------------------------------------------------------------------
# Deploy — promote to a canary via /v1/adapters/activate (v0.30.0)
# ---------------------------------------------------------------------------


# Adapter-name shape accepted by the v0.30.0 /v1/adapters/activate route.
_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


def _endpoint_is_local(endpoint: str) -> bool:
    """True iff ``endpoint``'s host is loopback or a private/link-local IP.

    The deploy surface must only POST to the operator's own box / LAN, never
    an arbitrary public host (the webhook validator permits any HTTPS host).
    Loopback hostnames are accepted by name; any other host must parse as a
    private / loopback / link-local IP literal. A non-IP public hostname is
    rejected (we do not resolve DNS — project policy).

    Deliberately narrower than ``net_guard.is_private_or_link_local``: this
    gates what counts as "the operator's own box / LAN" for an outbound
    POST, so reserved/multicast ranges stay rejected rather than being
    treated as trustworthy deploy targets (#616 review). Shares
    ``parse_ip_literal`` with the wider predicate for the abbreviated /
    decimal / hex / octal IPv4 parsing only.
    """
    from urllib.parse import urlparse

    host = (urlparse(endpoint).hostname or "").strip("[]").lower()
    if not host:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    ip = parse_ip_literal(host)
    if ip is None:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


#: Environment variable holding the ``soup serve --tool-auth-token`` value.
#: ``soup loop watch --tool-auth-token`` sets it, including for a ``--detach``
#: child, so the token never has to appear in argv (#1139).
TOOL_AUTH_TOKEN_ENV = "SOUP_TOOL_AUTH_TOKEN"


class _ActivateRefusedError(Exception):
    """The serve endpoint answered the activate POST with 401."""


def _post_activate(endpoint: str, name: str) -> bool:
    """POST to ``<endpoint>/v1/adapters/activate/<name>``; True on 2xx.

    ``endpoint`` is already SSRF-validated by the caller; ``name`` is
    re-validated here against the activate-route pattern before URL
    interpolation (defence-in-depth against a crafted adapter dir name).
    The tool token from ``SOUP_TOOL_AUTH_TOKEN``, when set, goes as a Bearer
    header: a server started with ``--tool-auth-token`` requires it on this
    route (#1139). A 401 raises ``_ActivateRefusedError`` so the caller can name
    the missing credential instead of reporting a bare failure.
    """
    if _DEPLOY_POSTER is not None:
        return _DEPLOY_POSTER(endpoint, name)
    if not _ADAPTER_NAME_RE.match(name):
        return False
    try:
        import httpx
    except ImportError:
        return False
    url = endpoint.rstrip("/") + f"/v1/adapters/activate/{name}"
    token = os.environ.get(TOOL_AUTH_TOKEN_ENV)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        # A plain-http endpoint (loopback only) ignores HTTP_PROXY / ALL_PROXY, which
        # would otherwise receive the Bearer token in clear text. An https endpoint keeps
        # trust_env: a proxy only sees a CONNECT tunnel there, and SSL_CERT_FILE /
        # SSL_CERT_DIR are how a TLS proxy's private CA gets trusted.
        plain_http = url.lower().startswith("http://")
        resp = httpx.post(url, headers=headers, timeout=5.0, trust_env=not plain_http)
    except Exception:  # noqa: BLE001 — deploy must never crash the loop
        return False
    if resp.status_code == 401:
        raise _ActivateRefusedError()
    return 200 <= resp.status_code < 300


def _post_load(endpoint: str, name: str, path: str) -> Tuple[int, str]:
    """POST ``{name, path}`` to ``<endpoint>/v1/adapters/load``.

    Returns ``(HTTP status, detail)``; status 0 when no response came back,
    or when the reply is longer than ``_SERVE_REPLY_LIMIT_BYTES`` (read
    through ``post_capped``, never whole). Same transport rules as
    ``_post_activate`` (the tool token as a Bearer header, no environment
    proxy for plain http). ``_LOAD_POSTER`` replaces
    the call in tests; the older ``_DEPLOY_POSTER(endpoint, name) -> bool``
    hook still works (True = loaded).
    """
    if _LOAD_POSTER is not None:
        return _LOAD_POSTER(endpoint, name, path)
    if _DEPLOY_POSTER is not None:
        return (200, "") if _DEPLOY_POSTER(endpoint, name) else (0, "")
    if not _ADAPTER_NAME_RE.match(name):
        return 0, "invalid adapter name"
    try:
        import httpx  # noqa: F401 -- post_capped sends through it
    except ImportError:
        return 0, "httpx is not installed"
    url = endpoint.rstrip("/") + "/v1/adapters/load"
    token = os.environ.get(TOOL_AUTH_TOKEN_ENV)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        from soup_cli.utils.http_response import post_capped

        resp = post_capped(
            url,
            max_bytes=_SERVE_REPLY_LIMIT_BYTES,
            endpoint_label="SOUP_LOOP_SERVE_ENDPOINT /v1/adapters/load",
            headers=headers,
            json={"name": name, "path": path},
            timeout=300.0,
            trust_env=not url.lower().startswith("http://"),
        )
    except Exception as exc:  # noqa: BLE001 — deploy must never crash the loop
        return 0, type(exc).__name__
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            detail = _printable(body["detail"], 200)
    except ValueError:
        pass
    return resp.status_code, detail


def _load_refused_note(status: int, detail: str) -> str:
    """Name why the serve endpoint did not load the adapter."""
    if status == 401:
        if os.environ.get(TOOL_AUTH_TOKEN_ENV):
            return (
                "adapter load refused (401): the serve endpoint rejected the tool token; "
                "check soup loop watch --tool-auth-token / SOUP_TOOL_AUTH_TOKEN against "
                "soup serve --tool-auth-token"
            )
        return (
            "adapter load refused (401): the serve endpoint requires its tool token; "
            "pass soup loop watch --tool-auth-token or set SOUP_TOOL_AUTH_TOKEN"
        )
    if status == 404:
        return (
            "adapter load refused (404): the serve endpoint does not load adapters at "
            "runtime; start soup serve with --adapter-dir .soup-loops/adapters"
        )
    if status == 403:
        return "adapter load refused (403): the adapter is outside the server's --adapter-dir"
    if status == 0:
        return f"adapter load failed: no response from the serve endpoint ({detail or 'error'})"
    return f"adapter load failed (HTTP {status}{': ' + detail if detail else ''})"


def _checked_endpoint() -> Tuple[Optional[str], str]:
    """``(SSRF-checked SOUP_LOOP_SERVE_ENDPOINT, "")`` or ``(None, why not)``."""
    raw_endpoint = os.environ.get("SOUP_LOOP_SERVE_ENDPOINT")
    if not raw_endpoint:
        return None, "no SOUP_LOOP_SERVE_ENDPOINT configured"
    from soup_cli.utils.drift_alarm import validate_webhook_url

    try:
        endpoint = validate_webhook_url(raw_endpoint, allow_private_hosts=True)
    except (TypeError, ValueError):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT rejected by SSRF guard")
        return None, "SOUP_LOOP_SERVE_ENDPOINT rejected (SSRF guard)"
    if not _endpoint_is_local(endpoint):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)")
        return None, "SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)"
    return endpoint, ""


def deploy_to_canary(
    state: LoopState, ctx: Mapping[str, object]
) -> Mapping[str, object]:
    """Load the trained adapter into the running server and start a canary.

    When the verdict is OK: POST the adapter to ``/v1/adapters/load`` on
    ``SOUP_LOOP_SERVE_ENDPOINT`` (``soup serve --adapter-dir``, tool token
    required), then return the canary fields the daemon writes to
    ``loop.yaml``: the new adapter gets ``deploy_traffic_pct`` (default 5%)
    of keyed traffic, everything else stays on production. Nothing is
    activated for unkeyed traffic; that happens on promotion. Returns
    ``{deployed, canary_verdict, notes}`` plus ``canary_active``,
    ``canary_adapter``, ``canary_traffic_pct``, ``canary_rollout_id`` and
    ``canary_started_ts`` (epoch seconds) on success. The endpoint is SSRF-validated
    (loopback/LAN only) before any POST.
    """
    if ctx.get("gate_verdict") != "OK":
        return {"deployed": False, "canary_verdict": None}
    adapter_path = ctx.get("adapter_path")
    if not adapter_path:
        return {"deployed": False, "canary_verdict": None}
    if state.canary_active:
        return {
            "deployed": False,
            "canary_verdict": None,
            "notes": f"canary {state.canary_active} is still running; not starting another",
        }
    raw_endpoint = os.environ.get("SOUP_LOOP_SERVE_ENDPOINT")
    if not raw_endpoint:
        return {
            "deployed": False,
            "canary_verdict": None,
            "notes": "no SOUP_LOOP_SERVE_ENDPOINT configured",
        }
    from soup_cli.utils.drift_alarm import validate_webhook_url

    try:
        # A serve endpoint legitimately targets the operator's own LAN box, so
        # opt into private hosts here; the deploy surface is then tightened to
        # loopback/LAN by `_endpoint_is_local` below. The default webhook policy
        # (allow_private_hosts=False) still blocks private IPs for --slack-url /
        # --discord-url, where the URL is not operator-trusted.
        endpoint = validate_webhook_url(raw_endpoint, allow_private_hosts=True)
    except (TypeError, ValueError):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT rejected by SSRF guard")
        return {
            "deployed": False,
            "canary_verdict": None,
            "failed": True,
            "notes": "SOUP_LOOP_SERVE_ENDPOINT rejected (SSRF guard)",
        }
    # A serve endpoint is the operator's own box / LAN, never an arbitrary
    # remote host. `validate_webhook_url` (a *webhook* policy) permits any
    # HTTPS host; tighten further for the deploy surface to loopback + RFC1918
    # private + link-local only, so a poisoned SOUP_LOOP_SERVE_ENDPOINT cannot
    # POST adapter names to an attacker-controlled HTTPS server (v0.71.4
    # review MEDIUM-4). Non-IP hostnames can't be verified private without DNS
    # (project policy: no DNS resolution), so the safe default rejects them.
    if not _endpoint_is_local(endpoint):
        _LOG.warning("SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)")
        return {
            "deployed": False,
            "canary_verdict": None,
            "failed": True,
            "notes": "SOUP_LOOP_SERVE_ENDPOINT must be loopback/LAN (SSRF guard)",
        }
    name = os.path.basename(str(adapter_path).rstrip("/\\")) or "canary"
    status, detail = _post_load(endpoint, name, os.path.abspath(str(adapter_path)))
    # 409 "already loaded": a re-run of this deploy; the adapter is there.
    if not (200 <= status < 300 or (status == 409 and "already loaded" in detail)):
        note = _load_refused_note(status, detail)
        _LOG.warning(note)
        return {"deployed": False, "canary_verdict": None, "failed": True, "notes": note}
    pct = float(state.deploy_traffic_pct)
    # The verdict comes later, from the bucket stats soup serve records.
    return {
        "deployed": True,
        "canary_verdict": None,
        "canary_active": name,
        "canary_adapter": str(adapter_path),
        "canary_traffic_pct": pct,
        "canary_rollout_id": uuid.uuid4().hex,
        # Answers given before this moment say nothing about the rollout (the quality
        # verdict, ``canary_quality``).
        "canary_started_ts": time.time(),
        "notes": f"canary {name}: {pct:g}% of keyed traffic",
    }


def canary_quality(state: LoopState) -> "Optional[QualityStats]":
    """The feedback-based side of a canary verdict, or ``None`` when it takes no part.

    Counts, per bucket, the answers of the active rollout that people rated and the share
    that got negative feedback (``loop_feedback.read_quality_stats``; the daemon turns it
    into a verdict and combines it with the request-error verdict). It takes part only
    when there is something to read and the rollout can be told from earlier ones: the
    feedback log exists (a loop that gets no feedback keeps judging by errors alone, as
    before it existed) and ``canary_started_ts`` was recorded with the canary (an older
    ``loop.yaml``, or a canary started by hand before this field, has none).
    """
    if not state.canary_active or state.canary_started_ts is None:
        return None
    feedback_path = _resolve_feedback_log(state)
    if feedback_path is None:
        return None
    return read_quality_stats(
        _resolve_trace_dir(state),
        feedback_path,
        stable=state.stable_name(),
        canary=state.canary_active,
        started_ts=state.canary_started_ts,
        stable_is_base=state.production_name is None,
    )


def _post_deactivate(endpoint: str) -> bool:
    """POST ``<endpoint>/v1/adapters/deactivate`` (back to the served model).

    True on 2xx. The reply is read through ``post_capped`` (at most
    ``_SERVE_REPLY_LIMIT_BYTES``); a longer one counts as no answer.
    """
    from soup_cli.utils.http_response import post_capped

    url = endpoint.rstrip("/") + "/v1/adapters/deactivate"
    token = os.environ.get(TOOL_AUTH_TOKEN_ENV)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = post_capped(
            url,
            max_bytes=_SERVE_REPLY_LIMIT_BYTES,
            endpoint_label="SOUP_LOOP_SERVE_ENDPOINT /v1/adapters/deactivate",
            headers=headers,
            timeout=5.0,
            trust_env=not url.lower().startswith("http://"),
        )
    except Exception:  # noqa: BLE001 — a rollback reports, it never crashes
        return False
    if resp.status_code == 401:
        raise _ActivateRefusedError()
    return 200 <= resp.status_code < 300


def switch_served_adapter(name: Optional[str]) -> Tuple[bool, str]:
    """Point the server's unkeyed traffic at ``name`` (None = the served model).

    ``(True, "")`` on success or when no ``SOUP_LOOP_SERVE_ENDPOINT`` is
    configured (the note then says the server was not told);
    ``(False, why)`` when the server refused or did not answer.
    """
    endpoint, why = _checked_endpoint()
    if endpoint is None:
        if why.startswith("no SOUP_LOOP_SERVE_ENDPOINT"):
            return True, "the server was not told (no SOUP_LOOP_SERVE_ENDPOINT)"
        return False, why
    try:
        ok = _post_activate(endpoint, name) if name else _post_deactivate(endpoint)
    except _ActivateRefusedError:
        return False, "the serve endpoint refused the tool token (401)"
    return (True, "") if ok else (False, "the serve endpoint did not switch the adapter")


def promote_canary(state: LoopState) -> Mapping[str, object]:
    """Make the passing canary the production version for unkeyed traffic too."""
    ok, note = switch_served_adapter(state.canary_active)
    return {"promoted": ok, "notes": note}


# ---------------------------------------------------------------------------
# Cost — placeholder estimate (v0.34.0 run_cost wiring is a follow-up)
# ---------------------------------------------------------------------------


def estimate_cost(state: LoopState) -> float:
    """Per-iteration cost estimate (USD) for the v0.58 budget gate (#245).

    Wires the cost gate to v0.34 ``run_cost.estimate_run_cost_usd`` using the
    most-recent *completed* run's GPU + duration as a forward estimate for the
    next iteration. ``cost_fn`` runs *pre-flight* (before this iteration
    trains, so the budget gate can refuse a run before spending) — and a
    pre-wired loop trains the same model on the same box repeatedly, so the
    previous run's actual GPU-time is the best forward signal.

    Falls back to ``0.0`` (no cost gate) when there is no prior priced run:
    the first iteration has no data, and a CPU / unpriced GPU returns ``None``
    from ``estimate_run_cost_usd`` (so the budget gate trips only on a real
    dollar estimate, never a fabricated one). Never raises — a tracker error
    must not crash the daemon.
    """
    try:
        from soup_cli.experiment.tracker import ExperimentTracker
        from soup_cli.utils.run_cost import estimate_run_cost_usd

        runs = ExperimentTracker().list_runs(limit=10)
    except Exception:  # noqa: BLE001 - cost estimation must never crash the loop
        return 0.0
    for run in runs:
        if run.get("status") != "completed":
            continue
        device = run.get("device_name")
        duration = run.get("duration_secs")
        if not device or duration is None:
            continue
        try:
            cost = estimate_run_cost_usd(str(device), float(duration))
        except (TypeError, ValueError):
            continue
        if cost is not None and cost > 0:
            return float(cost)
        # Most-recent completed run was on a CPU / unpriced GPU → no dollar
        # signal; report 0.0 (no cost gate) rather than scanning older runs.
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_prewired_watch_config(
    *,
    max_iterations: Optional[int] = None,
    poll_interval_sec: float = 60.0,
    state_path: Optional[str] = None,
    iteration_dir: Optional[str] = None,
    pack_iterations: bool = False,
    served_model: Optional[str] = None,
    base_model: str = "unknown",
) -> "WatchConfig":
    """Return a ``WatchConfig`` wired with the pre-wired production stages."""
    from soup_cli.utils.loop_daemon import WatchConfig

    return WatchConfig(
        poll_interval_sec=poll_interval_sec,
        max_iterations=max_iterations,
        state_path=state_path,
        iteration_dir=iteration_dir,
        harvest_fn=harvest_from_traces,
        train_fn=train_dpo_from_pairs,
        gate_fn=verify_candidate,
        deploy_fn=deploy_to_canary,
        promote_fn=promote_canary,
        cost_fn=estimate_cost,
        quality_fn=canary_quality,
        pack_iterations=pack_iterations,
        served_model=served_model,
        base_model=base_model,
    )
