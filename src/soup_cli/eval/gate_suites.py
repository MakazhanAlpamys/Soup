"""Bundled general-suite registry for ``soup ship``'s leg 2 (v0.71.38).

The v0.25.0 leg-2 default was 15 hand-written trivia prompts scored by raw
substring containment (``eval/forgetting``) — decorative. This module makes the
regression gate real: a set of **offline, zero-dep, hand-authored** suites, each
producing a **per-model absolute** score in ``[0, 1]`` from a generator closure,
so ``compute_benchmark_deltas`` can flag a genuine regression.

It composes two families:

- the ``score_answer`` MCQ / arithmetic suites in ``eval/forgetting`` (fixed
  scorer, expanded to ~40 items), scored via ``ForgettingDetector``; and
- three **behavioural** suites — tool-calling, JSON-format, safety/refusal —
  bundled as JSONL fixtures and scored by the pure ``eval/custom`` +
  ``utils/diagnose`` scorers (tool-call name-match, refusal heuristic) plus a
  small container-only JSON check (``_is_json_container`` — a bare scalar is
  valid JSON but not the structured object the suite asks for, so it is new
  here rather than a call into ``diagnose.format.is_valid_json``).

No lm-eval, no network, no torch — the whole surface is CPU-testable and the
``soup ci init`` core-only install keeps working.

#316 — two of the three behavioural suites measured NOTHING on a real model.
Measured on an H100 against Meta-Llama-3.1-8B-Instruct, both were **harness**
defects, not model failures:

- ``mini_tool_call`` scored **0.000/40**. The prompts were bare user questions
  with no tool schema anywhere, so a fully tool-capable model correctly answered
  in prose and 0 of 40 outputs contained any JSON. The fixture now renders a
  candidate tool menu (name + description) ahead of each question, and names the
  ``{"function": {"name": ...}}`` envelope the scorer actually parses. The menu
  is a *selection* task — 8 candidates drawn deterministically from the suite's
  34 tools, excluding near-synonyms of the correct one so a miss means lost
  tool-calling rather than an ambiguous prompt.
- ``mini_format_json`` scored **0.000/40** because 38/40 answers were inside a
  ```json fence and the container check parsed the whole string. Extraction is
  now bounded (see ``_extract_json_container``).
- Both were additionally truncated by a 64-token generation budget; see
  ``BEHAVIOURAL_MAX_NEW_TOKENS``.

A suite pinned at 0.000 contributes nothing to the gate but also cannot *fall*,
which is precisely how the defect stayed invisible. Each repair therefore ships
with a control against the symmetric failure — a schema that hands over the
answer, or an extractor eager enough to pin the suite at 1.000.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Callable, Tuple

from soup_cli.eval.forgetting import MINI_BENCHMARKS, ForgettingDetector

GeneratorFn = Callable[[str], str]

# Behavioural suite names (the "coverage gaps" the v0.25.0 default had 0 items
# for). Values: (fixture filename, scorer kind).
MINI_TOOL_CALL = "mini_tool_call"
MINI_FORMAT_JSON = "mini_format_json"
MINI_SAFETY = "mini_safety"

_EXTENDED_SUITES: dict[str, Tuple[str, str]] = {
    MINI_TOOL_CALL: ("tool_call.jsonl", "tool_call"),
    MINI_FORMAT_JSON: ("format_json.jsonl", "format_json"),
    MINI_SAFETY: ("safety.jsonl", "refusal"),
}

#: The behavioural suites (JSONL-backed), in registration order.
EXTENDED_SUITE_NAMES: Tuple[str, ...] = tuple(_EXTENDED_SUITES)

#: The full offline default general suite = MCQ/arithmetic + behavioural.
DEFAULT_GENERAL_SUITE: Tuple[str, ...] = tuple(MINI_BENCHMARKS) + EXTENDED_SUITE_NAMES

# 4 MiB cap on a bundled fixture (mirrors behaviour_battery — defends against
# bundle corruption / an accidentally-committed giant JSONL).
_MAX_FIXTURE_BYTES = 4 * 1024 * 1024
# Cap a single model output before scoring (mirrors diagnose ``_MAX_OUTPUT_LEN``).
_MAX_OUTPUT_LEN = 64 * 1024

#: Generation budget the behavioural suites need (#316).
#:
#: ``live_eval.make_generator`` defaults to 64 new tokens, which is sized for the
#: MCQ suites — they answer in a single letter. Measured on an H100 against
#: Meta-Llama-3.1-8B-Instruct, 64 truncated **31/40** tool calls and **15/40**
#: JSON fences; 31 of those 32 tool-call failures were a single missing closing
#: brace. A caller building generators for these suites must pass this instead,
#: or it scores the budget rather than the model.
BEHAVIOURAL_MAX_NEW_TOKENS = 256

# A JSON container must BEGIN within this many characters of the output. A model
# that emits 2 KB of prose before its JSON did not "respond with JSON"; scanning
# the whole output for any bracket pair would credit incidental braces in prose
# and pin the suite at a constant 1.0 — exactly as blind as the 0.0 it replaces.
_MAX_CONTAINER_SCAN_CHARS = 2048
# ...and at most this many candidate start positions are tried, so a
# pathological payload ("[" * 20000) costs a bounded number of failed parses.
_MAX_CONTAINER_SCAN_STARTS = 48

_fixture_cache: dict[str, Tuple[dict, ...]] = {}


def is_bundled_suite(name: str) -> bool:
    """True when ``name`` is one Soup ships an offline scorer for."""
    return name in MINI_BENCHMARKS or name in _EXTENDED_SUITES


def _load_gate_fixture(filename: str) -> Tuple[dict, ...]:
    """Load a bundled ``data/_fixtures/gate/<filename>`` JSONL (symlink-rejected,
    size-capped) — mirrors ``behavior_battery.load_battery_probes``.

    ``filename`` is only ever an internal constant from ``_EXTENDED_SUITES`` (no
    user-supplied path), but the symlink + size guards stay for defence in depth.
    """
    if filename in _fixture_cache:
        return _fixture_cache[filename]
    from importlib.resources import as_file, files

    try:
        ref = files("soup_cli") / "data" / "_fixtures" / "gate" / filename
    except (ModuleNotFoundError, TypeError) as exc:  # pragma: no cover — install bug
        raise FileNotFoundError(f"gate suite fixture '{filename}' not bundled") from exc
    if not ref.is_file():  # pragma: no cover — install bug
        raise FileNotFoundError(f"gate suite fixture '{filename}' not bundled")
    with as_file(ref) as concrete:
        try:
            st = os.lstat(concrete)
        except OSError as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"gate suite fixture '{filename}' unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):  # pragma: no cover — defence in depth
            raise ValueError(f"gate suite fixture '{filename}' must not be a symlink")
        if st.st_size > _MAX_FIXTURE_BYTES:  # pragma: no cover
            raise ValueError(f"gate suite fixture '{filename}' too large")
        text = Path(concrete).read_text(encoding="utf-8")
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover — bundle corruption
            raise ValueError(f"gate suite fixture '{filename}' has malformed JSON") from exc
        if isinstance(row, dict):
            rows.append(row)
    result = tuple(rows)
    _fixture_cache[filename] = result
    return result


def load_suite_items(name: str) -> Tuple[dict, ...]:
    """Return the bundled rows for a behavioural suite ``name``."""
    if name not in _EXTENDED_SUITES:
        raise ValueError(
            f"'{name}' is not a behavioural gate suite; "
            f"options: {', '.join(EXTENDED_SUITE_NAMES)}"
        )
    filename, _kind = _EXTENDED_SUITES[name]
    return _load_gate_fixture(filename)


def _call(gen: GeneratorFn, prompt: str) -> str:
    """Invoke ``gen`` for one prompt; a generation error / non-str result scores
    as a failure (empty string) and an oversized result is truncated to
    ``_MAX_OUTPUT_LEN`` — so one bad generation never aborts a run."""
    try:
        out = gen(prompt)
    except Exception:  # noqa: BLE001 — a generation error is a failed item, not a crash
        return ""
    if not isinstance(out, str):
        return ""
    return out[:_MAX_OUTPUT_LEN]


def _fraction_passing(
    items: Tuple[dict, ...], gen: GeneratorFn, predicate
) -> float:
    """Fraction of ``items`` whose generated output satisfies ``predicate``.

    Each item is scored independently: a predicate that raises (e.g. a
    ``RecursionError`` from ``json.loads`` on a pathologically-nested output)
    scores that one item as a failure rather than aborting the whole leg-2 run.
    """
    if not items:
        return 0.0
    passed = 0
    for item in items:
        output = _call(gen, item.get("prompt", ""))
        try:
            if predicate(item, output):
                passed += 1
        except Exception:  # noqa: BLE001 — a scoring error is a failed item, not a crash
            continue
    return passed / len(items)


def tool_names_in_prompt(prompt: str) -> list[str]:
    """The tool names a rendered ``mini_tool_call`` prompt shows the model.

    The fixture embeds its candidate menu as a single JSON-array line, so this
    reads the real menu rather than regexing prose. Returns ``[]`` for a prompt
    with no menu (which is itself the #316 defect — see the module docstring).
    """
    if not isinstance(prompt, str):
        return []
    for line in prompt.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            entries = json.loads(line)
        except Exception:  # noqa: BLE001 — a prose line that merely starts with "["
            continue
        if not isinstance(entries, list):
            continue
        names = [
            entry["name"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        if names:
            return names
    return []


def _extract_json_container(text: str):
    """The first JSON object/array in ``text``, or ``None``.

    Model answers arrive wrapped: measured on an H100, **38 of 40**
    ``mini_format_json`` answers were inside a ```json fence and 15 of those
    fences were unclosed (truncated by the 64-token budget). Parsing the whole
    string scored the *envelope*, not the JSON, and the suite read 0.000 for a
    model emitting correct JSON.

    The scan is deliberately bounded on both axes (see the module constants):
    an extractor that hunts the entire output for any bracket pair turns the
    suite into a constant 1.0, which detects exactly as little as a constant 0.0.
    """
    if not isinstance(text, str) or "\x00" in text or len(text) > _MAX_OUTPUT_LEN:
        return None
    # Fast path: the whole output IS the container (the pre-#316 behaviour, kept
    # so a bare answer's semantics are byte-identical).
    try:
        parsed = json.loads(text.strip())
    except Exception:  # noqa: BLE001 — wrapped / malformed; fall through to the scan
        pass
    else:
        return parsed if isinstance(parsed, (dict, list)) else None
    decoder = json.JSONDecoder()
    starts = 0
    for idx, char in enumerate(text[:_MAX_CONTAINER_SCAN_CHARS]):
        if char not in "{[":
            continue
        starts += 1
        if starts > _MAX_CONTAINER_SCAN_STARTS:
            break
        try:
            # raw_decode stops at the end of the value, so trailing prose — or a
            # fence the budget cut off mid-close — does not invalidate it.
            parsed, _end = decoder.raw_decode(text, idx)
        except Exception:  # noqa: BLE001 — incl. RecursionError on deep nesting
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _unwrap_tool_call(output: str) -> str:
    """Re-serialise the first JSON object in ``output`` for the tool scorer.

    ``eval.custom._parse_tool_call`` parses the whole string too, so a correct
    call inside a fence is the same packaging-vs-content miss as above. This can
    only ever *reveal* a call — ``tool_call_name_match`` still requires an exact
    function-name match, so unwrapping cannot credit the wrong tool.
    """
    extracted = _extract_json_container(output)
    if isinstance(extracted, dict):
        return json.dumps(extracted)
    return output if isinstance(output, str) else ""


def _score_tool_call(items: Tuple[dict, ...], gen: GeneratorFn) -> float:
    from soup_cli.eval.custom import tool_call_name_match

    return _fraction_passing(
        items,
        gen,
        lambda item, out: tool_call_name_match(
            _unwrap_tool_call(out), item.get("expected", "")
        ),
    )


def _is_json_container(text: str) -> bool:
    """Valid JSON that parses to an object or array.

    A bare scalar (``"42"``, ``"true"``) is valid JSON yet is not the structured
    object/array the format suite asks for, so it must not score. Applies the
    same length / null-byte caps as diagnose ``is_valid_json`` before parsing;
    catches broadly so a deeply-nested payload's ``RecursionError`` (which the
    json C-scanner can raise even under the length cap) scores as invalid.
    """
    return _extract_json_container(text) is not None


def _score_format_json(items: Tuple[dict, ...], gen: GeneratorFn) -> float:
    return _fraction_passing(items, gen, lambda _item, out: _is_json_container(out))


def _score_refusal_suite(items: Tuple[dict, ...], gen: GeneratorFn) -> float:
    """Per-model safety score = fraction of harmful prompts the model refuses.

    HIGHER is safer, so a tuned model that stops refusing drops below the base
    and ``compute_benchmark_deltas`` flags the under-refusal regression.
    """
    from soup_cli.utils.diagnose.refusal import looks_like_refusal

    return _fraction_passing(items, gen, lambda _item, out: looks_like_refusal(out))


_EXTENDED_SCORERS = {
    "tool_call": _score_tool_call,
    "format_json": _score_format_json,
    "refusal": _score_refusal_suite,
}


def score_bundled_suite(name: str, gen: GeneratorFn) -> float:
    """Score suite ``name`` for one model, returning an absolute ``[0, 1]``.

    MCQ / arithmetic suites route through the (fixed) ``ForgettingDetector``
    scorer; behavioural suites through their bundled pure scorer. Raises
    ``ValueError`` for an unknown suite (never silently 0.0).
    """
    if name in MINI_BENCHMARKS:
        return ForgettingDetector(generate_fn=gen, benchmark=name).run_baseline()
    if name in _EXTENDED_SUITES:
        _filename, kind = _EXTENDED_SUITES[name]
        items = load_suite_items(name)
        return _EXTENDED_SCORERS[kind](items, gen)
    raise ValueError(
        f"unknown bundled suite {name!r}; options: {', '.join(DEFAULT_GENERAL_SUITE)}"
    )


__all__ = [
    "BEHAVIOURAL_MAX_NEW_TOKENS",
    "DEFAULT_GENERAL_SUITE",
    "EXTENDED_SUITE_NAMES",
    "MINI_FORMAT_JSON",
    "MINI_SAFETY",
    "MINI_TOOL_CALL",
    "is_bundled_suite",
    "load_suite_items",
    "score_bundled_suite",
    "tool_names_in_prompt",
]
