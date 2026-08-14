"""Reward-verifier synthesis — `soup reward synth` (v0.71.40).

Auto-generate a *deterministic* reward verifier from a dataset of reference (gold)
outputs. The emitted artifact is a readable, editable, committable ``.py`` that rides
:func:`soup_cli.trainer.rewards.load_reward_fn`'s existing ``.py`` path — NO new
trusted-exec hot path.

Pipeline (all pure, NO top-level torch):
  detect_kind → induce_* → render_verifier_py → (load) → perturb_negatives + calibrate.

v1 verifier families (deterministic only): ``numeric`` · ``json_schema`` · ``regex`` ·
``tool_call``. Judge-based and LLM-codegen are explicitly out of scope.

The calibration report is the moat: a synthesized verifier is only emitted if it
accepts its own references AND rejects auto-perturbed negatives — a degenerate
always-1.0 verifier is refused.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

KINDS: tuple[str, ...] = ("numeric", "json_schema", "regex", "tool_call")
# Auto-detect precedence: most-specific shape wins.
_DETECT_ORDER: tuple[str, ...] = ("tool_call", "json_schema", "numeric", "regex")

# Fraction of golds that must fit a kind for auto-detection / regex confidence.
_MIN_CONFIDENCE = 0.9
DEFAULT_NUMERIC_TOLERANCE = 1e-6
DEFAULT_MIN_DISCRIMINATION = 0.5
# A verifier must accept at least this fraction of its OWN references, else the
# induced spec plainly does not fit the golds (mirrors _MIN_CONFIDENCE).
_MIN_SELF_ACCEPT = 0.9
_MAX_ROWS = 1_000_000

_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_WHOLE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# Frozen result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NumericSpec:
    is_float: bool
    tolerance: float


@dataclass(frozen=True)
class ToolCallSpec:
    """Per-tool argument signatures.

    ``tools`` maps each tool name to ``{"required": (...), "allowed": (...)}`` —
    ``required`` = keys present in EVERY observed call of that name (intersection),
    ``allowed`` = union of keys seen for that name. The verifier accepts a call iff
    ``required <= call_args <= allowed`` for the matching name, so a call cannot
    borrow another tool's argument keys or omit required ones.
    """

    tools: dict  # name -> {"required": tuple[str, ...], "allowed": tuple[str, ...]}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tools))

    @property
    def arg_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for sig in self.tools.values():
            keys.update(sig.get("allowed", ()))
        return tuple(sorted(keys))


@dataclass(frozen=True)
class CalibrationReport:
    kind: str
    positives: int
    negatives: int
    pos_accept: float
    neg_accept: float
    discrimination: float
    precision: float
    recall: float
    refused: bool
    reason: str


SpecType = Union[NumericSpec, dict, ToolCallSpec, str]


@dataclass(frozen=True)
class SynthResult:
    kind: str
    source: str
    spec: SpecType  # NumericSpec | dict | ToolCallSpec | str (pattern)
    report: Optional[CalibrationReport] = None


# ---------------------------------------------------------------------------
# Gold extraction
# ---------------------------------------------------------------------------
def _row_gold(row: Mapping[str, Any], field: str) -> Optional[str]:
    if not isinstance(row, Mapping):
        return None
    if field in row and row[field] is not None:
        value = row[field]
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)  # numbers, bools -> their str form
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, Mapping) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
    return None


def extract_golds(rows: Sequence[Mapping[str, Any]], *, field: str = "answer") -> list[str]:
    """Pull the gold output from each row (``field`` else chat last-assistant).

    Raises ``TypeError`` if ``rows`` is not a sequence of mappings; ``ValueError``
    if no row yields a gold.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mapping rows")
    if len(rows) > _MAX_ROWS:
        raise ValueError(f"rows exceed cap of {_MAX_ROWS}")
    golds: list[str] = []
    for row in rows:
        gold = _row_gold(row, field)
        if gold is not None:
            golds.append(gold)
    if not golds:
        raise ValueError(
            f"no gold outputs found (field={field!r}; rows need that key or a "
            "chat 'messages' list with an assistant turn)"
        )
    return golds


# ---------------------------------------------------------------------------
# Kind primitives
# ---------------------------------------------------------------------------
def _is_number(text: str) -> bool:
    return bool(_WHOLE_NUMBER_RE.match(text.strip()))


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _is_tool_call(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and isinstance(obj.get("arguments"), dict)
    )


def _fraction(golds: Sequence[str], predicate: Callable[[str], bool]) -> float:
    if not golds:
        return 0.0
    return sum(1 for g in golds if predicate(g)) / len(golds)


def detect_kind(golds: Sequence[str]) -> Optional[str]:
    """Infer the verifier family from the golds, or ``None`` if un-inferrable.

    Precedence: tool_call → json_schema → numeric → regex.
    """
    if not golds:
        return None
    for kind in _DETECT_ORDER:
        if kind == "tool_call":
            frac = _fraction(golds, lambda g: _is_tool_call(_json_or_none(g)))
        elif kind == "json_schema":
            frac = _fraction(
                golds, lambda g: isinstance(_json_or_none(g), (dict, list))
            )
        elif kind == "numeric":
            frac = _fraction(golds, _is_number)
        else:  # regex
            frac = 1.0 if induce_regex(golds) is not None else 0.0
        if frac >= _MIN_CONFIDENCE:
            return kind
    return None


# ---------------------------------------------------------------------------
# Inducers
# ---------------------------------------------------------------------------
def induce_numeric(
    golds: Sequence[str], *, tolerance: Optional[float] = None
) -> NumericSpec:
    if _fraction(golds, _is_number) < _MIN_CONFIDENCE:
        raise ValueError(
            "references are not numeric — cannot induce a numeric verifier "
            "(each gold must be a bare number); pass a different --kind"
        )
    is_float = any(("." in g) or ("e" in g.lower()) for g in golds)
    if tolerance is None:
        tol = DEFAULT_NUMERIC_TOLERANCE if is_float else 0.0
    else:
        tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0:
        raise ValueError(f"tolerance must be a finite, non-negative number, got {tol!r}")
    return NumericSpec(is_float=is_float, tolerance=tol)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def induce_json_schema(golds: Sequence[str]) -> dict:
    """Induce a top-level JSON schema (keys + types + required) from the golds.

    Refuses (``ValueError``) rather than emitting a wrong schema when the golds
    mix top-level shapes (some object, some array) or parse to no JSON container
    at all — mirroring :func:`induce_regex`'s refuse-don't-guess policy. A key
    whose type is inconsistent across rows is left un-typed (presence only), so
    an over-narrow ``type`` can never reject a valid completion.
    """
    parsed = [_json_or_none(g) for g in golds]
    containers = [o for o in parsed if isinstance(o, (dict, list))]
    if not containers:
        raise ValueError(
            "no JSON object/array references found — cannot induce a json_schema"
        )
    dicts = [o for o in containers if isinstance(o, dict)]
    lists = [o for o in containers if isinstance(o, list)]
    if dicts and lists:
        raise ValueError(
            "references mix JSON objects and arrays — split them or pass an "
            "explicit --kind; refusing to induce an ambiguous schema"
        )
    if lists:
        return {"type": "array"}
    key_types: dict[str, set[str]] = {}
    key_counts: dict[str, int] = {}
    for obj in dicts:
        for key, value in obj.items():
            key_counts[key] = key_counts.get(key, 0) + 1
            key_types.setdefault(key, set()).add(_json_type(value))
    properties: dict[str, dict] = {}
    for key, types in key_types.items():
        # Only constrain the type when EVERY row agreed on it; else presence-only.
        properties[key] = {"type": next(iter(types))} if len(types) == 1 else {}
    required = sorted(k for k, c in key_counts.items() if c == len(dicts))
    return {"type": "object", "properties": properties, "required": required}


def induce_tool_call(golds: Sequence[str]) -> ToolCallSpec:
    """Induce per-tool argument signatures (name -> required/allowed keys)."""
    per_name: dict[str, list[set]] = {}
    for gold in golds:
        obj = _json_or_none(gold)
        if _is_tool_call(obj):
            per_name.setdefault(obj["name"], []).append(
                {str(k) for k in obj["arguments"].keys()}
            )
    if not per_name:
        raise ValueError(
            "no tool-call references found (each needs a JSON object with string "
            "'name' + object 'arguments') — cannot induce a tool_call verifier"
        )
    tools: dict = {}
    for name, calls in per_name.items():
        allowed = set().union(*calls)
        required = set.intersection(*calls)
        tools[name] = {
            "required": tuple(sorted(required)),
            "allowed": tuple(sorted(allowed)),
        }
    return ToolCallSpec(tools=tools)


def _char_class(chars: set[str]) -> Optional[str]:
    if len(chars) == 1:
        return re.escape(next(iter(chars)))
    if all(c.isdigit() for c in chars):
        return r"\d"
    if all(c.isalpha() for c in chars):
        return r"[A-Za-z]"
    if all(c.isalnum() for c in chars):
        return r"[A-Za-z0-9]"
    return None


def induce_regex(golds: Sequence[str]) -> Optional[str]:
    """Conservative positional pattern induction over equal-length golds.

    Returns ``None`` unless the golds share a length and every position
    generalizes to a confident char class (the safe direction: refuse rather
    than emit a loose pattern).
    """
    golds = [g.strip() for g in golds if g.strip()]
    if len(golds) < 2:
        return None
    length = len(golds[0])
    if any(len(g) != length for g in golds) or length == 0:
        return None
    parts: list[str] = []
    for pos in range(length):
        cls = _char_class({g[pos] for g in golds})
        if cls is None:
            return None
        parts.append(cls)
    # By construction every gold matches (each position's class is a superset of
    # the observed chars), so no post-hoc confidence check is needed.
    return "^" + "".join(parts) + "$"


# ---------------------------------------------------------------------------
# Source rendering
# ---------------------------------------------------------------------------
_HEADER = '''"""Deterministic reward verifier — generated by `soup reward synth` (v0.71.40).

Kind: {kind}. Induced from {n_refs} reference output(s). EDIT FREELY — this is a
self-contained, committable reward function. Wire it via:

    training:
      reward_fn: {rel_hint}

Signature: reward_fn(completions, **kwargs) -> list[float]; the per-row gold answer
arrives as kwargs["answer"] (a list aligned with completions), matching TRL/GRPO.
"""

import json
import re


def _last_content(completion):
    return completion[-1]["content"] if completion else ""
'''

_NUMERIC_BODY = '''
_NUMBER_RE = re.compile(r"[+-]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?")
_BOXED_RE = re.compile(r"\\\\boxed\\{([^}]*)\\}")


def _extract_number(text):
    # Prefer an explicit answer marker (\\boxed{...} then ####) over "last number
    # in the whole string", so trailing chit-chat numbers do not hijack the score.
    boxed = _BOXED_RE.search(text)
    if boxed:
        nums = _NUMBER_RE.findall(boxed.group(1))
        if nums:
            return nums[-1]
    if "####" in text:
        nums = _NUMBER_RE.findall(text.split("####")[-1])
        if nums:
            return nums[-1]
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def _numbers_match(pred, gold, tol):
    if pred is None:
        return False
    try:
        if tol == 0:
            # Exact: integer compare first so large ints dodge float rounding.
            try:
                return int(pred) == int(gold)
            except (ValueError, TypeError):
                pass
        return abs(float(pred) - float(gold)) <= tol
    except (ValueError, TypeError):
        return False


def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [])
    out = []
    for completion, expected in zip(completions, answers):
        predicted = _extract_number(_last_content(completion))
        gold = str(expected).strip()
        out.append(1.0 if _numbers_match(predicted, gold, _TOLERANCE) else 0.0)
    return out
'''

_JSON_SCHEMA_BODY = '''

def _json_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _matches_schema(data):
    expected = _SCHEMA.get("type")
    if expected == "array":
        return isinstance(data, list)
    if expected == "object":
        if not isinstance(data, dict):
            return False
        for key in _SCHEMA.get("required", []):
            if key not in data:
                return False
        props = _SCHEMA.get("properties", {})
        _NUMERIC = {"integer", "number"}
        for key, spec in props.items():
            if key in data:
                want = spec.get("type")
                got = _json_type(data[key])
                if not want:
                    continue  # presence-only (type varied across references)
                if want in _NUMERIC and got in _NUMERIC:
                    continue  # int/float interchangeable
                if want != got:
                    return False
        return True
    return True


def reward_fn(completions, **kwargs):
    out = []
    for completion in completions:
        content = _last_content(completion)
        fenced = re.search(r"```(?:json)?\\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        try:
            data = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            out.append(0.0)
            continue
        out.append(1.0 if _matches_schema(data) else 0.0)
    return out
'''

_REGEX_BODY = '''

def reward_fn(completions, **kwargs):
    out = []
    for completion in completions:
        content = _last_content(completion).strip()
        out.append(1.0 if _PATTERN.fullmatch(content) else 0.0)
    return out
'''

_TOOL_CALL_BODY = '''

def reward_fn(completions, **kwargs):
    out = []
    for completion in completions:
        content = _last_content(completion)
        fenced = re.search(r"```(?:json)?\\s*(.*?)```", content, re.DOTALL)
        if fenced:
            content = fenced.group(1)
        try:
            data = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            out.append(0.0)
            continue
        if not (isinstance(data, dict) and isinstance(data.get("arguments"), dict)):
            out.append(0.0)
            continue
        sig = _TOOLS.get(data.get("name"))
        if sig is None:
            out.append(0.0)
            continue
        keys = set(data["arguments"].keys())
        required = set(sig["required"])
        allowed = set(sig["allowed"])
        out.append(1.0 if required <= keys <= allowed else 0.0)
    return out
'''


def _safe_hint(value: object) -> str:
    """One-line, quote/backslash-free display string for the header docstring.

    ``rel_hint`` (the ``-o`` path) is the only value ``.format``'d RAW into the
    generated source (everything else is baked via ``repr``). The file is later
    exec'd, so a path containing ``\"\"\"`` + code must not be able to break out
    of the docstring — strip control chars, quotes and backslashes and cap it.
    """
    text = str(value)
    text = "".join(ch for ch in text if ch >= " " and ch not in '"\\')
    return text[:200] or "reward.py"


def render_verifier_py(kind: str, spec: Any, *, meta: Mapping[str, Any]) -> str:
    """Render a self-contained reward-verifier ``.py`` source string.

    The induced spec is baked in as a top-level constant via ``repr`` (so the
    code bodies stay literal — no ``str.format`` over code containing braces).
    """
    n_refs = int(meta.get("n_refs", 0))
    rel_hint = _safe_hint(meta.get("rel_hint", "reward.py"))
    header = _HEADER.format(kind=kind, n_refs=n_refs, rel_hint=rel_hint)
    if kind == "numeric":
        if not isinstance(spec, NumericSpec):
            raise TypeError("numeric verifier requires a NumericSpec")
        const = f"\n_TOLERANCE = {spec.tolerance!r}\n"
        body = const + _NUMERIC_BODY
    elif kind == "json_schema":
        if not isinstance(spec, dict):
            raise TypeError("json_schema verifier requires a schema dict")
        const = f"\n_SCHEMA = {spec!r}\n"
        body = const + _JSON_SCHEMA_BODY
    elif kind == "regex":
        if not isinstance(spec, str):
            raise TypeError("regex verifier requires a pattern string")
        const = f"\n_PATTERN = re.compile({spec!r})\n"
        body = const + _REGEX_BODY
    elif kind == "tool_call":
        if not isinstance(spec, ToolCallSpec):
            raise TypeError("tool_call verifier requires a ToolCallSpec")
        const = f"\n_TOOLS = {spec.tools!r}\n"
        body = const + _TOOL_CALL_BODY
    else:
        raise ValueError(f"unknown verifier kind: {kind!r} (options: {', '.join(KINDS)})")
    return header + body


# ---------------------------------------------------------------------------
# Negatives + calibration
# ---------------------------------------------------------------------------
def perturb_negatives(golds: Sequence[str], kind: str) -> list[str]:
    """Deterministically corrupt golds into known-bad outputs for calibration."""
    negatives: list[str] = []
    for gold in golds:
        gold = gold.strip()
        if kind == "numeric":
            num = _NUMBER_RE.findall(gold)
            if num:
                try:
                    negatives.append(str(float(num[-1]) + 9999.0))
                except ValueError:
                    negatives.append("not-a-number")
            else:
                negatives.append("not-a-number")
        elif kind == "json_schema":
            negatives.append("this is definitely not json {")
        elif kind == "tool_call":
            negatives.append("this is definitely not json {")
            obj = _json_or_none(gold)
            if _is_tool_call(obj):
                # Valid JSON, WRONG tool name — a "just parse JSON" verifier would
                # accept this; a name-bound verifier must reject it.
                negatives.append(
                    json.dumps({"name": "__nonexistent_tool__",
                                "arguments": obj["arguments"]})
                )
                # Valid JSON, right name, a foreign arg key outside `allowed`.
                negatives.append(
                    json.dumps({"name": obj["name"],
                                "arguments": {"__foreign_arg__": 1}})
                )
        elif kind == "regex":
            negatives.append(gold + "ZZZ_definitely_wrong")
        else:
            raise ValueError(f"unknown kind: {kind!r} (options: {', '.join(KINDS)})")
    # Always include a couple of universal degenerate cases.
    negatives.extend(["", "the model rambled without answering at all"])
    return negatives


def calibrate(
    reward_fn: Callable[..., list[float]],
    positives: Sequence[str],
    negatives: Sequence[str],
    *,
    kind: str = "",
    min_discrimination: float = DEFAULT_MIN_DISCRIMINATION,
) -> CalibrationReport:
    """Run ``reward_fn`` over positives (should accept) + negatives (should reject).

    ``discrimination = pos_accept - neg_accept``. Refuses when: (a) the verifier
    fails to accept ``>= _MIN_SELF_ACCEPT`` of its own references (the induced
    spec does not fit the golds), (b) ``discrimination <= 0`` (no separation at
    all — a hard floor the user cannot disable via ``--min-discrimination 0``),
    or (c) ``discrimination < min_discrimination``.
    """
    def _accept_rate(
        items: Sequence[str], answers: Sequence[str]
    ) -> tuple[float, int]:
        if not items:
            return 0.0, 0
        completions = [[{"role": "assistant", "content": s}] for s in items]
        scores = reward_fn(completions, answer=list(answers))
        accepted = sum(1 for s in scores if float(s) >= 0.5)
        return accepted / len(items), accepted

    # Positives are scored as content==gold. Negatives are corrupted completions
    # but must be scored against the REAL gold (cycled), else a gold-comparison
    # verifier would accept a corrupted answer matched against itself.
    pos_accept, tp = _accept_rate(positives, positives)
    if positives:
        neg_answers = [positives[j % len(positives)] for j in range(len(negatives))]
    else:
        neg_answers = list(negatives)
    neg_accept, fp = _accept_rate(negatives, neg_answers)
    discrimination = pos_accept - neg_accept
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = pos_accept
    refused = False
    reason = "verifier discriminates references from perturbed negatives"
    if pos_accept < _MIN_SELF_ACCEPT:
        refused = True
        reason = (
            f"verifier rejects its own references (accepts only "
            f"{pos_accept:.0%}, need >= {_MIN_SELF_ACCEPT:.0%}) — the induced "
            "spec does not fit the golds"
        )
    elif discrimination <= 0.0:
        refused = True
        reason = (
            f"verifier accepts perturbed negatives as readily as references "
            f"(accepts {pos_accept:.0%} of refs and {neg_accept:.0%} of negatives, "
            "discrimination <= 0) — degenerate, refusing regardless of "
            "--min-discrimination"
        )
    elif discrimination < min_discrimination:
        refused = True
        reason = (
            f"discrimination {discrimination:.2f} < required {min_discrimination:.2f} "
            f"(accepts {pos_accept:.0%} of references but also {neg_accept:.0%} of "
            "perturbed negatives) — the verifier is too permissive to reward-train on"
        )
    return CalibrationReport(
        kind=kind,
        positives=len(positives),
        negatives=len(negatives),
        pos_accept=pos_accept,
        neg_accept=neg_accept,
        discrimination=discrimination,
        precision=precision,
        recall=recall,
        refused=refused,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Orchestration (pure — the CLI does write→load→calibrate)
# ---------------------------------------------------------------------------
def _induce(kind: str, golds: Sequence[str], tolerance: Optional[float]) -> Any:
    if kind == "numeric":
        return induce_numeric(golds, tolerance=tolerance)
    if kind == "json_schema":
        return induce_json_schema(golds)
    if kind == "tool_call":
        return induce_tool_call(golds)
    if kind == "regex":
        pattern = induce_regex(golds)
        if pattern is None:
            raise ValueError(
                "could not induce a confident regex from the references "
                "(golds must share a length + positional char classes)"
            )
        return pattern
    raise ValueError(f"unknown kind: {kind!r} (options: {', '.join(KINDS)})")


def synthesize(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str = "answer",
    kind: str = "auto",
    tolerance: Optional[float] = None,
    rel_hint: str = "reward.py",
) -> SynthResult:
    """Detect + induce + render (no calibration — the CLI loads then calibrates)."""
    if kind != "auto" and kind not in KINDS:
        raise ValueError(f"unknown kind: {kind!r} (options: auto, {', '.join(KINDS)})")
    golds = extract_golds(rows, field=field)
    resolved = kind
    if kind == "auto":
        resolved = detect_kind(golds)
        if resolved is None:
            raise ValueError(
                "could not auto-detect a deterministic verifier kind from the "
                "references — pass --kind numeric|json_schema|regex|tool_call"
            )
    spec = _induce(resolved, golds, tolerance)
    source = render_verifier_py(
        resolved, spec, meta={"n_refs": len(golds), "rel_hint": rel_hint}
    )
    return SynthResult(kind=resolved, source=source, spec=spec, report=None)
