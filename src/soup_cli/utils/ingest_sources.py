"""Universal trace importer (v0.63.0 Part A).

Adapters for production-observability ingest from Langfuse / LangSmith /
Helicone / OpenPipe / OpenTelemetry / OpenAI Stored Completions. Extends the
v0.26.0 Trace harvester (LangChain / OpenAI / Soup-serve) to the full SaaS
ecosystem so `soup loop` can close the train -> eval -> gate -> ship cycle
without ripping out existing dashboards.

Every parser normalises to a frozen `TraceRecord` that downstream
`build_pairs` (v0.26.0) can consume after a thin shim. PII reminder fires
once per ingest invocation, mirroring v0.26.0 Part C policy.

Design notes:
- All parsers are pure Iterable[dict] -> Iterator[TraceRecord]. No network
  code. SaaS API pulls happen out-of-band; users hand us a JSONL export.
- File reads enforce cwd containment + null-byte rejection (TOCTOU policy
  mirroring v0.26.0 / v0.40.3 / v0.55.0).
- `_MAX_INGEST_LINES` caps any single ingest to prevent OOM on tampered
  exports; default 1,000,000 (production-scale day of traces).
- Auth env-var lookup is read-only — Soup never makes the network call,
  it just tells the user which variable to set if they want SaaS pulls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from soup_cli.utils.paths import is_under_cwd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_INGEST_SOURCES: frozenset[str] = frozenset(
    {"langfuse", "langsmith", "helicone", "openpipe", "otel", "openai-stored"}
)

_MAX_SOURCE_NAME_LEN = 32
_MAX_INGEST_LINES = 1_000_000

# Per-source env vars used for SaaS auth. Soup itself only echoes which
# variable is set — never makes the network call.
_AUTH_ENV: Mapping[str, str] = {
    "langfuse": "LANGFUSE_KEY",
    "langsmith": "LANGSMITH_API_KEY",
    "helicone": "HELICONE_API_KEY",
    "openpipe": "OPENPIPE_API_KEY",
    "openai-stored": "OPENAI_API_KEY",
    "otel": "OTEL_EXPORTER_OTLP_HEADERS",
}


@dataclass(frozen=True)
class TraceRecord:
    """A single normalised trace record from any ingest source.

    Maps cleanly onto the v0.26.0 ``Trace`` shape so downstream
    ``build_pairs`` accepts it after a thin adapter pass.

    ``metadata`` is wrapped in :class:`types.MappingProxyType` post-init so
    callers cannot mutate the dict in place (code-review MEDIUM fix
    v0.63.0: previously a mutable Dict on a frozen dataclass let callers
    silently corrupt records shared across call sites).
    """

    trace_id: str
    prompt: str
    output: str
    source: str
    signal: str = "none"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # `metadata` may arrive as a plain dict or as a MappingProxyType
        # already; normalise via dict-copy then re-wrap. `object.__setattr__`
        # because the dataclass is frozen.
        meta = dict(self.metadata) if self.metadata else {}
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (unwraps the MappingProxyType).

        ``dataclasses.asdict`` cannot serialise ``MappingProxyType`` directly
        (pickling fails through json.dumps) — this helper is the canonical
        path from a record to a JSONL row.
        """
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "output": self.output,
            "source": self.source,
            "signal": self.signal,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Source-name validation
# ---------------------------------------------------------------------------


def validate_source_name(name: object) -> str:
    """Normalise + validate an ingest-source name.

    Mirrors v0.41.0 / v0.50.0 / v0.51.0 / v0.62.0 validator policy:
    bool-first, non-string -> TypeError, empty / null-byte / oversize ->
    ValueError. Case-insensitive: ``LANGFUSE`` and ``langfuse`` resolve
    to ``langfuse``.
    """
    if isinstance(name, bool):
        raise TypeError("source name must be str, not bool")
    if not isinstance(name, str):
        raise TypeError(f"source name must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("source name must be non-empty")
    if "\x00" in name:
        raise ValueError("source name must not contain null bytes")
    if len(name) > _MAX_SOURCE_NAME_LEN:
        raise ValueError(
            f"source name must be <= {_MAX_SOURCE_NAME_LEN} chars, got {len(name)}"
        )
    canonical = name.lower().strip()
    if canonical not in SUPPORTED_INGEST_SOURCES:
        raise ValueError(
            f"unknown ingest source {name!r}; supported: "
            f"{sorted(SUPPORTED_INGEST_SOURCES)}"
        )
    return canonical


def resolve_auth_env(name: object) -> Optional[str]:
    """Return the value of the env var that authenticates this source.

    Returns ``None`` when the var is not set. Soup never reads the actual
    SaaS API; we surface only whether the operator has wired up creds.
    Unknown sources raise.
    """
    import os

    canonical = validate_source_name(name)
    env_key = _AUTH_ENV[canonical]
    value = os.environ.get(env_key)
    if value is None or value == "":
        return None
    return value


# ---------------------------------------------------------------------------
# Per-source parsers (pure functions over Iterable[Any])
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str:
    """Best-effort string extraction for messy SaaS payloads."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common shapes: {messages: [...]}, {content: "..."}, {text: "..."}
        messages = value.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, dict):
                                text = chunk.get("text")
                                if isinstance(text, str):
                                    parts.append(text)
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        for key in ("content", "text", "prompt", "response", "output", "input"):
            sub = value.get(key)
            if isinstance(sub, str):
                return sub
            if isinstance(sub, list):
                # OpenAI Stored Completions style: input/output as list of msgs
                for msg in sub:
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if isinstance(c, str):
                            return c
    if isinstance(value, list):
        # OpenAI Stored Completions: list of role/content turns
        for msg in value:
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c
    return ""


def _signal_from_thumbs(score: Any) -> str:
    """Map a thumbs score to canonical signal vocabulary."""
    if isinstance(score, bool):
        return "thumbs_up" if score else "thumbs_down"
    if isinstance(score, (int, float)):
        if score >= 1:
            return "thumbs_up"
        if score <= 0:
            return "thumbs_down"
    if isinstance(score, str):
        low = score.lower()
        if low in ("up", "thumbs_up", "positive", "1"):
            return "thumbs_up"
        if low in ("down", "thumbs_down", "negative", "0"):
            return "thumbs_down"
    return "none"


def parse_langfuse(events: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse Langfuse-shaped trace exports.

    Langfuse exports use ``input`` / ``output`` top-level fields; both may
    be a string OR a structured object (messages list / content envelope).
    """
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        prompt = _coerce_str(event.get("input"))
        output = _coerce_str(event.get("output"))
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"langfuse-{idx}")
        signal = _signal_from_thumbs(event.get("score") or event.get("rating"))
        metadata: Dict[str, Any] = {}
        model = event.get("model")
        if isinstance(model, str):
            metadata["model"] = model
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="langfuse",
            signal=signal,
            metadata=metadata,
        )


def parse_langsmith(events: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse LangSmith run exports.

    LangSmith uses ``inputs`` / ``outputs`` (note the trailing s, distinct
    from Langfuse) plus a ``feedback_stats`` block with thumbs averages.
    """
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        prompt = _coerce_str(event.get("inputs"))
        outputs = event.get("outputs")
        # LangSmith generations: {generations: [[{text: "..."}]]}
        output = ""
        if isinstance(outputs, dict):
            gens = outputs.get("generations")
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, list) and first:
                    item = first[0]
                else:
                    item = first
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        output = text
            if not output:
                output = _coerce_str(outputs)
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"langsmith-{idx}")
        signal = "none"
        fb = event.get("feedback_stats")
        if isinstance(fb, dict):
            thumbs = fb.get("thumbs")
            if isinstance(thumbs, dict):
                signal = _signal_from_thumbs(thumbs.get("avg"))
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="langsmith",
            signal=signal,
            metadata={},
        )


def parse_helicone(events: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse Helicone request log exports.

    Helicone wraps the OpenAI request + response: ``request.body`` carries
    messages, ``response.body`` carries choices. Wide variation in shape
    across SDK versions, hence the defensive coercion.
    """
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        req = event.get("request")
        if isinstance(req, dict):
            prompt = _coerce_str(req.get("body") or req)
        else:
            prompt = _coerce_str(req)
        resp = event.get("response")
        output = ""
        if isinstance(resp, dict):
            body = resp.get("body")
            if isinstance(body, dict):
                choices = body.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        msg = first.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, str):
                                output = content
                        if not output:
                            output = _coerce_str(first.get("text"))
            if not output:
                output = _coerce_str(body)
        else:
            output = _coerce_str(resp)
        if not prompt or not output:
            continue
        trace_id = str(event.get("request_id") or event.get("id") or f"helicone-{idx}")
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="helicone",
            signal="none",
            metadata={},
        )


def parse_openpipe(events: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse OpenPipe trace exports.

    OpenPipe uses ``messages`` (OpenAI-style list) + ``response`` string.
    """
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        messages = event.get("messages")
        prompt = ""
        if isinstance(messages, list):
            user_msgs = [
                str(m.get("content") or "")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            prompt = "\n".join(p for p in user_msgs if p).strip()
        if not prompt:
            prompt = _coerce_str(event.get("prompt"))
        output = _coerce_str(event.get("response") or event.get("output"))
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"openpipe-{idx}")
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="openpipe",
            signal="none",
            metadata={},
        )


def parse_otel(spans: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse raw OpenTelemetry spans for LLM call attributes.

    Filters to spans carrying both ``llm.prompt`` and ``llm.completion``
    attributes (OpenTelemetry GenAI semantic conventions).
    """
    for idx, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        attrs = span.get("attributes")
        if not isinstance(attrs, dict):
            continue
        prompt = attrs.get("llm.prompt") or attrs.get("gen_ai.prompt")
        output = attrs.get("llm.completion") or attrs.get("gen_ai.completion")
        if not isinstance(prompt, str) or not isinstance(output, str):
            continue
        if not prompt or not output:
            continue
        trace_id = str(span.get("traceId") or span.get("spanId") or f"otel-{idx}")
        model = attrs.get("llm.model") or attrs.get("gen_ai.model")
        metadata: Dict[str, Any] = {}
        if isinstance(model, str):
            metadata["model"] = model
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="otel",
            signal="none",
            metadata=metadata,
        )


def parse_openai_stored(events: Iterable[Any]) -> Iterator[TraceRecord]:
    """Parse OpenAI Stored Completions exports.

    Stored Completions returns ``input`` (list of role/content msgs) +
    ``output`` (list of role/content msgs from the assistant).
    """
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        prompt = _coerce_str(event.get("input"))
        output = _coerce_str(event.get("output"))
        if not prompt or not output:
            continue
        trace_id = str(event.get("id") or f"openai-stored-{idx}")
        model = event.get("model")
        metadata: Dict[str, Any] = {}
        if isinstance(model, str):
            metadata["model"] = model
        yield TraceRecord(
            trace_id=trace_id,
            prompt=prompt,
            output=output,
            source="openai-stored",
            signal="none",
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Universal dispatcher
# ---------------------------------------------------------------------------


_PARSERS = {
    "langfuse": parse_langfuse,
    "langsmith": parse_langsmith,
    "helicone": parse_helicone,
    "openpipe": parse_openpipe,
    "otel": parse_otel,
    "openai-stored": parse_openai_stored,
}


def ingest_traces(*, source: str, path: str) -> Iterator[TraceRecord]:
    """Read a JSONL file and dispatch to the matching parser.

    Validates source name + path containment + null-byte rejection before
    any open. Caps the number of lines read at ``_MAX_INGEST_LINES`` to
    bound memory on tampered exports.
    """
    canonical = validate_source_name(source)
    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"path {path!r} is outside cwd")
    import os

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    parser = _PARSERS[canonical]

    def _stream() -> Iterator[Any]:
        count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if count >= _MAX_INGEST_LINES:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    # Malformed line -> skip (matches v0.26.0 parser policy)
                    continue
                count += 1

    yield from parser(_stream())
