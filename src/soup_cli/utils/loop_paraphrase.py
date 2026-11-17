"""Paraphrases of the prompts of corrected answers: written once, stored, read back.

The loop's verdict ("loop-v2", ``docs/loop-formats.md``) checks whether a new model version
learned the corrections people made, by asking it paraphrases of each corrected prompt
instead of the prompt it was trained on. This module creates and stores those paraphrases:

- :func:`paraphrase_edit` asks a chat endpoint for two paraphrases of an edit's prompt (the
  last user message), drops the ones that are the prompt again (``same_as_prompt``) or that
  hold the words the correction added (``leaks_answer``), asks once more when fewer than two
  survive, and returns the ``paraphrase`` record of ``docs/loop-formats.md`` (or ``None`` when
  nothing survives: then no line is written);
- :func:`paraphrase_edits` is what the pre-wired train stage runs before training: for every
  edit that has no current record it asks, appends the record to
  ``.soup-loops/paraphrases.jsonl`` and returns the notes the iteration keeps. An edit is
  asked once and reused by every later iteration;
- :func:`read_paraphrases` reads the file back (the last line per ``response_id`` wins).

Paraphrases are never training data: nothing here touches the datasets.

The method mirrors the consuming application's: same template text,
same filters, same added-words rule, one retry request, two paraphrases per edit.

Who answers: ``loop.yaml``'s ``paraphraser_endpoint`` / ``paraphraser_model`` (an
OpenAI-compatible endpoint on this machine or LAN), else the loop's own serve endpoint
(``SOUP_LOOP_SERVE_ENDPOINT``) with the ``X-Soup-Internal: loop-check`` header and the tool
token, which makes ``soup serve`` leave the call out of its trace log and canary routing so
the paraphrases cannot come back as training data. All HTTP goes through ``post_capped``.
"""

from __future__ import annotations

import logging
import os
import re
import string
import unicodedata
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.parse import urlparse

from soup_cli.utils.loop_feedback import EditedAnswer, LogStats, _iter_objects
from soup_cli.utils.loop_formats import DEFAULT_LAYOUT, SCHEMA_VERSION, prompt_hash
from soup_cli.utils.loop_log import append_json_line, utc_now_iso
from soup_cli.utils.loop_stages import TOOL_AUTH_TOKEN_ENV, _endpoint_is_local

_LOG = logging.getLogger(__name__)

#: The file the records are appended to (the contract layout).
PARAPHRASE_LOG = DEFAULT_LAYOUT["paraphrase"]

#: The instruction template the paraphrases are written with; recorded as ``generator.template``.
TEMPLATE = MappingProxyType({"id": "loop-paraphrase", "version": "1"})
#: Text of the template ``loop-paraphrase`` version ``1`` (the consuming application ships the
#: same text).
TEMPLATE_TEXT = (
    "Rewrite the question below in {n} different ways. Keep its meaning exactly and do not "
    "answer it. Reply with a numbered list only, one rewrite per line, like:\n1. ...\n2. ...\n\n"
    "Question: {question}"
)
#: Paraphrases kept per edit, and asked for in one request.
PARAPHRASES_WANTED = 2
PARAPHRASE_TEMPERATURE = 0.7
#: The first request and one more when fewer than ``PARAPHRASES_WANTED`` survive the filters.
PARAPHRASE_ATTEMPTS = 2
#: ``max_tokens`` of a request: two rewrites of a question are short.
MAX_REPLY_TOKENS = 1024
#: Bytes of a chat reply read (``post_capped``); a larger one counts as no answer.
REPLY_LIMIT_BYTES = 64 * 1024
#: Seconds one request may take (a small model on a CPU answers in seconds).
CHAT_TIMEOUT_SEC = 120.0

SAME_AS_PROMPT = "same_as_prompt"
LEAKS_ANSWER = "leaks_answer"

#: Request header and value ``soup serve`` honours, with a valid tool token, to answer a
#: loop check without recording it (``soup_cli.commands.serve`` repeats both; a test pins them).
#: A server that honoured it says so by echoing the header in its response.
INTERNAL_HEADER = "X-Soup-Internal"
INTERNAL_LOOP_CHECK = "loop-check"

#: The advice the notes give when no paraphraser is reachable or the answers are poor.
_FLAGS = "--paraphraser-endpoint <url> --paraphraser-model <name>"
MOSTLY_DROPPED = (
    "paraphrases mostly dropped: a small served model paraphrases poorly; "
    "set soup loop init --paraphraser-model"
)

# ---------------------------------------------------------------------------
# Words: what a correction added, and whether a text holds it (the consuming application's rules)
# ---------------------------------------------------------------------------

#: The share of the added words a text must hold to count as holding the answer.
THRESHOLD = 0.5
#: An edit that adds fewer content words than this is scored on all of its content words.
MIN_ADDED = 2

# One small fixed English stop list. Negations (not, no, never) and numbers are not on it:
# they can be the whole correction.
STOP_WORDS = frozenset(
    """
a an the and or but if so as of at by for in on to up with from into onto than then that this
these those is are was were be been being am do does did doing have has had having will would
can could shall should may might must it its it's i me my we our you your he him his she her
they them their what which who whom whose when where why how there here also just very too
about any some such own same each all both
""".split()
)

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE | re.ASCII)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE | re.ASCII)


def strip_think(text: str) -> str:
    """The text without its reasoning: closed ``<think>`` blocks go; a lone ``</think>`` means
    the opening was in the chat template, so what follows is the answer; an unclosed
    ``<think>`` is a thought cut off, so what precedes it is."""
    text = _THINK_BLOCK.sub("", text or "")
    # Offsets are taken on ``text`` itself: lower-casing first would move them (U+0130 lower-cases
    # to two code points) and cut the answer one character off.
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opening = _THINK_OPEN.search(text)
    if opening is not None:
        text = text[: opening.start()]
    return text.strip()


def content_words(text: str) -> frozenset:
    """Lowercase words without punctuation, minus the stop words."""
    return frozenset(
        word for word in _TOKEN.findall((text or "").lower()) if word not in STOP_WORDS
    )


def added_words(edit: str, original: str) -> frozenset:
    """What the correction added: the edit's content words the original answer lacked; all of
    the edit's content words when it adds fewer than two."""
    words = content_words(edit)
    added = words - content_words(original)
    return added if len(added) >= MIN_ADDED else words


def is_hit(text: str, words: frozenset) -> bool:
    """At least half of ``words`` are in ``text`` (its thinking removed). With no words to look
    for there is nothing to find: never a hit."""
    if not words:
        return False
    present = len(words & content_words(strip_think(text)))
    return present / len(words) >= THRESHOLD


# ---------------------------------------------------------------------------
# Paraphrasing one prompt
# ---------------------------------------------------------------------------

_LIST_ITEM = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*(.+?)\s*$")
_QUOTES = "\"'“”‘’`"


def parse_numbered_list(text: str) -> list:
    """The items of a numbered (or bulleted) list, one per line; anything else the model said
    around it is ignored."""
    items = []
    for line in (text or "").splitlines():
        found = _LIST_ITEM.match(line)
        item = found.group(1).strip().strip(_QUOTES).strip() if found else ""
        if item:
            items.append(item)
    return items


def paraphrase_key(text: str) -> str:
    """What two questions are compared by: casefolded, every punctuation character removed
    (Unicode punctuation and ASCII symbols such as ``$`` or ``+`` alike), whitespace collapsed
    to single spaces and stripped. "How  do I reset my router ?" and "How do I reset my
    router?" have the same key."""
    kept = (
        char
        for char in (text or "").casefold()
        if char not in string.punctuation and not unicodedata.category(char).startswith("P")
    )
    return " ".join("".join(kept).split())


def paraphrase_body(model: str, question: str, n: int = PARAPHRASES_WANTED) -> dict:
    """The chat request that asks for ``n`` paraphrases of ``question``."""
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": TEMPLATE_TEXT.format(n=n, question=question)}
        ],
        "temperature": PARAPHRASE_TEMPERATURE,
        "max_tokens": MAX_REPLY_TOKENS,
        "stream": False,
    }


class ParaphraserUnavailableError(Exception):
    """The paraphraser could not be asked; the message says why (no URL, no secret)."""


@dataclass
class Paraphrased:
    """What asking for the paraphrases of one prompt gave."""

    kept: list = field(default_factory=list)
    #: Paraphrases dropped, across both requests.
    dropped: int = 0
    #: Why paraphrases were dropped, each reason once, in the order it first happened.
    reasons: list = field(default_factory=list)
    #: How many were dropped for each reason.
    counts: dict = field(default_factory=dict)


def paraphrase(
    chat: Callable[[dict], str], model: str, question: str, added: frozenset
) -> Paraphrased:
    """Ask for two paraphrases of ``question`` in one numbered-list request (one more request
    when fewer than two survive; never a third).

    Dropped: one that is the question apart from case, punctuation and spacing
    (``same_as_prompt``, see :func:`paraphrase_key`), or that holds at least half of the
    ``added`` words of the correction (``leaks_answer``). The same rewording twice is one
    rewording and is not counted as dropped. ``chat`` takes the request body and returns the
    reply text; it raises :class:`ParaphraserUnavailableError` when the paraphraser cannot be asked.
    """
    result = Paraphrased()
    asked = paraphrase_key(question)
    seen = {asked}
    for _attempt in range(PARAPHRASE_ATTEMPTS):
        if len(result.kept) >= PARAPHRASES_WANTED:
            break
        reply = strip_think(chat(paraphrase_body(model, question)))
        for candidate in parse_numbered_list(reply):
            if len(result.kept) >= PARAPHRASES_WANTED:
                break
            key = paraphrase_key(candidate)
            reason = None
            if key == asked:
                reason = SAME_AS_PROMPT
            elif is_hit(candidate, added):
                reason = LEAKS_ANSWER
            if reason:
                result.dropped += 1
                result.counts[reason] = result.counts.get(reason, 0) + 1
                if reason not in result.reasons:
                    result.reasons.append(reason)
            elif key not in seen:
                seen.add(key)
                result.kept.append(candidate)
    return result


def last_user_message(messages: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The text of the last user message of a prompt: the question that is paraphrased."""
    for message in reversed(list(messages)):
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            return content
    return None


@dataclass
class ParaphraseRun:
    """What one pass over the edits did (for the iteration's notes)."""

    #: Edits that already had a current record and were not asked again.
    reused: int = 0
    #: Edits that needed paraphrases.
    edits: int = 0
    #: Edits the paraphraser answered for.
    asked: int = 0
    #: Edits that got a record, and the paraphrases in those records.
    written: int = 0
    kept: int = 0
    #: Paraphrases dropped by the filters, in all, and per reason.
    dropped: int = 0
    counts: dict = field(default_factory=dict)
    #: Edits the paraphraser answered for that were left without a record: nothing survived.
    empty: int = 0
    #: Edits not asked at all: the prompt has no user message to rewrite.
    unusable: int = 0
    #: Why the pass stopped early (the paraphraser or the file failed); ``None`` if it did not.
    stopped: Optional[str] = None

    @property
    def requested(self) -> int:
        """Paraphrases asked for: two per edit the paraphraser answered for."""
        return self.asked * PARAPHRASES_WANTED


def paraphrase_edit(
    edit: EditedAnswer,
    chat: Callable[[dict], str],
    *,
    model: str,
    created_at: Optional[str] = None,
    run: Optional[ParaphraseRun] = None,
) -> Optional[dict]:
    """The ``paraphrase`` record (``docs/loop-formats.md``) of one edit, or ``None``.

    Asks ``chat`` for paraphrases of the edit's prompt (its last user message), filtered as
    :func:`paraphrase` says. ``None`` when nothing survives (or the prompt has no user message):
    no line is written then. ``run``, when given, is updated with what was asked and dropped
    (the caller counts what it wrote).
    Raises :class:`ParaphraserUnavailableError` when the paraphraser cannot be asked.
    """
    question = last_user_message(edit.messages)
    try:
        digest = prompt_hash(edit.messages)
    except ValueError:
        digest = None
    if question is None or digest is None:
        if run is not None:
            run.unusable += 1
        return None
    found = paraphrase(chat, model, question, added_words(edit.edited, edit.original))
    if run is not None:
        run.asked += 1
        run.dropped += found.dropped
        for reason, count in found.counts.items():
            run.counts[reason] = run.counts.get(reason, 0) + count
    if not found.kept:
        if run is not None:
            run.empty += 1
        return None
    record: dict = {
        "schema_version": SCHEMA_VERSION,
        "response_id": edit.id,
        "prompt_hash": digest,
        "paraphrases": list(found.kept),
        "generator": {"model": model, "template": dict(TEMPLATE)},
        "created_at": created_at or utc_now_iso(),
    }
    if found.dropped:
        record["dropped"] = found.dropped
        record["dropped_reasons"] = list(found.reasons)
    return record


# ---------------------------------------------------------------------------
# Reading paraphrases.jsonl
# ---------------------------------------------------------------------------

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HASH = re.compile(r"[0-9a-f]{64}")


@dataclass
class ParaphraseLog:
    """A paraphrase log read: the record in force per ``response_id`` and what was skipped."""

    latest: dict = field(default_factory=dict)
    stats: LogStats = field(default_factory=LogStats)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _record(obj: dict) -> Optional[dict]:
    """``obj`` when it is a version 1 paraphrase record with what a reader needs, else ``None``."""
    version = obj.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        return None
    response_id, digest = obj.get("response_id"), obj.get("prompt_hash")
    if not (isinstance(response_id, str) and _UUID.fullmatch(response_id)):
        return None
    if not (isinstance(digest, str) and _HASH.fullmatch(digest)):
        return None
    paraphrases = obj.get("paraphrases")
    if not (isinstance(paraphrases, list) and paraphrases and all(_text(p) for p in paraphrases)):
        return None
    generator = obj.get("generator")
    template = generator.get("template") if isinstance(generator, dict) else None
    if not (
        isinstance(template, dict)
        and _text(generator.get("model"))
        and _text(template.get("id"))
        and _text(template.get("version"))
    ):
        return None
    if not _text(obj.get("created_at")):
        return None
    return obj


def read_paraphrases(path: Optional[str] = None) -> ParaphraseLog:
    """Read a paraphrase log by the contract's rules: the last line per ``response_id`` wins
    (file order). A missing file is an empty log; a line that is not a version 1 paraphrase
    record is skipped and counted in ``stats.skipped``. The read is bounded like the other
    logs (a line of at most 1 MiB, a file of at most 256 MiB)."""
    log = ParaphraseLog()
    target = PARAPHRASE_LOG if path is None else path
    if not os.path.isfile(target):
        return log
    for obj in _iter_objects(target, log.stats):
        record = _record(obj)
        if record is None:
            log.stats.skipped += 1
            continue
        log.latest[record["response_id"]] = record
    return log


def is_current(record: Optional[Mapping[str, Any]], edit: EditedAnswer) -> bool:
    """Whether ``record`` was written for this very prompt (same ``prompt_hash``)."""
    if record is None:
        return False
    try:
        return record.get("prompt_hash") == prompt_hash(edit.messages)
    except ValueError:
        return False


def current_paraphrases(log: ParaphraseLog, edits: Sequence[EditedAnswer]) -> dict:
    """The paraphrases per edit id for the edits whose record in force is current."""
    return {
        edit.id: list(log.latest[edit.id]["paraphrases"])
        for edit in edits
        if is_current(log.latest.get(edit.id), edit)
    }


# ---------------------------------------------------------------------------
# Who answers, and the HTTP call
# ---------------------------------------------------------------------------

_MAX_DETAIL = 200
#: What ``_post_chat`` reports (status 0) when the internal header was not echoed.
_UNCONFIRMED = "InternalCallNotConfirmed"


@dataclass(frozen=True)
class Paraphraser:
    """The chat endpoint that writes the paraphrases."""

    #: The validated base URL (loopback or LAN).
    endpoint: str
    #: The model name sent, and recorded as ``generator.model``.
    model: str
    #: True for the loop's own serve endpoint: the call carries ``X-Soup-Internal`` and the
    #: tool token so ``soup serve`` does not record it.
    internal: bool = False


def checked_endpoint(raw: object, label: str = "paraphraser endpoint") -> str:
    """``raw`` as a validated chat base URL, or ``ValueError`` saying why not.

    The same gate as ``SOUP_LOOP_SERVE_ENDPOINT``: ``validate_webhook_url`` (http or https, no
    control characters, no unspecified host, plain http only on loopback) and then loopback or
    private/link-local IP literals only: a paraphrase prompt is the user's own words and
    never leaves this machine or its LAN. A URL that carries credentials is refused.
    """
    from soup_cli.utils.drift_alarm import validate_webhook_url

    try:
        endpoint = validate_webhook_url(raw, allow_private_hosts=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} rejected ({exc})") from exc
    parsed = urlparse(endpoint)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not carry credentials in the URL")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must be a base URL without a query or fragment")
    if not _endpoint_is_local(endpoint):
        raise ValueError(f"{label} must be loopback/LAN (SSRF guard)")
    return endpoint


def chat_url(endpoint: str) -> str:
    """The chat-completions URL of a base URL: ``<base>/chat/completions`` when the base already
    ends in ``/v1`` (the OpenAI convention), else ``<base>/v1/chat/completions``."""
    base = endpoint.rstrip("/")
    if urlparse(base).path.rstrip("/").endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def resolve_paraphraser(state: Any) -> "tuple[Optional[Paraphraser], str]":
    """``(who answers, "")`` or ``(None, why nobody can)``.

    The ``paraphraser_endpoint`` and ``paraphraser_model`` of ``loop.yaml`` when set (both are:
    ``soup loop init`` refuses one without the other), else the loop's own serve endpoint
    (``SOUP_LOOP_SERVE_ENDPOINT``) with the tool token (``SOUP_TOOL_AUTH_TOKEN``) that
    ``soup serve`` needs to honour the internal header. The model recorded for the serve
    endpoint is the loop's stable side: the promoted adapter, else the served model.
    """
    endpoint, model = state.paraphraser_endpoint, state.paraphraser_model
    raw = os.environ.get("SOUP_LOOP_SERVE_ENDPOINT")
    served: Optional[str] = None
    if raw:
        try:
            served = checked_endpoint(raw, "SOUP_LOOP_SERVE_ENDPOINT")
        except ValueError as exc:
            if endpoint is None and model is None:
                return None, str(exc)
    if endpoint is not None or model is not None:
        if endpoint is None or model is None:
            return None, (
                "loop.yaml sets only one of paraphraser_endpoint and paraphraser_model "
                "(run soup loop init --force with both)"
            )
        try:
            chosen = checked_endpoint(endpoint, "paraphraser_endpoint")
        except ValueError as exc:
            return None, str(exc)
        # The loop's own server named as the paraphraser: its answers would be logged and
        # could come back as training data, so the call is made the internal way.
        if served is None or not _same_service(chosen, served):
            return Paraphraser(chosen, model), ""
        return _internal(served, model)
    if served is None:
        return None, "no paraphraser_endpoint in loop.yaml and no SOUP_LOOP_SERVE_ENDPOINT"
    return _internal(served, state.stable_name())


def _internal(served: str, model: str) -> "tuple[Optional[Paraphraser], str]":
    """The loop's own serve endpoint as the paraphraser: possible only with the tool token."""
    if not os.environ.get(TOOL_AUTH_TOKEN_ENV):
        return None, (
            "the serve endpoint takes an internal call only with its tool token and "
            f"{TOOL_AUTH_TOKEN_ENV} is not set"
        )
    return Paraphraser(served, model, internal=True), ""


_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _same_service(left: str, right: str) -> bool:
    """Whether two validated base URLs name the same server: the same scheme and port, and the
    same host (``localhost``, ``127.0.0.1`` and ``::1`` are one host)."""

    def key(url: str) -> "tuple[str, str, int]":
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if host in _LOOPBACK_NAMES:
            host = "loopback"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed.scheme, host, port

    try:
        return key(left) == key(right)
    except ValueError:  # an unparsable port
        return False


def _printable(text: object, limit: int = _MAX_DETAIL) -> str:
    """``text`` cut to ``limit`` characters with every control character replaced by a space."""
    cleaned = "".join(" " if ord(char) < 0x20 or ord(char) == 0x7F else char for char in str(text))
    return " ".join(cleaned.split())[:limit]


def _post_chat(
    url: str, headers: Mapping[str, str], body: Mapping[str, Any]
) -> "tuple[int, object]":
    """POST ``body`` to the chat endpoint; ``(HTTP status, parsed JSON or None)``.

    Status 0 means no usable answer (the call failed or the reply was larger than
    ``REPLY_LIMIT_BYTES``) and the second value is the exception's class name. The reply is
    read through ``post_capped``, never whole. A plain-http endpoint (loopback or LAN only)
    ignores HTTP_PROXY / ALL_PROXY, which would otherwise receive the tool token in clear text.
    A call that carried the internal header and got no echo of it back is status 0 too
    (``_UNCONFIRMED``): the server is older than the header and recorded the call.
    """
    try:
        from soup_cli.utils.http_response import post_capped

        resp = post_capped(
            url,
            max_bytes=REPLY_LIMIT_BYTES,
            endpoint_label="the paraphraser endpoint /v1/chat/completions",
            headers=dict(headers),
            json=dict(body),
            timeout=CHAT_TIMEOUT_SEC,
            trust_env=not url.lower().startswith("http://"),
        )
    except Exception as exc:  # noqa: BLE001 -- paraphrases must never crash the loop
        return 0, type(exc).__name__
    if (
        INTERNAL_HEADER in headers
        and 200 <= resp.status_code < 300
        and resp.headers.get(INTERNAL_HEADER) != INTERNAL_LOOP_CHECK
    ):
        return 0, _UNCONFIRMED
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, None


def _refusal(status: int, payload: object, paraphraser: Paraphraser) -> str:
    """Name why the endpoint did not answer."""
    if status == 0 and payload == _UNCONFIRMED:
        return (
            "the serve endpoint did not confirm the internal call, so it may have recorded it: "
            "restart soup serve from the same version as soup loop"
        )
    if status == 0:
        return f"no response from the paraphraser endpoint ({_printable(payload) or 'error'})"
    detail = ""
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        detail = _printable(payload["detail"])
    if status in (401, 403) and paraphraser.internal:
        return (
            f"the serve endpoint refused the internal call (HTTP {status}"
            f"{': ' + detail if detail else ''}): check SOUP_TOOL_AUTH_TOKEN against "
            "soup serve --tool-auth-token"
        )
    return f"the paraphraser endpoint answered HTTP {status}{': ' + detail if detail else ''}"


def make_chat(paraphraser: Paraphraser) -> Callable[[dict], str]:
    """A ``chat(body) -> reply text`` function that asks ``paraphraser``.

    The internal call carries ``X-Soup-Internal: loop-check`` and ``Authorization: Bearer
    <SOUP_TOOL_AUTH_TOKEN>``; a call to another endpoint carries no credential. Raises
    :class:`ParaphraserUnavailableError` (when called) for no answer, an HTTP error or a reply
    without a message.
    """
    url = chat_url(paraphraser.endpoint)
    headers: dict = {}
    if paraphraser.internal:
        headers[INTERNAL_HEADER] = INTERNAL_LOOP_CHECK
        token = os.environ.get(TOOL_AUTH_TOKEN_ENV)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    def chat(body: dict) -> str:
        status, payload = _post_chat(url, headers, body)
        if not 200 <= status < 300:
            raise ParaphraserUnavailableError(_refusal(status, payload, paraphraser))
        try:
            content = payload["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            content = None
        if not isinstance(content, str):
            raise ParaphraserUnavailableError("the paraphraser endpoint's reply holds no message")
        return content

    return chat


# ---------------------------------------------------------------------------
# The pass the train stage runs
# ---------------------------------------------------------------------------


def write_paraphrases(
    edits: Sequence[EditedAnswer],
    chat: Callable[[dict], str],
    *,
    model: str,
    path: Optional[str] = None,
    now: Optional[Callable[[], str]] = None,
    log: Optional[ParaphraseLog] = None,
) -> ParaphraseRun:
    """Append a paraphrase record for every edit that has no current one.

    An edit whose record in force (the last line for its id) was written for this very prompt
    is reused and never asked again. The pass ends at the first failure to ask (the paraphraser
    is down or refuses: it would be for the rest as well) or to write the file, and says why in
    ``stopped``; it never raises for those. ``log`` is the file as already read, when it was.
    """
    target = PARAPHRASE_LOG if path is None else path
    clock = now or utc_now_iso
    if log is None:
        log = read_paraphrases(target)
    missing = [edit for edit in edits if not is_current(log.latest.get(edit.id), edit)]
    run = ParaphraseRun(reused=len(edits) - len(missing), edits=len(missing))
    for edit in missing:
        try:
            record = paraphrase_edit(edit, chat, model=model, created_at=clock(), run=run)
        except ParaphraserUnavailableError as exc:
            run.stopped = str(exc)
            break
        if record is None:
            continue
        try:
            append_json_line(target, record)
        except (OSError, ValueError, TypeError) as exc:
            _LOG.warning("paraphrases.jsonl not written: %s", type(exc).__name__)
            run.stopped = f"paraphrases.jsonl not written ({type(exc).__name__})"
            break
        run.written += 1
        run.kept += len(record["paraphrases"])
    return run


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def run_notes(run: ParaphraseRun) -> list:
    """The notes of a pass: what was written and dropped, why it stopped, and the warning when
    more than half of the paraphrases asked for were dropped."""
    if not run.edits:
        return []
    text = f"paraphrases: {run.kept} new for {_plural(run.written, 'edit')}"
    if run.dropped:
        reasons = ", ".join(
            f"{reason} {run.counts[reason]}"
            for reason in (SAME_AS_PROMPT, LEAKS_ANSWER)
            if run.counts.get(reason)
        )
        text += f", {run.dropped} dropped ({reasons})"
    nothing = run.empty + run.unusable
    if nothing:
        text += f", {_plural(nothing, 'edit')} left without paraphrases"
    if run.stopped:
        unasked = max(run.edits - run.asked - run.unusable, 0)
        text += f". Stopped, {_plural(unasked, 'edit')} not asked: {run.stopped}"
    notes = [text]
    if run.requested and run.dropped * 2 > run.requested:
        notes.append(MOSTLY_DROPPED)
    return notes


def paraphrase_edits(
    state: Any,
    edits: Sequence[EditedAnswer],
    *,
    path: Optional[str] = None,
    now: Optional[Callable[[], str]] = None,
) -> list:
    """Write the paraphrases the edits still lack and return the notes for the iteration.

    The step of the pre-wired train stage that runs before training. Edits that all have
    current records give no note (nothing was done). With nobody to ask it writes nothing and
    the note names the reason and ``soup loop init --paraphraser-endpoint`` /
    ``--paraphraser-model``. Never raises.
    """
    target = PARAPHRASE_LOG if path is None else path
    log = read_paraphrases(target)
    pending = [edit for edit in edits if not is_current(log.latest.get(edit.id), edit)]
    if not pending:
        return []
    paraphraser, why = resolve_paraphraser(state)
    if paraphraser is None:
        return [
            f"paraphrases: none written for {_plural(len(pending), 'edit')}: {why}. "
            f"Pass soup loop init {_FLAGS}, or give the loop its own serve endpoint "
            f"(SOUP_LOOP_SERVE_ENDPOINT and {TOOL_AUTH_TOKEN_ENV})"
        ]
    run = write_paraphrases(
        edits, make_chat(paraphraser), model=paraphraser.model, path=target, now=now, log=log
    )
    return run_notes(run)
