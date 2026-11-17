"""Shared redaction for text and configs that are written to disk or handed to someone.

Used by the crash bundle (``utils/crash.py``) and by ``.can`` packing
(``cans/pack.py``), so both drop the same things:

* URLs: userinfo (``scheme://user:secret@host``, also ``//user:secret@host/x`` and
  ``user:secret@host/x``) and sensitive query or fragment parameters (``token``,
  ``api_key``, ``X-Amz-Signature``, ...).
* Credential shapes in free text: Hugging Face, ``sk-`` / ``sk-ant-`` keys, GitHub,
  AWS access key ids, Google and Slack keys, ``Authorization`` / ``Bearer`` headers, JWTs
  and PEM private-key blocks.
* ``KEY=value`` / ``key: value`` / ``"key": "value"`` pairs whose KEY NAME says the value
  is a credential, in text and in structured configs (nested dicts and lists). This is
  what catches an AWS secret access key and a W&B key, which have no shape of their own.

Every pattern is linear: no nested quantifiers over overlapping classes, every repeat is
bounded, possessive, or runs over a character class that cannot also match the next token,
and a pattern is skipped when the literal it needs is absent from the text. Stdlib only,
so importing this stays cheap.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

REDACTED = "<redacted>"


@dataclass
class RedactionReport:
    """What a redaction pass changed: how many values, and which config key paths.

    ``paths`` holds key paths only (``data.sources[0].api_key``), never a value.
    """

    count: int = 0
    paths: list[str] = field(default_factory=list)
    # membership test for ``add_path``: a scan of ``paths`` per call would be quadratic
    _seen: set[str] = field(default_factory=set, repr=False, compare=False)

    def add_path(self, path: str) -> None:
        if path and path not in self._seen:
            self._seen.add(path)
            self.paths.append(path)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

# Normalised (lower-case, no ``_`` / ``-`` / ``.`` / space) query-parameter names.
_SENSITIVE_PARAMS = frozenset({
    "key", "token", "secret", "password", "passwd", "pwd", "pass", "auth",
    "authorization", "sig", "signature", "credential", "credentials", "session",
    "sessionid",
})
_SENSITIVE_PARAM_SUFFIXES = (
    "token", "secret", "password", "passwd", "apikey", "accesskey", "secretkey",
    "privatekey", "signature", "credential", "credentials",
)
_PARAM_NOISE_RE = re.compile(r"[_\-.\s]")


def _is_sensitive_param(name: str) -> bool:
    flat = _PARAM_NOISE_RE.sub("", unquote(name)).lower()
    return flat in _SENSITIVE_PARAMS or flat.endswith(_SENSITIVE_PARAM_SUFFIXES)


def _mask_params(raw: str) -> str:
    """``a=1&token=x`` -> ``a=1&token=<redacted>``; names and other values stay."""
    if "=" not in raw:
        return raw
    pairs = []
    for pair in raw.split("&"):
        name, sep, value = pair.partition("=")
        if sep and value and _is_sensitive_param(name):
            pair = f"{name}={REDACTED}"
        pairs.append(pair)
    return "&".join(pairs)


def redact_url(url: str, *, keep_query: bool = False) -> str:
    """``url`` made safe to print, log or store. Raises ``ValueError`` if it cannot be parsed.

    By default (the judge's behaviour, ``eval/judge.py``) the userinfo, query and
    fragment are all dropped. With ``keep_query=True`` the URL keeps its shape for
    diagnosis: the userinfo becomes ``<redacted>`` and only the values of sensitive
    query / fragment parameters are replaced.
    """
    parts = urlsplit(str(url))
    if keep_query:
        netloc = parts.netloc
        if "@" in netloc:
            netloc = f"{REDACTED}@{netloc.rpartition('@')[2]}"
        query = _mask_params(parts.query)
        fragment = _mask_params(parts.fragment)
        if (netloc, query, fragment) == (parts.netloc, parts.query, parts.fragment):
            return str(url)
        return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


# A URL inside free text: stops at whitespace, quotes, backticks and backslashes. The scheme
# may be long (``postgresql+psycopg2``); the look-behind keeps a match from starting inside
# a run of scheme characters, so a long run is scanned once.
_URL_RE = re.compile(r"(?<![A-Za-z0-9+.\-])[A-Za-z][A-Za-z0-9+.\-]{1,40}://[^\s\"'`\\]+")
_URL_TRAILING = ".,;:)]}"

# Userinfo with no scheme in front. ``//user:secret@host/x`` (scheme-relative) is matched on
# the ``//``: a scan stops at the next ``/``, ``@`` or space, so scans never overlap and a long
# userinfo is handled. ``user:secret@host/x`` and ``user:secret@host:5432`` are matched only
# when a path or a port follows the host, so an e-mail address or ``git@github.com:org/repo``
# is never touched.
_REL_USERINFO_RE = re.compile(
    r"(?<![A-Za-z0-9+.\-])(?P<lead>//)[^/@\s\"'`\\<>]+@(?=[A-Za-z0-9\[])"
)
_BARE_USERINFO_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-+/@%])[A-Za-z0-9_.\-]{1,64}:[^\s@/:<>\"'`\\]{1,512}"
    r"@(?=[A-Za-z0-9][A-Za-z0-9.\-]{0,252}(?:/|:[0-9]{1,5}(?![A-Za-z0-9])))"
)


# A query string with no scheme in front: urllib3 / requests put ``url: /path?x=1&sig=...`` into
# the message of a failed request. The ``?`` has to follow a path or name character and be
# followed by ``name=``, so prose such as "what?" is left alone; the run ends at whitespace, a
# quote, a bracket or a fragment, and is consumed once.
_QUERY_RE = re.compile(r"(?<=[A-Za-z0-9_\-./~%])\?(?=[A-Za-z0-9_.\-%]{1,64}=)[^\s\"'`\\<>#]*")


def _redact_urls(text: str, report: RedactionReport) -> str:
    def _sub(match: re.Match[str]) -> str:
        raw = match.group(0)
        core = raw.rstrip(_URL_TRAILING)
        tail = raw[len(core):]
        try:
            new = redact_url(core, keep_query=True)
        except ValueError:
            # urlsplit refuses it: the query and userinfo passes below still apply to it
            return raw
        if new != core:
            report.count += max(1, new.count(REDACTED) - core.count(REDACTED))
        return new + tail

    def _sub_userinfo(match: re.Match[str]) -> str:
        report.count += 1
        return f"{match.groupdict().get('lead') or ''}{REDACTED}@"

    def _sub_query(match: re.Match[str]) -> str:
        raw = match.group(0)
        core = raw.rstrip(_URL_TRAILING)
        tail = raw[len(core):]
        new = "?" + _mask_params(core[1:])
        if new != core:
            report.count += max(1, new.count(REDACTED) - core.count(REDACTED))
        return new + tail

    if "://" in text:
        text = _URL_RE.sub(_sub, text)
    if "?" in text:
        text = _QUERY_RE.sub(_sub_query, text)
    if "@" in text:
        text = _BARE_USERINFO_RE.sub(_sub_userinfo, _REL_USERINFO_RE.sub(_sub_userinfo, text))
    return text


# ---------------------------------------------------------------------------
# Credential shapes in free text
# ---------------------------------------------------------------------------

_Q = r"\\?[\"']?"  # an optional (possibly backslash-escaped) quote


class _Shape(NamedTuple):
    """One credential shape: ``hints`` are lower-case literals, one of which must be in the
    text for ``pattern`` to run; a ``keep`` group is retained and the rest replaced."""

    name: str
    hints: tuple
    pattern: "re.Pattern[str]"


_SHAPES = (
    _Shape(
        "hf", ("hf_",),
        re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    ),
    _Shape(
        "openai", ("sk-",),
        re.compile(r"(?<![A-Za-z0-9])sk-(?!ant-)[A-Za-z0-9_\-]{20,}"),
    ),
    _Shape(
        "sk_ant", ("sk-ant-",),
        re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{20,}"),
    ),
    _Shape(
        "github", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
        re.compile(
            r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
        ),
    ),
    _Shape(
        "aws_access_key_id", ("akia", "asia"),
        re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"),
    ),
    _Shape(
        "google_api_key", ("aiza",),
        re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])"),
    ),
    _Shape(
        "slack", ("xox",),
        re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9\-]{10,}"),
    ),
    _Shape(
        "bearer", ("bearer",),
        re.compile(
            r"(?P<keep>(?<![A-Za-z0-9])bearer[ \t]+)[A-Za-z0-9._~+/=\-]{16,}", re.IGNORECASE
        ),
    ),
    _Shape(
        "authorization_header", ("authorization",),
        re.compile(
            r"(?P<keep>(?<![A-Za-z0-9])authorization" + _Q + r"[ \t]*[:=][ \t]*" + _Q
            + r"(?:(?:bearer|basic|digest|negotiate|token)[ \t]+)?)[^\s\"',;\\]{8,}",
            re.IGNORECASE,
        ),
    ),
    _Shape(
        "jwt", ("eyj",),
        re.compile(
            r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{5,}\.eyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]*"
        ),
    ),
)


def redact_token_shapes(text: str, report: Optional[RedactionReport] = None) -> str:
    """Replace the credential shapes in ``_SHAPES`` (URLs, PEM and key=value pairs excluded)."""
    if not isinstance(text, str) or not text:
        return text
    counter = report if report is not None else RedactionReport()
    lowered = text.lower()
    for shape in _SHAPES:
        if not any(hint in lowered for hint in shape.hints):
            continue
        has_keep = "keep" in shape.pattern.groupindex

        def _sub(match: re.Match[str], has_keep: bool = has_keep) -> str:
            new = (match.group("keep") if has_keep else "") + REDACTED
            if new != match.group(0):
                counter.count += 1
            return new

        text = shape.pattern.sub(_sub, text)
    return text


# ---------------------------------------------------------------------------
# PEM private-key blocks (searched with str.find: a regex with a lazy body would be
# quadratic on many BEGIN lines and no END line)
# ---------------------------------------------------------------------------

_PEM_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----")
_PEM_END_MARK = "-----END "
_PEM_DASHES = "-----"


def _redact_pem(text: str, report: RedactionReport) -> str:
    if "PRIVATE KEY" not in text:
        return text
    pieces: list[str] = []
    pos = 0
    for match in _PEM_BEGIN_RE.finditer(text):
        if match.start() < pos:
            continue
        end_at = text.find(_PEM_END_MARK, match.end())
        stop = len(text)  # an END line that is missing: redact to the end of the text
        if end_at != -1:
            close = text.find(_PEM_DASHES, end_at + len(_PEM_END_MARK))
            if close != -1:
                stop = close + len(_PEM_DASHES)
        pieces.append(text[pos:match.start()])
        pieces.append(REDACTED)
        pos = stop
        report.count += 1
    if not pieces:
        return text
    pieces.append(text[pos:])
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Key names that say "this value is a credential"
# ---------------------------------------------------------------------------

# ``{s}`` is the separator allowed inside a two-word name: ``_`` in a normalised name
# (``is_secret_key`` turns ``-``, ``.`` and camelCase into ``_`` first), any one non-
# alphanumeric character in raw text (``x-api-key``, ``client.api.key``). The text pre-filter
# below is built from the same words, so it can never skip a text the name test would flag.
_SECRET_WORDS = (
    r"token|secret|passw|pass{s}phrase|api{s}key|access{s}key|private{s}key|credential"
    r"|auth|bearer|cookie|session"
)
_SECRET_NAME_RE = re.compile(_SECRET_WORDS.replace("{s}", "_?"))
# Harmless names that contain one of the words above. Stripped before the test, so
# ``max_new_tokens``, ``tokenizer`` and ``auth_required`` stay readable while
# ``hf_token``, ``use_auth_token`` and ``admin_tokens`` do not.
_BENIGN_NAME_RE = re.compile(
    r"tokeniz[a-z0-9]*"
    r"|token_(?:count|counts|id|ids|len|length|limit|budget|ratio|type_ids)"
    r"|tokens_(?:per|seen|used|total)(?:_[a-z0-9]+)*"
    r"|(?<![a-z0-9])(?:max|min|num|n|new|total|special|added|extra|stop|reserved|prompt"
    r"|completion|input|output|response|generated|draft|speculative|assistant|think"
    r"|reasoning|judge|ignore|suppress)(?:_[a-z0-9]+){0,3}_tokens"
    r"|(?<![a-z0-9])(?:eos|bos|pad|unk|sep|cls|mask|special|stop|think)_token"
    r"|author(?!iz)"
    r"|authentic[a-z0-9]*"
    r"|^tokens$"
    r"|auth_(?:required|enabled|disabled|mode|type|method|scheme)"
    r"|(?:require|requires|use|enable|disable)_auth"
)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@functools.lru_cache(maxsize=4096)
def is_secret_key(name: str) -> bool:
    """Whether the config / environment key ``name`` says its value is a credential."""
    if not isinstance(name, str):
        return False
    flat = _NON_ALNUM_RE.sub("_", _CAMEL_RE.sub("_", name.strip()).lower())
    return bool(_SECRET_NAME_RE.search(_BENIGN_NAME_RE.sub("_", flat)))


# Values under a secret-named key that are plainly not a secret: switches, nulls, an
# auth scheme word, a reference to an environment variable.
_HARMLESS_LITERALS = frozenset({
    "", "true", "false", "none", "null", "nil", "yes", "no", "on", "off",
    "bearer", "basic", "digest", "negotiate", "[]", "{}", REDACTED,
})
_ENV_REF_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
# ``Bearer <redacted>``: an auth header the shape pass has already handled.
_SCHEME_REDACTED_RE = re.compile(
    r"(?:bearer|basic|digest|negotiate|token)[ \t]+" + re.escape(REDACTED), re.IGNORECASE
)


def _is_harmless_literal(value: str) -> bool:
    flat = value.strip()
    return (
        flat.lower() in _HARMLESS_LITERALS
        or bool(_ENV_REF_RE.fullmatch(flat))
        or bool(_SCHEME_REDACTED_RE.fullmatch(flat))
    )


def mask_secret_value(value: str) -> str:
    """``<redacted>`` for a string stored under a secret-named key, unless it is harmless."""
    return value if _is_harmless_literal(value) else REDACTED


# ``name`` + separator only: a name that is NOT a credential consumes just itself, so an
# assignment nested in its value is still seen. The value is read (``_VALUE_RE``) only
# for a credential-looking name, and a value that is read is consumed whole, so the work
# done for it is paid once. A name is at most 80 characters (a longer run of name
# characters is judged by its last 80), so a start costs a constant.
_ASSIGN_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_.\-]{1,80})"
    r"\\?[\"']?[ \t]{0,16}[:=][ \t]{0,16}(?=\S)"
)
# A value has no length cap (a cap would leave the tail of a long token in the text). Every
# repeat is possessive over characters that cannot also start the closing delimiter, so a
# value is read once and never backtracked into.
_QUOTED_RE = re.compile(
    r"\"(?:[^\"\\\n]|\\.)*+\"|'(?:[^'\\\n]|\\.)*+'"
    r"|\\\"(?:[^\"\\\n]|\\(?!\"))*+\\\""
)
# An unquoted value ends at whitespace, a quote or ``&``. ``, ; ) } ]`` end it only when the
# next character cannot continue a value: ``password=ab,cd`` is one value, ``token=ab, x=1`` is
# two. An ordinary call or list is not swallowed and a password that holds one is not cut.
_BARE_VALUE_RE = re.compile(
    r"(?:[^\s\"',;&)}\]\\]|\\(?![\"'])|[,;)}\]](?=[^\s\"',;&)}\]]))++"
)
# A quote that never closes (a text that was cut): the rest of the line is the value.
_UNCLOSED_RE = re.compile(r"\\?[\"'][^\n]*+")
_BLOCK_MARK_RE = re.compile(r"[|>][+\-0-9]*")
# ``/token=3/``: a partition directory, not a credential.
_PATH_NUMBER_RE = re.compile(r"[0-9]+(?:/|$)")
_BRACKET_SCAN_LIMIT = 65536
_ASSIGN_HINT_RE = re.compile(_SECRET_WORDS.replace("{s}", "[^a-z0-9]?"), re.IGNORECASE)


def _mask_assigned_value(value: str) -> str:
    """``value`` as captured (quotes included) with its content replaced."""
    if value.startswith('\\"'):
        quote = '\\"'
    elif value[0] in "\"'":
        quote = value[0]
    else:
        quote = ""
    inner = value[len(quote):len(value) - len(quote)] if quote else value
    if _is_harmless_literal(inner):
        return value
    return f"{quote}{REDACTED}{quote}"


def _bracket_end(text: str, start: int) -> int:
    """End of the ``[...]`` / ``{...}`` value that opens at ``start``.

    Quotes are honoured. The scan is bounded; a value that never closes ends at the end of
    its line (or at the bound) instead of being left in the text.
    """
    limit = min(len(text), start + _BRACKET_SCAN_LIMIT)
    depth = 0
    quote = ""
    index = start
    while index < limit:
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    newline = text.find("\n", start, limit)
    return limit if newline == -1 else newline


def _key_column(text: str, name_start: int) -> int:
    """Indentation a block scalar of the key at ``name_start`` has to exceed.

    A key that opens its line (``key:`` or ``- key:``) is measured at its own column. A key
    that follows other text (``RuntimeError: key: |`` in a message) is measured at the
    indentation of its line, since the lines of the block do not move with the prefix.
    """
    line_start = text.rfind("\n", 0, name_start) + 1
    prefix = text[line_start:name_start]
    if not prefix.strip(" \t-"):
        return len(prefix)
    return len(prefix) - len(prefix.lstrip(" \t"))


def _block_end(text: str, end: int, column: int) -> int:
    """End of the YAML block scalar whose indicator line ends at ``end``.

    The block is every following line that is indented deeper than the key (``column``);
    blank lines inside it belong to it, trailing ones do not.
    """
    last = end
    pos = end
    size = len(text)
    while pos < size and text[pos] in "\r\n":
        line_start = pos + (2 if text.startswith("\r\n", pos) else 1)
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = size
        line = text[line_start:line_end]
        if line.strip():
            if len(line) - len(line.lstrip(" \t")) <= column:
                break
            last = line_end
        pos = line_end
    return last


def _value_end(text: str, pos: int) -> int:
    """End of the value that starts at ``pos`` (``-1`` when nothing there is a value)."""
    if text.startswith(REDACTED, pos):
        return pos + len(REDACTED)  # already redacted: what follows it is not part of it
    first = text[pos]
    if first in "[{":
        return _bracket_end(text, pos)
    found = _QUOTED_RE.match(text, pos)
    if found is None:
        if first in "\"'\\":
            unclosed = _UNCLOSED_RE.match(text, pos)
            if unclosed is not None:
                return unclosed.end()
        found = _BARE_VALUE_RE.match(text, pos)
        if found is None:
            return -1
    end = found.end()
    # ``b'..'``, ``u"..."``, ``SecretStr('..')``: an unquoted run that stops at a quote opens
    # a quoted part, and ``'it''s'`` is one YAML string: take every quoted part that follows.
    while end < len(text) and text[end] in "\"'":
        more = _QUOTED_RE.match(text, end)
        if more is None:
            break
        end = more.end()
    return end


def _redact_assignments(text: str, report: RedactionReport) -> str:
    if not _ASSIGN_HINT_RE.search(text):
        return text
    pieces: list[str] = []
    pos = 0
    for match in _ASSIGN_RE.finditer(text):
        if match.start() < pos or not is_secret_key(match.group("name")):
            continue
        start = match.end()
        end = _value_end(text, start)
        if end < 0:
            continue
        value = text[start:end]
        if match.start() and text[match.start() - 1] == "/" and _PATH_NUMBER_RE.match(value):
            continue
        if _BLOCK_MARK_RE.fullmatch(value) and text[end:end + 1] in ("", "\n", "\r"):
            end = _block_end(text, end, _key_column(text, match.start()))
            value = text[start:end]
        masked = _mask_assigned_value(value)
        if masked == value:
            continue
        pieces.append(text[pos:start])
        pieces.append(masked)
        pos = end
        report.count += 1
    if not pieces:
        return text
    pieces.append(text[pos:])
    return "".join(pieces)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def redact_text(
    text: str, report: Optional[RedactionReport] = None, *, assignments: bool = True,
) -> str:
    """``text`` with credentials replaced by ``<redacted>``; not a string: returned as is.

    ``assignments=False`` skips the ``KEY=value`` pass (used for template source, where
    ``{% set auth = messages[0] %}`` is code, not a credential).
    """
    if not isinstance(text, str) or not text:
        return text
    counter = report if report is not None else RedactionReport()
    text = _redact_pem(text, counter)
    text = _redact_urls(text, counter)
    text = redact_token_shapes(text, counter)
    return _redact_assignments(text, counter) if assignments else text


def redact_config(
    value: Any, text_pass: Callable[[str, RedactionReport], str] = redact_text,
) -> tuple[Any, RedactionReport]:
    """A copy of a nested config with credentials replaced, plus what changed.

    Strings are redacted as text. A string under a secret-named key (see
    :func:`is_secret_key`) is replaced outright, and so is every string below a
    secret-named dict or list. Non-string values (numbers, booleans, ``None``) are
    never touched, and the input is not modified. ``text_pass`` redacts a plain string
    (a caller can add a layer of its own to ``redact_text``).
    """
    report = RedactionReport()
    return _walk(value, "", report, False, text_pass), report


# Config keys that hold a template instead of a setting (``data.chat_template``): a Jinja chat
# template is source code, so only the credential shapes are looked for in it.
_TEMPLATE_KEYS = frozenset({"chat_template"})


def _walk_template(value: str, path: str, report: RedactionReport) -> str:
    before = report.count
    new = redact_text(value, report, assignments=False)
    if report.count == before:
        return value
    report.add_path(path)
    return new


def find_redacted_paths(value: Any, limit: int = 50) -> list[str]:
    """Key paths of the strings in ``value`` that hold the ``<redacted>`` marker."""
    found: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(item, str):
            if REDACTED in item:
                found.append(path or "<root>")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return found


def _walk(
    value: Any, path: str, report: RedactionReport, secret: bool,
    text_pass: Callable[[str, RedactionReport], str],
) -> Any:
    if isinstance(value, str):
        if secret:
            new = mask_secret_value(value)
            if new == value:
                return value
            report.count += 1
        else:
            before = report.count
            new = text_pass(value, report)
            if report.count == before:
                return value
        report.add_path(path)
        return new
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if (not secret and isinstance(key, str) and isinstance(item, str)
                    and key.strip().lower() in _TEMPLATE_KEYS):
                out[key] = _walk_template(item, child, report)
                continue
            flagged = secret or is_secret_key(key)
            out[key] = _walk(item, child, report, flagged, text_pass)
        return out
    if isinstance(value, (list, tuple)):
        items = [
            _walk(item, f"{path}[{index}]", report, secret, text_pass)
            for index, item in enumerate(value)
        ]
        return tuple(items) if isinstance(value, tuple) else items
    return value
