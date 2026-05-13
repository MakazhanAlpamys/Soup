"""Data Pipeline Pro helpers (v0.42.0).

Schema-first surface for axolotl + LlamaFactory data parity. Live wiring for
heavy paths (S3/GCS loaders, AOT preprocess, PDF/DOCX ingestion, custom prompt
strategies) is intentionally deferred to v0.42.1 — mirrors v0.27.0 MII /
v0.37.0 multipack / v0.39.0 ReLoRA / v0.40.0 Part D / v0.41.0 LLaMA Pro
stub-then-live pattern.

Project policy followed:
- Bool rejected before int isinstance check (v0.30.0 Candidate / v0.34.0
  estimate_run_cost_usd / v0.36.0 Part D make_cache_key).
- Null-byte / oversize string rejection (v0.36.0 chat-template / v0.39.0
  Part E templates / v0.40.5 reward_model field validator).
- Frozen dataclasses for value objects (v0.32.0 SpikeRecoveryStrategy /
  v0.39.0 ReLoRAPolicy / v0.41.0 LrGroup).
- ``MappingProxyType`` for runtime-immutable registries (v0.36.0 _REGISTRY /
  v0.41.0 _OPTIMIZER_PACKAGES).
- ``is None`` over falsy guards (v0.34.0 / v0.39.0 / v0.40.6 policy).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import math
import os
import re
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)
from urllib.parse import urlparse

# --- Part A: New formats ---------------------------------------------------

# v0.42.0 new format strings on DataConfig.format Literal. Each is matched by
# an entry in ``soup_cli.data.formats._convert_*`` (Part A live wiring) and
# documented in the README's Data Formats section.
NEW_FORMATS_V0_42: FrozenSet[str] = frozenset({
    "prm",
    "pre_tokenized",
    "input_output",
    "video",
    "multimodal",
})


# --- Part B: Remote loading allowlist --------------------------------------

# Closed allowlist of remote URI schemes we recognise on ``DataConfig.train``.
# Each scheme requires a corresponding lazy-imported fsspec backend at load
# time. Live wiring of ``s3fs`` / ``gcsfs`` / ``adlfs`` / ``ocifs`` deferred to
# v0.42.1 — schema gate fires now so a misconfigured YAML fails loudly.
_REMOTE_SCHEMES: Mapping[str, str] = types.MappingProxyType({
    "s3": "s3fs",
    "gs": "gcsfs",
    "gcs": "gcsfs",
    "az": "adlfs",
    "abfs": "adlfs",
    "abfss": "adlfs",
    "oci": "ocifs",
})

# RFC 3986 — bucket / container names cannot contain control chars. We
# additionally reject characters that would let a crafted URL break out of the
# scheme://host/path pattern via embedded query / fragment / userinfo.
_BUCKET_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._\-]{0,62}$")

_MAX_REMOTE_PATH = 2048


def is_remote_uri(value: str) -> bool:
    """Return True iff ``value`` is one of the recognised remote schemes.

    Conservative — rejects bool (subclass of str via type-check below is not
    a concern, but defence-in-depth) and non-string values.
    """
    if not isinstance(value, str):
        return False
    if not value:
        return False
    if "://" not in value:
        return False
    scheme = value.split("://", 1)[0].lower()
    return scheme in _REMOTE_SCHEMES


def validate_remote_uri(value: str) -> str:
    """Validate ``value`` as a recognised remote URI.

    Returns the canonicalised URI (scheme lowercased) on success. Raises
    ``ValueError`` otherwise. Bucket / container name must match
    ``_BUCKET_RE`` to defeat URL-injection attacks against fsspec backends.
    """
    if not isinstance(value, str):
        raise ValueError("remote URI must be a string")
    if not value:
        raise ValueError("remote URI must not be empty")
    if "\x00" in value:
        raise ValueError("remote URI must not contain null bytes")
    if len(value) > _MAX_REMOTE_PATH:
        raise ValueError(f"remote URI must be <= {_MAX_REMOTE_PATH} chars")

    parsed = urlparse(value)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _REMOTE_SCHEMES:
        raise ValueError(
            f"remote URI scheme '{scheme}' not in allowlist: "
            f"{sorted(_REMOTE_SCHEMES)}"
        )
    if parsed.username or parsed.password:
        raise ValueError(
            "remote URI must not embed userinfo — use environment variables "
            "for cloud credentials instead"
        )
    if parsed.fragment:
        raise ValueError("remote URI must not contain a '#' fragment")
    if parsed.query:
        raise ValueError(
            "remote URI must not contain a '?' query string — fsspec backends "
            "interpret query parameters as configuration overrides (SSRF-adjacent)."
        )
    bucket = parsed.netloc
    if not bucket:
        raise ValueError("remote URI must include a bucket / container name")
    if not _BUCKET_RE.match(bucket):
        raise ValueError(
            f"remote URI bucket '{bucket}' must be alphanumeric + ._- "
            "(2-63 chars, leading alnum)"
        )
    return f"{scheme}://{bucket}{parsed.path}"


def required_remote_package(scheme: str) -> Optional[str]:
    """Return the pip-installable package name for a remote scheme."""
    if not isinstance(scheme, str):
        return None
    return _REMOTE_SCHEMES.get(scheme.lower())


# --- Part B: Streaming + sharding bounds -----------------------------------

# Tight bounds — beyond these the pipeline OOMs the DataLoader prefetch queue
# (cf. v0.32.0 GradAccumMonitor MAX_ACCUM=1024 policy).
_MIN_BUFFER_SIZE = 1
_MAX_BUFFER_SIZE = 1_000_000
_MIN_SHARDS = 1
_MAX_SHARDS = 1024


def validate_buffer_size(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("buffer_size must not be bool")
    if not isinstance(value, int):
        raise ValueError("buffer_size must be an integer")
    if value < _MIN_BUFFER_SIZE or value > _MAX_BUFFER_SIZE:
        raise ValueError(
            f"buffer_size must be in [{_MIN_BUFFER_SIZE}, {_MAX_BUFFER_SIZE}]"
        )
    return value


def validate_shards(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("shards must not be bool")
    if not isinstance(value, int):
        raise ValueError("shards must be an integer")
    if value < _MIN_SHARDS or value > _MAX_SHARDS:
        raise ValueError(f"shards must be in [{_MIN_SHARDS}, {_MAX_SHARDS}]")
    return value


# --- Part C: AOT preprocessing cache --------------------------------------

# Mirrors v0.36.0 batch_cache.json content-hash policy. Cache key includes
# tokenizer + max_length + format + dataset path → SHA-256 → cache filename.
def make_preprocess_cache_key(
    *,
    dataset_path: str,
    tokenizer_name: str,
    max_length: int,
    format_name: str,
) -> str:
    """Return a 16-char hex cache key for AOT-tokenized output.

    Defence-in-depth: every input is type-checked and bool-rejected so the
    SHA-256 input is always canonical. Mirrors v0.36.0 ``make_cache_key``.
    """
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("dataset_path must be a non-empty string")
    if "\x00" in dataset_path:
        raise ValueError("dataset_path must not contain null bytes")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError("tokenizer_name must be a non-empty string")
    if "\x00" in tokenizer_name:
        raise ValueError("tokenizer_name must not contain null bytes")
    if isinstance(max_length, bool):
        raise ValueError("max_length must not be bool")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive int")
    if not isinstance(format_name, str) or not format_name:
        raise ValueError("format_name must be a non-empty string")
    blob = f"{dataset_path}\x1f{tokenizer_name}\x1f{max_length}\x1f{format_name}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --- Part D: Interleave strategies ----------------------------------------

INTERLEAVE_STRATEGIES: FrozenSet[str] = frozenset({
    "concat", "under", "over", "probs",
})

_MAX_INTERLEAVE_DATASETS = 32


@dataclass(frozen=True)
class InterleaveSpec:
    """Frozen value object — interleave plan for multi-dataset training.

    ``probs`` (when ``strategy == 'probs'``) names the per-dataset sampling
    probability and must sum to 1.0 (±1e-6). For ``concat`` / ``under`` /
    ``over`` it is None.
    """

    strategy: str
    probs: Optional[Tuple[float, ...]]


def parse_interleave(
    raw: object, *, num_datasets: int,
) -> Optional[InterleaveSpec]:
    """Parse an interleave directive from a ``soup.yaml`` value.

    Accepts:
    - ``None`` → return ``None`` (legacy single-dataset path).
    - ``str`` (one of ``concat`` / ``under`` / ``over``) → ``InterleaveSpec``
      with probs=None.
    - ``dict`` of the form ``{"strategy": "probs", "probs": [...]}``.

    Raises ``ValueError`` on every other shape — silent acceptance was the
    Axolotl-mode footgun this validator is here to prevent.
    """
    if raw is None:
        return None
    if isinstance(num_datasets, bool):
        raise ValueError("num_datasets must not be bool")
    if not isinstance(num_datasets, int) or num_datasets < 2:
        raise ValueError(
            "interleave requires >= 2 datasets — use a single dataset instead"
        )
    if num_datasets > _MAX_INTERLEAVE_DATASETS:
        raise ValueError(
            f"interleave supports at most {_MAX_INTERLEAVE_DATASETS} datasets"
        )

    if isinstance(raw, str):
        if raw not in INTERLEAVE_STRATEGIES:
            raise ValueError(
                f"interleave strategy must be one of "
                f"{sorted(INTERLEAVE_STRATEGIES)} (got {raw!r})"
            )
        if raw == "probs":
            raise ValueError(
                "interleave='probs' requires a 'probs' list — use "
                "{strategy: probs, probs: [...]} dict form."
            )
        return InterleaveSpec(strategy=raw, probs=None)

    if isinstance(raw, dict):
        strategy = raw.get("strategy")
        probs = raw.get("probs")
        if strategy not in INTERLEAVE_STRATEGIES:
            raise ValueError(
                f"interleave.strategy must be one of "
                f"{sorted(INTERLEAVE_STRATEGIES)} (got {strategy!r})"
            )
        if strategy != "probs":
            if probs is not None:
                raise ValueError(
                    f"interleave strategy={strategy!r} must not set 'probs'"
                )
            return InterleaveSpec(strategy=strategy, probs=None)
        if not isinstance(probs, list) or len(probs) != num_datasets:
            raise ValueError(
                f"interleave.probs must be a list of length {num_datasets} "
                f"(got {type(probs).__name__})"
            )
        cleaned: List[float] = []
        for index, entry in enumerate(probs):
            if isinstance(entry, bool):
                raise ValueError(
                    f"interleave.probs[{index}] must not be bool"
                )
            if not isinstance(entry, (int, float)):
                raise ValueError(
                    f"interleave.probs[{index}] must be numeric"
                )
            entry_f = float(entry)
            if not math.isfinite(entry_f):
                raise ValueError(
                    f"interleave.probs[{index}] must be finite"
                )
            if entry_f <= 0.0 or entry_f > 1.0:
                raise ValueError(
                    f"interleave.probs[{index}] must be in (0.0, 1.0]"
                )
            cleaned.append(entry_f)
        if abs(sum(cleaned) - 1.0) > 1e-6:
            raise ValueError("interleave.probs must sum to 1.0 (±1e-6)")
        return InterleaveSpec(strategy="probs", probs=tuple(cleaned))

    raise ValueError(
        f"interleave must be None, a string, or a dict (got {type(raw).__name__})"
    )


# --- Part D: Image min/max pixels + video fps/maxlen + resize algorithm ---

IMAGE_RESIZE_ALGORITHMS: FrozenSet[str] = frozenset({
    "nearest", "bilinear", "bicubic", "lanczos",
})

_MAX_IMAGE_PIXELS = 1_073_741_824  # 1 GP — Pillow's default DOS guard.
_MAX_VIDEO_MAXLEN = 4096
_MAX_VIDEO_FPS = 240


def validate_image_pixels(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0 or value > _MAX_IMAGE_PIXELS:
        raise ValueError(f"{name} must be in (0, {_MAX_IMAGE_PIXELS}]")
    return value


def validate_video_maxlen(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("video_maxlen must not be bool")
    if not isinstance(value, int):
        raise ValueError("video_maxlen must be an integer")
    if value <= 0 or value > _MAX_VIDEO_MAXLEN:
        raise ValueError(f"video_maxlen must be in (0, {_MAX_VIDEO_MAXLEN}]")
    return value


def validate_video_fps(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("video_fps must not be bool")
    if not isinstance(value, (int, float)):
        raise ValueError("video_fps must be numeric")
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError("video_fps must be finite")
    if value_f <= 0.0 or value_f > _MAX_VIDEO_FPS:
        raise ValueError(f"video_fps must be in (0.0, {_MAX_VIDEO_FPS}]")
    return value_f


# --- Part E: Vocab expansion ----------------------------------------------

_MAX_NEW_TOKENS = 10_000  # ~ 30 MB embedding row * dim 8K → still RAM-safe.
_MAX_TOKEN_LEN = 256


def validate_new_tokens(value: Optional[List[str]]) -> Optional[List[str]]:
    """Validate ``add_new_tokens`` / ``new_special_tokens`` lists.

    Returns a new list (defensive copy) so caller mutation cannot leak into
    a stored Pydantic field. Mirrors v0.39.0 rank_pattern policy.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("must be a list of strings")
    if len(value) > _MAX_NEW_TOKENS:
        raise ValueError(f"too many tokens (max {_MAX_NEW_TOKENS})")
    cleaned: List[str] = []
    seen: set[str] = set()
    for index, token in enumerate(value):
        if not isinstance(token, str):
            raise ValueError(f"entry [{index}] must be a string")
        if not token:
            raise ValueError(f"entry [{index}] must not be empty")
        if "\x00" in token:
            raise ValueError(f"entry [{index}] must not contain null bytes")
        if len(token) > _MAX_TOKEN_LEN:
            raise ValueError(
                f"entry [{index}] must be <= {_MAX_TOKEN_LEN} chars"
            )
        if token in seen:
            raise ValueError(f"duplicate token {token!r}")
        seen.add(token)
        cleaned.append(token)
    return cleaned


# --- Part E: Custom prompt strategies -------------------------------------

# Module:fn syntax allowlist — mirrors how Axolotl exposes user transforms.
# Live runtime invocation deferred to v0.42.1; the schema validates the
# reference shape now so a typo fails loudly.
_PROMPT_STRATEGY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]{0,127}:[A-Za-z_][A-Za-z0-9_]{0,127}$"
)


def validate_prompt_strategy(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("prompt_strategy must be a string")
    if not value:
        raise ValueError("prompt_strategy must not be empty")
    if "\x00" in value:
        raise ValueError("prompt_strategy must not contain null bytes")
    # Regex allows up to 128 chars per side + colon = 257 chars; cap at 260
    # to absorb any future regex relaxation without mismatching the bounds.
    if len(value) > 260:
        raise ValueError("prompt_strategy must be <= 260 chars")
    if not _PROMPT_STRATEGY_RE.match(value):
        raise ValueError(
            "prompt_strategy must match 'module.path:function_name' "
            "(letters / digits / underscore / dot only)."
        )
    return value


# --- v0.53.7 #86: pre-tokenized Arrow shard loader ------------------------
#
# Reads a directory produced by ``soup data preprocess`` (Arrow shards from
# ``datasets.Dataset.save_to_disk``). Used by SFT + Pretrain wrappers to
# short-circuit tokenization when ``data.format='pre_tokenized'`` and
# ``data.tokenized_path`` is set. Containment + symlink TOCTOU defence
# enforced before the heavy ``datasets.load_from_disk`` call.

def load_pretokenized_dataset(
    tokenized_path: str,
    *,
    expected_cache_key: Optional[str] = None,
) -> Any:
    """Load a pre-tokenized Arrow dataset directory.

    Verifies cwd containment, rejects symlinks at the target path (TOCTOU —
    mirrors v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B policy), and
    cross-checks ``metadata.json`` cache_key when ``expected_cache_key`` is
    provided. Lazy-imports ``datasets`` so the helper itself loads on a
    fresh interpreter.

    Raises:
        TypeError: if ``tokenized_path`` is not a string.
        ValueError: on null bytes, non-existent path, out-of-cwd path,
            symlink at target, mismatched cache_key, or malformed metadata.
        ImportError: if ``datasets`` is not installed.
    """
    import stat as _stat

    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(tokenized_path, str):
        raise TypeError("tokenized_path must be a string")
    if not tokenized_path:
        raise ValueError("tokenized_path must be a non-empty string")
    if "\x00" in tokenized_path:
        raise ValueError("tokenized_path must not contain null bytes")
    real = os.path.realpath(tokenized_path)
    if not is_under_cwd(real):
        raise ValueError("tokenized_path must stay under cwd")
    # Lstat the ORIGINAL (unresolved) path to detect symlinks at the entry
    # point — mirrors v0.33.0 #22 / v0.43.0 Part C TOCTOU policy.
    try:
        lst = os.lstat(tokenized_path)
    except OSError as exc:
        raise ValueError(f"tokenized_path not found: {tokenized_path!r}") from exc
    if _stat.S_ISLNK(lst.st_mode):
        raise ValueError("tokenized_path must not be a symlink")
    if not os.path.isdir(real):
        raise ValueError(f"tokenized_path must be a directory: {tokenized_path!r}")

    metadata_path = os.path.join(real, "metadata.json")
    if os.path.isfile(metadata_path):
        try:
            import json as _json

            with open(metadata_path, encoding="utf-8") as f:
                metadata = _json.load(f)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"metadata.json in tokenized_path is unreadable: {exc}"
            ) from exc
        if (
            expected_cache_key is not None
            and metadata.get("cache_key") != expected_cache_key
        ):
            raise ValueError(
                f"cache_key mismatch: expected {expected_cache_key!r}, "
                f"got {metadata.get('cache_key')!r}"
            )

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError(
            "datasets is required for pre_tokenized short-circuit: "
            "pip install datasets"
        ) from exc
    return load_from_disk(real)


# --- v0.53.7 #87: prompt_strategy live resolver ---------------------------
#
# Resolves a validated ``module.path:function_name`` spec into a callable
# applied per-row at SFT format time. Validation surface (regex / null-byte /
# oversize) lives in ``validate_prompt_strategy`` above — this is the runtime
# resolver that lazy-imports the named module and verifies the callable
# signature accepts a single positional argument and returns a ``Mapping``.

_PROMPT_STRATEGY_LOG = logging.getLogger("soup_cli.utils.data_pipeline.prompt_strategy")


@functools.lru_cache(maxsize=64)
def resolve_prompt_strategy(spec: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Resolve ``module.path:fn_name`` into a callable.

    Lazy-imports the named module via :mod:`importlib`, fetches the
    referenced attribute, and verifies the callable accepts a single
    positional argument. Per-spec ``lru_cache`` so trainer loops do not pay
    the import cost on every row.

    Raises:
        TypeError: if ``spec`` is not a string.
        ValueError: if ``spec`` does not match the validator regex, the
            module cannot be imported, the attribute is missing, or the
            resolved object is not callable / has the wrong signature.
    """
    # Re-validate the shape — defence-in-depth so direct callers cannot
    # bypass the schema-validator that fires on YAML load.
    validate_prompt_strategy(spec)
    # v0.53.7 H-G: ``assert`` is stripped under ``python -O``. Replace with
    # an explicit TypeError so the type narrowing survives optimisation.
    if not isinstance(spec, str):
        raise TypeError(
            f"prompt_strategy must be a str, got {type(spec).__name__}"
        )

    module_path, _, fn_name = spec.partition(":")
    try:
        import importlib

        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"prompt_strategy module {module_path!r} could not be imported: {exc}"
        ) from exc
    if not hasattr(module, fn_name):
        raise ValueError(
            f"prompt_strategy {spec!r}: module {module_path!r} has no attribute "
            f"{fn_name!r}"
        )
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(
            f"prompt_strategy {spec!r}: resolved attribute is not callable "
            f"({type(fn).__name__})"
        )
    # Best-effort signature check — builtins / C-extensions may not expose a
    # signature; in that case we skip the check and rely on the per-row
    # exception swallow.
    # v0.53.7 H-E: narrow the catch to TypeError (signature unavailable for
    # builtins / C-extensions). The ValueError raised below is the
    # signature-shape error we explicitly want to surface — swallowing it
    # would let a bad callable through.
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Signature unavailable (builtin / C-extension); skip the check.
        return fn
    positional_count = 0
    has_var_positional = False
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                positional_count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
    if not has_var_positional and positional_count > 1:
        raise ValueError(
            f"prompt_strategy {spec!r}: callable must accept at most one "
            f"required positional argument (found {positional_count})"
        )
    return fn


def apply_prompt_strategy(
    spec: Optional[str], row: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Apply the resolved prompt_strategy to ``row`` if ``spec`` is set.

    Mirrors the v0.33.0 #47 ``CrossDocCollator`` silent-degrade policy: a
    per-row callable exception is logged at DEBUG and the original row is
    returned unchanged (defence so a single transform bug doesn't crash a
    multi-hour training run).
    """
    if spec is None:
        return row
    try:
        fn = resolve_prompt_strategy(spec)
    except (TypeError, ValueError):
        # Hard config errors must surface at trainer setup time, not silently
        # at every row. We re-raise here.
        raise
    try:
        result = fn(row)
    except Exception as exc:  # noqa: BLE001 — user callable can raise anything
        _PROMPT_STRATEGY_LOG.debug(
            "prompt_strategy %r raised on row: %s", spec, exc
        )
        return row
    if not isinstance(result, Mapping):
        _PROMPT_STRATEGY_LOG.debug(
            "prompt_strategy %r returned non-Mapping (%s); using original row",
            spec,
            type(result).__name__,
        )
        return row
    return result


# --- Part F: Document ingestion -------------------------------------------

INGEST_EXTENSIONS: FrozenSet[str] = frozenset({".pdf", ".docx", ".md", ".txt"})


# v0.53.7 #88 — markdown heading-aware splitter.
#
# Splits a markdown document on ATX headings (``^#{1,6}\s``) and emits one
# record per section. A preamble before the first heading is emitted as a
# record with ``section=None`` / ``level=None``. Pure function — no I/O.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_MAX_MD_SECTIONS = 10_000


def split_markdown_by_headings(text: str) -> List[Mapping[str, Any]]:
    """Split markdown ``text`` on ATX headings into section records.

    Each output record has three keys: ``section`` (heading text, or ``None``
    for preamble), ``level`` (1-6, or ``None`` for preamble), and ``text``
    (body content following the heading; may be empty).

    Trailing empty whitespace-only sections are emitted as-is; the caller
    decides whether to keep or drop. Capped at ``_MAX_MD_SECTIONS`` to defend
    against pathological inputs.

    Raises ``TypeError`` on non-string input.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lines = text.splitlines()
    sections: List[Mapping[str, Any]] = []
    current_section: Optional[str] = None
    current_level: Optional[int] = None
    body: List[str] = []

    def _flush() -> None:
        if current_section is None and not body:
            return
        sections.append(
            {
                "section": current_section,
                "level": current_level,
                "text": "\n".join(body).strip(),
            }
        )

    for line in lines:
        m = _MD_HEADING_RE.match(line)
        if m:
            _flush()
            if len(sections) >= _MAX_MD_SECTIONS:
                break
            current_section = m.group(2).strip()
            current_level = len(m.group(1))
            body = []
        else:
            body.append(line)
    _flush()
    return sections


def detect_ingest_format(path: str) -> str:
    """Return the canonical ingest format for ``path`` based on extension.

    Raises ``ValueError`` on unsupported extensions so the CLI can present a
    friendly error before the lazy import path triggers a confusing
    ImportError on a missing optional dep.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    ext = os.path.splitext(path)[1].lower()
    if ext not in INGEST_EXTENSIONS:
        raise ValueError(
            f"ingest only supports {sorted(INGEST_EXTENSIONS)} — got {ext!r}"
        )
    if ext == ".md":
        return "markdown"
    return ext.lstrip(".")


# --- Public introspection helpers (for tests + docs) ----------------------

def remote_schemes() -> Tuple[str, ...]:
    """Return a sorted, immutable view of recognised remote schemes."""
    return tuple(sorted(_REMOTE_SCHEMES))


def interleave_strategies() -> Tuple[str, ...]:
    return tuple(sorted(INTERLEAVE_STRATEGIES))


def new_formats() -> Tuple[str, ...]:
    return tuple(sorted(NEW_FORMATS_V0_42))


def ingest_extensions() -> Tuple[str, ...]:
    return tuple(sorted(INGEST_EXTENSIONS))


__all__ = [
    "InterleaveSpec",
    "NEW_FORMATS_V0_42",
    "INTERLEAVE_STRATEGIES",
    "IMAGE_RESIZE_ALGORITHMS",
    "INGEST_EXTENSIONS",
    "is_remote_uri",
    "validate_remote_uri",
    "required_remote_package",
    "validate_buffer_size",
    "validate_shards",
    "make_preprocess_cache_key",
    "load_pretokenized_dataset",
    "parse_interleave",
    "validate_image_pixels",
    "validate_video_maxlen",
    "validate_video_fps",
    "validate_new_tokens",
    "validate_prompt_strategy",
    "resolve_prompt_strategy",
    "apply_prompt_strategy",
    "detect_ingest_format",
    "split_markdown_by_headings",
    "remote_schemes",
    "interleave_strategies",
    "new_formats",
    "ingest_extensions",
]


# Generator-friendly probe used by tests — confirms the module loads on a
# fresh interpreter without importing torch / transformers / fsspec.
def _selfcheck() -> Dict[str, int]:
    return {
        "new_formats": len(NEW_FORMATS_V0_42),
        "remote_schemes": len(_REMOTE_SCHEMES),
        "interleave_strategies": len(INTERLEAVE_STRATEGIES),
        "image_resize_algorithms": len(IMAGE_RESIZE_ALGORITHMS),
        "ingest_extensions": len(INGEST_EXTENSIONS),
    }


def _iter_immutable(items: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(items))
