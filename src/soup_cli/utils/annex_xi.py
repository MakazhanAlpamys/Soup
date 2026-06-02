"""EU AI Act Annex XI/XII auto-doc generator (v0.59.0 Part C).

Pure-stdlib markdown renderer. Live PDF generation deferred to v0.59.1
(``reportlab`` integration); the markdown text shipped here is the
canonical source-of-truth that future PDF / DOCX exporters can transform.

Annex XI Section 1 covers model description + intended purpose; Section
2 covers training process + data + compute. Annex XII (Article 53(1)(d))
is the public summary for GPAI providers — top-10 % crawled domains,
modality breakdown, training compute.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Tuple
from urllib.parse import urlsplit

from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink

_LOG = logging.getLogger(__name__)

_MAX_NAME = 256
_MAX_TEXT = 16384
_VALID_SECTIONS = ("xi", "xii")

# #184 — top-domain extraction caps (DoS defence).
_MAX_DOMAIN_ROWS = 200_000
_MAX_ROW_CHARS = 1_000_000
_MAX_URLS_PER_ROW = 5_000
_MAX_JSONL_BYTES = 1024 * 1024 * 1024  # 1 GiB
# Linear-time, bounded URL matcher (ReDoS-safe — no nested quantifiers).
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]{1,2048}")
_TEXT_FIELDS = ("text", "content", "output", "response", "prompt", "instruction")


def _validate_text(value: str, field_name: str, *, max_len: int = _MAX_NAME) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field_name} too int ({len(value)} > {max_len})")
    return value


def _validate_non_negative(value: float, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be bool")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{field_name} must be finite")
    if f < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return f


@dataclass(frozen=True)
class AnnexXIData:
    """Per-run Annex XI/XII input."""

    model_name: str
    base_model: str
    task: str
    dataset_summary: str
    modalities: Tuple[str, ...]
    train_compute_flops: float
    train_energy_kwh: float
    train_co2_kg: float
    top_domains: Tuple[Tuple[str, float], ...]
    soup_version: str
    run_id: str
    created_at: str

    def __post_init__(self) -> None:
        _validate_text(self.model_name, "model_name")
        _validate_text(self.base_model, "base_model")
        _validate_text(self.task, "task", max_len=64)
        _validate_text(self.dataset_summary, "dataset_summary", max_len=_MAX_TEXT)
        if not isinstance(self.modalities, tuple) or not self.modalities:
            raise ValueError("modalities must be a non-empty tuple")
        for m in self.modalities:
            _validate_text(m, "modalities[*]", max_len=32)
        _validate_non_negative(self.train_compute_flops, "train_compute_flops")
        _validate_non_negative(self.train_energy_kwh, "train_energy_kwh")
        _validate_non_negative(self.train_co2_kg, "train_co2_kg")
        if not isinstance(self.top_domains, tuple):
            raise ValueError("top_domains must be a tuple")
        for entry in self.top_domains:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError("each top_domains entry must be a (domain, share) tuple")
            domain, share = entry
            _validate_text(domain, "top_domains[domain]", max_len=256)
            _validate_non_negative(share, "top_domains[share]")
        _validate_text(self.soup_version, "soup_version", max_len=32)
        _validate_text(self.run_id, "run_id", max_len=64)
        _validate_text(self.created_at, "created_at", max_len=64)


def _format_flops(flops: float) -> str:
    if isinstance(flops, bool):
        raise ValueError("flops must not be bool")
    if not isinstance(flops, (int, float)) or not math.isfinite(float(flops)):
        raise ValueError("flops must be a finite number")
    if flops <= 0:
        return "0"
    exp = int(math.floor(math.log10(flops)))
    mant = flops / (10 ** exp)
    return f"{mant:.2f}e{exp}"


# Markdown-active chars that can break downstream PDF/HTML renderers.
# Matches v0.29.0 model-card v2 escaping policy: neutralise `|[](){}!<>` plus
# newline/CR so an operator-controlled model_name with `\n## Forged Section`
# cannot inject a forged heading into the rendered document.
_MD_ESCAPE_PATTERN = re.compile(r"([|\[\]()!<>])")


def _md_escape(value: str) -> str:
    """Neutralise markdown-active chars in operator-supplied strings."""
    if not isinstance(value, str):
        return ""
    # Replace control chars (newline/tab/CR) with spaces — defends against
    # forged-heading injection inside an interpolated field.
    cleaned = "".join(ch if ch >= " " or ch == "\t" else " " for ch in value)
    cleaned = cleaned.replace("\t", " ")
    return _MD_ESCAPE_PATTERN.sub(r"\\\1", cleaned)


def _build_domains_block(data: AnnexXIData) -> str:
    """Render the top-10 domains as a markdown list (shared by XI + XII)."""
    return "\n".join(
        f"- {_md_escape(domain)}: {share:.2%}"
        for domain, share in data.top_domains[:10]
    ) or "_(no domains recorded)_"


def _build_modalities(data: AnnexXIData) -> str:
    return ", ".join(_md_escape(m) for m in data.modalities)


def render_annex_xi_markdown(data: AnnexXIData) -> str:
    """Render Annex XI Section 1 + Section 2 as markdown."""
    if not isinstance(data, AnnexXIData):
        raise TypeError(f"data must be AnnexXIData, got {type(data).__name__}")
    domains_block = _build_domains_block(data)
    modalities = _build_modalities(data)
    dataset_summary = _md_escape(data.dataset_summary) if data.dataset_summary else ""
    return f"""# Annex XI — Technical Documentation

_Generated by soup-cli {_md_escape(data.soup_version)} at {_md_escape(data.created_at)}._

## Section 1 — Model Description

- **Model name:** {_md_escape(data.model_name)}
- **Base model:** {_md_escape(data.base_model)}
- **Task:** {_md_escape(data.task)}
- **Run id:** {_md_escape(data.run_id)}
- **Modalities:** {modalities}

## Section 2 — Training Process + Data

- **Training compute (FLOPs):** {_format_flops(data.train_compute_flops)}
- **Energy consumed:** {data.train_energy_kwh:.3f} kWh
- **Estimated CO₂ emissions:** {data.train_co2_kg:.3f} kg

### Dataset summary

{dataset_summary or "_(no dataset summary supplied)_"}

### Top-10 domains in training corpus

{domains_block}
"""


def render_annex_xii_markdown(data: AnnexXIData) -> str:
    """Render Annex XII (Article 53(1)(d)) public training summary."""
    if not isinstance(data, AnnexXIData):
        raise TypeError(f"data must be AnnexXIData, got {type(data).__name__}")
    domains_block = _build_domains_block(data)
    modalities = _build_modalities(data)
    return f"""# Annex XII — Public Training Summary (Article 53(1)(d))

_Generated by soup-cli {_md_escape(data.soup_version)} at {_md_escape(data.created_at)}._

## Scope

This document is the publicly disclosed training summary required by
**Article 53(1)(d)** of the EU AI Act for general-purpose AI providers.
It enumerates the categories of training data, modalities, and a top-10
share of the data sources.

## Model

- **Model name:** {_md_escape(data.model_name)}
- **Base model:** {_md_escape(data.base_model)}
- **Task:** {_md_escape(data.task)}
- **Modalities:** {modalities}

## Data sources (top 10 by share)

{domains_block}

## Compute footprint

- **Training compute (FLOPs):** {_format_flops(data.train_compute_flops)}
- **Energy consumed:** {data.train_energy_kwh:.3f} kWh
- **Estimated CO₂ emissions:** {data.train_co2_kg:.3f} kg
"""


_VALID_FORMATS = ("markdown", "md", "pdf")


def _render_markdown(data: AnnexXIData, section_lc: str) -> str:
    return (
        render_annex_xi_markdown(data) if section_lc == "xi"
        else render_annex_xii_markdown(data)
    )


def render_annex_pdf(data: AnnexXIData, section: str) -> bytes:
    """Render an Annex XI or XII document as PDF bytes.

    Lazy-imports ``reportlab`` (install via ``pip install soup-cli[pdf]``).
    The PDF body is a block-level projection of the markdown rendering: ``#`` /
    ``##`` / ``###`` become heading styles, ``- `` becomes a bullet, blank
    lines become spacers (inline emphasis is not interpreted). Operator-
    controlled strings stay XML-escaped so a crafted ``model_name`` cannot
    inject reportlab markup.
    """
    if not isinstance(data, AnnexXIData):
        raise TypeError(f"data must be AnnexXIData, got {type(data).__name__}")
    if not isinstance(section, str) or section.lower() not in _VALID_SECTIONS:
        raise ValueError(
            f"section must be one of {_VALID_SECTIONS}, got {section!r}"
        )
    try:
        import io  # noqa: PLC0415
        from xml.sax.saxutils import escape as _xml_escape  # noqa: PLC0415

        from reportlab.lib.pagesizes import A4  # noqa: PLC0415
        from reportlab.lib.styles import getSampleStyleSheet  # noqa: PLC0415
        from reportlab.platypus import (  # noqa: PLC0415
            ListFlowable,
            ListItem,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without reportlab
        raise RuntimeError(
            "PDF rendering requires reportlab. Install with "
            "`pip install soup-cli[pdf]` (or `pip install reportlab`)."
        ) from exc

    markdown = _render_markdown(data, section.lower())
    styles = getSampleStyleSheet()
    flowables: list[object] = []
    bullets: list[object] = []

    def _flush_bullets() -> None:
        if bullets:
            flowables.append(ListFlowable(list(bullets), bulletType="bullet"))
            bullets.clear()

    def _clean(raw: str) -> str:
        # Undo the markdown backslash-escapes from _md_escape, then XML-escape
        # for reportlab's mini-markup parser.
        return _xml_escape(re.sub(r"\\([|\[\]()!<>])", r"\1", raw))

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            _flush_bullets()
            flowables.append(Spacer(1, 6))
            continue
        if stripped.startswith("### "):
            _flush_bullets()
            flowables.append(Paragraph(_clean(stripped[4:]), styles["Heading3"]))
        elif stripped.startswith("## "):
            _flush_bullets()
            flowables.append(Paragraph(_clean(stripped[3:]), styles["Heading2"]))
        elif stripped.startswith("# "):
            _flush_bullets()
            flowables.append(Paragraph(_clean(stripped[2:]), styles["Title"]))
        elif stripped.startswith("- "):
            bullets.append(
                ListItem(Paragraph(_clean(stripped[2:]), styles["BodyText"]))
            )
        else:
            _flush_bullets()
            flowables.append(Paragraph(_clean(stripped), styles["BodyText"]))
    _flush_bullets()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=f"Annex {section.upper()}")
    doc.build(flowables)
    return buffer.getvalue()


def write_annex_doc(
    data: AnnexXIData,
    section: str,
    output_path: str,
    *,
    fmt: str = "markdown",
) -> str:
    """Atomic write of an Annex XI or XII doc to ``output_path``.

    ``fmt`` is ``markdown`` (default) / ``md`` for the markdown body, or
    ``pdf`` (v0.71.3 #181) for a reportlab-rendered PDF.
    """
    if not isinstance(section, str) or section.lower() not in _VALID_SECTIONS:
        raise ValueError(
            f"section must be one of {_VALID_SECTIONS}, got {section!r}"
        )
    if not isinstance(fmt, str) or fmt.lower() not in _VALID_FORMATS:
        raise ValueError(f"fmt must be one of {_VALID_FORMATS}, got {fmt!r}")
    section_lc = section.lower()
    if fmt.lower() == "pdf":
        from soup_cli.utils.paths import atomic_write_bytes

        pdf_bytes = render_annex_pdf(data, section)
        return atomic_write_bytes(
            pdf_bytes, output_path, prefix=".annex.", suffix=".pdf.tmp",
        )
    text = _render_markdown(data, section_lc)
    return atomic_write_text(
        text, output_path, prefix=".annex.", suffix=".md.tmp",
    )


# ---------------------------------------------------------------------------
# #184 — auto-populate top_domains from the training corpus
# ---------------------------------------------------------------------------
def _row_text(row: object) -> str:
    """Best-effort concatenation of a row's text-bearing fields."""
    if isinstance(row, str):
        return row
    if not isinstance(row, Mapping):
        return ""
    parts: list[str] = []
    for field in _TEXT_FIELDS:
        val = row.get(field)
        if isinstance(val, str):
            parts.append(val)
    messages = row.get("messages")
    if isinstance(messages, (list, tuple)):
        for msg in messages:
            if isinstance(msg, Mapping):
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
    return " ".join(parts)


def _domain_of(url: str) -> str:
    """Extract a bare lowercased hostname from a URL (port stripped)."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host


def extract_top_domains(
    rows: Iterable[object], *, top_n: int = 10
) -> Tuple[Tuple[str, float], ...]:
    """Count URL domains across ``rows`` and return the top-N by share.

    Returns a tuple of ``(domain, share)`` where ``share`` is the domain's
    fraction of all extracted URLs. Deterministic: ties broken by domain
    ascending. DoS-capped (``_MAX_DOMAIN_ROWS`` rows × ``_MAX_ROW_CHARS``
    chars × ``_MAX_URLS_PER_ROW`` URLs/row).
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise ValueError("top_n must be an int")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise TypeError("rows must be an iterable of mappings/strings")
    counter: Counter[str] = Counter()
    total = 0
    for i, row in enumerate(rows):
        if i >= _MAX_DOMAIN_ROWS:
            break
        text = _row_text(row)
        if not text:
            continue
        if len(text) > _MAX_ROW_CHARS:
            text = text[:_MAX_ROW_CHARS]
        for j, match in enumerate(_URL_RE.finditer(text)):
            if j >= _MAX_URLS_PER_ROW:
                break
            domain = _domain_of(match.group(0))
            if domain:
                counter[domain] += 1
                total += 1
    if total == 0:
        return ()
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return tuple((domain, count / total) for domain, count in ranked)


def load_top_domains_from_jsonl(
    path: object, *, top_n: int = 10
) -> Tuple[Tuple[str, float], ...]:
    """Best-effort domain extraction from a cwd-contained JSONL file.

    Returns ``()`` on any failure (missing file, outside cwd, symlink,
    unreadable, oversize) — an Annex doc must never fail because the corpus
    is unavailable. cwd-contained + symlink-rejected via the shared helper.
    """
    if not isinstance(path, str) or not path:
        return ()
    try:
        realpath = enforce_under_cwd_and_no_symlink(path, "data.train")
    except (ValueError, OSError) as exc:
        _LOG.debug("load_top_domains_from_jsonl: rejected %r: %s", path, exc)
        return ()
    try:
        if not os.path.isfile(realpath):
            return ()
        if os.path.getsize(realpath) > _MAX_JSONL_BYTES:
            return ()
    except OSError:
        return ()

    def _row_iter():
        try:
            with open(realpath, encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except (ValueError, TypeError):
                        continue
        except OSError as exc:
            _LOG.debug("load_top_domains_from_jsonl: read failed: %s", exc)

    try:
        return extract_top_domains(_row_iter(), top_n=top_n)
    except (ValueError, TypeError) as exc:
        _LOG.debug("load_top_domains_from_jsonl: extract failed: %s", exc)
        return ()
