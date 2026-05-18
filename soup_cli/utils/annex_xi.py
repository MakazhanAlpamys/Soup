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

import math
import re
from dataclasses import dataclass
from typing import Tuple

from soup_cli.utils.paths import atomic_write_text

_MAX_NAME = 256
_MAX_TEXT = 16384
_VALID_SECTIONS = ("xi", "xii")


def _validate_text(value: str, field_name: str, *, max_len: int = _MAX_NAME) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be str")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field_name} too long ({len(value)} > {max_len})")
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


def write_annex_doc(data: AnnexXIData, section: str, output_path: str) -> str:
    """Atomic write of an Annex XI or XII markdown to ``output_path``."""
    if not isinstance(section, str) or section.lower() not in _VALID_SECTIONS:
        raise ValueError(
            f"section must be one of {_VALID_SECTIONS}, got {section!r}"
        )
    section_lc = section.lower()
    text = (
        render_annex_xi_markdown(data) if section_lc == "xi"
        else render_annex_xii_markdown(data)
    )
    return atomic_write_text(
        text, output_path, prefix=".annex.", suffix=".md.tmp",
    )
