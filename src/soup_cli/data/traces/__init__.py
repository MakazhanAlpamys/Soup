"""Trace-to-Preference harvester (v0.26.0 Part C).

Ingests production logs from LangChain / OpenAI-style / Soup-serve, extracts
preference pairs from explicit signals, and emits JSONL for DPO/KTO/ORPO.
"""

from __future__ import annotations

from soup_cli.data.traces.pair_builder import (
    SUPPORTED_SIGNALS,
    PreferencePair,
    build_pairs,
)
from soup_cli.data.traces.parsers import (
    SUPPORTED_FORMATS,
    Trace,
    parse_langchain,
    parse_openai,
    parse_soup_serve,
)

__all__ = [
    "SUPPORTED_FORMATS",
    "SUPPORTED_SIGNALS",
    "PreferencePair",
    "Trace",
    "build_pairs",
    "parse_langchain",
    "parse_openai",
    "parse_soup_serve",
]
