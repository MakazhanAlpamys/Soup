"""Soup Cans — shareable .can artifact format (v0.26.0 Part E).

A ``.can`` file is a tar.gz containing:
  manifest.yaml      — format version, name, author, etc.
  config.yaml        — full SoupConfig
  data_ref.yaml      — how to fetch training data (hash + URL/HF id)
  recipe.md          — human-readable description (optional)
"""

from __future__ import annotations

from soup_cli.cans.schema import CAN_FORMAT_VERSION, DataRef, Manifest

__all__ = ["CAN_FORMAT_VERSION", "DataRef", "Manifest"]
