"""v0.66.0 Part E — Activation probe pack (TDD).

``soup probe pack <base>`` assembles a pack of calibrated probes for a
given base model. The "pack" is a manifest of probes (sleeper / SAE /
generic) bundled with their metadata + targeted hidden_dim. Composes
with Parts C (sleeper) + D (interference) by exposing a single
``ProbePack`` surface that downstream tools consume.

This release ships the manifest + lookup + render. No live HF Hub
download (operators bring their own weights via the existing v0.66.0
Part A/C surfaces); v0.66.x can wire ``huggingface_hub.snapshot_download``
through ``utils.hubs`` later.

Public surface:

- ``ProbeEntry`` frozen dataclass — one probe metadata entry
- ``ProbePack`` frozen dataclass — full pack for a base
- ``PROBE_KINDS`` closed allowlist
- ``BUNDLED_PACKS`` ``MappingProxyType`` — per-base packs
- ``validate_pack_base(name)`` — allowlist canonicaliser
- ``get_probe_pack(base)`` — typed lookup
- ``list_probe_bases()`` — sorted list of supported bases
- ``render_pack_json`` / ``render_pack_markdown``
"""
from __future__ import annotations

import json

import pytest


def test_module_imports():
    from soup_cli.utils import probe_pack

    for name in (
        "ProbeEntry",
        "ProbePack",
        "PROBE_KINDS",
        "BUNDLED_PACKS",
        "validate_pack_base",
        "get_probe_pack",
        "list_probe_bases",
        "render_pack_json",
        "render_pack_markdown",
    ):
        assert hasattr(probe_pack, name), name


def test_probe_kinds_frozen():
    from soup_cli.utils.probe_pack import PROBE_KINDS

    assert isinstance(PROBE_KINDS, frozenset)
    # Should cover sleeper + sae + truth + harm
    for kind in ("sleeper", "sae"):
        assert kind in PROBE_KINDS


def test_bundled_packs_immutable():
    from types import MappingProxyType

    from soup_cli.utils.probe_pack import BUNDLED_PACKS

    assert isinstance(BUNDLED_PACKS, MappingProxyType)
    with pytest.raises(TypeError):
        BUNDLED_PACKS["x"] = "y"  # type: ignore[index]


def test_bundled_packs_overlap_with_sleeper_bases():
    """Every base in BUNDLED_PACKS must have a sleeper probe entry."""
    from soup_cli.utils.probe_pack import BUNDLED_PACKS
    from soup_cli.utils.sleeper_probe import BUNDLED_PROBES

    for base, pack in BUNDLED_PACKS.items():
        assert base in BUNDLED_PROBES, f"{base} missing from sleeper catalogue"
        # The pack must include at least the sleeper probe
        kinds = {e.kind for e in pack.probes}
        assert "sleeper" in kinds


def test_list_probe_bases_sorted():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, list_probe_bases

    result = list_probe_bases()
    assert isinstance(result, tuple)
    assert list(result) == sorted(result)
    assert set(result) == set(BUNDLED_PACKS)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_pack_base_happy():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, validate_pack_base

    base = next(iter(BUNDLED_PACKS))
    assert validate_pack_base(base) == base


def test_validate_pack_base_case_insensitive():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, validate_pack_base

    base = next(iter(BUNDLED_PACKS))
    assert validate_pack_base(base.upper()) == base


def test_validate_pack_base_unknown_raises():
    from soup_cli.utils.probe_pack import validate_pack_base

    with pytest.raises(ValueError, match="no probe pack"):
        validate_pack_base("unknown/model")


def test_validate_pack_base_bool_rejected():
    from soup_cli.utils.probe_pack import validate_pack_base

    with pytest.raises(TypeError):
        validate_pack_base(True)


def test_validate_pack_base_non_string_rejected():
    from soup_cli.utils.probe_pack import validate_pack_base

    with pytest.raises(TypeError):
        validate_pack_base(42)


def test_validate_pack_base_empty_rejected():
    from soup_cli.utils.probe_pack import validate_pack_base

    with pytest.raises(ValueError):
        validate_pack_base("")


def test_validate_pack_base_null_byte_rejected():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, validate_pack_base

    base = next(iter(BUNDLED_PACKS))
    with pytest.raises(ValueError, match="null"):
        validate_pack_base(base + "\x00")


def test_validate_pack_base_oversize_rejected():
    from soup_cli.utils.probe_pack import validate_pack_base

    with pytest.raises(ValueError):
        validate_pack_base("a" * 1000)


# ---------------------------------------------------------------------------
# ProbeEntry
# ---------------------------------------------------------------------------


def test_probe_entry_frozen():
    from soup_cli.utils.probe_pack import ProbeEntry

    e = ProbeEntry(name="x", kind="sleeper", hidden_dim=4096, description="d")
    with pytest.raises((AttributeError, Exception)):
        e.kind = "sae"  # type: ignore[misc]


def test_probe_entry_rejects_unknown_kind():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError, match="kind"):
        ProbeEntry(name="x", kind="weird", hidden_dim=4096, description="d")


def test_probe_entry_rejects_empty_name():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError):
        ProbeEntry(name="", kind="sleeper", hidden_dim=4096, description="d")


def test_probe_entry_rejects_negative_hidden_dim():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError):
        ProbeEntry(name="x", kind="sleeper", hidden_dim=-1, description="d")


def test_probe_entry_rejects_bool_hidden_dim():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(TypeError):
        ProbeEntry(name="x", kind="sleeper", hidden_dim=True, description="d")


def test_probe_entry_rejects_null_byte_name():
    from soup_cli.utils.probe_pack import ProbeEntry

    with pytest.raises(ValueError):
        ProbeEntry(name="a\x00b", kind="sleeper", hidden_dim=4096, description="d")


# ---------------------------------------------------------------------------
# ProbePack
# ---------------------------------------------------------------------------


def test_probe_pack_frozen():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    p = ProbePack(
        base="b",
        probes=(ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d"),),
        soup_version="0.66.0",
    )
    with pytest.raises((AttributeError, Exception)):
        p.base = "y"  # type: ignore[misc]


def test_probe_pack_rejects_non_tuple_probes():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    with pytest.raises(TypeError):
        ProbePack(
            base="b",
            probes=[
                ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d")
            ],  # list, not tuple
            soup_version="0.66.0",
        )


def test_probe_pack_rejects_empty_probes():
    from soup_cli.utils.probe_pack import ProbePack

    with pytest.raises(ValueError, match="at least"):
        ProbePack(base="b", probes=tuple(), soup_version="0.66.0")


def test_probe_pack_rejects_duplicate_probe_names():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    e1 = ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d")
    e2 = ProbeEntry(name="x", kind="sae", hidden_dim=4, description="d")
    with pytest.raises(ValueError, match="duplicate"):
        ProbePack(base="b", probes=(e1, e2), soup_version="0.66.0")


def test_probe_pack_rejects_empty_base():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    with pytest.raises(ValueError):
        ProbePack(
            base="",
            probes=(
                ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d"),
            ),
            soup_version="0.66.0",
        )


def test_probe_pack_rejects_null_byte_base():
    from soup_cli.utils.probe_pack import ProbeEntry, ProbePack

    with pytest.raises(ValueError):
        ProbePack(
            base="a\x00b",
            probes=(
                ProbeEntry(name="x", kind="sleeper", hidden_dim=4, description="d"),
            ),
            soup_version="0.66.0",
        )


# ---------------------------------------------------------------------------
# get_probe_pack
# ---------------------------------------------------------------------------


def test_get_probe_pack_happy():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, ProbePack, get_probe_pack

    base = next(iter(BUNDLED_PACKS))
    pack = get_probe_pack(base)
    assert isinstance(pack, ProbePack)
    assert pack.base == base


def test_get_probe_pack_unknown_raises():
    from soup_cli.utils.probe_pack import get_probe_pack

    with pytest.raises(ValueError, match="no probe pack"):
        get_probe_pack("unknown/model")


def test_get_probe_pack_case_insensitive():
    from soup_cli.utils.probe_pack import BUNDLED_PACKS, get_probe_pack

    base = next(iter(BUNDLED_PACKS))
    upper = base.upper()
    pack = get_probe_pack(upper)
    assert pack.base == base  # canonicalised


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_pack_json_roundtrip():
    from soup_cli.utils.probe_pack import (
        ProbeEntry,
        ProbePack,
        render_pack_json,
    )

    pack = ProbePack(
        base="b",
        probes=(
            ProbeEntry(name="sleeper-1", kind="sleeper", hidden_dim=4, description="d1"),
            ProbeEntry(name="sae-1", kind="sae", hidden_dim=4, description="d2"),
        ),
        soup_version="0.66.0",
    )
    text = render_pack_json(pack)
    payload = json.loads(text)
    assert payload["base"] == "b"
    assert len(payload["probes"]) == 2


def test_render_pack_json_rejects_non_pack():
    from soup_cli.utils.probe_pack import render_pack_json

    with pytest.raises(TypeError):
        render_pack_json("nope")


def test_render_pack_markdown_renders_probes():
    from soup_cli.utils.probe_pack import (
        ProbeEntry,
        ProbePack,
        render_pack_markdown,
    )

    pack = ProbePack(
        base="meta-llama/Llama-3-8B",
        probes=(
            ProbeEntry(name="sleeper-1", kind="sleeper", hidden_dim=4096,
                       description="Defection probe"),
        ),
        soup_version="0.66.0",
    )
    text = render_pack_markdown(pack)
    assert "Llama" in text
    assert "sleeper-1" in text


def test_render_pack_markdown_rejects_non_pack():
    from soup_cli.utils.probe_pack import render_pack_markdown

    with pytest.raises(TypeError):
        render_pack_markdown(None)


# ---------------------------------------------------------------------------
# Source-grep
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    import inspect

    from soup_cli.utils import probe_pack

    source = inspect.getsource(probe_pack)
    top_level_imports = [
        line for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in top_level_imports:
        for bad in ("torch", "transformers", "peft", "safetensors"):
            assert bad not in line, f"top-level {bad} import: {line!r}"
