"""``soup serve --kv-cache-type``: the help text and the generated reference say what the code does.

``utils/kv_cache.py`` and ``commands/serve.py`` accept ``q8_0`` / ``bf16`` / ``f16`` on
``--backend transformers`` and refuse everything else: ``fp8`` on every backend (with or without a
Hopper GPU), and any value on another backend. The option's help used to say that ``fp8`` is a
"vLLM+Hopper only" value, which sent a reader toward a combination the code refuses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_REPO = Path(__file__).resolve().parent.parent


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed output with the help table's borders removed."""
    return " ".join(strip_ansi(text).replace("│", " ").split())


def _serve_help(monkeypatch) -> str:
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output)


def _kv_cache_help(monkeypatch) -> str:
    """The help text of ``--kv-cache-type`` alone: from its flag to the next option's flag."""
    out = _serve_help(monkeypatch)
    # typer prints the metavar as TEXT up to 0.2x and as <str> from 0.27; the metavar also keeps
    # this from matching the "--kv-cache-type bf16/f16" mention in the --device help.
    match = re.search(r"--kv-cache-type (?:TEXT|<str>) (.*?) --tool-auth-token", out)
    assert match, f"--kv-cache-type is not followed by --tool-auth-token in: {out[:300]}"
    return match.group(1)


class TestServeHelpDescribesKvCacheType:
    def test_names_the_values_that_work(self, monkeypatch):
        text = _kv_cache_help(monkeypatch)

        assert "on --backend transformers only" in text
        assert "q8_0 (8-bit quantized cache, needs hqq)" in text
        assert "bf16 or f16 (cache dtype)" in text

    def test_says_fp8_is_refused(self, monkeypatch):
        text = _kv_cache_help(monkeypatch)

        assert "fp8 is refused on every backend, Hopper GPUs included" in text
        assert "every other backend refuses this option" in text

    def test_does_not_send_a_reader_to_vllm_hopper(self, monkeypatch):
        text = _kv_cache_help(monkeypatch)

        assert "vLLM+Hopper" not in text
        assert "blocked tail" not in text


class TestGeneratedReferenceMatchesTheHelp:
    def test_cli_reference_row_says_fp8_is_refused(self):
        page = (_REPO / "docs" / "cli-reference.md").read_text(encoding="utf-8")
        rows = [line for line in page.splitlines() if line.startswith("| `--kv-cache-type`")]

        assert len(rows) == 1, rows
        assert "fp8 is refused on every backend, Hopper GPUs included" in rows[0]
        assert "vLLM+Hopper" not in rows[0]


class TestTheHelpIsTrueOfTheCode:
    """The two claims the help makes, checked against the functions that enforce them."""

    @pytest.mark.parametrize("capability", [None, (8, 6), (9, 0)])
    def test_fp8_is_refused_on_transformers_with_or_without_hopper(self, capability):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(RuntimeError, match="fp8"):
            apply_kv_cache_type("fp8", backend="transformers", compute_capability=capability)

    @pytest.mark.parametrize("kv_type", ["q8_0", "bf16", "f16"])
    def test_the_other_values_resolve_on_transformers(self, kv_type):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        runtime = apply_kv_cache_type(kv_type, backend="transformers")

        assert runtime.kv_cache_type == kv_type

    @pytest.mark.parametrize("kv_type", ["q8_0", "bf16", "f16", "fp8"])
    def test_every_value_is_refused_on_sglang(self, kv_type):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(NotImplementedError, match="Use --backend transformers"):
            apply_kv_cache_type(kv_type, backend="sglang", compute_capability=(9, 0))

    def test_q8_0_is_refused_on_vllm(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(NotImplementedError, match="not supported by vLLM"):
            apply_kv_cache_type("q8_0", backend="vllm", compute_capability=(9, 0))

    @pytest.mark.parametrize(
        ("kv_type", "expected_dtype"),
        [("bf16", "bfloat16"), ("f16", "float16"), ("fp8", "fp8")],
    )
    def test_supported_values_resolve_on_vllm(self, kv_type, expected_dtype, monkeypatch):
        import sys
        from types import ModuleType
        from typing import Literal

        monkeypatch.setattr(
            "soup_cli.utils.vllm.is_vllm_available",
            lambda: True,
        )

        vllm_module = ModuleType("vllm")
        config_module = ModuleType("vllm.config")
        cache_module = ModuleType("vllm.config.cache")
        cache_module.CacheDType = Literal["auto", "bfloat16", "float16", "fp8"]

        monkeypatch.setitem(sys.modules, "vllm", vllm_module)
        monkeypatch.setitem(sys.modules, "vllm.config", config_module)
        monkeypatch.setitem(sys.modules, "vllm.config.cache", cache_module)

        from soup_cli.utils.kv_cache import apply_kv_cache_type

        runtime = apply_kv_cache_type(
            kv_type,
            backend="vllm",
            compute_capability=(9, 0),
        )

        assert runtime.kv_cache_type == kv_type
        assert runtime.vllm_cache_dtype == expected_dtype

    def test_mlx_refuses_the_option(self):
        from soup_cli.utils.mlx_serve import validate_serve_options

        with pytest.raises(ValueError, match="--backend mlx does not support --kv-cache-type"):
            validate_serve_options({"--kv-cache-type": ("bf16", None)})
