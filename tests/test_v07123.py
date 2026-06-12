"""v0.71.23 "Targeted training: native Spectrum" tests.

Closes #266 — native Spectrum (arXiv:2406.06623) targeted training:

- ``utils/spectrum_scan.py`` — pure-numpy singular-value SNR kernel +
  Marchenko-Pastur noise threshold + per-tensor safetensors streaming
  (NO model load) + ``~/.soup/spectrum/<slug>.json`` scan cache.
- ``soup spectrum scan`` — SNR table + ready-to-paste
  ``training.unfrozen_parameters`` YAML block.
- Schema ``training.unfrozen_parameters`` + cross-validators (mutually
  exclusive with LoRA features / freeze_layers / freeze_ratio /
  train_router_only) + ``apply_unfrozen_parameters`` SFT wiring
  (freeze-all → unfreeze-matching, full fine-tuning, LoRA off).

The SNR kernel is pure numpy and transpose-invariant (singular values are
identical for ``W`` and ``W.T``); the "Conv1D-aware" requirement is about
recognising GPT-2's ``c_attn``/``c_fc``/``c_proj`` module naming, not the
math. LISA (per-step layer sampling) is deliberately split to #267.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

runner = CliRunner()


def _module_head(rel_path: str) -> str:
    """Return a module's source text up to the first ``def``/``class``."""
    src = (_SRC / rel_path).read_text(encoding="utf-8").replace("\r\n", "\n")
    for marker in ("\ndef ", "\nclass "):
        src = src.split(marker, 1)[0]
    return src


def _assert_no_top_level_import(rel_path: str, mod: str) -> None:
    head = _module_head(rel_path)
    assert f"\nimport {mod}" not in head, f"top-level `import {mod}` in {rel_path}"
    assert f"\nfrom {mod} " not in head, f"top-level `from {mod} ` in {rel_path}"


def _write_tiny_safetensors(dir_path: Path) -> Path:
    """Synthesize a 2-layer Llama-shaped model dir (numpy safetensors).

    No torch / no download — exercises the streaming scanner end-to-end.
    """
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    tensors: dict[str, np.ndarray] = {}
    for layer in range(2):
        # attention projections (square)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{layer}.self_attn.{proj}.weight"] = (
                rng.standard_normal((32, 32)).astype(np.float32)
            )
        # mlp projections (rectangular)
        tensors[f"model.layers.{layer}.mlp.gate_proj.weight"] = (
            rng.standard_normal((64, 32)).astype(np.float32)
        )
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = (
            rng.standard_normal((32, 64)).astype(np.float32)
        )
        # 1-D norm — must be skipped by the scanner
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = (
            rng.standard_normal((32,)).astype(np.float32)
        )
    # embedding (2-D "other") + lm_head
    tensors["model.embed_tokens.weight"] = rng.standard_normal((100, 32)).astype(
        np.float32
    )
    dir_path.mkdir(parents=True, exist_ok=True)
    out = dir_path / "model.safetensors"
    save_file(tensors, str(out))
    return out


# ---------------------------------------------------------------------------
# Part A — SNR kernel (pure numpy)
# ---------------------------------------------------------------------------
class TestEstimateSigma:
    def test_iqr_over_1349(self):
        from soup_cli.utils.spectrum_scan import estimate_sigma

        s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q75, q25 = np.percentile(s, [75, 25])
        assert estimate_sigma(s) == pytest.approx((q75 - q25) / 1.349)

    def test_constant_singular_values_zero_sigma(self):
        from soup_cli.utils.spectrum_scan import estimate_sigma

        assert estimate_sigma(np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)


class TestMarchenkoPasturThreshold:
    def test_symmetric_in_n_m(self):
        from soup_cli.utils.spectrum_scan import marchenko_pastur_threshold

        # beta = min/max, so (n, m) and (m, n) give the same threshold.
        assert marchenko_pastur_threshold(1.0, 64, 32) == pytest.approx(
            marchenko_pastur_threshold(1.0, 32, 64)
        )

    def test_square_matrix_beta_one(self):
        from soup_cli.utils.spectrum_scan import marchenko_pastur_threshold

        # beta=1 → threshold = sigma * (1 + sqrt(1)) = 2*sigma.
        assert marchenko_pastur_threshold(3.0, 16, 16) == pytest.approx(6.0)

    def test_matches_reference_form(self):
        from soup_cli.utils.spectrum_scan import marchenko_pastur_threshold

        sigma, n, m = 1.7, 48, 96
        beta = min(n, m) / max(n, m)
        expected = sigma * math.sqrt((1 + math.sqrt(beta)) ** 2)
        assert marchenko_pastur_threshold(sigma, n, m) == pytest.approx(expected)

    def test_non_positive_dims_return_zero(self):
        from soup_cli.utils.spectrum_scan import marchenko_pastur_threshold

        assert marchenko_pastur_threshold(1.0, 0, 8) == 0.0
        assert marchenko_pastur_threshold(1.0, 8, -1) == 0.0


class TestComputeSnr:
    def test_signal_beats_pure_noise(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        rng = np.random.default_rng(0)
        noise = rng.standard_normal((64, 64))
        u1, v1 = rng.standard_normal(64), rng.standard_normal(64)
        u2, v2 = rng.standard_normal(64), rng.standard_normal(64)
        signal = 10 * np.outer(u1, v1) + 8 * np.outer(u2, v2) + 0.01 * noise
        assert compute_snr(signal) > compute_snr(noise)

    def test_transpose_invariant(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        rng = np.random.default_rng(1)
        w = rng.standard_normal((48, 96))
        assert compute_snr(w) == pytest.approx(compute_snr(w.T), rel=1e-9)

    def test_deterministic(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        rng = np.random.default_rng(2)
        w = rng.standard_normal((40, 40))
        assert compute_snr(w) == compute_snr(w.copy())

    def test_zero_matrix_returns_zero(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        assert compute_snr(np.zeros((8, 8))) == 0.0

    def test_always_finite(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        rng = np.random.default_rng(3)
        for shape in [(8, 8), (1, 1), (3, 9), (9, 3)]:
            val = compute_snr(rng.standard_normal(shape))
            assert math.isfinite(val), shape

    def test_non_2d_raises(self):
        from soup_cli.utils.spectrum_scan import compute_snr

        with pytest.raises(ValueError, match="2"):
            compute_snr(np.zeros((4, 4, 4)))


class TestClassifyModule:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("model.layers.0.self_attn.q_proj.weight", "attn"),
            ("model.layers.5.self_attn.o_proj.weight", "attn"),
            ("transformer.h.3.attn.c_attn.weight", "attn"),
            ("transformer.h.3.attn.c_proj.weight", "attn"),
            ("model.layers.0.mlp.gate_proj.weight", "mlp"),
            ("model.layers.0.mlp.down_proj.weight", "mlp"),
            ("transformer.h.3.mlp.c_fc.weight", "mlp"),
            ("transformer.h.3.mlp.c_proj.weight", "mlp"),
            ("model.embed_tokens.weight", "other"),
            ("lm_head.weight", "other"),
        ],
    )
    def test_classification(self, name, expected):
        from soup_cli.utils.spectrum_scan import classify_module

        assert classify_module(name) == expected

    def test_non_weight_returns_none(self):
        from soup_cli.utils.spectrum_scan import classify_module

        # a layernorm weight is a real .weight but neither mlp nor attn
        assert classify_module("model.layers.0.input_layernorm.weight") == "other"
        # a bias is not a .weight → not scannable
        assert classify_module("model.layers.0.self_attn.q_proj.bias") is None


class TestLayerTypeSignature:
    @pytest.mark.parametrize(
        "name,sig",
        [
            ("model.layers.5.self_attn.q_proj.weight", "self_attn.q_proj"),
            ("model.layers.0.mlp.down_proj.weight", "mlp.down_proj"),
            ("transformer.h.3.mlp.c_fc.weight", "mlp.c_fc"),
            ("lm_head.weight", "lm_head"),
        ],
    )
    def test_strips_layer_index(self, name, sig):
        from soup_cli.utils.spectrum_scan import layer_type_signature

        assert layer_type_signature(name) == sig


# ---------------------------------------------------------------------------
# Part A — streaming + scan + selection
# ---------------------------------------------------------------------------
class TestIterWeightMatrices:
    def test_yields_only_2d_weights(self, tmp_path):
        from soup_cli.utils.spectrum_scan import iter_weight_matrices

        _write_tiny_safetensors(tmp_path)
        names = {name for name, _ in iter_weight_matrices(str(tmp_path))}
        # 2 layers × (4 attn + 2 mlp) = 12, + embed_tokens = 13 two-D weights
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.embed_tokens.weight" in names
        # 1-D norm skipped
        assert "model.layers.0.input_layernorm.weight" not in names
        assert len(names) == 13

    def test_modules_filter_excludes_other(self, tmp_path):
        from soup_cli.utils.spectrum_scan import iter_weight_matrices

        _write_tiny_safetensors(tmp_path)
        names = {
            name
            for name, _ in iter_weight_matrices(str(tmp_path), modules=("mlp", "attn"))
        }
        assert "model.embed_tokens.weight" not in names
        assert "model.layers.0.mlp.gate_proj.weight" in names

    def test_arrays_are_2d_float(self, tmp_path):
        from soup_cli.utils.spectrum_scan import iter_weight_matrices

        _write_tiny_safetensors(tmp_path)
        for _name, arr in iter_weight_matrices(str(tmp_path)):
            assert arr.ndim == 2


class TestScanWeightsDir:
    def test_returns_finite_snr_per_matrix(self, tmp_path):
        from soup_cli.utils.spectrum_scan import scan_weights_dir

        _write_tiny_safetensors(tmp_path)
        layers = scan_weights_dir(str(tmp_path))
        assert len(layers) == 13
        for ls in layers:
            assert math.isfinite(ls.snr)
            assert ls.module_type in ("attn", "mlp", "other")

    def test_groups_carry_signature(self, tmp_path):
        from soup_cli.utils.spectrum_scan import scan_weights_dir

        _write_tiny_safetensors(tmp_path)
        layers = scan_weights_dir(str(tmp_path))
        groups = {ls.group for ls in layers}
        assert "self_attn.q_proj" in groups
        assert "mlp.down_proj" in groups


class TestSelectUnfrozenParameters:
    def _fake_layers(self):
        from soup_cli.utils.spectrum_scan import LayerSNR

        out = []
        # 4 q_proj layers with distinct SNR
        for i, snr in enumerate([0.1, 0.4, 0.2, 0.9]):
            out.append(
                LayerSNR(
                    name=f"model.layers.{i}.self_attn.q_proj.weight",
                    module_type="attn",
                    group="self_attn.q_proj",
                    snr=snr,
                    shape=(32, 32),
                )
            )
        return out

    def test_top_50_percent_per_group(self):
        from soup_cli.utils.spectrum_scan import select_unfrozen_parameters

        kept = select_unfrozen_parameters(self._fake_layers(), top_percent=50)
        # top 2 of 4 by SNR: layers 3 (0.9) and 1 (0.4)
        assert kept == [
            "model.layers.1.self_attn.q_proj",
            "model.layers.3.self_attn.q_proj",
        ]

    def test_at_least_one_kept_per_group(self):
        from soup_cli.utils.spectrum_scan import select_unfrozen_parameters

        kept = select_unfrozen_parameters(self._fake_layers(), top_percent=1)
        # ceil(0.01*4) = 1 → the single best (layer 3)
        assert kept == ["model.layers.3.self_attn.q_proj"]

    def test_top_percent_bounds(self):
        from soup_cli.utils.spectrum_scan import select_unfrozen_parameters

        for bad in (0, -5, 101, 1000):
            with pytest.raises(ValueError, match="top_percent"):
                select_unfrozen_parameters(self._fake_layers(), top_percent=bad)

    def test_modules_filter(self):
        from soup_cli.utils.spectrum_scan import LayerSNR, select_unfrozen_parameters

        layers = self._fake_layers() + [
            LayerSNR(
                name="model.embed_tokens.weight",
                module_type="other",
                group="model.embed_tokens",
                snr=5.0,
                shape=(100, 32),
            )
        ]
        kept = select_unfrozen_parameters(
            layers, top_percent=50, modules=("mlp", "attn")
        )
        assert all("embed_tokens" not in p for p in kept)


# ---------------------------------------------------------------------------
# Part A — cache + slug
# ---------------------------------------------------------------------------
class TestModelSlug:
    def test_sanitizes_separators(self):
        from soup_cli.utils.spectrum_scan import model_slug

        slug = model_slug("meta-llama/Llama-3-8B")
        assert "/" not in slug and "\\" not in slug
        assert slug

    def test_rejects_traversal(self):
        from soup_cli.utils.spectrum_scan import model_slug

        # ".." path components must not survive into a filename
        slug = model_slug("../../etc/passwd")
        assert ".." not in slug
        assert "/" not in slug


class TestScanCache:
    def test_roundtrip(self, tmp_path):
        from soup_cli.utils.spectrum_scan import (
            ScanResult,
            read_cached_scan,
            scan_weights_dir,
            write_cached_scan,
        )

        model_dir = tmp_path / "m"
        _write_tiny_safetensors(model_dir)
        layers = scan_weights_dir(str(model_dir))
        result = ScanResult(model="org/m", modules="all", layers=tuple(layers))
        cache = tmp_path / "cache"
        path = write_cached_scan(result, cache_dir=str(cache))
        assert Path(path).is_file()
        loaded = read_cached_scan("org/m", modules="all", cache_dir=str(cache))
        assert loaded is not None
        assert len(loaded.layers) == len(layers)
        assert {ls.name for ls in loaded.layers} == {ls.name for ls in layers}

    def test_missing_returns_none(self, tmp_path):
        from soup_cli.utils.spectrum_scan import read_cached_scan

        assert read_cached_scan("absent/model", cache_dir=str(tmp_path)) is None

    def test_modules_mismatch_returns_none(self, tmp_path):
        from soup_cli.utils.spectrum_scan import (
            ScanResult,
            read_cached_scan,
            write_cached_scan,
        )

        result = ScanResult(model="org/m", modules="all", layers=())
        write_cached_scan(result, cache_dir=str(tmp_path))
        # asked for a different module set → cache miss (do not reuse)
        assert read_cached_scan("org/m", modules="mlp,attn", cache_dir=str(tmp_path)) is None


class TestCacheDirOverride:
    def test_default_under_dot_soup(self):
        from soup_cli.utils.spectrum_scan import default_spectrum_cache_dir

        d = default_spectrum_cache_dir()
        assert d.endswith(str(Path(".soup") / "spectrum"))

    def test_env_override_under_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils import spectrum_scan

        target = Path.cwd() / "spectrum-cache-test"
        monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", str(target))
        try:
            resolved = spectrum_scan.resolve_cache_dir()
            assert Path(resolved) == Path(os.path.realpath(str(target)))
        finally:
            if target.exists():
                target.rmdir()

    def test_env_override_control_char_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils import spectrum_scan

        monkeypatch.setenv("SOUP_SPECTRUM_CACHE_DIR", "bad\nvalue")
        # falls through to the default rather than honouring a CRLF-laden path
        resolved = spectrum_scan.resolve_cache_dir()
        assert "bad" not in resolved


class TestNoTopLevelHeavyImport:
    @pytest.mark.parametrize("mod", ["torch", "transformers", "safetensors"])
    def test_kernel_is_numpy_only(self, mod):
        _assert_no_top_level_import("utils/spectrum_scan.py", mod)


# ---------------------------------------------------------------------------
# Part B — `soup spectrum scan` CLI
# ---------------------------------------------------------------------------
class TestSpectrumScanCli:
    def test_help_lists_scan(self):
        from soup_cli.commands import spectrum as spectrum_cmd

        res = runner.invoke(spectrum_cmd.app, ["--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "scan" in res.output

    def test_registered_in_main_app(self):
        from soup_cli.cli import app as main_app

        res = runner.invoke(main_app, ["spectrum", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "scan" in res.output

    def test_scan_prints_table_and_yaml(self, tmp_path):
        from soup_cli.commands import spectrum as spectrum_cmd

        model_dir = tmp_path / "m"
        _write_tiny_safetensors(model_dir)
        res = runner.invoke(
            spectrum_cmd.app,
            ["scan", "--model", str(model_dir), "--no-cache"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "unfrozen_parameters" in res.output
        assert "self_attn.q_proj" in res.output

    def test_scan_writes_output_patch(self):
        import yaml

        from soup_cli.commands import spectrum as spectrum_cmd

        with runner.isolated_filesystem():
            model_dir = Path("m")
            _write_tiny_safetensors(model_dir)
            res = runner.invoke(
                spectrum_cmd.app,
                [
                    "scan",
                    "--model",
                    str(model_dir),
                    "--top-percent",
                    "50",
                    "--output",
                    "patch.yaml",
                    "--no-cache",
                ],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert Path("patch.yaml").is_file()
            data = yaml.safe_load(Path("patch.yaml").read_text(encoding="utf-8"))
            assert "training" in data
            params = data["training"]["unfrozen_parameters"]
            assert isinstance(params, list) and params
            assert all(isinstance(p, str) for p in params)

    def test_modules_filter_excludes_embeddings(self, tmp_path):
        from soup_cli.commands import spectrum as spectrum_cmd

        model_dir = tmp_path / "m"
        _write_tiny_safetensors(model_dir)
        res = runner.invoke(
            spectrum_cmd.app,
            [
                "scan",
                "--model",
                str(model_dir),
                "--modules",
                "mlp,attn",
                "--no-cache",
            ],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "embed_tokens" not in res.output

    def test_bad_top_percent_rejected(self, tmp_path):
        from soup_cli.commands import spectrum as spectrum_cmd

        model_dir = tmp_path / "m"
        _write_tiny_safetensors(model_dir)
        for bad in ("0", "150"):
            res = runner.invoke(
                spectrum_cmd.app,
                ["scan", "--model", str(model_dir), "--top-percent", bad, "--no-cache"],
            )
            assert res.exit_code != 0
            assert "top-percent" in res.output.lower() or "top_percent" in res.output

    def test_output_outside_cwd_rejected(self, tmp_path):
        from soup_cli.commands import spectrum as spectrum_cmd

        model_dir = tmp_path / "m"
        _write_tiny_safetensors(model_dir)
        outside = tmp_path / "escape.yaml"
        res = runner.invoke(
            spectrum_cmd.app,
            [
                "scan",
                "--model",
                str(model_dir),
                "--output",
                str(outside),
                "--no-cache",
            ],
        )
        assert res.exit_code != 0
        assert "cwd" in res.output.lower()

    def test_missing_model_friendly_error(self, tmp_path):
        from soup_cli.commands import spectrum as spectrum_cmd

        res = runner.invoke(
            spectrum_cmd.app,
            ["scan", "--model", str(tmp_path / "nope"), "--no-cache"],
        )
        assert res.exit_code != 0
        # a Hub download would be attempted; offline that surfaces an error,
        # never a raw traceback to the user.
        assert res.exception is None or isinstance(res.exception, SystemExit)
        assert "failed" in res.output.lower()


# ---------------------------------------------------------------------------
# Part C — schema + trainer wiring
# ---------------------------------------------------------------------------
_BASE = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _sft_yaml(extra_training: str = "") -> str:
    return (
        f"base: {_BASE}\n"
        "task: sft\n"
        "data:\n"
        "  train: data.jsonl\n"
        "training:\n"
        "  quantization: none\n"  # Spectrum is full FT — quantization off
        "  unfrozen_parameters:\n"
        "  - model.layers.0.mlp.down_proj\n"
        "  - model.layers.1.self_attn.q_proj\n"
        f"{extra_training}"
    )


class TestUnfrozenParametersSchema:
    def test_happy_path_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_sft_yaml())
        assert cfg.training.unfrozen_parameters == [
            "model.layers.0.mlp.down_proj",
            "model.layers.1.self_attn.q_proj",
        ]

    def test_default_none(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
        )
        assert cfg.training.unfrozen_parameters is None

    def test_empty_string_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            "training:\n  unfrozen_parameters:\n  - ''\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "unfrozen_parameters" in str(exc.value)

    def test_null_byte_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            'training:\n  unfrozen_parameters:\n  - "bad\\u0000pat"\n'
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "null" in str(exc.value).lower()

    def test_invalid_regex_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            'training:\n  unfrozen_parameters:\n  - "model.layers.[0"\n'
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "regex" in str(exc.value).lower()

    def test_pattern_length_boundary(self):
        from soup_cli.config.loader import load_config_from_string

        head = f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
        # 512 chars accepted, 513 rejected
        ok = load_config_from_string(
            head + "training:\n  quantization: none\n"
            f"  unfrozen_parameters:\n  - \"{'a' * 512}\"\n"
        )
        assert ok.training.unfrozen_parameters == ["a" * 512]
        with pytest.raises(Exception) as exc:
            load_config_from_string(
                head + "training:\n  quantization: none\n"
                f"  unfrozen_parameters:\n  - \"{'a' * 513}\"\n"
            )
        assert "512" in str(exc.value)

    def test_too_many_patterns_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        patterns = "\n".join(f"  - p{i}" for i in range(50_001))
        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            f"training:\n  unfrozen_parameters:\n{patterns}\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "unfrozen_parameters" in str(exc.value)

    @pytest.mark.parametrize("redos", ["(x+)+y", "(a*)*", "(a+)*b", "(.*)*"])
    def test_redos_pattern_rejected(self, redos):
        from soup_cli.config.loader import load_config_from_string

        # a catastrophic-backtracking regex compiles fine but would hang
        # re.search during training — reject the pattern class at parse time.
        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            "training:\n  quantization: none\n  unfrozen_parameters:\n"
            f'  - "{redos}"\n'
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "redos" in str(exc.value).lower() or "quantifier" in str(exc.value).lower()

    def test_literal_prefix_with_parens_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        # a normal alternation group (no nested unbounded quantifier) is fine
        cfg = load_config_from_string(
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            "training:\n  quantization: none\n  unfrozen_parameters:\n"
            '  - "model.layers.0.self_attn.(q|k|v)_proj"\n'
        )
        assert cfg.training.unfrozen_parameters == [
            "model.layers.0.self_attn.(q|k|v)_proj"
        ]


class TestUnfrozenParametersCrossValidators:
    @pytest.mark.parametrize(
        "extra,keyword",
        [
            ("  freeze_layers: 2\n", "freeze_layers"),
            ("  freeze_ratio: 0.5\n", "freeze_ratio"),
            ("  train_router_only: true\n", "train_router_only"),
            ("  moe_lora: true\n", "lora"),
            ("  use_longlora: true\n", "lora"),
            ("  relora_steps: 100\n", "lora"),
            ("  loraplus_lr_ratio: 16.0\n", "lora"),
            ("  lora:\n    use_dora: true\n", "lora"),
            ("  lora:\n    use_vera: true\n", "lora"),
            ("  lora:\n    use_olora: true\n", "lora"),
            ("  lora:\n    use_rslora: true\n", "lora"),
            ("  expand_layers: 2\n  freeze_trainable_layers: 2\n", "expand_layers"),
        ],
    )
    def test_mutually_exclusive(self, extra, keyword):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(Exception) as exc:
            load_config_from_string(_sft_yaml(extra))
        assert keyword in str(exc.value).lower()

    def test_rejected_on_non_sft_task(self):
        from soup_cli.config.loader import load_config_from_string

        bad = (
            f"base: {_BASE}\ntask: dpo\ndata:\n  train: d.jsonl\n"
            "training:\n  unfrozen_parameters:\n  - model.layers.0.mlp.down_proj\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "sft" in str(exc.value).lower()

    def test_rejected_on_mlx_backend(self):
        from soup_cli.config.loader import load_config_from_string

        bad = (
            f"base: {_BASE}\ntask: sft\nbackend: mlx\ndata:\n  train: d.jsonl\n"
            "training:\n  unfrozen_parameters:\n  - model.layers.0.mlp.down_proj\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "transformers" in str(exc.value).lower()

    def test_rejected_when_quantized(self):
        from soup_cli.config.loader import load_config_from_string

        # Spectrum is full fine-tuning of float weights — a quantized load
        # (the 4bit default) cannot be trained directly.
        bad = (
            f"base: {_BASE}\ntask: sft\ndata:\n  train: d.jsonl\n"
            "training:\n  quantization: 4bit\n"
            "  unfrozen_parameters:\n  - model.layers.0.mlp.down_proj\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "quantization" in str(exc.value).lower()

    def test_rejected_on_vision_modality(self):
        from soup_cli.config.loader import load_config_from_string

        # the Spectrum branch is wired only in the text SFT trainer; a
        # vision/audio config would silently no-op without this gate.
        bad = (
            f"base: {_BASE}\ntask: sft\nmodality: vision\n"
            "data:\n  train: d.jsonl\n"
            "training:\n  quantization: none\n"
            "  unfrozen_parameters:\n  - model.layers.0.mlp.down_proj\n"
        )
        with pytest.raises(Exception) as exc:
            load_config_from_string(bad)
        assert "modality" in str(exc.value).lower()


class TestApplyUnfrozenParameters:
    def _tiny_model(self):
        pytest.importorskip("torch")
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        return Tiny()

    def test_freezes_all_then_unfreezes_matching(self):
        model = self._tiny_model()
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        n = apply_unfrozen_parameters(model, ["layers.1"])
        # layers.1 has weight + bias = 2 tensors unfrozen
        assert n == 2
        for name, param in model.named_parameters():
            if "layers.1." in name:
                assert param.requires_grad, name
            else:
                assert not param.requires_grad, name

    def test_multiple_patterns(self):
        model = self._tiny_model()
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        n = apply_unfrozen_parameters(model, ["layers.0", "layers.2"])
        assert n == 4  # (0 + 2) × (weight + bias)

    def test_no_match_returns_zero_all_frozen(self):
        model = self._tiny_model()
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        n = apply_unfrozen_parameters(model, ["nonexistent.module"])
        assert n == 0
        assert all(not p.requires_grad for p in model.parameters())

    def test_empty_patterns_rejected(self):
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        with pytest.raises(ValueError, match="non-empty"):
            apply_unfrozen_parameters(self._tiny_model(), [])

    def test_invalid_regex_rejected(self):
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        with pytest.raises(ValueError, match="invalid regex"):
            apply_unfrozen_parameters(self._tiny_model(), ["model.layers.[0"])

    def test_skips_non_float_matched(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Linear(4, 4)
                # a non-float (quantized-like) parameter that matches but
                # cannot require grad — must be skipped, not crash.
                self.idx = nn.Parameter(
                    torch.zeros(4, dtype=torch.long), requires_grad=False
                )

        model = M()
        from soup_cli.utils.freeze import apply_unfrozen_parameters

        n = apply_unfrozen_parameters(model, ["w", "idx"])
        assert n == 2  # w.weight + w.bias; idx skipped (non-float)
        assert not model.idx.requires_grad


class TestSftUnfrozenWiring:
    def test_sft_setup_has_unfrozen_branch(self):
        src = (_SRC / "trainer" / "sft.py").read_text(encoding="utf-8")
        assert "unfrozen_parameters" in src
        assert "apply_unfrozen_parameters" in src
        # the Spectrum branch must guard on the flag (full-FT, skip LoRA)
        assert "if tcfg.unfrozen_parameters" in src
        # and must re-enable input-require-grads so gradient checkpointing
        # works with frozen embeddings (review H1).
        assert "enable_input_require_grads" in src

    def test_grad_flows_with_gradient_checkpointing(self):
        # review H1: full-FT of a mid-stack layer with frozen embeddings +
        # gradient checkpointing must still backprop. The fix is
        # enable_input_require_grads() — exercise the exact mechanism.
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from transformers import LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.freeze import apply_unfrozen_parameters

        cfg = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        )
        model = LlamaForCausalLM(cfg)
        model.gradient_checkpointing_enable()
        apply_unfrozen_parameters(model, ["model.layers.1.mlp.down_proj"])
        model.enable_input_require_grads()  # the sft.py branch does this
        model.train()
        ids = torch.tensor([[1, 2, 3, 4]])
        model(input_ids=ids, labels=ids).loss.backward()
        grad = model.model.layers[1].mlp.down_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all()


class TestSpectrumScanRobustness:
    def test_oversized_matrix_skipped(self, tmp_path, monkeypatch):
        from soup_cli.utils import spectrum_scan

        _write_tiny_safetensors(tmp_path)
        # cap below the 32×32 attn matrices (1024 elements) → all skipped
        monkeypatch.setattr(spectrum_scan, "_MAX_MATRIX_ELEMENTS", 100)
        layers = spectrum_scan.scan_weights_dir(str(tmp_path), modules=("attn",))
        assert layers == ()

    @pytest.mark.skipif(
        os.name == "nt", reason="symlink creation needs privilege on Windows"
    )
    def test_symlinked_shard_skipped(self, tmp_path):
        from soup_cli.utils.spectrum_scan import _discover_safetensors

        _write_tiny_safetensors(tmp_path)
        real = tmp_path / "model.safetensors"
        link = tmp_path / "evil.safetensors"
        os.symlink(real, link)
        found = {os.path.basename(p) for p in _discover_safetensors(str(tmp_path))}
        assert "model.safetensors" in found
        assert "evil.safetensors" not in found


class TestParamPrefix:
    def test_strips_weight_suffix(self):
        from soup_cli.utils.spectrum_scan import param_prefix

        assert param_prefix("model.layers.0.mlp.down_proj.weight") == (
            "model.layers.0.mlp.down_proj"
        )
        assert param_prefix("lm_head") == "lm_head"
