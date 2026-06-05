"""v0.71.14 — Export QA + serve(transformers) + GPU smoke finale.

Closes the doable subset of the finale patch on the Windows + RTX 3050 4 GB box:

* **#96** — live torch runtime for ``soup merge-sharded-fsdp-weights``
  (``consolidate_shards`` + ``ConsolidationResult``).
* **#140** — live ``kv_cache_type`` wiring on the **transformers** backend
  (``apply_kv_cache_type`` lifts the v0.53.1 ``NotImplementedError`` stub;
  ``--kv-cache-type`` CLI flag on ``soup serve``).
* **#71** — ONNX export QA (recorded in ``tests/qa/v07114_qa.md``).

Infra/tooling-blocked items (#70 GGUF / #72 AWQ-GPTQ / #144 doc / #74 HF push /
#79 meta) stay OPEN with ``infra-blocked`` labels — see the patch notes.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX symlink semantics"
)


# =====================================================================
# #96 — live FSDP shard consolidation
# =====================================================================


def _write_shard(path: Path, tensors: dict) -> None:
    import torch

    torch.save(tensors, str(path))


class TestConsolidateShards:
    def test_merges_two_disjoint_shards(self, tmp_path, monkeypatch):
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(
            shards / "pytorch_model_fsdp_0.bin",
            {"model.layers.0.weight": torch.ones(2, 3)},
        )
        _write_shard(
            shards / "pytorch_model_fsdp_1.bin",
            {"model.layers.1.weight": torch.zeros(4, 5)},
        )
        target = tmp_path / "merged.safetensors"
        plan = plan_consolidation(str(shards), str(target))
        result = consolidate_shards(plan)
        assert result.num_tensors == 2
        assert result.num_shards == 2
        assert result.total_bytes > 0
        assert os.path.exists(target)
        loaded = load_file(str(target))
        assert set(loaded) == {"model.layers.0.weight", "model.layers.1.weight"}
        assert loaded["model.layers.0.weight"].shape == (2, 3)

    def test_streams_one_shard_at_a_time(self, tmp_path, monkeypatch):
        """Memory-friendly: torch.load is called exactly once per shard."""
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"a": torch.ones(2)})
        _write_shard(shards / "pytorch_model_fsdp_1.bin", {"b": torch.ones(2)})
        target = tmp_path / "merged.safetensors"
        plan = plan_consolidation(str(shards), str(target))

        calls = {"n": 0}
        real_load = torch.load

        def _counting_load(*args, **kwargs):
            calls["n"] += 1
            return real_load(*args, **kwargs)

        # consolidate_shards lazy-imports torch and calls torch.load — patch
        # the real module so the count reflects per-shard streaming.
        monkeypatch.setattr(torch, "load", _counting_load)
        consolidate_shards(plan)
        assert calls["n"] == 2

    def test_shape_conflict_rejected(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"w": torch.ones(2, 3)})
        _write_shard(shards / "pytorch_model_fsdp_1.bin", {"w": torch.ones(4, 5)})
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        with pytest.raises(ValueError, match="shape"):
            consolidate_shards(plan)

    def test_duplicate_key_same_shape_kept_once(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"w": torch.ones(2, 3)})
        _write_shard(shards / "pytorch_model_fsdp_1.bin", {"w": torch.ones(2, 3)})
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        result = consolidate_shards(plan)
        assert result.num_tensors == 1

    def test_duplicate_key_warns(self, tmp_path, monkeypatch, caplog):
        """A same-shape duplicate across shards logs a WARNING (no silent drop)."""
        import logging

        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"w": torch.ones(2, 3)})
        _write_shard(shards / "pytorch_model_fsdp_1.bin", {"w": torch.zeros(2, 3)})
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        with caplog.at_level(
            logging.WARNING, logger="soup_cli.utils.fsdp_consolidate"
        ):
            consolidate_shards(plan)
        assert any(
            "more than one shard" in rec.message for rec in caplog.records
        )

    def test_non_dict_shard_rejected(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        # A bare tensor, not a state-dict.
        torch.save(torch.ones(3), str(shards / "pytorch_model_fsdp_0.bin"))
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        with pytest.raises(ValueError, match="state-dict"):
            consolidate_shards(plan)

    def test_empty_merged_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {})
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        with pytest.raises(ValueError, match="no tensors"):
            consolidate_shards(plan)

    def test_result_frozen(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"a": torch.ones(2)})
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        result = consolidate_shards(plan)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.num_tensors = 99  # type: ignore[misc]

    def test_non_plan_rejected(self):
        from soup_cli.utils.fsdp_consolidate import consolidate_shards

        with pytest.raises(TypeError):
            consolidate_shards({"not": "a plan"})  # type: ignore[arg-type]

    @_POSIX_ONLY
    def test_symlinked_shard_rejected(self, tmp_path, monkeypatch):
        # A symlinked shard child could redirect torch.load to an arbitrary
        # target — consolidate_shards must reject it (TOCTOU defence). The
        # output-symlink case is handled earlier: plan_consolidation realpaths
        # the output and rejects an out-of-cwd target as "outside cwd".
        import torch

        from soup_cli.utils.fsdp_consolidate import (
            consolidate_shards,
            plan_consolidation,
        )

        monkeypatch.chdir(tmp_path)
        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(shards / "pytorch_model_fsdp_0.bin", {"a": torch.ones(2)})
        # A second "shard" that is actually a symlink (to a real under-cwd file).
        real_target = tmp_path / "secret.bin"
        _write_shard(real_target, {"b": torch.ones(2)})
        os.symlink(real_target, shards / "pytorch_model_fsdp_1.bin")
        plan = plan_consolidation(str(shards), str(tmp_path / "m.safetensors"))
        with pytest.raises(ValueError, match="symlink"):
            consolidate_shards(plan)

    def test_source_uses_atomic_write_bytes(self):
        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "soup_cli"
            / "utils"
            / "fsdp_consolidate.py"
        ).read_text(encoding="utf-8")
        assert "atomic_write_bytes" in src
        assert "weights_only=True" in src


class TestMergeShardedCli:
    def _make_shards(self, tmp_path):
        import torch

        shards = tmp_path / "shards"
        shards.mkdir()
        _write_shard(
            shards / "pytorch_model_fsdp_0.bin",
            {"model.layers.0.weight": torch.ones(2, 2)},
        )
        _write_shard(
            shards / "pytorch_model_fsdp_1.bin",
            {"model.layers.1.weight": torch.zeros(2, 2)},
        )
        return shards

    def test_live_consolidation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = self._make_shards(tmp_path)
        target = tmp_path / "merged.safetensors"
        result = runner.invoke(
            app,
            ["merge-sharded-fsdp-weights", str(shards), "-o", str(target)],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert target.exists()
        assert "2" in result.output  # 2 tensors / 2 shards reported

    def test_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = self._make_shards(tmp_path)
        target = tmp_path / "merged.safetensors"
        result = runner.invoke(
            app,
            [
                "merge-sharded-fsdp-weights",
                str(shards),
                "-o",
                str(target),
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Plan" in result.output
        assert not target.exists()  # plan-only writes nothing

    def test_outside_cwd_output_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shards = self._make_shards(tmp_path)
        result = runner.invoke(
            app,
            [
                "merge-sharded-fsdp-weights",
                str(shards),
                "-o",
                str(tmp_path.parent / "escape.safetensors"),
            ],
        )
        assert result.exit_code == 2

    def test_missing_shards_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(
            app,
            [
                "merge-sharded-fsdp-weights",
                str(empty),
                "-o",
                str(tmp_path / "m.safetensors"),
            ],
        )
        assert result.exit_code == 2

    def test_help_lists_plan_only(self):
        import re

        result = runner.invoke(app, ["merge-sharded-fsdp-weights", "--help"])
        assert result.exit_code == 0
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--plan-only" in clean


# =====================================================================
# #140 — live kv_cache_type wiring (transformers backend)
# =====================================================================


class TestApplyKvCacheType:
    def test_no_longer_raises_notimplemented(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("bf16", backend="transformers")
        assert rt.kv_cache_type == "bf16"

    def test_bf16_model_dtype(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("bf16", backend="transformers")
        assert rt.model_dtype == "bfloat16"
        assert dict(rt.generate_kwargs) == {}
        assert rt.requires_quant_backend is False

    def test_f16_model_dtype(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("f16", backend="transformers")
        assert rt.model_dtype == "float16"
        assert dict(rt.generate_kwargs) == {}

    def test_q8_0_quantized_cache_kwargs(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("q8_0", backend="transformers")
        gk = dict(rt.generate_kwargs)
        assert gk["cache_implementation"] == "quantized"
        cfg = gk["cache_config"]
        assert cfg["backend"] == "hqq"
        assert cfg["nbits"] == 8
        assert rt.model_dtype is None
        assert rt.requires_quant_backend is True

    def test_case_insensitive(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("Q8_0", backend="transformers")
        assert rt.kv_cache_type == "q8_0"

    def test_fp8_non_hopper_friendly_error(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(RuntimeError, match="Hopper"):
            apply_kv_cache_type(
                "fp8", backend="transformers", compute_capability=(8, 6)
            )

    def test_fp8_hopper_still_unsupported_on_transformers(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(RuntimeError, match="vLLM"):
            apply_kv_cache_type(
                "fp8", backend="transformers", compute_capability=(9, 0)
            )

    def test_fp8_unknown_cc_raises(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(RuntimeError):
            apply_kv_cache_type(
                "fp8", backend="transformers", compute_capability=None
            )

    def test_vllm_backend_deferred(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(NotImplementedError, match="vLLM|transformers"):
            apply_kv_cache_type("q8_0", backend="vllm")

    def test_sglang_backend_deferred(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(NotImplementedError):
            apply_kv_cache_type("bf16", backend="sglang")

    def test_invalid_type_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(ValueError, match="not supported"):
            apply_kv_cache_type("wat", backend="transformers")

    def test_oversize_type_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(ValueError, match="too long"):
            apply_kv_cache_type("x" * 17, backend="transformers")

    def test_null_byte_type_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(ValueError, match="null byte"):
            apply_kv_cache_type("bf\x0016", backend="transformers")

    def test_bool_type_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(TypeError):
            apply_kv_cache_type(True, backend="transformers")  # type: ignore[arg-type]

    def test_bool_backend_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(TypeError):
            apply_kv_cache_type("bf16", backend=True)  # type: ignore[arg-type]

    def test_non_str_backend_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(TypeError):
            apply_kv_cache_type("bf16", backend=123)  # type: ignore[arg-type]

    def test_bad_compute_capability_shape_rejected(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises((TypeError, ValueError)):
            apply_kv_cache_type(
                "fp8", backend="transformers", compute_capability=(9,)  # type: ignore[arg-type]
            )

    def test_runtime_frozen(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("bf16", backend="transformers")
        with pytest.raises(dataclasses.FrozenInstanceError):
            rt.kv_cache_type = "f16"  # type: ignore[misc]

    def test_generate_kwargs_immutable(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("q8_0", backend="transformers")
        assert isinstance(rt.generate_kwargs, MappingProxyType)
        with pytest.raises(TypeError):
            rt.generate_kwargs["x"] = 1  # type: ignore[index]

    def test_quantized_cache_backend_available(self):
        from soup_cli.utils.kv_cache import quantized_cache_backend_available

        # On this box neither hqq nor quanto is installed → None. The return
        # type is Optional[str]; assert it's None-or-str without asserting a
        # specific value (CI may differ).
        got = quantized_cache_backend_available()
        assert got is None or isinstance(got, str)


class TestPlainKvKwargs:
    def test_deep_converts_nested_mappingproxy(self):
        from soup_cli.commands.serve import _plain_kv_kwargs

        nested = MappingProxyType(
            {"cache_implementation": "quantized",
             "cache_config": MappingProxyType({"backend": "hqq", "nbits": 8})}
        )
        out = _plain_kv_kwargs(nested)
        assert isinstance(out, dict)
        assert isinstance(out["cache_config"], dict)
        # Plain dicts are mutable — proves the proxy was unwrapped.
        out["cache_config"]["nbits"] = 4
        assert out == {
            "cache_implementation": "quantized",
            "cache_config": {"backend": "hqq", "nbits": 4},
        }

    def test_empty_mapping(self):
        from soup_cli.commands.serve import _plain_kv_kwargs

        assert _plain_kv_kwargs(MappingProxyType({})) == {}

    def test_q8_0_runtime_round_trips_to_plain(self):
        from soup_cli.commands.serve import _plain_kv_kwargs
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        rt = apply_kv_cache_type("q8_0", backend="transformers")
        plain = _plain_kv_kwargs(rt.generate_kwargs)
        assert plain["cache_implementation"] == "quantized"
        assert plain["cache_config"]["backend"] == "hqq"
        assert plain["cache_config"]["nbits"] == 8


class TestServeKvCacheCli:
    def test_help_lists_flag(self):
        import re

        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--kv-cache-type" in clean

    def test_invalid_type_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        result = runner.invoke(
            app,
            ["serve", "--model", str(model), "--kv-cache-type", "wat"],
        )
        assert result.exit_code == 2

    def test_fp8_on_ampere_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        # Force a non-Hopper compute capability so the friendly Hopper error
        # fires regardless of the host GPU.
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))
        result = runner.invoke(
            app,
            ["serve", "--model", str(model), "--kv-cache-type", "fp8"],
        )
        assert result.exit_code == 2

    def test_vllm_backend_deferred_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(model),
                "--backend",
                "vllm",
                "--kv-cache-type",
                "q8_0",
            ],
        )
        assert result.exit_code == 2


# =====================================================================
# #71 — ONNX export QA (recorded in the QA log)
# =====================================================================


class TestOnnxQaLog:
    def test_qa_log_exists(self):
        log = (
            Path(__file__).resolve().parent
            / "qa"
            / "v07114_qa.md"
        )
        assert log.exists(), "v07114 QA log missing"
        body = log.read_text(encoding="utf-8")
        assert "#71" in body
        assert "onnx" in body.lower()


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(x) for x in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 14)

    def test_no_top_level_torch_in_kv_cache(self):
        src = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "soup_cli"
            / "utils"
            / "kv_cache.py"
        ).read_text(encoding="utf-8")
        for line in src.splitlines():
            assert not line.startswith("import torch")
            assert not line.startswith("from torch")
