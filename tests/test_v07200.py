"""v0.72.0 — Layer Streaming (BETA).

Pure-side truth tables + the four silent-failure regression classes the
correctness/throughput gates surfaced on a real RTX 3050 4 GB box.

The CUDA-gated tests are skipped without a GPU and say so.
"""

import ast
import os

import pytest


# ==========================================================================
# C1 — the pure planner (utils/layer_stream.py)
# ==========================================================================
class _Cfg:
    """Minimal AutoConfig stand-in: the planner only reads model_type."""

    def __init__(self, model_type=None, **kw):
        if model_type is not None:
            self.model_type = model_type
        for key, val in kw.items():
            setattr(self, key, val)


class TestStreamArchAllowlist:
    def test_llama_family_supported(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        assert stream_arch_of(_Cfg("llama")) == "llama"

    def test_qwen2_and_qwen3_supported(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        assert stream_arch_of(_Cfg("qwen2")) == "qwen2"
        assert stream_arch_of(_Cfg("qwen3")) == "qwen3"

    def test_case_insensitive(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        assert stream_arch_of(_Cfg("LLaMA")) == "llama"

    def test_unsupported_arch_names_the_allowlist(self):
        """v0.72.0 scope is Llama/Qwen. The refusal must say what IS allowed."""
        from soup_cli.utils.layer_stream import stream_arch_of

        with pytest.raises(ValueError, match="llama"):
            stream_arch_of(_Cfg("gpt2"))

    def test_multimodal_gemma3_is_rejected_while_gemma3_text_is_not(self):
        """v0.72.3 added the Gemma family, but ``gemma3`` and ``gemma3_text`` are
        NOT the same thing: a real ``google/gemma-3-*`` reports ``gemma3`` for its
        vision-capable wrapper. Streaming that as a causal LM is exactly the
        silent mis-train the allowlist exists to prevent, so the pair is pinned
        together — the accepted half is the control that makes the refusal
        meaningful rather than a spelling accident."""
        from soup_cli.utils.layer_stream import stream_arch_of

        assert stream_arch_of(_Cfg("gemma3_text")) == "gemma3_text"
        with pytest.raises(ValueError, match="gemma3"):
            stream_arch_of(_Cfg("gemma3"))

    def test_missing_model_type_raises(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        with pytest.raises(ValueError, match="model_type"):
            stream_arch_of(_Cfg())


class TestChooseTier:
    def test_fits_in_ram_returns_ram(self):
        from soup_cli.utils.layer_stream import TIER_RAM, choose_tier

        assert choose_tier(1_000, 10_000, "nvme") == TIER_RAM

    def test_headroom_is_strict_at_the_boundary(self):
        """model_bytes < free*0.7 — exactly at 0.7 is NOT ram (plan 5.1)."""
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        assert choose_tier(7_000, 10_000, "nvme") == TIER_DISK
        assert choose_tier(6_999, 10_000, "nvme") == "ram"

    def test_too_big_for_ram_falls_to_nvme(self):
        from soup_cli.utils.layer_stream import TIER_DISK, choose_tier

        assert choose_tier(9_000, 10_000, "nvme") == TIER_DISK

    def test_hdd_is_refused_loudly(self):
        """plan P11: 80 shards x 2 reads = 160 seeks per step."""
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="NVMe"):
            choose_tier(9_000, 10_000, "hdd")

    def test_sata_ssd_is_refused(self):
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="NVMe"):
            choose_tier(9_000, 10_000, "sata")

    def test_unknown_disk_kind_is_refused(self):
        from soup_cli.utils.layer_stream import choose_tier

        with pytest.raises(ValueError, match="NVMe"):
            choose_tier(9_000, 10_000, "wat")


class TestDecidePinning:
    """User requirement: the 7.12 GB pinned ceiling measured on the dev box is
    real behaviour, not a caveat. Exceeding it must fall back to a pageable
    store AND say the utilisation cost out loud."""

    def test_store_within_limit_is_pinned(self):
        from soup_cli.utils.layer_stream import decide_pinning

        decision = decide_pinning(2 * 10**9, 7 * 10**9)
        assert decision.pinned is True

    def test_store_above_limit_falls_back_to_pageable(self):
        from soup_cli.utils.layer_stream import decide_pinning

        decision = decide_pinning(8 * 10**9, 7 * 10**9)
        assert decision.pinned is False

    def test_fallback_reason_states_the_utilisation_cost(self):
        """Silently taking the 97% -> 79% hit is the failure mode."""
        from soup_cli.utils.layer_stream import decide_pinning

        decision = decide_pinning(8 * 10**9, 7 * 10**9)
        assert "utilisation" in decision.reason.lower()
        assert "pageable" in decision.reason.lower()

    def test_unknown_limit_attempts_pinning(self):
        """No probe result -> try pinned; the runtime falls back on failure."""
        from soup_cli.utils.layer_stream import decide_pinning

        assert decide_pinning(8 * 10**9, None).pinned is True

    def test_boundary_equal_is_pinned(self):
        from soup_cli.utils.layer_stream import decide_pinning

        assert decide_pinning(7 * 10**9, 7 * 10**9).pinned is True


class TestBufferValidation:
    def test_default_is_two(self):
        from soup_cli.utils.layer_stream import DEFAULT_STREAM_BUFFERS

        assert DEFAULT_STREAM_BUFFERS == 2

    def test_valid_range_accepted(self):
        from soup_cli.utils.layer_stream import validate_stream_buffers

        for n in (2, 3, 8):
            assert validate_stream_buffers(n) == n

    def test_one_buffer_is_a_scheduler_bug_not_a_config(self):
        """plan 8: '1 is a scheduler bug, not a config'."""
        from soup_cli.utils.layer_stream import validate_stream_buffers

        with pytest.raises(ValueError, match="2"):
            validate_stream_buffers(1)

    def test_above_max_rejected(self):
        from soup_cli.utils.layer_stream import validate_stream_buffers

        with pytest.raises(ValueError, match="8"):
            validate_stream_buffers(9)

    def test_bool_rejected(self):
        from soup_cli.utils.layer_stream import validate_stream_buffers

        with pytest.raises(ValueError, match="bool"):
            validate_stream_buffers(True)


class TestThroughputEstimate:
    def test_uses_c_equals_six_for_gradient_checkpointing(self):
        """plan 4.3: C=6 WITH checkpointing (2 fwd + 2 recompute + 2 dL/dx).
        The source docs used C=4 while also mandating checkpointing."""
        from soup_cli.utils.layer_stream import FLOPS_PER_PARAM_PER_TOKEN

        assert FLOPS_PER_PARAM_PER_TOKEN == 6

    def test_hand_computed_rate(self):
        from soup_cli.utils.layer_stream import estimate_stream_tokens_per_sec

        # 4.9 TFLOPS effective / (6 * 1.5437e9) = 529.1 tok/s
        got = estimate_stream_tokens_per_sec(1_543_714_304, 4.9)
        assert got == pytest.approx(529.1, rel=0.01)

    def test_matches_the_measured_15b_number_within_10pct(self):
        """Calibration: the dev box measured 525.0 tok/s on Qwen2.5-1.5B at
        4.86 TFLOPS effective. The model must reproduce it."""
        from soup_cli.utils.layer_stream import estimate_stream_tokens_per_sec

        got = estimate_stream_tokens_per_sec(1_543_714_304, 4.86)
        assert 0.9 * 525.0 <= got <= 1.1 * 525.0

    def test_zero_params_rejected(self):
        from soup_cli.utils.layer_stream import estimate_stream_tokens_per_sec

        with pytest.raises(ValueError):
            estimate_stream_tokens_per_sec(0, 4.9)

    def test_epoch_seconds(self):
        from soup_cli.utils.layer_stream import estimate_epoch_seconds

        assert estimate_epoch_seconds(1_000_000, 143.1) == pytest.approx(6987.0, rel=0.01)

    def test_epoch_seconds_rejects_non_positive_rate(self):
        from soup_cli.utils.layer_stream import estimate_epoch_seconds

        with pytest.raises(ValueError):
            estimate_epoch_seconds(1000, 0.0)


class TestVramEstimate:
    def test_sums_the_plan_41_terms(self):
        from soup_cli.utils.layer_stream import estimate_stream_vram

        got = estimate_stream_vram(
            layer_bytes=100,
            buffers=2,
            embed_bytes=50,
            adapter_bytes=10,
            activation_bytes=5,
            logits_bytes=20,
            workspace_bytes=1000,
        )
        assert got == 2 * 100 + 50 + 10 + 5 + 20 + 1000

    def test_buffer_count_scales_the_layer_term(self):
        from soup_cli.utils.layer_stream import estimate_stream_vram

        two = estimate_stream_vram(layer_bytes=100, buffers=2, embed_bytes=0, workspace_bytes=0)
        three = estimate_stream_vram(layer_bytes=100, buffers=3, embed_bytes=0, workspace_bytes=0)
        assert three - two == 100

    def test_logits_budget_includes_the_whole_loss_path(self):
        """plan P5: logits, not weights, OOM you first on a small card.

        v0.72.0 charged 2 + 4 here (bf16 logits + fp32 upcast) from first
        principles. v0.72.3 GATE 2 measured the real peak: 14 bytes per element,
        because ``ForCausalLMLoss`` also holds log-softmax's fp32 output and the
        fp32 gradient live at the same time. The old figure under-predicted this
        term by 2.33x — ~5 GB on a 152k-vocab model at batch 8."""
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(vocab_size=151936, seq_len=512, batch_size=1)
        assert got == 512 * 151936 * 14

    def test_logits_budget_without_upcast(self):
        from soup_cli.utils.layer_stream import estimate_logits_bytes

        got = estimate_logits_bytes(
            vocab_size=1000, seq_len=10, batch_size=1, upcast_fp32=False
        )
        assert got == 10 * 1000 * 2


class TestLayerSpec:
    def test_nbytes_from_shape_and_dtype(self):
        from soup_cli.utils.layer_stream import LayerSpec

        spec = LayerSpec(name="self_attn.q_proj.weight", shape=(2048, 2048), dtype="bfloat16")
        assert spec.nbytes == 2048 * 2048 * 2

    def test_is_frozen(self):
        import dataclasses

        from soup_cli.utils.layer_stream import LayerSpec

        spec = LayerSpec(name="w", shape=(2, 2), dtype="bfloat16")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "other"

    def test_unsupported_dtype_rejected(self):
        from soup_cli.utils.layer_stream import LayerSpec

        with pytest.raises(ValueError, match="dtype"):
            LayerSpec(name="w", shape=(2, 2), dtype="int4").nbytes


class TestStreamPlan:
    def test_build_plan_reports_tier_and_pinning(self):
        from soup_cli.utils.layer_stream import build_stream_plan

        plan = build_stream_plan(
            arch="qwen2",
            n_layers=36,
            layer_bytes=154 * 10**6,
            embed_bytes=622 * 10**6,
            available_ram_bytes=12 * 10**9,
            pinned_limit_bytes=7 * 10**9,
            buffers=2,
            disk_kind="nvme",
        )
        assert plan.tier == "ram"
        assert plan.n_layers == 36
        assert plan.store_bytes == 36 * 154 * 10**6
        # 5.5 GB store fits under a 7 GB pinned ceiling
        assert plan.pinned is True

    def test_plan_falls_back_to_pageable_and_records_a_note(self):
        from soup_cli.utils.layer_stream import build_stream_plan

        plan = build_stream_plan(
            arch="qwen2",
            n_layers=36,
            layer_bytes=300 * 10**6,
            embed_bytes=622 * 10**6,
            available_ram_bytes=40 * 10**9,
            pinned_limit_bytes=7 * 10**9,
            buffers=2,
            disk_kind="nvme",
        )
        assert plan.pinned is False
        assert any("pageable" in note.lower() for note in plan.notes)

    def test_plan_is_frozen(self):
        import dataclasses

        from soup_cli.utils.layer_stream import build_stream_plan

        plan = build_stream_plan(
            arch="llama", n_layers=2, layer_bytes=10, embed_bytes=10,
            available_ram_bytes=10**9, pinned_limit_bytes=None, buffers=2, disk_kind="nvme",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.tier = "disk"

    def test_render_panel_names_the_beta_status_and_tier(self):
        from rich.console import Console
        from rich.panel import Panel

        from soup_cli.utils.layer_stream import build_stream_plan, render_stream_panel

        plan = build_stream_plan(
            arch="llama", n_layers=2, layer_bytes=10, embed_bytes=10,
            available_ram_bytes=10**9, pinned_limit_bytes=None, buffers=2, disk_kind="nvme",
        )
        panel = render_stream_panel(plan)
        assert isinstance(panel, Panel)
        import io

        buf = io.StringIO()
        Console(file=buf, width=100).print(panel)
        out = buf.getvalue()
        assert "BETA" in out
        assert "ram" in out.lower()


class TestNoTopLevelTorch:
    """utils/layer_stream.py is the pure half — importing torch at module level
    would put torch on the light CLI's import path."""

    def test_layer_stream_has_no_top_level_torch(self):
        import soup_cli.utils.layer_stream as mod

        source = open(mod.__file__, encoding="utf-8").read()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("torch"), alias.name
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("torch"), node.module

    def test_importable_without_torch_installed(self):
        """The light install has no torch; the planner must still import."""
        import importlib
        import sys

        saved = {k: v for k, v in sys.modules.items() if k.startswith("torch")}
        for key in list(saved):
            sys.modules[key] = None
        try:
            importlib.reload(importlib.import_module("soup_cli.utils.layer_stream"))
        finally:
            for key, val in saved.items():
                sys.modules[key] = val
            importlib.reload(importlib.import_module("soup_cli.utils.layer_stream"))


# ==========================================================================
# C2 — the checkpoint sharder (utils/layer_shard.py)
# ==========================================================================
def _write_safetensors(path, tensors):
    """Write a fake HF weights shard (torch is a [train]-extra dep)."""
    from safetensors.torch import save_file

    save_file({k: v.clone() for k, v in tensors.items()}, path)
    return path


def _fake_weights_dir(tmp_path, n_layers=3, split=False):
    import torch

    torch.manual_seed(0)
    layer = {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        layer[pre + "self_attn.q_proj.weight"] = torch.randn(8, 8, dtype=torch.float32)
        layer[pre + "mlp.down_proj.weight"] = torch.randn(8, 16, dtype=torch.float32)
        layer[pre + "input_layernorm.weight"] = torch.randn(8, dtype=torch.float32)
    extras = {
        "model.embed_tokens.weight": torch.randn(32, 8, dtype=torch.float32),
        "model.norm.weight": torch.randn(8, dtype=torch.float32),
    }
    src = tmp_path / "weights"
    src.mkdir()
    if split:
        # a layer's tensors deliberately straddle two files
        keys = sorted(layer)
        half = len(keys) // 2
        _write_safetensors(
            str(src / "model-00001.safetensors"), {k: layer[k] for k in keys[:half]}
        )
        _write_safetensors(
            str(src / "model-00002.safetensors"),
            {**{k: layer[k] for k in keys[half:]}, **extras},
        )
    else:
        _write_safetensors(str(src / "model.safetensors"), {**layer, **extras})
    return str(src), layer, extras


class TestShardRoundTrip:
    def test_layer_tensors_survive_byte_identical(self, tmp_path):
        """A sharder that silently corrupts a tensor still trains — badly."""
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, layer, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(src, out, dtype="float32")

        assert index.n_layers == 3
        blob = load_file(layer_shard_path(out, 1))
        for key, tensor in layer.items():
            if key.startswith("model.layers.1."):
                short = key[len("model.layers.1.") :]
                assert torch.equal(blob[short], tensor), short

    def test_extras_are_separated_from_layers(self, tmp_path):
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import extras_shard_path, shard_checkpoint

        src, _, extras = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        blob = load_file(extras_shard_path(out))
        assert set(blob) == set(extras)

    def test_layers_split_across_source_files_are_gathered(self, tmp_path):
        """Real checkpoints shard by size, not by layer boundary."""
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, layer, _ = _fake_weights_dir(tmp_path, split=True)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(src, out, dtype="float32")
        assert index.n_layers == 3
        for idx in range(3):
            blob = load_file(layer_shard_path(out, idx))
            assert set(blob) == {
                "self_attn.q_proj.weight",
                "mlp.down_proj.weight",
                "input_layernorm.weight",
            }
            pre = f"model.layers.{idx}."
            assert torch.equal(
                blob["mlp.down_proj.weight"], layer[pre + "mlp.down_proj.weight"]
            )

    def test_dtype_is_converted(self, tmp_path):
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(src, out, dtype="bfloat16")
        assert index.dtype == "bfloat16"
        blob = load_file(layer_shard_path(out, 0))
        assert blob["self_attn.q_proj.weight"].dtype == torch.bfloat16

    def test_index_round_trips(self, tmp_path):
        from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        written = shard_checkpoint(src, out, dtype="float32", arch="llama")
        read = read_shard_index(out)
        assert read == written
        assert read.arch == "llama"
        assert read.total_params > 0
        assert "self_attn.q_proj.weight" in read.layer_keys

    def test_second_call_reuses_the_cache(self, tmp_path):
        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        before = os.path.getmtime(layer_shard_path(out, 0))
        shard_checkpoint(src, out, dtype="float32")
        assert os.path.getmtime(layer_shard_path(out, 0)) == before

    def test_force_reshards(self, tmp_path):
        from soup_cli.utils.layer_shard import read_shard_index, shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        index = shard_checkpoint(src, out, dtype="bfloat16", force=True)
        assert index.dtype == "bfloat16"
        assert read_shard_index(out).dtype == "bfloat16"

    def test_dtype_change_invalidates_a_stale_cache(self, tmp_path):
        """Reusing float32 shards for a bfloat16 request would stream the wrong
        dtype into the pool and silently mis-train."""
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        index = shard_checkpoint(src, out, dtype="bfloat16")
        assert index.dtype == "bfloat16"


class TestShardGuards:
    def test_no_safetensors_refuses(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="safetensors"):
            shard_checkpoint(str(empty), str(tmp_path / "out"))

    def test_no_decoder_layers_refuses(self, tmp_path):
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint

        src = tmp_path / "weights"
        src.mkdir()
        _write_safetensors(
            str(src / "model.safetensors"), {"model.embed_tokens.weight": torch.randn(4, 4)}
        )
        with pytest.raises(ValueError, match="decoder layer"):
            shard_checkpoint(str(src), str(tmp_path / "out"), dtype="float32")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_symlinked_source_shard_is_skipped(self, tmp_path):
        """Mirrors spectrum_scan._discover_safetensors."""
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        outside = tmp_path / "outside.safetensors"
        _write_safetensors(
            str(outside), {"model.layers.9.mlp.down_proj.weight": torch.randn(4, 4)}
        )
        os.symlink(str(outside), os.path.join(src, "model-evil.safetensors"))
        index = shard_checkpoint(src, str(tmp_path / "out"), dtype="float32")
        assert index.n_layers == 3  # the symlinked layer 9 was not picked up

    def test_too_many_layers_refused(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_shard as mod

        src, _, _ = _fake_weights_dir(tmp_path)
        monkeypatch.setattr(mod, "_MAX_LAYERS", 2)
        with pytest.raises(ValueError, match="layers"):
            mod.shard_checkpoint(src, str(tmp_path / "out"), dtype="float32")

    def test_oversized_tensor_refused(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_shard as mod

        src, _, _ = _fake_weights_dir(tmp_path)
        monkeypatch.setattr(mod, "_MAX_TENSOR_ELEMENTS", 4)
        with pytest.raises(ValueError, match="too large"):
            mod.shard_checkpoint(src, str(tmp_path / "out"), dtype="float32")

    def test_unsupported_dtype_refused(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        with pytest.raises(ValueError, match="dtype"):
            shard_checkpoint(src, str(tmp_path / "out"), dtype="int4")

    def test_corrupt_index_reshards_instead_of_crashing(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        index = shard_checkpoint(src, out, dtype="float32")
        assert index.n_layers == 3


class TestShardCacheDir:
    def test_slug_is_traversal_safe(self):
        from soup_cli.utils.layer_shard import model_slug

        slug = model_slug("../../etc/passwd")
        assert "/" not in slug and "\\" not in slug and ".." not in slug

    def test_slug_maps_org_slash_name(self):
        from soup_cli.utils.layer_shard import model_slug

        assert model_slug("Qwen/Qwen2.5-1.5B") == "Qwen__Qwen2.5-1.5B"

    def test_slug_rejects_empty(self):
        from soup_cli.utils.layer_shard import model_slug

        with pytest.raises(ValueError):
            model_slug("   ")

    def test_env_override_inside_tmp_is_honoured(self, tmp_path, monkeypatch):
        from soup_cli.utils.layer_shard import resolve_shard_dir

        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path))
        got = resolve_shard_dir("Qwen/Qwen2.5-1.5B")
        assert os.path.realpath(str(tmp_path)) in os.path.realpath(got)

    def test_env_override_outside_bounds_falls_back(self, monkeypatch):
        """An env var is operator config, not API input — silent fall-through."""
        from soup_cli.utils.layer_shard import (
            default_layer_stream_cache_dir,
            resolve_shard_dir,
        )

        bad = "C:\\Windows\\System32" if os.name == "nt" else "/etc/soup-evil"
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", bad)
        got = os.path.realpath(resolve_shard_dir("m"))
        assert got.startswith(os.path.realpath(default_layer_stream_cache_dir()))

    def test_control_chars_in_override_rejected(self, tmp_path, monkeypatch):
        """ESC, not NUL: a NUL byte cannot live in an environment variable at
        all (POSIX putenv and CPython >= 3.11 both reject it outright), so a
        NUL-based test only ever passed on one platform/interpreter pair.

        The override here is otherwise VALID — a real path under $TMPDIR — so
        the control character is the ONLY reason it can be refused. Drop the
        `ord(ch) < 0x20` guard and this test goes red.
        """
        from soup_cli.utils.layer_shard import (
            default_layer_stream_cache_dir,
            resolve_shard_dir,
        )

        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache") + "\x1b")
        got = os.path.realpath(resolve_shard_dir("m"))
        assert got.startswith(os.path.realpath(default_layer_stream_cache_dir()))


class TestShardNoTopLevelTorch:
    def test_layer_shard_has_no_top_level_torch(self):
        import soup_cli.utils.layer_shard as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("torch", "safetensors")), alias.name
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("torch", "safetensors"))


class TestShardUniformity:
    """The runtime builds its buffer-pool spec from layer 0. A checkpoint whose
    layers disagree would size the pool wrong and stream garbage into it."""

    def test_non_uniform_layer_parameter_set_refused(self, tmp_path):
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint

        src = tmp_path / "weights"
        src.mkdir()
        _write_safetensors(
            str(src / "model.safetensors"),
            {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.layers.1.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.layers.1.mlp.extra.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(4, 4),
            },
        )
        with pytest.raises(ValueError, match="uniform"):
            shard_checkpoint(str(src), str(tmp_path / "out"), dtype="float32")

    def test_non_contiguous_layer_indices_refused(self, tmp_path):
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint

        src = tmp_path / "weights"
        src.mkdir()
        _write_safetensors(
            str(src / "model.safetensors"),
            {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.layers.2.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(4, 4),
            },
        )
        with pytest.raises(ValueError, match="contiguous"):
            shard_checkpoint(str(src), str(tmp_path / "out"), dtype="float32")


# ==========================================================================
# C3 — schema fields + cross-validators (config/schema.py)
# ==========================================================================
def _stream_yaml(training=None, **top):
    """Minimal valid streaming config; `training` overlays the training block."""
    import yaml

    cfg = {
        "base": "HuggingFaceTB/SmolLM2-135M",
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {"train": "train.jsonl"},
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "stream_layers": True,
        },
    }
    cfg.update(top)
    if training:
        cfg["training"].update(training)
    return yaml.safe_dump(cfg)


def _load(yaml_text):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(yaml_text)


class TestStreamSchemaDefaults:
    def test_defaults_are_off(self):
        cfg = _load(_stream_yaml(training={"stream_layers": False}))
        assert cfg.training.stream_layers is False
        assert cfg.training.stream_source == "auto"
        assert cfg.training.stream_buffers == 2
        assert cfg.training.stream_vram_override is None

    def test_happy_path_parses(self):
        cfg = _load(_stream_yaml(training={"stream_source": "ram", "stream_buffers": 3}))
        assert cfg.training.stream_layers is True
        assert cfg.training.stream_source == "ram"
        assert cfg.training.stream_buffers == 3


class TestStreamVramOverride:
    def test_defaults_to_none(self):
        assert _load(_stream_yaml()).training.stream_vram_override is None

    def test_accepts_an_explicit_byte_count(self):
        cfg = _load(_stream_yaml(training={"stream_vram_override": 4_000_000_000}))
        assert cfg.training.stream_vram_override == 4_000_000_000

    def test_zero_is_accepted(self):
        """A degenerate but legal value: 'assume nothing is free'."""
        cfg = _load(_stream_yaml(training={"stream_vram_override": 0}))
        assert cfg.training.stream_vram_override == 0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="stream_vram_override"):
            _load(_stream_yaml(training={"stream_vram_override": -1}))

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            _load(_stream_yaml(training={"stream_vram_override": True}))

    def test_without_stream_layers_rejected(self):
        with pytest.raises(ValueError, match="stream_layers"):
            _load(
                _stream_yaml(
                    training={"stream_layers": False, "stream_vram_override": 4_000_000_000}
                )
            )


class TestStreamTaskAndBackendGates:
    def test_unsupported_task_rejected(self):
        """v0.72.4 opened the four preference losses, so `dpo` is no longer the
        example here. `reward_model` still has no streaming path, and `grpo` is
        excluded permanently — rollouts re-read every layer per generated
        token."""
        with pytest.raises(ValueError, match="stream_layers"):
            _load(_stream_yaml(task="reward_model"))
        with pytest.raises(ValueError, match="generation"):
            _load(_stream_yaml(task="grpo"))

    def test_preference_tasks_are_accepted_since_v0724(self):
        assert _load(_stream_yaml(task="dpo")).task == "dpo"

    def test_unsloth_backend_rejected(self):
        with pytest.raises(ValueError, match="transformers"):
            _load(_stream_yaml(backend="unsloth"))

    def test_mlx_backend_rejected(self):
        with pytest.raises(ValueError, match="transformers"):
            _load(_stream_yaml(backend="mlx"))

    def test_vision_modality_rejected(self):
        with pytest.raises(ValueError, match="text"):
            _load(_stream_yaml(modality="vision"))


class TestStreamScopeGates:
    def test_nf4_is_accepted_since_v0_72_2(self):
        """v0.72.0 refused 4-bit and named v0.72.2; v0.72.2 delivered it."""
        cfg = _load(_stream_yaml(training={"quantization": "4bit"}))
        assert cfg.training.quantization == "4bit"

    def test_other_quantisations_still_rejected(self):
        """The gate did not simply disappear — only NF4 was added to it."""
        with pytest.raises(ValueError, match="stream_layers"):
            _load(_stream_yaml(training={"quantization": "8bit"}))

    def test_disk_source_accepted_in_v0723(self):
        """The disk overflow tier shipped in v0.72.3. It stays NVMe-only —
        `choose_tier` refuses spinning disks (plan P11) — but that is a runtime
        decision about the machine, not a schema refusal about the config."""
        cfg = _load(_stream_yaml(training={"stream_source": "disk"}))
        assert cfg.training.stream_source == "disk"

    def test_batch_size_above_one_accepted_in_v0723(self):
        """Lifted in v0.72.3. Bigger batches are where streaming pays off — one
        weight read amortised over more tokens — and the v0.72.2 refusal already
        named this slot, so leaving it would have shipped a lie."""
        assert _load(_stream_yaml(training={"batch_size": 4})).training.batch_size == 4

    def test_auto_batch_size_rejected(self):
        """Still refused, and for a reason that does not expire: "auto" sizes a
        RESIDENT model by OOM-probing it, and a streaming run never loads one."""
        with pytest.raises(ValueError, match="batch_size"):
            _load(_stream_yaml(training={"batch_size": "auto"}))

    def test_gradient_accumulation_accepted_in_v0723(self):
        """Lifted in v0.72.3, and plan P9's premise turned out to be half wrong.
        Accumulation does re-read the base per micro-batch, but it also processes
        that many more tokens, so layer reads per 1k tokens are UNCHANGED
        (measured constant at 175.78 across accum 1/2/4). The real cost is
        opportunity cost against raising batch_size — 2.52x at matched effective
        batch — which is advice for the pre-flight, not grounds for a refusal:
        accumulation is the only way to raise effective batch once VRAM is
        spent."""
        cfg = _load(_stream_yaml(training={"gradient_accumulation_steps": 4}))
        assert cfg.training.gradient_accumulation_steps == 4

    def test_lora_disabled_is_a_no_op_and_rejected(self):
        """Streaming a frozen base with nothing trainable trains nothing."""
        with pytest.raises(ValueError, match="LoRA"):
            _load(_stream_yaml(training={"lora": {"r": 0}}))


class TestStreamMutualExclusions:
    def test_unfrozen_parameters_conflict(self):
        with pytest.raises(ValueError, match="unfrozen_parameters"):
            _load(_stream_yaml(training={"unfrozen_parameters": ["model.layers.0.mlp"]}))

    def test_lisa_conflict(self):
        with pytest.raises(ValueError, match="lisa_enabled"):
            _load(_stream_yaml(training={"lisa_enabled": True}))

    def test_packing_conflict(self):
        with pytest.raises(ValueError, match="packing"):
            _load(_stream_yaml(training={"packing": True}))

    def test_multipack_conflict(self):
        with pytest.raises(ValueError, match="multipack"):
            _load(_stream_yaml(training={"multipack": True}))

    def test_fsdp2_compile_conflict(self):
        with pytest.raises(ValueError, match="use_fsdp2_compile"):
            _load(_stream_yaml(training={"use_fsdp2_compile": True}))

    def test_train_router_only_conflict(self):
        """moe_lora is set so the PRE-EXISTING train_router_only validator is
        satisfied — only the streaming gate can reject this, and the message
        must say so (otherwise the test passes for the wrong reason)."""
        with pytest.raises(ValueError, match="stream_layers"):
            _load(_stream_yaml(training={"train_router_only": True, "moe_lora": True}))

    def test_expand_layers_conflict(self):
        """Likewise: freeze_trainable_layers satisfies the LLaMA-Pro validator."""
        with pytest.raises(ValueError, match="stream_layers"):
            _load(
                _stream_yaml(
                    training={"expand_layers": 2, "freeze_trainable_layers": 2}
                )
            )


class TestStreamFootgunRejection:
    def test_stream_buffers_without_stream_layers_rejected(self):
        with pytest.raises(ValueError, match="stream_layers"):
            _load(_stream_yaml(training={"stream_layers": False, "stream_buffers": 4}))

    def test_stream_source_without_stream_layers_rejected(self):
        with pytest.raises(ValueError, match="stream_layers"):
            _load(_stream_yaml(training={"stream_layers": False, "stream_source": "ram"}))


class TestStreamBufferBounds:
    def test_one_buffer_rejected(self):
        with pytest.raises(ValueError, match="stream_buffers"):
            _load(_stream_yaml(training={"stream_buffers": 1}))

    def test_nine_buffers_rejected(self):
        with pytest.raises(ValueError, match="stream_buffers"):
            _load(_stream_yaml(training={"stream_buffers": 9}))

    def test_bool_buffers_rejected(self):
        with pytest.raises(ValueError, match="bool"):
            _load(_stream_yaml(training={"stream_buffers": True}))

    def test_bad_stream_source_rejected(self):
        with pytest.raises(ValueError, match="stream_source"):
            _load(_stream_yaml(training={"stream_source": "network"}))


# ==========================================================================
# C4 — the streaming runtime (utils/layer_stream_runtime.py)
#
# The four classes below are SILENT failures: if any regresses, training still
# runs and still converges. That is exactly why they are tests, not comments.
# ==========================================================================
def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_is_the_accelerator():
    """True on an Apple-Silicon runner with no CUDA.

    transformers picks `mps` as its default device there, while this suite
    builds the streamed model on `cpu` — the two then disagree and any real
    training step raises "found at least two devices". v0.72.0 measured CUDA
    and CPU only; MPS is untested, so the step test is skipped rather than
    making an unverified claim about it.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return False
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def _tiny_llama_dir(tmp_path, n_layers=2, tie=True):
    """A real (tiny) Llama checkpoint on disk: config.json + model.safetensors."""
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights), model, config


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed_cpu(tmp_path, n_layers=2, tie=True, buffers=2):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights, resident, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers, tie=tie)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index,
        lora_config=_tiny_lora(), device="cpu", dtype="float32",
        buffers=buffers, pin=False, seed=3,
    )
    return model, runtime, resident, weights


class TestDecoderOwner:
    def test_finds_the_module_that_owns_layers(self, tmp_path):
        from peft import get_peft_model

        from soup_cli.utils.layer_stream_runtime import decoder_owner

        _, resident, _ = _tiny_llama_dir(tmp_path)
        peft_model = get_peft_model(resident, _tiny_lora())
        owner = decoder_owner(peft_model)
        assert hasattr(owner, "layers")
        assert len(owner.layers) == 2

    def test_raises_when_no_layer_container(self):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import decoder_owner

        with pytest.raises(ValueError, match="layers"):
            decoder_owner(nn.Linear(2, 2))


class TestRegressionPrefetchHookAttachPoint:
    """SILENT FAILURE 1 — PEFT's LoraModel calls ``self.model.forward(...)``
    DIRECTLY, which bypasses __call__ and therefore every forward hook on the
    CausalLM wrapper. Attaching the prefetch there means layer 0 is never
    primed; the buffer-ownership assert then fires (or, worse, a stale buffer
    is used). The hook must go on the module that owns ``.layers``."""

    def test_prefetch_primes_through_a_real_peft_wrapper(self, tmp_path):
        import torch

        model, runtime, _, _ = _build_streamed_cpu(tmp_path)
        assert runtime.prefetcher.primes == 0
        model(input_ids=torch.randint(0, 64, (1, 8)))
        assert runtime.prefetcher.primes >= 1, (
            "prefetch never primed — the hook was attached to a module PEFT "
            "bypasses with a direct .forward() call"
        )

    def test_hook_is_registered_on_the_layer_owner(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import decoder_owner

        model, runtime, _, _ = _build_streamed_cpu(tmp_path)
        owner = decoder_owner(model)
        assert len(owner._forward_pre_hooks) >= 1


class TestRegressionAttributeTransparency:
    """SILENT FAILURE 2 — transformers reads contract attributes straight off
    the layer object (this version reads ``decoder_layer.attention_type`` in
    Qwen2Model.forward). A wrapper that is not attribute-transparent raises
    AttributeError at forward time; worse, a wrapper that returns a DEFAULT
    would silently select the wrong attention path."""

    def test_wrapper_forwards_unknown_attributes_to_the_wrapped_layer(self, tmp_path):
        model, _, _, _ = _build_streamed_cpu(tmp_path)
        from soup_cli.utils.layer_stream_runtime import decoder_owner

        layer = decoder_owner(model).layers[0]
        assert layer.__class__.__name__ == "StreamedDecoderLayer"
        # a real attribute of the wrapped LlamaDecoderLayer
        assert layer.self_attn is layer.inner.self_attn

    def test_attention_type_is_visible_when_the_inner_layer_has_it(self, tmp_path):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        class _Inner(nn.Module):
            attention_type = "full_attention"

            def forward(self, x, **kw):
                return x

        wrapper = StreamedDecoderLayer(_Inner(), 0, pool=None, prefetcher=None)
        assert wrapper.attention_type == "full_attention"

    def test_genuinely_missing_attribute_still_raises(self, tmp_path):
        import torch.nn as nn

        from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

        class _Inner(nn.Module):
            def forward(self, x, **kw):
                return x

        wrapper = StreamedDecoderLayer(_Inner(), 0, pool=None, prefetcher=None)
        with pytest.raises(AttributeError):
            wrapper.definitely_not_a_real_attribute


class TestRegressionMetaAdapterMaterialisation:
    """SILENT FAILURE 3 — PEFT initialises adapter weights on the BASE layer's
    device, which is ``meta`` in a streaming build. Meta parameters have no
    storage: the optimizer sees them, the run proceeds, and nothing trains."""

    def test_no_adapter_parameter_is_left_on_meta(self, tmp_path):
        model, _, _, _ = _build_streamed_cpu(tmp_path)
        stranded = [
            name for name, param in model.named_parameters()
            if "lora_" in name and param.is_meta
        ]
        assert stranded == [], stranded

    def test_adapters_are_trainable_and_base_is_not(self, tmp_path):
        model, _, _, _ = _build_streamed_cpu(tmp_path)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable, "nothing is trainable"
        assert all("lora_" in n for n in trainable), trainable

    def test_lora_b_is_zero_and_lora_a_is_not(self, tmp_path):
        """PEFT's init scheme. B=0 makes the adapter a no-op at step 0; an
        all-zero A would instead freeze the adapter forever."""
        model, _, _, _ = _build_streamed_cpu(tmp_path)
        a_seen = b_seen = False
        for name, param in model.named_parameters():
            if "lora_A" in name:
                a_seen = True
                assert param.abs().sum().item() > 0, name
            if "lora_B" in name:
                b_seen = True
                assert param.abs().sum().item() == 0, name
        assert a_seen and b_seen

    def test_base_decoder_weights_never_materialise(self, tmp_path):
        """The whole point: the resident load must not happen."""
        model, runtime, _, _ = _build_streamed_cpu(tmp_path)
        meta_layer_params = [
            name for name, param in model.named_parameters()
            if param.is_meta and ".layers." in name
        ]
        assert meta_layer_params, "decoder weights were materialised — not streaming"
        assert runtime.n_layers == 2


class TestRegressionAllocateOnceThenCopy:
    """SILENT FAILURE 4 — the obvious ``load_file -> .to(dtype) -> .pin_memory()``
    costs three transient copies of every layer. On the dev box that transient,
    not the store, is what pushed a 5.55 GB base over the 7.12 GB page-locked
    ceiling and made a 3B run impossible. The store must be allocated once and
    filled with copy_."""

    def test_ram_source_never_calls_tensor_pin_memory(self, tmp_path, monkeypatch):
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")

        calls = []
        monkeypatch.setattr(
            torch.Tensor, "pin_memory",
            lambda self, *a, **k: calls.append(1) or self,
        )
        spec = RamSource.spec_from_shard(shards)
        source = RamSource(shards, index.n_layers, spec, pin=False)
        assert source.nbytes > 0
        assert calls == [], "RamSource used .pin_memory() instead of allocating once"

    def test_store_matches_the_shard_bytes(self, tmp_path):
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = RamSource.spec_from_shard(shards)
        source = RamSource(shards, index.n_layers, spec, pin=False)
        blob = load_file(layer_shard_path(shards, 1))
        for name in spec:
            assert torch.equal(source.get(1, name), blob[name]), name


class TestBufferPool:
    def test_slot_round_robin(self):
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool

        pool = LayerBufferPool({"w": ((2, 2), "float32")}, n_buffers=2, device="cpu")
        assert [pool.slot_for(i) for i in range(5)] == [0, 1, 0, 1, 0]

    def test_wait_on_an_unloaded_slot_is_the_p1_tripwire(self):
        """plan P1: a buffer overwritten early yields silently wrong gradients,
        not a crash. The ownership assert is what makes it loud."""
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool

        pool = LayerBufferPool({"w": ((2, 2), "float32")}, n_buffers=2, device="cpu")
        with pytest.raises(RuntimeError, match="scheduler"):
            pool.wait(0)

    def test_wait_after_load_returns_the_buffer(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, RamSource

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = RamSource.spec_from_shard(shards)
        source = RamSource(shards, index.n_layers, spec, pin=False)
        pool = LayerBufferPool(spec, n_buffers=2, device="cpu")
        pool.load_async(0, source)
        got = pool.wait(0)
        assert set(got) == set(spec)

    def test_early_overwrite_is_detected(self, tmp_path):
        """Layer 2 recycles slot 0 (2 % 2 == 0) while layer 0 still owns it."""
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import LayerBufferPool, RamSource

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=3)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = RamSource.spec_from_shard(shards)
        source = RamSource(shards, index.n_layers, spec, pin=False)
        pool = LayerBufferPool(spec, n_buffers=2, device="cpu")
        pool.load_async(0, source)
        pool.load_async(2, source)  # slot 0 again — layer 0 is gone
        with pytest.raises(RuntimeError, match="scheduler"):
            pool.wait(0)


class TestPrefetcherDirection:
    """Forward walks 0..L-1; backward recompute walks L-1..0. The prefetcher
    must reverse with it or every backward layer is a buffer miss."""

    class _FakePool:
        def __init__(self, n=2):
            self.n = n
            self.owner = [None] * n
            self.loaded = []

        def slot_for(self, idx):
            return idx % self.n

        def load_async(self, idx, source, stream=None):
            self.owner[self.slot_for(idx)] = idx
            self.loaded.append(idx)

    def test_forward_prefetches_ascending(self):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        pool = self._FakePool()
        pre = StreamPrefetcher(pool, source=None, n_layers=4)
        pre.prime()
        for idx in range(4):
            pre.advance(idx)
        assert pool.loaded == [0, 1, 2, 3]

    def test_backward_prefetches_descending(self):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        pool = self._FakePool()
        pre = StreamPrefetcher(pool, source=None, n_layers=4)
        pre.prime()
        for idx in range(4):
            pre.advance(idx)
        for idx in (3, 2, 1, 0):  # backward recompute order
            pre.advance(idx)
        assert pool.loaded[-2:] == [1, 0]

    def test_does_not_reload_a_layer_already_owned(self):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        pool = self._FakePool()
        pre = StreamPrefetcher(pool, source=None, n_layers=4)
        pre.prime()
        pre.advance(0)
        before = list(pool.loaded)
        pre.advance(0)
        assert pool.loaded == before, "re-prefetched a layer the pool already owns"


class TestExpandableSegments:
    """User requirement 4: the brief says to auto-set
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. It is silently IGNORED on
    Windows (torch warns 'not supported on this platform'). Attempt, detect,
    and never claim it is active when it is not."""

    def test_returns_a_bool(self):
        from soup_cli.utils.layer_stream_runtime import probe_expandable_segments

        assert isinstance(probe_expandable_segments(), bool)

    def test_reports_false_on_windows(self):
        import sys

        from soup_cli.utils.layer_stream_runtime import probe_expandable_segments

        if sys.platform.startswith("win"):
            assert probe_expandable_segments() is False


class TestStreamedForwardParityCpu:
    """The Gate-1 property, as a CI test: streaming substitutes the identical
    weight bytes into the identical kernels, so the logits must be bit-exact."""

    def test_streamed_logits_match_resident_bit_exactly(self, tmp_path):
        import torch
        from peft import get_peft_model

        model, _, resident, _ = _build_streamed_cpu(tmp_path)
        ref = get_peft_model(resident, _tiny_lora())
        _copy_lora(model, ref)

        ids = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            got = model(input_ids=ids).logits
            want = ref(input_ids=ids).logits
        assert torch.equal(got, want), (got - want).abs().max().item()

    def test_layer_zero_adapter_gradient_is_non_zero(self, tmp_path):
        """plan P2: a detach()/no_grad() around the base severs the graph.
        Loss still falls (upper layers learn), so only this catches it."""
        import torch

        model, _, _, _ = _build_streamed_cpu(tmp_path, n_layers=3)
        ids = torch.randint(0, 64, (1, 12))
        model(input_ids=ids, labels=ids).loss.backward()
        grads = {
            name: param.grad
            for name, param in model.named_parameters()
            if "lora_" in name and param.grad is not None
        }
        layer0 = [g for n, g in grads.items() if ".layers.0." in n]
        assert layer0, "layer 0 has no adapter gradient at all"
        assert sum(float(g.abs().sum()) for g in layer0) > 0.0

    def test_buffer_count_does_not_change_the_result(self, tmp_path):
        import torch

        two, _, _, _ = _build_streamed_cpu(tmp_path / "a", n_layers=3, buffers=2)
        three, _, _, _ = _build_streamed_cpu(tmp_path / "b", n_layers=3, buffers=3)
        ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            assert torch.equal(two(input_ids=ids).logits, three(input_ids=ids).logits)


def _copy_lora(src, dst):
    """StreamedDecoderLayer inserts an `.inner.` segment into state-dict keys."""
    def norm(key):
        return key.replace(".inner.", ".")

    src_lora = {norm(k): v for k, v in src.state_dict().items() if "lora_" in k}
    dst_lora = {norm(k): v for k, v in dst.state_dict().items() if "lora_" in k}
    assert src_lora and set(src_lora) == set(dst_lora)
    for key, val in src_lora.items():
        dst_lora[key].copy_(val)


CUDA = pytest.mark.skipif(
    not _cuda_available(), reason="requires CUDA (layer streaming is a GPU feature)"
)


@CUDA
class TestStreamedTrainingParityCuda:
    def test_five_steps_match_resident(self, tmp_path):
        import torch
        from peft import get_peft_model

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, resident, _ = _tiny_llama_dir(tmp_path, n_layers=3)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        streamed, _ = build_streamed_model(
            model_id=weights, shard_dir=shards, index=index,
            lora_config=_tiny_lora(), device="cuda", dtype="float32",
            buffers=2, pin=True, seed=3,
        )
        ref = get_peft_model(resident, _tiny_lora()).to("cuda")
        _copy_lora(streamed, ref)

        torch.manual_seed(0)
        batches = [torch.randint(0, 64, (1, 12), device="cuda") for _ in range(5)]
        losses = []
        for model in (streamed, ref):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            run = []
            for ids in batches:
                out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                run.append(float(out.loss))
            losses.append(run)
        for got, want in zip(*losses):
            assert abs(got - want) < 1e-4, losses

    def test_same_seed_twice_is_identical(self, tmp_path):
        """Catches stream races: a racy prefetch is non-deterministic."""
        import torch

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=3)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        runs = []
        for _ in range(2):
            model, _ = build_streamed_model(
                model_id=weights, shard_dir=shards, index=index,
                lora_config=_tiny_lora(), device="cuda", dtype="float32",
                buffers=2, pin=True, seed=3,
            )
            torch.manual_seed(0)
            batches = [torch.randint(0, 64, (1, 12), device="cuda") for _ in range(4)]
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3
            )
            run = []
            for ids in batches:
                out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                run.append(float(out.loss))
            runs.append(run)
            del model
            torch.cuda.empty_cache()
        assert runs[0] == runs[1], runs


class TestRuntimeNoTopLevelTorch:
    def test_runtime_is_torch_lazy(self):
        import soup_cli.utils.layer_stream_runtime as mod

        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("torch", "peft", "transformers"))
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("torch", "peft", "transformers")
                )


# ==========================================================================
# C5 — trainer/sft.py streaming branch
# ==========================================================================
class _StubTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, *a, **k):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _stream_config(base_dir, **training):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    cfg = {
        "base": base_dir,
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {"train": "train.jsonl"},
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "quantization": "none",
            "stream_layers": True,
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
            **training,
        },
    }
    return load_config_from_string(yaml.safe_dump(cfg))


class TestHfGradientCheckpointingInteraction:
    """StreamedDecoderLayer already wraps every layer in checkpoint(). Letting
    HF ALSO enable gradient checkpointing double-recomputes every layer: the
    run still converges, it is just silently ~1.5x slower."""

    def test_streaming_disables_hf_gradient_checkpointing(self):
        from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

        assert should_enable_hf_gradient_checkpointing(True, stream_layers=True) is False

    def test_non_streaming_keeps_hf_gradient_checkpointing(self):
        from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

        assert should_enable_hf_gradient_checkpointing(True, stream_layers=False) is True

    def test_tier_string_also_disabled_under_streaming(self):
        from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

        assert should_enable_hf_gradient_checkpointing("auto", stream_layers=True) is False

    def test_off_stays_off(self):
        from soup_cli.utils.layer_stream import should_enable_hf_gradient_checkpointing

        assert should_enable_hf_gradient_checkpointing(False, stream_layers=False) is False

    def test_sft_guards_the_hf_flag_with_the_helper(self):
        """Pin the call site: the trainer must consult the helper, not the raw
        tcfg.gradient_checkpointing."""
        import inspect

        from soup_cli.trainer import sft

        source = inspect.getsource(sft)
        assert "should_enable_hf_gradient_checkpointing" in source


class TestStreamingDispatch:
    def test_setup_routes_to_the_streaming_branch(self, tmp_path, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        weights, _, _ = _tiny_llama_dir(tmp_path)
        cfg = _stream_config(weights)

        calls = []

        class _StopError(Exception):
            pass

        def _stream(self, c, t):
            calls.append("streaming")
            raise _StopError()

        def _plain(self, c, t):
            calls.append("plain")
            raise _StopError()

        monkeypatch.setattr(SFTTrainerWrapper, "_setup_streaming_transformers", _stream)
        monkeypatch.setattr(SFTTrainerWrapper, "_setup_transformers", _plain)
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        with pytest.raises(_StopError):
            wrapper.setup({"train": [{"messages": []}]})
        assert calls == ["streaming"]

    def test_setup_uses_the_plain_branch_when_streaming_is_off(self, tmp_path, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        weights, _, _ = _tiny_llama_dir(tmp_path)
        cfg = _stream_config(weights, stream_layers=False)

        calls = []

        class _StopError(Exception):
            pass

        def _stream(self, c, t):
            calls.append("streaming")
            raise _StopError()

        def _plain(self, c, t):
            calls.append("plain")
            raise _StopError()

        monkeypatch.setattr(SFTTrainerWrapper, "_setup_streaming_transformers", _stream)
        monkeypatch.setattr(SFTTrainerWrapper, "_setup_transformers", _plain)
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        with pytest.raises(_StopError):
            wrapper.setup({"train": [{"messages": []}]})
        assert calls == ["plain"]


class TestStreamingSetupIntegration:
    """The real construction path on CPU: meta skeleton -> extras -> LoRA ->
    streaming, with NO resident base load at any point."""

    def _run(self, tmp_path, monkeypatch, n_layers=2):
        import transformers

        from soup_cli.trainer.sft import SFTTrainerWrapper

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained",
            classmethod(lambda cls, *a, **k: _StubTokenizer()),
        )
        cfg = _stream_config(weights)
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper._setup_streaming_transformers(cfg, cfg.training)
        return wrapper

    def test_decoder_weights_stay_on_meta(self, tmp_path, monkeypatch):
        wrapper = self._run(tmp_path, monkeypatch)
        meta_layer = [
            n for n, p in wrapper.model.named_parameters()
            if p.is_meta and ".layers." in n
        ]
        assert meta_layer, "the base was materialised — that defeats streaming"

    def test_only_adapters_are_trainable(self, tmp_path, monkeypatch):
        wrapper = self._run(tmp_path, monkeypatch)
        trainable = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n for n in trainable)

    def test_runtime_is_attached_with_stats(self, tmp_path, monkeypatch):
        wrapper = self._run(tmp_path, monkeypatch)
        stats = wrapper._stream_runtime.stats()
        assert stats["n_layers"] == 2
        assert stats["buffers"] == 2
        assert stats["store_bytes"] > 0

    def test_a_forward_pass_runs(self, tmp_path, monkeypatch):
        import torch

        wrapper = self._run(tmp_path, monkeypatch, n_layers=3)
        out = wrapper.model(input_ids=torch.randint(0, 64, (1, 8)))
        assert out.logits.shape == (1, 8, 64)

    def test_unsupported_architecture_is_refused(self, tmp_path, monkeypatch):
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import GPT2Config, GPT2LMHeadModel

        from soup_cli.trainer.sft import SFTTrainerWrapper

        config = GPT2Config(n_layer=2, n_embd=32, n_head=4, vocab_size=64)
        model = GPT2LMHeadModel(config)
        weights = tmp_path / "gpt2"
        weights.mkdir()
        state = {
            k: v.contiguous()
            for k, v in model.state_dict().items()
            if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")
        }
        state.pop("lm_head.weight", None)  # tied to wte; safetensors refuses aliases
        save_file(state, str(weights / "model.safetensors"))
        config.save_pretrained(str(weights))
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained",
            classmethod(lambda cls, *a, **k: _StubTokenizer()),
        )
        cfg = _stream_config(str(weights))
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        with pytest.raises(ValueError, match="gpt2"):
            wrapper._setup_streaming_transformers(cfg, cfg.training)
        assert torch  # keep the import meaningful


class TestStreamingResumeAccepted:
    """v0.72.0-.2 refused --resume: a streamed model's named_parameters() carry
    an `.inner.` segment that load_state_dict narrows away, so a canonical
    checkpoint matched NOTHING and training silently continued from scratch.
    v0.72.3 redirects canonical keys at load time; the refusal is gone and the
    round-trip is pinned in test_v07203.py."""

    def test_train_accepts_resume_with_streaming(self, tmp_path, monkeypatch):
        import json as _json

        import yaml
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        }
        with open("train.jsonl", "w", encoding="utf-8") as fh:
            fh.write(_json.dumps(row) + "\n")
        cfg = {
            "base": "HuggingFaceTB/SmolLM2-135M",
            "task": "sft",
            "backend": "transformers",
            "modality": "text",
            "data": {"train": "train.jsonl"},
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "quantization": "none",
                "stream_layers": True,
            },
        }
        with open("soup.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh)
        result = CliRunner().invoke(
            app, ["train", "--config", "soup.yaml", "--resume", "latest", "--yes"]
        )
        assert "are not supported with" not in result.output, result.output
        assert "lands in v0.72.3" not in result.output, result.output


class TestRegressionTrainerConstructsWithoutMovingMetaWeights:
    """SILENT-UNTIL-FIRST-RUN FAILURE — transformers' Trainer.__init__ calls
    _move_model_to_device -> model.to(args.device), and .to() on a module that
    still holds meta parameters raises NotImplementedError. Streaming keeps the
    decoder weights on meta ON PURPOSE, so without a device-map marker EVERY
    streaming run dies at trainer construction, right after printing
    'Layer streaming ready'. Unit tests that stop at model(input_ids=...) do not
    see it — only building a real Trainer does."""

    def test_to_device_is_safe_and_leaves_the_base_on_meta(self, tmp_path):
        """A bare nn.Module holding meta params raises on .to(); the streamed
        layer must pass meta tensors through so Trainer/accelerate can move the
        model, while the base placeholders STAY meta (or streaming is over)."""
        model, _, _, _ = _build_streamed_cpu(tmp_path)
        model.to("cpu")  # must not raise
        still_meta = [
            n for n, p in model.named_parameters() if p.is_meta and ".layers." in n
        ]
        assert still_meta, "the base was materialised by .to() — streaming defeated"
        assert all(
            not p.is_meta for n, p in model.named_parameters() if "lora_" in n
        ), "adapters must remain real after a device move"

    def test_streamed_model_declares_a_device_map(self, tmp_path):
        model, runtime, _, _ = _build_streamed_cpu(tmp_path)
        assert getattr(model, "hf_device_map", None) is not None, (
            "no hf_device_map — Trainer will try to .to() the meta weights"
        )
        assert runtime.device in str(model.hf_device_map)

    def test_real_trainer_constructs_from_a_streamed_model(self, tmp_path):
        from transformers import Trainer, TrainingArguments

        model, _, _, _ = _build_streamed_cpu(tmp_path)
        args = TrainingArguments(
            output_dir=str(tmp_path / "out"),
            per_device_train_batch_size=1,
            report_to=[],
            use_cpu=True,
        )
        Trainer(model=model, args=args)  # must not raise


class TestStreamLoraVariantGates:
    """DoRA / VeRA / PiSSA / OLoRA all READ the real base weight at
    get_peft_model() time to initialise. Under streaming the base is still on
    meta then, so they crash with an opaque torch error ("Fan in and fan out
    can not be computed...") instead of a refusal naming the incompatibility."""

    def test_dora_conflict(self):
        with pytest.raises(ValueError, match="use_dora"):
            _load(_stream_yaml(training={"lora": {"r": 4, "use_dora": True}}))

    def test_rslora_is_allowed(self):
        """rsLoRA only rescales alpha — it never reads the base weight."""
        cfg = _load(_stream_yaml(training={"lora": {"r": 4, "use_rslora": True}}))
        assert cfg.training.lora.use_rslora is True

    def test_pissa_init_conflict(self):
        with pytest.raises(ValueError, match="init_strategy"):
            _load(_stream_yaml(training={"lora": {"r": 4, "init_strategy": "pissa"}}))

    def test_olora_init_conflict(self):
        with pytest.raises(ValueError, match="init_strategy"):
            _load(_stream_yaml(training={"lora": {"r": 4, "init_strategy": "olora"}}))


class TestPrefetchDirectionIsExplicit:
    """Direction was inferred from call order, which is correct today only
    because the turnaround index happens to be the last layer. Make it explicit
    so a future lookahead change cannot silently turn a missed prefetch into a
    stall or a hazard."""

    def test_turnaround_resets_state(self):
        from soup_cli.utils.layer_stream_runtime import StreamPrefetcher

        class _Pool:
            n = 2

            def __init__(self):
                self.owner = [None, None]
                self.loaded = []

            def slot_for(self, idx):
                return idx % self.n

            def load_async(self, idx, source, stream=None):
                self.owner[self.slot_for(idx)] = idx
                self.loaded.append(idx)

        pool = _Pool()
        pre = StreamPrefetcher(pool, source=None, n_layers=4)
        pre.prime()
        assert pre.direction == 1
        for idx in range(4):
            pre.advance(idx)
        pre.advance(3)  # backward recompute starts at the SAME index
        pre.advance(2)
        assert pre.direction == -1
        pre.prime()  # next step's forward
        assert pre.direction == 1


class TestShardCacheIdentityBinding:
    """SILENT FAILURE — the shard cache is keyed by a model SLUG. If the source
    checkpoint changes (a local dir retrained in place) or two ids collide onto
    one slug, a dtype-only cache check happily reuses stale shards and streams
    the WRONG WEIGHTS into training. Nothing errors; the loss curve just
    describes a different model."""

    def test_index_records_a_source_fingerprint(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        index = shard_checkpoint(src, str(tmp_path / "shards"), dtype="float32")
        assert index.source_fingerprint

    def test_changed_source_weights_invalidate_the_cache(self, tmp_path):
        import torch
        from safetensors.torch import load_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, layer, extras = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")

        # retrain in place: same filenames, different weights
        changed = {k: torch.full_like(v, 0.5) for k, v in layer.items()}
        _write_safetensors(
            str(tmp_path / "weights" / "model.safetensors"), {**changed, **extras}
        )
        index = shard_checkpoint(src, out, dtype="float32")
        blob = load_file(layer_shard_path(out, 0))
        assert torch.allclose(blob["mlp.down_proj.weight"], torch.tensor(0.5)), (
            "stale shards reused after the source checkpoint changed"
        )
        assert index.source_fingerprint

    def test_identical_source_still_hits_the_cache(self, tmp_path):
        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        before = os.path.getmtime(layer_shard_path(out, 0))
        shard_checkpoint(src, out, dtype="float32")
        assert os.path.getmtime(layer_shard_path(out, 0)) == before


class TestShardWriteContainment:
    def test_out_dir_outside_the_allowed_roots_is_refused(self, tmp_path):
        """shard_checkpoint must bound its OWN writes, not rely on its caller."""
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        outside = (
            os.path.join("C:" + os.sep, "Windows", "System32", "soup-evil")
            if os.name == "nt"
            else "/etc/soup-evil"
        )
        with pytest.raises(ValueError, match="under"):
            shard_checkpoint(src, outside, dtype="float32")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_symlinked_ancestor_is_resolved_not_followed_blindly(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        os.symlink(str(real), str(link))
        index = shard_checkpoint(src, str(link / "shards"), dtype="float32")
        # resolved through the symlink to its real location, and recorded there
        assert (real / "shards" / "index.json").exists()
        assert index.n_layers == 3

    def test_total_tensor_count_is_capped(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_shard as mod

        src, _, _ = _fake_weights_dir(tmp_path)
        monkeypatch.setattr(mod, "_MAX_TOTAL_TENSORS", 3)
        with pytest.raises(ValueError, match="tensors"):
            mod.shard_checkpoint(src, str(tmp_path / "out"), dtype="float32")


class TestSourceSizeProbe:
    """Refuse an obviously-too-big base BEFORE spending minutes sharding it."""

    def test_reports_the_source_byte_size(self, tmp_path):
        from soup_cli.utils.layer_shard import source_weight_bytes

        src, _, _ = _fake_weights_dir(tmp_path)
        assert source_weight_bytes(src) > 0

    def test_missing_dir_raises(self, tmp_path):
        from soup_cli.utils.layer_shard import source_weight_bytes

        with pytest.raises(FileNotFoundError):
            source_weight_bytes(str(tmp_path / "nope"))


# ==========================================================================
# Review-round hardening (ECC round 4 — test-quality audit)
# ==========================================================================
def _write_tiny_tokenizer(directory):
    """A real, offline PreTrainedTokenizerFast so `setup()` can tokenize.

    The end-to-end streaming test must not depend on the HF cache or network.
    """
    import json as _json

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "yo", "the", "cat", "sat", "on", "mat"):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        _json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            fh,
        )


class TestStreamingEndToEndSetup:
    """THE integration gap. Unit tests stop at `model(input_ids=...)`; the real
    path is `wrapper.setup(dataset)` -> tokenize -> TRL SFTTrainer -> train().
    A crash-on-every-run bug already hid in exactly this gap once."""

    def _wrapper(self, tmp_path, monkeypatch):
        import yaml

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=2)
        _write_tiny_tokenizer(weights)
        monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            yaml.safe_dump(
                {
                    "base": weights,
                    "task": "sft",
                    "backend": "transformers",
                    "modality": "text",
                    "data": {
                        "train": "train.jsonl",
                        "max_length": 64,
                        "chat_template": "chatml",
                    },
                    "training": {
                        "batch_size": 1,
                        "gradient_accumulation_steps": 1,
                        "quantization": "none",
                        "stream_layers": True,
                        "epochs": 1,
                        "logging_steps": 1,
                        "save_steps": 1000,
                        "gradient_checkpointing": True,
                        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                    },
                    "output": str(tmp_path / "out"),
                }
            )
        )
        dataset = {
            "train": [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello world"},
                    ]
                }
                for _ in range(4)
            ]
        }
        # Use the REAL device: TrainingArguments picks cuda when it is
        # available, so forcing the model to cpu here would only produce a
        # device mismatch that no user would ever hit.
        device = "cuda" if _cuda_available() else "cpu"
        return SFTTrainerWrapper(cfg, device=device), dataset

    def test_setup_builds_a_real_trl_trainer(self, tmp_path, monkeypatch):
        wrapper, dataset = self._wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)
        assert wrapper.trainer is not None
        assert wrapper.model is not None

    def test_hf_gradient_checkpointing_is_off_under_streaming(self, tmp_path, monkeypatch):
        """The behavioural version of the helper test: streaming checkpoints
        per layer itself, so TrainingArguments must NOT also enable it."""
        wrapper, dataset = self._wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)
        assert wrapper.trainer.args.gradient_checkpointing is False

    @pytest.mark.skipif(
        _mps_is_the_accelerator(),
        reason="MPS is untested in v0.72.0 (measured on CUDA + CPU only)",
    )
    def test_one_training_step_actually_runs(self, tmp_path, monkeypatch):
        """Forward + backward + optimizer step through the streamed layers."""
        wrapper, dataset = self._wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)
        wrapper.trainer.args.max_steps = 1
        wrapper.trainer.train()
        loads = wrapper._stream_runtime.pool.loads
        assert loads > 0, "no layer was ever streamed during training"


class TestInstallStreamingGuards:
    def test_layer_count_mismatch_is_refused(self, tmp_path):
        import dataclasses

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=2)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        lying = dataclasses.replace(index, n_layers=5)
        with pytest.raises(ValueError, match="decoder layers"):
            build_streamed_model(
                model_id=weights, shard_dir=shards, index=lying,
                lora_config=_tiny_lora(), device="cpu", dtype="float32",
                buffers=2, pin=False, seed=1,
            )

    def test_shard_missing_a_decoder_weight_is_refused(self, tmp_path):
        from safetensors.torch import load_file, save_file

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, _, _ = _tiny_llama_dir(tmp_path, n_layers=2)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
        import gc

        for idx in range(index.n_layers):
            blob = {k: v.clone() for k, v in load_file(layer_shard_path(shards, idx)).items()}
            gc.collect()  # Windows refuses to rewrite a still-mmapped file (err 1224)
            blob.pop("self_attn.q_proj.weight")
            save_file(blob, layer_shard_path(shards, idx))
        with pytest.raises(ValueError, match="missing decoder weights"):
            build_streamed_model(
                model_id=weights, shard_dir=shards, index=index,
                lora_config=_tiny_lora(), device="cpu", dtype="float32",
                buffers=2, pin=False, seed=1,
            )


class TestPinnedFallbackRuntime:
    """decide_pinning() is byte math; this is the RUNTIME retry that actually
    catches a failed page-lock and downgrades to a pageable store."""

    def test_pin_failure_falls_back_and_warns(self, tmp_path, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as rt
        from soup_cli.utils.layer_shard import shard_checkpoint

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        spec = rt.RamSource.spec_from_shard(shards)

        real = rt.RamSource
        attempts = []

        class _FailsWhenPinned(real):
            def __init__(self, shard_dir, n_layers, spec, *, pin=True):
                attempts.append(pin)
                if pin:
                    raise RuntimeError("CUDA error: out of memory")
                super().__init__(shard_dir, n_layers, spec, pin=False)

        monkeypatch.setattr(rt, "RamSource", _FailsWhenPinned)
        printed = []

        class _Console:
            def print(self, msg):
                printed.append(str(msg))

        source, pinned = rt._build_source(shards, index.n_layers, spec, True, _Console())
        assert attempts == [True, False]
        assert pinned is False
        assert source.nbytes > 0
        assert any("pageable" in msg.lower() for msg in printed)
        assert any("utilisation" in msg.lower() for msg in printed)


class TestCachedIndexInvalidation:
    def _prepare(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        src, _, _ = _fake_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        shard_checkpoint(src, out, dtype="float32")
        return src, out

    def test_missing_extras_shard_reshards(self, tmp_path):
        from soup_cli.utils.layer_shard import extras_shard_path, shard_checkpoint

        src, out = self._prepare(tmp_path)
        os.unlink(extras_shard_path(out))
        shard_checkpoint(src, out, dtype="float32")
        assert os.path.exists(extras_shard_path(out))

    def test_missing_layer_shard_reshards(self, tmp_path):
        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint

        src, out = self._prepare(tmp_path)
        os.unlink(layer_shard_path(out, 1))
        shard_checkpoint(src, out, dtype="float32")
        assert os.path.exists(layer_shard_path(out, 1))

    def test_index_with_missing_keys_reshards(self, tmp_path):
        import json as _json

        from soup_cli.utils.layer_shard import shard_checkpoint

        src, out = self._prepare(tmp_path)
        with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as fh:
            _json.dump({"n_layers": 3}, fh)  # well-formed JSON, missing fields
        index = shard_checkpoint(src, out, dtype="float32")
        assert index.n_layers == 3
        assert index.layer_keys


class TestMaterializeExtrasGuard:
    def test_incomplete_extras_shard_is_refused(self, tmp_path):
        """A corrupt extras shard must fail loudly, not leave meta embeddings."""
        from safetensors.torch import load_file, save_file

        from soup_cli.utils.layer_shard import extras_shard_path, shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, materialize_extras

        weights, _, _ = _tiny_llama_dir(tmp_path)
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(weights, shards, dtype="float32")
        import gc

        blob = {k: v.clone() for k, v in load_file(extras_shard_path(shards)).items()}
        gc.collect()  # Windows refuses to rewrite a still-mmapped file (err 1224)
        blob.pop("model.embed_tokens.weight")
        save_file(blob, extras_shard_path(shards))

        model = build_meta_skeleton(weights, dtype="float32")
        with pytest.raises(RuntimeError, match="unmaterialised"):
            materialize_extras(model, shards, index, device="cpu", dtype="float32")


class TestHardwareFitGateIsStreamingAware:
    """The pre-flight VRAM gate models a RESIDENT run: full weights + optimizer
    + grads on the card. Under streaming, peak VRAM is bounded by ONE layer, so
    the resident prediction refuses precisely the runs layer streaming exists to
    enable (a 3B base on a 4 GB card predicts >4 GB and is blocked)."""

    def _cfg(self, stream: bool):
        import yaml

        from soup_cli.config.loader import load_config_from_string

        body = {
            "base": "Qwen/Qwen2.5-3B",
            "task": "sft",
            "backend": "transformers",
            "modality": "text",
            "data": {"train": "train.jsonl", "max_length": 512},
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "quantization": "none",
                "stream_layers": stream,
                "lora": {"r": 8, "alpha": 16},
            },
        }
        return load_config_from_string(yaml.safe_dump(body))

    def test_resident_run_is_still_gated(self):
        """Control: without streaming the gate must still fire on a 4 GB card,
        otherwise this test proves nothing about the streaming branch."""
        import typer

        from soup_cli.commands.train import _hardware_fit_preflight

        gpu = {"memory_total_bytes": 4 * 10**9}
        with pytest.raises(typer.Exit):
            _hardware_fit_preflight(self._cfg(False), gpu, allow_oom_attempt=False)

    def test_streaming_run_is_not_blocked_by_the_resident_prediction(self):
        from soup_cli.commands.train import _hardware_fit_preflight

        gpu = {"memory_total_bytes": 4 * 10**9}
        _hardware_fit_preflight(self._cfg(True), gpu, allow_oom_attempt=False)
