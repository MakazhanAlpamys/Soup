"""R4 — setup: the roots reach the sharder, the depth reaches every consumer."""

import ast
import inspect
import io
import json
import os
import struct
from types import SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.trainer import stream_setup
from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD, MAX_STREAM_READ_AHEAD


def _tcfg(value):
    return SimpleNamespace(stream_read_ahead=value)


def _console():
    buffer = io.StringIO()
    return Console(file=buffer, width=200, color_system=None), buffer


def _depth(value, n_roots):
    console, out = _console()
    decision = stream_setup._effective_read_ahead(_tcfg(value), n_roots, console)
    return decision.depth, out.getvalue()


class TestTheDepth:
    def test_one_root_leaves_the_setting_alone(self):
        assert _depth(2, 1)[0] == 2
        assert _depth(1, 1)[0] == 1

    def test_the_default_becomes_one_more_than_the_drives(self):
        depth, said = _depth(DEFAULT_STREAM_READ_AHEAD, 2)
        assert depth == 3 and "stream_read_ahead" in said
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 3)[0] == 4

    def test_a_value_already_deep_enough_is_kept(self):
        """Review Focus 5: never bump what the operator already set high enough."""
        assert _depth(5, 2) == (5, "")

    def test_a_shallower_value_is_kept_with_a_warning_naming_the_idle_drives(self):
        depth, said = _depth(1, 2)
        assert depth == 1 and "idle" in said and "3" in said

    def test_the_default_never_exceeds_the_schema_ceiling(self):
        """Review Focus 5: seven roots want 8, the most the setting allows."""
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 7)[0] == 8 == MAX_STREAM_READ_AHEAD

    def test_the_clamp_holds_past_the_most_roots_the_setting_can_serve(self):
        """Seven roots reach the ceiling exactly, so only a bigger count shows the clamp."""
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 9)[0] == MAX_STREAM_READ_AHEAD


def test_every_depth_consumer_in_the_setup_reads_the_effective_value():
    """The default N+1 is worth nothing if one consumer still reads the raw setting."""
    source = inspect.getsource(stream_setup)
    tree = ast.parse(source)
    raw_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "stream_read_ahead"
        and isinstance(node.value, ast.Name)
        and node.value.id == "tcfg"
    ]
    helper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_effective_read_ahead"
    )
    inside = range(helper.lineno, helper.end_lineno + 1)
    assert [line for line in raw_reads if line not in inside] == []


class TestTheDiskPreflight:
    def test_shares_split_the_estimate_across_roots_and_sum_to_it(self):
        assert stream_setup._stripe_write_shares(10, 1) == [10]
        shares = stream_setup._stripe_write_shares(10, 3)
        assert sum(shares) == 10 and max(shares) - min(shares) <= 1

    def test_a_full_stripe_volume_is_refused_by_name(self, tmp_path, monkeypatch):
        # `console` is the module-level Rich console: keep the panel off the real terminal.
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        full = str(tmp_path / "stripe")
        monkeypatch.setattr(
            stream_setup, "_disk_volume",
            lambda path: (2, 0) if path == full else (1, 10**15),
        )
        with pytest.raises(ValueError, match="layer-shard stripe"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=1, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=str(tmp_path), shard_bytes=10, shard_write_bytes=5,
                shard_path=str(tmp_path / "cache"), stripe_writes=((full, 5),),
            )

    def test_a_full_stripe_volume_names_the_stripe_variable(self, tmp_path, monkeypatch):
        """Final review Minor 6: the remedy named only the primary cache variables."""
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        full = str(tmp_path / "stripe")
        monkeypatch.setattr(
            stream_setup, "_disk_volume",
            lambda path: (2, 0) if path == full else (1, 10**15),
        )
        with pytest.raises(ValueError, match="SOUP_LAYER_STREAM_STRIPE_DIRS"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=1, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=str(tmp_path), shard_bytes=10, shard_write_bytes=5,
                shard_path=str(tmp_path / "cache"), stripe_writes=((full, 5),),
            )

    def test_the_additional_writes_line_counts_every_stripe(self, tmp_path, monkeypatch):
        """A striped cache writes to several drives; the headline figure must add them up."""
        recorder = io.StringIO()
        monkeypatch.setattr(
            stream_setup, "console", Console(file=recorder, width=200, color_system=None)
        )
        monkeypatch.setattr(stream_setup, "_disk_volume", lambda path: (hash(path), 10**15))
        stream_setup._render_stream_disk_preflight(
            source_bytes=0, materialized_copy_bytes=0, materialize_bytes=0,
            materialized_path=str(tmp_path), shard_bytes=int(3e9), shard_write_bytes=int(1e9),
            shard_path=str(tmp_path / "cache"),
            stripe_writes=((str(tmp_path / "a"), int(1e9)), (str(tmp_path / "b"), int(1e9))),
        )
        assert "Additional writes before training: 3.00 GB" in recorder.getvalue()

    def test_exact_shares_charge_primary_drive_more_than_even_split_two_roots(self, tmp_path):
        """#1612: 5 layers over 2 roots puts 3 layers + non-decoder weights on root 0."""
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        file_size = os.path.getsize(os.path.join(weights, "model.safetensors"))
        shares, approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=2, dtype="bfloat16"
        )
        assert not approx
        assert sum(shares) == file_size
        even_shares = stream_setup._stripe_write_shares(file_size, 2)
        # Primary drive receives 3 of 5 layers plus embedding/head; must exceed even split
        assert shares[0] > even_shares[0]
        assert shares[1] < even_shares[1]

    def test_exact_shares_three_roots_odd_layer_count(self, tmp_path):
        """#1612: 5 layers over 3 roots gives (2, 2, 1) layer placement."""
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        file_size = os.path.getsize(os.path.join(weights, "model.safetensors"))
        shares, approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=3, dtype="bfloat16"
        )
        assert not approx
        assert sum(shares) == file_size
        even_shares = stream_setup._stripe_write_shares(file_size, 3)
        assert shares[0] > even_shares[0]
        # Root 0 and 1 each have 2 layers; root 0 also has non-decoder weights
        assert shares[0] > shares[1]
        # Root 2 has only 1 layer
        assert shares[1] > shares[2]

    def test_preflight_refusal_when_primary_volume_short_two_roots(self, tmp_path, monkeypatch):
        """#1612: pre-flight refuses when primary drive fits even split but not real bytes."""
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        file_size = os.path.getsize(os.path.join(weights, "model.safetensors"))
        shares, approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=2, dtype="bfloat16"
        )
        assert not approx
        even_shares = stream_setup._stripe_write_shares(file_size, 2)
        assert shares[0] > even_shares[0]

        primary_dir = str(tmp_path / "cache")
        stripe_dir = str(tmp_path / "stripe")
        # Primary volume has room for even share, but NOT for real share
        primary_free = even_shares[0] + 1
        monkeypatch.setattr(
            stream_setup,
            "_disk_volume",
            lambda path: (1, primary_free) if path == primary_dir else (2, 10**15),
        )

        # Real shares: refusal fires and names layer-shard cache
        with pytest.raises(ValueError, match="layer-shard cache"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=file_size, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=weights, shard_bytes=file_size, shard_write_bytes=shares[0],
                shard_path=primary_dir, stripe_writes=((stripe_dir, shares[1]),),
            )

        # Mutation check: with even split, pre-flight would have falsely passed
        recorder = io.StringIO()
        monkeypatch.setattr(
            stream_setup, "console", Console(file=recorder, width=200, color_system=None)
        )
        stream_setup._render_stream_disk_preflight(
            source_bytes=file_size, materialized_copy_bytes=0, materialize_bytes=0,
            materialized_path=weights, shard_bytes=file_size, shard_write_bytes=even_shares[0],
            shard_path=primary_dir, stripe_writes=((stripe_dir, even_shares[1]),),
        )
        assert "Layer-shard cache" in recorder.getvalue()

    def test_preflight_refusal_when_primary_volume_short_three_roots(self, tmp_path, monkeypatch):
        """#1612: 3-root case refuses when primary volume is short."""
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        file_size = os.path.getsize(os.path.join(weights, "model.safetensors"))
        shares, _approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=3, dtype="bfloat16"
        )
        even_shares = stream_setup._stripe_write_shares(file_size, 3)
        assert shares[0] > even_shares[0]

        primary_dir = str(tmp_path / "cache")
        stripe_a = str(tmp_path / "stripe_a")
        stripe_b = str(tmp_path / "stripe_b")
        primary_free = even_shares[0] + 1
        monkeypatch.setattr(
            stream_setup,
            "_disk_volume",
            lambda path: (1, primary_free) if path == primary_dir else (hash(path), 10**15),
        )

        with pytest.raises(ValueError, match="layer-shard cache"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=file_size, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=weights, shard_bytes=file_size, shard_write_bytes=shares[0],
                shard_path=primary_dir,
                stripe_writes=((stripe_a, shares[1]), (stripe_b, shares[2])),
            )

    def test_preflight_panel_line_approximate_when_unreadable(self, tmp_path, monkeypatch):
        """#1612: when headers cannot be read, even split is kept and panel says approximately."""
        recorder = io.StringIO()
        monkeypatch.setattr(
            stream_setup, "console", Console(file=recorder, width=200, color_system=None)
        )
        monkeypatch.setattr(stream_setup, "_disk_volume", lambda path: (hash(path), 10**15))

        shares, approx = stream_setup.estimate_stripe_write_shares(
            str(tmp_path / "nonexistent"), total_bytes=1000, n_roots=2
        )
        assert approx is True
        assert shares == stream_setup._stripe_write_shares(1000, 2)

        stream_setup._render_stream_disk_preflight(
            source_bytes=0, materialized_copy_bytes=0, materialize_bytes=0,
            materialized_path=str(tmp_path), shard_bytes=1000, shard_write_bytes=shares[0],
            shard_path=str(tmp_path / "cache"), stripe_writes=((str(tmp_path / "a"), shares[1]),),
            approximate=approx,
        )
        assert "Layer-shard cache: approximately 0.00 GB (write required)" in recorder.getvalue()

    def test_exact_shares_match_actual_sharded_files_on_disk(self, tmp_path):
        """#1612: compare charged shares with actual sharded bytes on disk."""
        pytest.importorskip("torch")
        pytest.importorskip("safetensors")
        pytest.importorskip("transformers")
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, LlamaConfig

        from soup_cli.utils.layer_shard import shard_checkpoint

        weights = tmp_path / "model"
        weights.mkdir(parents=True, exist_ok=True)
        config = LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=64,
            num_hidden_layers=5, num_attention_heads=4,
            num_key_value_heads=2, tie_word_embeddings=False,
            max_position_embeddings=128,
        )
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
        save_file(
            {k: v.contiguous() for k, v in model.state_dict().items()},
            str(weights / "model.safetensors"),
        )
        config.save_pretrained(str(weights))

        file_size = os.path.getsize(str(weights / "model.safetensors"))
        shares, approx = stream_setup.estimate_stripe_write_shares(
            str(weights), total_bytes=file_size, n_roots=2, dtype="bfloat16"
        )
        assert not approx

        cache_dir = tmp_path / "cache" / "model"
        stripe_dir = tmp_path / "stripe"
        stripe_dir.mkdir(parents=True, exist_ok=True)
        shard_checkpoint(
            str(weights), str(cache_dir), dtype="bfloat16", stripe_roots=(str(stripe_dir),)
        )

        actual_root0 = sum(f.stat().st_size for f in cache_dir.glob("*.safetensors"))
        subdirs = [d for d in stripe_dir.iterdir() if d.is_dir()]
        stripe_sub = subdirs[0] if subdirs else stripe_dir
        actual_root1 = sum(f.stat().st_size for f in stripe_sub.glob("*.safetensors"))

        assert shares[0] >= actual_root0
        assert shares[1] >= actual_root1
        # Within 10% tolerance
        assert abs(shares[0] - actual_root0) / actual_root0 < 0.10
        assert abs(shares[1] - actual_root1) / actual_root1 < 0.10

        # Mutation check: with even split, root 0's share is below its actual bytes
        even_shares = stream_setup._stripe_write_shares(file_size, 2)
        assert even_shares[0] < actual_root0

    def test_exact_shares_nf4_charges_non_decoder_at_streaming_dtype(self, tmp_path):
        """#1612: NF4 applies to linears, non-decoder charged at streaming dtype."""
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        file_size = os.path.getsize(os.path.join(weights, "model.safetensors"))
        shares_bf16, _ = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=2, dtype="bfloat16", quant="none"
        )
        shares_nf4, approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=file_size, n_roots=2, dtype="bfloat16", quant="nf4"
        )
        assert not approx
        assert sum(shares_nf4) == file_size
        # Root 0 holds non-decoder weights charged at full itemsize, so its share ratio
        # under NF4 is higher than under unquantised bf16
        assert shares_nf4[0] / sum(shares_nf4) > shares_bf16[0] / sum(shares_bf16)

    def test_preflight_unparseable_header_falls_back_to_even_split(self, tmp_path):
        """#1612 mutation check: garbage safetensors header falls back to even split."""
        corrupt = tmp_path / "corrupt"
        corrupt.mkdir(parents=True, exist_ok=True)
        with open(corrupt / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", 100))
            f.write(b"NOT_VALID_JSON_GARBAGE_HEADER_DATA")
        shares, approx = stream_setup.estimate_stripe_write_shares(
            str(corrupt), total_bytes=1000, n_roots=2
        )
        assert approx is True
        assert shares == stream_setup._stripe_write_shares(1000, 2)

    def test_exact_shares_qwen4_exp_ple_skipped_and_even_split_used(self, tmp_path):
        """#1612: qwen4_exp arch falls back to even split so external PLE tables are not charged."""
        qwen_dir = tmp_path / "qwen4_model"
        qwen_dir.mkdir(parents=True, exist_ok=True)
        cur = 0
        tensors = {
            "model.embed_tokens.weight": {
                "dtype": "BF16", "shape": [128, 64], "data_offsets": [cur, cur + 16384]
            },
            "model.layers.0.ple.ple_embedding.ngram_embedding.weight": {
                "dtype": "BF16", "shape": [64, 64], "data_offsets": [cur + 16384, cur + 24576]
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16", "shape": [64, 64], "data_offsets": [cur + 24576, cur + 32768]
            },
        }
        cur += 32768
        data = json.dumps(tensors).encode("utf-8")
        with open(qwen_dir / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(data)))
            f.write(data)
            f.write(b"\x00" * cur)

        shares, approx = stream_setup.estimate_stripe_write_shares(
            str(qwen_dir), total_bytes=10000, n_roots=2, arch="qwen4_exp"
        )
        assert approx is True
        assert shares == stream_setup._stripe_write_shares(10000, 2)

    def test_exact_shares_never_charge_less_than_the_header_bytes(self, tmp_path):
        """#1612: an estimate below the real header bytes is raised to them, not trusted."""
        weights = _make_synthetic_untied_checkpoint(tmp_path, n_layers=5)
        shares, approx = stream_setup.estimate_stripe_write_shares(
            weights, total_bytes=1000, n_roots=2, dtype="float32"
        )
        assert not approx
        assert sum(shares) > 1000


def _make_synthetic_untied_checkpoint(tmp_path, n_layers: int = 5) -> str:
    """A synthetic safetensors checkpoint with n_layers and untied lm_head (no torch needed)."""
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    tensors = {}
    cur = 0
    tensors["model.embed_tokens.weight"] = {
        "dtype": "BF16", "shape": [128, 64], "data_offsets": [cur, cur + 16384]
    }
    cur += 16384
    tensors["lm_head.weight"] = {
        "dtype": "BF16", "shape": [128, 64], "data_offsets": [cur, cur + 16384]
    }
    cur += 16384
    tensors["model.norm.weight"] = {
        "dtype": "BF16", "shape": [64], "data_offsets": [cur, cur + 128]
    }
    cur += 128
    for i in range(n_layers):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            name = (
                f"model.layers.{i}.self_attn.{proj}.weight"
                if "proj" in proj and "gate" not in proj and "up" not in proj and "down" not in proj
                else f"model.layers.{i}.mlp.{proj}.weight"
            )
            tensors[name] = {"dtype": "BF16", "shape": [64, 64], "data_offsets": [cur, cur + 8192]}
            cur += 8192
        tensors[f"model.layers.{i}.input_layernorm.weight"] = {
            "dtype": "BF16", "shape": [64], "data_offsets": [cur, cur + 128]
        }
        cur += 128
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = {
            "dtype": "BF16", "shape": [64], "data_offsets": [cur, cur + 128]
        }
        cur += 128

    data = json.dumps(tensors).encode("utf-8")
    file_path = weights / "model.safetensors"
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(data)))
        f.write(data)
        f.write(b"\x00" * cur)
    return str(weights)


def _drive_setup(tmp_path, monkeypatch, *, striped):
    """The real `_setup_streaming_transformers` on a tiny checkpoint, runtime boundary spied.

    Same harness as test_issue623 (simulated CUDA, ``build_streamed_model`` spied). Two temp
    folders share one volume, so ``volume_of`` is faked to tell them apart; everything else —
    the variable, the validation, the disk-kind callable, the sharder, the depth — is real.
    """
    import torch

    import soup_cli.utils.layer_stream as layer_stream
    import soup_cli.utils.layer_stream_runtime as runtime_module
    import soup_cli.utils.stripe_roots as stripe_module
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_issue623_stream_pin_allocation import _cfg_yaml, _tiny_checkpoint

    free = 10_000_000_000
    # A second call in the same folder reuses the checkpoint: rebuilding it would move its
    # mtimes, and the cache (keyed on them) would rightly be rewritten.
    existing = tmp_path / "model"
    weights = str(existing) if existing.is_dir() else _tiny_checkpoint(tmp_path)
    second_drive = tmp_path / "second-drive"
    second_drive.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    if striped:
        monkeypatch.setenv("SOUP_LAYER_STREAM_STRIPE_DIRS", str(second_drive))
    else:
        monkeypatch.delenv("SOUP_LAYER_STREAM_STRIPE_DIRS", raising=False)
    monkeypatch.setattr(
        stripe_module, "volume_of", lambda path: 2 if "second-drive" in str(path) else 1
    )
    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: free)
    monkeypatch.setattr(
        layer_stream, "classify_disk_kind",
        lambda *_a, **_k: layer_stream.DiskClassification("nvme"),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (free, free))
    monkeypatch.setattr(runtime_module, "expandable_segments_status", lambda *_a, **_k: (True, ""))

    preflight_calls = []
    orig_preflight = stream_setup._render_stream_disk_preflight

    def spy_preflight(**kwargs):
        preflight_calls.append(kwargs)
        return orig_preflight(**kwargs)

    monkeypatch.setattr(stream_setup, "_render_stream_disk_preflight", spy_preflight)

    captured = {}

    def spy_build(**kwargs):
        from unittest.mock import MagicMock

        captured.update(kwargs)
        runtime = MagicMock()
        runtime.stats.return_value = {
            "tier": "ram", "store_bytes": 1_000_000, "disk_bytes": 1_000_000,
            "pinned": False, "read_ahead": kwargs.get("read_ahead"), "buffers": 2,
            "buffer_bytes": 4_000, "n_layers": 3,
        }
        return MagicMock(), runtime

    monkeypatch.setattr(runtime_module, "build_streamed_model", spy_build)
    buffer = io.StringIO()
    monkeypatch.setattr(
        stream_setup, "console", Console(file=buffer, width=400, color_system=None)
    )

    cfg = load_config_from_string(_cfg_yaml(weights, ""))
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cuda"
    wrapper._setup_streaming_transformers(cfg, cfg.training)
    return captured, " ".join(buffer.getvalue().split()), second_drive, preflight_calls


class TestTheSetupPathUsesTheRoots:
    """R4 end to end: the variable reaches the sharder, the pre-flight and the reader."""

    def test_two_drives_stripe_the_cache_and_deepen_the_reader(self, tmp_path, monkeypatch):
        captured, said, second_drive, preflight_calls = _drive_setup(
            tmp_path, monkeypatch, striped=True
        )
        index = captured["index"]
        assert index.stripe_roots == (os.path.realpath(second_drive),)
        assert index.layer_roots == (0, 1, 0)
        assert captured["read_ahead"] == DEFAULT_STREAM_READ_AHEAD + 1
        # Finding A: the page-lock advice downstream needs to know the depth was raised.
        assert captured["read_ahead_decision"].raised
        assert "Layer cache striped over" in said
        assert "layer-shard stripe 1" in said  # the second drive is charged on its own
        assert f"training.stream_read_ahead {DEFAULT_STREAM_READ_AHEAD} -> 3" in said
        stripe_folder = second_drive / os.listdir(second_drive)[0]
        assert any(name.startswith("layer_") for name in os.listdir(stripe_folder)), (
            "no layer file landed on the second drive"
        )
        # #1612: exact shares charge root 0 more than root 1 (2 of 3 layers plus extras on root 0)
        assert len(preflight_calls) >= 1
        assert preflight_calls[0]["shard_write_bytes"] > preflight_calls[0]["stripe_writes"][0][1]

    def test_one_drive_is_left_exactly_as_it_was(self, tmp_path, monkeypatch):
        captured, said, _second_drive, _preflight = _drive_setup(
            tmp_path, monkeypatch, striped=False
        )
        index = captured["index"]
        assert index.stripe_roots == () and index.layer_roots == ()
        assert captured["read_ahead"] == DEFAULT_STREAM_READ_AHEAD
        assert "striped" not in said and "stripe 1" not in said

    def test_a_striped_cache_is_reused_not_rewritten_on_the_next_run(self, tmp_path, monkeypatch):
        """The pre-flight asks the cache check about the SAME roots the sharder will use."""
        _drive_setup(tmp_path, monkeypatch, striped=True)
        _captured, said, _second_drive, _preflight = _drive_setup(
            tmp_path, monkeypatch, striped=True
        )
        assert "(reusable)" in said
        assert "layer-shard stripe 1" not in said  # no write is planned for a cache we keep

    def test_the_setup_hands_the_estimate_its_dtype_quant_and_arch(self, tmp_path, monkeypatch):
        calls = []
        real = stream_setup.estimate_stripe_write_shares

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(stream_setup, "estimate_stripe_write_shares", spy)
        _drive_setup(tmp_path, monkeypatch, striped=True)
        assert calls, "the striped setup never asked for the exact shares"
        assert {"dtype", "quant", "double_quant", "arch"} <= set(calls[0])
        assert calls[0]["arch"] == "llama"
