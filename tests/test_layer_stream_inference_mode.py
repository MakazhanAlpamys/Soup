"""Forward-only inference mode of the layer-streaming runtime.

``build_streamed_model(lora_config=None)`` builds the streamed model with no PEFT wrapper, no
trainable parameter and no checkpointing; ``utils/stream_inference.py`` is the public entry
point (build, forward logits, per-token log-probs, close) and
``benchmarks/harness/stream_infer_throughput.py`` is the measuring script.

The oracle is the resident Hugging Face model on CPU: streaming substitutes the same weight
bytes into the same kernels, so the logits are bit-exact (``torch.equal``), the tolerance
``tests/test_v07200.py::TestStreamedForwardParityCpu`` and ``benchmarks/harness/bitexact.py``
already use.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_v07200 import _build_streamed_cpu, _tiny_llama_dir

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "harness" / "stream_infer_throughput.py"
STREAM_INFERENCE = ROOT / "src" / "soup_cli" / "utils" / "stream_inference.py"
VOCAB = 64


def _shard(tmp_path, *, tie=True, n_layers=2):
    from soup_cli.utils.layer_shard import shard_checkpoint

    weights, resident, _ = _tiny_llama_dir(tmp_path, n_layers=n_layers, tie=tie)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    return weights, shards, index, resident


def _build(tmp_path, *, tie=True, tier="ram", buffers=2, n_layers=2):
    from soup_cli.utils.stream_inference import build_streamed_inference_model

    weights, shards, index, resident = _shard(tmp_path, tie=tie, n_layers=n_layers)
    handle = build_streamed_inference_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        device="cpu",
        dtype="float32",
        buffers=buffers,
        tier=tier,
    )
    return handle, resident


def _ids(batch, seq, seed=5):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=generator)


def _resident_logits(resident, input_ids, attention_mask=None):
    with torch.inference_mode():
        return resident(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


def _resident_log_probs(resident, input_ids, attention_mask=None):
    """The per-token log-probs, spelled the way ``score_sequences`` documents them."""
    logits = _resident_logits(resident, input_ids, attention_mask)
    out = torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32)
    for row in range(input_ids.shape[0]):
        log_probs = torch.log_softmax(logits[row, :-1].float(), dim=-1)
        out[row] = log_probs.gather(-1, input_ids[row, 1:].unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        out = out.masked_fill(attention_mask[:, 1:] == 0, 0.0)
    return out


# ==========================================================================
# the build: no peft, nothing trainable
# ==========================================================================
class TestBuildWithoutPeft:
    def test_inference_build_never_imports_peft(self, tmp_path, monkeypatch):
        """``sys.modules['peft'] = None`` makes any ``import peft`` raise ImportError."""
        weights, shards, index, resident = _shard(tmp_path)
        monkeypatch.setitem(sys.modules, "peft", None)
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        with build_streamed_inference_model(
            model_id=weights, shard_dir=shards, index=index, device="cpu", dtype="float32",
            pin=False,
        ) as handle:
            ids = _ids(1, 8)
            assert torch.equal(handle.logits(ids), _resident_logits(resident, ids))

    def test_the_blocked_peft_control_does_refuse_a_lora_build(self, tmp_path, monkeypatch):
        """The block above is real: the training path needs peft and says so."""
        from peft import LoraConfig, TaskType

        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, shards, index, _ = _shard(tmp_path)
        lora = LoraConfig(r=2, target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM)
        monkeypatch.setitem(sys.modules, "peft", None)
        with pytest.raises(ImportError):
            build_streamed_model(
                model_id=weights, shard_dir=shards, index=index, lora_config=lora,
                device="cpu", dtype="float32", pin=False,
            )

    def test_lora_config_defaults_to_inference_mode(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        weights, shards, index, _ = _shard(tmp_path)
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=shards, index=index, device="cpu", dtype="float32",
            pin=False,
        )
        try:
            assert type(model).__name__ == "LlamaForCausalLM"
            assert not hasattr(model, "peft_config")
        finally:
            runtime.close()

    @pytest.mark.parametrize("tie", [True, False])
    def test_no_parameter_requires_grad_and_nothing_is_peft(self, tmp_path, tie):
        handle, _ = _build(tmp_path, tie=tie)
        with handle:
            names = [name for name, _ in handle.model.named_parameters()]
            assert names, "fixture: the model exposes no parameters"
            assert [n for n, p in handle.model.named_parameters() if p.requires_grad] == []
            assert not any("lora_" in name or "base_layer" in name for name in names)
            assert type(handle.model).__name__ == "LlamaForCausalLM"
            assert handle.model.training is False
            assert handle.model.config.use_cache is False

    def test_no_gradient_checkpointing_wrapper_is_active(self, tmp_path):
        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        handle, _ = _build(tmp_path, n_layers=3)
        with handle:
            layers = streamed_decoder_layers(handle.model)
            assert [layer.idx for layer in layers] == [0, 1, 2]
            assert [layer.use_checkpoint for layer in layers] == [False, False, False]

    def test_a_grad_enabled_call_builds_no_graph(self, tmp_path):
        """A caller who forgets ``inference_mode`` still gets logits with no autograd graph."""
        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(2, 7)
            assert torch.is_grad_enabled()
            out = handle.model(input_ids=ids, use_cache=False).logits
            assert out.requires_grad is False
            assert torch.equal(out, _resident_logits(resident, ids))

    def test_training_path_keeps_the_peft_model_and_trainable_adapters(self, tmp_path):
        """With a ``lora_config`` the build is the one it always was."""
        model, runtime, _, _ = _build_streamed_cpu(tmp_path)
        try:
            assert type(model).__name__ == "PeftModelForCausalLM"
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            assert trainable and all("lora_" in name for name in trainable)
            frozen_base = [
                n for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad
            ]
            assert frozen_base == []
            streamed = [
                layer for layer in _layers_of(model) if getattr(layer, "use_checkpoint", False)
            ]
            assert len(streamed) == 2, "the training layers must keep their checkpointing"
        finally:
            runtime.close()


def _layers_of(model):
    from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

    return streamed_decoder_layers(model)


# ==========================================================================
# the forward: bit-exact against the resident model
# ==========================================================================
class TestForwardParity:
    @pytest.mark.parametrize("tie", [True, False])
    @pytest.mark.parametrize("tier", ["ram", "disk"])
    def test_logits_match_the_resident_model_bit_exactly(self, tmp_path, tie, tier):
        handle, resident = _build(tmp_path, tie=tie, tier=tier)
        with handle:
            ids = _ids(1, 12)
            got = handle.logits(ids)
            want = _resident_logits(resident, ids)
            assert got.shape == (1, 12, VOCAB)
            assert torch.equal(got, want), (got - want).abs().max().item()

    def test_batch_and_every_sequence_length_match(self, tmp_path):
        handle, resident = _build(tmp_path)
        with handle:
            for batch, seq in ((2, 5), (3, 16), (4, 1), (1, 33)):
                ids = _ids(batch, seq, seed=batch * 100 + seq)
                got = handle.logits(ids)
                assert got.shape == (batch, seq, VOCAB)
                assert torch.equal(got, _resident_logits(resident, ids)), (batch, seq)

    @pytest.mark.parametrize("side", ["right", "left"])
    def test_padded_attention_masks_match(self, tmp_path, side):
        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(3, 9)
            mask = torch.ones_like(ids)
            for row, pad in enumerate((0, 3, 5)):
                if pad:
                    if side == "right":
                        mask[row, 9 - pad :] = 0
                    else:
                        mask[row, :pad] = 0
            got = handle.logits(ids, mask)
            want = _resident_logits(resident, ids, mask)
            valid = mask.bool().unsqueeze(-1).expand_as(got)
            assert torch.equal(got[valid], want[valid])
            assert not torch.isnan(got[valid]).any()

    def test_the_mask_changes_the_result(self, tmp_path):
        """A mask that is silently dropped would pass every parity test above."""
        handle, _ = _build(tmp_path)
        with handle:
            ids = _ids(1, 8)
            mask = torch.ones_like(ids)
            mask[0, :4] = 0
            masked = handle.logits(ids, mask)
            plain = handle.logits(ids)
            assert not torch.equal(masked[0, -1], plain[0, -1])

    def test_repeated_forwards_reprime_the_prefetcher(self, tmp_path):
        handle, resident = _build(tmp_path, n_layers=3)
        with handle:
            first_ids, second_ids = _ids(1, 6, seed=1), _ids(2, 10, seed=2)
            first = handle.logits(first_ids)
            second = handle.logits(second_ids)
            again = handle.logits(first_ids)
            assert torch.equal(first, again)
            assert torch.equal(first, _resident_logits(resident, first_ids))
            assert torch.equal(second, _resident_logits(resident, second_ids))

    def test_the_result_is_an_inference_tensor_without_a_graph(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            out = handle.logits(_ids(1, 4))
            assert out.is_inference()
            assert out.requires_grad is False

    def test_a_deeper_buffer_pool_gives_the_same_logits(self, tmp_path):
        handle, resident = _build(tmp_path, buffers=3, n_layers=4)
        with handle:
            ids = _ids(2, 6)
            assert torch.equal(handle.logits(ids), _resident_logits(resident, ids))

    def test_the_module_level_function_takes_a_model(self, tmp_path):
        from soup_cli.utils.stream_inference import streamed_forward_logits

        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(2, 6)
            got = streamed_forward_logits(handle.model, ids)
            assert torch.equal(got, _resident_logits(resident, ids))


# ==========================================================================
# score_sequences
# ==========================================================================
class TestScoreSequences:
    def test_log_probs_match_a_resident_reference(self, tmp_path):
        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(3, 10)
            got = handle.score_sequences(ids)
            want = _resident_log_probs(resident, ids)
            assert got.shape == (3, 9)
            assert got.dtype == torch.float32
            assert torch.equal(got, want)
            assert bool((got <= 0).all())

    def test_padded_targets_score_zero_and_the_rest_match(self, tmp_path):
        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(2, 8)
            mask = torch.ones_like(ids)
            mask[1, 5:] = 0
            got = handle.score_sequences(ids, mask)
            want = _resident_log_probs(resident, ids, mask)
            assert torch.equal(got, want)
            assert torch.equal(got[1, 4:], torch.zeros(3))
            assert bool((got[1, :4] < 0).all())

    def test_scores_are_log_probabilities_of_the_next_token(self, tmp_path):
        """``exp(score)`` is the softmax probability the model gives the actual next token."""
        handle, _ = _build(tmp_path)
        with handle:
            ids = _ids(1, 6)
            scores = handle.score_sequences(ids)
            probs = torch.softmax(handle.logits(ids)[0, :-1].float(), dim=-1)
            expected = probs.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            assert torch.allclose(scores[0].exp(), expected, rtol=1e-5)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)

    def test_the_module_level_function_takes_a_model(self, tmp_path):
        from soup_cli.utils.stream_inference import score_sequences

        handle, resident = _build(tmp_path)
        with handle:
            ids = _ids(2, 7)
            assert torch.equal(
                score_sequences(handle.model, ids), _resident_log_probs(resident, ids)
            )


# ==========================================================================
# close
# ==========================================================================
class TestClose:
    def test_close_detaches_the_prefetch_hook_and_is_idempotent(self, tmp_path):
        handle, _ = _build(tmp_path)
        runtime = handle.runtime
        assert runtime.hook is not None
        handle.close()
        assert runtime.hook is None, "the prefetch hook was left attached"
        assert handle.closed is True
        handle.close()

    def test_disk_tier_close_stops_the_reader_thread(self, tmp_path):
        from soup_cli.utils.async_disk_source import AsyncDiskSource

        handle, _ = _build(tmp_path, tier="disk")
        source = handle.runtime.source
        assert isinstance(source, AsyncDiskSource)
        assert source._thread.is_alive(), "the reader never started"
        handle.logits(_ids(1, 4))
        handle.close()
        assert not source._thread.is_alive()
        assert source._slots == []

    def test_the_context_manager_closes_on_exit_and_on_error(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle as entered:
            assert entered is handle
            assert handle.closed is False
        assert handle.closed is True

        failing, _ = _build(tmp_path / "second")
        with pytest.raises(RuntimeError, match="boom"):
            with failing:
                raise RuntimeError("boom")
        assert failing.closed is True
        assert failing.runtime.hook is None

    def test_a_closed_model_refuses_to_run_with_a_clear_message(self, tmp_path):
        handle, _ = _build(tmp_path)
        handle.close()
        with pytest.raises(RuntimeError, match="closed"):
            handle.logits(_ids(1, 4))
        with pytest.raises(RuntimeError, match="closed"):
            handle.score_sequences(_ids(1, 4))

    def test_stats_report_the_loads_of_a_forward(self, tmp_path):
        handle, _ = _build(tmp_path, n_layers=3)
        with handle:
            before = handle.stats()["layer_loads"]
            handle.logits(_ids(1, 5))
            assert handle.stats()["layer_loads"] - before >= 3
            assert handle.stats()["n_layers"] == 3


# ==========================================================================
# bad inputs
# ==========================================================================
class TestBadInputs:
    def test_empty_input_ids_are_refused(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            with pytest.raises(ValueError, match=r"input_ids is empty \(shape \(0, 4\)\)"):
                handle.logits(torch.zeros((0, 4), dtype=torch.long))
            with pytest.raises(ValueError, match=r"input_ids is empty \(shape \(2, 0\)\)"):
                handle.logits(torch.zeros((2, 0), dtype=torch.long))

    def test_input_ids_must_be_a_two_dimensional_tensor(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            with pytest.raises(ValueError, match=r"2-D \(batch, seq_len\).*got shape \(5,\)"):
                handle.logits(torch.arange(5))
            with pytest.raises(TypeError, match="input_ids must be a torch.Tensor"):
                handle.logits([[1, 2, 3]])

    def test_input_ids_must_be_integers(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            with pytest.raises(TypeError, match="integer token ids.*float32"):
                handle.logits(torch.ones((1, 4), dtype=torch.float32))

    def test_a_token_outside_the_vocabulary_is_refused(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            with pytest.raises(ValueError, match=r"token id 64 .*\[0, 64\)"):
                handle.logits(torch.tensor([[1, 2, VOCAB]]))
            with pytest.raises(ValueError, match=r"token id -1 .*\[0, 64\)"):
                handle.logits(torch.tensor([[1, -1, 2]]))

    def test_the_mask_must_match_the_ids(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            ids = _ids(2, 5)
            with pytest.raises(ValueError, match=r"attention_mask shape \(2, 4\).*\(2, 5\)"):
                handle.logits(ids, torch.ones((2, 4), dtype=torch.long))
            with pytest.raises(ValueError, match="only 0 and 1"):
                handle.logits(ids, torch.full((2, 5), 2, dtype=torch.long))
            with pytest.raises(ValueError, match="no real token"):
                handle.logits(ids, torch.zeros((2, 5), dtype=torch.long))
            with pytest.raises(TypeError, match="attention_mask must be a torch.Tensor"):
                handle.logits(ids, [[1] * 5, [1] * 5])

    def test_scoring_needs_a_context_token(self, tmp_path):
        handle, _ = _build(tmp_path)
        with handle:
            with pytest.raises(ValueError, match="at least 2 tokens.*got 1"):
                handle.score_sequences(_ids(2, 1))

    def test_a_missing_shard_index_names_the_directory(self, tmp_path):
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        weights, _, _, _ = _shard(tmp_path)
        empty = tmp_path / "no-shards"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no-shards"):
            build_streamed_inference_model(
                model_id=weights, shard_dir=str(empty), device="cpu", dtype="float32",
                pin=False,
            )

    def test_the_index_is_read_from_the_shard_dir_when_not_given(self, tmp_path):
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        weights, shards, _, resident = _shard(tmp_path)
        with build_streamed_inference_model(
            model_id=weights, shard_dir=shards, device="cpu", pin=False
        ) as handle:
            # dtype and quant come from the index: float32 / none
            ids = _ids(1, 6)
            assert torch.equal(handle.logits(ids), _resident_logits(resident, ids))

    @pytest.mark.parametrize("buffers", [1, 9, True, 2.0])
    def test_buffers_outside_the_streaming_bounds_are_refused(self, tmp_path, buffers):
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        weights, shards, index, _ = _shard(tmp_path)
        with pytest.raises(ValueError, match="buffers must be an integer between 2 and 8"):
            build_streamed_inference_model(
                model_id=weights, shard_dir=shards, index=index, device="cpu",
                dtype="float32", buffers=buffers, pin=False,
            )

    def test_a_shard_cache_of_another_depth_is_refused(self, tmp_path):
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        weights, _, _, _ = _shard(tmp_path / "a", n_layers=2)
        _, other_shards, _, _ = _shard(tmp_path / "b", n_layers=3)
        with pytest.raises(ValueError, match="reshard"):
            build_streamed_inference_model(
                model_id=weights, shard_dir=other_shards, device="cpu", dtype="float32",
                pin=False,
            )


# ==========================================================================
# the module stays import-light
# ==========================================================================
class TestModuleIsLazy:
    def test_no_module_level_heavy_import(self):
        tree = ast.parse(STREAM_INFERENCE.read_text(encoding="utf-8"))
        heavy = {"torch", "transformers", "peft", "bitsandbytes", "safetensors"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names & heavy, f"module-level import of {names & heavy}"

    def test_importing_it_does_not_import_torch(self):
        import soup_cli

        source_root = str(Path(soup_cli.__file__).resolve().parents[1])
        code = (
            "import sys; import soup_cli.utils.stream_inference as module; "
            "assert hasattr(module, 'build_streamed_inference_model'); "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'peft' not in sys.modules, 'peft was imported'"
        )
        env = {**os.environ, "PYTHONPATH": source_root}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
        )
        assert result.returncode == 0, (result.stdout, result.stderr)


# ==========================================================================
# the throughput harness
# ==========================================================================
def _harness():
    spec = importlib.util.spec_from_file_location("stream_infer_throughput", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestThroughputHarness:
    def test_help_runs_without_a_gpu_and_names_the_grid_flags(self, capsys):
        harness = _harness()
        with pytest.raises(SystemExit) as excinfo:
            harness.main(["--help"])
        assert excinfo.value.code == 0
        text = capsys.readouterr().out
        for flag in ("--shard-dir", "--model", "--batches", "--seq-lens", "--repeats", "--out"):
            assert flag in text, flag

    def test_import_is_light(self):
        source = HARNESS.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = (
                    node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
                )
                assert name.split(".")[0] not in {"torch", "transformers", "peft"}, name

    @pytest.mark.parametrize(
        "text,message",
        [
            ("", "at least one"),
            ("1,x", "integers"),
            ("0", "positive"),
            ("2,-1", "positive"),
            ("1,1", "duplicate"),
        ],
    )
    def test_the_grid_parser_refuses_bad_lists(self, text, message):
        harness = _harness()
        with pytest.raises(ValueError, match=message):
            harness.parse_int_list(text, "--batches")

    def test_the_grid_parser_keeps_order(self):
        harness = _harness()
        assert harness.parse_int_list("4, 1,2", "--batches") == [4, 1, 2]

    def test_median_is_the_middle_run(self):
        harness = _harness()
        assert harness.median([3.0, 1.0, 2.0]) == 2.0
        assert harness.median([4.0, 1.0, 3.0, 2.0]) == 2.5
        with pytest.raises(ValueError, match="no runs"):
            harness.median([])

    def test_a_tiny_cpu_run_writes_the_whole_grid(self, tmp_path, capsys):
        harness = _harness()
        weights, shards, index, _ = _shard(tmp_path, n_layers=4)
        out = tmp_path / "result.json"
        code = harness.main(
            [
                "--shard-dir", shards, "--model", weights, "--device", "cpu",
                "--dtype", "float32", "--batches", "1,2", "--seq-lens", "4,8",
                "--warmup", "1", "--repeats", "3", "--out", str(out),
            ]
        )
        assert code == 0, capsys.readouterr()
        record = json.loads(out.read_text(encoding="utf-8"))
        assert record["schema"] == 1
        assert record["device"] == "cpu"
        assert record["index"]["n_layers"] == 4
        assert record["settings"]["repeats"] == 3
        points = record["points"]
        assert [(p["batch"], p["seq_len"]) for p in points] == [(1, 4), (1, 8), (2, 4), (2, 8)]
        for point in points:
            assert point["tokens"] == point["batch"] * point["seq_len"]
            assert len(point["runs_s"]) == 3
            assert point["median_s"] == sorted(point["runs_s"])[1]
            assert point["tokens_per_s"] == pytest.approx(point["tokens"] / point["median_s"])
            assert point["peak_vram_bytes"] is None
            # 4 layers through 2 slots: every forward reloads every layer
            assert point["layer_loads_per_forward"] == 4.0
            assert point["large_loads_per_forward"] == 0.0
            assert point["bytes_loaded_est_per_forward"] > 0
        assert "tok/s" in capsys.readouterr().out

    def test_a_missing_shard_dir_is_refused_with_its_name(self, tmp_path, capsys):
        harness = _harness()
        with pytest.raises(SystemExit) as excinfo:
            harness.main(
                [
                    "--shard-dir", str(tmp_path / "nope"), "--model", "x", "--device", "cpu",
                    "--out", str(tmp_path / "o.json"),
                ]
            )
        assert excinfo.value.code == 2
        assert "nope" in capsys.readouterr().err

    def test_a_bad_grid_is_an_argument_error(self, tmp_path, capsys):
        harness = _harness()
        with pytest.raises(SystemExit) as excinfo:
            harness.main(
                [
                    "--shard-dir", str(tmp_path), "--model", "x", "--batches", "0",
                    "--out", str(tmp_path / "o.json"),
                ]
            )
        assert excinfo.value.code == 2
        assert "positive" in capsys.readouterr().err
