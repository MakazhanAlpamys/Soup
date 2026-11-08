"""Loading a saved LoRA adapter into a layer-streamed inference model.

``build_streamed_inference_model(adapter=...)`` is the "exact reload" instrument for a LoRA run
trained on a streamed base: a directory written by ``PeftModel.save_pretrained`` (what ``soup
train`` writes) goes into a streamed base built in a fresh runtime, and the model's outputs are
the trained model's outputs.

The oracle is the trained streamed model itself: the same shards, the same kernels, the same
adapter bytes, so on CPU the logits are compared with ``torch.equal`` (bit for bit), the way
``tests/test_layer_stream_inference_mode.py`` compares streamed and resident logits.

Everything here is CPU only: a tiny Llama / Mistral checkpoint, sharded to fp32, trained for two
real optimizer steps through ``build_streamed_model(lora_config=...)``. The trained adapter of
each variant is built once per module and copied into a per-test working directory, because the
adapter path must stay under the current directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import namedtuple
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

VOCAB = 64
TARGETS = ("q_proj", "v_proj")

#: name -> (arch, tied embeddings, LoRA targets)
VARIANTS = {
    "llama-tied": ("llama", True, TARGETS),
    "llama-untied": ("llama", False, TARGETS),
    "llama-head": ("llama", False, TARGETS + ("lm_head",)),
    "mistral-tied": ("mistral", True, TARGETS),
}

Run = namedtuple("Run", "root weights shards index adapter ids mask trained_logits trained_masked")
_RUNS = {}


# ==========================================================================
# fixtures
# ==========================================================================
def _make_checkpoint(root, *, arch, tie, hidden=32, n_layers=2):
    """A real (tiny) checkpoint on disk, sharded to fp32. Returns (weights, shards, index)."""
    import transformers
    from safetensors.torch import save_file

    from soup_cli.utils.layer_shard import shard_checkpoint

    torch.manual_seed(7)
    common = dict(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=tie, max_position_embeddings=128,
    )
    if arch == "llama":
        config = transformers.LlamaConfig(**common)
        model = transformers.LlamaForCausalLM(config)
    else:
        config = transformers.MistralConfig(**common)
        model = transformers.MistralForCausalLM(config)
    model = model.to(torch.float32).eval()
    weights = Path(root) / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    shards = str(Path(root) / "shards")
    index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)
    return str(weights), shards, index


def _ids(batch=2, seq=10, seed=5):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, seq), generator=generator)


def _train_and_save(root, *, arch, tie, targets, steps=2):
    """Two real optimizer steps on a streamed LoRA model, then save_pretrained + eval logits."""
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_stream_runtime import (
        assert_streamed_adapter_saved,
        build_streamed_model,
    )

    weights, shards, index = _make_checkpoint(root, arch=arch, tie=tie)
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=list(targets),
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=weights, shard_dir=shards, index=index, lora_config=lora, device="cpu",
        dtype="float32", buffers=2, pin=False, seed=3,
    )
    ids = _ids()
    mask = torch.ones_like(ids)
    mask[1, :3] = 0
    adapter = Path(root) / "adapter"
    try:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        for _ in range(steps):
            model(input_ids=ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.save_pretrained(str(adapter))
        assert_streamed_adapter_saved(model, str(adapter))
        model.eval()
        with torch.inference_mode():
            plain = model(input_ids=ids, use_cache=False).logits.clone()
            masked = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.clone()
    finally:
        runtime.close()
    return Run(Path(root), weights, shards, index, adapter, ids, mask, plain, masked)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    def get(name):
        if name not in _RUNS:
            arch, tie, targets = VARIANTS[name]
            root = tmp_path_factory.mktemp(name)
            _RUNS[name] = _train_and_save(root, arch=arch, tie=tie, targets=targets)
        return _RUNS[name]

    return get


@pytest.fixture
def run(runs):
    """The plain tied Llama run most refusal tests use."""
    return runs("llama-tied")


@pytest.fixture
def work(tmp_path, monkeypatch, run):
    """A working directory holding a private copy of the trained adapter, as the cwd."""
    shutil.copytree(run.adapter, tmp_path / "adapter")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def no_build(monkeypatch):
    """Refusals that need only the adapter directory must happen before any model is built."""
    import soup_cli.utils.layer_stream_runtime as runtime_module

    def refuse(**_kwargs):
        raise AssertionError("the streamed model was built before the adapter was validated")

    monkeypatch.setattr(runtime_module, "build_streamed_model", refuse)


def _open(run, adapter="adapter", **kwargs):
    from soup_cli.utils.stream_inference import build_streamed_inference_model

    settings = dict(device="cpu", dtype="float32", pin=False)
    settings.update(kwargs)
    return build_streamed_inference_model(
        model_id=run.weights, shard_dir=run.shards, index=run.index, adapter=adapter, **settings
    )


def _edit_config(directory, **changes):
    path = Path(directory) / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(changes)
    path.write_text(json.dumps(config), encoding="utf-8")


def _edit_weights(directory, mutate):
    from safetensors.torch import load_file, save_file

    path = Path(directory) / "adapter_model.safetensors"
    state = load_file(str(path))
    mutate(state)
    save_file(state, str(path), metadata={"format": "pt"})


def _lora_keys(directory):
    from safetensors.torch import load_file

    return sorted(load_file(str(Path(directory) / "adapter_model.safetensors")))


# ==========================================================================
# the round trip: the trained model's outputs, reproduced
# ==========================================================================
class TestRoundTrip:
    @pytest.mark.parametrize("variant", list(VARIANTS))
    @pytest.mark.parametrize("tier", ["ram", "disk"])
    def test_logits_equal_the_trained_models_bit_for_bit(
        self, runs, tmp_path, monkeypatch, variant, tier
    ):
        """``torch.equal``: the strictest comparison there is, and it holds on CPU because the
        reloaded model runs the same kernels on the same shard bytes and the same adapter bytes."""
        trained = runs(variant)
        shutil.copytree(trained.adapter, tmp_path / "adapter")
        monkeypatch.chdir(tmp_path)
        with _open(trained, tier=tier) as handle:
            got = handle.logits(trained.ids)
            assert got.shape == trained.trained_logits.shape
            assert torch.equal(got, trained.trained_logits), (
                (got - trained.trained_logits).abs().max().item()
            )
            masked = handle.logits(trained.ids, trained.mask)
            assert torch.equal(masked, trained.trained_masked)

    def test_a_second_forward_and_a_fresh_runtime_give_the_same_bytes(self, run, work):
        with _open(run) as first:
            once = first.logits(run.ids).clone()
            again = first.logits(run.ids)
            assert torch.equal(once, again)
        with _open(run, buffers=3) as second:
            assert torch.equal(second.logits(run.ids), once)

    def test_log_probs_follow_the_reloaded_adapter(self, run, work):
        with _open(run) as handle:
            scores = handle.score_sequences(run.ids)
            logits = run.trained_logits
            expected = torch.zeros(run.ids.shape[0], run.ids.shape[1] - 1)
            for row in range(run.ids.shape[0]):
                log_probs = torch.log_softmax(logits[row, :-1].float(), dim=-1)
                expected[row] = log_probs.gather(-1, run.ids[row, 1:].unsqueeze(-1)).squeeze(-1)
            assert torch.equal(scores, expected)

    def test_the_adapter_is_actually_applied(self, run, work):
        """A reload that silently kept the LoRA at its zero-B init would match the base model."""
        with _open(run, adapter=None) as base:
            without = base.logits(run.ids)
        with _open(run) as adapted:
            with_adapter = adapted.logits(run.ids)
        assert not torch.equal(with_adapter, without)
        assert (with_adapter - without).abs().max().item() > 1e-4
        assert torch.equal(with_adapter, run.trained_logits)

    def test_loaded_tensors_equal_the_saved_file(self, run, work):
        from safetensors.torch import load_file

        stored = load_file(str(work / "adapter" / "adapter_model.safetensors"))
        with _open(run) as handle:
            live = {
                name.replace(".default", ""): param
                for name, param in handle.model.named_parameters()
                if "lora_" in name
            }
            assert live.keys() == stored.keys()
            assert any(
                bool(value.abs().sum() > 0) for key, value in live.items() if "lora_B" in key
            )
            for key, value in stored.items():
                assert torch.equal(live[key].detach().cpu(), value), key


# ==========================================================================
# the contract: forward-only
# ==========================================================================
class TestInferenceContract:
    def test_no_parameter_is_trainable_and_every_adapter_tensor_is_real(self, run, work):
        from soup_cli.utils.layer_stream_runtime import streamed_decoder_layers

        with _open(run) as handle:
            adapters = [(n, p) for n, p in handle.model.named_parameters() if "lora_" in n]
            assert adapters, "fixture: the reloaded model holds no LoRA tensor"
            assert [n for n, p in handle.model.named_parameters() if p.requires_grad] == []
            assert [n for n, p in adapters if p.is_meta] == []
            assert handle.model.training is False
            assert handle.model.config.use_cache is False
            layers = streamed_decoder_layers(handle.model)
            assert [layer.use_checkpoint for layer in layers] == [False] * len(layers)

    def test_loading_does_not_make_peft_warn_about_a_renamed_base(self, run, work):
        """The recorded base name is checked here; PEFT's own "renamed from" warning is noise."""
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*was renamed.*")
            with _open(run) as handle:
                recorded = handle.model.peft_config["default"].base_model_name_or_path
        assert recorded == run.weights

    def test_a_grad_enabled_call_builds_no_graph(self, run, work):
        with _open(run) as handle:
            assert torch.is_grad_enabled()
            out = handle.model(input_ids=run.ids, use_cache=False).logits
            assert out.requires_grad is False
            assert torch.equal(out, run.trained_logits)

    def test_the_result_is_an_inference_tensor(self, run, work):
        with _open(run) as handle:
            out = handle.logits(run.ids)
            assert out.is_inference()
            assert out.requires_grad is False

    def test_without_an_adapter_the_model_is_still_the_plain_hugging_face_class(self, run):
        with _open(run, adapter=None) as handle:
            assert type(handle.model).__name__ == "LlamaForCausalLM"
            assert not hasattr(handle.model, "peft_config")
            assert [n for n, p in handle.model.named_parameters() if p.requires_grad] == []
            assert handle.adapter is None

    def test_the_default_is_no_adapter(self, run):
        from soup_cli.utils.stream_inference import build_streamed_inference_model

        with build_streamed_inference_model(
            model_id=run.weights, shard_dir=run.shards, index=run.index, device="cpu",
            dtype="float32", pin=False,
        ) as handle:
            assert type(handle.model).__name__ == "LlamaForCausalLM"

    def test_the_handle_records_the_adapter_it_loaded(self, run, work):
        with _open(run) as handle:
            assert os.path.samefile(handle.adapter, work / "adapter")
            assert hasattr(handle.model, "peft_config")

    def test_close_and_the_with_block_release_the_runtime(self, run, work):
        handle = _open(run)
        runtime = handle.runtime
        assert runtime.hook is not None
        handle.logits(run.ids)
        handle.close()
        assert runtime.hook is None, "the prefetch hook was left attached"
        assert handle.closed is True
        handle.close()
        with pytest.raises(RuntimeError, match="closed"):
            handle.logits(run.ids)
        with _open(run) as scoped:
            inner = scoped.runtime
        assert inner.hook is None
        assert scoped.closed is True

    def test_a_closed_disk_tier_model_stops_its_reader_thread(self, run, work):
        with _open(run, tier="disk") as handle:
            source = handle.runtime.source
            assert source._thread.is_alive()
            handle.logits(run.ids)
        assert not source._thread.is_alive()


# ==========================================================================
# the adapter argument: what is refused, before a model is built
# ==========================================================================
class TestAdapterDirectory:
    @pytest.mark.parametrize("value", [123, True, 1.5, ["adapter"]])
    def test_the_argument_must_be_a_path(self, run, no_build, value):
        with pytest.raises(TypeError, match="adapter must be a path"):
            _open(run, adapter=value)

    def test_an_empty_path_is_refused(self, run, no_build):
        with pytest.raises(ValueError, match="adapter must be non-empty"):
            _open(run, adapter="")

    def test_a_missing_directory_is_named(self, run, work, no_build):
        with pytest.raises(FileNotFoundError, match="no-such-adapter"):
            _open(run, adapter="no-such-adapter")

    def test_a_file_is_not_an_adapter_directory(self, run, work, no_build):
        (work / "plain.txt").write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match="plain.txt.*not a directory"):
            _open(run, adapter="plain.txt")

    def test_a_path_that_escapes_the_working_directory_is_refused(
        self, run, tmp_path, monkeypatch, no_build
    ):
        outside = tmp_path / "outside"
        shutil.copytree(run.adapter, outside)
        inside = tmp_path / "inside"
        inside.mkdir()
        monkeypatch.chdir(inside)
        with pytest.raises(ValueError, match="adapter .*must stay under cwd"):
            _open(run, adapter=str(outside))
        with pytest.raises(ValueError, match="adapter .*must stay under cwd"):
            _open(run, adapter=os.path.join("..", "outside"))

    @pytest.mark.requires_symlink
    def test_a_symlinked_adapter_directory_is_refused(self, run, work, no_build):
        os.symlink(str(work / "adapter"), str(work / "linked"), target_is_directory=True)
        with pytest.raises(ValueError, match="adapter must not be a (symlink|reparse point)"):
            _open(run, adapter="linked")

    @pytest.mark.requires_symlink
    def test_a_symlinked_weights_file_is_refused(self, run, work, no_build):
        weights = work / "adapter" / "adapter_model.safetensors"
        real = work / "real.safetensors"
        weights.replace(real)
        os.symlink(str(real), str(weights))
        with pytest.raises(ValueError, match="adapter_model.safetensors must not be a symlink"):
            _open(run)

    @pytest.mark.requires_symlink
    def test_a_symlinked_config_file_is_refused(self, run, work, no_build):
        config = work / "adapter" / "adapter_config.json"
        real = work / "real.json"
        config.replace(real)
        os.symlink(str(real), str(config))
        with pytest.raises(ValueError, match="adapter_config.json must not be a symlink"):
            _open(run)


class TestAdapterConfig:
    def test_a_missing_config_is_named(self, run, work, no_build):
        (work / "adapter" / "adapter_config.json").unlink()
        with pytest.raises(FileNotFoundError, match="no adapter_config.json in"):
            _open(run)

    def test_a_config_that_is_not_json_is_refused(self, run, work, no_build):
        (work / "adapter" / "adapter_config.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="adapter_config.json is not valid JSON"):
            _open(run)

    def test_a_config_that_is_not_an_object_is_refused(self, run, work, no_build):
        (work / "adapter" / "adapter_config.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="adapter_config.json must hold a JSON object"):
            _open(run)

    def test_an_oversized_config_is_refused_unread(self, run, work, no_build):
        (work / "adapter" / "adapter_config.json").write_text(
            '{"peft_type": "LORA", "pad": "' + "x" * (300 * 1024) + '"}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="adapter_config.json is larger than"):
            _open(run)

    @pytest.mark.parametrize("peft_type", ["IA3", "PREFIX_TUNING", "VERA", "LOKR", "", None, 7])
    def test_an_adapter_that_is_not_lora_is_refused(self, run, work, no_build, peft_type):
        _edit_config(work / "adapter", peft_type=peft_type)
        with pytest.raises(ValueError, match=r"not a LoRA adapter \(peft_type="):
            _open(run)

    def test_a_config_without_a_peft_type_is_refused(self, run, work, no_build):
        path = work / "adapter" / "adapter_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        del config["peft_type"]
        path.write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(ValueError, match=r"not a LoRA adapter \(peft_type=None\)"):
            _open(run)

    @pytest.mark.parametrize(
        "field,value,words",
        [
            ("modules_to_save", ["lm_head"], "modules_to_save"),
            ("modules_to_save", ["embed_tokens", "lm_head"], "modules_to_save"),
            ("use_dora", True, "DoRA"),
            ("target_parameters", ["experts.gate_up_proj"], "target_parameters"),
            ("trainable_token_indices", [1, 2], "trainable_token_indices"),
            ("layer_replication", [[0, 2]], "layer_replication"),
            ("bias", "all", "bias"),
            ("bias", "lora_only", "bias"),
            ("init_lora_weights", "pissa", "init_lora_weights"),
            ("init_lora_weights", "olora", "init_lora_weights"),
            ("init_lora_weights", "loftq", "init_lora_weights"),
        ],
    )
    def test_what_the_streamed_base_cannot_hold_is_refused_by_name(
        self, run, work, no_build, field, value, words
    ):
        _edit_config(work / "adapter", **{field: value})
        with pytest.raises(ValueError, match=f"{words}.*streamed"):
            _open(run)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("modules_to_save", None),
            ("modules_to_save", []),
            ("use_dora", False),
            ("target_parameters", None),
            ("bias", "none"),
            ("init_lora_weights", True),
            ("init_lora_weights", "gaussian"),
        ],
    )
    def test_the_same_fields_switched_off_are_accepted(self, run, work, field, value):
        _edit_config(work / "adapter", **{field: value})
        with _open(run) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)


class TestAdapterBase:
    def test_an_adapter_trained_on_another_base_is_refused_by_name(self, run, work, no_build):
        _edit_config(work / "adapter", base_model_name_or_path="some-org/other-model")
        with pytest.raises(ValueError) as caught:
            _open(run)
        message = str(caught.value)
        assert "some-org/other-model" in message
        assert run.weights in message
        assert "allow_adapter_base_mismatch" in message

    def test_the_check_can_be_waived_by_name(self, run, work):
        _edit_config(work / "adapter", base_model_name_or_path="some-org/other-model")
        with _open(run, allow_adapter_base_mismatch=True) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)

    def test_a_hub_id_matches_a_local_copy_of_the_same_name(self, run, work):
        _edit_config(
            work / "adapter", base_model_name_or_path="some-org/" + Path(run.weights).name
        )
        with _open(run) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)

    def test_the_same_directory_spelled_differently_matches(self, run, work):
        spelled = run.weights.replace("\\", "/") + "/"
        _edit_config(work / "adapter", base_model_name_or_path=spelled)
        with _open(run) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)

    def test_a_different_local_directory_of_the_same_name_is_refused(
        self, run, work, no_build
    ):
        twin = work / "elsewhere" / Path(run.weights).name
        twin.mkdir(parents=True)
        _edit_config(work / "adapter", base_model_name_or_path=str(twin))
        with pytest.raises(ValueError, match="was trained on base .*elsewhere"):
            _open(run)

    @pytest.mark.parametrize("value", [None, ""])
    def test_an_adapter_that_names_no_base_is_not_name_checked(self, run, work, value):
        _edit_config(work / "adapter", base_model_name_or_path=value)
        with _open(run) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)

    def test_a_base_name_that_is_not_text_is_refused(self, run, work, no_build):
        _edit_config(work / "adapter", base_model_name_or_path=["a", "b"])
        with pytest.raises(ValueError, match="base_model_name_or_path must be text"):
            _open(run)


class TestAdapterWeightsFile:
    def test_a_missing_weights_file_is_named(self, run, work, no_build):
        (work / "adapter" / "adapter_model.safetensors").unlink()
        with pytest.raises(FileNotFoundError, match="no adapter_model.safetensors in"):
            _open(run)

    def test_a_pickle_bin_is_not_loaded(self, run, work, no_build):
        (work / "adapter" / "adapter_model.safetensors").unlink()
        (work / "adapter" / "adapter_model.bin").write_bytes(b"not a pickle")
        with pytest.raises(FileNotFoundError, match=r"adapter_model\.bin.*pickle.*safetensors"):
            _open(run)

    def test_a_weights_path_that_is_a_directory_is_refused(self, run, work, no_build):
        weights = work / "adapter" / "adapter_model.safetensors"
        weights.unlink()
        weights.mkdir()
        with pytest.raises(ValueError, match="adapter_model.safetensors is not a regular file"):
            _open(run)

    def test_a_corrupt_weights_file_is_refused_by_name(self, run, work, no_build):
        (work / "adapter" / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError, match="adapter_model.safetensors cannot be read"):
            _open(run)

    def test_a_non_floating_tensor_is_refused(self, run, work, no_build):
        victim = next(k for k in _lora_keys(work / "adapter") if "lora_B" in k)
        _edit_weights(
            work / "adapter",
            lambda state: state.update({victim: state[victim].to(torch.int64)}),
        )
        with pytest.raises(ValueError, match=r"non-floating tensor .*lora_B.*\(I64\)"):
            _open(run)


# ==========================================================================
# the adapter against the built base: refused by name, and the runtime released
# ==========================================================================
class TestAdapterAgainstTheBase:
    def test_a_partial_adapter_is_refused(self, run, work):
        keys = _lora_keys(work / "adapter")
        dropped = [k for k in keys if "layers.1" in k and "v_proj" in k and "lora_B" in k]
        assert len(dropped) == 1
        _edit_weights(work / "adapter", lambda state: state.pop(dropped[0]))
        with pytest.raises(ValueError) as caught:
            _open(run)
        message = str(caught.value)
        assert "partial adapter" in message
        assert dropped[0] in message
        assert "missing 1 " in message

    def test_a_tensor_the_base_has_no_slot_for_is_refused(self, run, work):
        extra = "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"
        _edit_weights(
            work / "adapter", lambda state: state.update({extra: torch.zeros(4, 32)})
        )
        with pytest.raises(ValueError) as caught:
            _open(run)
        message = str(caught.value)
        assert "no slot" in message
        assert extra in message

    def test_the_streaming_wrappers_spelling_of_a_key_is_refused(self, run, work):
        """v0.72.0 wrote ``.inner.`` keys that reload as nothing (#1011)."""
        keys = _lora_keys(work / "adapter")
        victim = next(k for k in keys if "layers.0" in k and "q_proj" in k and "lora_A" in k)
        wrapped = victim.replace("layers.0.", "layers.0.inner.")

        def rename(state):
            state[wrapped] = state.pop(victim)

        _edit_weights(work / "adapter", rename)
        with pytest.raises(ValueError) as caught:
            _open(run)
        message = str(caught.value)
        assert wrapped in message and victim in message
        assert "'.inner.' key is the streaming wrapper's own spelling" in message

    def test_a_wrong_shape_is_refused_by_name_and_not_inside_peft(self, run, work):
        keys = _lora_keys(work / "adapter")
        victim = next(k for k in keys if "layers.0" in k and "q_proj" in k and "lora_A" in k)
        _edit_weights(work / "adapter", lambda state: state.update({victim: torch.zeros(4, 48)}))
        with pytest.raises(ValueError) as caught:
            _open(run)
        message = str(caught.value)
        assert victim in message
        assert "(4, 48)" in message and "(4, 32)" in message
        assert "different base" in message

    def test_an_adapter_of_another_base_width_is_refused(self, run, work, tmp_path_factory):
        """The real wrong-base case: the base is 64 wide, the adapter was trained on 32."""
        root = tmp_path_factory.mktemp("wide")
        weights, shards, index = _make_checkpoint(root, arch="llama", tie=True, hidden=64)
        wide = run._replace(weights=weights, shards=shards, index=index)
        with pytest.raises(ValueError) as caught:
            _open(wide, allow_adapter_base_mismatch=True)
        message = str(caught.value)
        assert "lora_" in message and "(4, 32)" in message and "(4, 64)" in message

    def test_a_refusal_after_the_build_releases_the_runtime(self, run, work, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as runtime_module

        built = []
        real = runtime_module.build_streamed_model

        def spy(**kwargs):
            model, runtime = real(**kwargs)
            built.append(runtime)
            return model, runtime

        monkeypatch.setattr(runtime_module, "build_streamed_model", spy)
        victim = next(k for k in _lora_keys(work / "adapter") if "lora_A" in k)
        _edit_weights(work / "adapter", lambda state: state.pop(victim))
        with pytest.raises(ValueError, match="partial adapter"):
            _open(run)
        assert len(built) == 1
        assert built[0].hook is None, "the refused build left its prefetch hook attached"

    def test_a_stored_dtype_is_cast_to_the_adapters_dtype(self, run, work):
        """An fp16 adapter loads (cast up); the model still runs and is not the base model."""
        _edit_weights(
            work / "adapter",
            lambda state: state.update({k: v.to(torch.float16) for k, v in state.items()}),
        )
        with _open(run) as handle:
            out = handle.logits(run.ids)
            assert out.dtype == torch.float32
            assert (out - run.trained_logits).abs().max().item() < 5e-2
            assert not any(
                p.dtype != torch.float32 for n, p in handle.model.named_parameters() if "lora_" in n
            )


class TestAdaptedBoundary:
    """A LoRA on ``lm_head`` makes PEFT save that module's frozen base weight beside the LoRA
    (``save_embedding_layers='auto'``). The shards are the base, so it is checked, not loaded."""

    COPY = "base_model.model.lm_head.base_layer.weight"

    @pytest.fixture
    def head(self, runs, tmp_path, monkeypatch):
        trained = runs("llama-head")
        shutil.copytree(trained.adapter, tmp_path / "adapter")
        monkeypatch.chdir(tmp_path)
        assert self.COPY in _lora_keys(tmp_path / "adapter"), "fixture: no frozen copy was saved"
        return trained

    def test_the_frozen_copy_is_not_loaded(self, head, tmp_path):
        """Garbage in the stored copy changes nothing: the base is the shard, not the file."""
        _edit_weights(
            tmp_path / "adapter",
            lambda state: state.update({self.COPY: torch.full_like(state[self.COPY], 9.0)}),
        )
        with _open(head) as handle:
            assert torch.equal(handle.logits(head.ids), head.trained_logits)

    def test_a_frozen_copy_of_the_wrong_shape_is_refused(self, head, tmp_path):
        _edit_weights(
            tmp_path / "adapter",
            lambda state: state.update({self.COPY: torch.zeros(VOCAB, 48)}),
        )
        with pytest.raises(ValueError) as caught:
            _open(head)
        message = str(caught.value)
        assert self.COPY in message and f"({VOCAB}, 48)" in message
        assert "different base" in message

    def test_a_frozen_copy_of_a_module_with_no_lora_is_refused(self, head, tmp_path):
        stray = "base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight"
        _edit_weights(
            tmp_path / "adapter",
            lambda state: state.update({stray: torch.zeros(16, 32)}),
        )
        with pytest.raises(ValueError, match=r"no slot.*k_proj\.base_layer\.weight"):
            _open(head)

    def test_the_head_adapter_is_applied(self, head, tmp_path):
        with _open(head, adapter=None) as base:
            without = base.logits(head.ids)
        assert not torch.equal(without, head.trained_logits)


class TestLoadResultIsChecked:
    """``set_peft_model_state_dict`` reports what it could not place; a reload that lands
    nothing used to be a UserWarning at some later load (#1011). Each check is exercised by
    replacing the loader, since the preflight above refuses everything the files can express."""

    def _patch(self, monkeypatch, behaviour):
        import peft

        real = peft.set_peft_model_state_dict

        def fake(model, state, adapter_name="default", **kwargs):
            return behaviour(real, model, state, adapter_name, **kwargs)

        monkeypatch.setattr(peft, "set_peft_model_state_dict", fake)

    def test_missing_keys_in_the_result_are_refused(self, run, work, monkeypatch):
        from types import SimpleNamespace

        def behaviour(real, model, state, adapter_name, **kwargs):
            real(model, state, adapter_name=adapter_name, **kwargs)
            return SimpleNamespace(
                missing_keys=["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"],
                unexpected_keys=[],
            )

        self._patch(monkeypatch, behaviour)
        with pytest.raises(RuntimeError, match=r"reported 1 missing LoRA tensor.*layers\.0"):
            _open(run)

    def test_unexpected_keys_in_the_result_are_refused(self, run, work, monkeypatch):
        from types import SimpleNamespace

        def behaviour(real, model, state, adapter_name, **kwargs):
            real(model, state, adapter_name=adapter_name, **kwargs)
            return SimpleNamespace(missing_keys=[], unexpected_keys=["stray.weight"])

        self._patch(monkeypatch, behaviour)
        with pytest.raises(RuntimeError, match=r"reported 1 unexpected key.*stray\.weight"):
            _open(run)

    def test_non_adapter_missing_keys_are_the_base_models_and_are_ignored(
        self, run, work, monkeypatch
    ):
        from types import SimpleNamespace

        def behaviour(real, model, state, adapter_name, **kwargs):
            real(model, state, adapter_name=adapter_name, **kwargs)
            return SimpleNamespace(
                missing_keys=["base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"],
                unexpected_keys=[],
            )

        self._patch(monkeypatch, behaviour)
        with _open(run) as handle:
            assert torch.equal(handle.logits(run.ids), run.trained_logits)

    def test_a_load_that_places_nothing_is_caught_by_comparing_the_tensors(
        self, run, work, monkeypatch
    ):
        from types import SimpleNamespace

        def behaviour(real, model, state, adapter_name, **kwargs):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        self._patch(monkeypatch, behaviour)
        with pytest.raises(RuntimeError, match=r"did not land.*lora_"):
            _open(run)

    def test_a_load_that_places_part_of_it_is_caught(self, run, work, monkeypatch):
        from types import SimpleNamespace

        def behaviour(real, model, state, adapter_name, **kwargs):
            half = dict(list(state.items())[: len(state) // 2])
            real(model, half, adapter_name=adapter_name, **kwargs)
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        self._patch(monkeypatch, behaviour)
        with pytest.raises(RuntimeError, match=r"did not land"):
            _open(run)

    def test_a_failed_check_releases_the_runtime(self, run, work, monkeypatch):
        from types import SimpleNamespace

        import soup_cli.utils.layer_stream_runtime as runtime_module

        built = []
        real_build = runtime_module.build_streamed_model

        def spy(**kwargs):
            model, runtime = real_build(**kwargs)
            built.append(runtime)
            return model, runtime

        monkeypatch.setattr(runtime_module, "build_streamed_model", spy)
        self._patch(
            monkeypatch,
            lambda real, model, state, name, **kw: SimpleNamespace(
                missing_keys=[], unexpected_keys=[]
            ),
        )
        with pytest.raises(RuntimeError, match="did not land"):
            _open(run)
        assert built and built[0].hook is None


# ==========================================================================
# the module stays import-light
# ==========================================================================
class TestModuleStaysLight:
    def test_importing_it_still_does_not_import_torch_or_peft(self):
        import soup_cli

        source_root = str(Path(soup_cli.__file__).resolve().parents[1])
        code = (
            "import sys; import soup_cli.utils.stream_inference as module; "
            "assert hasattr(module, 'build_streamed_inference_model'); "
            "assert 'torch' not in sys.modules, 'torch was imported'; "
            "assert 'peft' not in sys.modules, 'peft was imported'; "
            "assert 'safetensors' not in sys.modules, 'safetensors was imported'"
        )
        env = {**os.environ, "PYTHONPATH": source_root}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
        )
        assert result.returncode == 0, (result.stdout, result.stderr)

    def test_building_without_an_adapter_never_imports_peft(self, run, monkeypatch):
        monkeypatch.setitem(sys.modules, "peft", None)
        with _open(run, adapter=None) as handle:
            assert handle.logits(run.ids[:, :4]).shape == (2, 4, VOCAB)
