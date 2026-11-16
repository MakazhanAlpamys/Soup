"""The estimate's three terms taken from the capped runs of Qwen3-0.6B (2026-11-14 and -16).

A bf16 LoRA run's overhead is the CUDA context, the cuBLAS workspace, the temporaries of TRL's
entropy metric and 1.1 times the fp32 logits tensor, in place of 0.5 GB and 5% of everything
allocated; a Qwen3 base keeps the fp32 input of its QK-norm for the backward pass (4 bytes a
element of q and k, per token per layer); and a tied head the checkpoint stores twice is counted
once, as the model loads it. Everything else keeps the numbers it had: the 4-bit runs, which the
capped measurements show are already priced above their need, a full fine-tune and the tasks
without a vocabulary-sized loss.

The outcomes of the capped runs themselves (``trains at`` / ``fails at``) are pinned in
``test_vram_estimate_realised_length.py``; this file pins the formulas, so a change to one of
them fails here by name.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from soup_cli.utils import model_shape
from soup_cli.utils.model_shape import ModelShape, read_model_shape, shape_from_config
from soup_cli.utils.vram_estimate import (
    OVERHEAD_FIXED_BYTES,
    OVERHEAD_SLACK,
    RunSpec,
    estimate_vram,
    measured_activations_apply,
    measured_overhead_applies,
)
from tests._tiny_checkpoint import TINY_LLAMA, tensors_for, write_safetensors

MIB = 2**20
EVIDENCE = (
    Path(__file__).parents[1] / "benchmarks/evidence/vram-qwen3-0.6b-realised-length.json"
)


def _models() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))["models"]


def _shape(name: str = "Qwen/Qwen3-0.6B", **replace) -> ModelShape:
    known = _models()[name]
    shape = shape_from_config(known["config"], params=known["params"], params_source="safetensors")
    return dataclasses.replace(shape, **replace) if replace else shape


def _run(seq_len: int = 256, batch_size: int = 2, **kwargs) -> RunSpec:
    base = dict(seq_len=seq_len, batch_size=batch_size, quant="none", peft="lora", lora_r=16)
    base.update(kwargs)
    return RunSpec(**base)


# --- the overhead -----------------------------------------------------------------------


class TestWhichRunsGetTheMeasuredOverhead:
    def test_bf16_lora_with_a_vocabulary_sized_loss_does(self):
        assert measured_overhead_applies(_shape(), _run())

    def test_dora_is_lora_here(self):
        assert measured_overhead_applies(_shape(), _run(peft="dora"))

    @pytest.mark.parametrize(
        "change",
        [
            dict(quant="4bit", peft="qlora"),
            dict(quant="8bit"),
            dict(quant="gptq"),
            dict(quant="mxfp4"),
            dict(peft="full"),
            dict(peft="qlora"),
            dict(lm_head=False),
            dict(logits_loss=False),
            dict(task="dpo"),
            dict(task="grpo"),
            dict(task="tts"),
            dict(backend="unsloth"),
            dict(backend="mlx"),
        ],
        ids=lambda change: ",".join(f"{k}={v}" for k, v in change.items()),
    )
    def test_every_other_run_keeps_the_old_overhead(self, change):
        run = _run(**change)
        assert not measured_overhead_applies(_shape(), run)
        terms = estimate_vram(_shape(), run).terms
        allocated = sum(value for key, value in terms.items() if key != "overhead")
        assert terms["overhead"] == math.ceil(OVERHEAD_FIXED_BYTES + OVERHEAD_SLACK * allocated)

    def test_a_shape_with_no_vocabulary_keeps_the_old_overhead(self):
        shape = _shape(vocab_size=None)
        assert not measured_overhead_applies(shape, _run())


class TestTheMeasuredOverheadIsTheSumOfItsFourParts:
    def test_it_is_exactly_the_context_the_workspace_the_entropy_chunks_and_1_1_blocks(self):
        shape, run = _shape(), _run(seq_len=512, batch_size=2)
        vocab, tokens = shape.vocab_size, 1024
        assert vocab == 151_936
        expected = (
            162 * MIB  # the CUDA context outside torch (device - reserved, every bench row)
            + 64 * MIB  # the cuBLAS workspace
            + 2 * 128 * vocab * 4  # entropy_from_logits: two fp32 [128, vocab] temporaries
            + 1.1 * tokens * vocab * 4  # the unusable cache around the fp32 logits tensor
        )
        assert estimate_vram(shape, run).terms["overhead"] == pytest.approx(expected, abs=2)

    def test_it_is_1027_mib_for_the_micro_batch_that_fills_512_tokens(self):
        # Pinned in MiB: 162 + 64 + 148.4 + 1.1 x 593.5 = 1027.2.
        terms = estimate_vram(_shape(), _run(seq_len=512)).terms
        assert terms["overhead"] / MIB == pytest.approx(1027.2, abs=0.1)

    def test_a_micro_batch_under_128_rows_has_fewer_entropy_rows_than_the_chunk(self):
        shape = _shape()
        short = estimate_vram(shape, _run(seq_len=32, batch_size=2)).terms["overhead"]
        vocab = shape.vocab_size
        expected = 226 * MIB + 2 * 64 * vocab * 4 + 1.1 * 64 * vocab * 4
        assert short == pytest.approx(expected, abs=2)

    def test_it_grows_with_the_tokens_and_no_other_term_of_it_does(self):
        shape = _shape()
        small = estimate_vram(shape, _run(seq_len=64)).terms["overhead"]
        large = estimate_vram(shape, _run(seq_len=1024)).terms["overhead"]
        assert large > small > 226 * MIB
        batch_four = estimate_vram(shape, _run(seq_len=64, batch_size=4)).terms["overhead"]
        assert batch_four > small

    def test_the_4_bit_overhead_is_the_one_it_was(self):
        run = _run(seq_len=512, quant="4bit", peft="qlora")
        terms = estimate_vram(_shape(), run).terms
        allocated = sum(value for key, value in terms.items() if key != "overhead")
        assert terms["overhead"] == math.ceil(0.5e9 + 0.05 * allocated)


# --- the QK-norm term -------------------------------------------------------------------


def _activations(shape: ModelShape, **run_kwargs) -> int:
    return estimate_vram(shape, _run(**run_kwargs)).terms["activations"]


class TestTheQkNormTerm:
    def _extra(self, run_kwargs: dict) -> tuple[int, int]:
        shape = _shape()
        plain = dataclasses.replace(shape, model_type="qwen2")
        return _activations(shape, **run_kwargs), _activations(plain, **run_kwargs)

    def test_qwen3_keeps_4_bytes_of_q_and_k_per_token_per_layer(self):
        shape = _shape()
        with_qk, without = self._extra(dict(seq_len=256, batch_size=2))
        width = (shape.num_attention_heads + shape.num_kv_heads) * shape.head_dim
        assert (shape.num_attention_heads, shape.num_kv_heads, shape.head_dim) == (16, 8, 128)
        assert with_qk - without == pytest.approx(512 * 4 * width * shape.num_layers, abs=2)

    def test_it_is_0_328_mib_a_token_for_the_0_6b(self):
        with_qk, without = self._extra(dict(seq_len=512, batch_size=2))
        assert (with_qk - without) / MIB / 1024 == pytest.approx(0.328, abs=0.001)

    def test_with_full_checkpointing_it_is_one_layer_not_all(self):
        shape = _shape()
        with_qk, without = self._extra(dict(seq_len=256, batch_size=2, gradient_checkpointing=True))
        width = (shape.num_attention_heads + shape.num_kv_heads) * shape.head_dim
        assert with_qk - without == pytest.approx(512 * 4 * width, abs=2)

    def test_with_medium_checkpointing_it_is_the_kept_layers_and_the_recomputed_one(self):
        shape = _shape()
        with_qk, without = self._extra(
            dict(seq_len=256, batch_size=2, gradient_checkpointing="medium")
        )
        width = (shape.num_attention_heads + shape.num_kv_heads) * shape.head_dim
        kept = math.ceil(shape.num_layers / 2)
        assert with_qk - without == pytest.approx((kept + 1) * 512 * 4 * width, abs=2)

    @pytest.mark.parametrize(
        "change",
        [
            dict(quant="4bit", peft="qlora"),
            dict(quant="8bit"),
            dict(peft="full"),
            dict(task="dpo"),
            dict(backend="unsloth"),
        ],
        ids=lambda change: ",".join(f"{k}={v}" for k, v in change.items()),
    )
    def test_k_bit_loading_a_full_fine_tune_another_task_or_backend_do_not_get_it(self, change):
        assert not measured_activations_apply(_shape(), _run(**change))
        with_qk, without = self._extra(dict(seq_len=256, batch_size=2, **change))
        assert with_qk == without

    @pytest.mark.parametrize("model_type", ["qwen2", "llama", "gemma3", "olmo2", "qwen3_moe", ""])
    def test_no_other_family_gets_it(self, model_type):
        shape = _shape(model_type=model_type)
        assert not measured_activations_apply(shape, _run())

    def test_the_other_mode_terms_are_untouched_by_it(self):
        shape = _shape()
        plain = dataclasses.replace(shape, model_type="qwen2")
        a, b = estimate_vram(shape, _run()).terms, estimate_vram(plain, _run()).terms
        for term in ("weights", "adapters", "gradients", "optimizer", "logits", "dequant"):
            assert a[term] == b[term], term


# --- the tied head, counted once --------------------------------------------------------


class TestTiedHeadStoredTwice:
    def test_qwen3_0_6b_loads_596_million_not_the_751_million_its_headers_sum(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        assert known["params"] == 751_632_384
        assert _shape().params == 596_049_920
        assert known["params"] - _shape().params == 151_936 * 1024

    def test_qwen3_1_7b_too(self):
        known = _models()["Qwen/Qwen3-1.7B"]
        assert known["params"] - _shape("Qwen/Qwen3-1.7B").params == 151_936 * 2048

    def test_a_checkpoint_that_stores_the_tied_head_once_is_left_as_it_is(self):
        known = _models()["Qwen/Qwen3-4B"]
        assert _shape("Qwen/Qwen3-4B").params == known["params"]

    def test_an_untied_model_is_left_as_it_is(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        config = {**known["config"], "tie_word_embeddings": False}
        shape = shape_from_config(config, params=known["params"], params_source="safetensors")
        assert shape.params == known["params"]

    def test_an_untied_model_is_never_adjusted_even_when_its_count_looks_doubled(self):
        # The config says untied, so an extra table of parameters is not a stored tied head.
        known = _models()["Qwen/Qwen3-0.6B"]
        config = {**known["config"], "tie_word_embeddings": False}
        untied = shape_from_config(config, params=None, params_source="config.json").params
        params = untied + 151_936 * 1024
        shape = shape_from_config(config, params=params, params_source="safetensors")
        assert shape.params == params

    def test_a_count_read_from_the_config_is_already_the_loaded_one(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        shape = shape_from_config(known["config"], params=None, params_source="config.json")
        assert shape.params == 596_042_752  # tied: the table once

    def test_a_count_labelled_as_read_from_the_config_is_never_adjusted(self):
        # A header sum that arrives labelled "config.json" is not a stored count to second-guess.
        known = _models()["Qwen/Qwen3-0.6B"]
        shape = shape_from_config(
            known["config"], params=known["params"], params_source="config.json"
        )
        assert shape.params == known["params"]

    def test_the_hub_metadata_count_is_corrected_the_same_way(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        shape = shape_from_config(
            known["config"], params=known["params"], params_source="hub safetensors metadata"
        )
        assert shape.params == 596_049_920

    def test_a_mixture_of_experts_count_is_not_second_guessed(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        config = {**known["config"], "num_experts": 8}
        shape = shape_from_config(config, params=known["params"], params_source="safetensors")
        assert shape.params == known["params"]

    def test_a_multimodal_checkpoint_is_not_second_guessed(self):
        known = _models()["Qwen/Qwen3-0.6B"]
        config = {"model_type": "wrapper", "text_config": dict(known["config"])}
        shape = shape_from_config(config, params=known["params"], params_source="safetensors")
        assert shape.params == known["params"]

    @pytest.mark.parametrize(
        ("extra_fraction", "corrected"),
        [(0.0, True), (0.01, True), (-0.01, True), (0.03, False), (-0.03, False), (1.0, False)],
    )
    def test_the_count_must_be_the_configs_plus_one_table_within_two_percent(
        self, extra_fraction, corrected
    ):
        known = _models()["Qwen/Qwen3-0.6B"]
        table = 151_936 * 1024
        params = known["params"] + int(extra_fraction * table)
        shape = shape_from_config(known["config"], params=params, params_source="safetensors")
        assert (shape.params == params - table) is corrected

    def test_the_weights_term_follows_the_loaded_count(self):
        header_sum = _models()["Qwen/Qwen3-0.6B"]["params"]
        stored = dataclasses.replace(_shape(), params=header_sum)
        run = _run()
        weights = estimate_vram(stored, run).terms["weights"]
        assert weights - estimate_vram(_shape(), run).terms["weights"] == 2 * 151_936 * 1024

    def test_a_local_checkpoint_that_stores_both_tensors_is_read_as_loaded(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        config = {**TINY_LLAMA, "tie_word_embeddings": True}
        tensors = tensors_for(config)
        loaded = sum(math.prod(shape) for shape in tensors.values())
        tensors["lm_head.weight"] = [config["vocab_size"], config["hidden_size"]]
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        stored = write_safetensors(tmp_path / "model.safetensors", tensors)
        assert stored - loaded == config["vocab_size"] * config["hidden_size"]
        shape = read_model_shape(str(tmp_path), allow_network=False)
        assert shape.params == loaded
        assert shape.params_source == "safetensors"

    def test_a_local_checkpoint_that_stores_the_head_once_is_read_as_stored(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        config = {**TINY_LLAMA, "tie_word_embeddings": True}
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        stored = write_safetensors(tmp_path / "model.safetensors", tensors_for(config))
        assert read_model_shape(str(tmp_path), allow_network=False).params == stored

    def test_the_hub_path_goes_through_the_same_function(self, monkeypatch):
        known = _models()["Qwen/Qwen3-0.6B"]
        monkeypatch.setattr(
            model_shape, "_hub_lookup", lambda repo, **kw: (dict(known["config"]), None)
        )
        monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: known["params"])
        shape = read_model_shape("Qwen/Qwen3-0.6B", allow_network=False)
        assert shape.params == 596_049_920


# --- only the measured task -------------------------------------------------------------------


def _tasks() -> tuple[str, ...]:
    from typing import get_args

    from soup_cli.config.schema import SoupConfig

    return get_args(SoupConfig.model_fields["task"].annotation)


class TestOnlySftIsMeasured:
    def test_the_schema_has_a_task_literal_to_derive_from(self):
        assert "sft" in _tasks() and len(_tasks()) > 20

    @pytest.mark.parametrize("task", [name for name in _tasks() if name != "sft"])
    def test_every_other_task_keeps_the_old_overhead_and_no_qk_norm_term(self, task):
        # Derived from the schema, so a task added later is not measured until someone measures it.
        run = _run(task=task)
        assert not measured_overhead_applies(_shape(), run)
        assert not measured_activations_apply(_shape(), run)

    def test_sft_on_the_transformers_backend_is_the_measured_run(self):
        run = _run(task="sft", backend="transformers")
        assert measured_overhead_applies(_shape(), run)
        assert measured_activations_apply(_shape(), run)

    def test_the_pre_flights_input_carries_the_task_into_the_estimate(self, tmp_path):
        import yaml

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils.hardware_fit import (
            estimate_peak_vram_gb,
            hardware_fit_input_or_reason,
        )
        from tests._tiny_checkpoint import write_tiny_model

        write_tiny_model(tmp_path / "tiny")

        def terms(task: str, fmt: str) -> tuple[str, dict]:
            cfg = load_config_from_string(
                yaml.safe_dump(
                    {
                        "base": str(tmp_path / "tiny"),
                        "task": task,
                        "data": {"train": "t.jsonl", "format": fmt},
                        "training": {"batch_size": 2, "quantization": "none"},
                    }
                )
            )
            inp = hardware_fit_input_or_reason(cfg, allow_network=False)
            return inp.task, estimate_peak_vram_gb(inp).estimate.terms

        sft_task, sft_terms = terms("sft", "alpaca")
        dpo_task, dpo_terms = terms("dpo", "dpo")
        assert (sft_task, dpo_task) == ("sft", "dpo")

        def old_overhead(found: dict) -> int:
            allocated = sum(value for key, value in found.items() if key != "overhead")
            return math.ceil(0.5e9 + 0.05 * allocated)

        assert dpo_terms["overhead"] == old_overhead(dpo_terms)
        assert sft_terms["overhead"] != old_overhead(sft_terms)
