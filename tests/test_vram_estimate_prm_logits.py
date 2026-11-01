"""#2332: a PRM step is charged its logits tensor, not the cross-entropy chain.

``trainer/prm.py`` runs the causal-LM forward without labels and takes its loss from
``reward_head``, so the step builds the logits (a half-precision output and the fp32
copy Accelerate's ``convert_outputs_to_fp32`` makes of it) and no loss over them.
Every other task with an LM head keeps the logits-and-loss charge.

#2368: that forward passes ``logits_to_keep=1`` where the model takes it, so the PRM
logits cover one position. The model types whose forward lacks the argument
(``FULL_LOGITS_MODEL_TYPES``) and remote-code bases (``auto_map``) keep every position.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import yaml

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, vram_estimate
from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT
from soup_cli.utils.model_shape import read_model_shape
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import make_cli_runner

_VOCAB = 32_000
_SEQ = 2048
_BATCH = 4
_ELEMENTS = _VOCAB * _SEQ * _BATCH
#: A PRM step's logits under ``logits_to_keep=1``: one position (#2368).
_ONE_POSITION = _VOCAB * 1 * _BATCH


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny", vocab_size=_VOCAB)
    return tmp_path


def _yaml(base: str, task: str, quantization: str = "none") -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": task,
            "data": {"train": "train.jsonl", "max_length": _SEQ},
            "training": {"quantization": quantization, "batch_size": _BATCH},
        }
    )


def _input(workdir: Path, task: str, quantization: str = "none"):
    cfg = load_config_from_string(_yaml(str(workdir / "tiny"), task, quantization))
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    assert inp is not None
    return inp


def _terms(inp) -> dict:
    return hardware_fit.estimate_peak_vram_gb(inp).estimate.terms


# --- which tasks ------------------------------------------------------------------


def test_prm_is_the_one_logits_only_task():
    assert hardware_fit.LOGITS_ONLY_TASKS == {"prm"}
    assert not hardware_fit.LOGITS_ONLY_TASKS & hardware_fit.NO_LM_HEAD_TASKS


def test_the_logits_are_six_bytes_an_element():
    # The half-precision output (2) and Accelerate's fp32 copy of it (4), both live.
    assert vram_estimate.LOGITS_TENSOR_BYTES_PER_ELEMENT == 6


@pytest.mark.parametrize(
    ("task", "logits_loss"),
    [("prm", False), ("sft", True), ("dpo", True), ("kto", True), ("grpo", True)],
)
def test_the_config_carries_logits_loss(workdir, task, logits_loss):
    assert _input(workdir, task).logits_loss is logits_loss


# --- the term for PRM -------------------------------------------------------------


def test_prm_is_charged_the_logits_tensor_only(workdir):
    # #2368: one position, since the forward keeps the last position's logits only.
    logits = _terms(_input(workdir, "prm"))["logits"]
    assert logits == _ONE_POSITION * 6
    assert logits == _ONE_POSITION * vram_estimate.LOGITS_TENSOR_BYTES_PER_ELEMENT


def _shape(workdir: Path):
    return read_model_shape(str(workdir / "tiny"), allow_network=False)


def test_the_logits_are_eight_bytes_under_4bit(workdir):
    # The k-bit output head is fp32 (KBIT_LOGITS_EXTRA_BYTES). A prm config never
    # trains 4-bit (its trainer never quantises the base), so this is the estimate's
    # own arithmetic for a direct caller.
    run = vram_estimate.RunSpec(
        seq_len=_SEQ, batch_size=_BATCH, quant="4bit", peft="qlora", logits_loss=False
    )
    logits = vram_estimate.logits_bytes(_shape(workdir), run)
    assert logits == _ONE_POSITION * 8
    assert logits == _ONE_POSITION * (
        vram_estimate.LOGITS_TENSOR_BYTES_PER_ELEMENT + vram_estimate.KBIT_LOGITS_EXTRA_BYTES
    )


def test_a_model_without_an_lm_head_still_gets_no_logits(workdir):
    run = vram_estimate.RunSpec(
        seq_len=_SEQ, batch_size=_BATCH, lm_head=False, logits_loss=False
    )
    assert vram_estimate.logits_bytes(_shape(workdir), run) == 0


def test_logits_loss_defaults_to_true():
    assert vram_estimate.RunSpec(seq_len=_SEQ, batch_size=_BATCH).logits_loss is True
    inp = _input_fields()
    assert hardware_fit.HardwareFitInput(**inp).logits_loss is True


def _input_fields() -> dict:
    return {
        "params_b": 0.5,
        "seq_len": _SEQ,
        "batch_size": _BATCH,
        "optimizer": "adamw_torch",
        "quant": "none",
        "peft": "lora",
        "gradient_checkpointing": False,
    }


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_logits_loss_must_be_a_bool(value):
    with pytest.raises(TypeError, match="logits_loss must be bool"):
        vram_estimate.RunSpec(seq_len=_SEQ, batch_size=_BATCH, logits_loss=value)


# --- every other LM-head task is unchanged ----------------------------------------


def _main_logits(quantization: str) -> int:
    """``main``'s formula: logits and the loss chain, plus the fp32 head under k-bit."""
    per = LOGITS_LOSS_BYTES_PER_ELEMENT
    if quantization in ("4bit", "8bit"):
        per += vram_estimate.KBIT_LOGITS_EXTRA_BYTES
    return _VOCAB * _SEQ * _BATCH * per


@pytest.mark.parametrize("quantization", ["none", "4bit", "8bit"])
@pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto", "grpo"])
def test_other_tasks_keep_the_logits_and_loss_charge(workdir, task, quantization):
    assert _terms(_input(workdir, task, quantization))["logits"] == _main_logits(quantization)


def test_the_default_is_the_logits_and_loss_charge(workdir):
    run = vram_estimate.RunSpec(seq_len=_SEQ, batch_size=_BATCH)
    assert vram_estimate.logits_bytes(_shape(workdir), run) == _main_logits("none")


# --- soup profile --json ----------------------------------------------------------


def _profile(workdir: Path, task: str) -> dict:
    (workdir / "soup.yaml").write_text(_yaml(str(workdir / "tiny"), task), encoding="utf-8")
    result = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return json.loads(result.stdout)


def test_soup_profile_reports_the_prm_logits_term(workdir):
    report = _profile(workdir, "prm")
    terms = report["memory_terms_bytes"]
    # #2368: one position of logits, not the whole sequence.
    assert terms["logits"] == _ONE_POSITION * vram_estimate.LOGITS_TENSOR_BYTES_PER_ELEMENT

    # The same run with the loss chain charged, as before #2332.
    inp = _input(workdir, "prm")
    before = _terms(dataclasses.replace(inp, logits_loss=True))
    assert before["logits"] == _ELEMENTS * LOGITS_LOSS_BYTES_PER_ELEMENT

    # Only the logits term moves, and the overhead's slack on it.
    unchanged = [name for name in terms if name not in ("logits", "overhead")]
    assert {name: terms[name] for name in unchanged} == {
        name: before[name] for name in unchanged
    }
    allocated = sum(terms[name] for name in terms if name != "overhead")
    assert terms["overhead"] == math.ceil(
        vram_estimate.OVERHEAD_FIXED_BYTES + vram_estimate.OVERHEAD_SLACK * allocated
    )
    drop = before["logits"] - terms["logits"]
    assert drop == (
        _ELEMENTS * LOGITS_LOSS_BYTES_PER_ELEMENT
        - _ONE_POSITION * vram_estimate.LOGITS_TENSOR_BYTES_PER_ELEMENT
    )
    assert report["total_memory_bytes"] == sum(terms.values())
    assert sum(before.values()) - report["total_memory_bytes"] == drop + (
        before["overhead"] - terms["overhead"]
    )


# --- #2368: the forwards without logits_to_keep keep every position ---------------

#: transformers 5.20.2: the causal-LM classes (``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES``)
#: whose ``forward`` has no ``logits_to_keep``.
_WITHOUT_LOGITS_TO_KEEP = {
    "gemma4_assistant",
    "gemma4_unified_assistant",
    "musicgen",
    "musicgen_melody",
    "prophetnet",
    "trocr",
    "whisper",
    "xlstm",
}


def test_the_full_logits_model_types_are_the_eight_without_the_argument():
    assert vram_estimate.FULL_LOGITS_MODEL_TYPES == _WITHOUT_LOGITS_TO_KEEP


def _prm_logits(base: Path) -> int:
    cfg = load_config_from_string(_yaml(str(base), "prm"))
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    assert inp is not None
    return _terms(inp)["logits"]


@pytest.mark.parametrize("model_type", sorted(_WITHOUT_LOGITS_TO_KEEP))
def test_a_model_type_without_logits_to_keep_is_charged_every_position(
    tmp_path, monkeypatch, model_type
):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "base", vocab_size=_VOCAB, model_type=model_type)
    assert _prm_logits(tmp_path / "base") == _ELEMENTS * 6


def test_the_outer_model_type_counts_under_a_nested_text_config(tmp_path, monkeypatch):
    # gemma4_assistant nests a gemma4_text text_config (transformers 5.20.2), and the
    # shape's model_type is the nested one; the outer one decides here.
    from tests._tiny_checkpoint import TINY_LLAMA, tensors_for, write_safetensors

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "assistant"
    base.mkdir()
    text = {**TINY_LLAMA, "vocab_size": _VOCAB, "model_type": "gemma4_text"}
    config = {"model_type": "gemma4_assistant", "text_config": text}
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(base / "model.safetensors", tensors_for(text))
    shape = read_model_shape(str(base), allow_network=False)
    assert (shape.model_type, shape.config_model_type) == ("gemma4_text", "gemma4_assistant")
    assert _prm_logits(base) == _ELEMENTS * 6


def test_a_remote_code_base_is_charged_every_position(tmp_path, monkeypatch):
    # The estimate cannot see a remote-code forward, so it does not assume the argument.
    monkeypatch.chdir(tmp_path)
    auto_map = {"AutoModelForCausalLM": "modeling_tiny.TinyForCausalLM"}
    write_tiny_model(tmp_path / "remote", vocab_size=_VOCAB, auto_map=auto_map)
    assert read_model_shape(str(tmp_path / "remote"), allow_network=False).remote_code
    assert _prm_logits(tmp_path / "remote") == _ELEMENTS * 6


def test_a_native_base_reads_as_no_remote_code(workdir):
    shape = _shape(workdir)
    assert shape.remote_code is False
    assert shape.config_model_type == shape.model_type == "llama"

