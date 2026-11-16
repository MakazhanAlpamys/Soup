"""#2909: the VRAM estimate priced ``task: distill`` without its teacher.

``trainer/distill.py`` loads ``training.teacher_model`` frozen beside the student, at the
dtype its checkpoint stores, and every token-level step builds the teacher's logits and an
fp32 divergence over both models' vocabularies. The estimate behind the ``soup train``
pre-flight, ``soup profile``, the MCP ``profile`` tool and ``soup plan`` read only
``base``: it priced the run as a plain LoRA step on the student, at 0.26 of the peak the
repo's own H200 record measured (``benchmarks/distill-chunked-h200-2026-11-10.md``) and at
0.11 to 0.45 of the rows measured for this fix
(``benchmarks/evidence/vram-distill-teacher.json``,
``benchmarks/gate-distill-teacher-vram.md``).

The estimate now charges:

- ``teacher``: a token-level teacher's weights, resident for the whole run;
- ``distill``: bytes per element of batch x sequence x vocabulary, by kernel and
  divergence, plus what one chunk of ``distill_chunk_size`` positions costs (unset is one
  chunk of every position); ``uld_strategy`` and ``minillm_enabled`` are not measured and
  are priced at the largest measured figure, with a note;
- ``teacher_generation``: with ``distill_mode: sequence`` the teacher writes the training
  text during setup and is freed, so its weights, its cache and its prefill belong to
  that phase; the term is how far it exceeds training.

A distillation step counts as outside the calibrated range, so a run whose student fits
the card but not with its teacher gets a warning that names the new terms, and launches.
The student's own estimate is still inside the range: a run whose student alone does not
fit the card is refused, as on ``main``. The teacher's weights stay out of the floor
(``device_map: auto`` can keep part of a frozen teacher in CPU memory).
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import pytest
import typer
import yaml
from rich.console import Console

import soup_cli.commands.train as train_mod
import soup_cli.utils.model_shape as model_shape_mod
from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, vram_estimate
from soup_cli.utils.model_shape import ModelShape, read_model_shape, shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram
from tests._tiny_checkpoint import TINY_LLAMA, tensors_for, write_safetensors, write_tiny_model
from tests.conftest import make_cli_runner

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "benchmarks/evidence/vram-distill-teacher.json"
H200 = ROOT / "benchmarks/evidence/distill-722-h200"

_VOCAB = 32_000
_SEQ = 1024
_BATCH = 2
_POSITIONS = _SEQ * _BATCH
_TEACHER = {"vocab_size": _VOCAB, "hidden_size": 256, "intermediate_size": 640,
            "num_hidden_layers": 6}

#: Allowed estimate / measured peak for a token-level row. The rows measured for this fix
#: land at 1.001 to 1.134, the H200 record's two arms at 1.047 and 1.024.
BAND = (1.00, 1.16)
#: A ``distill_mode: sequence`` row whose training step, not its generation, sets the peak
#: is priced by the student's own terms, which this fix does not change (#2049 holds bf16
#: LoRA to 0.98 to 1.10). On the 40M-parameter student they leave out about 13 MB the
#: step holds (the cuBLAS workspaces among it): 0.973 and 0.988 of the peak.
SEQUENCE_BAND = (0.97, 1.16)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "student", vocab_size=_VOCAB)
    write_tiny_model(tmp_path / "teacher", **_TEACHER)
    return tmp_path


def _yaml(workdir: Path, task: str = "distill", **training) -> str:
    block = {"quantization": "none", "batch_size": _BATCH, "gradient_checkpointing": True}
    if task == "distill":
        block["teacher_model"] = str(workdir / "teacher")
    block.update(training)
    return yaml.safe_dump({
        "base": str(workdir / "student"),
        "task": task,
        "data": {"train": "train.jsonl", "max_length": _SEQ},
        "training": block,
    })


def _input(workdir: Path, task: str = "distill", **training):
    cfg = load_config_from_string(_yaml(workdir, task, **training))
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    assert inp is not None
    return inp


def _estimate(inp):
    return hardware_fit.estimate_peak_vram_gb(inp).estimate


def _terms(inp) -> dict:
    return _estimate(inp).terms


def _teacher(workdir: Path) -> ModelShape:
    return read_model_shape(str(workdir / "teacher"), allow_network=False)


def _divergence(kernel: str, divergence: str, chunk, positions: int = _POSITIONS) -> int:
    """The ``distill`` term, as the docstring of ``distill_bytes`` states it."""
    one = vram_estimate.DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT[kernel][divergence]
    kept = vram_estimate.DISTILL_KEPT_BYTES_PER_ELEMENT[kernel][divergence]
    span = positions if chunk is None else min(chunk, positions)
    return math.ceil(_VOCAB * (kept * positions + (one - kept) * span))


# --- the teacher and the divergence ----------------------------------------------------


def test_a_distill_run_is_charged_its_teacher_weights(workdir):
    """On ``main`` the estimate read ``base`` only: there was no teacher term."""
    terms = _terms(_input(workdir))
    assert terms["teacher"] == _teacher(workdir).params * 2  # a 16-bit checkpoint


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize(
    ("training", "kernel", "chunk"),
    [
        ({}, "dense", None),
        ({"distill_chunk_size": 128}, "dense", 128),
        ({"distill_checkpoint": True}, "checkpoint", None),
        ({"distill_checkpoint": True, "distill_chunk_size": 128}, "checkpoint", 128),
    ],
    ids=["dense", "chunked", "checkpoint-one-chunk", "checkpoint-chunked"],
)
def test_the_divergence_term_follows_kernel_divergence_and_chunk(
    workdir, training, kernel, chunk, divergence
):
    inp = _input(workdir, distill_divergence=divergence, **training)
    assert (inp.distill_kernel, inp.distill_divergence, inp.distill_chunk_size) == (
        kernel, divergence, chunk
    )
    assert _terms(inp)["distill"] == _divergence(kernel, divergence, chunk)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_a_chunk_of_every_position_is_one_chunk(workdir, checkpoint):
    """``distill_chunk_size`` unset runs the divergence as one chunk (``distill.py``), and
    so does a chunk as long as the step or longer."""
    unset = _terms(_input(workdir, distill_checkpoint=checkpoint))["distill"]
    for chunk in (_POSITIONS, 10 * _POSITIONS):
        terms = _terms(_input(workdir, distill_checkpoint=checkpoint, distill_chunk_size=chunk))
        assert terms["distill"] == unset


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl"])
def test_one_checkpointed_chunk_is_priced_as_the_dense_kernel(workdir, divergence):
    """Measured: with ``distill_chunk_size`` unset, ``distill_checkpoint`` peaks exactly as
    high as the dense kernel for both KL divergences (the record's rows 1/20, 2/21, 6/24
    and 14/23), so it is not priced at the 128-token chunk the first rows used."""
    dense = _terms(_input(workdir, distill_divergence=divergence))["distill"]
    checkpointed = _terms(
        _input(workdir, distill_divergence=divergence, distill_checkpoint=True)
    )["distill"]
    assert checkpointed == dense
    chunked = _terms(_input(
        workdir, distill_divergence=divergence, distill_checkpoint=True, distill_chunk_size=128
    ))["distill"]
    assert chunked == _divergence("checkpoint", divergence, 128) < 0.6 * dense


@pytest.mark.parametrize("checkpoint", [False, True])
def test_the_divergence_term_is_linear_in_the_chunk_size(workdir, checkpoint):
    sizes = (32, 128, 512)
    values = [
        _terms(_input(workdir, distill_checkpoint=checkpoint, distill_chunk_size=c))["distill"]
        for c in sizes
    ]
    slope = (values[2] - values[0]) / (sizes[2] - sizes[0])
    assert values[1] - values[0] == pytest.approx(slope * (sizes[1] - sizes[0]), abs=2)
    assert slope > 0


def test_the_coefficients_are_the_measured_maxima():
    assert vram_estimate.DISTILL_ONE_CHUNK_BYTES_PER_ELEMENT == {
        "dense": {"forward_kl": 38.0, "reverse_kl": 38.0, "js": 50.0},
        "checkpoint": {"forward_kl": 38.0, "reverse_kl": 38.0, "js": 46.0},
    }
    assert vram_estimate.DISTILL_KEPT_BYTES_PER_ELEMENT == {
        "dense": {"forward_kl": 26.0, "reverse_kl": 26.0, "js": 38.0},
        "checkpoint": {"forward_kl": 18.0, "reverse_kl": 18.0, "js": 18.0},
    }
    assert vram_estimate.DISTILL_UNMEASURED_BYTES_PER_ELEMENT == 50.0


@pytest.mark.parametrize(
    ("training", "kernel", "setting"),
    [
        ({"uld_strategy": "wasserstein"}, "uld", "uld_strategy"),
        ({"minillm_enabled": True, "minillm_teacher_mix_ratio": 0.5}, "minillm",
         "minillm_enabled"),
        ({"minillm_enabled": True, "minillm_on_policy": True}, "minillm", "minillm_enabled"),
    ],
    ids=["uld", "minillm-offline", "minillm-on-policy"],
)
def test_uld_and_minillm_are_priced_as_the_largest_measured_divergence(
    workdir, training, kernel, setting
):
    """Neither path is measured, and neither reads ``distill_divergence``,
    ``distill_chunk_size`` or ``distill_checkpoint``: both are charged their teacher and
    50 bytes an element (dense JS in one chunk, the largest measured), and a note says so."""
    inp = _input(workdir, distill_divergence="forward_kl", **training)
    assert inp.distill_kernel == kernel
    est = _estimate(inp)
    assert est.terms["teacher"] == _teacher(workdir).params * 2
    assert est.terms["distill"] == math.ceil(_VOCAB * _POSITIONS * 50.0)
    assert any(f"training.{setting} is set" in n and "not measured" in n for n in est.notes)
    assert ("a distillation step",) == hardware_fit.outside_calibrated_range(inp)


def test_an_fp32_teacher_checkpoint_is_charged_four_bytes_a_parameter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "student", vocab_size=_VOCAB)
    write_tiny_model(tmp_path / "teacher", vocab_size=_VOCAB, torch_dtype="float32")
    terms = _terms(_input(tmp_path))
    assert terms["teacher"] == _teacher(tmp_path).params * 4


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"torch_dtype": "bfloat16"}, 2.0),
        ({"dtype": "float16"}, 2.0),
        ({"torch_dtype": "float32"}, 4.0),
        ({"dtype": "torch.float32"}, 4.0),
        ({}, None),
        ({"torch_dtype": "int8"}, None),
    ],
)
def test_the_shape_reads_the_stored_dtype(config, expected):
    full = {
        "model_type": "llama", "hidden_size": 64, "intermediate_size": 160,
        "num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 1000, **config,
    }
    shape = shape_from_config(full, params=1000, params_source="safetensors")
    assert shape.stored_bytes_per_param == expected


def test_a_teacher_without_a_config_is_guessed_and_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "student", vocab_size=_VOCAB)
    (tmp_path / "teacher").mkdir()
    params = write_safetensors(
        tmp_path / "teacher" / "model.safetensors", tensors_for({**TINY_LLAMA, **_TEACHER})
    )
    est = _estimate(_input(tmp_path))
    assert est.terms["teacher"] == params * 2
    assert any("No config.json could be read for training.teacher_model" in n for n in est.notes)


def test_a_teacher_over_the_size_limit_is_not_priced(workdir, monkeypatch):
    real = model_shape_mod.read_model_shape
    teacher = str(workdir / "teacher")

    def read(base, **kwargs):
        shape = real(base, **kwargs)
        if base == teacher:
            return ModelShape(**{**shape.__dict__, "params": 2 * 10**12})
        return shape

    monkeypatch.setattr(model_shape_mod, "read_model_shape", read)
    cfg = load_config_from_string(_yaml(workdir))
    result = hardware_fit.hardware_fit_input_or_reason(cfg, allow_network=False)
    assert isinstance(result, hardware_fit.FitUnavailable)
    assert result.field == "training.teacher_model"


# --- distill_mode: sequence -----------------------------------------------------------


def _generation(workdir: Path, inp) -> int:
    """The writing phase, as ``teacher_generation_bytes`` states it, plus the student."""
    teacher = _teacher(workdir)
    terms = _terms(inp)
    tokens = _SEQ + min(_SEQ, 256)
    cache = 2 * teacher.num_layers * teacher.num_kv_heads * teacher.head_dim * 2 * tokens
    prefill = 4 * teacher.intermediate_size * 2 * _SEQ
    return (
        terms["weights"] + terms["adapters"] + teacher.params * 2 + cache + prefill
        + vram_estimate.GENERATION_WORKSPACE_BYTES
    )


def _big_teacher(workdir: Path) -> None:
    """A 1.5B-shaped teacher (a config.json, no weights) that outweighs the step."""
    (workdir / "teacher" / "model.safetensors").unlink()
    (workdir / "teacher" / "config.json").write_text(json.dumps({
        **TINY_LLAMA, "vocab_size": _VOCAB, "hidden_size": 1536, "intermediate_size": 8960,
        "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2,
    }), encoding="utf-8")


def test_sequence_mode_prices_the_teacher_in_its_generation_phase(workdir):
    """The teacher writes the training text during setup and is freed before the first step
    (``trainer/distill.py``): its weights belong to that phase with the student, its cache
    and its prefill, not to the step, and the peak is the larger of the two."""
    _big_teacher(workdir)
    inp = _input(workdir, distill_mode="sequence")
    est = _estimate(inp)
    sft = _terms(_input(workdir, "sft"))
    training = sum(v for k, v in sft.items() if k != "overhead")
    assert est.terms["teacher"] == est.terms["distill"] == 0
    assert est.terms["teacher_generation"] == _generation(workdir, inp) - training > 0
    allocated = est.total_bytes - est.terms["overhead"]
    assert allocated == _generation(workdir, inp)
    assert any("distill_mode: sequence" in n and "writing." in n for n in est.notes)


def test_sequence_mode_whose_step_is_larger_is_priced_as_the_step(workdir):
    """The small teacher writes in less than the step holds: the estimate is the step's."""
    est = _estimate(_input(workdir, distill_mode="sequence"))
    sft = _terms(_input(workdir, "sft"))
    assert est.terms["teacher_generation"] == 0
    assert est.terms == sft
    assert any("distill_mode: sequence" in n and "training." in n for n in est.notes)


def test_the_optimizer_step_holds_a_token_teacher_but_not_a_sequence_one():
    """A short full fine-tune is set by the optimizer step (#2277). A token-level teacher is
    resident through it; a sequence-level one is freed before it."""
    shape = ModelShape(
        params=135_000_000, hidden_size=576, intermediate_size=1536, num_layers=30,
        num_attention_heads=9, num_kv_heads=3, head_dim=64, vocab_size=49152,
        tie_word_embeddings=True, model_type="llama", params_source="safetensors",
        shape_source="config.json",
    )
    teacher = ModelShape(
        params=10**8, hidden_size=512, intermediate_size=1024, num_layers=4,
        num_attention_heads=8, num_kv_heads=2, head_dim=64, vocab_size=49152,
        tie_word_embeddings=True, model_type="llama", params_source="safetensors",
        shape_source="config.json",
    )
    base = estimate_vram(shape, RunSpec(seq_len=64, batch_size=1, peft="full")).terms
    token = estimate_vram(
        shape, RunSpec(seq_len=64, batch_size=1, peft="full", teacher=teacher)
    ).terms
    sequence = estimate_vram(
        shape,
        RunSpec(seq_len=64, batch_size=1, peft="full", teacher=teacher, distill_mode="sequence"),
    ).terms
    assert base["optimizer_step"] > 0
    # The teacher is in both phases, so only the divergence narrows the step's lead.
    assert token["optimizer_step"] == max(base["optimizer_step"] - token["distill"], 0)
    assert token["teacher"] == 2 * 10**8
    assert sequence["teacher"] == 0
    assert sequence["optimizer_step"] == base["optimizer_step"]
    assert sequence["teacher_generation"] == 0  # 0.2 GB of teacher under a 2.7 GB step


# --- every other run ------------------------------------------------------------------


@pytest.mark.parametrize("task", ["sft", "dpo", "kto"])
def test_other_tasks_are_priced_as_before(workdir, task):
    inp = _input(workdir, task)
    assert inp.teacher_shape is None
    terms = _terms(inp)
    assert terms["teacher"] == terms["distill"] == terms["teacher_generation"] == 0


def test_the_student_terms_are_unchanged(workdir):
    """Only ``teacher``, ``distill`` and the overhead's slack on them move."""
    distill = _terms(_input(workdir))
    sft = _terms(_input(workdir, "sft"))
    same = [name for name in vram_estimate.TERM_NAMES
            if name not in ("teacher", "distill", "overhead")]
    assert {n: distill[n] for n in same} == {n: sft[n] for n in same}
    allocated = sum(v for k, v in distill.items() if k != "overhead")
    assert distill["overhead"] == math.ceil(
        vram_estimate.OVERHEAD_FIXED_BYTES + vram_estimate.OVERHEAD_SLACK * allocated
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("teacher", "Qwen/Qwen2.5-1.5B", TypeError),
        ("distill_mode", "logit", ValueError),
        ("distill_kernel", "chunked", ValueError),
        ("distill_chunk_size", 0, ValueError),
        ("distill_chunk_size", True, TypeError),
        ("distill_divergence", "kl", ValueError),
    ],
)
def test_run_spec_refuses_bad_distill_fields(field, value, error):
    with pytest.raises(error):
        RunSpec(seq_len=64, batch_size=1, **{field: value})


def test_run_spec_refuses_a_teacher_dtype_it_has_no_figure_for():
    teacher = ModelShape(
        params=10**6, hidden_size=64, intermediate_size=160, num_layers=2,
        num_attention_heads=8, num_kv_heads=2, head_dim=8, vocab_size=1000,
        tie_word_embeddings=False, model_type="llama", params_source="safetensors",
        shape_source="config.json", stored_bytes_per_param=1.0,
    )
    with pytest.raises(ValueError):
        RunSpec(seq_len=64, batch_size=1, teacher=teacher)


# --- measured peaks ------------------------------------------------------------------


def _all_rows() -> list[dict]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _rows(mode: str) -> list[dict]:
    return [
        row for row in _all_rows()
        if not row["oom"] and not row["void"] and row["run"].get("mode", "token") == mode
    ]


def _ident(row: dict) -> str:
    run = row["run"]
    teacher = run["teacher"].split("/")[-1]
    if run.get("mode") == "sequence":
        return f"{teacher}-sequence-{run['seq']}x{run['batch']}"
    chunk = run.get("chunk", 128 if run["kernel"] == "checkpoint" else None)
    return (
        f"{teacher}-{run['kernel']}-chunk{chunk}-{run.get('divergence', 'forward_kl')}-"
        f"{run['seq']}x{run['batch']}-gc{run.get('gradient_checkpointing', True)}"
    )


def _spec(row: dict, *, teacher: bool = True) -> tuple[ModelShape, RunSpec]:
    run, student = row["run"], row["student"]
    shape = shape_from_config(student, params=student["params"], params_source="safetensors")
    mode = run.get("mode", "token")
    kernel = run.get("kernel", "dense")
    teacher_shape = (
        shape_from_config(row["teacher"], params=row["teacher"]["params"],
                          params_source="safetensors")
        if teacher else None
    )
    spec = RunSpec(
        seq_len=run["seq"],
        batch_size=run["batch"],
        gradient_checkpointing=run.get("gradient_checkpointing", True),
        lora_r=run.get("lora_r", 8),
        target_modules=["q_proj", "v_proj"],
        teacher=teacher_shape,
        distill_mode=mode,
        distill_kernel=kernel,
        # Rows recorded before the harness's "chunk" key could be null used 128.
        distill_chunk_size=run.get("chunk", 128 if kernel == "checkpoint" else None),
        distill_divergence=run.get("divergence", "forward_kl"),
    )
    return shape, spec


def _predicted(row: dict, *, teacher: bool = True) -> int:
    est = estimate_vram(*_spec(row, teacher=teacher))
    return est.total_bytes - est.terms["overhead"]


def _measured(row: dict) -> int:
    """The larger of the setup (both models, and the teacher's generation in sequence mode)
    and the steps; the first rows recorded the steps only, which hold both models too."""
    return max(row["peak_allocated"], row.get("setup_peak_allocated", 0))


def test_the_evidence_is_complete():
    rows = _all_rows()
    assert len(rows) == 39
    token, sequence = _rows("token"), _rows("sequence")
    assert (len(token), len(sequence)) == (34, 4)
    assert all(row["realised"] == row["run"]["seq"] for row in token + sequence)
    assert all(row["steps"] == 2 for row in token + sequence)
    assert {row["run"]["kernel"] for row in token} == {"dense", "checkpoint"}
    assert {_spec(row)[1].distill_chunk_size for row in token} == {None, 32, 128, 512}
    assert {row["run"].get("divergence", "forward_kl") for row in token} == {
        "forward_kl", "reverse_kl", "js"
    }
    assert {row["run"]["batch"] for row in token} == {1, 2}
    assert {row["teacher"]["params"] > 10**9 for row in token} == {True, False}
    assert {row["teacher"]["params"] > 10**9 for row in sequence} == {True, False}
    assert all(row["stack"]["device"] == "NVIDIA RTX PRO 5000 Blackwell" for row in rows)


@pytest.mark.parametrize("row", _rows("token"), ids=_ident)
def test_each_token_level_peak_is_within_the_band(row):
    predicted, measured = _predicted(row), _measured(row)
    ratio = predicted / measured
    assert BAND[0] <= ratio <= BAND[1], (
        f"predicted {predicted / 1e9:.3f} GB, measured {measured / 1e9:.3f} GB ({ratio:.3f}x)"
    )


@pytest.mark.parametrize("row", _rows("token"), ids=_ident)
def test_without_the_new_terms_every_token_level_row_is_under_half_its_peak(row):
    """What ``main`` priced: the student's LoRA step alone."""
    assert _predicted(row, teacher=False) / _measured(row) < 0.5


@pytest.mark.parametrize("row", _rows("sequence"), ids=_ident)
def test_each_sequence_generation_is_priced_over_its_setup_peak(row):
    """The writing phase alone (both models, the teacher's cache and prefill, the
    workspaces) against ``max_memory_allocated()`` over the setup that ran it."""
    shape, spec = _spec(row)
    est = estimate_vram(shape, spec)
    generation = (
        est.terms["weights"] + est.terms["adapters"] + vram_estimate.teacher_generation_bytes(spec)
    )
    ratio = generation / row["setup_peak_allocated"]
    assert BAND[0] <= ratio <= BAND[1], ratio


@pytest.mark.parametrize("row", _rows("sequence"), ids=_ident)
def test_each_sequence_peak_is_within_its_band(row):
    ratio = _predicted(row) / _measured(row)
    assert SEQUENCE_BAND[0] <= ratio <= SEQUENCE_BAND[1], ratio
    if row["setup_peak_allocated"] > row["peak_allocated"]:
        # The 1.5B teacher's writing sets the peak; main priced the student's step alone.
        assert BAND[0] <= ratio
        assert _predicted(row, teacher=False) / _measured(row) < 0.5


def test_the_oom_row_is_priced_over_the_cap_it_ran_out_of():
    """Run under ``set_per_process_memory_fraction(0.19)`` of a 47.3 GiB card."""
    (row,) = [r for r in _all_rows() if r["oom"]]
    cap = 0.19 * row["total"]
    assert _predicted(row) > cap
    assert _predicted(row, teacher=False) < cap / 2


# The H200 record of #722 (Qwen2.5-0.5B-Instruct student, Qwen2.5-1.5B-Instruct teacher,
# B=2, S=2048, LoRA r8 on q/v, checkpointing on): its config names local snapshot
# paths, so the shapes are the two models' config.json values and safetensors counts.
_QWEN25_05B = {
    "model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
    "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
    "vocab_size": 151936, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
}
_QWEN25_15B = {**_QWEN25_05B, "hidden_size": 1536, "intermediate_size": 8960,
               "num_hidden_layers": 28, "num_attention_heads": 12}
_QWEN25_05B_PARAMS = 494_032_768
_QWEN25_15B_PARAMS = 1_543_714_304


@pytest.mark.parametrize(("record", "kernel", "chunk"), [("run-01-dense", "dense", None),
                                                         ("run-02-chunked", "checkpoint", 128)])
def test_the_h200_record_is_within_the_band(record, kernel, chunk):
    data = json.loads((H200 / f"{record}.json").read_text(encoding="utf-8"))
    training = data["config"]["training"]
    assert (
        training["distill_checkpoint"], training["distill_chunk_size"],
        data["shape"]["batch"], data["shape"]["sequence"],
    ) == (kernel == "checkpoint", chunk, 2, 2048)
    shape = shape_from_config(
        _QWEN25_05B, params=_QWEN25_05B_PARAMS, params_source="safetensors"
    )
    teacher = shape_from_config(
        _QWEN25_15B, params=_QWEN25_15B_PARAMS, params_source="safetensors"
    )
    spec = RunSpec(
        seq_len=2048, batch_size=2, gradient_checkpointing=True, lora_r=8,
        target_modules=["q_proj", "v_proj"], teacher=teacher, distill_kernel=kernel,
        distill_chunk_size=chunk,
    )
    est = estimate_vram(shape, spec)
    predicted = est.total_bytes - est.terms["overhead"]
    measured = data["summary"]["peak_allocated_bytes"]
    assert BAND[0] <= predicted / measured <= BAND[1], (predicted, measured)
    # The teacher term is the record's own before-step allocation less the student.
    assert est.terms["teacher"] == _QWEN25_15B_PARAMS * 2


# --- the pre-flight and soup profile -------------------------------------------------


@pytest.fixture
def screen(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, width=400, force_terminal=False)
    )
    return buffer


def _preflight(cfg, card_gb: float):
    try:
        train_mod._hardware_fit_preflight(
            cfg, {"memory_total_bytes": int(card_gb * 1e9)}, allow_oom_attempt=False
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


def _big(tmp_path: Path, student: dict = _QWEN25_05B) -> Path:
    """A student config and a 1.5B-shaped teacher config, no weights."""
    for name, config in (("student", student), ("teacher", _QWEN25_15B)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _h200_cfg(tmp_path: Path, seq: int = 2048, batch: int = 2):
    return load_config_from_string(yaml.safe_dump({
        "base": str(tmp_path / "student"), "task": "distill",
        "data": {"train": "train.jsonl", "max_length": seq},
        "training": {"teacher_model": str(tmp_path / "teacher"), "quantization": "none",
                     "batch_size": batch, "gradient_checkpointing": True,
                     "lora": {"r": 8, "target_modules": ["q_proj", "v_proj"]}},
    }))


def test_a_run_whose_student_fits_but_not_its_teacher_is_warned(tmp_path, monkeypatch, screen):
    """The H200 run (34 GB measured) on a 24 GB card. Its student alone is priced at 9.8 GB,
    which fits, so ``main`` printed nothing. With the teacher and the divergence it does not
    fit, and a distillation step is outside the calibrated range: a warning that names both
    terms, and the run launches."""
    monkeypatch.chdir(_big(tmp_path))
    assert _preflight(_h200_cfg(tmp_path), 24) is None
    text = _plain(screen.getvalue())
    assert "Hardware-fit warning" in text
    assert "a distillation step" in text
    assert "teacher 3.1" in text
    assert "distillation 23.6" in text
    assert "Warning only: launching anyway." in text


def test_the_teacher_stays_out_of_the_floor(tmp_path, monkeypatch, screen):
    """A small student whose own step fits a 3 GB card, which ``main`` launched without a
    word, beside a 3.1 GB teacher. ``device_map: auto`` can keep part of a frozen teacher in
    CPU memory, so the run is warned, not refused: no run ``main`` launched is refused for
    its teacher."""
    student = {**_QWEN25_05B, "hidden_size": 256, "intermediate_size": 768,
               "num_hidden_layers": 2, "num_attention_heads": 4}
    monkeypatch.chdir(_big(tmp_path, student))
    cfg = _h200_cfg(tmp_path, seq=256, batch=1)
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    breakdown = hardware_fit.estimate_peak_vram_gb(inp)
    floor_gb, _ = hardware_fit.unextrapolated_floor(cfg, inp, breakdown)
    student_only = breakdown.total_gb - breakdown.teacher_gb - breakdown.distill_gb
    assert student_only * 1.1 < 3 < floor_gb + breakdown.teacher_gb
    assert _preflight(cfg, 3) is None
    text = _plain(screen.getvalue())
    assert "Hardware-fit warning" in text
    assert "a distillation step" in text


def test_a_run_whose_student_alone_does_not_fit_is_refused(tmp_path, monkeypatch, screen):
    """The same run on a 4 GB card. Its student alone needs 10.8 GB with the margin, so
    ``main`` refused it inside the calibrated range, and a teacher only adds to that: it is
    still refused, with ``--allow-oom-attempt`` to launch it anyway. Its floor fits the
    card, so the refusal is the student's estimate, not the floor."""
    monkeypatch.chdir(_big(tmp_path))
    cfg = _h200_cfg(tmp_path)
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    floor_gb, _ = hardware_fit.unextrapolated_floor(
        cfg, inp, hardware_fit.estimate_peak_vram_gb(inp)
    )
    assert floor_gb < 4
    assert _preflight(cfg, 4) == 1
    text = _plain(screen.getvalue())
    assert "Hardware-fit gate" in text
    assert "--allow-oom-attempt" in text
    assert "Its student alone needs 10.8 GB with the margin" in text


def test_the_student_floor_still_refuses(tmp_path, monkeypatch, screen):
    """As on ``main``: a card smaller than the student's own floor is refused."""
    monkeypatch.chdir(_big(tmp_path))
    assert _preflight(_h200_cfg(tmp_path), 1) == 1
    assert "Hardware-fit gate" in _plain(screen.getvalue())


def test_soup_profile_reports_the_teacher_and_the_divergence(workdir):
    (workdir / "soup.yaml").write_text(_yaml(workdir), encoding="utf-8")
    result = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    report = json.loads(result.stdout)
    terms = report["memory_terms_bytes"]
    assert terms["teacher"] == _teacher(workdir).params * 2
    assert terms["distill"] == _divergence("dense", "forward_kl", None)
    assert terms["teacher_generation"] == 0
    assert report["teacher_memory_gb"] == round(terms["teacher"] / 1e9, 2)
    assert report["distill_memory_gb"] == round(terms["distill"] / 1e9, 2)
    assert report["teacher_generation_memory_gb"] == 0
    assert report["total_memory_bytes"] == sum(terms.values())

    text = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml"])
    assert text.exit_code == 0, text.output
    assert "Distillation teacher" in _plain(text.output)
    assert "Distillation divergence" in _plain(text.output)


def test_soup_profile_reports_a_sequence_teacher_s_generation(workdir):
    _big_teacher(workdir)
    (workdir / "soup.yaml").write_text(_yaml(workdir, distill_mode="sequence"), encoding="utf-8")
    result = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    report = json.loads(result.stdout)
    terms = report["memory_terms_bytes"]
    assert terms["teacher"] == terms["distill"] == 0
    assert terms["teacher_generation"] > 0
    assert report["teacher_generation_memory_gb"] == round(terms["teacher_generation"] / 1e9, 2)
    assert any("distill_mode: sequence" in note for note in report["notes"])

    text = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml"])
    assert text.exit_code == 0, text.output
    assert "Teacher generation (over training)" in _plain(text.output)


# --- a real step on a CUDA card -----------------------------------------------------


@pytest.mark.gpu
def test_real_card_distill_peak_is_within_the_band(tmp_path, monkeypatch):
    """One real ``DistillTrainerWrapper`` run, two steps, against the estimate.

    On ``main``, on an RTX PRO 5000 Blackwell, this was priced at about 0.25 of its peak.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.trainer.distill import DistillTrainerWrapper
    from tests._tiny_hf_models import build_tokenizer

    monkeypatch.chdir(tmp_path)
    tokenizer = build_tokenizer()
    for name, seed in (("student", 0), ("teacher", 1)):
        torch.manual_seed(seed)
        LlamaForCausalLM(LlamaConfig(
            vocab_size=_VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        )).to(torch.bfloat16).save_pretrained(str(tmp_path / name))
        tokenizer.save_pretrained(str(tmp_path / name))
    seq, batch = 512, 2
    rows = [{"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "word " * (seq * 3)}]}] * (2 * batch)
    cfg = load_config_from_string(yaml.safe_dump({
        "base": str(tmp_path / "student"), "task": "distill",
        "data": {"train": "train.jsonl", "max_length": seq, "chat_template": "chatml",
                 "val_split": 0.0},
        "output": str(tmp_path / "out"),
        "training": {"teacher_model": str(tmp_path / "teacher"), "quantization": "none",
                     "batch_size": batch, "gradient_accumulation_steps": 1, "epochs": 1,
                     "gradient_checkpointing": True, "save_steps": 100000,
                     "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}},
    }))
    inp = hardware_fit.hardware_fit_input_from_config(cfg, allow_network=False)
    est = hardware_fit.estimate_peak_vram_gb(inp).estimate
    predicted = est.total_bytes - est.terms["overhead"]

    wrapper = DistillTrainerWrapper(cfg, device="cuda")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    assert next(iter(trainer.get_train_dataloader()))["input_ids"].shape == (batch, seq)
    trainer.args.max_steps = 2
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    measured = torch.cuda.max_memory_allocated()
    assert BAND[0] <= predicted / measured <= 1.25, (predicted / 1e9, measured / 1e9)
