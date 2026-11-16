"""#ISSUE: the VRAM estimate priced ``task: distill`` without its teacher.

``trainer/distill.py`` loads ``training.teacher_model`` frozen beside the student, at the
dtype its checkpoint stores, and every token-level step builds the teacher's logits and an
fp32 divergence over both models' vocabularies. The estimate behind the ``soup train``
pre-flight, ``soup profile``, the MCP ``profile`` tool and ``soup plan`` read only
``base``: it priced the run as a plain LoRA step on the student, at 0.26 of the peak the
repo's own H200 record measured (``benchmarks/distill-chunked-h200-2026-11-10.md``) and at
0.11 to 0.43 of the 18 rows measured for this fix
(``benchmarks/evidence/vram-distill-teacher.json``,
``benchmarks/gate-distill-teacher-vram.md``).

The estimate now charges a ``teacher`` term (its parameter count times its stored dtype)
and a ``distill`` term (bytes per element of batch x sequence x vocabulary, by kernel and
divergence). A distillation step counts as outside the calibrated range, so the pre-flight
warns instead of refusing unless the student and teacher weights alone exceed the card.
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
from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, vram_estimate
from soup_cli.utils.model_shape import ModelShape, read_model_shape, shape_from_config
from soup_cli.utils.vram_estimate import RunSpec, estimate_vram
from tests._tiny_checkpoint import write_tiny_model
from tests.conftest import make_cli_runner

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "benchmarks/evidence/vram-distill-teacher.json"
H200 = ROOT / "benchmarks/evidence/distill-722-h200"

_VOCAB = 32_000
_SEQ = 1024
_BATCH = 2
_ELEMENTS = _VOCAB * _SEQ * _BATCH

#: Allowed estimate / measured peak for a distillation row. The rows measured for this
#: fix land at 1.006 to 1.151, the H200 record's two arms at 1.047 and 1.088.
BAND = (1.00, 1.16)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "student", vocab_size=_VOCAB)
    write_tiny_model(
        tmp_path / "teacher", vocab_size=_VOCAB, hidden_size=256, intermediate_size=640,
        num_hidden_layers=6,
    )
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


def _terms(inp) -> dict:
    return hardware_fit.estimate_peak_vram_gb(inp).estimate.terms


def _teacher_params(workdir: Path) -> int:
    return read_model_shape(str(workdir / "teacher"), allow_network=False).params


# --- the terms -----------------------------------------------------------------------


def test_a_distill_run_is_charged_its_teacher_weights(workdir):
    """On ``main`` the estimate read ``base`` only: there was no teacher term."""
    terms = _terms(_input(workdir))
    assert terms["teacher"] == _teacher_params(workdir) * 2  # a 16-bit checkpoint


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "js"])
@pytest.mark.parametrize(
    ("training", "kernel"),
    [
        ({}, "dense"),
        ({"distill_chunk_size": 128}, "dense"),  # chunks kept for the backward
        ({"distill_checkpoint": True}, "checkpoint"),
        ({"distill_checkpoint": True, "distill_chunk_size": 128}, "checkpoint"),
    ],
    ids=["dense", "chunked", "checkpoint", "chunked-checkpoint"],
)
def test_the_divergence_term_follows_the_kernel_and_divergence(
    workdir, training, kernel, divergence
):
    inp = _input(workdir, distill_divergence=divergence, **training)
    assert (inp.distill_kernel, inp.distill_divergence) == (kernel, divergence)
    per = vram_estimate.DISTILL_KD_BYTES_PER_ELEMENT[kernel][divergence]
    assert _terms(inp)["distill"] == math.ceil(_ELEMENTS * per)


def test_the_coefficients_are_the_measured_maxima():
    assert vram_estimate.DISTILL_KD_BYTES_PER_ELEMENT == {
        "dense": {"forward_kl": 38.0, "reverse_kl": 38.0, "js": 50.0},
        "checkpoint": {"forward_kl": 21.0, "reverse_kl": 21.0, "js": 22.0},
    }


def test_an_fp32_teacher_checkpoint_is_charged_four_bytes_a_parameter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "student", vocab_size=_VOCAB)
    write_tiny_model(tmp_path / "teacher", vocab_size=_VOCAB, torch_dtype="float32")
    terms = _terms(_input(tmp_path))
    assert terms["teacher"] == _teacher_params(tmp_path) * 4


def test_sequence_mode_charges_the_teacher_but_no_divergence(workdir):
    inp = _input(workdir, distill_mode="sequence")
    est = hardware_fit.estimate_peak_vram_gb(inp).estimate
    assert est.terms["teacher"] == _teacher_params(workdir) * 2
    assert est.terms["distill"] == 0
    assert any("distill_mode: sequence" in note for note in est.notes)


@pytest.mark.parametrize("task", ["sft", "dpo", "kto"])
def test_other_tasks_are_priced_as_before(workdir, task):
    inp = _input(workdir, task)
    assert inp.teacher_shape is None and inp.distill_kernel is None
    terms = _terms(inp)
    assert terms["teacher"] == terms["distill"] == 0


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


def test_the_optimizer_step_phase_holds_the_teacher_too():
    """A short full fine-tune is set by the optimizer step (#2277); the teacher is
    resident through it."""
    shape = ModelShape(
        params=135_000_000, hidden_size=576, intermediate_size=1536, num_layers=30,
        num_attention_heads=9, num_kv_heads=3, head_dim=64, vocab_size=49152,
        tie_word_embeddings=True, model_type="llama", params_source="safetensors",
        shape_source="config.json",
    )
    base = RunSpec(seq_len=64, batch_size=1, peft="full")
    with_teacher = RunSpec(seq_len=64, batch_size=1, peft="full", teacher_params=10**9)
    a, b = estimate_vram(shape, base).terms, estimate_vram(shape, with_teacher).terms
    assert a["optimizer_step"] > 0
    assert b["optimizer_step"] == a["optimizer_step"]
    assert b["teacher"] == 2 * 10**9


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("teacher_params", -1, ValueError),
        ("teacher_params", True, TypeError),
        ("teacher_bytes_per_param", 1.0, ValueError),
        ("distill_kernel", "chunked", ValueError),
        ("distill_divergence", "kl", ValueError),
    ],
)
def test_run_spec_refuses_bad_distill_fields(field, value, error):
    with pytest.raises(error):
        RunSpec(seq_len=64, batch_size=1, **{field: value})


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


# --- measured peaks ------------------------------------------------------------------


def _rows() -> list[dict]:
    rows = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    return [row for row in rows if not row["oom"] and not row["void"]]


def _ident(row: dict) -> str:
    run = row["run"]
    return (
        f"{run['teacher'].split('/')[-1]}-{run['kernel']}-"
        f"{run.get('divergence', 'forward_kl')}-{run['seq']}x{run['batch']}-"
        f"gc{run.get('gradient_checkpointing', True)}"
    )


def _predicted(row: dict, *, teacher_and_distill: bool = True) -> int:
    run, student, teacher = row["run"], row["student"], row["teacher"]
    shape = shape_from_config(student, params=student["params"], params_source="safetensors")
    spec = RunSpec(
        seq_len=run["seq"],
        batch_size=run["batch"],
        gradient_checkpointing=run.get("gradient_checkpointing", True),
        lora_r=run.get("lora_r", 8),
        target_modules=["q_proj", "v_proj"],
        teacher_params=teacher["params"] if teacher_and_distill else 0,
        distill_kernel=run["kernel"] if teacher_and_distill else None,
        distill_divergence=run.get("divergence", "forward_kl"),
    )
    est = estimate_vram(shape, spec)
    return est.total_bytes - est.terms["overhead"]


def test_the_evidence_is_complete():
    rows = _rows()
    assert len(rows) == 18
    assert all(row["realised"] == row["run"]["seq"] for row in rows)
    assert all(row["steps"] == 2 for row in rows)
    assert {row["run"]["kernel"] for row in rows} == {"dense", "checkpoint"}
    assert {row["run"].get("divergence", "forward_kl") for row in rows} == {
        "forward_kl", "reverse_kl", "js"
    }
    assert {row["run"]["batch"] for row in rows} == {1, 2}
    assert any(row["teacher"]["params"] > 10**9 for row in rows)
    assert all(row["stack"]["device"] == "NVIDIA RTX PRO 5000 Blackwell" for row in rows)


@pytest.mark.parametrize("row", _rows(), ids=_ident)
def test_each_measured_peak_is_within_the_band(row):
    predicted, measured = _predicted(row), row["peak_allocated"]
    ratio = predicted / measured
    assert BAND[0] <= ratio <= BAND[1], (
        f"predicted {predicted / 1e9:.3f} GB, measured {measured / 1e9:.3f} GB ({ratio:.3f}x)"
    )


@pytest.mark.parametrize("row", _rows(), ids=_ident)
def test_without_the_new_terms_every_row_is_under_half_its_peak(row):
    """What ``main`` priced: the student's LoRA step alone."""
    assert _predicted(row, teacher_and_distill=False) / row["peak_allocated"] < 0.5


def test_the_oom_row_is_priced_over_the_cap_it_ran_out_of():
    """Run under ``set_per_process_memory_fraction(0.19)`` of a 47.3 GiB card."""
    (row,) = [r for r in json.loads(EVIDENCE.read_text(encoding="utf-8")) if r["oom"]]
    cap = 0.19 * row["total"]
    assert _predicted(row) > cap
    assert _predicted(row, teacher_and_distill=False) < cap / 2


# The H200 record of #722 (Qwen2.5-0.5B-Instruct student, Qwen2.5-1.5B-Instruct teacher,
# B=2, S=2048, LoRA r8 on q/v, checkpointing on): its config names local snapshot
# paths, so the shapes are the two models' config.json values and safetensors counts.
_QWEN25_05B = {
    "model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
    "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
    "vocab_size": 151936, "tie_word_embeddings": True, "torch_dtype": "bfloat16",
}
_QWEN25_05B_PARAMS = 494_032_768
_QWEN25_15B_PARAMS = 1_543_714_304


@pytest.mark.parametrize(("record", "kernel"), [("run-01-dense", "dense"),
                                                ("run-02-chunked", "checkpoint")])
def test_the_h200_record_is_within_the_band(record, kernel):
    data = json.loads((H200 / f"{record}.json").read_text(encoding="utf-8"))
    training = data["config"]["training"]
    assert (training["distill_checkpoint"], data["shape"]["batch"], data["shape"]["sequence"]) == (
        kernel == "checkpoint", 2, 2048
    )
    shape = shape_from_config(
        _QWEN25_05B, params=_QWEN25_05B_PARAMS, params_source="safetensors"
    )
    spec = RunSpec(
        seq_len=2048, batch_size=2, gradient_checkpointing=True, lora_r=8,
        target_modules=["q_proj", "v_proj"], teacher_params=_QWEN25_15B_PARAMS,
        distill_kernel=kernel,
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


def _big(tmp_path: Path) -> Path:
    """A 0.5B-shaped student config and a 1.5B-shaped teacher config, no weights."""
    for name, config in (
        ("student", _QWEN25_05B),
        ("teacher", {**_QWEN25_05B, "hidden_size": 1536, "intermediate_size": 8960,
                     "num_hidden_layers": 28, "num_attention_heads": 12}),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _h200_cfg(tmp_path: Path):
    return load_config_from_string(yaml.safe_dump({
        "base": str(tmp_path / "student"), "task": "distill",
        "data": {"train": "train.jsonl", "max_length": 2048},
        "training": {"teacher_model": str(tmp_path / "teacher"), "quantization": "none",
                     "batch_size": 2, "gradient_checkpointing": True,
                     "lora": {"r": 8, "target_modules": ["q_proj", "v_proj"]}},
    }))


def test_the_pre_flight_warns_where_main_passed_silently(tmp_path, monkeypatch, screen):
    """The H200 run (34 GB measured) on a 24 GB card. ``main`` priced it at 9.8 GB and
    printed nothing; it is now a warning, not a refusal: outside the calibrated range."""
    monkeypatch.chdir(_big(tmp_path))
    assert _preflight(_h200_cfg(tmp_path), 24) is None
    text = _plain(screen.getvalue())
    assert "Hardware-fit warning" in text
    assert "a distillation step" in text
    assert "teacher 3.1" in text
    assert "Warning only: launching anyway." in text


def test_the_pre_flight_refuses_when_the_two_models_alone_exceed_the_card(
    tmp_path, monkeypatch, screen
):
    monkeypatch.chdir(_big(tmp_path))
    assert _preflight(_h200_cfg(tmp_path), 4) == 1
    text = _plain(screen.getvalue())
    assert "Hardware-fit gate" in text
    assert "teacher weights" in text


def test_soup_profile_reports_the_teacher_and_the_divergence(workdir):
    (workdir / "soup.yaml").write_text(_yaml(workdir), encoding="utf-8")
    result = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    report = json.loads(result.stdout)
    terms = report["memory_terms_bytes"]
    assert terms["teacher"] == _teacher_params(workdir) * 2
    assert terms["distill"] == math.ceil(_ELEMENTS * 38.0)
    assert report["teacher_memory_gb"] == round(terms["teacher"] / 1e9, 2)
    assert report["distill_memory_gb"] == round(terms["distill"] / 1e9, 2)
    assert report["total_memory_bytes"] == sum(terms.values())

    text = make_cli_runner().invoke(app, ["profile", "--config", "soup.yaml"])
    assert text.exit_code == 0, text.output
    assert "Distillation teacher" in _plain(text.output)
    assert "Distillation divergence" in _plain(text.output)


# --- a real step on a CUDA card -----------------------------------------------------


@pytest.mark.gpu
def test_real_card_distill_peak_is_within_the_band(tmp_path, monkeypatch):
    """One real ``DistillTrainerWrapper`` run, two steps, against the estimate.

    On ``main``, on an RTX PRO 5000 Blackwell, this was priced at about 0.3 of its peak.
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
