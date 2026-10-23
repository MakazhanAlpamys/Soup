"""#2049 acceptance: the train pre-flight and ``soup profile`` return the same number.

The same config through both paths, and through the MCP ``profile`` tool, compared
in bytes before the pre-flight's 10% margin. Before #2049 the pre-flight and
``soup profile`` used two formulas in two units and disagreed on every row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string
from soup_cli.mcp_server.registry import tool_profile
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.layer_stream import LOGITS_LOSS_BYTES_PER_ELEMENT
from soup_cli.utils.vram_estimate import VRAM_SAFETY_MARGIN
from tests._tiny_checkpoint import TINY_LLAMA, write_tiny_model
from tests.conftest import strip_ansi

_ANY_CARD = {"memory_total_bytes": 10**15}

_RUNS = {
    "lora bf16": {"quantization": "none", "batch_size": 2, "lora": {"r": 16}},
    "lora bf16, checkpointing": {
        "quantization": "none",
        "batch_size": 4,
        "gradient_checkpointing": True,
        "lora": {"r": 8, "target_modules": "all-linear"},
    },
    "qlora": {"quantization": "4bit", "batch_size": 1, "lora": {"r": 64}},
    "full fine-tuning": {"quantization": "none", "batch_size": 1, "lora": {"r": 0}},
    "8-bit optimizer": {"quantization": "none", "batch_size": 2, "optimizer": "adamw_bnb_8bit"},
}


def _yaml(base: str, training: dict, max_length: int = 1024) -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": "sft",
            "data": {"train": "train.jsonl", "max_length": max_length},
            "training": training,
        }
    )


def _preflight_report(cfg, card=_ANY_CARD):
    seen = []
    real = hardware_fit.decide_hardware_fit

    def recording(inp, **kwargs):
        seen.append(real(inp, **kwargs))
        return seen[-1]

    hardware_fit.decide_hardware_fit = recording
    try:
        _hardware_fit_preflight(cfg, card, allow_oom_attempt=False)
    finally:
        hardware_fit.decide_hardware_fit = real
    assert len(seen) == 1, "the pre-flight did not price the run"
    return seen[0]


def _three_ways(tmp_path: Path, text: str) -> tuple[int, int, int, object]:
    (tmp_path / "soup.yaml").write_text(text, encoding="utf-8")
    report = _preflight_report(load_config_from_string(text))
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, result.output
    profile = json.loads(result.output)["total_memory_bytes"]
    mcp = tool_profile({"config": "soup.yaml"})["total_memory_bytes"]
    return report.breakdown.estimate.total_bytes, profile, mcp, report


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_tiny_model(tmp_path / "tiny")
    return tmp_path


@pytest.mark.parametrize("name", sorted(_RUNS))
def test_a_local_model_gets_one_number(workdir, name):
    text = _yaml(str(workdir / "tiny"), _RUNS[name])
    preflight, profile, mcp, report = _three_ways(workdir, text)
    assert preflight == profile == mcp
    # The comparison is before the margin, which only the gate adds.
    assert report.peak_vram_gb == pytest.approx(preflight / 1e9)
    assert report.required_with_margin_gb == pytest.approx(
        report.peak_vram_gb * (1 + VRAM_SAFETY_MARGIN)
    )


@pytest.mark.parametrize("name", sorted(_RUNS))
def test_a_cached_hub_model_gets_one_number(workdir, monkeypatch, name):
    # Found with or without the network: the model is in the local cache, which the
    # offline MCP tool reads too.
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 1_234_567)
    preflight, profile, mcp, _ = _three_ways(workdir, _yaml("someone/tiny-llama", _RUNS[name]))
    assert preflight == profile == mcp


def test_a_model_with_no_readable_config_gets_one_number_too(workdir):
    preflight, profile, mcp, report = _three_ways(
        workdir, _yaml("someone/Unreadable-8B", _RUNS["lora bf16"])
    )
    assert preflight == profile == mcp
    assert report.breakdown.logits_gb == 0  # the vocabulary is unknown, and it says so
    assert report.breakdown.estimate.notes


# --- the logits term follows what the run builds -------------------------------------

# The review's example: a 152,064-token vocabulary at batch 4 x 2,048 tokens. With an
# LM head that is 152,064 x 8,192 elements of logits and loss, about 14.9 GB.
_BIG_VOCAB = {**TINY_LLAMA, "vocab_size": 152_064}
_BIG_VOCAB_LOGITS = 152_064 * 2048 * 4 * LOGITS_LOSS_BYTES_PER_ELEMENT

# What each task needs besides the shared fields to load.
_TASK_FIELDS = {
    "classifier": {"num_labels": 2},
    "cross_encoder": {"num_labels": 1},
    "embedding": {},
    "reranker": {"num_labels": 1},
    "reward_model": {},
}


def _task_yaml(task: str, training: dict) -> str:
    return yaml.safe_dump(
        {
            "base": "someone/big-vocab",
            "task": task,
            "data": {"train": "train.jsonl", "max_length": 2048},
            "training": {"quantization": "none", "batch_size": 4, **training},
        }
    )


@pytest.fixture
def big_vocab(workdir, monkeypatch):
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(_BIG_VOCAB), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 1_234_567)
    return workdir


def _terms(task: str, **training) -> dict:
    return _preflight_report(load_config_from_string(_task_yaml(task, training)))


def test_the_tasks_without_an_lm_head():
    # The classifier family and reward_model load AutoModelForSequenceClassification,
    # embedding loads AutoModel: none has a vocabulary-sized output.
    assert hardware_fit.NO_LM_HEAD_TASKS == {
        "classifier",
        "cross_encoder",
        "embedding",
        "reranker",
        "reward_model",
    }


@pytest.mark.parametrize(
    "task", ["classifier", "cross_encoder", "embedding", "reranker", "reward_model"]
)
def test_a_task_without_an_lm_head_is_charged_no_logits(big_vocab, task):
    preflight, profile, mcp, report = _three_ways(big_vocab, _task_yaml(task, _TASK_FIELDS[task]))
    assert preflight == profile == mcp
    assert report.breakdown.estimate.terms["logits"] == 0
    assert _terms("sft").breakdown.estimate.terms["logits"] == _BIG_VOCAB_LOGITS


@pytest.mark.parametrize("task", ["dpo", "pretrain", "sft"])
def test_a_task_with_an_lm_head_is_charged_its_logits(big_vocab, task):
    assert _terms(task).breakdown.estimate.terms["logits"] == _BIG_VOCAB_LOGITS


@pytest.mark.parametrize(
    ("flags", "said"),
    [
        ({"use_liger": True}, "training.use_liger is set"),
        ({"use_cut_ce": True}, "training.use_cut_ce is set"),
        (
            {"use_liger": True, "use_cut_ce": True},
            "training.use_cut_ce and training.use_liger are set",
        ),
    ],
)
def test_a_fused_loss_keeps_the_logits_term_and_says_so(big_vocab, flags, said):
    # Whether the patch lands is known only at train time (architecture, install),
    # so the term stays and the note says it may overstate the peak.
    plain, fused = _terms("sft"), _terms("sft", **flags)
    assert fused.breakdown.estimate.terms == plain.breakdown.estimate.terms
    notes = [note for note in fused.breakdown.estimate.notes if said in note]
    assert len(notes) == 1
    assert "the logits term (14.9 GB) overstates the peak" in notes[0]
    assert not any("fused loss" in note for note in plain.breakdown.estimate.notes)


def test_use_cut_ce_on_a_task_that_never_applies_it_is_not_noted(big_vocab):
    # Only the SFT-family trainer applies the fused losses; dpo loads the flag and
    # builds its logits anyway.
    report = _terms("dpo", use_cut_ce=True)
    assert report.breakdown.estimate.terms["logits"] == _BIG_VOCAB_LOGITS
    assert not any("fused loss" in note for note in report.breakdown.estimate.notes)


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, renderable):
        self.printed.append(renderable)


def test_the_pre_flight_panel_carries_the_fused_loss_note(big_vocab, monkeypatch):
    console = _Console()
    monkeypatch.setattr(train_mod, "console", console)
    cfg = load_config_from_string(_task_yaml("sft", {"use_liger": True}))
    _hardware_fit_preflight(cfg, {"memory_total_bytes": int(8e9)}, allow_oom_attempt=True)
    panel = console.printed[-1].renderable
    assert "logits 14.9" in panel
    # After the advice, with the other notes, so the #1631 layout holds.
    assert panel.index("--allow-oom-attempt set") < panel.index("training.use_liger is set")


# --- gradient_checkpointing as the trainer applies it --------------------------------


def _tier_yaml(base: str, task: str, value) -> str:
    return yaml.safe_dump(
        {
            "base": base,
            "task": task,
            "data": {"train": "train.jsonl", "max_length": 1024},
            "training": {"batch_size": 2, "gradient_checkpointing": value},
        }
    )


@pytest.mark.parametrize(
    ("task", "value", "gib", "priced"),
    [
        ("sft", False, None, False),
        ("sft", True, None, "full"),
        ("sft", "medium", None, "medium"),
        ("sft", "selective", None, "selective"),
        # `auto` resolves on the card as trainer/sft.py does, and is full without one.
        ("sft", "auto", None, "full"),
        ("sft", "auto", 16.0, "full"),
        ("sft", "auto", 40.0, "medium"),
        ("sft", "auto", 96.0, "selective"),
        # Every other trainer turns any truthy value into plain (full) checkpointing.
        ("dpo", "medium", None, "full"),
        ("dpo", "selective", 40.0, "full"),
        ("dpo", "auto", 96.0, "full"),
        ("dpo", False, None, False),
    ],
)
def test_checkpointing_is_priced_as_the_trainer_applies_it(workdir, task, value, gib, priced):
    cfg = load_config_from_string(_tier_yaml(str(workdir / "tiny"), task, value))
    inp = hardware_fit.hardware_fit_input_from_config(cfg, gpu_memory_gb=gib)
    assert inp.gradient_checkpointing == priced


@pytest.mark.parametrize(
    ("card_gib", "tier"),
    # 23 GiB is 24.7e9 bytes and 80 GiB is 85.9e9: the card is read in GiB, as the
    # trainer reads it, or these two would resolve one tier lower.
    [(16, "full"), (23, "full"), (40, "medium"), (80, "medium"), (96, "selective")],
)
def test_the_pre_flight_resolves_auto_on_the_card(workdir, card_gib, tier):
    cfg = load_config_from_string(_tier_yaml(str(workdir / "tiny"), "sft", "auto"))
    report = _preflight_report(cfg, card={"memory_total_bytes": card_gib * 1024**3})
    assert report.breakdown.estimate.run.gradient_checkpointing == tier


def _profile_bytes(workdir, value, *gpu) -> int:
    (workdir / "soup.yaml").write_text(
        _tier_yaml(str(workdir / "tiny"), "sft", value), encoding="utf-8"
    )
    args = ["profile", "--config", "soup.yaml", "--json", *(("--gpu", gpu[0]) if gpu else ())]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return json.loads(result.output)["total_memory_bytes"]


@pytest.mark.parametrize(("gpu", "tier"), [("t4", True), ("a100", "medium"), ("h200", "selective")])
def test_soup_profile_resolves_auto_on_the_gpu_it_is_given(workdir, gpu, tier):
    assert _profile_bytes(workdir, "auto", gpu) == _profile_bytes(workdir, tier, gpu)


def test_soup_profile_prices_auto_as_full_on_an_assumed_card(workdir, monkeypatch):
    def _no_device():
        raise RuntimeError("no GPU")

    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", _no_device)
    assert _profile_bytes(workdir, "auto") == _profile_bytes(workdir, True)
    # The assumed 24 GB would have resolved to medium had it been used.
    assert _profile_bytes(workdir, "auto") != _profile_bytes(workdir, "medium")


def test_the_mcp_tool_resolves_auto_on_the_gpu_it_is_given(workdir):
    (workdir / "soup.yaml").write_text(
        _tier_yaml(str(workdir / "tiny"), "sft", "auto"), encoding="utf-8"
    )
    auto = tool_profile({"config": "soup.yaml", "gpu": "a100"})["total_memory_bytes"]
    (workdir / "soup.yaml").write_text(
        _tier_yaml(str(workdir / "tiny"), "sft", "medium"), encoding="utf-8"
    )
    assert auto == tool_profile({"config": "soup.yaml", "gpu": "a100"})["total_memory_bytes"]


# --- the fit verdict of soup profile and the MCP tool ------------------------------


@pytest.mark.parametrize(
    ("factor", "fit", "line"),
    [
        (1.2, "fits", "OK Fits in"),
        # Fits, but not with the pre-flight's 10% margin.
        (1.05, "tight", "~ Tight: fits in"),
        (0.95, "does not fit", "X Does NOT fit in"),
    ],
)
def test_the_fit_verdict_on_a_detected_card(workdir, monkeypatch, factor, fit, line):
    (workdir / "soup.yaml").write_text(
        _yaml(str(workdir / "tiny"), _RUNS["lora bf16"]), encoding="utf-8"
    )
    total = tool_profile({"config": "soup.yaml"})["total_memory_bytes"]
    # The card is read in GiB and compared in bytes, so a card a few percent above the
    # estimate is tight, not short: 1.05 x total in 1e9-byte GB would not fit.
    card = int(total * factor)
    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", lambda: {"memory_total_bytes": card})

    assert tool_profile({"config": "soup.yaml"})["fit"] == fit
    as_json = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert as_json.exit_code == 0, (as_json.output, repr(as_json.exception))
    assert json.loads(as_json.output)["fit"] == fit
    rendered = CliRunner().invoke(app, ["profile", "--config", "soup.yaml"])
    assert rendered.exit_code == 0, (rendered.output, repr(rendered.exception))
    assert line in " ".join(strip_ansi(rendered.output).split())


# --- the recommendations compare in bytes, as the fit verdict does ----------------


@pytest.mark.parametrize("total_gb", [2.0, 5.6, 7.5, 14.0, 21.5, 40.0, 70.0, 150.0])
def test_a_recommended_gpu_is_one_the_fit_verdict_says_fits(total_gb):
    from soup_cli.utils.profiler import GPU_MEMORY, recommend_gpu
    from soup_cli.utils.vram_estimate import fit_verdict

    listed = set(recommend_gpu(total_gb))
    for name, vram in GPU_MEMORY.items():
        fits = fit_verdict(total_gb * 1e9, vram * 1024**3) == "fits"
        assert (f"{name.upper()} ({vram} GB)" in listed) == fits, name


def test_a_6_gb_card_is_recommended_for_a_run_that_fits_it():
    from soup_cli.utils.profiler import recommend_gpu

    # 6 GiB is 6.44e9 bytes and 5.6e9 x 1.1 is 6.16e9: it fits with the margin.
    assert "RTX4050LAPTOP (6 GB)" in recommend_gpu(5.6)


def test_the_batch_recommendation_reads_the_card_in_gib():
    from soup_cli.utils.profiler import recommend_batch_size

    # 24 GiB is 25.8e9 bytes: 2.8e9 free beside 23e9, room for one more sample.
    assert recommend_batch_size(23.0, 24.0) == 2


def test_a_model_over_1000b_parameters_is_named_as_the_reason(workdir, monkeypatch):
    monkeypatch.setattr(model_shape, "_hub_lookup", lambda repo, **kw: (dict(TINY_LLAMA), None))
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 2 * 10**12)
    (workdir / "soup.yaml").write_text(
        _yaml("someone/huge", _RUNS["lora bf16"]), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "size (over 1,000B parameters)" in " ".join(strip_ansi(result.output).split())
    with pytest.raises(Exception, match=r"over 1,000B parameters"):
        tool_profile({"config": "soup.yaml"})


# --- the MCP tool stays offline ----------------------------------------------------


def test_the_mcp_tool_never_reaches_the_hub(workdir, monkeypatch):
    """A Hub model not in the cache: the CLI reads it, the MCP tool does not ask."""
    asked = []

    def lookup(repo, *, allow_network):
        asked.append(("config", allow_network))
        return (dict(TINY_LLAMA), None) if allow_network else (None, None)

    def count(repo, *, allow_network):
        asked.append(("count", allow_network))
        return 1_234_567 if allow_network else None

    monkeypatch.setattr(model_shape, "_hub_lookup", lookup)
    monkeypatch.setattr(model_shape, "_hub_param_count", count)
    (workdir / "soup.yaml").write_text(
        _yaml("someone/not-cached", _RUNS["lora bf16"]), encoding="utf-8"
    )

    mcp = tool_profile({"config": "soup.yaml"})
    assert asked and all(allow_network is False for _, allow_network in asked)
    assert mcp["shape_source"] == "guessed"
    assert any("No config.json could be read" in note for note in mcp["notes"])

    asked.clear()
    result = CliRunner().invoke(app, ["profile", "--config", "soup.yaml", "--json"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert json.loads(result.output)["shape_source"] == "config.json"
    assert ("config", True) in asked


def test_the_mcp_tool_still_reads_a_local_model(workdir):
    (workdir / "soup.yaml").write_text(
        _yaml(str(workdir / "tiny"), _RUNS["lora bf16"]), encoding="utf-8"
    )
    assert tool_profile({"config": "soup.yaml"})["shape_source"] == "config.json"
