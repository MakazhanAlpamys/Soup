"""soup ship --base-adapter: compare against base + a production adapter.

The continual-learning case: the candidate (base + new adapter) is judged against the
version already in production (base + previous adapter), both trained from one base.
No GPU and no downloads: ``live_eval.make_generator`` is replaced by a recorder, so the
real ``_resolve_generators`` / ``_verdict_live`` wiring runs and the test sees exactly
which adapter each side was built with.
"""

from __future__ import annotations

import collections
import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import typer
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import ship as ship_cmd
from soup_cli.utils import live_eval
from soup_cli.utils.ship_verdict import (
    DECISION_SHIP,
    ShipVerdict,
    TaskWin,
    render_ship_panel,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()

Q_A = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
Q_B = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
TASK_PROMPT = "task"

HELP_TEXT = (
    "LoRA adapter applied to the base side (before = base + this adapter), e.g. the "
    "current production version. Live run with --adapter only."
)


def _flat(text: str) -> str:
    """ANSI-free, box-drawing-free, whitespace-collapsed CLI output.

    Rich puts colour escapes inside flag names on Linux/macOS CI, and wraps help text
    between table borders at 80 columns; Windows hides both.
    """
    plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", strip_ansi(text))
    return " ".join(re.sub(r"[^\x20-\x7e]+", " ", plain).split())


def _write_adapter(path: Path, b_value: float) -> None:
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path.mkdir()
    save_file(
        {
            Q_A: np.ones((2, 4), dtype=np.float32),
            Q_B: np.full((4, 2), b_value, dtype=np.float32),
        },
        str(path / "adapter_model.safetensors"),
    )
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}), encoding="utf-8"
    )


class _FakeLive:
    """Stand-in for ``live_eval.make_generator`` that records how each side was built.

    A generator is labelled by the adapter it was built with (``bare`` for none). Only the
    ``cand`` adapter has "learned" the task, so ``cand`` vs ``prod`` is a SHIP and every
    other pairing is not.
    """

    def __init__(self) -> None:
        self.calls: list = []
        self.usage: collections.Counter = collections.Counter()

    def make_generator(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        label = kwargs.get("adapter") or "bare"

        def generate(prompt: str) -> str:
            self.usage[label] += 1
            if prompt == TASK_PROMPT:
                return "answer" if label == "cand" else "wrong"
            return "B"

        return generate

    def adapters(self) -> list:
        return [kwargs.get("adapter") for _model, kwargs in self.calls]


def _install(monkeypatch) -> _FakeLive:
    fake = _FakeLive()
    monkeypatch.setattr(live_eval, "make_generator", fake.make_generator)
    return fake


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_adapter(tmp_path / "prod", 0.5)
    _write_adapter(tmp_path / "cand", 0.7)
    _write_adapter(tmp_path / "zero", 0.0)
    (tmp_path / "task.jsonl").write_text(
        json.dumps({"prompt": TASK_PROMPT, "expected": "answer"}) + "\n", encoding="utf-8"
    )
    fake = _install(monkeypatch)
    return tmp_path, fake


def _ship(*args: str):
    return runner.invoke(app, ["ship", *args])


def _live(*extra: str):
    """The common live invocation: base ``m``, task file, CPU."""
    return _ship("--base", "m", "--task-eval", "task.jsonl", "--device", "cpu", *extra)


def _verdict_json(root: Path) -> dict:
    return json.loads((root / "out.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Wiring: which adapter each side is built with
# ---------------------------------------------------------------------------


def test_base_side_is_built_with_the_base_adapter_and_tuned_side_with_the_adapter(work):
    root, fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand", "--output", "out.json")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert [model for model, _kwargs in fake.calls] == ["m", "m"]
    # base side first, then the tuned side
    assert fake.adapters() == ["prod", "cand"], fake.calls
    verdict = _verdict_json(root)
    assert verdict["decision"] == DECISION_SHIP
    # each side was scored by its OWN generator: prod missed the task, cand got it
    assert verdict["task_win"]["base"] == 0.0
    assert verdict["task_win"]["tuned"] == 1.0


def test_without_the_flag_the_base_side_gets_no_adapter_keyword(work):
    _root, fake = work
    res = _live("--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert "adapter" not in fake.calls[0][1], fake.calls[0]
    assert fake.calls[1][1]["adapter"] == "cand"


def test_noise_floor_repeats_are_served_by_the_base_adapter_generator(work, monkeypatch):
    _root, plain = work
    res = _live("--base-adapter", "prod", "--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))

    floored = _install(monkeypatch)
    res = _live("--base-adapter", "prod", "--adapter", "cand", "--noise-floor", "2")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    # no generator of its own for the repeats, and none of them touched a bare base
    assert floored.adapters() == ["prod", "cand"], floored.calls
    assert "bare" not in floored.usage
    assert floored.usage["cand"] == plain.usage["cand"]
    assert floored.usage["prod"] > plain.usage["prod"]


def test_the_base_adapter_is_stated_in_the_load_message(work):
    res = _live("--base-adapter", "prod", "--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert "base side = base + --base-adapter prod" in _flat(res.output)


# ---------------------------------------------------------------------------
# The all-zero shortcut means "tuned == base", which --base-adapter breaks
# ---------------------------------------------------------------------------


def test_all_zero_adapter_still_short_circuits_without_the_flag(work):
    root, fake = work
    res = _live("--adapter", "zero", "--output", "out.json")
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert fake.calls == []
    assert _verdict_json(root)["failed_rule"] == "adapter_noop"


def test_all_zero_adapter_shortcut_is_skipped_with_a_base_adapter(work):
    root, fake = work
    res = _live("--base-adapter", "prod", "--adapter", "zero", "--output", "out.json")
    assert res.exit_code == 2, (res.output, repr(res.exception))
    # tuned = the bare base here, before = base + prod: that is a real comparison
    assert fake.adapters() == ["prod", "zero"], fake.calls
    verdict = _verdict_json(root)
    assert verdict["failed_rule"] == "task_win"
    assert verdict["failed_rule"] != "adapter_noop"


# ---------------------------------------------------------------------------
# Refusals: each one is a usage error (exit 3) that names both flags
# ---------------------------------------------------------------------------

_LIVE_BASE = ["--base", "m", "--task-eval", "task.jsonl", "--device", "cpu"]

_REFUSALS = {
    "with_tuned_instead_of_adapter": (
        [*_LIVE_BASE, "--base-adapter", "prod", "--tuned", "t"],
        "--base-adapter needs --adapter",
    ),
    "alone": (
        ["--base-adapter", "prod"],
        "--base-adapter needs --adapter",
    ),
    "with_evidence": (
        ["--base-adapter", "prod", "--adapter", "cand", "--evidence", "ev.json"],
        "--base-adapter cannot be used with --evidence",
    ),
    "with_require_provenance": (
        ["--base-adapter", "prod", "--adapter", "cand", "--require-provenance"],
        "--base-adapter cannot be used with --require-provenance",
    ),
    "with_emit_evidence": (
        [*_LIVE_BASE, "--base-adapter", "prod", "--adapter", "cand",
         "--emit-evidence", "emitted.json"],
        "--base-adapter cannot be used with --emit-evidence",
    ),
    "with_baseline": (
        [*_LIVE_BASE, "--base-adapter", "prod", "--adapter", "cand",
         "--baseline", "base.json"],
        "--base-adapter cannot be used with --baseline",
    ),
    "with_lm_eval_benchmark": (
        [*_LIVE_BASE, "--base-adapter", "prod", "--adapter", "cand",
         "--general-suite", "mini_mmlu,hellaswag"],
        "--base-adapter cannot score non-bundled benchmark(s) hellaswag",
    ),
    "same_path_as_adapter": (
        [*_LIVE_BASE, "--base-adapter", "cand", "--adapter", "./cand"],
        "--base-adapter and --adapter are the same path",
    ),
    "same_path_through_dotdot": (
        [*_LIVE_BASE, "--base-adapter", "prod/../cand", "--adapter", "cand"],
        "--base-adapter and --adapter are the same path",
    ),
    "missing_directory": (
        [*_LIVE_BASE, "--base-adapter", "nope", "--adapter", "cand"],
        "--base-adapter must be an existing directory",
    ),
    "a_file_not_a_directory": (
        [*_LIVE_BASE, "--base-adapter", "task.jsonl", "--adapter", "cand"],
        "--base-adapter must be an existing directory",
    ),
    "empty_value": (
        [*_LIVE_BASE, "--base-adapter", "", "--adapter", "cand"],
        "--base-adapter must be an existing directory",
    ),
}


@pytest.mark.parametrize("argv, message", list(_REFUSALS.values()), ids=list(_REFUSALS))
def test_refusal_is_a_usage_error_naming_the_flag_and_loads_nothing(work, argv, message):
    root, fake = work
    res = _ship(*argv)
    assert res.exit_code == 3, (res.output, repr(res.exception))
    assert message in _flat(res.output), res.output
    assert fake.calls == []
    assert not (root / "emitted.json").exists()


@pytest.mark.requires_symlink
def test_a_symlink_to_the_adapter_is_the_same_path(work):
    root, fake = work
    (root / "link").symlink_to(root / "cand", target_is_directory=True)
    res = _live("--base-adapter", "link", "--adapter", "cand")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    assert "--base-adapter and --adapter are the same path" in _flat(res.output)
    assert fake.calls == []


def test_refusal_covers_a_non_bundled_benchmark_set_only(work):
    """Bundled names alone run; the refusal is about lm-eval routing, not the flag."""
    _root, fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand",
                "--general-suite", "mini_mmlu,mini_arithmetic")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert fake.adapters() == ["prod", "cand"]


def test_lm_eval_names_from_a_config_are_refused_too(work):
    """The suite can come from --config; the CLI-level check alone would miss it."""
    root, fake = work
    (root / "train.jsonl").write_text('{"text":"train"}\n', encoding="utf-8")
    (root / "soup.yaml").write_text(
        "base: owner/base\ndata: {train: train.jsonl}\n"
        "training: {quantization: none}\neval:\n  ship:\n"
        "    task_eval: task.jsonl\n    general_suite: hellaswag\n",
        encoding="utf-8",
    )
    res = _ship("--base", "m", "--base-adapter", "prod", "--adapter", "cand",
                "--config", "soup.yaml", "--device", "cpu")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    assert "--base-adapter cannot score non-bundled benchmark(s) hellaswag" in _flat(res.output)
    assert fake.calls == []


def test_verdict_live_refuses_a_baseline_even_when_called_without_the_cli(work):
    """``--config`` can supply the baseline, and the SDK skips ``ship()`` altogether."""
    root, fake = work
    with pytest.raises(typer.Exit) as excinfo:
        ship_cmd._verdict_live(
            base="m", tuned=None, adapter="cand", base_adapter="prod",
            task_eval="task.jsonl", task_mode="metric", judge_model=None,
            general_suite=None, baseline_spec="registry://abc", device="cpu",
            forgetting_threshold=0.05,
        )
    assert excinfo.value.exit_code == 3
    assert fake.calls == []


def test_verdict_live_refuses_a_base_adapter_without_an_adapter(work):
    _root, fake = work
    with pytest.raises(typer.Exit) as excinfo:
        ship_cmd._verdict_live(
            base="m", tuned="t", adapter=None, base_adapter="prod",
            task_eval="task.jsonl", task_mode="metric", judge_model=None,
            general_suite=None, baseline_spec=None, device="cpu",
            forgetting_threshold=0.05,
        )
    assert excinfo.value.exit_code == 3
    assert fake.calls == []


def test_refusal_message_does_not_echo_terminal_escapes(work):
    _root, fake = work
    res = _live("--base-adapter", "bad\x1b[2Jname", "--adapter", "cand")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    assert "\x1b[2J" not in res.output
    assert "--base-adapter must be an existing directory" in _flat(res.output)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# The verdict records the before side honestly
# ---------------------------------------------------------------------------


def test_output_json_records_the_base_adapter(work):
    root, _fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand", "--output", "out.json")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert _verdict_json(root)["base_adapter"] == "prod"


def test_output_json_base_adapter_is_null_without_the_flag(work):
    root, _fake = work
    res = _live("--adapter", "cand", "--output", "out.json")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    verdict = _verdict_json(root)
    assert "base_adapter" in verdict
    assert verdict["base_adapter"] is None


def test_existing_output_keys_are_unchanged(work):
    root, _fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand", "--output", "out.json")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    verdict = _verdict_json(root)
    existing = {
        "decision", "task_win", "benchmark_deltas", "failed_rule",
        "forgetting_threshold", "soup_version", "noise_floor", "numerics",
    }
    assert existing <= set(verdict)
    assert set(verdict["task_win"]) == {"mode", "base", "tuned", "won"}
    assert verdict["numerics"] == "float32"


def test_panel_names_the_base_adapter(work):
    _root, _fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert "Base adapter: prod" in _flat(res.output)


def test_panel_has_no_base_adapter_line_without_the_flag(work):
    _root, _fake = work
    res = _live("--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert "Base adapter" not in _flat(res.output)


def test_panel_escapes_markup_and_control_bytes_in_the_base_adapter():
    verdict = ShipVerdict(
        decision=DECISION_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule=None,
        forgetting_threshold=0.05,
        soup_version="0",
        base_adapter="a[red]b\x1b[2Jc",
    )
    buffer = StringIO()
    Console(file=buffer, width=100, force_terminal=False).print(render_ship_panel(verdict))
    out = buffer.getvalue()
    assert "Base adapter: a[red]b" in out
    assert "\x1b" not in out


def test_a_verdict_defaults_to_no_base_adapter():
    verdict = ShipVerdict(
        decision=DECISION_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.9, won=True),
        benchmark_deltas=(),
        failed_rule=None,
        forgetting_threshold=0.05,
        soup_version="0",
    )
    assert verdict.base_adapter is None


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_lists_the_flag():
    res = _ship("--help")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    flat = _flat(res.output)
    assert "--base-adapter" in flat
    assert HELP_TEXT in flat


# ---------------------------------------------------------------------------
# The weight screen: pickle weights are refused before PEFT reads the directory
# ---------------------------------------------------------------------------


def _pickle_adapter(path: Path) -> None:
    """An adapter directory whose weights are a ``.bin`` pickle (code runs when it loads)."""
    path.mkdir()
    (path / "adapter_model.bin").write_bytes(b"\x80\x04N.")
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4}), encoding="utf-8"
    )


def test_a_base_adapter_with_pickle_weights_is_refused_by_name(work):
    root, fake = work
    _pickle_adapter(root / "pickled")
    res = _live("--base-adapter", "pickled", "--adapter", "cand")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    flat = _flat(res.output)
    assert "--base-adapter" in flat and "adapter_model.bin" in flat
    assert "unsafe weight file" in flat
    assert fake.calls == []


def test_an_adapter_with_pickle_weights_is_refused_with_a_base_adapter(work):
    root, fake = work
    _pickle_adapter(root / "pickled")
    res = _live("--base-adapter", "prod", "--adapter", "pickled")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    flat = _flat(res.output)
    assert "--adapter" in flat and "adapter_model.bin" in flat
    assert fake.calls == []


def test_an_adapter_with_pickle_weights_is_refused_without_a_base_adapter(work):
    root, fake = work
    _pickle_adapter(root / "pickled")
    res = _live("--adapter", "pickled")
    assert res.exit_code == 3, (res.output, repr(res.exception))
    assert "adapter_model.bin" in _flat(res.output)
    assert fake.calls == []


def test_safetensors_adapters_still_pass_the_screen(work):
    _root, fake = work
    res = _live("--base-adapter", "prod", "--adapter", "cand")
    assert res.exit_code == 0, (res.output, repr(res.exception))
    assert fake.adapters() == ["prod", "cand"]
