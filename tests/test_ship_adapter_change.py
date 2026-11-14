"""soup ship --adapter prints how much of the base the adapter changes (#2762).

Display only: a warning line below ADAPTER_CHANGE_WARN_BELOW, never a different
verdict. The adapters and bases here are built from tensors; no model is loaded
(_resolve_generators is mocked).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.utils.ship_verdict import (
    ADAPTER_CHANGE_WARN_BELOW,
    FAILED_ADAPTER_NOOP,
    FAILED_TASK_WIN,
    AdapterChange,
    format_ship_rubric,
    verdict_to_dict,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.adapter_change import measure_adapter_change  # noqa: E402

runner = CliRunner()

KEY = "model.layers.0.self_attn.q_proj"
OUT, IN = 10, 100  # 1,000 elements, so k changed elements is a fraction of k / 1000
WARNING = "fewer than 1% of the elements"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _write_base(path: Path, dtype) -> None:
    """One target weight of ones, with a -0.0 in a row the update leaves at zero."""
    path.mkdir()
    weight = torch.ones((OUT, IN), dtype=dtype)
    weight[1, 0] = -0.0
    save_file({f"{KEY}.weight": weight}, str(path / "model.safetensors"))


def _write_adapter(path: Path, changed: int, *, config: dict | None = None) -> None:
    """r = 1, scaling 2. Row 0 of the update gets +0.5 on ``changed`` columns
    (representable next to 1.0 in bf16 and fp16) and +1e-4 on 20 others, which
    rounds back to 1.0 in both dtypes and must not count."""
    path.mkdir()
    lora_a = torch.zeros((1, IN))
    lora_a[0, :changed] = 0.25
    lora_a[0, IN - 20:] = 5e-5
    lora_b = torch.zeros((OUT, 1))
    lora_b[0, 0] = 1.0
    save_file(
        {
            f"base_model.model.{KEY}.lora_A.weight": lora_a,
            f"base_model.model.{KEY}.lora_B.weight": lora_b,
        },
        str(path / "adapter_model.safetensors"),
    )
    cfg = {"peft_type": "LORA", "r": 1, "lora_alpha": 2, "bias": "none"}
    cfg.update(config or {})
    (path / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")


# --------------------------------------------------------------------------
# The measurement and the boundary, from tensors, in bf16 and fp16
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    ("changed", "warned"),
    [(9, True), (10, False), (11, False)],
    ids=["just-below", "at-constant", "just-above"],
)
def test_fraction_and_warning_at_the_constant(tmp_path, dtype, changed, warned):
    _write_base(tmp_path / "base", dtype)
    _write_adapter(tmp_path / "lora", changed)

    change = measure_adapter_change(tmp_path / "lora", tmp_path / "base")

    assert change.measured, change.reason
    # Only the +0.5 elements count: the +1e-4 ones round back to 1.0, and the
    # -0.0 that becomes +0.0 is the same value.
    assert (change.changed, change.elements, change.modules) == (changed, OUT * IN, 1)
    assert change.base_dtypes == (str(dtype).replace("torch.", ""),)
    assert change.fraction == changed / (OUT * IN)
    assert change.warning is warned
    assert (change.fraction < ADAPTER_CHANGE_WARN_BELOW) is warned


def test_constant_is_one_percent():
    assert ADAPTER_CHANGE_WARN_BELOW == 1e-2


def test_all_zero_update_changes_nothing_even_with_negative_zero(tmp_path):
    _write_base(tmp_path / "base", torch.bfloat16)
    _write_adapter(tmp_path / "lora", 0)
    lora = tmp_path / "lora" / "adapter_model.safetensors"
    from safetensors.torch import load_file

    weights = load_file(str(lora))
    weights[f"base_model.model.{KEY}.lora_A.weight"].zero_()
    save_file(weights, str(lora))

    change = measure_adapter_change(tmp_path / "lora", tmp_path / "base")

    assert change.measured and change.changed == 0 and change.fraction == 0.0


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"use_dora": True}, "use_dora"),
        ({"bias": "all"}, "bias"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
        ({"peft_type": "VERA"}, "peft_type"),
    ],
)
def test_adapters_the_reader_refuses_are_not_measured(tmp_path, config, expected):
    _write_base(tmp_path / "base", torch.bfloat16)
    _write_adapter(tmp_path / "lora", 9, config=config)

    change = measure_adapter_change(tmp_path / "lora", tmp_path / "base")

    assert not change.measured
    assert expected in change.reason
    assert change.fraction is None and change.warning is False


def test_adapter_on_a_module_the_base_lacks_is_not_measured(tmp_path):
    _write_base(tmp_path / "base", torch.bfloat16)
    _write_adapter(tmp_path / "lora", 9)
    (tmp_path / "base" / "model.safetensors").unlink()
    save_file({"other.weight": torch.ones((OUT, IN), dtype=torch.bfloat16)},
              str(tmp_path / "base" / "model.safetensors"))

    change = measure_adapter_change(tmp_path / "lora", tmp_path / "base")

    assert not change.measured and "match no base tensor" in change.reason


def test_verdict_dict_and_rubric_carry_the_number():
    from soup_cli.utils.ship_verdict import DECISION_SHIP, ShipVerdict, TaskWin

    verdict = ShipVerdict(
        decision=DECISION_SHIP,
        task_win=TaskWin(mode="metric", base=0.1, tuned=0.2, won=True),
        benchmark_deltas=(),
        failed_rule=None,
        forgetting_threshold=0.05,
        soup_version="x",
        adapter_change=AdapterChange(
            changed=9, elements=1000, modules=1, base_dtypes=("bfloat16",)
        ),
    )
    data = verdict_to_dict(verdict)["adapter_change"]
    assert data == {
        "measured": True, "fraction": 0.009, "changed": 9, "elements": 1000,
        "modules": 1, "base_dtypes": ["bfloat16"], "warn_below": 0.01,
        "warning": True, "reason": None,
    }
    rubric = format_ship_rubric(verdict)
    assert "fraction 9.000e-03" in rubric and WARNING in rubric
    assert verdict_to_dict(
        ShipVerdict(**{**verdict.__dict__, "adapter_change": None})
    )["adapter_change"] is None


# --------------------------------------------------------------------------
# Through `soup ship`: printed, written, and never a different verdict
# --------------------------------------------------------------------------

def _run(monkeypatch, changed, *, base="base", config=None, extra=()):
    loads = []

    def _fake_resolve(base, tuned, adapter, device, quantization=None):
        loads.append(adapter)
        return (lambda p: "hi", lambda p: "hi")

    monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
    with runner.isolated_filesystem():
        _write_base(Path("base"), torch.bfloat16)
        _write_adapter(Path("lora"), changed, config=config)
        Path("task.jsonl").write_text(
            json.dumps({"prompt": "say hi", "expected": "hi", "scoring": "contains"}) + "\n",
            encoding="utf-8",
        )
        res = runner.invoke(
            ship_cmd.app,
            ["--base", base, "--adapter", "lora", "--task-eval", "task.jsonl",
             "--device", "cpu", "--output", "out.json", *extra],
        )
        out = (
            json.loads(Path("out.json").read_text(encoding="utf-8"))
            if Path("out.json").exists() else None
        )
    return res, loads, out


@pytest.mark.parametrize(("changed", "warned"), [(9, True), (10, False)])
def test_ship_prints_the_fraction_and_keeps_the_verdict(monkeypatch, changed, warned):
    res, loads, out = _run(monkeypatch, changed)

    assert loads == ["lora"], (res.output, repr(res.exception))
    # Same verdict, failed rule and exit code as without the measurement.
    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert out["adapter_change"]["fraction"] == changed / 1000
    assert out["adapter_change"]["warning"] is warned
    plain = _plain(res.output)
    assert f"changes {changed} of 1,000 elements" in plain
    assert (WARNING in plain) is warned
    if warned:
        assert "does not change the verdict" in plain


def test_ship_says_not_measured_for_a_dora_adapter(monkeypatch):
    res, loads, out = _run(monkeypatch, 9, config={"use_dora": True})

    assert loads == ["lora"] and res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert out["adapter_change"]["measured"] is False
    assert "use_dora" in out["adapter_change"]["reason"]
    plain = _plain(res.output)
    assert "Adapter update: not measured" in plain and WARNING not in plain


def test_hub_base_not_in_cache_is_not_measured_and_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty-cache"))

    res, loads, out = _run(monkeypatch, 9, base="org/never-downloaded")

    assert res.exit_code == 2, (res.output, repr(res.exception))
    assert out["adapter_change"]["measured"] is False
    assert "local Hugging Face cache" in out["adapter_change"]["reason"]
    assert not (tmp_path / "empty-cache").exists()


def test_update_below_resolution_is_measured_as_zero_but_not_refused(monkeypatch):
    # lora_B is not zero, but every contribution rounds away in the base dtype.
    res, loads, out = _run(monkeypatch, 0)

    assert loads == ["lora"] and res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_TASK_WIN
    assert out["adapter_change"]["fraction"] == 0.0
    assert out["adapter_change"]["warning"] is True


def test_all_zero_check_still_runs_first_and_measures_nothing(monkeypatch):
    monkeypatch.setattr(ship_cmd, "_adapter_is_noop", lambda adapter: True)

    res, loads, out = _run(monkeypatch, 9)

    assert loads == [] and res.exit_code == 2, (res.output, repr(res.exception))
    assert out["failed_rule"] == FAILED_ADAPTER_NOOP
    assert out["adapter_change"] is None
