"""``soup probe`` shows bases, paths, adapter names and error text as typed (#2489).

Every line of ``soup probe pack``, ``sleeper``, ``truth``, ``harm``,
``interference`` and ``sae-diff`` that carries text from outside the program --
a typed base or path, an adapter name from a losses JSON, an SAE id, a probe
description, an exception message -- keeps ``:name:`` aliases and backslashes
before brackets, and drops control and bidi characters. Fixed messages keep
their colours.
"""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import probe as probe_cmd
from soup_cli.utils import (
    harm_probe,
    interference,
    probe_kernel,
    probe_pack,
    sae_diff,
    sleeper_probe,
    truth_probe,
)
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

CONTROLS = "\x9b\u202e"
VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x" + CONTROLS + "n", "x\x1b\x07n"]
#: Values that can also be part of a file name on every CI platform.
FILE_VALUES = [
    "plain",
    "x[red]n",
    "x" + CONTROLS + "n",
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(sys.platform == "win32", reason="backslash in a file name"),
    ),
    pytest.param("n:rocket:", marks=skip_colon_on_windows),
]

#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")


def shown(value: str) -> str:
    for char in DROPPED:
        value = value.replace(char, "")
    return value


def plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def assert_shown(result, text: str) -> None:
    """``text`` is in the output, and no dropped character is anywhere in it."""
    assert " ".join(text.split()) in plain(result), result.output
    output = strip_ansi(result.output)
    assert [char for char in DROPPED if char in output] == []


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(probe_cmd, "console", Console(width=1000, height=100))
    Path("evidence.json").write_text(json.dumps({"activations": [[0.0, 0.0]]}), "utf-8")


def _probe(*args: str):
    return runner.invoke(app, ["probe", *args])


def _fail_with(exc_type, message):
    def fail(*_args, **_kwargs):
        raise exc_type(message)

    return fail


# ---------------------------------------------------------------------------
# soup probe pack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_pack_list_shows_each_base_as_typed(monkeypatch, value):
    monkeypatch.setattr(probe_pack, "list_probe_bases", lambda: (f"base-{value}",))

    result = _probe("pack", "--list")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert_shown(result, f"base-{shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_pack_refusal_names_the_base_as_typed(value):
    result = _probe("pack", f"base-{value}")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no probe pack for base {f'base-{value}'!r}")


@pytest.mark.parametrize("value", VALUES)
def test_pack_report_shows_the_base_and_descriptions_as_typed(monkeypatch, value):
    entry = probe_pack.ProbeEntry(
        name=f"sleeper:{value}", kind="sleeper", hidden_dim=4, description=f"about {value}"
    )
    pack = probe_pack.ProbePack(
        base=f"base-{value}",
        probes=(
            entry,
            probe_pack.ProbeEntry(name="truth:b", kind="truth", hidden_dim=4, description="t"),
            probe_pack.ProbeEntry(name="harm:b", kind="harm", hidden_dim=4, description="h"),
        ),
        soup_version="0.0.0",
    )
    monkeypatch.setattr(probe_pack, "get_probe_pack", lambda base: pack)

    result = _probe("pack", "anything")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert_shown(result, f"# Probe pack: base-{shown(value)}")
    assert_shown(result, f"| `sleeper:{shown(value)}` | sleeper | 4 | about {shown(value)} |")


@pytest.mark.parametrize("value", VALUES)
def test_pack_write_failure_is_named_as_typed(monkeypatch, value):
    monkeypatch.setattr(probe_cmd, "atomic_write_text", _fail_with(ValueError, f"no {value}"))

    result = _probe("pack", probe_pack.list_probe_bases()[0], "--output", "pack.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"Cannot write --output: no {shown(value)}")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_pack_receipt_names_the_output_as_typed(value):
    output = f"pack-{value}.json"

    result = _probe("pack", probe_pack.list_probe_bases()[0], "--output", output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path(output).is_file()
    assert_shown(result, f"Wrote probe pack → {shown(output)}")


# ---------------------------------------------------------------------------
# soup probe sleeper / truth / harm (the shared shape)
# ---------------------------------------------------------------------------

KINDS = {
    "sleeper": dict(
        module=sleeper_probe,
        validate="validate_base_for_probe",
        bundled="BUNDLED_PROBES",
        run="run_sleeper_probe",
        weights_module=sleeper_probe,
        spec=sleeper_probe.SleeperProbeSpec,
        result=lambda base: sleeper_probe.SleeperProbeResult(
            base=base, num_tokens=1, defection_rate=0.0, max_score=0.0, verdict="OK"
        ),
        heading="Sleeper probe",
    ),
    "truth": dict(
        module=truth_probe,
        validate="validate_base_for_truth",
        bundled="BUNDLED_TRUTH_PROBES",
        run="run_truth_probe",
        weights_module=probe_kernel,
        spec=probe_kernel.ProbeSpec,
        result=lambda base: probe_kernel.ProbeResult(
            kind="truth", base=base, num_tokens=1, flag_rate=0.0, max_score=0.0, verdict="OK"
        ),
        heading="Honesty probe",
    ),
    "harm": dict(
        module=harm_probe,
        validate="validate_base_for_harm",
        bundled="BUNDLED_HARM_PROBES",
        run="run_harm_probe",
        weights_module=probe_kernel,
        spec=probe_kernel.ProbeSpec,
        result=lambda base: probe_kernel.ProbeResult(
            kind="harm", base=base, num_tokens=1, flag_rate=0.0, max_score=0.0, verdict="OK"
        ),
        heading="Misuse probe",
    ),
}


def _bundle(monkeypatch, kind: str, base: str, description: str = "d") -> None:
    """Make ``base`` a bundled base of ``kind`` with ``description``."""
    k = KINDS[kind]
    spec = k["spec"](base=base, hidden_dim=2, threshold=1.0, description=description)
    monkeypatch.setattr(k["module"], k["validate"], lambda name: base)
    monkeypatch.setattr(k["module"], k["bundled"], {base: spec})


def _real_base(kind: str) -> str:
    return sorted(getattr(KINDS[kind]["module"], KINDS[kind]["bundled"]))[0]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_an_unknown_base_is_named_as_typed(kind, value):
    result = _probe(kind, f"base-{value}")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no bundled probe for base {f'base-{value}'!r}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_a_weights_refusal_is_shown_as_typed(monkeypatch, kind, value):
    monkeypatch.setattr(
        KINDS[kind]["weights_module"], "load_probe_weights", _fail_with(ValueError, f"bad {value}")
    )

    result = _probe(kind, "any", "--weights", "w.npz")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"--weights: bad {shown(value)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_the_no_evidence_panel_shows_base_and_description_as_typed(monkeypatch, kind, value):
    _bundle(monkeypatch, kind, f"base-{value}", description=f"about {value}")

    result = _probe(kind, "any")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert_shown(result, f"Base: base-{shown(value)}")
    assert_shown(result, f"Description: about {shown(value)}")
    assert_shown(result, "Pass --evidence <activations.json> to run the probe.")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_neutral_report_receipt_names_the_output_as_typed(kind, value):
    output = f"neutral-{value}.json"

    result = _probe(kind, _real_base(kind), "--output", output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path(output).is_file()
    assert_shown(result, f"Wrote neutral report → {shown(output)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_a_neutral_report_write_failure_is_shown_as_typed(monkeypatch, kind, value):
    monkeypatch.setattr(probe_cmd, "atomic_write_text", _fail_with(ValueError, f"no {value}"))

    result = _probe(kind, _real_base(kind), "--output", "neutral.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no {shown(value)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_an_evidence_refusal_is_shown_as_typed(monkeypatch, kind, value):
    import typer

    monkeypatch.setattr(
        probe_cmd, "_read_json_evidence", _fail_with(typer.BadParameter, f"--evidence: {value}")
    )

    result = _probe(kind, _real_base(kind), "--evidence", "evidence.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"--evidence: {shown(value)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_invalid_activations_are_shown_as_typed(kind, value):
    Path("bad.json").write_text(json.dumps({"activations": [[value]]}), "utf-8")

    result = _probe(kind, _real_base(kind), "--evidence", "bad.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(
        result, f"--evidence: invalid activations — could not convert string to float: {value!r}"
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_a_probe_failure_is_shown_as_typed(monkeypatch, kind, value):
    _bundle(monkeypatch, kind, "base")
    monkeypatch.setattr(KINDS[kind]["module"], KINDS[kind]["run"], _fail_with(ValueError, value))

    result = _probe(kind, "base", "--evidence", "evidence.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, shown(value))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_the_report_names_the_base_as_typed(monkeypatch, kind, value):
    base = f"base-{value}"
    _bundle(monkeypatch, kind, base)
    monkeypatch.setattr(
        KINDS[kind]["module"], KINDS[kind]["run"], lambda *a, **k: KINDS[kind]["result"](base)
    )

    result = _probe(kind, "any", "--evidence", "evidence.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert_shown(result, f"# {KINDS[kind]['heading']}: base-{shown(value)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_result_receipt_names_the_output_as_typed(monkeypatch, kind, value):
    _bundle(monkeypatch, kind, "base")
    monkeypatch.setattr(
        KINDS[kind]["module"], KINDS[kind]["run"], lambda *a, **k: KINDS[kind]["result"]("base")
    )
    output = f"result-{value}.json"

    result = _probe(kind, "base", "--evidence", "evidence.json", "--output", output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path(output).is_file()
    assert_shown(result, f"Wrote probe result → {shown(output)}")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", VALUES)
def test_a_result_write_failure_is_shown_as_typed(monkeypatch, kind, value):
    _bundle(monkeypatch, kind, "base")
    monkeypatch.setattr(
        KINDS[kind]["module"], KINDS[kind]["run"], lambda *a, **k: KINDS[kind]["result"]("base")
    )
    monkeypatch.setattr(probe_cmd, "atomic_write_text", _fail_with(ValueError, f"no {value}"))

    result = _probe(kind, "base", "--evidence", "evidence.json", "--output", "result.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no {shown(value)}")


# ---------------------------------------------------------------------------
# soup probe interference
# ---------------------------------------------------------------------------


def _losses(name: str) -> None:
    payload = {
        "adapters": [name, "b"],
        "losses": {f"{name}|{name}": 1.0, f"{name}|b": 1.05, "b|b": 1.0, f"b|{name}": 1.08},
    }
    Path("losses.json").write_text(json.dumps(payload), "utf-8")


@pytest.mark.parametrize("value", [*VALUES, "a]b", "x\\]n"])
def test_the_matrix_names_adapters_as_typed(value):
    name = f"ad-{value}"
    _losses(name)

    result = _probe("interference", "losses.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert_shown(result, f"| A \\ B | {shown(name)} | b |")
    assert_shown(result, f"| {shown(name)} | +0.000 (OK) | +0.050 (MINOR) |")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_matrix_receipt_names_the_output_as_typed(value):
    _losses("a")
    output = f"matrix-{value}.json"

    result = _probe("interference", "losses.json", "--output", output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path(output).is_file()
    assert_shown(result, f"Wrote matrix → {shown(output)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_matrix_write_failure_is_shown_as_typed(monkeypatch, value):
    _losses("a")
    monkeypatch.setattr(probe_cmd, "atomic_write_text", _fail_with(ValueError, f"no {value}"))

    result = _probe("interference", "losses.json", "--output", "matrix.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_losses_refusal_is_shown_as_typed(monkeypatch, value):
    import typer

    monkeypatch.setattr(
        probe_cmd, "_read_json_evidence", _fail_with(typer.BadParameter, f"--losses: {value}")
    )

    result = _probe("interference", "losses.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"--losses: {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_losses_key_without_a_bar_is_named_as_typed(value):
    payload = {"adapters": ["a", "b"], "losses": {f"k-{value}": 1.0}}
    Path("losses.json").write_text(json.dumps(payload), "utf-8")

    result = _probe("interference", "losses.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"Invalid losses key {f'k-{value}'!r}; expected 'a|b' shape.")


@pytest.mark.parametrize("value", VALUES)
def test_a_non_numeric_loss_is_named_as_typed(value):
    payload = {"adapters": ["a", "b"], "losses": {f"a|{value}": "x"}}
    Path("losses.json").write_text(json.dumps(payload), "utf-8")

    result = _probe("interference", "losses.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"losses[{f'a|{value}'!r}] must be numeric, got str")


@pytest.mark.parametrize("value", VALUES)
def test_a_matrix_build_failure_is_shown_as_typed(monkeypatch, value):
    _losses("a")
    monkeypatch.setattr(interference, "build_interference_matrix", _fail_with(ValueError, value))

    result = _probe("interference", "losses.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, shown(value))


def _measure(*extra: str):
    Path("suite.jsonl").write_text('{"prompt": "p", "response": "r"}\n', "utf-8")
    return _probe(
        "interference", "--measure", "suite.jsonl", "--adapter", "a=a", "--adapter", "b=b", *extra
    )


@pytest.mark.parametrize("value", VALUES)
def test_an_adapter_spec_refusal_is_named_as_typed(value):
    result = _probe(
        "interference", "--measure", "suite.jsonl", "--base-model", "m", "--adapter", f"s-{value}"
    )

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"--adapter {f's-{value}'!r}: expected name=path")


@pytest.mark.parametrize("value", VALUES)
def test_an_eval_suite_refusal_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(probe_cmd, "_load_jsonl_rows", _fail_with(ValueError, f"bad {value}"))

    result = _measure("--base-model", "m")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"--measure: bad {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_the_measure_line_and_its_failure_name_the_base_as_typed(monkeypatch, value):
    from soup_cli.utils import interference_live

    monkeypatch.setattr(
        interference_live, "measure_interference_losses", _fail_with(RuntimeError, f"x {value}")
    )

    result = _measure("--base-model", f"m-{value}")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(
        result, f"--measure: loading 2 adapters on m-{shown(value)} and measuring 1 eval rows ..."
    )
    assert_shown(result, f"--measure failed: x {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_measured_matrix_build_failure_is_shown_as_typed(monkeypatch, value):
    from soup_cli.utils import interference_live

    monkeypatch.setattr(interference_live, "measure_interference_losses", lambda *a, **k: {})
    monkeypatch.setattr(interference, "build_interference_matrix", _fail_with(ValueError, value))

    result = _measure("--base-model", "m")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, shown(value))


# ---------------------------------------------------------------------------
# soup probe sae-diff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_the_auto_download_line_and_refusal_name_the_sae_as_typed(monkeypatch, value):
    monkeypatch.setattr(sae_diff, "download_sae", _fail_with(ValueError, f"no {value}"))

    result = _probe("sae-diff", f"sae-{value}", "a.json", "b.json", "--auto-download")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(
        result, f"--auto-download: fetching SAE sae-{shown(value)} into ~/.soup/sae-cache/ ..."
    )
    assert_shown(result, f"no {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_missing_sae_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(sae_diff, "load_sae_weights", _fail_with(FileNotFoundError, f"no {value}"))

    result = _probe("sae-diff", "sae.safetensors", "a.json", "b.json")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert_shown(result, f"no {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_an_unreadable_sae_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(sae_diff, "load_sae_weights", _fail_with(ValueError, f"bad {value}"))

    result = _probe("sae-diff", "sae.safetensors", "a.json", "b.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"bad {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_an_activations_file_refusal_is_shown_as_typed(monkeypatch, value):
    import typer

    monkeypatch.setattr(sae_diff, "load_sae_weights", lambda path: object())
    monkeypatch.setattr(
        probe_cmd, "_read_json_evidence", _fail_with(typer.BadParameter, f"pre_acts: {value}")
    )

    result = _probe("sae-diff", "sae.safetensors", "a.json", "b.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"pre_acts: {shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_invalid_sae_activations_are_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(sae_diff, "load_sae_weights", lambda path: object())
    Path("bad.json").write_text(json.dumps({"activations": [[value]]}), "utf-8")

    result = _probe("sae-diff", "sae.safetensors", "bad.json", "evidence.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(
        result, f"Invalid activations JSON: could not convert string to float: {value!r}"
    )


@pytest.mark.parametrize("value", VALUES)
def test_a_diff_failure_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(sae_diff, "load_sae_weights", lambda path: object())
    monkeypatch.setattr(sae_diff, "compute_sae_diff", _fail_with(ValueError, value))

    result = _probe("sae-diff", "sae.safetensors", "evidence.json", "evidence.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, shown(value))


def _sae_report(monkeypatch) -> None:
    report = sae_diff.SaeFeatureDiffReport(num_features=2, num_tokens=1, l2_drift=0.0, changes=())
    monkeypatch.setattr(sae_diff, "load_sae_weights", lambda path: object())
    monkeypatch.setattr(sae_diff, "compute_sae_diff", lambda *a, **k: report)


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_diff_receipt_names_the_output_as_typed(monkeypatch, value):
    _sae_report(monkeypatch)
    output = f"diff-{value}.json"

    result = _probe("sae-diff", "sae.safetensors", "evidence.json", "evidence.json", "-o", output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert Path(output).is_file()
    assert_shown(result, "# SAE feature diff")
    assert_shown(result, f"Wrote diff → {shown(output)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_diff_write_failure_is_shown_as_typed(monkeypatch, value):
    _sae_report(monkeypatch)
    monkeypatch.setattr(probe_cmd, "atomic_write_text", _fail_with(ValueError, f"no {value}"))

    result = _probe("sae-diff", "sae.safetensors", "evidence.json", "evidence.json", "-o", "d.json")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert_shown(result, f"no {shown(value)}")


# ---------------------------------------------------------------------------
# The labels keep their colours
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(monkeypatch):
    monkeypatch.setattr(
        probe_cmd,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )


def test_a_refusal_stays_red(coloured):
    result = _probe("pack", "nope")

    assert result.exit_code == 2
    assert "\x1b[31mno probe pack for base 'nope'" in result.output


def test_a_receipt_stays_green(coloured):
    result = _probe("pack", probe_pack.list_probe_bases()[0], "--output", "pack.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32mWrote probe pack → pack.json\x1b[0m" in result.output


def test_the_list_table_keeps_its_bold_column(coloured, monkeypatch):
    monkeypatch.setattr(probe_pack, "list_probe_bases", lambda: ("one-base",))

    result = _probe("pack", "--list")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[1mone-base" in result.output


def test_the_panel_keeps_its_bold_base_and_dim_hint(coloured, monkeypatch):
    _bundle(monkeypatch, "sleeper", "the-base")

    result = _probe("sleeper", "any")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Base: \x1b[1mthe-base\x1b[0m" in result.output
    assert "\x1b[2mPass --evidence <activations.json> to run the probe.\x1b[0m" in result.output


@pytest.mark.parametrize(
    "args,label",
    [
        (
            (
                "interference", "--measure", "suite.jsonl", "--adapter", "a=a",
                "--adapter", "b=b", "--base-model", "m",
            ),
            "--measure:",
        ),
        (("sae-diff", "sae", "a.json", "b.json", "--auto-download"), "--auto-download:"),
    ],
)
def test_a_progress_line_keeps_its_cyan_label(coloured, monkeypatch, args, label):
    from soup_cli.utils import interference_live

    Path("suite.jsonl").write_text('{"prompt": "p", "response": "r"}\n', "utf-8")
    monkeypatch.setattr(
        interference_live, "measure_interference_losses", _fail_with(RuntimeError, "stop")
    )
    monkeypatch.setattr(sae_diff, "download_sae", _fail_with(ValueError, "stop"))

    result = _probe(*args)

    assert result.exit_code == 2
    assert f"\x1b[36m{label}\x1b[0m " in result.output
