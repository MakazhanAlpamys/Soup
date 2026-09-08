"""#367 criteria 2 and 4 -- soup ship stamps judge numerics and gates replay.

Criteria 1 and 3 (the load argument reaching from_pretrained) live in
test_issue367_live_eval_quantization.py. This file covers:

- the verdict / --output / --emit-evidence stamp matching the actual load
  (not merely that the call does not raise)
- a family-compare staleness gate (4bit vs full, 4bit vs 8bit) in the same
  spirit as config_sha
- missing stamps warn rather than refuse (pre-#367 evidence)
- --evidence --emit-evidence copies an existing stamp through

No GPU: live paths mock _resolve_generators; the gate is pure JSON.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils.ship_verdict import (
    NUMERICS_FAMILY_FULL,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    format_ship_rubric,
    numerics_family,
    numerics_from_evidence,
    parse_numerics,
    render_ship_panel,
    verdict_to_dict,
    verdict_to_evidence,
)

runner = CliRunner()

_CONFIG_MIN = "base: sshleifer/tiny-gpt2\ndata:\n  train: train.jsonl\n"
_CONFIG_NONE = _CONFIG_MIN + "training:\n  quantization: none\n"


def _config_sha(text: str) -> str:
    from soup_cli.commands.ship import _config_sha_of
    from soup_cli.config.loader import load_config_from_string

    return _config_sha_of(load_config_from_string(text))


def _ship_evidence() -> dict:
    return {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.70},
        "benchmarks": {"mini_mmlu": {"base": 0.60, "tuned": 0.60}},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _verdict(numerics=None):
    win = build_task_win("metric", 0.50, 0.70)
    deltas = compute_benchmark_deltas({"mini_mmlu": 0.60}, {"mini_mmlu": 0.60})
    verdict = decide_ship(win, deltas)
    if numerics is not None:
        return replace(verdict, numerics=numerics)
    return verdict


class TestNumericsFamily:
    def test_quantized_stays_itself(self):
        assert numerics_family("4bit") == "4bit"
        assert numerics_family("8bit") == "8bit"

    def test_full_precision_collapses(self):
        assert numerics_family("bfloat16") == NUMERICS_FAMILY_FULL
        assert numerics_family("float32") == NUMERICS_FAMILY_FULL

    def test_unknown_raises_without_echoing_the_value(self):
        with pytest.raises(ValueError, match="must be one of") as exc:
            numerics_family("gptq")
        assert "gptq" not in str(exc.value)


class TestParseNumerics:
    def test_accepts_known_stamps(self):
        for value in ("4bit", "8bit", "bfloat16", "float32"):
            assert parse_numerics(value) == value

    def test_rejects_non_string_without_echoing(self):
        with pytest.raises(ValueError, match="must be a string"):
            parse_numerics(4)

    def test_absent_key_is_none(self):
        assert numerics_from_evidence(None) is None

    def test_malformed_does_not_echo_esc(self):
        with pytest.raises(ValueError) as exc:
            parse_numerics("\x1b[2JPWNED")
        assert "\x1b" not in str(exc.value)
        assert "PWNED" not in str(exc.value)


class TestVerdictSerialisesNumerics:
    def test_evidence_omits_unstamped(self):
        ev = verdict_to_evidence(_verdict())
        assert "numerics" not in ev

    def test_evidence_includes_stamp(self):
        ev = verdict_to_evidence(_verdict("4bit"))
        assert ev["numerics"] == "4bit"

    def test_output_dict_always_has_the_key(self):
        assert verdict_to_dict(_verdict())["numerics"] is None
        assert verdict_to_dict(_verdict("8bit"))["numerics"] == "8bit"

    def test_rubric_and_panel_report_the_stamp(self):
        rubric = format_ship_rubric(_verdict("4bit"))
        assert "Judge numerics: 4bit" in rubric
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        Console(file=buf, width=100).print(render_ship_panel(_verdict("bfloat16")))
        assert "bfloat16" in buf.getvalue()

    def test_rubric_says_unstamped_when_absent(self):
        assert "Judge numerics: unstamped" in format_ship_rubric(_verdict())


class TestLiveEvalNumericsHelper:
    def test_4bit_stamps_4bit_not_a_dtype(self):
        from soup_cli.commands.ship import _live_eval_numerics

        assert _live_eval_numerics("4bit", "cpu") == "4bit"

    def test_cpu_stamps_float32(self):
        from soup_cli.commands.ship import _live_eval_numerics

        assert _live_eval_numerics(None, "cpu") == "float32"

    def test_indexed_cuda_stamps_bfloat16_without_torch(self):
        from soup_cli.commands.ship import _live_eval_numerics

        assert _live_eval_numerics(None, "cuda:0") == "bfloat16"

    def test_gptq_config_family_is_full(self):
        from soup_cli.commands.ship import _expected_numerics_family
        from soup_cli.config.schema import SoupConfig

        cfg = SoupConfig(
            base="m", data={"train": "t.jsonl"}, training={"quantization": "gptq"}
        )
        assert _expected_numerics_family(cfg) == NUMERICS_FAMILY_FULL

    def test_default_config_family_is_4bit(self):
        from soup_cli.commands.ship import _expected_numerics_family
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_CONFIG_MIN)
        assert _expected_numerics_family(cfg) == "4bit"


class TestEmitEvidenceStampsNumerics:
    def test_live_4bit_config_stamp_matches_the_load(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd

        captured = {}

        def _fake_resolve(base, tuned, adapter, device, quantization=None):
            captured["quantization"] = quantization
            return (lambda p: "hi", lambda p: "hi")

        monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
        cfg = "base: m\ndata:\n  train: t.jsonl\ntraining:\n  quantization: 4bit\n"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(cfg, encoding="utf-8")
            Path("task.jsonl").write_text(
                json.dumps(
                    {"prompt": "say hi", "expected": "hi", "scoring": "contains"}
                )
                + "\n",
                encoding="utf-8",
            )
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "m", "--adapter", "a", "--task-eval", "task.jsonl",
                    "--device", "cpu", "--config", "soup.yaml",
                    "--emit-evidence", "ev.json", "--output", "out.json",
                ],
            )
            assert res.exit_code in (0, 2), (res.output, repr(res.exception))
            assert captured["quantization"] == "4bit"
            emitted = json.loads(Path("ev.json").read_text(encoding="utf-8"))
            output = json.loads(Path("out.json").read_text(encoding="utf-8"))
            assert emitted["numerics"] == "4bit"
            assert output["numerics"] == "4bit"
            assert "4bit" in res.output

    def test_live_without_config_on_cpu_stamps_float32(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd

        captured = {}

        def _fake_resolve(base, tuned, adapter, device, quantization=None):
            captured["quantization"] = quantization
            return (lambda p: "hi", lambda p: "hi")

        monkeypatch.setattr(ship_cmd, "_resolve_generators", _fake_resolve)
        with runner.isolated_filesystem():
            Path("task.jsonl").write_text(
                json.dumps(
                    {"prompt": "say hi", "expected": "hi", "scoring": "contains"}
                )
                + "\n",
                encoding="utf-8",
            )
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--base", "m", "--adapter", "a", "--task-eval", "task.jsonl",
                    "--device", "cpu", "--emit-evidence", "ev.json",
                ],
            )
            assert res.exit_code in (0, 2), (res.output, repr(res.exception))
            assert captured["quantization"] is None
            emitted = json.loads(Path("ev.json").read_text(encoding="utf-8"))
            assert emitted["numerics"] == "float32"


class TestNumericsStalenessGate:
    def test_matching_4bit_stamp_passes(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        ev["numerics"] = "4bit"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_float32_stamp_against_default_4bit_config_is_stale(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        ev["numerics"] = "float32"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "stale" in res.output.lower()
            assert "numerics" in res.output.lower()

    def test_8bit_stamp_against_4bit_config_is_stale(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        ev["numerics"] = "8bit"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "stale" in res.output.lower()

    def test_bfloat16_and_float32_are_the_same_family(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_NONE)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        ev["numerics"] = "bfloat16"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_NONE, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_missing_stamp_warns_and_does_not_refuse(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "numerics stamp" in res.output.lower()

    def test_malformed_stamp_is_not_echoed(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["provenance"] = {"config_sha": sha}
        ev["numerics"] = "\x1b[2JPWNED"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "\x1b" not in res.output
            assert "PWNED" not in res.output

    def test_restamp_copies_numerics_through(self):
        from soup_cli.commands import ship as ship_cmd

        sha = _config_sha(_CONFIG_MIN)
        ev = _ship_evidence()
        ev["numerics"] = "4bit"
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_json(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app,
                [
                    "--evidence", "ev.json", "--config", "soup.yaml",
                    "--emit-evidence", "out.json",
                ],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            emitted = json.loads(Path("out.json").read_text(encoding="utf-8"))
            assert emitted["numerics"] == "4bit"
            assert emitted["provenance"]["config_sha"] == sha
