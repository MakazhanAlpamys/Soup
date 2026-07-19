"""v0.71.39 "CI for weights, not prompts: close the evidence loop" tests.

Four features, all offline / CPU-testable:

- Part A: ``soup ship --emit-evidence`` — a verdict re-serialised into the
  ``--evidence`` INPUT schema, so ``--emit-evidence -> --evidence`` round-trips
  to an identical verdict (#312). Output was not input before this.
- Part B: ``ShipConfig`` under ``EvalConfig`` + ``soup ship --config soup.yaml``
  reading leg-1/leg-2 defaults (CLI flag > config > hard default).
- Part C: ``soup ship --push owner/repo#N`` — the verdict as a PR comment
  (reuses ``utils/adapter_pr.post_pr_comment``).
- Part D: an optional ``provenance`` block on the evidence schema
  (``config_sha`` via the Registry hasher) + a ``--config`` staleness refusal
  (evidence whose config_sha != the committed config's is rejected) + a
  ``ci init --config`` binding that wires the gate to the committed config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils.ship_verdict import (
    TASK_MODES,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    verdict_to_evidence,
)

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

runner = CliRunner()


def _write_evidence(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ship_evidence() -> dict:
    """A SHIP evidence payload (task improves, no regression)."""
    return {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.70},
        "benchmarks": {
            "mini_mmlu": {"base": 0.60, "tuned": 0.60},
            "mini_arithmetic": {"base": 0.40, "tuned": 0.42},
        },
    }


def _dont_ship_evidence() -> dict:
    """A DON'T-SHIP evidence payload (leg 2 regresses)."""
    return {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.70},
        "benchmarks": {"mini_mmlu": {"base": 0.80, "tuned": 0.50}},
    }


# ---------------------------------------------------------------------------
# Part A — verdict_to_evidence (pure) + round-trip
# ---------------------------------------------------------------------------

class TestVerdictToEvidence:
    def _verdict(self, payload: dict, threshold: float = 0.05):
        task = payload["task"]
        win = build_task_win(task["mode"], task["base"], task["tuned"])
        base = {k: v["base"] for k, v in payload["benchmarks"].items()}
        tuned = {k: v["tuned"] for k, v in payload["benchmarks"].items()}
        deltas = compute_benchmark_deltas(base, tuned, forgetting_threshold=threshold)
        return decide_ship(win, deltas, forgetting_threshold=threshold)

    def test_shape_is_the_evidence_input_schema(self):
        ev = verdict_to_evidence(self._verdict(_ship_evidence()))
        assert set(ev) >= {"task", "benchmarks"}
        assert set(ev["task"]) == {"mode", "base", "tuned"}
        assert ev["task"]["mode"] == "metric"
        assert ev["task"]["base"] == 0.50
        assert ev["task"]["tuned"] == 0.70
        assert ev["benchmarks"]["mini_mmlu"] == {"base": 0.60, "tuned": 0.60}

    def test_no_provenance_by_default(self):
        ev = verdict_to_evidence(self._verdict(_ship_evidence()))
        assert "provenance" not in ev

    def test_provenance_included_when_supplied(self):
        prov = {"config_sha": "abc123", "base_model": "sshleifer/tiny-gpt2"}
        ev = verdict_to_evidence(self._verdict(_ship_evidence()), provenance=prov)
        assert ev["provenance"] == prov

    def test_round_trip_verdict_identical(self):
        """The headline: emit -> read -> identical decision + legs + failed_rule.

        Covers every leg-1 mode (metric / judge_score / pairwise) so a bug that
        hardcodes ``mode`` in the serialiser would be caught.
        """
        pairwise = {
            "task": {"mode": "pairwise", "base": 0.5, "tuned": 0.65},
            "benchmarks": {"mini_mmlu": {"base": 0.6, "tuned": 0.6}},
        }
        judge = {
            "task": {"mode": "judge_score", "base": 0.4, "tuned": 0.4},
            "benchmarks": {"mini_mmlu": {"base": 0.6, "tuned": 0.6}},
        }
        for payload in (_ship_evidence(), _dont_ship_evidence(), pairwise, judge):
            original = self._verdict(payload)
            ev = verdict_to_evidence(original)
            assert ev["task"]["mode"] == payload["task"]["mode"]
            # Feed the emitted evidence back through the SAME construction path.
            back = self._verdict(ev)
            assert back.decision == original.decision
            assert back.failed_rule == original.failed_rule
            assert back.task_win == original.task_win
            assert back.benchmark_deltas == original.benchmark_deltas

    def test_to_dict_is_json_serialisable(self):
        ev = verdict_to_evidence(self._verdict(_ship_evidence()))
        assert json.loads(json.dumps(ev)) == ev

    def test_rejects_non_verdict(self):
        with pytest.raises(TypeError):
            verdict_to_evidence({"not": "a verdict"})


# ---------------------------------------------------------------------------
# Part A — CLI --emit-evidence
# ---------------------------------------------------------------------------

class TestEmitEvidenceCli:
    def test_emit_from_evidence_round_trips_through_cli(self):
        """`ship --evidence A --emit-evidence B` then `ship --evidence B` agree."""
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("a.json"), _ship_evidence())
            res1 = runner.invoke(
                ship_cmd.app,
                ["--evidence", "a.json", "--emit-evidence", "b.json",
                 "--output", "v1.json"],
            )
            assert res1.exit_code == 0, (res1.output, repr(res1.exception))
            assert Path("b.json").exists()
            emitted = json.loads(Path("b.json").read_text(encoding="utf-8"))
            assert emitted["task"]["mode"] == "metric"
            assert "mini_mmlu" in emitted["benchmarks"]
            # Round-trip: the emitted evidence yields the SAME exit code AND the
            # SAME verdict content (not just the same SHIP/DON'T boundary).
            res2 = runner.invoke(
                ship_cmd.app, ["--evidence", "b.json", "--output", "v2.json"]
            )
            assert res2.exit_code == res1.exit_code
            v1 = json.loads(Path("v1.json").read_text(encoding="utf-8"))
            v2 = json.loads(Path("v2.json").read_text(encoding="utf-8"))
            assert v1 == v2

    def test_emit_dont_ship_round_trips_exit_2(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("a.json"), _dont_ship_evidence())
            res1 = runner.invoke(
                ship_cmd.app, ["--evidence", "a.json", "--emit-evidence", "b.json"]
            )
            assert res1.exit_code == 2, (res1.output, repr(res1.exception))
            res2 = runner.invoke(ship_cmd.app, ["--evidence", "b.json"])
            assert res2.exit_code == 2

    def test_emit_evidence_outside_cwd_rejected(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("a.json"), _ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "a.json", "--emit-evidence", "../out.json"]
            )
            assert res.exit_code == 1, (res.output, repr(res.exception))


# ---------------------------------------------------------------------------
# Part B — ShipConfig schema + soup ship --config reader
# ---------------------------------------------------------------------------

_CONFIG_LENIENT = (
    "base: sshleifer/tiny-gpt2\n"
    "data:\n"
    "  train: x.jsonl\n"
    "eval:\n"
    "  ship:\n"
    "    forgetting_threshold: 0.20\n"
    "    task_mode: metric\n"
    "    general_suite: mini_mmlu,mini_arithmetic\n"
)


class TestShipConfigSchema:
    def test_defaults(self):
        from soup_cli.config.schema import ShipConfig

        cfg = ShipConfig()
        assert cfg.task_mode == "metric"
        assert cfg.forgetting_threshold == 0.05
        assert cfg.task_eval is None
        assert cfg.general_suite is None
        assert cfg.judge_model is None
        assert cfg.baseline is None

    def test_parses_under_eval(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_CONFIG_LENIENT)
        assert cfg.eval is not None
        assert cfg.eval.ship is not None
        assert cfg.eval.ship.forgetting_threshold == 0.20
        assert cfg.eval.ship.general_suite == "mini_mmlu,mini_arithmetic"

    @pytest.mark.parametrize("mode", TASK_MODES)
    def test_task_mode_accepts_every_ship_mode(self, mode):
        """Every ship_verdict.TASK_MODES value must be accepted (no drift)."""
        from soup_cli.config.schema import ShipConfig

        assert ShipConfig(task_mode=mode).task_mode == mode

    def test_rejects_bad_task_mode(self):
        import pydantic

        from soup_cli.config.schema import ShipConfig

        with pytest.raises(pydantic.ValidationError):
            ShipConfig(task_mode="bogus")

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_rejects_out_of_bounds_threshold(self, bad):
        import pydantic

        from soup_cli.config.schema import ShipConfig

        with pytest.raises(pydantic.ValidationError):
            ShipConfig(forgetting_threshold=bad)


class TestShipConfigReader:
    def _ev_010_bound(self) -> dict:
        # --config in offline mode enforces staleness, so stamp matching
        # provenance (Part D) onto the 0.10-drop evidence.
        ev = _dont_ship_evidence_010()
        ev["provenance"] = {"config_sha": _config_sha(_CONFIG_LENIENT)}
        return ev

    def test_config_threshold_flips_offline_verdict(self):
        """A 0.10 drop is DON'T SHIP at the 0.05 default, SHIP at config's 0.20."""
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), self._ev_010_bound())
            Path("soup.yaml").write_text(_CONFIG_LENIENT, encoding="utf-8")
            # No config -> default 0.05 -> regression -> DON'T SHIP.
            no_cfg = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert no_cfg.exit_code == 2, (no_cfg.output, repr(no_cfg.exception))
            # With config -> 0.20 threshold -> SHIP.
            with_cfg = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert with_cfg.exit_code == 0, (with_cfg.output, repr(with_cfg.exception))

    def test_explicit_flag_overrides_config(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), self._ev_010_bound())
            Path("soup.yaml").write_text(_CONFIG_LENIENT, encoding="utf-8")
            # Config says 0.20 (would SHIP), but explicit 0.05 wins -> DON'T SHIP.
            res = runner.invoke(
                ship_cmd.app,
                ["--evidence", "ev.json", "--config", "soup.yaml",
                 "--forgetting-threshold", "0.05"],
            )
            assert res.exit_code == 2, (res.output, repr(res.exception))

    def test_bad_config_file_is_usage_error(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())
            Path("bad.yaml").write_text("base: [unterminated\n", encoding="utf-8")
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "bad.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))

    def test_config_fills_every_live_leg_flag(self, monkeypatch):
        """task_eval/task_mode/general_suite/judge_model/baseline thread from config."""
        from soup_cli.commands import ship as ship_cmd

        captured = {}

        def _fake_live(**kwargs):
            captured.update(kwargs)
            win = build_task_win("metric", 0.5, 0.7)
            deltas = compute_benchmark_deltas({"b": 0.6}, {"b": 0.6})
            return decide_ship(win, deltas)

        monkeypatch.setattr(ship_cmd, "_verdict_live", _fake_live)
        cfg = (
            _CONFIG_MIN + "eval:\n  ship:\n"
            "    task_eval: tasks.jsonl\n"
            "    task_mode: judge_score\n"
            "    general_suite: mini_mmlu\n"
            "    judge_model: https://judge.example\n"
            "    baseline: registry://abc\n"
        )
        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(cfg, encoding="utf-8")
            res = runner.invoke(
                ship_cmd.app, ["--base", "m", "--adapter", "a", "--config", "soup.yaml"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert captured["task_eval"] == "tasks.jsonl"
            assert captured["task_mode"] == "judge_score"
            assert captured["general_suite"] == "mini_mmlu"
            assert captured["judge_model"] == "https://judge.example"
            assert captured["baseline_spec"] == "registry://abc"


def _dont_ship_evidence_010() -> dict:
    """Task improves but mini_mmlu drops exactly 0.10 (regresses at 0.05, ok at 0.20)."""
    return {
        "task": {"mode": "metric", "base": 0.50, "tuned": 0.70},
        "benchmarks": {"mini_mmlu": {"base": 0.70, "tuned": 0.60}},
    }


# ---------------------------------------------------------------------------
# Part C — render_ship_pr_markdown + soup ship --push
# ---------------------------------------------------------------------------

def _make_verdict(payload: dict, threshold: float = 0.05):
    task = payload["task"]
    win = build_task_win(task["mode"], task["base"], task["tuned"])
    base = {k: v["base"] for k, v in payload["benchmarks"].items()}
    tuned = {k: v["tuned"] for k, v in payload["benchmarks"].items()}
    deltas = compute_benchmark_deltas(base, tuned, forgetting_threshold=threshold)
    return decide_ship(win, deltas, forgetting_threshold=threshold)


class TestRenderShipPrMarkdown:
    def test_heading_names_decision(self):
        from soup_cli.utils.ship_verdict import render_ship_pr_markdown

        md = render_ship_pr_markdown(_make_verdict(_ship_evidence()))
        assert "SHIP" in md
        assert md.lstrip().startswith("## soup ship")

    def test_body_is_fenced(self):
        from soup_cli.utils.ship_verdict import render_ship_pr_markdown

        md = render_ship_pr_markdown(_make_verdict(_ship_evidence()))
        assert "```" in md
        assert "Leg 1 task win" in md

    def test_hostile_benchmark_name_cannot_break_fence(self):
        """A benchmark name containing ``` must not close the fence early."""
        from soup_cli.utils.ship_verdict import render_ship_pr_markdown

        payload = {
            "task": {"mode": "metric", "base": 0.5, "tuned": 0.7},
            "benchmarks": {"evil```name": {"base": 0.6, "tuned": 0.6}},
        }
        md = render_ship_pr_markdown(_make_verdict(payload))
        assert "```" in md
        # The opening fence must be strictly longer than any backtick run in the
        # body, so the content cannot inject markdown after it.
        opening = md[md.index("```"):]
        fence_len = len(opening) - len(opening.lstrip("`"))
        assert fence_len >= 4  # longer than the 3-backtick run in the name

    def test_dont_ship_heading_and_emoji(self):
        """The DON'T-SHIP branch must render its own heading + ❌ (not a false ✅)."""
        from soup_cli.utils.ship_verdict import render_ship_pr_markdown

        md = render_ship_pr_markdown(_make_verdict(_dont_ship_evidence()))
        heading = md.split("\n", 1)[0]
        assert "DON'T SHIP" in heading
        assert "❌" in md and "✅" not in md

    def test_rejects_non_verdict(self):
        from soup_cli.utils.ship_verdict import render_ship_pr_markdown

        with pytest.raises(TypeError):
            render_ship_pr_markdown({"nope": 1})


class TestShipPushCli:
    def test_push_posts_comment_and_preserves_exit_code(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import adapter_pr

        calls = {}

        def _fake_post(target, body, **kwargs):
            calls["target"] = target
            calls["body"] = body
            return "https://github.com/o/r/pull/1#issuecomment-1"

        monkeypatch.setattr(adapter_pr, "post_pr_comment", _fake_post)
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--push", "owner/repo#1"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert calls["target"] == "owner/repo#1"
            assert "soup ship" in calls["body"]

    def test_push_on_dont_ship_still_posts_and_exits_2(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import adapter_pr

        posted = {}
        monkeypatch.setattr(
            adapter_pr, "post_pr_comment",
            lambda target, body, **kw: posted.setdefault("t", target) or "",
        )
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _dont_ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--push", "owner/repo#2"]
            )
            assert res.exit_code == 2, (res.output, repr(res.exception))
            assert posted["t"] == "owner/repo#2"

    def test_push_bad_target_is_usage_error(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--push", "not-a-target"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))

    def test_push_transport_failure_warns_but_preserves_verdict_exit(self, monkeypatch):
        """A missing token / gh failure must NOT flip a real SHIP into exit 1."""
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import adapter_pr

        def _raise(target, body, **kwargs):
            raise RuntimeError("no GitHub token found")

        monkeypatch.setattr(adapter_pr, "post_pr_comment", _raise)
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--push", "owner/repo#1"]
            )
            # SHIP verdict is preserved (exit 0); the post failure is a warning.
            assert res.exit_code == 0, (res.output, repr(res.exception))
            assert "could not post" in res.output.lower()

    def test_push_transport_failure_preserves_dont_ship_exit(self, monkeypatch):
        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import adapter_pr

        def _raise(target, body, **kwargs):
            raise RuntimeError("gh api failed")

        monkeypatch.setattr(adapter_pr, "post_pr_comment", _raise)
        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _dont_ship_evidence())
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--push", "owner/repo#2"]
            )
            assert res.exit_code == 2, (res.output, repr(res.exception))


# ---------------------------------------------------------------------------
# Part D — provenance (config_sha) + staleness refusal + ci init v2
# ---------------------------------------------------------------------------

_CONFIG_MIN = "base: sshleifer/tiny-gpt2\ndata:\n  train: train.jsonl\n"


def _config_sha(text: str) -> str:
    # Delegate to the production fingerprint so test + prod cannot drift (it
    # excludes the eval.ship gate policy from the recipe hash).
    from soup_cli.commands.ship import _config_sha_of
    from soup_cli.config.loader import load_config_from_string

    return _config_sha_of(load_config_from_string(text))


def _ship_with_provenance(config_sha: str) -> dict:
    ev = _ship_evidence()
    ev["provenance"] = {"config_sha": config_sha, "base_model": "sshleifer/tiny-gpt2"}
    return ev


class TestComputeProvenance:
    def test_provenance_shape(self):
        from soup_cli.commands.ship import _compute_provenance
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_CONFIG_MIN)
        prov = _compute_provenance(cfg)
        assert prov["config_sha"] == _config_sha(_CONFIG_MIN)
        assert len(prov["config_sha"]) == 64
        assert prov["base_model"] == "sshleifer/tiny-gpt2"

    def test_config_sha_is_semantic_not_textual(self):
        """Whitespace / comment / key-order changes must not change the sha."""
        reordered = "data:\n  train: train.jsonl\n# a comment\nbase: sshleifer/tiny-gpt2\n"
        assert _config_sha(_CONFIG_MIN) == _config_sha(reordered)

    def test_config_sha_excludes_eval_ship_policy(self):
        """Tuning the gate's own policy (eval.ship) must NOT change the sha.

        The staleness fingerprint is the training recipe, not the read-time gate
        policy — else loosening forgetting_threshold would falsely reject valid
        evidence about an unchanged model.
        """
        with_ship = (
            _CONFIG_MIN + "eval:\n  auto_eval: false\n  ship:\n"
            "    forgetting_threshold: 0.20\n"
        )
        different_ship = (
            _CONFIG_MIN + "eval:\n  auto_eval: false\n  ship:\n"
            "    forgetting_threshold: 0.03\n"
        )
        no_ship = _CONFIG_MIN + "eval:\n  auto_eval: false\n"
        # Two configs differing ONLY in the eval.ship policy hash identically...
        assert _config_sha(with_ship) == _config_sha(different_ship)
        # ...and a ship block hashes the same as no ship block at all.
        assert _config_sha(with_ship) == _config_sha(no_ship)

    def test_config_sha_changes_on_real_recipe_edit(self):
        """A genuine base/data change DOES change the sha (the gate still bites)."""
        other_base = "base: gpt2\ndata:\n  train: train.jsonl\n"
        assert _config_sha(_CONFIG_MIN) != _config_sha(other_base)

    def test_config_sha_exclusion_is_scoped_to_ship_only(self):
        """A non-ship eval field (auto_eval) MUST still change the sha.

        Guards against an over-broad exclusion (dropping the whole eval block)
        that would silently stop the gate noticing eval changes.
        """
        a = _CONFIG_MIN + "eval:\n  auto_eval: false\n"
        b = _CONFIG_MIN + "eval:\n  auto_eval: true\n"
        assert _config_sha(a) != _config_sha(b)


class TestStalenessGate:
    def test_matching_provenance_passes(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_evidence(
                Path("ev.json"), _ship_with_provenance(_config_sha(_CONFIG_MIN))
            )
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_mismatched_config_sha_is_stale(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_evidence(Path("ev.json"), _ship_with_provenance("deadbeef" * 8))
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "stale" in res.output.lower()

    def test_missing_provenance_refused(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            _write_evidence(Path("ev.json"), _ship_evidence())  # no provenance
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            assert "provenance" in res.output.lower()

    def test_no_config_means_no_staleness_check(self):
        """Back-compat: --evidence alone (no --config) never staleness-checks."""
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())  # no provenance
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json"])
            assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_empty_config_fails_loud_not_silent(self):
        """`--config ""` must fail loud (exit 3), not silently skip the gate."""
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            _write_evidence(Path("ev.json"), _ship_evidence())  # no provenance
            res = runner.invoke(ship_cmd.app, ["--evidence", "ev.json", "--config", ""])
            assert res.exit_code == 3, (res.output, repr(res.exception))

    def test_producer_stamps_raw_evidence_without_gating(self):
        """--emit-evidence + --config STAMPS provenance onto RAW scores (no gate).

        The natural producer flow: raw scores from an external eval tool have no
        provenance yet, so the staleness gate must NOT fire when emitting — it
        must stamp the config's provenance so the downstream GATE then accepts it.
        """
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            Path("train.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
            _write_evidence(Path("raw.json"), _ship_evidence())  # NO provenance
            res = runner.invoke(
                ship_cmd.app,
                ["--evidence", "raw.json", "--config", "soup.yaml",
                 "--emit-evidence", "out.json"],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            out = json.loads(Path("out.json").read_text(encoding="utf-8"))
            assert out["provenance"]["config_sha"] == _config_sha(_CONFIG_MIN)
            # The GATE (no --emit-evidence) now accepts the stamped evidence.
            gate = runner.invoke(
                ship_cmd.app, ["--evidence", "out.json", "--config", "soup.yaml"]
            )
            assert gate.exit_code == 0, (gate.output, repr(gate.exception))

    def test_bound_evidence_reemits_with_provenance(self):
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            sha = _config_sha(_CONFIG_MIN)
            _write_evidence(Path("ev.json"), _ship_with_provenance(sha))
            res = runner.invoke(
                ship_cmd.app,
                ["--evidence", "ev.json", "--config", "soup.yaml",
                 "--emit-evidence", "out.json"],
            )
            assert res.exit_code == 0, (res.output, repr(res.exception))
            out = json.loads(Path("out.json").read_text(encoding="utf-8"))
            assert out["provenance"]["config_sha"] == sha


class TestCiInitV2:
    def test_render_without_config_has_no_config_flag(self):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        yaml_text = render_soup_gate_workflow(
            data_path="data/train.jsonl",
            suite_path="expectations.yaml",
            evidence_path="ship_evidence.json",
        )
        assert "soup ship --evidence" in yaml_text
        assert "--config" not in yaml_text

    def test_render_with_config_binds_the_gate(self):
        """The --config must land on the ship `run:` STEP, not just a comment."""
        import yaml

        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        yaml_text = render_soup_gate_workflow(
            data_path="data/train.jsonl",
            suite_path="expectations.yaml",
            evidence_path="ship_evidence.json",
            config_path="soup.yaml",
        )
        run = yaml.safe_load(yaml_text)["jobs"]["soup-gate"]["steps"][-1]["run"]
        assert "--config" in run and "soup.yaml" in run

    def test_ci_init_with_config_writes_bound_workflow(self):
        import yaml

        from soup_cli.cli import app as main_app

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            res = runner.invoke(main_app, ["ci", "init", "--config", "soup.yaml"])
            assert res.exit_code == 0, (res.output, repr(res.exception))
            wf = Path(".github/workflows/soup-gate.yml").read_text(encoding="utf-8")
            run = yaml.safe_load(wf)["jobs"]["soup-gate"]["steps"][-1]["run"]
            assert "--config" in run and "soup.yaml" in run

    def test_render_config_outside_cwd_rejected(self):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        with runner.isolated_filesystem():
            with pytest.raises(ValueError):
                render_soup_gate_workflow(
                    data_path="data/train.jsonl",
                    suite_path="expectations.yaml",
                    evidence_path="ship_evidence.json",
                    config_path="../soup.yaml",
                )


class TestSecurityHardening:
    def test_malformed_config_sha_not_echoed_to_terminal(self):
        """An ESC-laden provenance.config_sha must be rejected without echoing it."""
        from soup_cli.commands import ship as ship_cmd

        with runner.isolated_filesystem():
            Path("soup.yaml").write_text(_CONFIG_MIN, encoding="utf-8")
            ev = _ship_evidence()
            ev["provenance"] = {"config_sha": "\x1b[2J\x1b[HPWNED"}
            _write_evidence(Path("ev.json"), ev)
            res = runner.invoke(
                ship_cmd.app, ["--evidence", "ev.json", "--config", "soup.yaml"]
            )
            assert res.exit_code == 3, (res.output, repr(res.exception))
            # The raw ESC byte must NOT reach the terminal.
            assert "\x1b" not in res.output
            assert "PWNED" not in res.output

    def test_data_train_symlink_out_of_cwd_not_hashed(self, tmp_path):
        """A symlinked data.train must not be followed when computing data_sha."""
        import os

        from soup_cli.commands.ship import _compute_provenance
        from soup_cli.config.loader import load_config_from_string

        with runner.isolated_filesystem():
            outside = tmp_path / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            try:
                os.symlink(outside, "train.jsonl")
            except (OSError, NotImplementedError):
                pytest.skip("symlink not permitted on this platform")
            cfg = load_config_from_string(_CONFIG_MIN)  # data.train = train.jsonl
            prov = _compute_provenance(cfg)
            # config_sha always present; data_sha omitted for the symlink.
            assert "config_sha" in prov
            assert "data_sha" not in prov

    def test_data_train_real_file_is_hashed(self):
        from soup_cli.commands.ship import _compute_provenance
        from soup_cli.config.loader import load_config_from_string

        with runner.isolated_filesystem():
            Path("train.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
            cfg = load_config_from_string(_CONFIG_MIN)
            prov = _compute_provenance(cfg)
            assert len(prov["data_sha"]) == 64

    @pytest.mark.parametrize("bad", ["a #comment", "a\x85b", "path#frag"])
    def test_ci_workflow_rejects_comment_and_linebreak_paths(self, bad):
        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        with pytest.raises(ValueError, match="single line|must not contain '#'"):
            render_soup_gate_workflow(
                data_path=bad,
                suite_path="expectations.yaml",
                evidence_path="ship_evidence.json",
            )

    def test_safe_read_text_enforces_byte_cap(self):
        from soup_cli.commands.ship import _safe_read_text

        with runner.isolated_filesystem():
            Path("big.txt").write_text("x" * 100, encoding="utf-8")
            with pytest.raises(ValueError, match="exceeds"):
                _safe_read_text("big.txt", "big", 10)

    def test_safe_hash_file_skips_oversize(self):
        from soup_cli.commands.ship import _safe_hash_file

        with runner.isolated_filesystem():
            Path("big.bin").write_bytes(b"x" * 100)
            assert _safe_hash_file("big.bin", 10) is None
            assert len(_safe_hash_file("big.bin", 1000)) == 64

    def test_bound_workflow_yaml_parses(self):
        """The generated workflow (with --config) must be valid YAML."""
        import yaml

        from soup_cli.utils.ci_workflow import render_soup_gate_workflow

        text = render_soup_gate_workflow(
            data_path="data/train.jsonl",
            suite_path="expectations.yaml",
            evidence_path="ship_evidence.json",
            config_path="soup.yaml",
        )
        doc = yaml.safe_load(text)
        run = doc["jobs"]["soup-gate"]["steps"][-1]["run"]
        assert "--config" in run and "soup.yaml" in run


class TestNoTopLevelTorch:
    def test_ship_verdict_no_torch(self):
        head = (_SRC / "utils" / "ship_verdict.py").read_text(encoding="utf-8")
        for marker in ("\ndef ", "\nclass "):
            head = head.split(marker, 1)[0]
        assert "import torch" not in head
