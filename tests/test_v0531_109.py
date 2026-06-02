"""v0.53.1 #109 — soup deploy autopilot --measure live wiring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# --- compute_cache_key ------------------------------------------------------


class TestComputeCacheKey:
    def test_basic(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        key = compute_cache_key(
            base_sha="abc123", profile_name="rtx-4090-24gb",
            tasks_sha="def456",
        )
        assert isinstance(key, str)
        assert len(key) == 32

    def test_deterministic(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        k1 = compute_cache_key(
            base_sha="x", profile_name="p", tasks_sha="t",
        )
        k2 = compute_cache_key(
            base_sha="x", profile_name="p", tasks_sha="t",
        )
        assert k1 == k2

    def test_diff_on_each_arg(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        base = compute_cache_key(base_sha="a", profile_name="b", tasks_sha="c")
        for changed in (
            compute_cache_key(base_sha="A", profile_name="b", tasks_sha="c"),
            compute_cache_key(base_sha="a", profile_name="B", tasks_sha="c"),
            compute_cache_key(base_sha="a", profile_name="b", tasks_sha="C"),
        ):
            assert changed != base

    def test_bool_rejected(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        with pytest.raises(TypeError):
            compute_cache_key(base_sha=True, profile_name="x", tasks_sha="y")

    def test_null_byte_rejected(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        with pytest.raises(ValueError):
            compute_cache_key(
                base_sha="ev\x00il", profile_name="x", tasks_sha="y",
            )

    def test_empty_rejected(self):
        from soup_cli.utils.deploy_measure import compute_cache_key

        with pytest.raises(ValueError):
            compute_cache_key(base_sha="", profile_name="x", tasks_sha="y")


# --- sha_of_file ------------------------------------------------------------


class TestShaOfFile:
    def test_basic(self, tmp_path):
        from soup_cli.utils.deploy_measure import sha_of_file

        f = tmp_path / "x.txt"
        f.write_bytes(b"hello world")
        h = sha_of_file(str(f))
        assert isinstance(h, str)
        assert len(h) == 64  # full sha256 hex

    def test_missing_file(self, tmp_path):
        from soup_cli.utils.deploy_measure import sha_of_file

        with pytest.raises(FileNotFoundError):
            sha_of_file(str(tmp_path / "missing"))

    def test_null_byte_rejected(self):
        from soup_cli.utils.deploy_measure import sha_of_file

        with pytest.raises(ValueError):
            sha_of_file("ev\x00il")


# --- measure_candidate ------------------------------------------------------


def _write_tasks(tmp_path: Path) -> Path:
    """Write a 2-row JSONL eval task file."""
    f = tmp_path / "tasks.jsonl"
    f.write_text(
        '{"prompt": "say hello", "expected": "hello", "scoring": "exact"}\n'
        '{"prompt": "say world", "expected": "world", "scoring": "exact"}\n',
        encoding="utf-8",
    )
    return f


class TestMeasureCandidate:
    def test_ok_when_after_matches(self, tmp_path):
        from soup_cli.utils.deploy_measure import measure_candidate

        tasks = _write_tasks(tmp_path)
        def gen(p):
            if "hello" in p:
                return "hello"
            return "world"
        result = measure_candidate(
            candidate="gptq", tasks_file=str(tasks),
            before_gen=gen, after_gen=gen,
        )
        assert result.candidate == "gptq"
        assert result.verdict == "OK"
        assert result.delta == 0.0

    def test_minor_band_at_3pct_drop(self, tmp_path):
        """L1: explicitly exercise the MINOR verdict band (2% < drop < 5%)."""
        from soup_cli.utils.deploy_measure import MeasureResult, measure_candidate

        # Build a 100-task fixture so we can hit a 3% drop
        big = tmp_path / "tasks_big.jsonl"
        big.write_text(
            "\n".join(
                '{"prompt": "p%d", "expected": "ok", "scoring": "exact"}' % i
                for i in range(100)
            ),
            encoding="utf-8",
        )

        def before(p):
            return "ok"

        # After: miss exactly 3 out of 100 → drop=0.03 → MINOR band
        miss_set = {"p7", "p23", "p64"}

        def after(p):
            return "WRONG" if p in miss_set else "ok"

        r = measure_candidate(
            candidate="awq", tasks_file=str(big),
            before_gen=before, after_gen=after,
        )
        assert isinstance(r, MeasureResult)
        assert r.verdict == "MINOR"
        assert 0.02 <= -r.delta < 0.05

    def test_minor_drop(self, tmp_path):
        from soup_cli.utils.deploy_measure import measure_candidate

        tasks = _write_tasks(tmp_path)
        def before(p):
            return "hello" if "hello" in p else "world"
        # Always wrong → score 0.0; before score 1.0 → drop 1.0 → MAJOR
        def after(p):
            return "WRONG"
        r = measure_candidate(
            candidate="awq", tasks_file=str(tasks),
            before_gen=before, after_gen=after,
        )
        assert r.verdict == "MAJOR"
        assert r.delta < 0

    def test_invalid_candidate(self, tmp_path):
        from soup_cli.utils.deploy_measure import measure_candidate

        tasks = _write_tasks(tmp_path)
        with pytest.raises(TypeError):
            measure_candidate(
                candidate=True,  # type: ignore[arg-type]
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen=lambda p: "",
            )

    def test_empty_candidate(self, tmp_path):
        from soup_cli.utils.deploy_measure import measure_candidate

        tasks = _write_tasks(tmp_path)
        with pytest.raises(ValueError):
            measure_candidate(
                candidate="",
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen=lambda p: "",
            )

    def test_null_byte_candidate(self, tmp_path):
        from soup_cli.utils.deploy_measure import measure_candidate

        tasks = _write_tasks(tmp_path)
        with pytest.raises(ValueError):
            measure_candidate(
                candidate="ev\x00il",
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen=lambda p: "",
            )


# --- pick_best --------------------------------------------------------------


class TestPickBest:
    def test_empty_returns_none(self):
        from soup_cli.utils.deploy_measure import pick_best

        assert pick_best([]) is None

    def test_first_ok_wins(self):
        from soup_cli.utils.deploy_measure import MeasureResult, pick_best

        rows = [
            MeasureResult("a", 0.8, 0.79, -0.01, "OK"),
            MeasureResult("b", 0.8, 0.78, -0.02, "MINOR"),
        ]
        assert pick_best(rows).candidate == "a"

    def test_no_ok_picks_highest_after(self):
        from soup_cli.utils.deploy_measure import MeasureResult, pick_best

        rows = [
            MeasureResult("a", 0.8, 0.5, -0.3, "MAJOR"),
            MeasureResult("b", 0.8, 0.6, -0.2, "MAJOR"),
            MeasureResult("c", 0.8, 0.55, -0.25, "MAJOR"),
        ]
        assert pick_best(rows).candidate == "b"


# --- cache load/save round-trip ---------------------------------------------


class TestCacheRoundtrip:
    def test_save_then_load(self, tmp_path):
        from soup_cli.utils.deploy_measure import load_cache, save_cache

        cache_path = tmp_path / "cache.json"
        payload = {"abc123": {"rows": [{"candidate": "gptq",
                                        "before": 0.8, "after": 0.79,
                                        "delta": -0.01, "verdict": "OK"}]}}
        save_cache(payload, str(cache_path))
        loaded = load_cache(str(cache_path))
        assert loaded == payload

    def test_load_missing_returns_empty(self, tmp_path):
        from soup_cli.utils.deploy_measure import load_cache

        assert load_cache(str(tmp_path / "missing.json")) == {}

    def test_load_malformed_returns_empty(self, tmp_path):
        from soup_cli.utils.deploy_measure import load_cache

        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{", encoding="utf-8")
        assert load_cache(str(bad)) == {}


# --- run_measure (full loop) ------------------------------------------------


class TestRunMeasure:
    def test_first_run_misses_then_hits(self, tmp_path):
        from soup_cli.utils.deploy_measure import run_measure

        tasks = _write_tasks(tmp_path)
        cache_path = tmp_path / "cache.json"

        def before(p):
            return "hello" if "hello" in p else "world"

        def after_factory(candidate):
            def gen(p):
                # awq matches; gptq always wrong (MAJOR drop)
                if candidate == "awq":
                    return "hello" if "hello" in p else "world"
                return "WRONG"
            return gen

        results1, hit1 = run_measure(
            profile_name="rtx-4090-24gb",
            base_sha="basetestsha",
            candidates=("awq", "gptq"),
            tasks_file=str(tasks),
            before_gen=before,
            after_gen_factory=after_factory,
            cache_path=str(cache_path),
        )
        assert hit1 is False
        assert [r.candidate for r in results1] == ["awq", "gptq"]
        assert results1[0].verdict == "OK"
        assert results1[1].verdict == "MAJOR"

        # Second invocation must hit cache and skip after_factory entirely
        called = {"count": 0}

        def boom_factory(candidate):
            called["count"] += 1
            return lambda p: "should not be called"

        results2, hit2 = run_measure(
            profile_name="rtx-4090-24gb",
            base_sha="basetestsha",
            candidates=("awq", "gptq"),
            tasks_file=str(tasks),
            before_gen=before,
            after_gen_factory=boom_factory,
            cache_path=str(cache_path),
        )
        assert hit2 is True
        assert called["count"] == 0
        assert [r.candidate for r in results2] == ["awq", "gptq"]

    def test_candidates_empty_rejected(self, tmp_path):
        from soup_cli.utils.deploy_measure import run_measure

        tasks = _write_tasks(tmp_path)
        with pytest.raises(ValueError):
            run_measure(
                profile_name="p", base_sha="b",
                candidates=(),
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen_factory=lambda c: (lambda p: ""),
                cache_path=str(tmp_path / "cache.json"),
            )

    def test_candidates_string_rejected(self, tmp_path):
        from soup_cli.utils.deploy_measure import run_measure

        tasks = _write_tasks(tmp_path)
        with pytest.raises(TypeError):
            run_measure(
                profile_name="p", base_sha="b",
                candidates="awq",  # type: ignore[arg-type]
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen_factory=lambda c: (lambda p: ""),
                cache_path=str(tmp_path / "cache.json"),
            )


# --- CLI plumbing -----------------------------------------------------------


class TestDeployAutopilotMeasureCLI:
    def test_help_lists_measure_flag(self):
        import typer
        from typer.testing import CliRunner

        from soup_cli.commands.deploy import autopilot

        app = typer.Typer()
        app.command()(autopilot)
        # Force wide terminal so Rich doesn't wrap option names; CI runners
        # default to 80-col which splits `--measure` mid-line.
        runner = CliRunner()
        result = runner.invoke(
            app, ["--help"], env={"COLUMNS": "200", "TERM": "dumb"},
        )
        assert result.exit_code == 0
        # Inspect registered click params directly so the assertion doesn't
        # depend on Rich's wrapping behaviour at all (CI runners default to
        # 80-col which splits int option names mid-line).
        click_cmd = typer.main.get_command(app)
        registered = {
            opt
            for param in click_cmd.params
            for opt in (param.opts + param.secondary_opts)
        }
        assert "--measure" in registered, registered
        assert "--tasks" in registered, registered

    def test_measure_without_tasks_rejected(self, tmp_path, monkeypatch):
        import typer
        from typer.testing import CliRunner

        from soup_cli.commands.deploy import autopilot

        monkeypatch.chdir(tmp_path)
        app = typer.Typer()
        app.command()(autopilot)
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["--target", "rtx-4090-24gb", "--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
             "--measure"],
        )
        assert result.exit_code != 0
        assert "--tasks" in result.output

    def test_measure_with_injected_generators(self, tmp_path, monkeypatch):
        """End-to-end with injected generators bypassing real model loading."""
        import typer
        from typer.testing import CliRunner

        from soup_cli.commands.deploy import autopilot
        from soup_cli.utils import deploy_measure as _dm

        monkeypatch.chdir(tmp_path)
        tasks = _write_tasks(tmp_path)

        # Inject generators
        def before(p):
            return "hello" if "hello" in p else "world"

        def after_factory(candidate):
            return lambda p: ("hello" if "hello" in p else "world")

        monkeypatch.setattr(
            _dm, "_DEPLOY_MEASURE_BEFORE_GEN", before, raising=False
        )
        monkeypatch.setattr(
            _dm, "_DEPLOY_MEASURE_AFTER_FACTORY", after_factory, raising=False
        )
        # Redirect cache to tmp
        monkeypatch.setenv(
            "SOUP_DEPLOY_AUTOPILOT_CACHE",
            str(tmp_path / "cache.json"),
        )

        app = typer.Typer()
        app.command()(autopilot)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--target", "rtx-4090-24gb",
                "--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--recipe-out", str(tmp_path / "recipe.yaml"),
                "--script-out", str(tmp_path / "deploy.sh"),
                "--measure",
                "--tasks", str(tasks),
                "--measure-candidates", "awq,gptq",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # L4: assert the specific verdict-recommendation line + that the
        # measured candidate name appears in the table.
        assert "Recommended" in result.output
        assert "awq" in result.output


# --- M3: _MAX_CANDIDATES upper bound ---------------------------------------


class TestMaxCandidatesCap:
    def test_too_many_candidates_rejected(self, tmp_path):
        from soup_cli.utils.deploy_measure import run_measure

        tasks = _write_tasks(tmp_path)
        with pytest.raises(ValueError, match="too many candidates"):
            run_measure(
                profile_name="p", base_sha="b",
                candidates=tuple(f"q{i}" for i in range(33)),
                tasks_file=str(tasks),
                before_gen=lambda p: "",
                after_gen_factory=lambda c: (lambda p: ""),
                cache_path=str(tmp_path / "cache.json"),
            )


# --- M4: cache symlink TOCTOU rejection ------------------------------------


class TestCacheSymlinkRejection:
    @pytest.mark.skipif(
        os.name == "nt", reason="symlink rejection POSIX-only"
    )
    def test_load_cache_rejects_symlink_target(self, tmp_path):
        from soup_cli.utils.deploy_measure import load_cache

        real = tmp_path / "real_cache.json"
        real.write_text('{"k": {"rows": []}}', encoding="utf-8")
        link = tmp_path / "link_cache.json"
        link.symlink_to(real)
        # load_cache must refuse to follow the symlink — returns {}
        assert load_cache(str(link)) == {}

    @pytest.mark.skipif(
        os.name == "nt", reason="symlink rejection POSIX-only"
    )
    def test_save_cache_refuses_symlink_target(self, tmp_path):
        from soup_cli.utils.deploy_measure import save_cache

        real = tmp_path / "real_target.json"
        real.write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        # save_cache silently refuses on a pre-placed symlink — no exception,
        # but the underlying real file must NOT be overwritten.
        original = real.read_text(encoding="utf-8")
        save_cache({"k": {"rows": []}}, str(link))
        assert real.read_text(encoding="utf-8") == original


# --- H3: render_measure_table markup-escape regression ---------------------


class TestRenderMeasureTableEscape:
    def test_candidate_with_markup_metacharacters_escaped(self):
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.deploy_measure import (
            MeasureResult,
            render_measure_table,
        )

        rows = [MeasureResult("[red]evil[/]", 0.8, 0.79, -0.01, "OK")]
        table = render_measure_table(rows)
        buf = StringIO()
        Console(file=buf, force_terminal=False, no_color=True, width=200).print(
            table
        )
        # The raw bracketed text must appear (escaped); the colour markup
        # must NOT have been interpreted as Rich styling.
        output = buf.getvalue()
        assert "[red]evil[/]" in output
