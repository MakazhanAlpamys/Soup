"""Release review of the commands that first ship in v0.76.0.

Security (commit A)
    1. ``strip_boilerplate`` ran a quadratic regex: one whitespace run in a
       poisoned row hung ``soup data clean --strip-boilerplate``.
    2. ``soup rewind`` and ``soup data clean`` printed dataset / log text through
       ``rich.markup.escape`` only, which passes ESC, BEL and C1 controls.
    3. ``soup bench train -o`` and ``soup rewind --json`` wrote with
       ``Path.write_text``: no cwd containment, a symlink was followed, and the
       bench path was only checked after the whole run.
    4. ``soup adapters check`` capped the tensor count, not the bytes.

Smoke findings (commit B)
    5. ``soup data clean --json``: the loader's warning went to stdout before the JSON.
    6. ``soup rewind --top -1`` dropped the last row; ``--top 0`` printed nothing.
    7. ``soup rewind``: a NaN-loss row ranked last at 0%, and the headline named
       the finite row instead.
"""

from __future__ import annotations

import json
import math
import re
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tests.test_rewind_command import (
    _build,
    _chat_config,
    _patch_tracker,
    _real_fingerprint,
    _run_entries,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _invoke(args):
    from soup_cli.cli import app

    return CliRunner().invoke(app, args)


#: The three kinds of byte that must never reach a terminal from untrusted text:
#: a C0 escape (ESC), a C0 bell (BEL, closes an OSC title), and a C1 CSI (U+009B).
_HOSTILE = "\x1b]0;pwned\x07 and \u009b31m done"


def _assert_no_control(text: str) -> None:
    assert "\x1b" not in text, repr(text)
    assert "\x07" not in text, repr(text)
    assert "\u009b" not in text, repr(text)


# ============================================================ 1. ReDoS =====


class TestStripBoilerplateIsLinear:
    """A poisoned dataset must not be able to hang ``soup data clean``."""

    BUDGET_SECONDS = 2.0

    def test_one_long_interior_whitespace_run_is_fast(self):
        from soup_cli.utils.data_clean import strip_boilerplate

        text = "start" + " " * 200_000 + "end"
        began = time.perf_counter()
        cleaned, modified = strip_boilerplate(text)
        elapsed = time.perf_counter() - began
        assert elapsed < self.BUDGET_SECONDS, f"{elapsed:.1f}s for a 200,000-char row"
        assert (cleaned, modified) == (text, False)

    def test_repeated_signoff_without_a_terminator_is_fast(self):
        """The unbounded ``[^.\\n]*`` inside the sign-off was quadratic by itself."""
        from soup_cli.utils.data_clean import strip_boilerplate

        text = "hope this helps let me know if you need anything " * 4000 + "\nfoo"
        assert len(text) > 190_000
        began = time.perf_counter()
        strip_boilerplate(text)
        elapsed = time.perf_counter() - began
        assert elapsed < self.BUDGET_SECONDS, f"{elapsed:.1f}s for a 190,000-char row"

    def test_the_command_path_is_fast_on_a_poisoned_row(self):
        from soup_cli.utils.data_clean import clean_dataset

        row = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a" + " " * 200_000 + "b"},
            ]
        }
        began = time.perf_counter()
        cleaned, report = clean_dataset([row], "chatml", strip_ai_boilerplate=True)
        elapsed = time.perf_counter() - began
        assert elapsed < self.BUDGET_SECONDS, f"{elapsed:.1f}s through clean_dataset"
        assert report.total_scanned == 1
        assert len(cleaned) == 1

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Answer.\n\nI hope this helps!", "Answer."),
            ("Answer.   \t\n  hope that helps.", "Answer."),
            ("Answer. I hope this helps! Please let me know if you need anything else.",
             "Answer."),
            ("Answer.\n\nLet me know if you have any further questions.", "Answer."),
            ("Answer.  \n Please let me know if you need more help!", "Answer."),
            ("Answer with a hope this helps in the middle. And more.",
             "Answer with a hope this helps in the middle. And more."),
        ],
    )
    def test_normal_signoffs_are_still_stripped_the_same(self, text, expected):
        from soup_cli.utils.data_clean import strip_boilerplate

        cleaned, _ = strip_boilerplate(text)
        assert cleaned == expected


# ===================================================== 2. terminal output ==


@pytest.fixture
def captured_console(monkeypatch):
    """Swap a command module's Console for one that writes into a buffer."""

    def install(module) -> StringIO:
        buffer = StringIO()
        monkeypatch.setattr(
            module,
            "console",
            Console(file=buffer, force_terminal=False, no_color=True, width=10_000),
        )
        return buffer

    return install


class TestRewindTerminalOutput:
    def _tracker(self, monkeypatch, tmp_path, config):
        _patch_tracker(
            monkeypatch,
            {
                "run_id": "r1",
                "output_dir": str(tmp_path),
                "status": "completed",
                "config_json": json.dumps(config),
            },
        )

    def test_a_dataset_row_preview_cannot_inject_terminal_controls(
        self, tmp_path, monkeypatch, captured_console
    ):
        import soup_cli.commands.rewind as rewind_cmd

        config = _chat_config(tmp_path)
        (tmp_path / "train.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": f"reply {index} {_HOSTILE}"},
                        ]
                    }
                )
                for index in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        _build(
            tmp_path / "rewind.jsonl",
            [(step, 0, [step % 3], [1.0], [10]) for step in range(1, 41)]
            + [(41, 0, [0, 1, 2], [1.0, 1.0, 8.0], [100, 100, 3000])],
            fingerprint=_real_fingerprint(config),
        )
        self._tracker(monkeypatch, tmp_path, config)
        buffer = captured_console(rewind_cmd)

        result = _invoke(["rewind", "--step", "41"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        rendered = buffer.getvalue()
        assert "reply 2" in rendered, "the preview must still be shown"
        _assert_no_control(rendered)

    def test_header_values_from_the_log_cannot_inject_terminal_controls(
        self, tmp_path, monkeypatch, captured_console
    ):
        import soup_cli.commands.rewind as rewind_cmd

        _build(tmp_path / "rewind.jsonl", _run_entries(), task=f"sft{_HOSTILE}")
        self._tracker(monkeypatch, tmp_path, {})
        buffer = captured_console(rewind_cmd)

        result = _invoke(["rewind", "--no-preview"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        rendered = buffer.getvalue()
        assert "Task: sft" in rendered
        _assert_no_control(rendered)

    def test_an_unreadable_log_error_cannot_inject_terminal_controls(
        self, tmp_path, monkeypatch, captured_console
    ):
        import soup_cli.commands.rewind as rewind_cmd
        import soup_cli.monitoring.rewind_log as rewind_log

        _build(tmp_path / "rewind.jsonl", _run_entries())
        self._tracker(monkeypatch, tmp_path, {})
        buffer = captured_console(rewind_cmd)

        def refuse(path):
            raise rewind_log.RewindLogError(f"bad line {_HOSTILE}")

        monkeypatch.setattr(rewind_log, "read_rewind_log", refuse)
        result = _invoke(["rewind"])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        rendered = buffer.getvalue()
        assert "Cannot read rewind log" in rendered
        _assert_no_control(rendered)


class TestDataCleanTerminalOutput:
    def _dataset(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text(
            json.dumps({"instruction": "q", "output": "a"}) + "\n", encoding="utf-8"
        )
        return path

    def test_a_load_failure_cannot_inject_terminal_controls(
        self, tmp_path, monkeypatch, captured_console
    ):
        import soup_cli.commands.data_clean as clean_cmd

        monkeypatch.chdir(tmp_path)
        dataset = self._dataset(tmp_path)
        buffer = captured_console(clean_cmd)

        def explode(path):
            raise ValueError(f"cannot parse {_HOSTILE}")

        monkeypatch.setattr(clean_cmd, "load_raw_data", explode)
        result = _invoke(["data", "clean", str(dataset)])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        rendered = buffer.getvalue()
        assert "Failed to load dataset" in rendered
        _assert_no_control(rendered)

    def test_a_write_failure_cannot_inject_terminal_controls(
        self, tmp_path, monkeypatch, captured_console
    ):
        import soup_cli.commands.data_clean as clean_cmd

        monkeypatch.chdir(tmp_path)
        dataset = self._dataset(tmp_path)
        buffer = captured_console(clean_cmd)

        def explode(text, output_path, **kwargs):
            raise OSError(f"disk said {_HOSTILE}")

        monkeypatch.setattr(clean_cmd, "atomic_write_text", explode)
        result = _invoke(["data", "clean", str(dataset), "-o", "out.jsonl"])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        rendered = buffer.getvalue()
        assert "Failed to write output file" in rendered
        _assert_no_control(rendered)


# ======================================================= 3. output paths ===


def _fake_bench_report() -> dict:
    return {
        "steps_measured": 3,
        "timing": {"warmup_steps_discarded": 1, "median_seconds": 0.1, "p95_seconds": 0.2},
        "tokens": {"useful": 5, "total": 10},
        "throughput": {"useful_tokens_per_second": 50.0},
        "valid": True,
        "failures": [],
    }


@pytest.fixture
def bench_workdir(tmp_path, monkeypatch):
    """A cwd with a config file, and the heavy steps of ``bench train`` replaced by
    sentinels that record whether they ran."""
    import soup_cli.bench.train_run as train_run
    import soup_cli.config.loader as config_loader
    import soup_cli.utils.gpu as gpu

    work = tmp_path / "work"
    work.mkdir()
    (work / "soup.yaml").write_text("base: x\n", encoding="utf-8")
    monkeypatch.chdir(work)

    calls: dict[str, int] = {"load_config": 0, "run_bench_train": 0}

    def fake_load_config(path):
        calls["load_config"] += 1
        return type("Cfg", (), {"backend": "transformers"})()

    def fake_run(cfg, **kwargs):
        calls["run_bench_train"] += 1
        return _fake_bench_report()

    monkeypatch.setattr(config_loader, "load_config", fake_load_config)
    monkeypatch.setattr(train_run, "bench_scope_error", lambda cfg: None)
    monkeypatch.setattr(train_run, "run_bench_train", fake_run)
    monkeypatch.setattr(gpu, "detect_device", lambda backend=None: ("cpu", "CPU"))
    return work, calls


class TestBenchTrainOutputPath:
    def test_an_output_outside_cwd_is_refused_before_any_work(self, bench_workdir):
        work, calls = bench_workdir
        outside = work.parent / "x.json"
        result = _invoke(["bench", "train", "--config", "soup.yaml", "-o", "../x.json"])
        out = _plain(result.output)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must stay under cwd" in out
        # The refusal names the path the user typed, not only its basename.
        assert "'../x.json'" in out
        assert calls == {"load_config": 0, "run_bench_train": 0}, "refusal came too late"
        assert not outside.exists()

    @pytest.mark.requires_symlink
    def test_a_symlink_output_is_refused_and_its_target_untouched(self, bench_workdir):
        work, calls = bench_workdir
        victim = work / "victim.txt"
        victim.write_text("precious", encoding="utf-8")
        (work / "link.json").symlink_to(victim)
        result = _invoke(["bench", "train", "--config", "soup.yaml", "-o", "link.json"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must not be a symlink" in _plain(result.output)
        assert calls == {"load_config": 0, "run_bench_train": 0}
        assert victim.read_text(encoding="utf-8") == "precious"

    def test_a_directory_output_is_refused_before_any_work(self, bench_workdir):
        work, calls = bench_workdir
        (work / "reports").mkdir()
        result = _invoke(["bench", "train", "--config", "soup.yaml", "-o", "reports"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must name a file, not a directory" in _plain(result.output)
        assert calls == {"load_config": 0, "run_bench_train": 0}

    def test_the_report_is_written_atomically_and_replaces_an_older_one(
        self, bench_workdir, monkeypatch
    ):
        import soup_cli.utils.paths as paths

        work, calls = bench_workdir
        (work / "r.json").write_text("{}", encoding="utf-8")
        real = paths.atomic_write_text
        seen: list[str] = []

        def spy(text, output_path, **kwargs):
            seen.append(output_path)
            return real(text, output_path, **kwargs)

        monkeypatch.setattr(paths, "atomic_write_text", spy)
        result = _invoke(["bench", "train", "--config", "soup.yaml", "-o", "r.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen == ["r.json"]
        assert json.loads((work / "r.json").read_text(encoding="utf-8"))["valid"] is True
        assert [p.name for p in work.iterdir() if p.name.startswith(".")] == []
        assert calls["run_bench_train"] == 1

    def test_a_write_failure_after_the_run_is_reported_not_a_traceback(
        self, bench_workdir, monkeypatch
    ):
        import soup_cli.utils.paths as paths

        def refuse(text, output_path, **kwargs):
            raise OSError(f"disk full {_HOSTILE}")

        monkeypatch.setattr(paths, "atomic_write_text", refuse)
        result = _invoke(["bench", "train", "--config", "soup.yaml", "-o", "r.json"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "Cannot write the report" in _plain(result.output)
        _assert_no_control(result.output)


class TestRewindJsonPath:
    def _ready(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _build(tmp_path / "rewind.jsonl", _run_entries())
        _patch_tracker(
            monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
        )

    def test_a_path_outside_cwd_is_refused_before_the_tracker_is_read(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.experiment.tracker as tracker_mod

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

        def must_not_run(self, *args, **kwargs):
            raise AssertionError("the tracker was opened before the output was validated")

        monkeypatch.setattr(tracker_mod.ExperimentTracker, "__init__", must_not_run)
        result = _invoke(["rewind", "--json", "../x.json"])
        out = _plain(result.output)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must stay under cwd" in out
        assert "'../x.json'" in out
        assert not (tmp_path / "x.json").exists()

    @pytest.mark.requires_symlink
    def test_a_symlink_path_is_refused_and_its_target_untouched(self, tmp_path, monkeypatch):
        self._ready(tmp_path, monkeypatch)
        victim = tmp_path / "victim.txt"
        victim.write_text("precious", encoding="utf-8")
        (tmp_path / "link.json").symlink_to(victim)
        result = _invoke(["rewind", "--no-preview", "--json", "link.json"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must not be a symlink" in _plain(result.output)
        assert victim.read_text(encoding="utf-8") == "precious"

    def test_a_directory_path_is_refused(self, tmp_path, monkeypatch):
        self._ready(tmp_path, monkeypatch)
        (tmp_path / "reports").mkdir()
        result = _invoke(["rewind", "--no-preview", "--json", "reports"])
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "must name a file, not a directory" in _plain(result.output)

    def test_the_json_is_written_atomically_and_replaces_an_older_one(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.utils.paths as paths

        self._ready(tmp_path, monkeypatch)
        (tmp_path / "out.json").write_text("stale", encoding="utf-8")
        real = paths.atomic_write_text
        seen: list[str] = []

        def spy(text, output_path, **kwargs):
            seen.append(output_path)
            return real(text, output_path, **kwargs)

        monkeypatch.setattr(paths, "atomic_write_text", spy)
        result = _invoke(["rewind", "--no-preview", "--json", "out.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert seen == ["out.json"]
        assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["step"] == 41


class TestRefusalShowsThePathAsTyped:
    """The shared helper named only the basename: ``'att_out.json'`` for ``../att_out.json``."""

    def test_the_helper_quotes_the_whole_path(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError) as caught:
            enforce_under_cwd_and_no_symlink("../att_out.json", "--output")
        assert str(caught.value) == "--output '../att_out.json' must stay under cwd"

    def test_a_control_character_in_the_path_is_escaped(self, tmp_path, monkeypatch):
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError) as caught:
            enforce_under_cwd_and_no_symlink("../\x1b[2Jx.json", "--output")
        assert "\x1b" not in str(caught.value)
        assert "\\x1b" in str(caught.value)


# ======================================================== 4. adapter size ===


class TestAdapterByteCap:
    def _adapter(self, directory: Path) -> Path:
        pytest.importorskip("safetensors")
        import numpy as np
        from safetensors.numpy import save_file

        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "base_model.model.layers.0.q_proj.lora_A.weight": np.ones((4, 8), "float32"),
                "base_model.model.layers.0.q_proj.lora_B.weight": np.ones((8, 4), "float32"),
            },
            str(directory / "adapter_model.safetensors"),
        )
        (directory / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 8}), encoding="utf-8"
        )
        return directory

    def test_the_cap_is_a_documented_two_gib(self):
        from soup_cli.utils import adapter_check

        assert adapter_check.MAX_ADAPTER_BYTES == 2 * 1024**3

    def test_a_file_above_the_cap_is_refused_naming_size_and_cap(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils import adapter_check

        monkeypatch.chdir(tmp_path)
        adapter = self._adapter(tmp_path / "adapter")
        size = (adapter / "adapter_model.safetensors").stat().st_size
        monkeypatch.setattr(adapter_check, "MAX_ADAPTER_BYTES", size - 1)

        def must_not_open(*args, **kwargs):
            raise AssertionError("the file was opened although it is over the cap")

        monkeypatch.setattr("safetensors.safe_open", must_not_open)
        with pytest.raises(ValueError) as caught:
            adapter_check.check_adapter("adapter")
        message = str(caught.value)
        assert f"{size:,} bytes" in message
        assert f"{size - 1:,} bytes" in message
        assert "adapter_model.safetensors" in message

    def test_a_file_at_the_cap_is_still_checked(self, tmp_path, monkeypatch):
        from soup_cli.utils import adapter_check

        monkeypatch.chdir(tmp_path)
        adapter = self._adapter(tmp_path / "adapter")
        size = (adapter / "adapter_model.safetensors").stat().st_size
        monkeypatch.setattr(adapter_check, "MAX_ADAPTER_BYTES", size)
        report = adapter_check.check_adapter("adapter")
        assert report.total_layers == 1


# ================================================= 5. data clean --json =====


class TestDataCleanJsonStdout:
    def test_stdout_is_pure_json_when_the_loader_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dataset = tmp_path / "rows.jsonl"
        dataset.write_text(
            json.dumps({"instruction": "q", "output": "a\x00b"})
            + "\n{this is not json\n"
            + json.dumps({"instruction": "q2", "output": "fine"})
            + "\n",
            encoding="utf-8",
        )
        result = _invoke(["data", "clean", str(dataset), "--dry-run", "--json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        payload = json.loads(result.stdout)
        assert payload["total_scanned"] == 2
        assert payload["dry_run"] is True
        # The warning is not lost: it moved to stderr.
        assert "invalid JSON on line 2" in _plain(result.stderr)

    def test_the_loader_console_is_put_back_afterwards(self, tmp_path, monkeypatch):
        import soup_cli.data.loader as loader

        monkeypatch.chdir(tmp_path)
        dataset = tmp_path / "rows.jsonl"
        dataset.write_text(json.dumps({"instruction": "q", "output": "a"}) + "\n")
        before = loader.console
        result = _invoke(["data", "clean", str(dataset), "--dry-run", "--json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert loader.console is before

    def test_the_human_output_keeps_the_warning_on_stdout(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dataset = tmp_path / "rows.jsonl"
        dataset.write_text(
            json.dumps({"instruction": "q", "output": "a"}) + "\n{nope\n", encoding="utf-8"
        )
        result = _invoke(["data", "clean", str(dataset), "--dry-run"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "invalid JSON on line 2" in _plain(result.stdout)


# ======================================================= 6. rewind --top ====


class TestRewindTopBound:
    @pytest.fixture
    def ready(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _build(tmp_path / "rewind.jsonl", _run_entries())
        _patch_tracker(
            monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
        )
        return tmp_path

    @pytest.mark.parametrize("value", ["-1", "0"])
    def test_a_top_below_one_is_a_usage_error(self, ready, value):
        result = _invoke(["rewind", "--step", "41", "--top", value])
        out = _plain(result.output)
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "Invalid value for '--top'" in out
        assert f"{value} is not in the range x>=1" in out

    def test_one_is_accepted_and_lists_exactly_one_row(self, ready, monkeypatch):
        import soup_cli.commands.rewind as rewind_cmd

        buffer = StringIO()
        monkeypatch.setattr(
            rewind_cmd,
            "console",
            Console(file=buffer, force_terminal=False, no_color=True, width=10_000),
        )
        result = _invoke(["rewind", "--step", "41", "--top", "1", "--no-preview", "--json",
                          "o.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert len(json.loads((ready / "o.json").read_text(encoding="utf-8"))["rows"]) == 1


# ================================================= 7. NaN headline ==========


def _nan_entries():
    """A flat 1.0 run whose step 20 holds one NaN-loss row and one ordinary row."""
    entries = [(step, 0, [step + 100], [1.0], [10]) for step in range(1, 31) if step != 20]
    entries.append((20, 0, [7, 8], [math.nan, 2.0], [10, 10]))
    return entries


class TestNonFiniteHeadline:
    @pytest.fixture
    def ready(self, tmp_path, monkeypatch):
        import soup_cli.commands.rewind as rewind_cmd

        monkeypatch.chdir(tmp_path)
        _build(tmp_path / "rewind.jsonl", _nan_entries())
        _patch_tracker(
            monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
        )
        buffer = StringIO()
        monkeypatch.setattr(
            rewind_cmd,
            "console",
            Console(file=buffer, force_terminal=False, no_color=True, width=10_000),
        )
        return tmp_path, buffer

    def test_the_headline_names_the_nan_row_not_the_finite_one(self, ready):
        work, buffer = ready
        result = _invoke(["rewind", "--no-preview", "--json", "o.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(buffer.getvalue())
        assert "row 7: loss is non-finite (nan) in step 20" in out
        assert "Inspect or drop this row, then retrain." in out
        assert "row 8: 100%" not in out

    def test_the_nan_row_is_listed_first_in_the_table_and_the_json(self, ready):
        work, _ = ready
        result = _invoke(["rewind", "--step", "20", "--top", "1", "--no-preview",
                          "--json", "o.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        rows = json.loads((work / "o.json").read_text(encoding="utf-8"))["rows"]
        # --top 1 would have dropped the NaN row when it sorted last.
        assert [row["row"] for row in rows] == [7]

    def test_every_non_finite_row_is_counted(self, tmp_path, monkeypatch):
        import soup_cli.commands.rewind as rewind_cmd

        monkeypatch.chdir(tmp_path)
        entries = [(s, 0, [s + 100], [1.0], [10]) for s in range(1, 31) if s != 20]
        entries.append((20, 0, [7, 8, 9], [math.nan, 2.0, math.nan], [10, 10, 10]))
        _build(tmp_path / "rewind.jsonl", entries)
        _patch_tracker(
            monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
        )
        buffer = StringIO()
        monkeypatch.setattr(
            rewind_cmd,
            "console",
            Console(file=buffer, force_terminal=False, no_color=True, width=10_000),
        )
        result = _invoke(["rewind", "--no-preview"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(buffer.getvalue())
        assert "row 7: loss is non-finite (nan) in step 20" in out
        assert "2 rows in this step are non-finite" in out

    def test_a_finite_step_keeps_the_share_headline(self, tmp_path, monkeypatch):
        import soup_cli.commands.rewind as rewind_cmd

        monkeypatch.chdir(tmp_path)
        _build(tmp_path / "rewind.jsonl", _run_entries())
        _patch_tracker(
            monkeypatch, {"run_id": "r1", "output_dir": str(tmp_path), "status": "completed"}
        )
        buffer = StringIO()
        monkeypatch.setattr(
            rewind_cmd,
            "console",
            Console(file=buffer, force_terminal=False, no_color=True, width=10_000),
        )
        result = _invoke(["rewind", "--no-preview"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _plain(buffer.getvalue())
        assert "row 13: 99% of step 41's loss, 3,000 tokens" in out
        assert "non-finite" not in out.split("row 13:")[1]


class TestRankRowsNonFinite:
    def test_a_nan_row_sorts_first_and_the_finite_shares_sum_to_one(self, tmp_path):
        from soup_cli.monitoring.rewind_log import RewindLog
        from soup_cli.utils.rewind import rank_rows

        run = _build(
            tmp_path / RewindLog.FILENAME,
            [(1, 0, [7, 8, 9], [2.0, math.nan, 4.0], [10, 10, 10])],
        )
        ranked = rank_rows(run, 1)
        assert [r.row for r in ranked] == [8, 9, 7]
        assert math.isnan(ranked[0].loss)
        assert ranked[0].share == 0.0
        assert sum(r.share for r in ranked) == pytest.approx(1.0)

    def test_an_infinite_row_does_not_turn_every_share_into_nan(self):
        """The log stores inf as null (read back as NaN), so a run object is faked here:
        ``inf / inf`` would otherwise make the finite row's share NaN as well."""
        from types import SimpleNamespace

        from soup_cli.utils.rewind import rank_rows

        batch = SimpleNamespace(rows=[7, 8], row_loss=[math.inf, 2.0], row_tokens=[10, 10])
        run = SimpleNamespace(batches_for=lambda step: [batch])
        ranked = rank_rows(run, 1)
        assert ranked[0].row == 7
        assert all(math.isfinite(r.share) for r in ranked)
        assert ranked[1].share == pytest.approx(1.0)
