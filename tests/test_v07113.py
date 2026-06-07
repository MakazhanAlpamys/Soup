"""v0.71.13 — "Prompt-compile family" live wiring.

Closes #225 (soup compile), #226 (soup distill-prompt), #227 (soup
compile-tools), #229 (soup local-rl train). Lifts the v0.68.1 deferred
stubs from ``utils/prompt_compile.py`` / ``prompt_distill.py`` /
``compile_tools.py`` / ``local_rl.py``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest
import typer
from typer.testing import CliRunner

runner = CliRunner()


# ===========================================================================
# #229 — soup local-rl train (live DPO/KTO/ORPO runner + scheduler)
# ===========================================================================


def _seed_db(path: str, n_pairs: int) -> None:
    from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

    init_local_rl_db(path)
    for i in range(n_pairs):
        record_thumb(
            db_path=path, prompt=f"q{i}", response=f"good{i}", thumb="up"
        )
        record_thumb(
            db_path=path, prompt=f"q{i}", response=f"bad{i}", thumb="down"
        )


class TestStateTable:
    def test_init_creates_state_table(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db

        init_local_rl_db("db.sqlite")
        with sqlite3.connect(str(tmp_path / "db.sqlite")) as conn:
            names = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "state" in names

    def test_get_set_state_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            get_state,
            init_local_rl_db,
            set_state,
        )

        init_local_rl_db("db.sqlite")
        assert get_state("db.sqlite", "k") is None
        set_state("db.sqlite", "k", "v1")
        assert get_state("db.sqlite", "k") == "v1"
        set_state("db.sqlite", "k", "v2")
        assert get_state("db.sqlite", "k") == "v2"

    def test_get_state_empty_key_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db, set_state

        init_local_rl_db("db.sqlite")
        with pytest.raises(ValueError):
            set_state("db.sqlite", "", "v")

    def test_set_state_non_str_value_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db, set_state

        init_local_rl_db("db.sqlite")
        with pytest.raises(TypeError):
            set_state("db.sqlite", "k", 5)

    def test_state_table_auto_migrates_old_db(self, tmp_path, monkeypatch):
        # A pre-v0.71.13 DB without the state table still works.
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import get_state

        with sqlite3.connect(str(tmp_path / "old.sqlite")) as conn:
            conn.execute(
                "CREATE TABLE thumbs (id INTEGER PRIMARY KEY, ts REAL, "
                "prompt TEXT, response TEXT, thumb TEXT)"
            )
            conn.commit()
        assert get_state("old.sqlite", "k") is None


class TestCountNewThumbs:
    def test_counts_all_when_since_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import count_new_thumbs_since

        _seed_db("db.sqlite", 3)  # 6 thumbs
        assert count_new_thumbs_since("db.sqlite", None) == 6

    def test_counts_after_ts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            count_new_thumbs_since,
            init_local_rl_db,
            record_thumb,
        )

        init_local_rl_db("db.sqlite")
        record_thumb(db_path="db.sqlite", prompt="a", response="x", thumb="up")
        with sqlite3.connect(str(tmp_path / "db.sqlite")) as conn:
            cutoff = conn.execute("SELECT MAX(ts) FROM thumbs").fetchone()[0]
        record_thumb(db_path="db.sqlite", prompt="b", response="y", thumb="down")
        assert count_new_thumbs_since("db.sqlite", cutoff) == 1

    def test_bool_since_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import count_new_thumbs_since

        _seed_db("db.sqlite", 1)
        with pytest.raises(TypeError):
            count_new_thumbs_since("db.sqlite", True)


class TestPairsToRows:
    def _pairs(self):
        from soup_cli.utils.local_rl import DpoPair

        return (DpoPair(prompt="p", chosen="c", rejected="r"),)

    def test_dpo_shape(self):
        from soup_cli.utils.local_rl import pairs_to_rows

        rows = pairs_to_rows(self._pairs(), "dpo")
        assert rows == [{"prompt": "p", "chosen": "c", "rejected": "r"}]

    def test_orpo_same_as_dpo(self):
        from soup_cli.utils.local_rl import pairs_to_rows

        assert pairs_to_rows(self._pairs(), "orpo") == [
            {"prompt": "p", "chosen": "c", "rejected": "r"}
        ]

    def test_kto_two_rows_per_pair(self):
        from soup_cli.utils.local_rl import pairs_to_rows

        rows = pairs_to_rows(self._pairs(), "kto")
        assert rows == [
            {"prompt": "p", "completion": "c", "label": True},
            {"prompt": "p", "completion": "r", "label": False},
        ]

    def test_unknown_method_rejected(self):
        from soup_cli.utils.local_rl import pairs_to_rows

        with pytest.raises(ValueError):
            pairs_to_rows(self._pairs(), "ppo")

    def test_non_pair_rejected(self):
        from soup_cli.utils.local_rl import pairs_to_rows

        with pytest.raises(TypeError):
            pairs_to_rows(({"prompt": "p"},), "dpo")


class TestRunNightlyTrain:
    def _cfg(self, db, method="dpo"):
        from soup_cli.utils.local_rl import LocalRLConfig

        return LocalRLConfig(
            backend="ollama",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            db_path=db,
            train_method=method,
        )

    def test_first_run_trains_and_stamps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            get_state,
            run_nightly_train,
        )

        _seed_db("db.sqlite", 5)
        seen = {}

        def fake_train(*, base_model, pairs_path, output_dir, train_method):
            seen["base_model"] = base_model
            seen["train_method"] = train_method
            seen["output_dir"] = output_dir
            with open(pairs_path, encoding="utf-8") as fh:
                seen["rows"] = [json.loads(line) for line in fh if line.strip()]

        res = run_nightly_train(
            self._cfg("db.sqlite"),
            min_pairs=3,
            output_dir="adapter",
            train_fn=fake_train,
        )
        assert res.status == "trained"
        assert res.num_pairs == 5
        assert seen["base_model"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert seen["train_method"] == "dpo"
        assert seen["output_dir"] == "adapter"
        assert len(seen["rows"]) == 5
        assert set(seen["rows"][0]) == {"prompt", "chosen", "rejected"}
        # last_train_at stamped.
        assert get_state("db.sqlite", "last_train_at") is not None

    def test_kto_rows_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import run_nightly_train

        _seed_db("db.sqlite", 4)
        seen = {}

        def fake_train(*, base_model, pairs_path, output_dir, train_method):
            with open(pairs_path, encoding="utf-8") as fh:
                seen["rows"] = [json.loads(line) for line in fh if line.strip()]

        run_nightly_train(
            self._cfg("db.sqlite", "kto"),
            min_pairs=1,
            train_fn=fake_train,
        )
        # 4 pairs -> 8 KTO rows.
        assert len(seen["rows"]) == 8
        assert set(seen["rows"][0]) == {"prompt", "completion", "label"}

    def test_skip_no_new_thumbs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import run_nightly_train, set_state

        _seed_db("db.sqlite", 5)
        # Stamp last_train_at AFTER all thumbs -> no new thumbs.
        import time

        set_state("db.sqlite", "last_train_at", repr(time.time() + 100.0))
        calls = []
        res = run_nightly_train(
            self._cfg("db.sqlite"),
            min_pairs=1,
            train_fn=lambda **kw: calls.append(kw),
        )
        assert res.status == "skipped_no_new_thumbs"
        assert calls == []

    def test_skip_insufficient_pairs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import run_nightly_train

        _seed_db("db.sqlite", 2)
        calls = []
        res = run_nightly_train(
            self._cfg("db.sqlite"),
            min_pairs=10,
            train_fn=lambda **kw: calls.append(kw),
        )
        assert res.status == "skipped_insufficient_pairs"
        assert res.num_pairs == 2
        assert calls == []

    def test_non_config_rejected(self):
        from soup_cli.utils.local_rl import run_nightly_train

        with pytest.raises(TypeError):
            run_nightly_train({"backend": "ollama"})

    def test_bad_min_pairs_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import run_nightly_train

        _seed_db("db.sqlite", 1)
        with pytest.raises(ValueError):
            run_nightly_train(self._cfg("db.sqlite"), min_pairs=0, train_fn=lambda **k: None)
        with pytest.raises(TypeError):
            run_nightly_train(self._cfg("db.sqlite"), min_pairs=True, train_fn=lambda **k: None)

    def test_non_callable_train_fn_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import run_nightly_train

        _seed_db("db.sqlite", 5)
        with pytest.raises(TypeError):
            run_nightly_train(self._cfg("db.sqlite"), train_fn=123)

    def test_default_train_fn_renders_schema_valid_config(self, tmp_path, monkeypatch):
        """The real ``_default_train_fn`` must render a YAML that the SoupConfig
        schema accepts — the injected-``train_fn`` tests never exercise the YAML
        rendering, so this guards the ``output:`` shape (regression: it was once
        emitted as ``{"dir": ...}`` which the schema rejects as not-a-string)."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils import local_rl as lr

        captured = {}

        def fake_run(argv, check):  # noqa: ARG001 — capture the rendered config
            cfg_idx = argv.index("--config") + 1
            with open(argv[cfg_idx], encoding="utf-8") as fh:
                captured["yaml"] = fh.read()

        # ``_default_train_fn`` does a lazy ``import subprocess`` -> the global
        # module; patch that module's ``run``.
        import subprocess as _sp

        monkeypatch.setattr(_sp, "run", fake_run)

        with open("pairs.jsonl", "w", encoding="utf-8") as fh:
            fh.write('{"prompt": "p", "chosen": "c", "rejected": "r"}\n')

        lr._default_train_fn(
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            pairs_path="pairs.jsonl",
            output_dir="adapter_out",
            train_method="dpo",
        )
        assert "yaml" in captured, "subprocess.run was not invoked"
        cfg = load_config_from_string(captured["yaml"])  # must NOT raise
        assert cfg.output == "adapter_out"
        assert cfg.task == "dpo"


class TestScheduler:
    def test_build_train_argv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db
        from soup_cli.utils.local_rl_scheduler import build_train_argv

        init_local_rl_db("db.sqlite")
        argv = build_train_argv(
            soup_python=sys.executable,
            db_path="db.sqlite",
            model="org/model",
            train_method="dpo",
        )
        assert argv[0] == sys.executable
        assert "local-rl" in argv and "train" in argv and "--once" in argv
        assert "db.sqlite" in argv and "org/model" in argv
        # No shell metacharacters injected — it's a plain list.
        assert all(isinstance(a, str) for a in argv)

    def test_systemd_service_has_execstart(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db
        from soup_cli.utils.local_rl_scheduler import render_systemd_service

        init_local_rl_db("db.sqlite")
        out = render_systemd_service(
            soup_python=sys.executable,
            db_path="db.sqlite",
            model="org/model",
            train_method="dpo",
        )
        assert "ExecStart=" in out
        assert "Type=oneshot" in out
        assert "org/model" in out

    def test_systemd_timer_oncalendar(self):
        from soup_cli.utils.local_rl_scheduler import render_systemd_timer

        out = render_systemd_timer(hour=3, minute=0)
        assert "OnCalendar=*-*-* 03:00:00" in out
        assert "WantedBy=timers.target" in out

    def test_systemd_timer_bad_hour(self):
        from soup_cli.utils.local_rl_scheduler import render_systemd_timer

        with pytest.raises(ValueError):
            render_systemd_timer(hour=24, minute=0)
        with pytest.raises(TypeError):
            render_systemd_timer(hour=True, minute=0)

    def test_launchd_plist(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db
        from soup_cli.utils.local_rl_scheduler import render_launchd_plist

        init_local_rl_db("db.sqlite")
        out = render_launchd_plist(
            soup_python=sys.executable,
            db_path="db.sqlite",
            model="org/model",
            train_method="dpo",
            hour=3,
            minute=30,
        )
        assert "StartCalendarInterval" in out
        assert "<integer>3</integer>" in out
        assert "<integer>30</integer>" in out
        assert "com.soup.local-rl" in out

    def test_write_scheduler_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db
        from soup_cli.utils.local_rl_scheduler import (
            LAUNCHD_PLIST_NAME,
            SYSTEMD_SERVICE_NAME,
            SYSTEMD_TIMER_NAME,
            write_scheduler_files,
        )

        init_local_rl_db("db.sqlite")
        written = write_scheduler_files(
            "sched",
            soup_python=sys.executable,
            db_path="db.sqlite",
            model="org/model",
            train_method="dpo",
        )
        for name in (SYSTEMD_SERVICE_NAME, SYSTEMD_TIMER_NAME, LAUNCHD_PLIST_NAME):
            assert name in written
            assert os.path.isfile(written[name])

    def test_systemd_quote_escapes(self):
        from soup_cli.utils.local_rl_scheduler import _systemd_quote

        assert _systemd_quote('a "b" c') == '"a \\"b\\" c"'
        assert _systemd_quote("a\\b") == '"a\\\\b"'


class TestTrainCli:
    def _import_app(self):
        from soup_cli.commands.local_rl import app

        return app

    def test_once_runs_live(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _seed_db("db.sqlite", 5)

        import soup_cli.utils.local_rl as lr

        calls = []
        monkeypatch.setattr(
            lr, "_default_train_fn", lambda **kw: calls.append(kw)
        )
        res = runner.invoke(
            self._import_app(),
            [
                "train",
                "--db",
                "db.sqlite",
                "--model",
                "org/model",
                "--train-method",
                "dpo",
                "--once",
                "--min-pairs",
                "1",
                "--output",
                "adapter",
            ],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert len(calls) == 1
        assert calls[0]["base_model"] == "org/model"

    def test_once_skip_insufficient(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _seed_db("db.sqlite", 1)
        import soup_cli.utils.local_rl as lr

        monkeypatch.setattr(
            lr, "_default_train_fn", lambda **kw: pytest.fail("should not train")
        )
        res = runner.invoke(
            self._import_app(),
            [
                "train",
                "--db",
                "db.sqlite",
                "--model",
                "org/model",
                "--once",
                "--min-pairs",
                "10",
            ],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "Skipped" in res.output

    def test_no_once_renders_scheduler(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db

        init_local_rl_db("db.sqlite")
        res = runner.invoke(
            self._import_app(),
            [
                "train",
                "--db",
                "db.sqlite",
                "--model",
                "org/model",
                "--scheduler-dir",
                "sched",
            ],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert os.path.isfile(str(tmp_path / "sched" / "soup-local-rl.timer"))

    def test_no_longer_deferred(self):
        # The CLI must NOT surface the v0.68.1 deferred marker any more.
        import soup_cli.utils.local_rl as lr

        assert "deferred" not in (lr.run_nightly_train.__doc__ or "").lower()


# ===========================================================================
# #226 — soup distill-prompt (live teacher/student dataset prep)
# ===========================================================================


def _write_traces(path: str, prompts):
    with open(path, "w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps({"prompt": p}) + "\n")


class TestExtractPrompt:
    def test_prompt_field(self):
        from soup_cli.utils.prompt_distill import extract_prompt

        assert extract_prompt({"prompt": "hi"}) == "hi"

    def test_input_instruction_question(self):
        from soup_cli.utils.prompt_distill import extract_prompt

        assert extract_prompt({"input": "a"}) == "a"
        assert extract_prompt({"instruction": "b"}) == "b"
        assert extract_prompt({"question": "c"}) == "c"

    def test_messages_last_user(self):
        from soup_cli.utils.prompt_distill import extract_prompt

        row = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        }
        assert extract_prompt(row) == "second"

    def test_no_prompt_returns_none(self):
        from soup_cli.utils.prompt_distill import extract_prompt

        assert extract_prompt({"foo": "bar"}) is None
        assert extract_prompt({"prompt": "   "}) is None
        assert extract_prompt("not a mapping") is None


class TestPrepareDistillDataset:
    def _plan(self, tmp_path, strategy):
        from soup_cli.utils.prompt_distill import build_distill_prompt_plan

        traces = str(tmp_path / "traces.jsonl")
        _write_traces(traces, ["q1", "q2"])
        return build_distill_prompt_plan(
            traces_path="traces.jsonl",
            teacher="qwen2.5:0.5b",
            student="smol",
            strategy=strategy,
            output_path="out.jsonl",
        )

    def test_sft_writes_messages(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "sft")
        n = prepare_distill_dataset(
            plan, teacher_fn=lambda p: {"text": f"T:{p}"}
        )
        assert n == 2
        with open(tmp_path / "out.jsonl", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert rows[0]["messages"][0] == {"role": "user", "content": "q1"}
        assert rows[0]["messages"][1] == {"role": "assistant", "content": "T:q1"}

    def test_kl_same_shape_as_sft(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "kl")
        n = prepare_distill_dataset(plan, teacher_fn=lambda p: {"text": "T"})
        assert n == 2
        with open(tmp_path / "out.jsonl", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert "messages" in rows[0]

    def test_preference_uses_student(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "preference")
        n = prepare_distill_dataset(
            plan,
            teacher_fn=lambda p: {"text": "teach"},
            student_fn=lambda p: {"text": "stud"},
        )
        assert n == 2
        with open(tmp_path / "out.jsonl", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert rows[0] == {"prompt": "q1", "chosen": "teach", "rejected": "stud"}

    def test_empty_teacher_reply_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "sft")
        n = prepare_distill_dataset(plan, teacher_fn=lambda p: {"text": ""})
        assert n == 0

    def test_max_rows_cap(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "sft")
        n = prepare_distill_dataset(
            plan, teacher_fn=lambda p: {"text": "T"}, max_rows=1
        )
        assert n == 1

    def test_bad_max_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "sft")
        with pytest.raises(ValueError):
            prepare_distill_dataset(plan, teacher_fn=lambda p: {"text": "T"}, max_rows=0)
        with pytest.raises(ValueError):
            prepare_distill_dataset(plan, teacher_fn=lambda p: {"text": "T"}, max_rows=True)

    def test_non_plan_rejected(self):
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        with pytest.raises(TypeError):
            prepare_distill_dataset({"traces_path": "x"})

    def test_teacher_exception_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        plan = self._plan(tmp_path, "sft")

        def boom(p):
            raise RuntimeError("provider down")

        n = prepare_distill_dataset(plan, teacher_fn=boom)
        assert n == 0


class TestDistillPromptCli:
    def test_cli_live(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.distill_prompt import distill_prompt_cmd

        _write_traces(str(tmp_path / "traces.jsonl"), ["a", "b"])
        import soup_cli.utils.prompt_distill as pd

        monkeypatch.setattr(
            pd,
            "_build_provider_fn",
            lambda *a, **k: (lambda p: {"text": f"R:{p}"}),
        )
        app = typer.Typer()
        app.command()(distill_prompt_cmd)
        res = runner.invoke(
            app,
            [
                "--traces",
                "traces.jsonl",
                "--teacher",
                "qwen2.5:0.5b",
                "--student",
                "smol",
                "--strategy",
                "sft",
                "--output",
                "out.jsonl",
            ],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert os.path.isfile(str(tmp_path / "out.jsonl"))

    def test_cli_no_longer_deferred(self):
        import soup_cli.utils.prompt_distill as pd

        assert "deferred" not in (
            pd.prepare_distill_dataset.__doc__ or ""
        ).lower()


# ===========================================================================
# #225 — soup compile (DSPy / GEPA / TextGrad dispatcher)
# ===========================================================================


class TestLoadEvalExamples:
    def test_json_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import load_eval_examples

        (tmp_path / "eval.json").write_text(
            json.dumps([{"q": "1"}, {"q": "2"}, "skip-non-dict"]),
            encoding="utf-8",
        )
        ex = load_eval_examples("eval.json")
        assert ex == [{"q": "1"}, {"q": "2"}]

    def test_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import load_eval_examples

        (tmp_path / "eval.jsonl").write_text(
            '{"q": "1"}\n\n{"q": "2"}\nbad-line\n', encoding="utf-8"
        )
        ex = load_eval_examples("eval.jsonl")
        assert ex == [{"q": "1"}, {"q": "2"}]

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        sub = tmp_path / "work"
        sub.mkdir()
        monkeypatch.chdir(sub)
        (tmp_path / "evil.json").write_text("[]", encoding="utf-8")
        from soup_cli.utils.prompt_compile import load_eval_examples

        with pytest.raises(ValueError):
            load_eval_examples(str(tmp_path / "evil.json"))


class TestRunCompileSeam:
    def _plan(self, tmp_path):
        from soup_cli.utils.prompt_compile import build_compile_plan

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        return build_compile_plan(
            program_path="prog.py",
            eval_suite_path="eval.jsonl",
            optimizer="mipro",
            max_iters=3,
            output_path="out.py",
        )

    def test_override_seam(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        plan = self._plan(tmp_path)
        monkeypatch.setattr(
            pc,
            "_OPTIMIZER_RUN_OVERRIDE",
            lambda p: pc.CompileResult(
                program_text="optimised!", score=0.9, iterations=3, converged=True
            ),
        )
        result = pc.run_compile(plan)
        assert result.program_text == "optimised!"
        assert result.score == 0.9

    def test_override_must_return_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        plan = self._plan(tmp_path)
        monkeypatch.setattr(pc, "_OPTIMIZER_RUN_OVERRIDE", lambda p: "nope")
        with pytest.raises(TypeError):
            pc.run_compile(plan)

    def test_non_plan_rejected(self):
        from soup_cli.utils.prompt_compile import run_compile

        with pytest.raises(TypeError):
            run_compile({"optimizer": "mipro"})

    def test_missing_dspy_friendly_importerror(self, tmp_path, monkeypatch):
        # dspy/textgrad/gepa are genuinely absent in this env -> the real
        # branch raises a friendly ImportError naming the [compile] extra.
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import run_compile

        plan = self._plan(tmp_path)  # optimizer=mipro -> dspy branch
        with pytest.raises(ImportError) as exc:
            run_compile(plan)
        assert "soup-cli[compile]" in str(exc.value)

    def test_missing_textgrad_friendly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        plan = build_compile_plan(
            program_path="prog.py",
            eval_suite_path="eval.jsonl",
            optimizer="textgrad",
            max_iters=2,
            output_path="out.py",
        )
        with pytest.raises(ImportError) as exc:
            run_compile(plan)
        assert "TextGrad" in str(exc.value)

    def test_missing_gepa_friendly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        plan = build_compile_plan(
            program_path="prog.py",
            eval_suite_path="eval.jsonl",
            optimizer="gepa",
            max_iters=2,
            output_path="out.py",
        )
        with pytest.raises(ImportError) as exc:
            run_compile(plan)
        assert "GEPA" in str(exc.value)

    def test_resolve_program_via_get_program(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        (tmp_path / "prog2.py").write_text(
            "def get_program():\n    return 42\n", encoding="utf-8"
        )
        mod = pc._load_program_module("prog2.py")
        assert pc._resolve_program(mod) == 42

    def test_resolve_program_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        (tmp_path / "prog3.py").write_text("x = 1\n", encoding="utf-8")
        mod = pc._load_program_module("prog3.py")
        with pytest.raises(ValueError):
            pc._resolve_program(mod)


class TestCompileCli:
    def test_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_cmd import compile_cmd

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        app = typer.Typer()
        app.command()(compile_cmd)
        res = runner.invoke(
            app,
            ["prog.py", "--eval", "eval.jsonl", "--optimizer", "mipro", "--plan-only"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_live_via_seam(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_cmd import compile_cmd

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        import soup_cli.utils.prompt_compile as pc

        monkeypatch.setattr(
            pc,
            "_OPTIMIZER_RUN_OVERRIDE",
            lambda p: pc.CompileResult(
                program_text="# compiled\n", score=0.5, iterations=3, converged=True
            ),
        )
        app = typer.Typer()
        app.command()(compile_cmd)
        res = runner.invoke(
            app, ["prog.py", "--eval", "eval.jsonl", "--optimizer", "mipro"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert (tmp_path / "compiled_program.py").read_text(
            encoding="utf-8"
        ) == "# compiled\n"

    def test_live_missing_dep_exit2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_cmd import compile_cmd

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        app = typer.Typer()
        app.command()(compile_cmd)
        res = runner.invoke(
            app, ["prog.py", "--eval", "eval.jsonl", "--optimizer", "mipro"]
        )
        assert res.exit_code == 2
        assert "compile" in res.output.lower()

    def test_unknown_optimizer_exit2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_cmd import compile_cmd

        (tmp_path / "prog.py").write_text("program = 1\n", encoding="utf-8")
        (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")
        app = typer.Typer()
        app.command()(compile_cmd)
        res = runner.invoke(
            app, ["prog.py", "--eval", "eval.jsonl", "--optimizer", "nope"]
        )
        assert res.exit_code == 2


# ===========================================================================
# #227 — soup compile-tools (TextGrad / GEPA tool-schema optimiser)
# ===========================================================================


_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "t", "version": "1"},
    "paths": {
        "/widgets": {
            "get": {
                "operationId": "listWidgets",
                "description": "List all the widgets",
            },
            "post": {
                "operationId": "createWidget",
                "description": "Create a widget",
            },
        }
    },
}


def _write_spec(tmp_path, name="spec.json"):
    (tmp_path / name).write_text(json.dumps(_OPENAPI_SPEC), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text('{"q":"1"}\n', encoding="utf-8")


class TestOptimiseDescription:
    def test_override_returns_str(self, monkeypatch):
        import soup_cli.utils.compile_tools as ct

        monkeypatch.setattr(
            ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: d.upper()
        )
        assert ct._optimise_description("hi", [], "textgrad") == "HI"

    def test_override_non_str_rejected(self, monkeypatch):
        import soup_cli.utils.compile_tools as ct

        monkeypatch.setattr(ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: 5)
        with pytest.raises(TypeError):
            ct._optimise_description("hi", [], "textgrad")

    def test_missing_textgrad_friendly(self):
        from soup_cli.utils.compile_tools import _optimise_description

        with pytest.raises(ImportError) as exc:
            _optimise_description("hi", [], "textgrad")
        assert "soup-cli[compile]" in str(exc.value)

    def test_missing_gepa_friendly(self):
        from soup_cli.utils.compile_tools import _optimise_description

        with pytest.raises(ImportError) as exc:
            _optimise_description("hi", [], "gepa")
        assert "GEPA" in str(exc.value)


class TestRunToolCompile:
    def _plan(self, tmp_path, output="out.json"):
        from soup_cli.utils.compile_tools import build_tool_compile_plan

        _write_spec(tmp_path)
        return build_tool_compile_plan(
            spec_path="spec.json",
            eval_suite_path="eval.jsonl",
            optimizer="textgrad",
            output_path=output,
        )

    def test_json_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.compile_tools as ct

        monkeypatch.setattr(
            ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: f"OPT:{d}"
        )
        plan = self._plan(tmp_path)
        n = ct.run_tool_compile(plan)
        assert n == 2
        out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert out["spec_kind"] == "openapi"
        tools = {t["tool"]: t["description"] for t in out["tools"]}
        assert tools["listWidgets"] == "OPT:List all the widgets"

    def test_yaml_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.compile_tools as ct

        monkeypatch.setattr(ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: "X")
        plan = self._plan(tmp_path, output="out.yaml")
        ct.run_tool_compile(plan)
        import yaml

        out = yaml.safe_load((tmp_path / "out.yaml").read_text(encoding="utf-8"))
        assert out["tools"][0]["description"] == "X"

    def test_non_plan_rejected(self):
        from soup_cli.utils.compile_tools import run_tool_compile

        with pytest.raises(TypeError):
            run_tool_compile({"spec_path": "x"})

    def test_live_missing_dep(self, tmp_path, monkeypatch):
        # Real branch (no override) -> textgrad absent -> friendly ImportError.
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.compile_tools import run_tool_compile

        plan = self._plan(tmp_path)
        with pytest.raises(ImportError):
            run_tool_compile(plan)


class TestCompileToolsCli:
    def test_plan_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_tools import compile_tools_cmd

        _write_spec(tmp_path)
        app = typer.Typer()
        app.command()(compile_tools_cmd)
        res = runner.invoke(
            app, ["spec.json", "--eval", "eval.jsonl", "--plan-only"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_live_via_seam(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.compile_tools as ct
        from soup_cli.commands.compile_tools import compile_tools_cmd

        _write_spec(tmp_path)
        monkeypatch.setattr(ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: "OK")
        app = typer.Typer()
        app.command()(compile_tools_cmd)
        res = runner.invoke(
            app, ["spec.json", "--eval", "eval.jsonl", "--output", "out.json"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert os.path.isfile(str(tmp_path / "out.json"))

    def test_live_missing_dep_exit2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.commands.compile_tools import compile_tools_cmd

        _write_spec(tmp_path)
        app = typer.Typer()
        app.command()(compile_tools_cmd)
        res = runner.invoke(
            app, ["spec.json", "--eval", "eval.jsonl", "--output", "out.json"]
        )
        assert res.exit_code == 2


# ===========================================================================
# Review-fix coverage
# ===========================================================================


class TestReviewFixes:
    def test_db_path_rejects_newline(self, tmp_path, monkeypatch):
        # SEC HIGH: db_path flows into a systemd ExecStart -> reject newlines.
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import validate_db_path

        with pytest.raises(ValueError):
            validate_db_path("a\nb.db")
        with pytest.raises(ValueError):
            validate_db_path("a\rb.db")

    def test_model_rejects_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import LocalRLConfig, init_local_rl_db

        init_local_rl_db("db.sqlite")
        with pytest.raises(ValueError):
            LocalRLConfig(
                backend="ollama",
                model="org/model\nExecStartPre=/bin/sh -c evil",
                db_path="db.sqlite",
                train_method="dpo",
            )

    def test_scheduler_model_rejects_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import init_local_rl_db
        from soup_cli.utils.local_rl_scheduler import build_train_argv

        init_local_rl_db("db.sqlite")
        with pytest.raises(ValueError):
            build_train_argv(
                soup_python=sys.executable,
                db_path="db.sqlite",
                model="m\nevil",
                train_method="dpo",
            )

    def test_systemd_quote_rejects_newline(self):
        from soup_cli.utils.local_rl_scheduler import _systemd_quote

        with pytest.raises(ValueError):
            _systemd_quote("a\nb")

    def test_stamp_before_train_keeps_concurrent_thumbs(
        self, tmp_path, monkeypatch
    ):
        # CORRECTNESS MEDIUM: thumbs recorded DURING the train window keep
        # ts > last_train_at and are counted by the next run (not dropped).
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            LocalRLConfig,
            count_new_thumbs_since,
            get_state,
            record_thumb,
            run_nightly_train,
        )

        _seed_db("db.sqlite", 5)
        cfg = LocalRLConfig(
            backend="ollama",
            model="org/model",
            db_path="db.sqlite",
            train_method="dpo",
        )

        def slow_train(*, base_model, pairs_path, output_dir, train_method):
            # Simulate a thumb landing mid-train. Sleep one clock tick first so
            # the thumbs' time.time() ts is strictly greater than run_started on
            # coarse-resolution clocks (Windows time.time() ~15.6ms) — otherwise
            # a same-tick ts == run_started is dropped by the strict `>` count.
            import time as _t

            _t.sleep(0.05)
            record_thumb(
                db_path="db.sqlite", prompt="qZ", response="x", thumb="up"
            )
            record_thumb(
                db_path="db.sqlite", prompt="qZ", response="y", thumb="down"
            )

        run_nightly_train(cfg, min_pairs=1, train_fn=slow_train)
        last = float(get_state("db.sqlite", "last_train_at"))
        # The 2 thumbs recorded during the train survive the stamp.
        assert count_new_thumbs_since("db.sqlite", last) == 2

    def test_cli_once_subprocess_failure_exit_1(self, tmp_path, monkeypatch):
        # CORRECTNESS MEDIUM: a failed soup-train subprocess -> friendly exit 1.
        monkeypatch.chdir(tmp_path)
        import subprocess

        import soup_cli.utils.local_rl as lr
        from soup_cli.commands.local_rl import app

        _seed_db("db.sqlite", 5)

        def boom(**kw):
            raise subprocess.CalledProcessError(1, ["soup", "train"])

        monkeypatch.setattr(lr, "_default_train_fn", boom)
        res = runner.invoke(
            app,
            [
                "train",
                "--db",
                "db.sqlite",
                "--model",
                "org/model",
                "--once",
                "--min-pairs",
                "1",
            ],
        )
        assert res.exit_code == 1, (res.output, repr(res.exception))

    def test_load_program_module_revalidates(self, tmp_path, monkeypatch):
        # SEC LOW: _load_program_module re-validates the path before exec.
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        (tmp_path / "notpy.txt").write_text("program = 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            pc._load_program_module("notpy.txt")


# ===========================================================================
# TDD-review coverage gaps
# ===========================================================================


class TestCoverageGaps:
    # --- #229 skip-ordering + harvest-dedup -------------------------------
    def _cfg(self, db, method="dpo"):
        from soup_cli.utils.local_rl import LocalRLConfig

        return LocalRLConfig(
            backend="ollama", model="org/model", db_path=db, train_method=method
        )

    def test_new_thumbs_but_insufficient_pairs(self, tmp_path, monkeypatch):
        # last_train_at in the past + fresh thumbs that yield < min_pairs ->
        # status must be insufficient_pairs (NOT no_new_thumbs).
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            init_local_rl_db,
            record_thumb,
            run_nightly_train,
            set_state,
        )

        init_local_rl_db("db.sqlite")
        set_state("db.sqlite", "last_train_at", "0.0")  # far past
        record_thumb(db_path="db.sqlite", prompt="q", response="g", thumb="up")
        record_thumb(db_path="db.sqlite", prompt="q", response="b", thumb="down")
        res = run_nightly_train(
            self._cfg("db.sqlite"),
            min_pairs=5,
            train_fn=lambda **k: pytest.fail("must not train"),
        )
        assert res.status == "skipped_insufficient_pairs"

    def test_harvest_dedup_one_pair_per_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
            run_nightly_train,
        )

        init_local_rl_db("db.sqlite")
        for _ in range(2):
            record_thumb(db_path="db.sqlite", prompt="q", response="g", thumb="up")
            record_thumb(db_path="db.sqlite", prompt="q", response="b", thumb="down")
        assert len(harvest_dpo_pairs("db.sqlite")) == 1
        res = run_nightly_train(
            self._cfg("db.sqlite"),
            min_pairs=2,
            train_fn=lambda **k: pytest.fail("dedup -> 1 pair < 2"),
        )
        assert res.status == "skipped_insufficient_pairs"

    def test_count_new_thumbs_exact_ts_excluded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.local_rl import (
            count_new_thumbs_since,
            init_local_rl_db,
            record_thumb,
        )

        init_local_rl_db("db.sqlite")
        record_thumb(db_path="db.sqlite", prompt="q", response="g", thumb="up")
        with sqlite3.connect(str(tmp_path / "db.sqlite")) as conn:
            ts = conn.execute("SELECT MAX(ts) FROM thumbs").fetchone()[0]
        assert count_new_thumbs_since("db.sqlite", ts) == 0  # strict >

    # --- #225 load_eval_examples + CompileResult -------------------------
    def test_eval_size_cap(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_compile as pc

        monkeypatch.setattr(pc, "_MAX_EVAL_BYTES", 8)
        (tmp_path / "big.json").write_text("[" + "1," * 50 + "1]", encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds"):
            pc.load_eval_examples("big.json")

    def test_eval_json_object_root(self, tmp_path, monkeypatch):
        # A single top-level JSON object is read as one example (not rejected).
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import load_eval_examples

        (tmp_path / "obj.json").write_text('{"q": "1"}', encoding="utf-8")
        assert load_eval_examples("obj.json") == [{"q": "1"}]

    def test_eval_pretty_json_array(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_compile import load_eval_examples

        (tmp_path / "arr.json").write_text(
            "[\n  {\"q\": \"1\"},\n  {\"q\": \"2\"}\n]", encoding="utf-8"
        )
        assert load_eval_examples("arr.json") == [{"q": "1"}, {"q": "2"}]

    def test_compile_result_validation(self):
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(ValueError):
            CompileResult(
                program_text="x", score=float("nan"), iterations=1, converged=True
            )
        with pytest.raises(ValueError):
            CompileResult(
                program_text="x", score=0.0, iterations=-1, converged=True
            )
        with pytest.raises(TypeError):
            CompileResult(
                program_text="x", score=True, iterations=1, converged=True
            )

    # --- #226 distill edge cases -----------------------------------------
    def test_extract_prompt_query_key(self):
        from soup_cli.utils.prompt_distill import extract_prompt

        assert extract_prompt({"query": "z"}) == "z"

    def test_extract_prompt_truncation(self):
        from soup_cli.utils.prompt_distill import _MAX_PROMPT_CHARS, extract_prompt

        out = extract_prompt({"prompt": "x" * (_MAX_PROMPT_CHARS + 50)})
        assert len(out) == _MAX_PROMPT_CHARS

    def test_preference_empty_student_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import (
            build_distill_prompt_plan,
            prepare_distill_dataset,
        )

        _write_traces(str(tmp_path / "t.jsonl"), ["q"])
        plan = build_distill_prompt_plan(
            traces_path="t.jsonl",
            teacher="t",
            student="s",
            strategy="preference",
            output_path="o.jsonl",
        )
        n = prepare_distill_dataset(
            plan,
            teacher_fn=lambda p: {"text": "T"},
            student_fn=lambda p: {"text": ""},
        )
        assert n == 0

    def test_preference_student_exception_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.prompt_distill import (
            build_distill_prompt_plan,
            prepare_distill_dataset,
        )

        _write_traces(str(tmp_path / "t.jsonl"), ["q"])
        plan = build_distill_prompt_plan(
            traces_path="t.jsonl",
            teacher="t",
            student="s",
            strategy="preference",
            output_path="o.jsonl",
        )

        def boom(p):
            raise RuntimeError("down")

        n = prepare_distill_dataset(
            plan, teacher_fn=lambda p: {"text": "T"}, student_fn=boom
        )
        assert n == 0

    # --- #227 compile-tools ----------------------------------------------
    def test_spec_path_extension_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.utils.compile_tools import validate_spec_path

        (tmp_path / "spec.txt").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="extension"):
            validate_spec_path("spec.txt")

    def test_tool_full_shape(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.compile_tools as ct

        monkeypatch.setattr(ct, "_TOOL_OPTIMIZER_OVERRIDE", lambda d, e, o: "D")
        _write_spec(tmp_path)
        plan = ct.build_tool_compile_plan(
            spec_path="spec.json",
            eval_suite_path="eval.jsonl",
            optimizer="textgrad",
            output_path="out.json",
        )
        ct.run_tool_compile(plan)
        out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        tool = next(t for t in out["tools"] if t["tool"] == "listWidgets")
        assert set(tool) == {"tool", "description", "method", "path", "parameters"}
        assert tool["method"] == "get"


# ===========================================================================
# Coverage cushion — reachable internals exercised network/lib-free so the
# 77% gate does not sit right on the edge (the DSPy/TextGrad/GEPA optimiser
# bodies are genuinely uncoverable without the [compile] extra installed).
# ===========================================================================


class TestReachableInternals:
    def test_resolve_metric_callable(self):
        import types

        import soup_cli.utils.prompt_compile as pc

        mod = types.SimpleNamespace(metric=lambda *a, **k: 1.0)
        assert callable(pc._resolve_metric(mod))

    def test_resolve_metric_absent_returns_none(self):
        import types

        import soup_cli.utils.prompt_compile as pc

        assert pc._resolve_metric(types.SimpleNamespace()) is None

    def test_resolve_metric_non_callable_returns_none(self):
        import types

        import soup_cli.utils.prompt_compile as pc

        assert pc._resolve_metric(types.SimpleNamespace(metric=42)) is None

    def test_build_provider_fn_delegates_to_make_judge(self, monkeypatch):
        import soup_cli.utils.data_forge as df
        import soup_cli.utils.prompt_distill as pd

        seen = {}

        def fake_make(provider, *, model, base_url, temperature):
            seen.update(
                provider=provider, model=model, base_url=base_url, temperature=temperature
            )
            return lambda prompt: {"text": f"R:{prompt}"}

        monkeypatch.setattr(df, "make_judge_provider_fn", fake_make)
        fn = pd._build_provider_fn(
            "ollama", "qwen2.5:0.5b", base_url="http://localhost:11434", temperature=0.2
        )
        assert fn("hi") == {"text": "R:hi"}
        assert seen == {
            "provider": "ollama",
            "model": "qwen2.5:0.5b",
            "base_url": "http://localhost:11434",
            "temperature": 0.2,
        }

    def test_prepare_distill_default_teacher_wires_provider(self, tmp_path, monkeypatch):
        """teacher_fn=None routes through _build_provider_fn (default-provider
        wiring) — covers the no-injected-seam branch of prepare_distill_dataset."""
        monkeypatch.chdir(tmp_path)
        import soup_cli.utils.prompt_distill as pd

        _write_traces(str(tmp_path / "traces.jsonl"), ["q1", "q2"])
        calls = {"n": 0}

        def fake_build(provider, model, *, base_url, temperature):  # noqa: ARG001
            def _gen(prompt):
                calls["n"] += 1
                return {"text": f"T:{prompt}"}

            return _gen

        monkeypatch.setattr(pd, "_build_provider_fn", fake_build)
        plan = pd.build_distill_prompt_plan(
            traces_path="traces.jsonl",
            teacher="qwen2.5:0.5b",
            student="smol",
            strategy="sft",
            output_path="out.jsonl",
        )
        n = pd.prepare_distill_dataset(plan, provider="ollama")
        assert n == 2
        assert calls["n"] == 2


# ===========================================================================
# Patch invariants
# ===========================================================================


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(x) for x in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 13)
