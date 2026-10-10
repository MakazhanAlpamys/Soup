"""#1449 — ``--trust-remote-code`` is accepted but dropped on five paths.

``soup data download`` and ``soup eval auto`` never read the parameter at all.
``soup train --find-lr`` hardcodes ``trust_remote_code=False`` in the live-sweep
loaders. ``soup draft distill`` and ``soup shrink`` (heal step) resolve the flag
in the parent but never pass it to the child ``soup train`` subprocess. In every
case, opting in changes nothing: the load that needs the flag never receives it.

Each test below mirrors the issue's own stub-based repro: the real load call is
replaced with a stub that records the kwargs/argv it received, so the assertion
is on what actually reaches the consumer, not on whether a real model loads.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi

runner = CliRunner()


def _flat(text: str) -> str:
    return " ".join(strip_ansi(text).split())


# ─── 1. soup data download -> _hf_download_dataset -> load_dataset ───


class TestDataDownloadTrust:
    def test_trust_remote_code_reaches_load_dataset(self, tmp_path, monkeypatch):
        import datasets

        import soup_cli.commands.data as data_cmd
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        # datasets<4 still honors trust_remote_code, so the version gate
        # (below) must let it through and the flag must reach load_dataset.
        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        seen = []
        monkeypatch.setattr(
            datasets, "load_dataset",
            lambda *a, **k: seen.append(k) or iter([{"text": "x"}]),
        )
        result = runner.invoke(
            app,
            ["data", "download", "org/ds", "-n", "1", "-o", "out.jsonl",
             "--trust-remote-code"],
        )
        assert result.exit_code == 0, result.output
        assert seen, "load_dataset was never called"
        assert seen[0].get("trust_remote_code") is True, seen[0]
        assert data_cmd._datasets_major_version() == 3

    def test_without_flag_load_dataset_gets_false(self, tmp_path, monkeypatch):
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        seen = []
        monkeypatch.setattr(
            datasets, "load_dataset",
            lambda *a, **k: seen.append(k) or iter([{"text": "x"}]),
        )
        result = runner.invoke(
            app, ["data", "download", "org/ds", "-n", "1", "-o", "out2.jsonl"],
        )
        assert result.exit_code == 0, result.output
        assert seen[0].get("trust_remote_code") is False, seen[0]

    def test_trust_remote_code_is_refused_on_datasets_v4_plus(
        self, tmp_path, monkeypatch,
    ):
        """On datasets>=4, `load_dataset` pops `trust_remote_code` and only
        logs an error (it no longer supports it at all) — so forwarding the
        flag there would silently do nothing, same bug as before this fix,
        one layer down. The CLI must refuse instead of forwarding it."""
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(datasets, "__version__", "5.0.1")
        seen = []
        monkeypatch.setattr(
            datasets, "load_dataset",
            lambda *a, **k: seen.append(k) or iter([{"text": "x"}]),
        )
        result = runner.invoke(
            app,
            ["data", "download", "org/ds", "-n", "1", "-o", "out3.jsonl",
             "--trust-remote-code"],
        )
        assert result.exit_code != 0, result.output
        assert "refused" in result.output.lower(), result.output
        assert not seen, "load_dataset must not be called when the flag is refused"

    def test_without_flag_datasets_v4_plus_is_unaffected(self, tmp_path, monkeypatch):
        """The version gate only fires when --trust-remote-code is actually
        requested; a plain download on datasets>=4 must still work."""
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(datasets, "__version__", "5.0.1")
        seen = []
        monkeypatch.setattr(
            datasets, "load_dataset",
            lambda *a, **k: seen.append(k) or iter([{"text": "x"}]),
        )
        result = runner.invoke(
            app, ["data", "download", "org/ds", "-n", "1", "-o", "out4.jsonl"],
        )
        assert result.exit_code == 0, result.output
        assert seen[0].get("trust_remote_code") is False, seen[0]


# ─── 2. soup eval auto -> benchmark()/custom() -> lm-eval model_args / pipeline ───


class TestEvalAutoTrust:
    def _setup(self, tmp_path, monkeypatch):
        import soup_cli.commands.eval as eval_cmd
        import soup_cli.eval.custom as custom

        monkeypatch.chdir(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        (out / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/custom-code-model"})
        )
        (tmp_path / "tasks.jsonl").write_text(
            json.dumps({"prompt": "2+2?", "expected": "4", "scoring": "exact"}) + "\n"
        )
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"instruction": "i", "output": "o"}) + "\n"
        )
        (tmp_path / "soup.yaml").write_text(
            "base: org/custom-code-model\ntask: sft\ndata:\n  train: train.jsonl\n"
            "output: out\neval:\n  benchmarks: [mmlu]\n  custom_tasks: tasks.jsonl\n"
        )
        monkeypatch.setitem(sys.modules, "lm_eval", types.ModuleType("lm_eval"))

        seen_model_args = []
        seen_generator_calls = []
        monkeypatch.setattr(
            eval_cmd, "_run_lm_eval",
            lambda model_arg, *a, **k: seen_model_args.append(model_arg) or {"results": {}},
        )
        monkeypatch.setattr(eval_cmd, "_save_benchmark_results", lambda *a, **k: None)
        monkeypatch.setattr(eval_cmd, "_save_custom_results", lambda *a, **k: None)
        monkeypatch.setattr(
            custom, "_create_default_generator",
            lambda *a, **k: seen_generator_calls.append((a, k)) or (lambda prompt: "4"),
        )
        return seen_model_args, seen_generator_calls

    def test_trust_remote_code_reaches_lm_eval_model_args_and_generator(
        self, tmp_path, monkeypatch,
    ):
        from soup_cli.cli import app

        seen_model_args, seen_generator_calls = self._setup(tmp_path, monkeypatch)
        result = runner.invoke(
            app, ["eval", "auto", "-c", "soup.yaml", "--trust-remote-code"],
        )
        assert result.exit_code == 0, result.output
        assert seen_model_args, "benchmark() was never reached"
        assert "trust_remote_code=True" in seen_model_args[0], seen_model_args[0]
        assert seen_generator_calls, "custom() was never reached"
        _, gen_kwargs = seen_generator_calls[0]
        assert gen_kwargs.get("trust_remote_code") is True, seen_generator_calls[0]

    def test_lm_eval_model_args_still_reject_injection_from_adapter_metadata(
        self, tmp_path, monkeypatch,
    ):
        """The `,`/`=` guard on `base_model_name_or_path` must survive: the only
        source of `trust_remote_code=True` is the resolved CLI flag, never
        adapter_config.json (#1449 acceptance criteria)."""
        from soup_cli.cli import app

        out = tmp_path / "out"
        out.mkdir()
        (out / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/model,trust_remote_code=True"})
        )
        (tmp_path / "soup.yaml").write_text(
            "base: org/model\ntask: sft\ndata:\n  train: train.jsonl\n"
            "output: out\neval:\n  benchmarks: [mmlu]\n"
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["eval", "auto", "-c", "soup.yaml", "--trust-remote-code"],
        )
        # benchmark() must refuse the smuggled base model, not silently swallow it.
        assert "Refusing to evaluate" in result.output, result.output


class TestCreateDefaultGeneratorTrust:
    """Unit-level coverage of `_create_default_generator` itself (`eval.auto`'s
    fallback generator, and every other caller of it), stubbing
    `transformers.pipeline` directly rather than going through the CLI."""

    def test_trust_remote_code_reaches_pipeline_call(self, monkeypatch):
        import transformers

        # Reading `transformers.pipeline` for the first time in a process can
        # replace `sys.modules["transformers"]` with a new module object
        # (this transformers version's lazy-loading backend check does a
        # fresh re-import under the hood) — so touch it once here and
        # re-fetch the *live* module before patching. Patching the
        # `transformers` reference obtained above would silently patch a
        # now-stale object and the real `pipeline` would still run.
        _ = transformers.pipeline
        live_transformers = sys.modules["transformers"]

        from soup_cli.eval.custom import _create_default_generator

        seen = []

        def fake_pipeline(*args, **kwargs):
            seen.append(kwargs.get("trust_remote_code"))
            return lambda *a, **k: [{"generated_text": "x"}]

        monkeypatch.setattr(live_transformers, "pipeline", fake_pipeline)

        _create_default_generator("org/model", trust_remote_code=True)
        assert seen == [True], seen

        _create_default_generator("org/model", trust_remote_code=False)
        assert seen == [True, False], seen


# ─── 3. soup train --find-lr -> _live_lr_sweep_from_config -> from_pretrained ───


class TestFindLrTrust:
    """Drives the real `soup train --find-lr` CLI path (like the issue's own
    repro), stubbing only `transformers.Auto*.from_pretrained` so the sweep
    reaches the loaders without needing a real model. `torch` is stubbed too
    only when it is not actually installed, with just enough surface
    (`cuda.is_available`) for `_live_lr_sweep_from_config` to reach the loads
    it makes before any real tensor math.
    """

    def _invoke(self, tmp_path, monkeypatch, *, requested_flag: bool):
        import importlib.util
        from types import SimpleNamespace

        import transformers

        # Force the lazy Auto* attributes to resolve now, while `torch`'s
        # real availability is still whatever this interpreter actually has.
        # Do this BEFORE any fake `torch` goes into sys.modules: once a fake
        # is there, transformers' own availability check can see it, decide
        # torch "is" available, and cache a real (but broken, half-fake)
        # class under these names — poisoning every other test in this
        # process that imports transformers afterward.
        _ = transformers.AutoTokenizer
        _ = transformers.AutoModelForCausalLM

        if importlib.util.find_spec("torch") is None:
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            monkeypatch.setitem(sys.modules, "torch", fake_torch)

        seen = []

        def fake_tokenizer_from_pretrained(*args, **kwargs):
            seen.append(kwargs.get("trust_remote_code"))
            return SimpleNamespace(pad_token="x", eos_token="x")

        def fake_model_from_pretrained(*args, **kwargs):
            seen.append(kwargs.get("trust_remote_code"))
            raise ValueError("custom code: set trust_remote_code=True")

        # `transformers.Auto*` classes gate attribute ACCESS to
        # `from_pretrained` behind a metaclass backend check (raises
        # ImportError without torch), so `monkeypatch.setattr(cls,
        # "from_pretrained", ...)` cannot even read the old value to patch
        # it. Replace the module-level class names instead — that's a plain
        # module attribute, no metaclass involved — so
        # `_live_lr_sweep_from_config`'s `from transformers import
        # AutoModelForCausalLM, AutoTokenizer` picks up the fakes. Separate
        # fakes for each class so both from_pretrained calls are actually
        # reached (the tokenizer must succeed for the model load to happen
        # at all).
        class _FakeAutoTokenizer:
            from_pretrained = staticmethod(fake_tokenizer_from_pretrained)

        class _FakeAutoModel:
            from_pretrained = staticmethod(fake_model_from_pretrained)

        monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)
        monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAutoModel)

        monkeypatch.chdir(tmp_path)
        (tmp_path / "rows.jsonl").write_text(
            "".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(8))
        )
        (tmp_path / "lr.yaml").write_text(
            "base: org/custom-code-model\ntask: sft\n"
            "data:\n  train: rows.jsonl\n  format: plaintext\noutput: out2\n"
        )

        from soup_cli.cli import app

        argv = ["train", "-c", "lr.yaml", "--find-lr", "--find-lr-steps", "8"]
        if requested_flag:
            argv.append("--trust-remote-code")
        result = runner.invoke(app, argv)
        return seen, result

    def test_trust_remote_code_reaches_both_from_pretrained_calls(
        self, tmp_path, monkeypatch,
    ):
        seen, result = self._invoke(tmp_path, monkeypatch, requested_flag=True)
        assert len(seen) == 2, (
            "expected both AutoTokenizer and AutoModelForCausalLM."
            f"from_pretrained to be reached, got {seen}: {result.output}"
        )
        assert all(v is True for v in seen), (seen, result.output)

    def test_without_flag_from_pretrained_gets_false(self, tmp_path, monkeypatch):
        seen, result = self._invoke(tmp_path, monkeypatch, requested_flag=False)
        assert len(seen) == 2, (
            "expected both AutoTokenizer and AutoModelForCausalLM."
            f"from_pretrained to be reached, got {seen}: {result.output}"
        )
        assert all(v is False for v in seen), (seen, result.output)


# ─── 4./5. soup draft distill and soup shrink --heal: child `soup train` argv ───


class TestChildTrainArgvTrust:
    def test_draft_distill_appends_trust_remote_code_when_trc(self, tmp_path, monkeypatch):
        import soup_cli.commands.draft as draft

        monkeypatch.chdir(tmp_path)
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            return subprocess.CompletedProcess(argv, 1, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        data_path = tmp_path / "d.jsonl"
        data_path.write_text(
            "".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(8)
            )
        )
        try:
            draft._run_distill(
                draft_base="org/draft", target="org/target", data=str(data_path),
                out_dir="draft_out", steps=8, data_rows=8, trc=True,
            )
        except Exception:
            pass  # the stubbed child "fails" (rc=1); only its argv matters here
        assert argvs, "subprocess.run (child `soup train`) was never called"
        assert "--trust-remote-code" in argvs[-1], argvs[-1]

    def test_draft_distill_omits_flag_when_not_trc(self, tmp_path, monkeypatch):
        import soup_cli.commands.draft as draft

        monkeypatch.chdir(tmp_path)
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            return subprocess.CompletedProcess(argv, 1, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        data_path = tmp_path / "d.jsonl"
        data_path.write_text(
            "".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(8)
            )
        )
        try:
            draft._run_distill(
                draft_base="org/draft", target="org/target", data=str(data_path),
                out_dir="draft_out2", steps=8, data_rows=8, trc=False,
            )
        except Exception:
            pass
        assert argvs
        assert "--trust-remote-code" not in argvs[-1], argvs[-1]

    def test_shrink_heal_appends_trust_remote_code_when_trc(self, tmp_path, monkeypatch):
        import soup_cli.commands.shrink as shrink

        monkeypatch.chdir(tmp_path)
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            return subprocess.CompletedProcess(argv, 1, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        data_path = tmp_path / "d.jsonl"
        data_path.write_text(
            "".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(8)
            )
        )
        pruned_dir = tmp_path / "pruned"
        pruned_dir.mkdir()
        try:
            shrink._run_heal(
                pruned_dir=str(pruned_dir), teacher="org/target", heal_data=str(data_path),
                steps=8, out_dir="heal_out", heal_rows=8, trc=True,
            )
        except Exception:
            pass
        assert argvs, "subprocess.run (child `soup train`) was never called"
        assert "--trust-remote-code" in argvs[-1], argvs[-1]

    def test_shrink_heal_omits_flag_when_not_trc(self, tmp_path, monkeypatch):
        import soup_cli.commands.shrink as shrink

        monkeypatch.chdir(tmp_path)
        argvs = []

        def fake_run(argv, **kwargs):
            argvs.append(argv)
            return subprocess.CompletedProcess(argv, 1, b"", b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        data_path = tmp_path / "d.jsonl"
        data_path.write_text(
            "".join(
                json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n"
                for i in range(8)
            )
        )
        pruned_dir = tmp_path / "pruned"
        pruned_dir.mkdir()
        try:
            shrink._run_heal(
                pruned_dir=str(pruned_dir), teacher="org/target", heal_data=str(data_path),
                steps=8, out_dir="heal_out2", heal_rows=8, trc=False,
            )
        except Exception:
            pass
        assert argvs, "subprocess.run (child `soup train`) was never called"
        assert "--trust-remote-code" not in argvs[-1], argvs[-1]


# ─── Default-deny: without the flag, no site may opt in to remote code ───


class TestDefaultsStayDeny:
    def test_eval_auto_without_flag_trusts_nothing(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        seen_model_args, seen_generator_calls = TestEvalAutoTrust()._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, ["eval", "auto", "-c", "soup.yaml"])
        assert result.exit_code == 0, result.output
        assert seen_model_args and "trust_remote_code" not in seen_model_args[0], seen_model_args
        assert seen_generator_calls[0][1].get("trust_remote_code") is False, seen_generator_calls

    @pytest.mark.parametrize(("flag", "expected"), [([], False), (["--trust-remote-code"], True)])
    def test_eval_benchmark_cli_model_args(self, tmp_path, monkeypatch, flag, expected):
        from soup_cli.cli import app

        seen_model_args, _ = TestEvalAutoTrust()._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, ["eval", "benchmark", "-m", "out", "-b", "mmlu", *flag])
        assert result.exit_code == 0, result.output
        assert ("trust_remote_code=True" in seen_model_args[0]) is expected, seen_model_args

    @pytest.mark.parametrize(("flag", "expected"), [([], False), (["--trust-remote-code"], True)])
    def test_eval_custom_cli_generator_kwargs(self, tmp_path, monkeypatch, flag, expected):
        from soup_cli.cli import app

        _, seen_generator_calls = TestEvalAutoTrust()._setup(tmp_path, monkeypatch)
        result = runner.invoke(app, ["eval", "custom", "-t", "tasks.jsonl", "-m", "out", *flag])
        assert result.exit_code == 0, result.output
        assert seen_generator_calls[0][1].get("trust_remote_code") is expected, seen_generator_calls

    def test_training_auto_eval_callback_never_opts_in(self, monkeypatch):
        import soup_cli.commands.eval as ce
        import soup_cli.commands.train as train_cmd

        calls: dict = {}
        monkeypatch.setattr(ce, "benchmark", lambda **kw: calls.__setitem__("benchmark", kw))
        monkeypatch.setattr(ce, "custom", lambda **kw: calls.__setitem__("custom", kw))
        monkeypatch.setattr(train_cmd, "_should_run_diagnose_gate_on_rank", lambda: True)

        eval_config = type(
            "EvalCfg", (),
            {"auto_eval": True, "benchmarks": ["mmlu"], "custom_tasks": "tasks.jsonl"},
        )()
        train_cmd._run_auto_eval_after_training(eval_config, "out", "run-1")
        assert calls["benchmark"]["trust_remote_code"] is False, calls
        assert calls["custom"]["trust_remote_code"] is False, calls

    @pytest.mark.parametrize(("flag", "panel"), [([], False), (["--trust-remote-code"], True)])
    def test_data_download_warning_panel_only_with_flag(self, tmp_path, monkeypatch, flag, panel):
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: iter([{"text": "x"}]))
        result = runner.invoke(
            app, ["data", "download", "org/ds", "-n", "1", "-o", "o.jsonl", *flag],
        )
        assert result.exit_code == 0, result.output
        assert ("Remote Code Warning" in _flat(result.output)) is panel, _flat(result.output)

    def test_default_generator_defaults_to_false(self, monkeypatch):
        """Callers that omit the kwarg must get an explicit False, not None."""
        import transformers

        _ = transformers.pipeline
        live_transformers = sys.modules["transformers"]

        from soup_cli.eval.custom import _create_default_generator

        seen = []

        def fake_pipeline(*args, **kwargs):
            seen.append(kwargs.get("trust_remote_code", "missing"))
            return lambda *a, **k: [{"generated_text": "x"}]

        monkeypatch.setattr(live_transformers, "pipeline", fake_pipeline)
        _create_default_generator("org/model")
        assert seen == [False], seen

    def test_flag_without_datasets_installed_gets_the_install_hint(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(sys.modules, "datasets", None)  # `import datasets` raises ImportError
        result = runner.invoke(
            app, ["data", "download", "org/ds", "-n", "1", "-o", "o.jsonl", "--trust-remote-code"],
        )
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "datasets library not available" in _flat(result.output)
