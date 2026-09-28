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

from typer.testing import CliRunner

runner = CliRunner()


# ─── 1. soup data download -> _hf_download_dataset -> load_dataset ───


class TestDataDownloadTrust:
    def test_trust_remote_code_reaches_load_dataset(self, tmp_path, monkeypatch):
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
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

    def test_without_flag_load_dataset_gets_false(self, tmp_path, monkeypatch):
        import datasets

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
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
        assert "Refusing to evaluate" in result.output or result.exit_code != 0


# ─── 3. soup train --find-lr -> _live_lr_sweep_from_config -> from_pretrained ───


class TestFindLrTrust:
    """Drives the real `soup train --find-lr` CLI path (like the issue's own
    repro), stubbing only `transformers.Auto*.from_pretrained` so the sweep
    reaches the loaders without needing torch or a real model. `torch` itself
    is stubbed too (never installed on this box) with just enough surface
    (`cuda.is_available`) for `_live_lr_sweep_from_config` to reach the loads
    it makes before any real tensor math.
    """

    def _invoke(self, tmp_path, monkeypatch, *, requested_flag: bool):
        import transformers

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        seen = []

        def fake_from_pretrained(*args, **kwargs):
            seen.append(kwargs.get("trust_remote_code"))
            raise ValueError("custom code: set trust_remote_code=True")

        # `transformers.Auto*` classes gate attribute ACCESS behind a
        # metaclass backend check (raises ImportError without torch), so
        # `monkeypatch.setattr(cls, "from_pretrained", ...)` cannot even read
        # the old value to patch it. Replace the module-level class names
        # instead — that's a plain module attribute, no metaclass involved —
        # so `_live_lr_sweep_from_config`'s `from transformers import
        # AutoModelForCausalLM, AutoTokenizer` picks up the fakes.
        class _FakeAuto:
            from_pretrained = staticmethod(fake_from_pretrained)

        monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAuto)
        monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAuto)

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
        assert seen, "AutoTokenizer/AutoModelForCausalLM.from_pretrained was never called"
        assert all(v is True for v in seen), (seen, result.output)

    def test_without_flag_from_pretrained_gets_false(self, tmp_path, monkeypatch):
        seen, result = self._invoke(tmp_path, monkeypatch, requested_flag=False)
        assert seen, "AutoTokenizer/AutoModelForCausalLM.from_pretrained was never called"
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
