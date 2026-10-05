"""#1445 — three commands validated their destination only AFTER the
expensive step, then threw the finished work away.

- ``soup compile -o ../out.py`` ran the whole optimizer (LLM calls) and only
  then refused the path, exiting 1 with a generic error line instead of 2.
- Quantized GGUF export into a directory that does not exist yet crashed in
  ``tempfile.mkdtemp`` AFTER the adapter merge, and the cleanup deleted the
  merged model.
- ``task: unlearn`` loaded the policy and a frozen reference copy, ran every
  NPO / SimNPO / RMU step, and only then refused an ``output`` outside cwd
  or a symlinked one.

Each test keeps a passing in-cwd control, as the issue's acceptance
criteria ask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# soup compile
# ---------------------------------------------------------------------------


def _compile_project(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "prog.py").write_text('program = "p"\n', encoding="utf-8")
    (proj / "suite.jsonl").write_text(
        json.dumps({"q": "a", "answer": "b"}) + "\n", encoding="utf-8"
    )
    monkeypatch.chdir(proj)
    return proj


def _stub_optimizer(monkeypatch, ran):
    from soup_cli.utils import prompt_compile

    def fake_optimizer(plan):
        ran.append(plan.output_path)
        return prompt_compile.CompileResult(
            program_text="COMPILED", score=0.9, iterations=3, converged=True
        )

    monkeypatch.setattr(prompt_compile, "_OPTIMIZER_RUN_OVERRIDE", fake_optimizer)


class TestCompileRefusesBeforeRunningTheOptimizer:
    @pytest.mark.parametrize("bad", ["../out.py", "link.py"])
    @pytest.mark.requires_symlink
    def test_the_optimizer_is_never_called_and_exit_is_2(self, tmp_path, monkeypatch, bad):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        proj = _compile_project(tmp_path, monkeypatch)
        if bad == "link.py":
            target = proj / "real.py"
            target.write_text("x = 1\n", encoding="utf-8")
            os.symlink(target, proj / "link.py")

        ran: list = []
        _stub_optimizer(monkeypatch, ran)

        result = CliRunner().invoke(
            app,
            ["compile", "prog.py", "--eval", "suite.jsonl", "-o", bad],
        )

        assert ran == [], "the optimizer ran before the destination was refused"
        assert result.exit_code == 2
        assert (
            "must stay under cwd" in result.output
            or (result.exception is not None and "must stay under cwd" in str(result.exception))
            or "symlink" in result.output
        )

    def test_control_an_in_cwd_destination_still_compiles(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        _compile_project(tmp_path, monkeypatch)
        ran: list = []
        _stub_optimizer(monkeypatch, ran)

        result = CliRunner().invoke(
            app,
            ["compile", "prog.py", "--eval", "suite.jsonl", "-o", "ok_out.py"],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (Path.cwd() / "ok_out.py").read_text(encoding="utf-8") == "COMPILED"

    def test_a_write_time_refusal_exits_2_not_1(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import compile_cmd as compile_mod

        _compile_project(tmp_path, monkeypatch)
        _stub_optimizer(monkeypatch, [])

        def refuse(*args, **kwargs):
            raise ValueError("output_path must not be a symlink (TOCTOU defence)")

        monkeypatch.setattr(compile_mod, "atomic_write_text", refuse)
        result = CliRunner().invoke(
            app, ["compile", "prog.py", "--eval", "suite.jsonl", "-o", "ok_out.py"]
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "TOCTOU" in result.output


# ---------------------------------------------------------------------------
# GGUF export into a missing parent directory
# ---------------------------------------------------------------------------


def _adapter_dir(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "gpt2"}), encoding="utf-8"
    )
    return adapter


def _stub_export_backend(monkeypatch, tmp_path, merges):
    from soup_cli.commands import export as export_mod

    llama = tmp_path / "llama"
    llama.mkdir(exist_ok=True)
    (llama / "convert_hf_to_gguf.py").write_text("# stub\n", encoding="utf-8")

    def fake_merge(adapter_path, base_model, output_dir, trust_remote_code=False):
        merges.append(output_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"\0" * 16)

    monkeypatch.setattr(export_mod, "_merge_adapter", fake_merge)
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda path=None: llama)
    monkeypatch.setattr(export_mod, "_install_convert_deps", lambda: None)

    def fake_convert(script, model_dir, output_path, outtype):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GGUF")

    monkeypatch.setattr(export_mod, "_run_convert", fake_convert)
    monkeypatch.setattr(
        export_mod,
        "_run_quantize",
        lambda ldir, f16, out, quant: out.write_bytes(b"GGUF"),
    )


class TestQuantisedExportIntoAMissingDirectory:
    def test_the_parent_is_created_and_the_merge_is_kept(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        adapter = _adapter_dir(tmp_path)
        merges: list = []
        _stub_export_backend(monkeypatch, tmp_path, merges)

        out = tmp_path / "new_dir" / "model.gguf"
        assert not out.parent.exists()  # the parent really is missing
        result = CliRunner().invoke(
            app,
            ["export", "--model", str(adapter), "--quant", "q4_k_m", "--output", str(out)],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert merges, "the merge ran"
        assert out.read_bytes() == b"GGUF"

    def test_control_an_existing_directory_still_exports(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        adapter = _adapter_dir(tmp_path)
        merges: list = []
        _stub_export_backend(monkeypatch, tmp_path, merges)

        out = tmp_path / "model.gguf"
        result = CliRunner().invoke(
            app,
            ["export", "--model", str(adapter), "--quant", "q4_k_m", "--output", str(out)],
        )

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()


# ---------------------------------------------------------------------------
# task: unlearn
# ---------------------------------------------------------------------------


def _unlearn_wrapper(output):
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    cfg = SimpleNamespace(
        output=output,
        training=SimpleNamespace(unlearn_method="npo"),
    )
    return UnlearnTrainerWrapper(cfg)


class TestUnlearnRefusesTheOutputBeforeLoadingAnything:
    @pytest.mark.parametrize("bad", ["../unlearn-out"])
    def test_setup_refuses_an_output_outside_cwd(self, tmp_path, monkeypatch, bad):
        monkeypatch.chdir(tmp_path)
        wrapper = _unlearn_wrapper(bad)

        # The heavy imports must never even run: torch / peft / from_pretrained
        # are all behind the validation, so a missing-import sentinel proves
        # the ordering.
        import builtins

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name in ("torch", "peft", "transformers"):
                raise AssertionError(f"setup() imported {name!r} before refusing the output dir")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guard):
            with pytest.raises(ValueError, match="output dir must stay under cwd"):
                wrapper.setup()

    @pytest.mark.requires_symlink
    def test_setup_refuses_a_symlinked_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real-out"
        real.mkdir()
        os.symlink(real, tmp_path / "link-out")

        wrapper = _unlearn_wrapper("link-out")
        with pytest.raises(ValueError, match="symlink"):
            wrapper.setup()

    def test_control_a_plain_in_cwd_output_passes_the_early_check(self, tmp_path, monkeypatch):
        from soup_cli.trainer.unlearn import _validated_output_dir

        monkeypatch.chdir(tmp_path)
        (tmp_path / "plain-out").mkdir()
        assert _validated_output_dir("plain-out") == "plain-out"
