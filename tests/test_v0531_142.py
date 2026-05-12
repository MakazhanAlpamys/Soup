"""v0.53.1 #142 — merge --save-format + export --format torchao live wiring.

Tests cover:
* ``merge_4bit`` validators + happy path with mocked transformers / BNB
* ``export_torchao`` validators + happy path with mocked torchao
* CLI plumbing for ``soup merge --save-format`` (4bit / 4bit_forced / fp16)
* CLI plumbing for ``soup export --format torchao --quant-config <yaml>``
* Path containment + symlink TOCTOU rejection at dispatch time
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

runner = CliRunner()


# --- merge_4bit live wiring -------------------------------------------------


class TestMerge4bitWiring:
    def test_imports(self):
        from soup_cli.utils.save_formats import merge_4bit

        assert callable(merge_4bit)

    def test_no_longer_raises_not_implemented(self, tmp_path):
        from soup_cli.utils.save_formats import merge_4bit

        # The live wiring lands in v0.53.1; calling without args used to
        # raise NotImplementedError. Now it accepts named args and runs
        # the path-validation path before raising on missing model dir.
        with pytest.raises((TypeError, ValueError, FileNotFoundError)):
            # Missing source dir → FileNotFoundError or ValueError
            merge_4bit(
                merged_dir=str(tmp_path / "missing"),
                output_dir=str(tmp_path / "out"),
                forced=False,
            )

    def test_rejects_outside_cwd_source(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import merge_4bit

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="under cwd"):
            merge_4bit(
                merged_dir=str(outside),
                output_dir=str(tmp_path / "out"),
                forced=False,
            )

    def test_rejects_outside_cwd_output(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import merge_4bit

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="under cwd"):
            merge_4bit(
                merged_dir=str(src),
                output_dir=str(tmp_path.parent / "out"),
                forced=False,
            )

    def test_rejects_symlink_output(self, tmp_path, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("symlink rejection POSIX-only")
        from soup_cli.utils.save_formats import merge_4bit

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        evil_target = tmp_path / "evil_target"
        evil_target.mkdir()
        link = tmp_path / "out_link"
        link.symlink_to(evil_target)
        with pytest.raises(ValueError, match="symlink"):
            merge_4bit(
                merged_dir=str(src),
                output_dir=str(link),
                forced=False,
            )

    def test_non_bool_forced_rejected(self, tmp_path, monkeypatch):
        """forced must be a real bool (project bool-before-int policy).

        Renamed from ``test_bool_forced_rejected`` to match the actual
        behaviour: the guard is ``if not isinstance(forced, bool)``, so
        any non-bool value (including ``"yes"`` or ``1``) is rejected,
        while ``True`` / ``False`` pass through.
        """
        from soup_cli.utils.save_formats import merge_4bit

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError):
            merge_4bit(
                merged_dir=str(src),
                output_dir=str(tmp_path / "out"),
                forced="yes",  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError):
            merge_4bit(
                merged_dir=str(src),
                output_dir=str(tmp_path / "out"),
                forced=1,  # type: ignore[arg-type]
            )

    def test_happy_path_with_mocks(self, tmp_path, monkeypatch):
        from soup_cli.utils import save_formats

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "merged"
        src.mkdir()
        (src / "config.json").write_text(
            '{"model_type": "llama"}', encoding="utf-8"
        )

        # Patch the from_pretrained class methods on the real transformers
        # module — this avoids the `patch.dict(sys.modules, ...)` approach
        # which leaks state into later test files that import real torch.
        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        import transformers  # noqa: F401 — needed before patching attrs
        out_dir = tmp_path / "out_4bit"
        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ), patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "transformers.BitsAndBytesConfig",
            MagicMock(return_value=MagicMock()),
        ):
            save_formats.merge_4bit(
                merged_dir=str(src),
                output_dir=str(out_dir),
                forced=False,
            )

        fake_model.save_pretrained.assert_called_once_with(str(out_dir))
        fake_tokenizer.save_pretrained.assert_called_once_with(str(out_dir))


# --- export_torchao live wiring ---------------------------------------------


class TestExportTorchAOWiring:
    def test_imports(self):
        from soup_cli.utils.save_formats import export_torchao

        assert callable(export_torchao)

    def test_invalid_scheme_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import export_torchao

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="not supported"):
            export_torchao(
                model_dir=str(src),
                output_dir=str(tmp_path / "out"),
                scheme="EvilScheme",
            )

    def test_rejects_outside_cwd_model(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import export_torchao

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_torchao"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="under cwd"):
            export_torchao(
                model_dir=str(outside),
                output_dir=str(tmp_path / "out"),
                scheme="Int4WeightOnly",
            )

    def test_rejects_outside_cwd_output(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import export_torchao

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="under cwd"):
            export_torchao(
                model_dir=str(src),
                output_dir=str(tmp_path.parent / "out"),
                scheme="Int4WeightOnly",
            )

    def test_happy_path_with_mocks(self, tmp_path, monkeypatch):
        from soup_cli.utils import save_formats

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "model"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        # Build a minimal in-process torchao stand-in. We do swap sys.modules
        # for torchao + torchao.quantization (real torchao isn't installed in
        # CI), but we patch transformers attrs directly to avoid the torch
        # reload issue that breaks downstream tests.
        fake_torchao = MagicMock()
        fake_torchao.quantization.Int4WeightOnlyConfig.return_value = MagicMock()
        fake_torchao.quantize_ = MagicMock()

        original_torchao = sys.modules.get("torchao")
        original_torchao_q = sys.modules.get("torchao.quantization")
        sys.modules["torchao"] = fake_torchao
        sys.modules["torchao.quantization"] = fake_torchao.quantization
        try:
            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ):
                out_dir = tmp_path / "out_torchao"
                save_formats.export_torchao(
                    model_dir=str(src),
                    output_dir=str(out_dir),
                    scheme="Int4WeightOnly",
                )
        finally:
            if original_torchao is None:
                sys.modules.pop("torchao", None)
            else:
                sys.modules["torchao"] = original_torchao
            if original_torchao_q is None:
                sys.modules.pop("torchao.quantization", None)
            else:
                sys.modules["torchao.quantization"] = original_torchao_q

        fake_model.save_pretrained.assert_called_once()


# --- CLI plumbing for `soup merge --save-format` ----------------------------


class TestMergeSaveFormatCLI:
    def test_save_format_help_lists_flag(self):
        from soup_cli.commands.merge import merge

        app = typer.Typer()
        app.command()(merge)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--save-format" in result.output

    def test_invalid_save_format(self, tmp_path, monkeypatch):
        from soup_cli.commands.merge import merge

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "some/base"}', encoding="utf-8"
        )

        app = typer.Typer()
        app.command()(merge)
        result = runner.invoke(
            app,
            [
                "--adapter", str(adapter),
                "--save-format", "weird",
                "--output", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "save_format" in result.output or "save-format" in result.output


# --- CLI plumbing for `soup export --format torchao` ------------------------


class TestExportTorchaoCLI:
    def test_torchao_in_supported_formats(self):
        from soup_cli.commands import export as export_mod

        assert "torchao" in export_mod.SUPPORTED_FORMATS

    def test_torchao_help_lists_quant_config(self):
        from soup_cli.commands.export import export

        app = typer.Typer()
        app.command()(export)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--quant-config" in result.output

    def test_torchao_requires_quant_config(self, tmp_path, monkeypatch):
        from soup_cli.commands.export import export

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")

        app = typer.Typer()
        app.command()(export)
        result = runner.invoke(
            app,
            [
                "--model", str(model),
                "--format", "torchao",
                "--output", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "--quant-config" in result.output or "quant_config" in result.output

    def test_torchao_quant_config_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.commands.export import export

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")

        outside_yaml = tmp_path.parent / "evil_quant.yaml"
        outside_yaml.write_text("scheme: Int4WeightOnly\n", encoding="utf-8")

        app = typer.Typer()
        app.command()(export)
        result = runner.invoke(
            app,
            [
                "--model", str(model),
                "--format", "torchao",
                "--quant-config", str(outside_yaml),
                "--output", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "under cwd" in result.output or "cwd" in result.output.lower()


# --- Path-containment helpers exposed by save_formats -----------------------


class TestValidateQuantConfigPath:
    def test_existing_shape_validators_still_work(self):
        from soup_cli.utils.save_formats import validate_quant_config_path

        assert validate_quant_config_path("config.yaml") == "config.yaml"

    def test_null_byte_rejected(self):
        from soup_cli.utils.save_formats import validate_quant_config_path

        with pytest.raises(ValueError):
            validate_quant_config_path("ev\x00il.yaml")

    def test_load_quant_config_yaml_happy(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        yaml_path = tmp_path / "q.yaml"
        yaml_path.write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
        data = load_quant_config(str(yaml_path))
        assert data == {"scheme": "Int4WeightOnly"}

    def test_load_quant_config_yaml_outside_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.yaml"
        outside.write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
        with pytest.raises(ValueError, match="under cwd"):
            load_quant_config(str(outside))

    def test_load_quant_config_yaml_symlink(self, tmp_path, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("symlink rejection POSIX-only")
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.yaml"
        real.write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
        link = tmp_path / "link.yaml"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            load_quant_config(str(link))

    def test_load_quant_config_yaml_size_cap(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        big = tmp_path / "big.yaml"
        # 300 KB blob — exceeds 256 KB cap
        big.write_text("scheme: Int4WeightOnly\n" + ("x" * (300 * 1024)), encoding="utf-8")
        with pytest.raises(ValueError, match="too large"):
            load_quant_config(str(big))

    def test_load_quant_config_yaml_invalid_extension(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "config.txt"
        bad.write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
        with pytest.raises(ValueError, match="extension"):
            load_quant_config(str(bad))

    def test_load_quant_config_yaml_missing(self, tmp_path, monkeypatch):
        from soup_cli.utils.save_formats import load_quant_config

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_quant_config(str(tmp_path / "missing.yaml"))


# --- H1 (TDD review): torchao kwarg allowlist ------------------------------


class TestTorchAOKwargAllowlist:
    def _setup(self, tmp_path):
        src = tmp_path / "model"
        src.mkdir()
        (src / "config.json").write_text("{}", encoding="utf-8")
        return src

    def _run(self, src, tmp_path, scheme, quant_config_data):
        from soup_cli.utils import save_formats

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_torchao = MagicMock()
        fake_torchao.quantization.Int4WeightOnlyConfig.return_value = MagicMock()
        fake_torchao.quantization.NVFP4Config.return_value = MagicMock()
        fake_torchao.quantize_ = MagicMock()

        original_torchao = sys.modules.get("torchao")
        original_torchao_q = sys.modules.get("torchao.quantization")
        sys.modules["torchao"] = fake_torchao
        sys.modules["torchao.quantization"] = fake_torchao.quantization
        try:
            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ), patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ):
                save_formats.export_torchao(
                    model_dir=str(src),
                    output_dir=str(tmp_path / "out"),
                    scheme=scheme,
                    quant_config_data=quant_config_data,
                )
        finally:
            if original_torchao is None:
                sys.modules.pop("torchao", None)
            else:
                sys.modules["torchao"] = original_torchao
            if original_torchao_q is None:
                sys.modules.pop("torchao.quantization", None)
            else:
                sys.modules["torchao.quantization"] = original_torchao_q

    def test_dunder_key_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = self._setup(tmp_path)
        with pytest.raises(ValueError, match="not allowed"):
            self._run(src, tmp_path, "Int4WeightOnly", {"__class__": "evil"})

    def test_unknown_key_on_int4_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = self._setup(tmp_path)
        with pytest.raises(ValueError, match="not allowed"):
            self._run(src, tmp_path, "Int4WeightOnly", {"unknown_key": 1})

    def test_allowed_int4_group_size(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = self._setup(tmp_path)
        # group_size is on the Int4 allowlist — should not raise
        self._run(src, tmp_path, "Int4WeightOnly", {"group_size": 32})

    def test_nvfp4_rejects_any_extra_kwargs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = self._setup(tmp_path)
        with pytest.raises(ValueError, match="not allowed"):
            self._run(src, tmp_path, "NVFP4", {"group_size": 32})


# --- M2 (TDD review): config.json symlink TOCTOU guard ---------------------


class TestDetectPrequantizedSymlinkRejection:
    @pytest.mark.skipif(
        sys.platform == "win32", reason="symlink rejection POSIX-only"
    )
    def test_config_json_symlink_returns_none(self, tmp_path, monkeypatch):
        """Security regression — `config.json` as a symlink is refused."""
        from soup_cli.autopilot.decisions import detect_prequantized_format_from_path

        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        real_config = tmp_path / "real_config.json"
        real_config.write_text(
            '{"quantization_config": {"quant_method": "gptq"}}',
            encoding="utf-8",
        )
        link = model_dir / "config.json"
        link.symlink_to(real_config)
        # Symlink config.json → soft-probe returns None instead of reading
        assert detect_prequantized_format_from_path("./model") is None

