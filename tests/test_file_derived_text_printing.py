"""Text read from files reaches the terminal as text, not as markup or control.

Several printers interpolated a value read from a file -- an adapter config's
``base_model_name_or_path``, an attestation sidecar's ``backend``, a layer name
from ``adapter_model.safetensors``, a benchmark name from evidence, a dataset
row's media path, a pydantic message quoting a config key -- into a Rich markup
string either raw, through ``rich.markup.escape`` alone (which leaves ESC and
C1 controls in place), or as ``{escape(x)!r}``, where ``repr()`` doubles the
backslash ``escape()`` added and the tag is live again. Each now goes through
``soup_cli.utils.terminal.for_terminal`` (or ``strip_control`` for a plain log
line).

Every test drives the real command or function that prints, with the module's
console swapped for ``Console(file=StringIO(), color_system=None)`` so Rich
emits no styling of its own: any ESC in the capture came from the value.
Non-ASCII characters are built with ``chr()`` so this file stays ASCII.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()

ESC = chr(27)
BEL = chr(7)
CSI_C1 = chr(0x9B)
BACKSLASH = chr(92)
MARKUP = "[b]x[/b][/]"

#: A value as it can sit in an adapter config, an evidence file or a dataset
#: row: an OSC title sequence, a C1 CSI, a tag pair and a bare closing tag.
FILE_TEXT = "org/m" + ESC + "]0;T" + BEL + CSI_C1 + "2J" + MARKUP

#: What the terminal must show for it: controls gone, brackets literal.
SHOWN = "org/m]0;T2J" + MARKUP


def _console(buffer: StringIO) -> Console:
    return Console(file=buffer, color_system=None, width=400)


def _swap_console(monkeypatch, module) -> StringIO:
    buffer = StringIO()
    monkeypatch.setattr(module, "console", _console(buffer))
    return buffer


def _assert_shown_literally(out: str, *, times: int = 1) -> None:
    assert out.count(SHOWN) >= times, out
    assert ESC not in out, repr(out)
    assert CSI_C1 not in out, repr(out)


def _adapter_dir(root, base_model: str = FILE_TEXT, name: str = "adp", **extra):
    adapter = root / name
    adapter.mkdir()
    config = {"base_model_name_or_path": base_model, "peft_type": "LORA", "r": 8}
    config.update(extra)
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return adapter


class _StopLoadingError(Exception):
    """Raised by the stand-in loaders once the line under test has printed."""


def _stand_in_hf_modules(monkeypatch, message: str = "stopped before the model load") -> list:
    """Replace ``transformers`` / ``peft`` with stand-ins that stop at the first load.

    The tokenizer loads (so the base-model line is reached); the first model
    load raises :class:`_StopLoadingError` with ``message``, before anything
    is downloaded.
    """
    calls: list = []

    def _model_from_pretrained(name, *args, **kwargs):
        calls.append(name)
        raise _StopLoadingError(message)

    tokenizer = types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=_model_from_pretrained
    )
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: tokenizer
    )
    peft = types.ModuleType("peft")
    peft.PeftModel = types.SimpleNamespace(from_pretrained=_model_from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    return calls


def _yaml_escape(code: int) -> str:
    return f"{BACKSLASH}x{code:02x}"


# ---------------------------------------------------------------------------
# soup attest verify: the sidecar's backend name
# ---------------------------------------------------------------------------


class TestAttestVerifyBackendName:
    def test_a_backend_name_containing_markup_keeps_exit_code_3(
        self, tmp_path, monkeypatch
    ) -> None:
        """Exit 3 is "invalid signature / policy mismatch"; exit 1 means the
        verifier was unavailable. A sidecar is the input being judged, so its
        content must not be able to turn the first into the second."""
        from soup_cli.commands import attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        (tmp_path / "stmt.json").write_text('{"_type": "x"}', encoding="utf-8")
        (tmp_path / "stmt.json.sig").write_text(
            json.dumps({"backend": "[/]x" + ESC + CSI_C1, "signature": "00"}),
            encoding="utf-8",
        )
        buffer = _swap_console(monkeypatch, attest_cmd)
        result = runner.invoke(
            attest_cmd.app, ["verify", "stmt.json", "--signature", "stmt.json.sig"]
        )
        out = buffer.getvalue()
        assert result.exit_code == 3, (out, repr(result.exception))
        # The quoted form is repr()'s, so the two controls arrive as their
        # escape text and the tag as literal brackets.
        quoted = "'[/]x" + BACKSLASH + "x1b" + BACKSLASH + "x9b'"
        assert f"Signature backend {quoted} is not cryptographically verifiable" in out, out

    def test_an_ordinary_backend_name_is_still_shown_quoted(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import attest as attest_cmd

        monkeypatch.chdir(tmp_path)
        (tmp_path / "stmt.json").write_text('{"_type": "x"}', encoding="utf-8")
        (tmp_path / "stmt.json.sig").write_text(
            json.dumps({"backend": "hmac", "signature": "00"}), encoding="utf-8"
        )
        buffer = _swap_console(monkeypatch, attest_cmd)
        result = runner.invoke(
            attest_cmd.app, ["verify", "stmt.json", "--signature", "stmt.json.sig"]
        )
        assert result.exit_code == 3, (buffer.getvalue(), repr(result.exception))
        assert "Signature backend 'hmac' is not" in buffer.getvalue()


# ---------------------------------------------------------------------------
# The adapter config's base_model_name_or_path: chat / merge / serve / export
# ---------------------------------------------------------------------------


class TestAdapterBaseModelName:
    def test_chat_loading_panel(self, tmp_path, monkeypatch) -> None:
        from soup_cli.cli import app
        from soup_cli.commands import chat as chat_cmd

        adapter = _adapter_dir(tmp_path)
        buffer = _swap_console(monkeypatch, chat_cmd)

        def _stop_loading(**kwargs):
            raise _StopLoadingError("stopped before loading")

        monkeypatch.setattr(chat_cmd, "_load_model", _stop_loading)
        result = runner.invoke(app, ["chat", "--model", str(adapter), "--device", "cpu"])
        assert isinstance(result.exception, _StopLoadingError), (
            buffer.getvalue(),
            repr(result.exception),
        )
        _assert_shown_literally(buffer.getvalue())

    def test_chat_loading_base_model_line(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import chat as chat_cmd

        calls = _stand_in_hf_modules(monkeypatch)
        buffer = _swap_console(monkeypatch, chat_cmd)
        with pytest.raises(_StopLoadingError):
            chat_cmd._load_model(
                model_path=str(tmp_path), base_model=FILE_TEXT, is_adapter=True, device="cpu"
            )
        assert calls == [FILE_TEXT]
        out = buffer.getvalue()
        assert "Loading base model: " + SHOWN in out, out
        _assert_shown_literally(out)

    def test_merge_plan_panel_loading_line_and_failure(self, tmp_path, monkeypatch) -> None:
        """The plan panel, the loading line, and the failure line: a loader
        error commonly quotes the repo id it could not load."""
        from soup_cli.cli import app
        from soup_cli.commands import merge as merge_cmd

        monkeypatch.chdir(tmp_path)
        _adapter_dir(tmp_path)
        _stand_in_hf_modules(monkeypatch, message="cannot load " + FILE_TEXT)
        buffer = _swap_console(monkeypatch, merge_cmd)
        result = runner.invoke(app, ["merge", "--adapter", "adp", "--output", "merged"])
        out = buffer.getvalue()
        assert result.exit_code == 1, (out, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "Merge failed: cannot load " + SHOWN in out, out
        assert "Loading base model: " + SHOWN in out, out
        _assert_shown_literally(out, times=3)

    def test_serve_loading_panel(self, tmp_path, monkeypatch) -> None:
        pytest.importorskip("fastapi")
        pytest.importorskip("uvicorn")
        from soup_cli.cli import app
        from soup_cli.commands import serve as serve_cmd

        adapter = _adapter_dir(tmp_path)
        buffer = _swap_console(monkeypatch, serve_cmd)
        # An unknown --structured-output mode is refused right after the
        # panel, so the command stops before any model is loaded.
        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(adapter),
                "--device",
                "cpu",
                "--structured-output",
                "unknown-mode",
            ],
        )
        out = buffer.getvalue()
        assert result.exit_code == 1, (out, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "Unknown structured-output mode" in out, out
        _assert_shown_literally(out)

    def test_serve_loading_base_model_line(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import serve as serve_cmd

        _stand_in_hf_modules(monkeypatch)
        buffer = _swap_console(monkeypatch, serve_cmd)
        with pytest.raises(_StopLoadingError):
            serve_cmd._load_model(
                model_path=str(tmp_path), base_model=FILE_TEXT, is_adapter=True, device="cpu"
            )
        out = buffer.getvalue()
        assert "Loading base model: " + SHOWN in out, out
        _assert_shown_literally(out)

    def test_export_merge_step_loading_line(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import export as export_cmd

        adapter = _adapter_dir(tmp_path)
        _stand_in_hf_modules(monkeypatch)
        buffer = _swap_console(monkeypatch, export_cmd)
        with pytest.raises(_StopLoadingError):
            export_cmd._merge_adapter(str(adapter), FILE_TEXT, str(tmp_path / "merged"))
        out = buffer.getvalue()
        assert "Loading base model: " + SHOWN in out, out
        _assert_shown_literally(out)

    def test_remote_code_warning_panel(self) -> None:
        from soup_cli.utils.trust_remote import resolve_trust_remote_code

        buffer = StringIO()
        assert resolve_trust_remote_code(FILE_TEXT, requested=True, console=_console(buffer))
        out = buffer.getvalue()
        assert "--trust-remote-code is enabled for" in out, out
        _assert_shown_literally(out)


class TestEvalBenchmarkBaseModel:
    """``soup eval benchmark`` reads the base model from the adapter config.

    ``lm_eval`` is made unimportable, so the command stops at its own
    "not installed" message, which comes right after the line under test.
    """

    def test_the_detected_base_model_line(self, tmp_path, monkeypatch) -> None:
        from soup_cli.cli import app
        from soup_cli.commands import eval as eval_cmd

        adapter = _adapter_dir(tmp_path)
        monkeypatch.setitem(sys.modules, "lm_eval", None)
        buffer = _swap_console(monkeypatch, eval_cmd)
        result = runner.invoke(app, ["eval", "benchmark", "--model", str(adapter)])
        out = buffer.getvalue()
        assert result.exit_code == 1, (out, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "lm-eval not installed" in out, out
        assert "LoRA adapter detected. Base model: " + SHOWN in out, out
        _assert_shown_literally(out)

    def test_a_refused_base_model_is_quoted_literally(self, tmp_path, monkeypatch) -> None:
        """A ',' or '=' in the base model is refused before lm-eval sees it,
        and the refusal quotes the value back."""
        from soup_cli.cli import app
        from soup_cli.commands import eval as eval_cmd

        adapter = _adapter_dir(tmp_path, base_model="org/m,[/]x")
        buffer = _swap_console(monkeypatch, eval_cmd)
        result = runner.invoke(app, ["eval", "benchmark", "--model", str(adapter)])
        out = buffer.getvalue()
        assert result.exit_code == 1, (out, repr(result.exception))
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "Refusing to evaluate:" in out and "('org/m,[/]x')" in out, out


# ---------------------------------------------------------------------------
# soup adapters list / info / compare / diff
# ---------------------------------------------------------------------------


class TestAdaptersCommands:
    def test_list(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import adapters as adapters_cmd

        scan = tmp_path / "scan"
        scan.mkdir()
        _adapter_dir(scan, peft_type=FILE_TEXT)
        buffer = _swap_console(monkeypatch, adapters_cmd)
        result = runner.invoke(adapters_cmd.app, ["list", str(scan)])
        assert result.exit_code == 0, (buffer.getvalue(), repr(result.exception))
        _assert_shown_literally(buffer.getvalue(), times=2)

    def test_list_directory_names(self, tmp_path, monkeypatch) -> None:
        """Adapter directory names come from the disk being scanned.

        POSIX file names may hold control characters; Windows file names may
        not, so discovery is stubbed to hand back such names without creating
        them. One adapter reads cleanly, the other takes the unreadable-config
        branch, which prints its full path.
        """
        from soup_cli.commands import adapters as adapters_cmd

        scan = (tmp_path / "scan").resolve()
        scan.mkdir()
        readable = scan / ("ok" + ESC + "]0;T" + CSI_C1 + "2J[b]x")
        unreadable = scan / ("bad" + ESC + "]0;T" + CSI_C1 + "2J[b]y")

        def _read(adapter_path):
            if adapter_path == unreadable:
                raise json.JSONDecodeError("unreadable", "{", 0)
            return {"base_model_name_or_path": "m", "r": 8, "peft_type": "LORA"}

        monkeypatch.setattr(
            adapters_cmd, "_find_adapters", lambda directory: [readable, unreadable]
        )
        monkeypatch.setattr(adapters_cmd, "_read_adapter_config", _read)
        monkeypatch.setattr(adapters_cmd, "_get_adapter_size", lambda adapter_path: "1 B")
        buffer = _swap_console(monkeypatch, adapters_cmd)
        result = runner.invoke(adapters_cmd.app, ["list", str(scan)])
        out = buffer.getvalue()
        assert result.exit_code == 0, (out, repr(result.exception))
        assert "ok]0;T2J[b]x" in out and "bad]0;T2J[b]y" in out, out
        assert ESC not in out and CSI_C1 not in out, repr(out)

    def test_info(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import adapters as adapters_cmd

        adapter = _adapter_dir(tmp_path, task_type=FILE_TEXT, target_modules=[FILE_TEXT])
        buffer = _swap_console(monkeypatch, adapters_cmd)
        result = runner.invoke(adapters_cmd.app, ["info", str(adapter)])
        assert result.exit_code == 0, (buffer.getvalue(), repr(result.exception))
        _assert_shown_literally(buffer.getvalue(), times=3)

    def test_compare(self, tmp_path, monkeypatch) -> None:
        from soup_cli.commands import adapters as adapters_cmd

        first = _adapter_dir(tmp_path, name="a1")
        second = _adapter_dir(tmp_path, base_model="other", name="a2")
        buffer = _swap_console(monkeypatch, adapters_cmd)
        result = runner.invoke(adapters_cmd.app, ["compare", str(first), str(second)])
        assert result.exit_code == 0, (buffer.getvalue(), repr(result.exception))
        _assert_shown_literally(buffer.getvalue())

    def test_diff_layer_names(self, tmp_path, monkeypatch) -> None:
        np = pytest.importorskip("numpy")
        safetensors_numpy = pytest.importorskip("safetensors.numpy")
        from soup_cli.commands import adapters as adapters_cmd

        monkeypatch.chdir(tmp_path)
        for name, scale in (("a1", 1.0), ("a2", 2.0)):
            adapter = _adapter_dir(tmp_path, name=name)
            safetensors_numpy.save_file(
                {FILE_TEXT: np.full((2, 2), scale, dtype=np.float32)},
                str(adapter / "adapter_model.safetensors"),
            )
        buffer = _swap_console(monkeypatch, adapters_cmd)
        result = runner.invoke(adapters_cmd.app, ["diff", "a1", "a2"])
        assert result.exit_code == 0, (buffer.getvalue(), repr(result.exception))
        _assert_shown_literally(buffer.getvalue())


# ---------------------------------------------------------------------------
# soup ship: benchmark names in the verdict footer
# ---------------------------------------------------------------------------


class TestShipVerdictFooter:
    def test_regressed_benchmark_names(self) -> None:
        from soup_cli.utils.ship_verdict import (
            build_task_win,
            compute_benchmark_deltas,
            decide_ship,
            render_ship_panel,
        )

        verdict = decide_ship(
            build_task_win("metric", 0.40, 0.99),
            compute_benchmark_deltas({FILE_TEXT: 0.80}, {FILE_TEXT: 0.50}),
        )
        buffer = StringIO()
        _console(buffer).print(render_ship_panel(verdict))
        out = buffer.getvalue()
        assert "regressed past" in out, out
        _assert_shown_literally(out, times=2)


# ---------------------------------------------------------------------------
# soup edit diff: run ids and probe text
# ---------------------------------------------------------------------------


class TestEditDiffTable:
    def test_run_ids_prompts_and_outputs(self) -> None:
        from soup_cli import __version__
        from soup_cli.utils.edit_diff import DiffReport, FactChange, render_diff_table

        report = DiffReport(
            before_run_id=FILE_TEXT,
            after_run_id="run-2",
            changes=(FactChange(prompt=FILE_TEXT, before=FILE_TEXT, after="y", changed=True),),
            total_probes=1,
            soup_version=__version__,
        )
        buffer = StringIO()
        render_diff_table(report, _console(buffer))
        _assert_shown_literally(buffer.getvalue(), times=3)


# ---------------------------------------------------------------------------
# SFT vision / audio: a dataset row's media path in the "cannot open" warning
# ---------------------------------------------------------------------------


class TestSftMediaPathWarnings:
    def test_vision_image_path(self, monkeypatch) -> None:
        pytest.importorskip("PIL")
        pytest.importorskip("datasets")
        from soup_cli.trainer import sft as sft_module

        buffer = _swap_console(monkeypatch, sft_module)
        wrapper = object.__new__(sft_module.SFTTrainerWrapper)
        row = {"messages": [{"role": "user", "content": "hi"}], "image": "imgs/" + FILE_TEXT}
        wrapper._prepare_vision_dataset({"train": [row]})
        out = buffer.getvalue()
        assert "cannot open image: imgs/" + SHOWN in out, out
        _assert_shown_literally(out)

    def test_audio_path(self, monkeypatch) -> None:
        pytest.importorskip("datasets")
        from soup_cli.trainer import sft as sft_module

        def _load(path, *args, **kwargs):
            raise FileNotFoundError(path)

        librosa = types.ModuleType("librosa")
        librosa.load = _load
        monkeypatch.setitem(sys.modules, "librosa", librosa)
        buffer = _swap_console(monkeypatch, sft_module)
        wrapper = object.__new__(sft_module.SFTTrainerWrapper)
        wrapper.processor = types.SimpleNamespace()
        row = {"messages": [{"role": "user", "content": "hi"}], "audio": "clips/" + FILE_TEXT}
        wrapper._prepare_audio_dataset({"train": [row]})
        out = buffer.getvalue()
        assert "cannot open audio: clips/" + SHOWN in out, out
        _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# soup infer --task asr: a dataset row's audio file name in the skip warnings
# ---------------------------------------------------------------------------


class TestInferAsrAudioName:
    """The skip and metric-skip warnings quote the row's audio file name.

    The name was escaped and then quoted (``name = for_terminal(...)``, then
    ``{name!r}``): ``repr()`` doubled the backslash the escape added, so
    ``[b]`` became a live tag. Quoting first keeps it text.
    """

    NAME = "[b]a.wav"

    def _run(self, tmp_path, monkeypatch, row: dict, transcriber):
        from soup_cli.cli import app
        from soup_cli.commands import infer as infer_cmd

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        monkeypatch.setattr(infer_cmd, "_ASR_TRANSCRIBER_OVERRIDE", transcriber)
        buffer = _swap_console(monkeypatch, infer_cmd)
        result = runner.invoke(
            app,
            [
                "infer",
                "--task",
                "asr",
                "--model",
                "m",
                "--input",
                "in.jsonl",
                "--output",
                "out.jsonl",
            ],
        )
        return result, buffer.getvalue()

    def test_a_skipped_row(self, tmp_path, monkeypatch) -> None:
        def _cannot_decode(path):
            raise OSError("cannot decode")

        result, out = self._run(tmp_path, monkeypatch, {"audio": self.NAME}, _cannot_decode)
        # The only row was skipped, which the command reports with exit 2.
        assert result.exit_code == 2, (out, repr(result.exception))
        assert "Skipped '[b]a.wav': cannot decode" in out, out

    def test_a_row_whose_metric_is_skipped(self, tmp_path, monkeypatch) -> None:
        # A reference over the WER/CER input cap is transcribed but unscored.
        row = {"audio": self.NAME, "text": "word " * 300_000}
        result, out = self._run(tmp_path, monkeypatch, row, lambda path: "hello")
        assert result.exit_code == 0, (out, repr(result.exception))
        assert "Metric skipped for '[b]a.wav':" in out, out


# ---------------------------------------------------------------------------
# Config loader: a validation message quoting a config key
# ---------------------------------------------------------------------------


class TestConfigValidationErrorLines:
    def test_a_message_quoting_a_key_is_printed_literally(self, tmp_path, monkeypatch) -> None:
        from soup_cli.config import loader

        buffer = _swap_console(monkeypatch, loader)
        path = tmp_path / "soup.yaml"
        path.write_text(
            "base: test-model\n"
            "data:\n"
            "  train: ./data.jsonl\n"
            "training:\n"
            "  lora:\n"
            "    rank_pattern:\n"
            '      "q[/]x": "eight"\n',
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        out = buffer.getvalue()
        assert "rank_pattern" in out and "'q[/]x'" in out, out

    def test_a_location_naming_a_key_is_printed_literally(self, tmp_path, monkeypatch) -> None:
        """No schema field puts a key from the file into ``loc`` today -- the
        dict-typed fields are rebuilt by their own validators -- so the error
        is built by hand: the printer must not depend on that staying true."""
        from pydantic_core import InitErrorDetails, ValidationError

        from soup_cli.config import loader

        def _refuse(raw):
            raise ValidationError.from_exception_data(
                "SoupConfig",
                [InitErrorDetails(type="missing", loc=("training", "q[/]" + ESC + "x"), input={})],
            )

        monkeypatch.setattr(loader, "_build_config", _refuse)
        buffer = _swap_console(monkeypatch, loader)
        path = tmp_path / "soup.yaml"
        path.write_text("base: test-model\ndata:\n  train: ./data.jsonl\n", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            loader.load_config(path)
        assert excinfo.value.code == 1
        out = buffer.getvalue()
        assert "training -> q[/]x:" in out, out
        assert ESC not in out, repr(out)


# ---------------------------------------------------------------------------
# soup probe interference: keys of the losses JSON
# ---------------------------------------------------------------------------


class TestProbeInterferenceLossKeys:
    @pytest.mark.parametrize(
        "losses, expected",
        [
            ({"[/]x": 1.0}, "Invalid losses key '[/]x'"),
            ({"a|[/]x": "high"}, "losses['a|[/]x'] must be numeric"),
        ],
        ids=["key-without-separator", "non-numeric-value"],
    )
    def test_a_key_containing_markup_keeps_exit_code_2(
        self, tmp_path, monkeypatch, losses, expected
    ) -> None:
        from soup_cli.commands import probe as probe_cmd

        monkeypatch.chdir(tmp_path)
        (tmp_path / "losses.json").write_text(
            json.dumps({"adapters": ["a", "b"], "losses": losses}), encoding="utf-8"
        )
        buffer = _swap_console(monkeypatch, probe_cmd)
        result = runner.invoke(probe_cmd.app, ["interference", "losses.json"])
        out = buffer.getvalue()
        assert result.exit_code == 2, (out, repr(result.exception))
        assert expected in out, out


# ---------------------------------------------------------------------------
# Web UI: the server-side log line for a refused config
# ---------------------------------------------------------------------------


class TestWebUiConfigLogLines:
    LOGGER = "soup_cli.ui.app"
    KEY = ESC + "]0;T" + CSI_C1 + "2Jquantizaton"

    @staticmethod
    def _client():
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.ui.app import create_app, get_auth_token

        return TestClient(create_app()), {"Authorization": f"Bearer {get_auth_token()}"}

    def _logged(self, caplog) -> list[str]:
        return [record.getMessage() for record in caplog.records if record.name == self.LOGGER]

    def _assert_clean(self, messages: list[str], visible: str) -> None:
        assert any(visible in message for message in messages), messages
        for message in messages:
            assert ESC not in message and CSI_C1 not in message, repr(message)

    def test_train_start_refused_config(self, caplog) -> None:
        client, headers = self._client()
        config_yaml = (
            "base: m\n"
            "data:\n"
            "  train: ./t.jsonl\n"
            "training:\n"
            f'  "{_yaml_escape(0x1B)}]0;T{_yaml_escape(0x9B)}2Jquantizaton": 4bit\n'
        )
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            response = client.post(
                "/api/train/start", json={"config_yaml": config_yaml}, headers=headers
            )
        assert response.status_code == 400, response.text
        self._assert_clean(self._logged(caplog), "2Jquantizaton")

    def test_config_from_form_refused_config(self, caplog) -> None:
        client, headers = self._client()
        body = {"base": "m", "data": {"train": "./t.jsonl"}, "training": {self.KEY: "4bit"}}
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            response = client.post("/api/config/from-form", json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert "error" in response.json(), response.json()
        self._assert_clean(self._logged(caplog), "2Jquantizaton")

    @pytest.mark.parametrize(
        "route, raised",
        [("/api/train/start", RuntimeError), ("/api/config/from-form", TypeError)],
        ids=["train-start", "from-form"],
    )
    def test_the_other_error_branch(self, caplog, monkeypatch, route, raised) -> None:
        from soup_cli.config import loader

        def _refuse(text):
            raise raised("refused " + self.KEY)

        monkeypatch.setattr(loader, "load_config_from_string", _refuse)
        client, headers = self._client()
        body = (
            {"config_yaml": "base: m\n"}
            if route == "/api/train/start"
            else {"base": "m", "data": {"train": "./t.jsonl"}}
        )
        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            client.post(route, json=body, headers=headers)
        self._assert_clean(self._logged(caplog), "2Jquantizaton")
