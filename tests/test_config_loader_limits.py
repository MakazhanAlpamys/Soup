"""soup.yaml is loaded under the same size, nesting and alias limits as a can's config."""

from __future__ import annotations

import os
import re
import sys

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config import loader
from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.utils import yaml_limits
from soup_cli.utils.yaml_limits import (
    ALIAS_EXPANSION_SLACK,
    MAX_CONFIG_BYTES,
    MAX_YAML_NESTING,
    YamlLimitError,
)
from tests.ui_helpers import ui_client

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_VALID = (
    "base: meta-llama/Llama-3.1-8B-Instruct\n"
    "task: sft\n"
    "data:\n"
    "  train: ./data.jsonl\n"
    "  format: alpaca\n"
    "training:\n"
    "  epochs: 1\n"
    "  lora:\n"
    "    r: 8\n"
    "    alpha: 16\n"
    "output: ./output\n"
)


def _plain(text: str) -> str:
    """Rich colours numbers and wraps at the terminal width: undo both."""
    return " ".join(_ANSI_RE.sub("", text).split())


# ---------------------------------------------------------------------------
# Configs named by the limit they sit under or cross
# ---------------------------------------------------------------------------


def _config_of_size(size: int) -> str:
    """A valid config padded with a trailing comment to exactly ``size`` bytes."""
    padding = size - len(_VALID.encode()) - 2
    assert padding >= 0
    text = _VALID + "#" + "x" * padding + "\n"
    assert len(text.encode()) == size
    return text


def _config_nested(levels: int) -> str:
    """The valid config plus one key whose value brings the nesting to ``levels``."""
    inner = levels - 1  # the root mapping is the first level
    return _VALID + "extra: " + "[" * inner + "]" * inner + "\n"


def _config_with_alias_fan_out(levels: int) -> str:
    """The valid config plus lists that each reference the previous one ten times."""
    text = _VALID + "l0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for index in range(1, levels + 1):
        text += f"l{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]\n"
    return text


_WITH_MODEST_ALIASES = _VALID.replace(
    "base: meta-llama/Llama-3.1-8B-Instruct\n",
    "base: &name meta-llama/Llama-3.1-8B-Instruct\n",
).replace("output: ./output\n", "output: *name\n")


def _expansion_message(what: str, text: str) -> str:
    size = len(text.encode())
    return (
        f"{what} expands to more than {size + ALIAS_EXPANSION_SLACK} nodes "
        f"from {size} bytes of YAML; refusing"
    )


_SIZE_MESSAGE = f"is larger than {MAX_CONFIG_BYTES} bytes; refusing"
_NESTING_MESSAGE = f"nests collections more than {MAX_YAML_NESTING} levels deep; refusing"


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _refused(capsys, path="soup.yaml") -> str:
    """Load ``path``, require a clean exit 1, and return what was printed."""
    with pytest.raises(SystemExit) as raised:
        load_config(path)
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out + captured.err
    return _plain(captured.out)


def test_the_limits_are_the_shared_ones():
    from soup_cli.cans import unpack

    assert MAX_CONFIG_BYTES == 1024 * 1024
    assert loader.MAX_CONFIG_BYTES is yaml_limits.MAX_CONFIG_BYTES
    assert unpack.MAX_CONFIG_BYTES is yaml_limits.MAX_CONFIG_BYTES
    assert MAX_YAML_NESTING == 100
    assert ALIAS_EXPANSION_SLACK == 10_000


# ---------------------------------------------------------------------------
# load_config: a file, SystemExit(1) with a message
# ---------------------------------------------------------------------------


class TestConfigFile:
    def test_config_exactly_at_the_size_limit_loads(self, in_tmp):
        (in_tmp / "soup.yaml").write_bytes(_config_of_size(MAX_CONFIG_BYTES).encode())
        assert load_config("soup.yaml").base == "meta-llama/Llama-3.1-8B-Instruct"

    def test_config_one_byte_over_the_size_limit_is_refused(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_bytes(_config_of_size(MAX_CONFIG_BYTES + 1).encode())
        assert _refused(capsys) == f"Config file soup.yaml {_SIZE_MESSAGE}"

    def test_size_limit_is_the_one_the_loader_module_names(self, in_tmp, capsys, monkeypatch):
        monkeypatch.setattr(loader, "MAX_CONFIG_BYTES", len(_VALID.encode()) - 1)
        (in_tmp / "soup.yaml").write_bytes(_VALID.encode())
        expected = f"Config file soup.yaml is larger than {len(_VALID.encode()) - 1} bytes"
        assert expected in _refused(capsys)

    def test_config_nested_to_the_limit_is_not_refused_for_nesting(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text(_config_nested(MAX_YAML_NESTING), encoding="utf-8")
        output = _refused(capsys)  # 'extra' is not a config key, so the load still stops
        assert "nests collections" not in output
        assert "Config validation error" in output
        assert "extra" in output

    def test_config_nested_one_level_past_the_limit_is_refused(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text(
            _config_nested(MAX_YAML_NESTING + 1), encoding="utf-8"
        )
        output = _refused(capsys)
        assert f"Config file soup.yaml {_NESTING_MESSAGE}" in output
        assert "Config validation error" not in output

    def test_config_nested_far_past_the_limit_is_refused_the_same_way(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text(_config_nested(20_000), encoding="utf-8")
        assert f"Config file soup.yaml {_NESTING_MESSAGE}" in _refused(capsys)

    def test_config_with_modest_alias_reuse_loads(self, in_tmp):
        (in_tmp / "soup.yaml").write_text(_WITH_MODEST_ALIASES, encoding="utf-8")
        config = load_config("soup.yaml")
        assert config.output == config.base == "meta-llama/Llama-3.1-8B-Instruct"

    def test_config_over_the_node_allowance_is_refused(self, in_tmp, capsys):
        text = _config_with_alias_fan_out(9)
        (in_tmp / "soup.yaml").write_bytes(text.encode())
        output = _refused(capsys)
        assert _expansion_message("Config file soup.yaml", text) in output
        assert "Config validation error" not in output

    def test_limits_are_checked_before_the_config_is_walked(self, in_tmp, capsys, monkeypatch):
        def _no_walk(*args, **kwargs):
            raise AssertionError("an over-limit config must not be walked")

        monkeypatch.setattr(loader, "find_unknown_config_keys", _no_walk)
        monkeypatch.setattr(loader, "_build_config", _no_walk)
        text = _config_with_alias_fan_out(9)
        (in_tmp / "soup.yaml").write_bytes(text.encode())
        assert _expansion_message("Config file soup.yaml", text) in _refused(capsys)

    def test_refusal_names_the_path_as_it_was_given(self, in_tmp, capsys):
        (in_tmp / "configs").mkdir()
        (in_tmp / "configs" / "run.yaml").write_text(
            _config_nested(MAX_YAML_NESTING + 1), encoding="utf-8"
        )
        output = _refused(capsys, in_tmp.joinpath("configs", "run.yaml").relative_to(in_tmp))
        assert "Config file configs" in output
        assert f"run.yaml {_NESTING_MESSAGE}" in output

    @pytest.mark.parametrize(
        "encode",
        [
            lambda text: text.encode(),
            lambda text: text.replace("\n", "\r\n").encode(),
            lambda text: b"\xef\xbb\xbf" + text.encode(),
            lambda text: b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode(),
        ],
        ids=["lf", "crlf", "bom", "bom-crlf"],
    )
    def test_line_ends_and_bom_load_the_same_config(self, in_tmp, encode):
        (in_tmp / "soup.yaml").write_bytes(encode(_VALID))
        assert load_config("soup.yaml") == load_config_from_string(_VALID)

    def test_non_ascii_utf8_config_loads(self, in_tmp):
        text = _VALID.replace("output: ./output\n", "output: ./sortie-été-☃\n")
        (in_tmp / "soup.yaml").write_bytes(text.encode("utf-8"))
        assert load_config("soup.yaml").output == "./sortie-été-☃"

    @pytest.mark.parametrize("text", ["", "\n\n", "# nothing but a comment\n"])
    def test_empty_config_keeps_its_message(self, in_tmp, capsys, text):
        (in_tmp / "soup.yaml").write_text(text, encoding="utf-8")
        assert _refused(capsys) == "Config file is empty"

    @pytest.mark.parametrize(
        ("text", "kind"), [("- a\n- b\n", "list"), ("just text\n", "str"), ("42\n", "int")]
    )
    def test_non_mapping_config_keeps_its_message(self, in_tmp, capsys, text, kind):
        (in_tmp / "soup.yaml").write_text(text, encoding="utf-8")
        assert _refused(capsys) == f"Config must be a YAML mapping, got {kind}"

    def test_unknown_key_refusal_is_unchanged(self, in_tmp, capsys):
        from soup_cli.config.unknown_keys import find_unknown_config_keys, format_unknown_keys

        text = _VALID + "trainning:\n  epochs: 3\n"
        (in_tmp / "soup.yaml").write_text(text, encoding="utf-8")
        expected = format_unknown_keys(
            find_unknown_config_keys(yaml.safe_load(text)), include_deadline=False
        )
        assert _refused(capsys) == _plain(f"Config validation error: {expected}")

    def test_syntax_error_is_one_line_that_names_the_file_and_the_place(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text("base: [unclosed\ntask: }\n", encoding="utf-8")
        output = _refused(capsys)
        assert output == (
            "Config file soup.yaml is not valid YAML: "
            "expected ',' or ']', but got ':' (line 2, column 5)"
        )

    def test_bytes_that_are_not_utf8_are_one_line_that_names_the_file(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_bytes(b"base: caf\xe9\n")
        assert _refused(capsys) == (
            "Config file soup.yaml is not valid UTF-8: invalid continuation byte at byte 9"
        )

    def test_a_directory_is_one_line_that_names_it(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").mkdir()
        assert _refused(capsys) == "Config file soup.yaml is a directory, not a file"

    def test_a_date_that_does_not_exist_is_one_line_that_names_the_place(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text(_VALID + "extra: 2026-02-30\n", encoding="utf-8")
        assert _refused(capsys) == (
            "Config file soup.yaml is not valid YAML: cannot build this value: "
            "day is out of range for month (line 12, column 8)"
        )

    @pytest.mark.skipif(
        not hasattr(sys, "get_int_max_str_digits"), reason="no integer digit limit here"
    )
    def test_a_number_too_long_to_build_is_one_line_that_names_the_place(self, in_tmp, capsys):
        text = _VALID + "extra: " + "7" * 5000 + "\n"
        (in_tmp / "soup.yaml").write_text(text, encoding="utf-8")
        output = _refused(capsys)
        assert output.startswith(
            "Config file soup.yaml is not valid YAML: cannot build this value: "
        )
        assert output.endswith("(line 12, column 8)")

    @pytest.mark.parametrize(
        "line",
        [
            "base: *" + "a" * 5000,
            "base: !" + "t" * 5000 + " x",
            "base: [" + "x" * 5000 + "\ntask: }",
        ],
        ids=["alias-name", "tag-name", "flow-text"],
    )
    def test_text_quoted_from_the_file_is_cut_to_two_hundred_characters(self, in_tmp, capsys, line):
        (in_tmp / "soup.yaml").write_text(line + "\n", encoding="utf-8")
        output = _refused(capsys)
        prefix = "Config file soup.yaml is not valid YAML: "
        assert output.startswith(prefix)
        detail, _, place = output[len(prefix):].rpartition(" (line ")
        assert place.endswith(")") and "column" in place
        # Rich wraps the long word, so count it without the breaks.
        assert len(detail.replace(" ", "")) <= loader._MAX_DETAIL_CHARS + 3
        assert len(output) < 300

    @pytest.mark.parametrize(
        ("scalar", "name"),
        [("!!bool foo", "bool"), ("!!timestamp foo", "timestamp"), ('!!int ""', "int")],
    )
    def test_text_an_explicit_tag_cannot_convert_is_one_line_too(
        self, in_tmp, capsys, scalar, name
    ):
        (in_tmp / "soup.yaml").write_text(_VALID + f"extra: {scalar}\n", encoding="utf-8")
        assert _refused(capsys) == (
            "Config file soup.yaml is not valid YAML: cannot build this value: "
            f"not a valid {name} (line 12, column 8)"
        )

    def test_a_long_detail_ends_with_an_ellipsis_and_a_short_one_does_not(self):
        assert loader._MAX_DETAIL_CHARS == 200
        assert loader._clip("a  b\n c") == "a b c"
        assert loader._clip("x" * 200) == "x" * 200
        assert loader._clip("x" * 201) == "x" * 200 + "..."

    def test_a_file_the_system_will_not_open_is_one_line_with_the_reason(
        self, in_tmp, capsys, monkeypatch
    ):
        def _denied(path, what, *, max_bytes):
            raise PermissionError(13, "Permission denied", str(path))

        monkeypatch.setattr(loader, "load_yaml_file_limited", _denied)
        assert _refused(capsys) == "Config file soup.yaml cannot be read: Permission denied"

    def test_an_error_that_is_not_about_the_file_is_passed_on_as_itself(self, in_tmp, monkeypatch):
        def _broken(path, what, *, max_bytes):
            raise ValueError("an internal mistake")

        monkeypatch.setattr(loader, "load_yaml_file_limited", _broken)
        with pytest.raises(ValueError, match="^an internal mistake$"):
            load_config("soup.yaml")

    def test_a_path_with_control_characters_and_markup_is_printed_escaped(
        self, capsys, monkeypatch
    ):
        def _over(path, what, *, max_bytes):
            raise YamlLimitError(f"{what} is larger than {max_bytes} bytes; refusing")

        monkeypatch.setattr(loader, "load_yaml_file_limited", _over)
        with pytest.raises(SystemExit):
            load_config("run\x1b[31m\x07[bold]x.yaml")
        printed = capsys.readouterr().out
        # Rich may add colour codes of its own, so compare the text without them: the
        # escape typed in the path is gone, what followed it is text, the tag is literal.
        assert "\x07" not in printed
        assert _plain(printed) == (
            f"Config file run[31m[bold]x.yaml is larger than {MAX_CONFIG_BYTES} bytes; refusing"
        )

    def test_missing_config_is_still_file_not_found(self, in_tmp):
        with pytest.raises(FileNotFoundError):
            load_config("absent.yaml")

    @pytest.mark.parametrize(
        ("content", "error"),
        [
            (b"base: [unclosed\ntask: }\n", yaml.YAMLError),
            (b"base: caf\xe9\n", UnicodeDecodeError),
            (b"extra: 2026-02-30\n", yaml.YAMLError),
            (b"#" + b"x" * MAX_CONFIG_BYTES, YamlLimitError),
            (("deep: " + "[" * MAX_YAML_NESTING + "]" * MAX_YAML_NESTING).encode(), YamlLimitError),
        ],
        ids=["syntax", "not-utf8", "date", "size", "nesting"],
    )
    def test_a_caller_that_asks_gets_the_exception_and_no_output(
        self, in_tmp, capsys, content, error
    ):
        (in_tmp / "soup.yaml").write_bytes(content)
        with pytest.raises(error):
            load_config("soup.yaml", raise_unreadable=True)
        captured = capsys.readouterr()
        assert captured.out == captured.err == ""

    def test_a_caller_that_asks_gets_the_os_error_for_a_directory(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").mkdir()
        with pytest.raises(OSError):
            load_config("soup.yaml", raise_unreadable=True)
        assert capsys.readouterr().out == ""

    def test_asking_changes_nothing_for_a_config_that_reads(self, in_tmp, capsys):
        (in_tmp / "soup.yaml").write_text(_VALID, encoding="utf-8")
        assert load_config("soup.yaml", raise_unreadable=True) == load_config("soup.yaml")
        (in_tmp / "soup.yaml").write_text(_VALID + "trainning: {}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_config("soup.yaml", raise_unreadable=True)
        assert "Config validation error" in _plain(capsys.readouterr().out)

    def test_find_lr_words_an_unreadable_config_in_its_own_sentence(self, in_tmp):
        """It quotes the reason itself, so the loader hands the exception over."""
        from soup_cli.cli import app

        (in_tmp / "soup.yaml").write_text("base: m\ndata: [unclosed\n", encoding="utf-8")
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--find-lr"])
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "--find-lr: config load failed (while parsing a flow sequence" in output, output
        assert "data: [unclosed" in output, output
        assert "is not valid YAML" not in output, output

    @pytest.mark.parametrize(
        ("argv", "code", "also"),
        [
            (["bench", "train", "--config", "soup.yaml", "--steps", "4", "--warmup", "1"], 1, ""),
            (["doctor", "--config", "soup.yaml"], 2, "Config could not be loaded (see above)."),
        ],
        ids=["bench-train", "doctor"],
    )
    def test_a_command_that_calls_the_loader_plainly_shows_its_line_and_keeps_its_exit_code(
        self, in_tmp, argv, code, also
    ):
        from soup_cli.cli import app

        (in_tmp / "soup.yaml").write_text("base: m\ndata: [unclosed\n", encoding="utf-8")
        result = CliRunner().invoke(app, argv)
        output = _plain(result.output)
        assert result.exit_code == code, (result.output, repr(result.exception))
        assert "Config file soup.yaml is not valid YAML: expected ',' or ']'" in output, output
        assert "(line 3, column 1)" in output, output
        assert also in output, output
        assert "Traceback" not in output

    def test_training_overrides_still_reach_the_config(self, in_tmp):
        (in_tmp / "soup.yaml").write_text(_VALID, encoding="utf-8")
        assert load_config("soup.yaml", training_overrides={"epochs": 7}).training.epochs == 7


# ---------------------------------------------------------------------------
# load_config_from_string: text from the API / UI, ValueError only
# ---------------------------------------------------------------------------


class TestConfigText:
    def test_text_exactly_at_the_size_limit_loads(self):
        config = load_config_from_string(_config_of_size(MAX_CONFIG_BYTES))
        assert config.base == "meta-llama/Llama-3.1-8B-Instruct"

    def test_text_one_byte_over_the_size_limit_is_refused(self):
        with pytest.raises(YamlLimitError, match=rf"^Config {re.escape(_SIZE_MESSAGE)}$"):
            load_config_from_string(_config_of_size(MAX_CONFIG_BYTES + 1))

    def test_text_nested_to_the_limit_is_not_refused_for_nesting(self):
        with pytest.raises(ValueError, match="extra") as raised:
            load_config_from_string(_config_nested(MAX_YAML_NESTING))
        assert not isinstance(raised.value, YamlLimitError)

    def test_text_nested_one_level_past_the_limit_is_refused(self):
        with pytest.raises(YamlLimitError, match=rf"^Config {re.escape(_NESTING_MESSAGE)}$"):
            load_config_from_string(_config_nested(MAX_YAML_NESTING + 1))

    def test_text_nested_far_past_the_limit_is_a_value_error(self):
        """API and UI callers catch ValueError and nothing else."""
        with pytest.raises(ValueError, match=rf"^Config {re.escape(_NESTING_MESSAGE)}$"):
            load_config_from_string(_config_nested(20_000))

    def test_text_with_modest_alias_reuse_loads(self):
        config = load_config_from_string(_WITH_MODEST_ALIASES)
        assert config.output == config.base

    def test_text_over_the_node_allowance_is_refused(self):
        text = _config_with_alias_fan_out(9)
        with pytest.raises(YamlLimitError) as raised:
            load_config_from_string(text)
        assert str(raised.value) == _expansion_message("Config", text)

    def test_limits_are_checked_before_the_config_is_walked(self, monkeypatch):
        def _no_walk(*args, **kwargs):
            raise AssertionError("an over-limit config must not be walked")

        monkeypatch.setattr(loader, "find_unknown_config_keys", _no_walk)
        monkeypatch.setattr(loader, "_build_config", _no_walk)
        with pytest.raises(YamlLimitError, match="Config expands to more than"):
            load_config_from_string(_config_with_alias_fan_out(9))

    @pytest.mark.parametrize("text", ["", "\n", "# nothing but a comment\n"])
    def test_empty_text_keeps_its_message(self, text):
        with pytest.raises(ValueError, match=r"^Config is empty$"):
            load_config_from_string(text)

    def test_non_mapping_text_keeps_its_message(self):
        with pytest.raises(ValueError, match=r"^Config must be a YAML mapping, got list$"):
            load_config_from_string("- a\n- b\n")

    def test_syntax_error_is_still_a_yaml_error(self):
        with pytest.raises(yaml.YAMLError):
            load_config_from_string("base: [unclosed\ntask: }\n")

    def test_a_date_that_does_not_exist_is_still_a_value_error(self):
        with pytest.raises(ValueError, match="cannot build this value: day is out of range"):
            load_config_from_string(_VALID + "extra: 2026-02-30\n")


class TestCallerWithItsOwnSizeLimit:
    def test_text_api_takes_the_callers_limit(self):
        text = _config_of_size(4096)
        assert load_config_from_string(text, max_bytes=4096).task == "sft"
        with pytest.raises(YamlLimitError, match=r"^Config is larger than 4095 bytes; refusing$"):
            load_config_from_string(text, max_bytes=4095)

    def test_ship_accepts_two_mebibytes_and_train_refuses_them(self, in_tmp):
        from soup_cli.cli import app
        from soup_cli.commands import ship

        assert ship._MAX_CONFIG_BYTES == 4 * 1024 * 1024
        (in_tmp / "soup.yaml").write_bytes(_config_of_size(2 * 1024 * 1024).encode())
        config, _ = ship._parse_ship_config("soup.yaml")
        assert config.base == "meta-llama/Llama-3.1-8B-Instruct"
        result = CliRunner().invoke(app, ["train", "--config", "soup.yaml", "--dry-run"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert f"Config file soup.yaml {_SIZE_MESSAGE}" in _plain(result.output)

    def test_ship_refuses_a_config_over_its_own_limit(self, in_tmp):
        import click

        from soup_cli.commands import ship

        (in_tmp / "soup.yaml").write_bytes(b"#" + b"x" * ship._MAX_CONFIG_BYTES)
        with pytest.raises(click.exceptions.Exit):
            ship._parse_ship_config("soup.yaml")

    def test_recipes_verify_reads_no_further_than_the_limit(self, in_tmp, capsys):
        import click

        from soup_cli.commands.recipes import _configs_to_verify

        (in_tmp / "soup.yaml").write_bytes(_config_of_size(MAX_CONFIG_BYTES + 1).encode())
        with pytest.raises(click.exceptions.Exit):
            _configs_to_verify("soup.yaml", False)
        captured = capsys.readouterr()
        assert f"Config file soup.yaml {_SIZE_MESSAGE}" in _plain(captured.out + captured.err)

    def test_adapters_audit_reads_no_further_than_the_limit(self, in_tmp):
        from soup_cli.cli import app

        (in_tmp / "adapter").mkdir()
        (in_tmp / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (in_tmp / "soup.yaml").write_bytes(_config_of_size(MAX_CONFIG_BYTES + 1).encode())
        result = CliRunner().invoke(app, ["adapters", "audit", "adapter", "--config", "soup.yaml"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert f"Config file soup.yaml {_SIZE_MESSAGE}" in _plain(result.output)


# ---------------------------------------------------------------------------
# Commands that take --config print the refusal and exit 1
# ---------------------------------------------------------------------------


_COMMANDS = [
    ["train", "--config", "soup.yaml", "--dry-run"],
    ["cost", "--config", "soup.yaml"],
    ["profile", "--config", "soup.yaml"],
]

_OVER_LIMIT = {
    "size": (lambda: _config_of_size(MAX_CONFIG_BYTES + 1), lambda text: _SIZE_MESSAGE),
    "nesting": (lambda: _config_nested(MAX_YAML_NESTING + 1), lambda text: _NESTING_MESSAGE),
    "aliases": (
        lambda: _config_with_alias_fan_out(9),
        lambda text: _expansion_message("", text).strip(),
    ),
}


@pytest.mark.parametrize("argv", _COMMANDS, ids=[argv[0] for argv in _COMMANDS])
@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_command_prints_the_refusal_and_exits_1(in_tmp, argv, limit):
    from soup_cli.cli import app

    make_text, make_message = _OVER_LIMIT[limit]
    text = make_text()
    (in_tmp / "soup.yaml").write_bytes(text.encode())

    result = CliRunner().invoke(app, argv)

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert f"Config file soup.yaml {make_message(text)}" in output
    assert "Traceback" not in output


@pytest.mark.parametrize("argv", _COMMANDS, ids=[argv[0] for argv in _COMMANDS])
def test_command_still_loads_a_config_at_the_size_limit(in_tmp, argv):
    from soup_cli.cli import app

    (in_tmp / "soup.yaml").write_bytes(_config_of_size(MAX_CONFIG_BYTES).encode())
    result = CliRunner().invoke(app, argv)
    output = _plain(result.output)
    assert "is larger than" not in output, output
    assert "refusing" not in output, output


# ---------------------------------------------------------------------------
# Callers that take config text and catch ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", ["nesting", "aliases"])
def test_web_ui_reports_the_refusal_as_an_invalid_config(limit):
    pytest.importorskip("fastapi")

    from soup_cli.ui.app import create_app, get_auth_token

    make_text, make_message = _OVER_LIMIT[limit]
    text = make_text()
    with ui_client(create_app()) as client:
        response = client.post(
            "/api/config/validate",
            json={"yaml": text},
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"valid": False, "error": f"Config {make_message(text)}"}


@pytest.mark.parametrize("limit", ["nesting", "aliases"])
def test_mcp_reports_the_refusal_as_an_invalid_config(in_tmp, limit):
    from soup_cli.mcp_server import registry

    make_text, _ = _OVER_LIMIT[limit]
    (in_tmp / "soup.yaml").write_bytes(make_text().encode())
    with pytest.raises(registry.McpToolError, match=r"invalid config \(YamlLimitError\)"):
        registry._load_config_under_cwd("soup.yaml")


# ---------------------------------------------------------------------------
# A path with control characters reaches the terminal without them
# ---------------------------------------------------------------------------

#: CSI (U+009B) and DEL fit in a file name everywhere; ESC does not on Windows,
#: where the file system refuses every character below U+0020.
_CONTROLS = "\x9b\x7f" + ("" if os.name == "nt" else "\x1b")
_ODD_NAME = "run" + "31m".join(_CONTROLS) + "Z.yaml"


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--config", _ODD_NAME, "--find-lr"],
        ["train", "--config", _ODD_NAME, "--dry-run"],
        ["adapters", "audit", "adapter", "--config", _ODD_NAME],
        ["cost", "--config", _ODD_NAME],
    ],
    ids=["train-find-lr", "train", "adapters-audit", "cost"],
)
def test_a_refusal_that_quotes_an_odd_path_carries_no_control_character(in_tmp, argv):
    from soup_cli.cli import app

    (in_tmp / "adapter").mkdir()
    (in_tmp / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (in_tmp / _ODD_NAME).write_text(_config_nested(MAX_YAML_NESTING + 1), encoding="utf-8")

    result = CliRunner().invoke(app, argv)

    # Rich may colour the line; those codes are its own, so they are set aside first.
    shown = _ANSI_RE.sub("", result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _NESTING_MESSAGE in _plain(result.output), result.output
    assert not [char for char in shown if char in "\x1b\x7f\x9b"], repr(shown)


def test_a_missing_config_with_an_odd_path_is_named_without_control_characters(in_tmp):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["train", "--config", _ODD_NAME, "--dry-run"])
    shown = _ANSI_RE.sub("", result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Config not found: run" in _plain(result.output), result.output
    assert not [char for char in shown if char in "\x1b\x7f\x9b"], repr(shown)


def test_adapters_audit_shows_a_syntax_error_as_written(in_tmp):
    """The quoted line holds a bracket; it is text, not markup."""
    from soup_cli.cli import app

    (in_tmp / "adapter").mkdir()
    (in_tmp / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (in_tmp / "soup.yaml").write_text("base: m\ndata: [unclosed\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["adapters", "audit", "adapter", "--config", "soup.yaml"])
    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Could not parse soup.yaml: while parsing a flow sequence" in output, output
    assert "data: [unclosed" in output, output


@pytest.mark.parametrize(
    ("content", "line_start", "line_rest"),
    [
        ("base: m\ndata: [unclosed\n", "Could not parse run", "while parsing a flow sequence"),
        ("#" * (MAX_CONFIG_BYTES + 1), "Could not load run", "is larger than"),
    ],
    ids=["syntax-error", "over-size"],
)
def test_adapters_audit_shows_an_odd_config_path_and_its_message_without_control_characters(
    in_tmp, content, line_start, line_rest
):
    """Both audit refusal lines quote the path; the size refusal quotes it in its message too."""
    from soup_cli.cli import app

    (in_tmp / "adapter").mkdir()
    (in_tmp / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (in_tmp / _ODD_NAME).write_text(content, encoding="utf-8")

    result = CliRunner().invoke(app, ["adapters", "audit", "adapter", "--config", _ODD_NAME])

    shown = _ANSI_RE.sub("", result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert line_start in _plain(result.output), result.output
    assert line_rest in _plain(result.output), result.output
    assert not [char for char in shown if char in "\x1b\x7f\x9b"], repr(shown)
