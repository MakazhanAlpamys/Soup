"""Credential options are found by reading the command tree, whatever built it."""

from __future__ import annotations

import pytest
import typer

from soup_cli.utils import argv_redaction
from soup_cli.utils.argv_redaction import (
    NOT_SECRET_OPTIONS,
    REDACTED,
    SECRET_OPTIONS,
    _long_names,
    all_secret_option_strings,
    is_secret_name,
    iter_commands,
    mask_secret_args,
    resolve_command,
    secret_option_strings,
    subcommands,
)

# ---------------------------------------------------------------------------
# A command tree whose classes share nothing with the importable click package,
# as Typer builds it when it carries its own copy.
# ---------------------------------------------------------------------------


class _Option:
    def __init__(self, *opts, secondary=(), is_flag=False, count=False):
        self.opts = list(opts)
        self.secondary_opts = list(secondary)
        self.is_flag = is_flag
        self.count = count


class _Argument:
    def __init__(self, name):
        self.opts = [name]
        self.secondary_opts = []


class _Command:
    def __init__(self, name, params=()):
        self.name = name
        self.params = list(params)


class _Group(_Command):
    def __init__(self, name, commands, params=()):
        super().__init__(name, params)
        self.commands = {command.name: command for command in commands}


@pytest.fixture
def tree():
    serve = _Command(
        "serve",
        [
            _Option("--model", "-m"),
            _Option("--api-key"),
            _Option("--tool-auth-token", "-t"),
            _Option("--use-auth-token", is_flag=True),
            _Option("--token-verbosity", "-v", count=True),
            _Option("--max-tokens"),
            _Option("--cache", secondary=["--no-key-cache"], is_flag=True),
            _Argument("token"),
        ],
    )
    generate = _Command("generate", [_Option("--api-key", "-k"), _Option("--rows")])
    data = _Group("data", [generate], [_Option("--data-secret")])
    return _Group("soup", [serve, data])


def test_every_command_of_a_foreign_tree_is_visited(tree):
    assert [command.name for command in iter_commands(tree)] == [
        "soup",
        "serve",
        "data",
        "generate",
    ]


def test_credential_options_of_a_foreign_tree_are_found(tree):
    assert all_secret_option_strings(tree) == {
        "--api-key",
        "--tool-auth-token",
        "-t",
        "-k",
        "--data-secret",
    }


def test_only_value_taking_options_of_a_command_count(tree):
    """A switch, a counter, a listed non-credential name and an argument do not."""
    serve = tree.commands["serve"]
    assert secret_option_strings(serve) == {"--api-key", "--tool-auth-token", "-t"}


def test_group_and_leaf_are_told_apart_by_their_subcommands(tree):
    data = tree.commands["data"]
    assert subcommands(tree) is tree.commands
    assert subcommands(data) is data.commands
    assert subcommands(tree.commands["serve"]) is None
    assert subcommands(_Group("empty", [])) == {}


@pytest.mark.parametrize(
    ("args", "name"),
    [
        (["serve", "--api-key", "VALUE"], "serve"),
        (["--verbose", "data", "generate", "--rows", "3"], "generate"),
        (["data"], "data"),
        (["data", "no-such-command"], "data"),
        (["no-such-command"], "soup"),
        ([], "soup"),
    ],
)
def test_arguments_resolve_down_a_foreign_tree(tree, args, name):
    assert resolve_command(tree, args).name == name


def test_values_are_masked_with_the_options_of_a_foreign_tree(tree):
    args = ["--model", "m", "--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO", "-t", "V3"]
    assert mask_secret_args(args, all_secret_option_strings(tree)) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
        "-t",
        REDACTED,
    )


# ---------------------------------------------------------------------------
# The real command tree
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    import soup_cli.cli as cli_module

    monkeypatch.setattr(cli_module, "_ALL_SECRET_OPTION_STRINGS", None)
    return cli_module


def test_the_real_tree_is_walked_and_has_credential_options(cli):
    root = typer.main.get_command(cli.app)
    assert sum(1 for _ in iter_commands(root)) > 200
    assert {"--api-key", "--auth-token", "--tool-auth-token"} <= all_secret_option_strings(root)


def test_the_real_tree_masks_by_option_name(cli):
    args = ["--model", "m", "--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO"]
    assert cli._mask_audit_args("serve", args) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
    )


def test_a_group_is_masked_with_every_option_of_the_app(cli):
    """`-t` is a credential option of some command, so a bare group masks it."""
    assert cli._mask_audit_args("data", ["-t", "VALUE"]) == ("-t", REDACTED)


# ---------------------------------------------------------------------------
# The names typed are masked by the same rule, whatever the tree yields
# ---------------------------------------------------------------------------


def test_an_empty_option_set_masks_by_the_names_on_the_command_line(cli, monkeypatch):
    monkeypatch.setattr(cli, "_ALL_SECRET_OPTION_STRINGS", frozenset())
    args = [
        "--model",
        "m",
        "--api-key",
        "VALUE-ONE",
        "--tool-auth-token=VALUE-TWO",
        "--max-tokens",
        "64",
    ]
    assert cli._mask_audit_args("serve", args) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
        "--max-tokens",
        "64",
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--api-key", "V"], ("--api-key", REDACTED)),
        (["--api-key=V"], (f"--api-key={REDACTED}",)),
        (["--api-key="], (f"--api-key={REDACTED}",)),
        (["--hf-token", "V", "next"], ("--hf-token", REDACTED, "next")),
        (["--password=V"], (f"--password={REDACTED}",)),
        (["--passwd", "V"], ("--passwd", REDACTED)),
        (["--client-secret", "V"], ("--client-secret", REDACTED)),
        (["--credential-file", "V"], ("--credential-file", REDACTED)),
        (["--API-KEY", "V"], ("--API-KEY", REDACTED)),
        (["--api-key", "-S1"], ("--api-key", REDACTED)),
        (["--api-key", "-"], ("--api-key", REDACTED)),
        (
            ["--api-key", "V1", "--auth-token", "V2"],
            ("--api-key", REDACTED, "--auth-token", REDACTED),
        ),
        (["--api-key"], ("--api-key",)),
        (["--slack-url", "https://example.invalid/hook/V"], ("--slack-url", REDACTED)),
        (["--discord-url=https://example.invalid/hook/V"], (f"--discord-url={REDACTED}",)),
    ],
)
def test_names_that_look_like_a_credential_have_their_value_masked(args, expected):
    assert mask_secret_args(args, frozenset()) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "--model", "m", "--port", "8000"],
        ["--max-tokens", "64"],
        ["--max-new-tokens=64"],
        ["--min-tokens", "5"],
        ["--base-url", "https://example.invalid/v1"],
        ["--tokenizer", "org/model"],
        ["--public-key", "key.pem"],
        ["--special-token", "<|im_end|>"],
        ["-t", "V"],
        ["--", "token"],
        ["token", "key", "secret"],
        [],
    ],
)
def test_other_arguments_are_kept_as_typed(args):
    assert mask_secret_args(args, frozenset()) == tuple(args)


def test_names_and_tree_options_are_masked_together():
    """A declared short alias and an undeclared long name in one command line."""
    args = ["-t", "V1", "--api-key", "V2", "-x", "kept", "--port", "1"]
    assert mask_secret_args(args, frozenset({"--token", "-t"})) == (
        "-t",
        REDACTED,
        "--api-key",
        REDACTED,
        "-x",
        "kept",
        "--port",
        "1",
    )


def test_a_single_dash_option_of_several_letters_is_masked_as_declared():
    """Not a long name and not a one-letter alias: only the tree can name it."""
    assert mask_secret_args(["-tok", "V", "x"], frozenset({"-tok"})) == ("-tok", REDACTED, "x")


# ---------------------------------------------------------------------------
# A tree that can only partly be read
# ---------------------------------------------------------------------------


class _Unreadable:
    """A parameter that exposes none of the attributes the walk reads."""


@pytest.fixture
def partly_readable(cli, monkeypatch):
    """`push` declares its credential option; nothing can be read from `serve`."""
    push = _Command("push", [_Option("--token", "-t"), _Option("--repo")])
    serve = _Command("serve", [_Unreadable(), _Unreadable()])
    root = _Group("soup", [push, serve])
    monkeypatch.setattr(typer.main, "get_command", lambda app: root)
    assert all_secret_option_strings(root) == {"--token", "-t"}
    return cli


def test_an_option_the_tree_did_not_yield_is_still_masked(partly_readable):
    args = ["--model", "m", "--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO", "--port", "1"]
    assert partly_readable._mask_audit_args("serve", args) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
        "--port",
        "1",
    )


def test_a_short_alias_the_tree_did_yield_is_masked(partly_readable):
    assert partly_readable._mask_audit_args("push", ["--repo", "u/m", "-t", "VALUE"]) == (
        "--repo",
        "u/m",
        "-t",
        REDACTED,
    )


# ---------------------------------------------------------------------------
# The two lists and the real options agree
# ---------------------------------------------------------------------------


def test_webhook_urls_are_credential_options_of_the_real_tree(cli):
    assert SECRET_OPTIONS == {"--slack-url", "--discord-url"}
    assert SECRET_OPTIONS <= all_secret_option_strings(typer.main.get_command(cli.app))


def test_a_webhook_url_is_masked_for_a_real_command(cli):
    args = ["--slack-url", "https://example.invalid/hook/VALUE", "--discord-url=VALUE-TWO"]
    assert cli._mask_audit_args("train", args) == (
        "--slack-url",
        REDACTED,
        f"--discord-url={REDACTED}",
    )


def test_a_length_threshold_is_not_a_credential_option(cli):
    assert "--min-tokens" in NOT_SECRET_OPTIONS
    assert "--min-tokens" not in all_secret_option_strings(typer.main.get_command(cli.app))
    assert cli._mask_audit_args("data", ["clean", "--min-tokens", "5"]) == (
        "clean",
        "--min-tokens",
        "5",
    )


def test_no_real_switch_has_a_name_the_rule_would_mask(cli):
    """A switch takes no value, so masking by its name would hide the next argument."""
    unlisted = set()
    for command in iter_commands(typer.main.get_command(cli.app)):
        for param in getattr(command, "params", None) or ():
            if getattr(param, "is_flag", False) or getattr(param, "count", False):
                unlisted.update(name for name in _long_names(param) if is_secret_name(name))
    assert unlisted == set(), "list each in NOT_SECRET_OPTIONS"


# ---------------------------------------------------------------------------
# A long option is never taken as the value of the option before it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--keep-key-cache", "--api-key", "S1"], ("--keep-key-cache", "--api-key", REDACTED)),
        (["--api-key", "--auth-token", "S1"], ("--api-key", "--auth-token", REDACTED)),
        (["--api-key", "--auth-token=S1"], ("--api-key", f"--auth-token={REDACTED}")),
        (["--hf-token", "--abc123"], ("--hf-token", REDACTED)),
        (["--hf-token", "--abc123", "next"], ("--hf-token", REDACTED, "next")),
        (["--hf-token=--abc123"], (f"--hf-token={REDACTED}",)),
        (["--api-key", "--model", "m"], ("--api-key", REDACTED, "m")),
        (["--use-token", "--", "S1"], ("--use-token", REDACTED, "S1")),
        (["--api-key", "--api-key", "--api-key", "S1"], ("--api-key",) * 3 + (REDACTED,)),
        # A single dash is never a long option, whatever its letters spell.
        (["--hf-token", "-my-key"], ("--hf-token", REDACTED)),
        (["--hf-token", "-my-key", "next"], ("--hf-token", REDACTED, "next")),
        (["--hf-token", "-my-key=S1"], ("--hf-token", REDACTED)),
    ],
)
def test_only_a_credential_named_option_is_read_as_an_option_without_a_tree(args, expected):
    """Anything else that follows a credential option is its value, dashes or not."""
    assert mask_secret_args(args, frozenset()) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--token", "--repo", "u/m"], ("--token", "--repo", "u/m")),
        (["--token", "--repo=u/m"], ("--token", "--repo=u/m")),
        (["--token", "--abc123", "--repo", "u/m"], ("--token", REDACTED, "--repo", "u/m")),
        (["-t", "--abc123"], ("-t", REDACTED)),
        (["-t", "--token", "S1"], ("-t", "--token", REDACTED)),
        (["-xt", "--repo", "u/m"], ("-xt", "--repo", "u/m")),
        (["-t", "-S1"], ("-t", REDACTED)),
    ],
)
def test_a_long_option_the_tree_knows_is_read_as_an_option(args, expected):
    known = frozenset({"--token", "--repo"})
    assert mask_secret_args(args, frozenset({"--token", "-t"}), known) == expected


def test_a_long_option_the_caller_lists_as_secret_is_read_as_an_option():
    """Listed in `secret_opts`, so an option, though its name says nothing about a credential."""
    secret_opts = frozenset({"--webhook", "--sink"})
    known = frozenset({"--repo"})
    assert mask_secret_args(["--webhook", "--sink", "S1"], secret_opts, known) == (
        "--webhook",
        "--sink",
        REDACTED,
    )
    assert mask_secret_args(["--webhook", "--sink=S1"], secret_opts, known) == (
        "--webhook",
        f"--sink={REDACTED}",
    )


# ---------------------------------------------------------------------------
# A group of a foreign tree is masked with the short aliases of its commands
# ---------------------------------------------------------------------------


def test_a_foreign_group_masks_the_short_alias_of_its_command(cli, monkeypatch):
    push = _Command("push", [_Option("--token", "-t"), _Option("--repo")])
    root = _Group("soup", [_Group("hub", [push])])
    monkeypatch.setattr(typer.main, "get_command", lambda app: root)
    assert cli._mask_audit_args("hub", ["-t", "VALUE", "--repo", "u/m"]) == (
        "-t",
        REDACTED,
        "--repo",
        "u/m",
    )


def test_the_real_tree_tells_a_known_option_from_a_value_that_starts_with_dashes(cli):
    assert cli._mask_audit_args("serve", ["--api-key", "--model", "m"]) == (
        "--api-key",
        "--model",
        "m",
    )
    assert cli._mask_audit_args("serve", ["--api-key", "--abc123", "--model", "m"]) == (
        "--api-key",
        REDACTED,
        "--model",
        "m",
    )


def test_every_long_option_of_a_tree_is_collected(tree):
    assert argv_redaction.all_long_option_strings(tree) == {
        "--model",
        "--api-key",
        "--tool-auth-token",
        "--use-auth-token",
        "--token-verbosity",
        "--max-tokens",
        "--cache",
        "--no-key-cache",
        "--rows",
        "--data-secret",
    }


# ---------------------------------------------------------------------------
# A command that cannot be resolved is masked with every option of the app
# ---------------------------------------------------------------------------


def test_a_failed_resolution_still_masks_short_aliases_and_long_names(cli, monkeypatch):
    def _fails(root, args):
        raise RuntimeError("the command could not be resolved")

    monkeypatch.setattr(argv_redaction, "resolve_command", _fails)
    assert cli._mask_audit_args("push", ["-t", "S3CRET", "--token", "V2", "--repo", "u/m"]) == (
        "-t",
        REDACTED,
        "--token",
        REDACTED,
        "--repo",
        "u/m",
    )
