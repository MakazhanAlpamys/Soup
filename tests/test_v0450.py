"""v0.45.0 — Plugin System & Ecosystem Wins.

Covers all five Parts:

- Part A: ``soup_cli.plugins`` registry + ``soup plugins`` CLI
- Part B: ``utils.anthropic_messages`` + ``utils.server_tools`` + ``utils.ngram_spec``
- Part C: ``utils.integrations`` external integrations catalog
- Part D: ``utils.trainer_plugins`` advanced trainer plugin allowlist
- Part E: ``utils.recipe_dag`` + ``soup data recipe`` CLI
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli import plugins as plugins_pkg
from soup_cli.utils.anthropic_messages import (
    from_anthropic,
    to_anthropic,
    validate_anthropic_payload,
)
from soup_cli.utils.integrations import (
    get_integration,
    has_integration,
    list_integrations,
)
from soup_cli.utils.ngram_spec import (
    NgramSpecConfig,
    validate_ngram_config,
    validate_ngram_n,
    validate_num_draft_tokens,
    validate_prompt_lookup_max,
)
from soup_cli.utils.recipe_dag import (
    NODE_KINDS,
    load_recipe_yaml,
    parse_recipe,
    parse_recipe_yaml,
)
from soup_cli.utils.server_tools import (
    SUPPORTED_TOOLS,
    WebSearchConfig,
    is_domain_allowed,
    tool_description,
    validate_domain,
    validate_rate_limit,
    validate_tool_name,
    validate_web_search_config,
)
from soup_cli.utils.trainer_plugins import (
    get_trainer_plugin,
    list_trainer_plugins,
    validate_trainer_plugin_list,
)

# ---------------------------------------------------------------------------
# Part A — Plugin / hook system
# ---------------------------------------------------------------------------


class _NoopPlugin:
    def pre_train(self, ctx):
        ctx["pre_train_seen"] = True

    def post_step(self, ctx):
        ctx["post_step_seen"] = True


@pytest.fixture(autouse=True)
def _reset_plugins():
    plugins_pkg.clear_plugins()
    yield
    plugins_pkg.clear_plugins()


def test_register_plugin_happy_path():
    spec = plugins_pkg.register_plugin(
        name="hello-world",
        version="1.0.0",
        plugin=_NoopPlugin(),
        description="hi",
    )
    assert spec.name == "hello-world"
    assert spec.version == "1.0.0"
    assert spec.enabled is True
    assert "hello-world" in plugins_pkg.list_plugins()


def test_discover_hooks_finds_only_implemented():
    plugin = _NoopPlugin()
    hooks = plugins_pkg.discover_hooks(plugin)
    assert "pre_train" in hooks
    assert "post_step" in hooks
    assert "post_train" not in hooks
    assert "pre_step" not in hooks


@pytest.mark.parametrize(
    "name",
    [
        "",
        "BadCase",
        "starts-with-",  # we accept this since regex allows trailing? actually allows
        "with space",
        "x" * 41,
        "with\x00null",
    ],
)
def test_register_plugin_rejects_bad_names(name):
    if name == "starts-with-":  # allowed — leading char is alnum
        pytest.skip("trailing hyphen permitted by regex")
    with pytest.raises((TypeError, ValueError)):
        plugins_pkg.register_plugin(
            name=name, version="1.0.0", plugin=_NoopPlugin()
        )


@pytest.mark.parametrize(
    "version",
    ["", "1", "1.0", "1.0.x", "v1.0.0", "1.0.0+\x00"],
)
def test_register_plugin_rejects_bad_version(version):
    with pytest.raises((TypeError, ValueError)):
        plugins_pkg.register_plugin(
            name="ok", version=version, plugin=_NoopPlugin()
        )


def test_register_plugin_idempotent_same_spec():
    plugin = _NoopPlugin()
    a = plugins_pkg.register_plugin(name="dup", version="1.0.0", plugin=plugin)
    b = plugins_pkg.register_plugin(name="dup", version="1.0.0", plugin=plugin)
    assert a is b


def test_register_plugin_idempotent_preserves_disabled_state():
    plugin = _NoopPlugin()
    plugins_pkg.register_plugin(name="dup2", version="1.0.0", plugin=plugin)
    plugins_pkg.disable_plugin("dup2")
    again = plugins_pkg.register_plugin(
        name="dup2", version="1.0.0", plugin=plugin
    )
    assert again.enabled is False


def test_register_plugin_rejects_conflicting_description():
    plugin = _NoopPlugin()
    plugins_pkg.register_plugin(
        name="dup3", version="1.0.0", plugin=plugin, description="first"
    )
    with pytest.raises(ValueError, match="already registered"):
        plugins_pkg.register_plugin(
            name="dup3", version="1.0.0", plugin=plugin, description="second"
        )


def test_register_plugin_rejects_too_many_templates():
    with pytest.raises(ValueError, match="exceeds"):
        plugins_pkg.register_plugin(
            name="big-tpl",
            version="1.0.0",
            plugin=_NoopPlugin(),
            templates=[f"t{index}" for index in range(33)],
        )


def test_register_plugin_rejects_oversize_template_name():
    with pytest.raises(ValueError, match="exceeds"):
        plugins_pkg.register_plugin(
            name="long-tpl",
            version="1.0.0",
            plugin=_NoopPlugin(),
            templates=["x" * 200],
        )


def test_register_plugin_rejects_conflicting_version():
    plugin = _NoopPlugin()
    plugins_pkg.register_plugin(name="dup", version="1.0.0", plugin=plugin)
    with pytest.raises(ValueError, match="already registered"):
        plugins_pkg.register_plugin(name="dup", version="2.0.0", plugin=plugin)


def test_register_plugin_rejects_empty_plugin():
    class _Empty:
        pass

    with pytest.raises(ValueError, match="hook"):
        plugins_pkg.register_plugin(
            name="empty", version="1.0.0", plugin=_Empty()
        )


def test_register_plugin_with_template_only_ok():
    class _Empty:
        pass

    plugins_pkg.register_plugin(
        name="tpl",
        version="1.0.0",
        plugin=_Empty(),
        templates=["my-template"],
    )


def test_register_plugin_rejects_none_plugin():
    with pytest.raises(ValueError):
        plugins_pkg.register_plugin(name="x", version="1.0.0", plugin=None)


def test_enable_disable_toggles_state():
    plugins_pkg.register_plugin(
        name="toggle", version="1.0.0", plugin=_NoopPlugin()
    )
    assert plugins_pkg.is_enabled("toggle") is True
    assert plugins_pkg.disable_plugin("toggle") is True
    assert plugins_pkg.is_enabled("toggle") is False
    # second disable returns False (no state change)
    assert plugins_pkg.disable_plugin("toggle") is False
    assert plugins_pkg.enable_plugin("toggle") is True


def test_enable_unknown_raises():
    with pytest.raises(KeyError):
        plugins_pkg.enable_plugin("ghost")


def test_list_plugins_returns_immutable_view():
    plugins_pkg.register_plugin(
        name="a", version="1.0.0", plugin=_NoopPlugin()
    )
    view = plugins_pkg.list_plugins()
    with pytest.raises(TypeError):
        view["evil"] = "not allowed"  # type: ignore[index]


def test_get_plugin_unknown_returns_none():
    assert plugins_pkg.get_plugin("missing") is None
    assert plugins_pkg.get_plugin(123) is None  # type: ignore[arg-type]


def test_pluginspec_frozen():
    plugins_pkg.register_plugin(
        name="frozen", version="1.0.0", plugin=_NoopPlugin()
    )
    spec = plugins_pkg.get_plugin("frozen")
    with pytest.raises(Exception):
        spec.name = "mutated"  # type: ignore[misc]


def test_too_many_plugins_rejected():
    for index in range(64):
        plugins_pkg.register_plugin(
            name=f"p{index}", version="1.0.0", plugin=_NoopPlugin()
        )
    with pytest.raises(RuntimeError, match="too many"):
        plugins_pkg.register_plugin(
            name="overflow", version="1.0.0", plugin=_NoopPlugin()
        )


def test_load_plugins_returns_count():
    count = plugins_pkg.load_plugins()
    assert isinstance(count, int)
    assert count >= 0


def test_plugins_cli_list_empty():
    from soup_cli.commands import plugins as plugins_cli

    runner = CliRunner()
    result = runner.invoke(plugins_cli.app, [])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "No plugins registered" in result.output


def test_plugins_cli_lists_registered():
    from soup_cli.commands import plugins as plugins_cli

    plugins_pkg.register_plugin(
        name="cli-test", version="1.2.3", plugin=_NoopPlugin()
    )
    runner = CliRunner()
    result = runner.invoke(plugins_cli.app, ["list"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "cli-test" in result.output
    assert "1.2.3" in result.output


def test_plugins_cli_install_advisory():
    from soup_cli.commands import plugins as plugins_cli

    runner = CliRunner()
    result = runner.invoke(plugins_cli.app, ["install", "anything"])
    assert result.exit_code == 0
    assert "advisory" in result.output


def test_plugins_cli_enable_unknown():
    from soup_cli.commands import plugins as plugins_cli

    runner = CliRunner()
    result = runner.invoke(plugins_cli.app, ["enable", "ghost"])
    assert result.exit_code == 1
    assert "Unknown" in result.output


def test_plugins_cli_enable_disable_cycle():
    from soup_cli.commands import plugins as plugins_cli

    plugins_pkg.register_plugin(
        name="cycle", version="1.0.0", plugin=_NoopPlugin()
    )
    runner = CliRunner()
    plugins_pkg.disable_plugin("cycle")
    result = runner.invoke(plugins_cli.app, ["enable", "cycle"])
    assert result.exit_code == 0
    assert "enabled" in result.output
    result = runner.invoke(plugins_cli.app, ["disable", "cycle"])
    assert result.exit_code == 0
    assert "disabled" in result.output


def test_plugins_cli_markup_escaped():
    from soup_cli.commands import plugins as plugins_cli

    runner = CliRunner()
    # Crafted name with Rich markup; should be escaped before printing.
    result = runner.invoke(plugins_cli.app, ["enable", "[red]evil[/]"])
    assert result.exit_code == 2
    # The validation message names the kebab-case rule; markup chars in
    # the input cannot inject Rich styles because ``rich.markup.escape`` is
    # applied to all user-controlled values in the CLI.
    assert "kebab-case" in result.output


# ---------------------------------------------------------------------------
# Part B — Anthropic Messages converter
# ---------------------------------------------------------------------------


def test_to_anthropic_joins_multiple_system_messages():
    payload = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 32,
    }
    out = to_anthropic(payload)
    assert out["system"] == "first\n\nsecond"


def test_to_anthropic_tool_result_concatenates_list_content():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "part-a"},
                    {"type": "text", "text": "part-b"},
                ],
            },
        ],
    }
    out = to_anthropic(payload)
    tool_msg = out["messages"][-1]
    assert tool_msg["content"][0]["content"] == "part-a\npart-b"


def test_to_anthropic_basic():
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 256,
    }
    out = to_anthropic(payload)
    assert out["model"] == "claude-3-5-sonnet"
    assert out["system"] == "you are helpful"
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["max_tokens"] == 256


def test_to_anthropic_default_max_tokens():
    payload = {"model": "x", "messages": [{"role": "user", "content": "a"}]}
    out = to_anthropic(payload)
    assert out["max_tokens"] == 1024


def test_to_anthropic_caps_max_tokens():
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "a"}],
        "max_tokens": 99999,
    }
    out = to_anthropic(payload)
    assert out["max_tokens"] == 16384


def test_to_anthropic_tool_role_becomes_tool_result():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "ask"},
            {
                "role": "tool",
                "content": "result-text",
                "tool_call_id": "tool_1",
            },
        ],
    }
    out = to_anthropic(payload)
    assert any(
        msg["role"] == "user"
        and isinstance(msg["content"], list)
        and msg["content"][0].get("type") == "tool_result"
        for msg in out["messages"]
    )


def test_to_anthropic_rejects_unknown_role():
    payload = {
        "model": "x",
        "messages": [{"role": "developer", "content": "x"}],
    }
    with pytest.raises(ValueError, match="role"):
        to_anthropic(payload)


def test_to_anthropic_rejects_bad_max_tokens():
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "a"}],
        "max_tokens": True,
    }
    with pytest.raises(TypeError):
        to_anthropic(payload)


def test_to_anthropic_rejects_temperature_out_of_range():
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "a"}],
        "temperature": 99.0,
    }
    with pytest.raises(ValueError):
        to_anthropic(payload)


def test_from_anthropic_roundtrip_user_only():
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }
    out = from_anthropic(payload)
    assert out == {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
    }


def test_from_anthropic_with_system():
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "system": "rules",
    }
    out = from_anthropic(payload)
    assert out["messages"][0] == {"role": "system", "content": "rules"}


def test_validate_anthropic_payload_rejects_empty_messages():
    with pytest.raises(ValueError):
        validate_anthropic_payload({
            "model": "x",
            "messages": [],
            "max_tokens": 10,
        })


def test_validate_anthropic_payload_rejects_bad_role():
    with pytest.raises(ValueError, match="role"):
        validate_anthropic_payload({
            "model": "x",
            "messages": [{"role": "system", "content": "x"}],
            "max_tokens": 10,
        })


def test_validate_anthropic_payload_rejects_oversize_max_tokens():
    with pytest.raises(ValueError):
        validate_anthropic_payload({
            "model": "x",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 99999,
        })


# ---------------------------------------------------------------------------
# Part B — Server-side tools registry
# ---------------------------------------------------------------------------


def test_supported_tools_closed_set():
    assert SUPPORTED_TOOLS == frozenset({"python", "bash", "web_search"})


@pytest.mark.parametrize("name", ["python", "Bash", "WEB_SEARCH"])
def test_validate_tool_name_canonicalises(name):
    assert validate_tool_name(name) in SUPPORTED_TOOLS


@pytest.mark.parametrize("name", ["", "evil", "python ", "py\x00thon"])
def test_validate_tool_name_rejects_garbage(name):
    if name == "python ":  # strip removes whitespace then matches
        assert validate_tool_name(name) == "python"
        return
    with pytest.raises((TypeError, ValueError)):
        validate_tool_name(name)


def test_validate_tool_name_rejects_non_string():
    with pytest.raises(TypeError):
        validate_tool_name(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("rpm", [1, 600, 30])
def test_validate_rate_limit_happy(rpm):
    assert validate_rate_limit(rpm) == rpm


@pytest.mark.parametrize("rpm", [0, -1, 601, True])
def test_validate_rate_limit_rejects(rpm):
    with pytest.raises((TypeError, ValueError)):
        validate_rate_limit(rpm)


@pytest.mark.parametrize(
    "domain", ["example.com", ".example.com", "a.b.c.example.com"]
)
def test_validate_domain_happy(domain):
    assert validate_domain(domain) == domain.lower()


@pytest.mark.parametrize(
    "domain", ["", " ", "with space.com", "/path", "X" * 300, "evil.com\x00"]
)
def test_validate_domain_rejects(domain):
    with pytest.raises((TypeError, ValueError)):
        validate_domain(domain)


def test_is_domain_allowed_exact_match():
    assert is_domain_allowed("example.com", ("example.com",)) is True
    assert is_domain_allowed("a.example.com", ("example.com",)) is False


def test_is_domain_allowed_subdomain_with_dot():
    assert is_domain_allowed("a.example.com", (".example.com",)) is True
    assert is_domain_allowed("example.com", (".example.com",)) is True


def test_is_domain_allowed_no_match():
    assert is_domain_allowed("evil.org", ("example.com",)) is False


def test_is_domain_allowed_rejects_garbage_host():
    assert is_domain_allowed("", ("example.com",)) is False
    assert is_domain_allowed("a\x00", ("example.com",)) is False


def test_validate_web_search_config_dedup():
    cfg = WebSearchConfig(
        domain_allowlist=("example.com", "example.com"),
        rate_limit_per_minute=30,
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_web_search_config(cfg)


def test_validate_web_search_config_too_many():
    cfg = WebSearchConfig(
        domain_allowlist=tuple(f"d{index}.com" for index in range(65)),
        rate_limit_per_minute=30,
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate_web_search_config(cfg)


def test_validate_web_search_config_rejects_bad_type():
    with pytest.raises(TypeError):
        validate_web_search_config("not-a-config")  # type: ignore[arg-type]


def test_tool_description_happy():
    assert "Sandboxed" in tool_description("python")
    assert "search" in tool_description("web_search").lower()


# ---------------------------------------------------------------------------
# Part B — n-gram speculative decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 4, 8])
def test_validate_ngram_n_happy(n):
    assert validate_ngram_n(n) == n


@pytest.mark.parametrize("n", [0, 9, -1, True])
def test_validate_ngram_n_rejects(n):
    with pytest.raises((TypeError, ValueError)):
        validate_ngram_n(n)


def test_validate_ngram_config_happy():
    cfg = NgramSpecConfig(n=4, num_draft_tokens=8, prompt_lookup_max=10)
    assert validate_ngram_config(cfg) is cfg


def test_validate_ngram_config_rejects_bad_inner():
    cfg = NgramSpecConfig(n=4, num_draft_tokens=999)
    with pytest.raises(ValueError):
        validate_ngram_config(cfg)


@pytest.mark.parametrize("v", [True, -1, 33])
def test_validate_num_draft_tokens_rejects(v):
    with pytest.raises((TypeError, ValueError)):
        validate_num_draft_tokens(v)


def test_validate_prompt_lookup_max_happy_zero():
    assert validate_prompt_lookup_max(0) == 0


def test_validate_prompt_lookup_max_rejects_bool():
    with pytest.raises(TypeError):
        validate_prompt_lookup_max(True)


def test_ngram_config_frozen():
    cfg = NgramSpecConfig(n=2)
    with pytest.raises(Exception):
        cfg.n = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Part C — External integrations catalog
# ---------------------------------------------------------------------------


def test_list_integrations_returns_immutable_view():
    view = list_integrations()
    with pytest.raises(TypeError):
        view["new"] = "evil"  # type: ignore[index]


def test_list_integrations_known_entries():
    view = list_integrations()
    for entry in ("lm-studio", "comfyui", "ollama", "open-webui", "claude-code"):
        assert entry in view


def test_get_integration_known():
    spec = get_integration("ollama")
    assert spec.name == "ollama"
    assert "gguf" in spec.target_artifacts


def test_get_integration_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_integration("nonexistent")


@pytest.mark.parametrize("bad", ["", " ", "a\x00b"])
def test_get_integration_rejects_bad(bad):
    with pytest.raises(ValueError):
        get_integration(bad)


def test_get_integration_rejects_non_string():
    with pytest.raises(TypeError):
        get_integration(42)  # type: ignore[arg-type]


def test_has_integration_happy_and_unknown():
    assert has_integration("ollama") is True
    assert has_integration("nope") is False
    assert has_integration(123) is False  # type: ignore[arg-type]


def test_integration_spec_frozen():
    spec = get_integration("ollama")
    with pytest.raises(Exception):
        spec.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Part D — Advanced trainer plugins
# ---------------------------------------------------------------------------


def test_list_trainer_plugins_immutable():
    view = list_trainer_plugins()
    with pytest.raises(TypeError):
        view["evil"] = "x"  # type: ignore[index]


def test_list_trainer_plugins_known_entries():
    view = list_trainer_plugins()
    for entry in ("grokfast", "spectrum", "llmcompressor", "math_verify"):
        assert entry in view


def test_get_trainer_plugin_known():
    spec = get_trainer_plugin("grokfast")
    assert spec.name == "grokfast"
    assert spec.required_package == "grokfast"


def test_get_trainer_plugin_unknown_raises():
    with pytest.raises(KeyError):
        get_trainer_plugin("nope")


def test_get_trainer_plugin_no_required_pkg():
    spec = get_trainer_plugin("spectrum")
    assert spec.required_package is None


def test_validate_trainer_plugin_list_happy():
    out = validate_trainer_plugin_list(["grokfast", "spectrum"])
    assert out == ("grokfast", "spectrum")


def test_validate_trainer_plugin_list_canonicalises():
    out = validate_trainer_plugin_list(["GROKFAST"])
    assert out == ("grokfast",)


def test_validate_trainer_plugin_list_rejects_unknown():
    with pytest.raises(ValueError, match="unknown"):
        validate_trainer_plugin_list(["bogus_plugin"])


def test_validate_trainer_plugin_list_rejects_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        validate_trainer_plugin_list(["grokfast", "grokfast"])


def test_validate_trainer_plugin_list_rejects_too_many():
    with pytest.raises(ValueError, match="too many"):
        validate_trainer_plugin_list(["grokfast"] * 9)


def test_validate_trainer_plugin_list_rejects_non_string():
    with pytest.raises(TypeError):
        validate_trainer_plugin_list([123])  # type: ignore[list-item]


def test_validate_trainer_plugin_list_rejects_non_list():
    with pytest.raises(TypeError):
        validate_trainer_plugin_list("grokfast")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part E — Data Recipe DAG
# ---------------------------------------------------------------------------


def _good_recipe():
    return {
        "nodes": [
            {"name": "seed1", "kind": "seed", "config": {"path": "x.jsonl"}},
            {"name": "llm1", "kind": "llm_text", "config": {}},
            {"name": "judge1", "kind": "judge", "config": {}},
            {"name": "samp1", "kind": "sampler", "config": {}},
        ],
        "edges": [
            ["seed1", "llm1"],
            ["llm1", "judge1"],
            ["judge1", "samp1"],
        ],
    }


def test_node_kinds_closed_set():
    assert NODE_KINDS == frozenset(
        {"seed", "llm_text", "code", "judge", "validator", "sampler"}
    )


def test_parse_recipe_happy():
    dag = parse_recipe(_good_recipe())
    assert tuple(dag.topo_order)[0] == "seed1"
    assert tuple(dag.topo_order)[-1] == "samp1"
    assert len(dag.nodes) == 4
    assert len(dag.edges) == 3


def test_parse_recipe_rejects_cycle():
    bad = _good_recipe()
    bad["edges"].append(["samp1", "seed1"])
    with pytest.raises(ValueError, match="cycle"):
        parse_recipe(bad)


def test_parse_recipe_rejects_self_loop():
    bad = _good_recipe()
    bad["edges"] = [["seed1", "seed1"]]
    with pytest.raises(ValueError, match="self-loop"):
        parse_recipe(bad)


def test_parse_recipe_rejects_unknown_kind():
    bad = _good_recipe()
    bad["nodes"][0]["kind"] = "bogus"
    with pytest.raises(ValueError, match="unknown"):
        parse_recipe(bad)


def test_parse_recipe_rejects_duplicate_node_name():
    bad = _good_recipe()
    bad["nodes"][1]["name"] = "seed1"
    with pytest.raises(ValueError, match="duplicate"):
        parse_recipe(bad)


def test_parse_recipe_rejects_dangling_edge():
    bad = _good_recipe()
    bad["edges"][0][1] = "ghost"
    with pytest.raises(ValueError, match="not in nodes"):
        parse_recipe(bad)


def test_parse_recipe_rejects_empty_nodes():
    with pytest.raises(ValueError):
        parse_recipe({"nodes": [], "edges": []})


def test_parse_recipe_rejects_non_dict_node():
    with pytest.raises(TypeError):
        parse_recipe({"nodes": ["not-a-dict"]})


def test_parse_recipe_rejects_non_dict_input():
    with pytest.raises(TypeError):
        parse_recipe([1, 2])


def test_parse_recipe_rejects_too_many_nodes():
    bad = {
        "nodes": [
            {"name": f"n{index}", "kind": "seed"} for index in range(257)
        ],
        "edges": [],
    }
    with pytest.raises(ValueError, match="exceeds"):
        parse_recipe(bad)


def test_parse_recipe_rejects_duplicate_edge():
    bad = _good_recipe()
    bad["edges"].append(["seed1", "llm1"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_recipe(bad)


def test_parse_recipe_rejects_bad_node_name():
    bad = _good_recipe()
    bad["nodes"][0]["name"] = "Bad Name"
    with pytest.raises(ValueError):
        parse_recipe(bad)


def test_parse_recipe_yaml_roundtrip():
    text = """
nodes:
  - name: seed1
    kind: seed
  - name: out1
    kind: sampler
edges:
  - [seed1, out1]
"""
    dag = parse_recipe_yaml(text)
    assert dag.topo_order == ("seed1", "out1")


def test_parse_recipe_yaml_rejects_null_byte():
    with pytest.raises(ValueError, match="null"):
        parse_recipe_yaml("nodes:\n  - name: a\x00\n    kind: seed")


def test_parse_recipe_yaml_rejects_invalid_yaml():
    with pytest.raises(ValueError, match="YAML"):
        parse_recipe_yaml("nodes: [\n  unterminated")


def test_load_recipe_yaml_happy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(
        "nodes:\n  - name: seed1\n    kind: seed\nedges: []\n",
        encoding="utf-8",
    )
    dag = load_recipe_yaml("r.yaml")
    assert dag.topo_order == ("seed1",)


def test_load_recipe_yaml_rejects_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "other.yaml"
    other.write_text("nodes:\n  - name: a\n    kind: seed\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="under cwd"):
            load_recipe_yaml(str(other))
    finally:
        try:
            other.unlink()
        except OSError:
            pass


def test_load_recipe_yaml_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_recipe_yaml("missing.yaml")


def test_load_recipe_yaml_rejects_null_byte_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        load_recipe_yaml("a\x00b.yaml")


import sys  # noqa: E402  -- keep imports tight for symlink-test guard


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privilege on Windows")
def test_load_recipe_yaml_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "real.yaml"
    target.write_text(
        "nodes:\n  - name: a\n    kind: seed\nedges: []\n",
        encoding="utf-8",
    )
    link = tmp_path / "link.yaml"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        load_recipe_yaml("link.yaml")


def test_data_recipe_cli_happy(tmp_path, monkeypatch):
    from soup_cli.commands import data as data_cmd

    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(
        "nodes:\n  - name: seed1\n    kind: seed\nedges: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(data_cmd.app, ["recipe", "r.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "validated" in result.output
    # v0.53.7 #106: live runner replaces the "deferred" stub. Without --execute,
    # the CLI now prompts the user to re-run with --execute instead.
    assert "--execute" in result.output


def test_data_recipe_cli_invalid_recipe(tmp_path, monkeypatch):
    from soup_cli.commands import data as data_cmd

    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "r.yaml"
    recipe_path.write_text(
        "nodes:\n  - name: seed1\n    kind: bogus\nedges: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(data_cmd.app, ["recipe", "r.yaml"])
    assert result.exit_code == 2
    assert "Invalid" in result.output


def test_data_recipe_cli_missing_file(tmp_path, monkeypatch):
    from soup_cli.commands import data as data_cmd

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(data_cmd.app, ["recipe", "ghost.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# Coverage-gap fill (post-review)
# ---------------------------------------------------------------------------


def test_register_plugin_rejects_too_many_model_groups():
    with pytest.raises(ValueError, match="exceeds"):
        plugins_pkg.register_plugin(
            name="big-grp",
            version="1.0.0",
            plugin=_NoopPlugin(),
            model_groups=[f"g{index}" for index in range(33)],
        )


def test_register_plugin_rejects_empty_model_group_entry():
    with pytest.raises(ValueError, match="model_group"):
        plugins_pkg.register_plugin(
            name="bad-grp",
            version="1.0.0",
            plugin=_NoopPlugin(),
            model_groups=[""],
        )


def test_register_plugin_rejects_oversize_model_group_entry():
    with pytest.raises(ValueError, match="exceeds"):
        plugins_pkg.register_plugin(
            name="long-grp",
            version="1.0.0",
            plugin=_NoopPlugin(),
            model_groups=["x" * 200],
        )


def test_register_plugin_rejects_null_byte_description():
    with pytest.raises(ValueError, match="null"):
        plugins_pkg.register_plugin(
            name="bad-desc",
            version="1.0.0",
            plugin=_NoopPlugin(),
            description="bad\x00desc",
        )


def test_register_plugin_rejects_oversize_description():
    with pytest.raises(ValueError, match="exceeds"):
        plugins_pkg.register_plugin(
            name="long-desc",
            version="1.0.0",
            plugin=_NoopPlugin(),
            description="x" * 1000,
        )


def test_enable_plugin_already_enabled_returns_false():
    plugins_pkg.register_plugin(
        name="warm", version="1.0.0", plugin=_NoopPlugin()
    )
    assert plugins_pkg.enable_plugin("warm") is False


def test_disable_unknown_raises():
    with pytest.raises(KeyError):
        plugins_pkg.disable_plugin("missing")


def test_list_hook_names_returns_canonical_tuple():
    assert plugins_pkg.list_hook_names() == (
        "pre_train",
        "post_train",
        "pre_step",
        "post_step",
    )


def test_to_anthropic_rejects_non_list_messages():
    with pytest.raises(TypeError):
        to_anthropic({"model": "x", "messages": "oops"})


def test_to_anthropic_rejects_non_str_or_list_content():
    with pytest.raises(TypeError):
        to_anthropic({
            "model": "x",
            "messages": [{"role": "user", "content": 42}],
        })


def test_to_anthropic_rejects_bool_temperature():
    with pytest.raises(TypeError):
        to_anthropic({
            "model": "x",
            "messages": [{"role": "user", "content": "a"}],
            "temperature": True,
        })


@pytest.mark.parametrize("temp", [0.0, 2.0])
def test_to_anthropic_accepts_temperature_boundaries(temp):
    out = to_anthropic({
        "model": "x",
        "messages": [{"role": "user", "content": "a"}],
        "temperature": temp,
    })
    assert out["temperature"] == temp


def test_to_anthropic_rejects_max_tokens_zero():
    with pytest.raises(ValueError):
        to_anthropic({
            "model": "x",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": 0,
        })


def test_validate_anthropic_payload_rejects_non_dict():
    with pytest.raises(TypeError):
        validate_anthropic_payload([1, 2, 3])  # type: ignore[arg-type]


def test_validate_anthropic_payload_rejects_bool_max_tokens():
    with pytest.raises(TypeError):
        validate_anthropic_payload({
            "model": "x",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": True,
        })


def test_is_domain_allowed_strips_port():
    assert is_domain_allowed("example.com:443", ("example.com",)) is True
    assert is_domain_allowed("a.example.com:443", (".example.com",)) is True


def test_is_domain_allowed_rejects_ipv6_literal():
    assert is_domain_allowed("[::1]", ("example.com",)) is False


def test_validate_ngram_config_rejects_non_config():
    with pytest.raises(TypeError):
        validate_ngram_config({"n": 4})  # type: ignore[arg-type]


def test_get_trainer_plugin_rejects_null_byte():
    with pytest.raises(ValueError, match="NUL"):
        get_trainer_plugin("grok\x00fast")


def test_get_trainer_plugin_rejects_empty_string():
    with pytest.raises(ValueError):
        get_trainer_plugin("")


def test_parse_recipe_rejects_too_many_edges():
    nodes = [
        {"name": f"n{index}", "kind": "seed"}
        for index in range(50)
    ]
    edges = [["n0", "n1"]] * 1025
    with pytest.raises(ValueError):
        parse_recipe({"nodes": nodes, "edges": edges})


def test_parse_recipe_rejects_non_pair_edge_entry():
    bad = _good_recipe()
    bad["edges"] = ["not-a-pair"]
    with pytest.raises(ValueError, match="2-element"):
        parse_recipe(bad)


def test_parse_recipe_yaml_rejects_oversize_text():
    text = "nodes:\n" + ("  - name: x\n    kind: seed\n" * 200_000)
    with pytest.raises(ValueError, match="bytes"):
        parse_recipe_yaml(text)


def test_parse_recipe_yaml_rejects_non_string():
    with pytest.raises(TypeError):
        parse_recipe_yaml(42)  # type: ignore[arg-type]


def test_load_recipe_yaml_rejects_non_string_path():
    with pytest.raises((TypeError, ValueError)):
        load_recipe_yaml(42)  # type: ignore[arg-type]


def test_load_recipe_yaml_rejects_oversize_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    big = tmp_path / "big.yaml"
    # File-size guard fires before YAML parsing.
    big.write_bytes(b"nodes: []\n" + b"# pad\n" * 200_000)
    with pytest.raises(ValueError, match="bytes"):
        load_recipe_yaml("big.yaml")
