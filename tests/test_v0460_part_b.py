"""v0.46.0 Part B — Agent Forge tests."""

from __future__ import annotations

import json
import os
import sys

import pytest
from typer.testing import CliRunner

from soup_cli.utils.agent_forge import (
    Endpoint,
    SpecReport,
    SynthRow,
    detect_spec_kind,
    endpoint_to_rows,
    load_spec_file,
    parse_graphql,
    parse_mcp,
    parse_openapi,
    parse_spec,
    synthesise_dataset,
    write_dataset,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# parse_openapi
# ---------------------------------------------------------------------------


_OPENAPI_SAMPLE = {
    "openapi": "3.0.0",
    "info": {"title": "Sample", "version": "1.0"},
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List pets",
                "parameters": [
                    {"name": "limit", "in": "query"},
                ],
            },
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
            },
        },
        "/pets/{id}": {
            "get": {
                "operationId": "getPet",
                "parameters": [
                    {"name": "id", "in": "path"},
                ],
            }
        },
    },
}


def test_parse_openapi_basic():
    endpoints, warnings = parse_openapi(_OPENAPI_SAMPLE)
    tools = {ep.tool for ep in endpoints}
    assert "listPets" in tools
    assert "createPet" in tools
    assert "getPet" in tools
    assert warnings == [] or all("$ref" not in w for w in warnings)


def test_parse_openapi_extracts_parameter_names():
    endpoints, _ = parse_openapi(_OPENAPI_SAMPLE)
    list_pets = next(ep for ep in endpoints if ep.tool == "listPets")
    assert "limit" in list_pets.parameters


def test_parse_openapi_method_lowercased():
    endpoints, _ = parse_openapi(_OPENAPI_SAMPLE)
    for ep in endpoints:
        assert ep.method == ep.method.lower()


def test_parse_openapi_non_dict_raises():
    with pytest.raises(TypeError):
        parse_openapi("not a dict")  # type: ignore[arg-type]


def test_parse_openapi_missing_paths_returns_empty():
    eps, warnings = parse_openapi({"openapi": "3.0.0"})
    assert eps == []
    assert warnings


def test_parse_openapi_wrong_version_warns():
    _, warnings = parse_openapi({"openapi": "2.0.0", "paths": {}})
    assert any("openapi" in w.lower() for w in warnings)


def test_parse_openapi_skips_invalid_methods():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/x": {
                "BOGUS": {"operationId": "skipMe"},
                "get": {"operationId": "keepMe"},
            }
        },
    }
    endpoints, _ = parse_openapi(spec)
    tools = {ep.tool for ep in endpoints}
    assert "keepMe" in tools
    assert "skipMe" not in tools


def test_parse_openapi_generates_id_when_missing():
    spec = {
        "openapi": "3.0.0",
        "paths": {"/widgets": {"get": {}}},
    }
    endpoints, _ = parse_openapi(spec)
    assert endpoints
    # Generated id sanitised, starts with letter or underscore
    assert endpoints[0].tool[0].isalpha() or endpoints[0].tool.startswith("_")


def test_parse_openapi_ref_param_skipped_with_warning():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/x": {
                "get": {
                    "operationId": "getX",
                    "parameters": [{"$ref": "#/components/parameters/X"}],
                }
            }
        },
    }
    eps, warnings = parse_openapi(spec)
    assert eps
    assert any("$ref" in w for w in warnings)


# ---------------------------------------------------------------------------
# parse_mcp
# ---------------------------------------------------------------------------


_MCP_SAMPLE = {
    "tools": [
        {
            "name": "search_files",
            "description": "Search files by pattern",
            "inputSchema": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            },
        },
        {
            "name": "read_file",
            "description": "Read a file by path",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    ]
}


def test_parse_mcp_basic():
    endpoints, _ = parse_mcp(_MCP_SAMPLE)
    tools = {ep.tool for ep in endpoints}
    assert "search_files" in tools
    assert "read_file" in tools


def test_parse_mcp_extracts_input_schema_props():
    endpoints, _ = parse_mcp(_MCP_SAMPLE)
    search = next(ep for ep in endpoints if ep.tool == "search_files")
    assert "pattern" in search.parameters
    assert "path" in search.parameters


def test_parse_mcp_method_is_invoke():
    endpoints, _ = parse_mcp(_MCP_SAMPLE)
    for ep in endpoints:
        assert ep.method == "invoke"


def test_parse_mcp_path_uses_mcp_scheme():
    endpoints, _ = parse_mcp(_MCP_SAMPLE)
    for ep in endpoints:
        assert ep.path.startswith("mcp://")


def test_parse_mcp_missing_tools_returns_empty():
    eps, warnings = parse_mcp({})
    assert eps == []
    assert warnings


def test_parse_mcp_missing_name_warns():
    spec = {"tools": [{"description": "no name"}]}
    _, warnings = parse_mcp(spec)
    assert any("name" in w.lower() for w in warnings)


def test_parse_mcp_non_dict_raises():
    with pytest.raises(TypeError):
        parse_mcp([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_graphql
# ---------------------------------------------------------------------------


_GRAPHQL_SAMPLE = {
    "data": {
        "__schema": {
            "queryType": {"name": "Query"},
            "mutationType": {"name": "Mutation"},
            "types": [
                {
                    "name": "Query",
                    "fields": [
                        {
                            "name": "user",
                            "description": "Fetch a user by id",
                            "args": [{"name": "id"}],
                        }
                    ],
                },
                {
                    "name": "Mutation",
                    "fields": [
                        {
                            "name": "createUser",
                            "description": "Create a user",
                            "args": [{"name": "name"}, {"name": "email"}],
                        }
                    ],
                },
            ],
        }
    }
}


def test_parse_graphql_basic():
    endpoints, _ = parse_graphql(_GRAPHQL_SAMPLE)
    tools = {ep.tool for ep in endpoints}
    assert any("user" in t for t in tools)
    assert any("createUser" in t for t in tools)


def test_parse_graphql_methods():
    endpoints, _ = parse_graphql(_GRAPHQL_SAMPLE)
    methods = {ep.method for ep in endpoints}
    assert "query" in methods
    assert "mutation" in methods


def test_parse_graphql_args_captured():
    endpoints, _ = parse_graphql(_GRAPHQL_SAMPLE)
    create_user = next(ep for ep in endpoints if "createUser" in ep.tool)
    assert "name" in create_user.parameters
    assert "email" in create_user.parameters


def test_parse_graphql_missing_schema_returns_empty():
    eps, warnings = parse_graphql({"data": {}})
    assert eps == []
    assert warnings


def test_parse_graphql_non_dict_raises():
    with pytest.raises(TypeError):
        parse_graphql("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# detect_spec_kind + parse_spec
# ---------------------------------------------------------------------------


def test_detect_openapi():
    assert detect_spec_kind(_OPENAPI_SAMPLE) == "openapi"


def test_detect_mcp():
    assert detect_spec_kind(_MCP_SAMPLE) == "mcp"


def test_detect_graphql():
    assert detect_spec_kind(_GRAPHQL_SAMPLE) == "graphql"


def test_detect_unknown_raises():
    with pytest.raises(ValueError, match="cannot detect"):
        detect_spec_kind({"random": "stuff"})


def test_detect_non_dict_raises():
    with pytest.raises(TypeError):
        detect_spec_kind("not a dict")  # type: ignore[arg-type]


def test_parse_spec_auto_detect():
    endpoints, report = parse_spec(_OPENAPI_SAMPLE)
    assert report.spec_kind == "openapi"
    assert report.endpoint_count >= 3
    assert isinstance(report, SpecReport)


def test_parse_spec_explicit_kind():
    endpoints, report = parse_spec(_MCP_SAMPLE, kind="mcp")
    assert report.spec_kind == "mcp"
    assert endpoints


def test_parse_spec_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown spec kind"):
        parse_spec(_OPENAPI_SAMPLE, kind="evil")


def test_parse_spec_non_string_kind():
    with pytest.raises(TypeError):
        parse_spec(_OPENAPI_SAMPLE, kind=123)  # type: ignore[arg-type]


def test_parse_spec_deduplicates_tools():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/x": {
                "get": {"operationId": "myTool"},
            },
            "/y": {
                "get": {"operationId": "myTool"},
            },
        },
    }
    endpoints, report = parse_spec(spec)
    assert len({ep.tool for ep in endpoints}) == 1
    assert report.skipped == 1


def test_parse_mcp_rejects_newline_in_name():
    spec = {"tools": [{"name": "evil\nhost", "description": "x"}]}
    eps, warnings = parse_mcp(spec)
    # Either skipped with warning, or path validation strips/rejects newline
    if eps:
        for ep in eps:
            assert "\n" not in ep.path
    assert any("invalid" in w.lower() or "skip" in w.lower() for w in warnings) or not eps


def test_parse_graphql_rejects_newline_in_field_name():
    spec = {
        "__schema": {
            "queryType": {"name": "Query"},
            "types": [{
                "name": "Query",
                "fields": [{"name": "evil\nhost", "args": []}],
            }],
        }
    }
    eps, _ = parse_graphql(spec)
    for ep in eps:
        assert "\n" not in ep.path


# ---------------------------------------------------------------------------
# endpoint_to_rows + synthesise_dataset
# ---------------------------------------------------------------------------


def test_endpoint_to_rows_basic():
    ep = Endpoint(
        tool="search", method="get", path="/search",
        description="Search the index", parameters=("query",),
        spec_kind="openapi",
    )
    rows = endpoint_to_rows(ep, examples_per_endpoint=2)
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, SynthRow)
        assert row.tool == "search"
        assert row.source_endpoint == "/search"
        # 2 messages: user + assistant-with-tool-call
        assert len(row.messages) == 2
        assert row.messages[0]["role"] == "user"
        assert row.messages[1]["role"] == "assistant"
        assert "tool_calls" in row.messages[1]


def test_endpoint_to_rows_arguments_are_json_string():
    ep = Endpoint(
        tool="search", method="get", path="/search",
        description="", parameters=("q", "limit"), spec_kind="openapi",
    )
    rows = endpoint_to_rows(ep, examples_per_endpoint=1)
    tc = rows[0].messages[1]["tool_calls"][0]
    parsed = json.loads(tc["function"]["arguments"])
    assert set(parsed.keys()) == {"q", "limit"}


def test_endpoint_to_rows_bool_examples_rejected():
    ep = Endpoint(
        tool="x", method="get", path="/x",
        description="", parameters=(), spec_kind="openapi",
    )
    with pytest.raises(TypeError):
        endpoint_to_rows(ep, examples_per_endpoint=True)  # type: ignore[arg-type]


def test_endpoint_to_rows_zero_rejected():
    ep = Endpoint(
        tool="x", method="get", path="/x",
        description="", parameters=(), spec_kind="openapi",
    )
    with pytest.raises(ValueError):
        endpoint_to_rows(ep, examples_per_endpoint=0)


def test_endpoint_to_rows_oversize_rejected():
    ep = Endpoint(
        tool="x", method="get", path="/x",
        description="", parameters=(), spec_kind="openapi",
    )
    with pytest.raises(ValueError):
        endpoint_to_rows(ep, examples_per_endpoint=33)


def test_endpoint_to_rows_type_check():
    with pytest.raises(TypeError):
        endpoint_to_rows("not-an-endpoint", 1)  # type: ignore[arg-type]


def test_synthesise_dataset_flat_list():
    endpoints, _ = parse_openapi(_OPENAPI_SAMPLE)
    rows = synthesise_dataset(endpoints, examples_per_endpoint=2)
    assert len(rows) == 2 * len(endpoints)
    assert all(isinstance(r, SynthRow) for r in rows)


def test_synthesise_dataset_rejects_string():
    with pytest.raises(TypeError):
        synthesise_dataset("not a list", 1)  # type: ignore[arg-type]


def test_synth_row_to_dict_serialisable():
    ep = Endpoint(
        tool="x", method="get", path="/x",
        description="", parameters=(), spec_kind="openapi",
    )
    row = endpoint_to_rows(ep, 1)[0]
    d = row.to_dict()
    # Round-trips through JSON
    json.dumps(d)


# ---------------------------------------------------------------------------
# load_spec_file + write_dataset (cwd containment + symlink)
# ---------------------------------------------------------------------------


def test_load_spec_file_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "spec.yaml"
    p.write_text("openapi: '3.0.0'\npaths: {}\n", encoding="utf-8")
    out = load_spec_file("spec.yaml")
    assert out.get("openapi") == "3.0.0"


def test_load_spec_file_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_OPENAPI_SAMPLE), encoding="utf-8")
    out = load_spec_file("spec.json")
    assert out["openapi"] == "3.0.0"


def test_load_spec_file_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path.parent / "evil.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must stay under cwd"):
        load_spec_file(str(other))


def test_load_spec_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_spec_file("missing.json")


def test_load_spec_file_null_byte_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_spec_file("evil\x00.json")


def test_load_spec_file_non_string():
    with pytest.raises(TypeError):
        load_spec_file(123)  # type: ignore[arg-type]


def test_load_spec_file_oversize_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "big.json"
    p.write_bytes(b"{" + b"x" * (6 * 1024 * 1024) + b"}")
    with pytest.raises(ValueError, match="exceeds"):
        load_spec_file("big.json")


def test_load_spec_file_non_dict_root_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_spec_file("list.json")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink ACL on Windows CI")
def test_load_spec_file_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    with pytest.raises(ValueError, match="symlink"):
        load_spec_file("link.json")


def test_write_dataset_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    endpoints, _ = parse_openapi(_OPENAPI_SAMPLE)
    rows = synthesise_dataset(endpoints, 1)
    out = write_dataset(rows, "dataset.jsonl")
    assert os.path.exists(out)
    # Each line valid JSON
    with open(out, encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            assert "messages" in data
            assert "tool" in data


def test_write_dataset_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [SynthRow(messages=({"role": "u", "content": "x"},), tool="t",
                     source_endpoint="/")]
    abs_outside = str(tmp_path.parent / "evil.jsonl")
    with pytest.raises(ValueError, match="must stay under cwd"):
        write_dataset(rows, abs_outside)


def test_write_dataset_null_byte_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        write_dataset([], "x\x00.jsonl")


def test_write_dataset_non_string_path():
    with pytest.raises(TypeError):
        write_dataset([], 123)  # type: ignore[arg-type]


def test_write_dataset_invalid_row_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        write_dataset([{"not": "a SynthRow"}], "out.jsonl")  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_agent_synth_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    result = runner.invoke(
        agent.app, ["synth", "--spec", "spec.json", "--output", "ds.jsonl"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "ds.jsonl").exists()
    assert "listPets" in result.output


def test_cli_agent_synth_unknown_spec_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    result = runner.invoke(agent.app, ["synth", "--spec", "missing.json"])
    assert result.exit_code == 1, result.output


def test_cli_agent_synth_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    abs_outside = str(tmp_path.parent / "evil.json")
    result = runner.invoke(agent.app, ["synth", "--spec", abs_outside])
    assert result.exit_code == 1, result.output


def test_cli_agent_train_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    result = runner.invoke(
        agent.app,
        ["train", "--spec", "spec.json", "--base", "meta-llama/Llama-3.2-1B",
         "--dataset-out", "ds.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (tmp_path / "ds.jsonl").exists()
    assert "Planned" in result.output or "agent_train.yaml" in result.output


def test_cli_agent_eval_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    preds = [
        {"tool": "listPets", "arguments": {"limit": "10"}},
        {"tool": "listPets", "arguments": {"unknownParam": "x"}},
        {"tool": "nopeTool", "arguments": {}},
    ]
    (tmp_path / "preds.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8"
    )
    result = runner.invoke(
        agent.app,
        ["eval", "--spec", "spec.json", "--predictions", "preds.jsonl"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # 2/3 tools match (listPets x2), 1/3 args fully valid
    assert "Tool match" in result.output
    assert "Args valid" in result.output


def test_cli_agent_eval_outside_cwd_predictions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    abs_outside = str(tmp_path.parent / "preds.jsonl")
    result = runner.invoke(
        agent.app,
        ["eval", "--spec", "spec.json", "--predictions", abs_outside],
    )
    assert result.exit_code == 1, result.output


def test_cli_agent_synth_help():
    from soup_cli.commands import agent

    result = runner.invoke(agent.app, ["synth", "--help"])
    assert result.exit_code == 0


def test_cli_agent_help():
    from soup_cli.commands import agent

    result = runner.invoke(agent.app, ["--help"])
    assert result.exit_code == 0
    assert "synth" in result.output
    assert "train" in result.output
    assert "eval" in result.output


def test_cli_agent_train_help():
    from soup_cli.commands import agent

    result = runner.invoke(agent.app, ["train", "--help"])
    assert result.exit_code == 0


def test_cli_agent_eval_help():
    from soup_cli.commands import agent

    result = runner.invoke(agent.app, ["eval", "--help"])
    assert result.exit_code == 0


def test_cli_agent_eval_missing_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "preds.jsonl").write_text("{}\n", encoding="utf-8")
    result = runner.invoke(
        agent.app, ["eval", "--spec", "missing.json", "--predictions", "preds.jsonl"],
    )
    assert result.exit_code == 1, result.output


def test_cli_agent_eval_outside_cwd_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "preds.jsonl").write_text("{}\n", encoding="utf-8")
    abs_outside = str(tmp_path.parent / "evil.json")
    result = runner.invoke(
        agent.app,
        ["eval", "--spec", abs_outside, "--predictions", "preds.jsonl"],
    )
    assert result.exit_code == 1, result.output


def test_cli_agent_train_rejects_newline_in_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    result = runner.invoke(
        agent.app,
        ["train", "--spec", "spec.json",
         "--base", "evil\ntraining: { epochs: 9999 }"],
    )
    assert result.exit_code == 2, result.output
    assert "newline" in result.output.lower() or "base" in result.output.lower()


def test_cli_agent_train_rejects_null_byte_in_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "spec.json").write_text(
        json.dumps(_OPENAPI_SAMPLE), encoding="utf-8"
    )
    result = runner.invoke(
        agent.app,
        ["train", "--spec", "spec.json", "--base", "evil\x00"],
    )
    assert result.exit_code == 2, result.output


def test_write_dataset_partial_failure_no_partial_file(tmp_path, monkeypatch):
    """Mid-stream TypeError must not leave a partial file at target."""
    monkeypatch.chdir(tmp_path)
    rows = [
        SynthRow(messages=(), tool="t1", source_endpoint="/"),
        "not a SynthRow",  # type: ignore[list-item]
    ]
    with pytest.raises(TypeError):
        write_dataset(rows, "out.jsonl")  # type: ignore[arg-type]
    # Atomic write: target file should NOT exist after partial failure
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="symlink ACL on Windows")
def test_write_dataset_symlink_target_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.jsonl"
    real.write_text("", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")
    rows = [SynthRow(messages=(), tool="t", source_endpoint="/")]
    with pytest.raises(ValueError, match="symlink"):
        write_dataset(rows, "link.jsonl")


def test_cli_agent_synth_no_endpoints_exits_1(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands import agent

    (tmp_path / "empty.json").write_text('{"openapi": "3.0.0", "paths": {}}',
                                          encoding="utf-8")
    result = runner.invoke(agent.app, ["synth", "--spec", "empty.json"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Endpoint dataclass invariants
# ---------------------------------------------------------------------------


def test_endpoint_is_frozen():
    import dataclasses

    ep = Endpoint(
        tool="x", method="get", path="/x",
        description="", parameters=(), spec_kind="openapi",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.tool = "y"  # type: ignore[misc]


def test_synth_row_is_frozen():
    import dataclasses

    row = SynthRow(messages=(), tool="x", source_endpoint="/")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.tool = "y"  # type: ignore[misc]


def test_spec_report_is_frozen_explicit():
    import dataclasses

    _, report = parse_spec(_OPENAPI_SAMPLE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.spec_kind = "x"  # type: ignore[misc]
