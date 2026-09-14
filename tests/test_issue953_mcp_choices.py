"""MCP choice errors name the input and the live choices without leaking paths."""

from pathlib import Path

import pytest

from soup_cli.commands import export
from soup_cli.mcp_server import registry
from soup_cli.utils import profiler

pytestmark = pytest.mark.integration


@pytest.fixture
def tool_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "private-config.yaml"
    config.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\ndata:\n  train: train.jsonl\n",
        encoding="utf-8",
    )
    return {"model": str(tmp_path / "private-model"), "config": str(config)}


@pytest.mark.parametrize("tool,key", [("export", "format"), ("profile", "gpu")])
def test_unknown_choice_names_input_and_live_choices(tool_args: dict, tool: str, key: str) -> None:
    handler = getattr(registry, f"tool_{tool}")
    choices = export.SUPPORTED_FORMATS if tool == "export" else profiler.GPU_MEMORY
    with pytest.raises(registry.McpToolError) as exc:
        handler({**tool_args, key: "BOGUS - value"})
    message = str(exc.value)
    assert repr("BOGUS - value") in message
    assert all(choice in message for choice in choices)
    assert tool_args["model"] not in message
    assert tool_args["config"] not in message


def test_export_accepts_and_lists_new_format(
    tool_args: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export, "SUPPORTED_FORMATS", (*export.SUPPORTED_FORMATS, "new_format"))
    result = registry.tool_export({**tool_args, "format": "new_format"})
    assert result["format"] == "new_format"
    assert "--format new_format" in result["would_run"]
    with pytest.raises(registry.McpToolError, match="new_format"):
        registry.tool_export({**tool_args, "format": "bogus"})


def test_profile_accepts_and_lists_new_gpu(
    tool_args: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(profiler.GPU_MEMORY, "newcard", 123)
    result = registry.tool_profile({**tool_args, "gpu": "NEW-CARD"})
    assert result["gpu_memory_gb"] == 123.0
    with pytest.raises(registry.McpToolError, match="newcard"):
        registry.tool_profile({**tool_args, "gpu": "bogus"})


@pytest.mark.parametrize("tool,key", [("export", "format"), ("profile", "gpu")])
@pytest.mark.parametrize("value", ["", "   ", "bogus", "\x1b[31m\n\x00\u202e"])
def test_invalid_choices_remain_refused(tool_args: dict, tool: str, key: str, value: str) -> None:
    with pytest.raises(registry.McpToolError):
        getattr(registry, f"tool_{tool}")({**tool_args, key: value})


@pytest.mark.parametrize("tool,key", [("export", "format"), ("profile", "gpu")])
def test_control_characters_are_escaped_on_the_wire(tool_args: dict, tool: str, key: str) -> None:
    import anyio

    from soup_cli.mcp_server.server import build_server
    from tests.mcp_roundtrip import connected_session, is_error

    value = "\x1b[31m\n\x00\u202e"

    async def call_tool():
        server = build_server(registry.build_registry(allow_mutating=True))
        async with connected_session(server) as session:
            return await session.call_tool(tool, {**tool_args, key: value})

    result = anyio.run(call_tool)
    assert is_error(result)
    message = " ".join(block.text for block in result.content if block.type == "text")
    assert repr(value) in message
    assert all(char not in message for char in ("\x1b", "\n", "\x00", "\u202e"))


@pytest.mark.parametrize("tool,key", [("export", "format"), ("profile", "gpu")])
def test_oversize_values_keep_existing_bound(tool_args: dict, tool: str, key: str) -> None:
    with pytest.raises(registry.McpToolError) as exc:
        getattr(registry, f"tool_{tool}")({**tool_args, key: "x" * 4097})
    assert "4096" in str(exc.value)
    assert len(str(exc.value)) < 100


def test_existing_export_and_normalized_gpu_still_work(tool_args: dict) -> None:
    for fmt in export.SUPPORTED_FORMATS:
        assert registry.tool_export({**tool_args, "format": fmt})["format"] == fmt
    assert registry.tool_profile({**tool_args, "gpu": "RTX 4090"})["gpu_memory_gb"] == 24
