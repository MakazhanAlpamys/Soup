"""MCP runs_show says when a run id prefix matches several runs (left over from #2470).

The tool accepts an id prefix. ``get_run`` returns None both for a prefix that
matches nothing and for one that matches several, and the tool answered
"run not found" for both, while ``soup runs show`` and the Web UI name the
ambiguous prefix and ask for more characters.
"""

import anyio
import pytest

from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server import registry as reg

AMBIGUOUS_LINE = "Run id prefix 'run_dup_' matches 2 runs; give more characters"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    tracker = ExperimentTracker(db_path=path)
    try:
        for run_id in ("run_dup_1", "run_dup_2", "run_[x]\x9b1", "run_[x]\x9b2", "run_solo_1"):
            tracker.start_run({"base": "m", "task": "sft"}, "cpu", "CPU", {}, run_id=run_id)
    finally:
        tracker.close()
    return path


@pytest.mark.parametrize(
    ("run_id", "message"),
    [
        ("run_dup_", AMBIGUOUS_LINE),
        ("run_[x]\x9b", "Run id prefix 'run_[x]' matches 2 runs; give more characters"),
        ("run_gone", "run not found"),
    ],
)
def test_runs_show_tells_an_ambiguous_prefix_from_a_missing_run(run_id, message):
    with pytest.raises(reg.McpToolError) as exc_info:
        reg.call_tool("runs_show", {"run_id": run_id})

    assert str(exc_info.value) == message


@pytest.mark.parametrize("run_id", ["run_solo_1", "run_solo"])
def test_runs_show_still_resolves_an_exact_id_and_a_unique_prefix(run_id):
    assert reg.call_tool("runs_show", {"run_id": run_id})["run_id"] == "run_solo_1"


def test_runs_show_over_an_mcp_session_names_the_ambiguous_prefix():
    pytest.importorskip("mcp")
    from soup_cli.mcp_server.server import build_server
    from tests.mcp_roundtrip import connected_session, is_error

    async def _call():
        server = build_server(reg.build_registry(allow_mutating=False))
        async with connected_session(server) as session:
            return await session.call_tool("runs_show", {"run_id": "run_dup_"})

    result = anyio.run(_call)

    assert is_error(result)
    assert result.content[0].text == AMBIGUOUS_LINE
