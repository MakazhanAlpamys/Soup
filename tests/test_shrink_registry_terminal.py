"""Best-effort shrink registry warnings display the supplied reference literally."""

import pytest
from rich.console import Console

from soup_cli.commands import shrink
from soup_cli.registry.store import RegistryStore
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    ("suffix", "display_suffix"),
    [
        pytest.param("-missing", "-missing", id="plain"),
        pytest.param("[red]", "[red]", id="markup"),
        pytest.param("[/]", "[/]", id="closing-markup"),
        pytest.param(":warning:", ":warning:", id="emoji-alias"),
        pytest.param("\x9b\u202e", "", id="controls-do-not-affect-lookup"),
    ],
)
def test_shrink_missing_registry_ref_is_displayed_literally(
    suffix, display_suffix, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setattr(shrink, "console", Console(width=200, height=50))
    with RegistryStore() as store:
        entry_id = store.push(
            name="pruned",
            tag="test",
            base_model="test-model",
            task="sft",
            run_id=None,
            config={"task": "sft"},
        )
    report = tmp_path / "shrink_report.json"
    report_text = '{"decision":"SHIP"}'
    report.write_text(report_text, encoding="utf-8")

    shrink._attach_to_registry(entry_id + suffix, str(report))

    # In the controls case, cleaning the reference before lookup would
    # attach the report to the real entry instead of warning about the miss.
    with RegistryStore() as store:
        assert store.get_artifacts(entry_id) == []
    assert report.read_text(encoding="utf-8") == report_text
    output = capsys.readouterr().out
    assert "\x9b" not in output
    assert "\u202e" not in output
    assert strip_ansi(output) == (
        "Warning: could not attach to registry: registry entry not found: "
        f"{entry_id}{display_suffix}\n"
    )
