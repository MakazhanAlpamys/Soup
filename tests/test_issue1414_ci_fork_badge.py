"""Regression coverage for issue #1414's fork-safe CI badge update."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_test_count_badge_runs_only_in_upstream_repository() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("      - name: Update test count badge")
    end = workflow.find("\n      - name:", start + 1)
    block = workflow[start:] if end == -1 else workflow[start:end]

    assert "github.repository == 'MakazhanAlpamys/Soup'" in block
    assert "continue-on-error" not in block
