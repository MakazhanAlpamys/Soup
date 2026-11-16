"""Run prefixes must select the same stored metrics as full IDs (#2751)."""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.ui_helpers import ui_client

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

RUN_A = "run_alpha_111"
RUN_B = "run_beta_222"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator["TestClient"]:
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.ui.app import create_app, get_auth_token

    db_path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(db_path))
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(db_path=db_path)
    try:
        for run_id, task, losses in (
            (RUN_A, "sft", (2.5, 2.1)),
            (RUN_B, "dpo", (1.8, 1.2)),
        ):
            tracker.start_run(
                {"base": f"model-{task}", "task": task}, "cpu", "CPU", {}, run_id=run_id,
            )
            for step, loss in zip((10, 20), losses, strict=True):
                tracker.log_metrics(run_id, step=step, loss=loss, lr=1e-5)
    finally:
        tracker.close()

    with ui_client(create_app()) as test_client:
        test_client.headers["Authorization"] = f"Bearer {get_auth_token()}"
        yield test_client


@pytest.mark.parametrize("requested_id", [RUN_A, "run_alpha"], ids=["full-id", "prefix"])
def test_metrics_returns_full_id_and_stored_rows(client: "TestClient", requested_id: str) -> None:
    response = client.get(f"/api/runs/{requested_id}/metrics")

    assert response.status_code == 200
    data = response.json()
    # Check the actual values, not just equality between two possibly empty replies.
    assert [(row["step"], row["loss"]) for row in data["metrics"]] == [(10, 2.5), (20, 2.1)]
    assert data["run_id"] == RUN_A
    assert all(row["run_id"] == RUN_A for row in data["metrics"])

    exact = client.get(f"/api/runs/{RUN_A}/metrics")
    assert exact.status_code == 200
    assert data == exact.json()


@pytest.mark.parametrize(
    "ids",
    [(RUN_A, RUN_B), ("run_alpha", "run_beta"), (RUN_A, "run_beta")],
    ids=["full-ids", "prefixes", "mixed"],
)
def test_compare_returns_resolved_runs_and_their_metrics(
    client: "TestClient", ids: tuple[str, str],
) -> None:
    response = client.get("/api/runs/compare", params={"ids": ",".join(ids)})

    assert response.status_code == 200
    data = response.json()
    runs = data["runs"]
    assert len(runs) == 2
    for run, full_id, task, losses in zip(
        runs, (RUN_A, RUN_B), ("sft", "dpo"), ((2.5, 2.1), (1.8, 1.2)), strict=True,
    ):
        assert [(row["step"], row["loss"]) for row in run["metrics"]] == list(
            zip((10, 20), losses, strict=True)
        )
        assert run["run_id"] == full_id
        assert run["config"] == {"base": f"model-{task}", "task": task}
        assert all(row["run_id"] == full_id for row in run["metrics"])

    exact = client.get("/api/runs/compare", params={"ids": f"{RUN_A},{RUN_B}"})
    assert exact.status_code == 200
    assert data == exact.json()


@pytest.mark.parametrize(
    "requested_id, detail",
    [
        ("run_missing", "Run not found"),
        ("run_", "Run id prefix 'run_' matches 2 runs; give more characters"),
    ],
    ids=["missing", "ambiguous"],
)
def test_metrics_preserves_lookup_errors(
    client: "TestClient", requested_id: str, detail: str,
) -> None:
    response = client.get(f"/api/runs/{requested_id}/metrics")

    assert response.status_code == 404
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize("unresolved_id", ["run_missing", "run_"], ids=["missing", "ambiguous"])
def test_compare_keeps_unresolved_entry_and_other_runs(
    client: "TestClient", unresolved_id: str,
) -> None:
    response = client.get("/api/runs/compare", params={"ids": f"{unresolved_id},{RUN_A}"})

    assert response.status_code == 200
    runs = response.json()["runs"]
    assert len(runs) == 2
    assert runs[0] == {"run_id": unresolved_id, "config": {}, "metrics": []}
    assert runs[1]["run_id"] == RUN_A
    assert [(row["step"], row["loss"]) for row in runs[1]["metrics"]] == [(10, 2.5), (20, 2.1)]
