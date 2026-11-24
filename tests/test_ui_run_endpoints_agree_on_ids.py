"""GET, DELETE and /eval of a UI run agree on ids by the CLI's rule (#3088 B-3).

``GET /api/runs/{id}``, ``DELETE /api/runs/{id}`` and ``GET /api/runs/{id}/eval``
resolve the id through ``tracker.get_run``, as ``soup runs show`` and
``soup runs delete`` do: a unique prefix names its run, and an unknown id or a
prefix that matches several runs answers 404 with the detail of
``run_lookup_failure(..., markup=False, name_missing_id=False)``. DELETE and
/eval act on, and return, the full id.
"""

from __future__ import annotations

import pytest

from tests.ui_helpers import ui_client

pytest.importorskip("fastapi")

_FIRST = "run_20261121_135252_3181d6f4"
_SECOND = "run_20261121_140000_9a0b1c2d"
_OTHER = "run_20261019_090000_00ff00ff"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    return path


def _add_run(db_path, run_id: str, *, evals: int = 0, metrics: int = 0) -> None:
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker(db_path=db_path)
    try:
        tracker.start_run(
            config_dict={"base": "test"},
            device="cpu",
            device_name="CPU",
            gpu_info={"memory_total": "N/A"},
            run_id=run_id,
        )
        for step in range(1, metrics + 1):
            tracker.log_metrics(run_id, step=step, loss=1.0 / step)
        for index in range(evals):
            tracker.save_eval_result(
                model_path="./output",
                benchmark=f"bench{index}",
                score=0.5,
                details={},
                run_id=run_id,
            )
    finally:
        tracker.close()


def _rows(db_path, table: str, run_id: str) -> int:
    from soup_cli.experiment.tracker import ExperimentTracker

    tracker = ExperimentTracker(db_path=db_path)
    try:
        conn = tracker._get_conn()
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    finally:
        tracker.close()


def _client():
    from soup_cli.ui.app import create_app, get_auth_token

    client = ui_client(create_app())
    client.headers["Authorization"] = f"Bearer {get_auth_token()}"
    return client


def test_delete_with_a_unique_prefix_removes_that_run_and_returns_its_full_id(db_path):
    _add_run(db_path, _FIRST, evals=2, metrics=3)
    _add_run(db_path, _OTHER, evals=1, metrics=2)

    response = _client().delete("/api/runs/run_20261121")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "run_id": _FIRST}
    for table in ("runs", "metrics", "eval_results"):
        assert _rows(db_path, table, _FIRST) == 0
    assert _rows(db_path, "runs", _OTHER) == 1
    assert _rows(db_path, "metrics", _OTHER) == 2
    assert _rows(db_path, "eval_results", _OTHER) == 1


def test_eval_with_a_unique_prefix_returns_the_runs_rows_under_its_full_id(db_path):
    _add_run(db_path, _FIRST, evals=2)
    _add_run(db_path, _OTHER, evals=1)

    response = _client().get("/api/runs/run_20261121/eval")

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == _FIRST
    assert sorted(row["benchmark"] for row in data["eval_results"]) == ["bench0", "bench1"]
    assert {row["run_id"] for row in data["eval_results"]} == {_FIRST}


def test_eval_for_an_unknown_id_is_404_with_the_detail_of_get(db_path):
    _add_run(db_path, _FIRST, evals=1)
    client = _client()

    response = client.get("/api/runs/nope/eval")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}
    assert response.json() == client.get("/api/runs/nope").json()


def test_eval_for_a_run_with_no_eval_rows_is_200_with_an_empty_list(db_path):
    _add_run(db_path, _FIRST)

    response = _client().get(f"/api/runs/{_FIRST}/eval")

    assert response.status_code == 200
    assert response.json() == {"run_id": _FIRST, "eval_results": []}


def test_delete_of_an_unknown_id_is_404_with_the_detail_of_get(db_path):
    _add_run(db_path, _FIRST)
    client = _client()

    response = client.delete("/api/runs/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}
    assert response.json() == client.get("/api/runs/nope").json()
    assert _rows(db_path, "runs", _FIRST) == 1


def test_delete_that_finds_the_run_but_removes_no_row_is_404(db_path, monkeypatch):
    """A run that another request removes between the lookup and the delete."""
    from soup_cli.experiment.tracker import ExperimentTracker

    _add_run(db_path, _FIRST)
    monkeypatch.setattr(ExperimentTracker, "delete_run", lambda self, run_id: False)

    response = _client().delete(f"/api/runs/{_FIRST}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Run not found"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/runs/run_20261121"),
        ("DELETE", "/api/runs/run_20261121"),
        ("GET", "/api/runs/run_20261121/eval"),
    ],
)
def test_an_ambiguous_prefix_is_404_on_all_three_and_deletes_nothing(db_path, method, path):
    _add_run(db_path, _FIRST, evals=1, metrics=1)
    _add_run(db_path, _SECOND, evals=1, metrics=1)

    response = _client().request(method, path)

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Run id prefix 'run_20261121' matches 2 runs; give more characters"
    }
    for run_id in (_FIRST, _SECOND):
        assert _rows(db_path, "runs", run_id) == 1
        assert _rows(db_path, "metrics", run_id) == 1
        assert _rows(db_path, "eval_results", run_id) == 1


def test_the_exact_id_still_works_on_all_three(db_path):
    _add_run(db_path, _FIRST, evals=1)
    _add_run(db_path, _SECOND, evals=1)
    client = _client()

    shown = client.get(f"/api/runs/{_FIRST}")
    assert shown.status_code == 200
    assert shown.json()["run_id"] == _FIRST

    evals = client.get(f"/api/runs/{_FIRST}/eval")
    assert evals.status_code == 200
    assert evals.json()["run_id"] == _FIRST
    assert len(evals.json()["eval_results"]) == 1

    deleted = client.delete(f"/api/runs/{_FIRST}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "run_id": _FIRST}
    assert _rows(db_path, "runs", _FIRST) == 0
    assert _rows(db_path, "runs", _SECOND) == 1
