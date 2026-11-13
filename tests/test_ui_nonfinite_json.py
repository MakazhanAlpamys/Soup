"""Web UI routes send a non-finite float as null instead of answering 500.

Starlette renders JSON with ``allow_nan=False``. The tracker keeps an infinite
grad_norm (an fp16 overflow step), loss or eval score as logged, so every route
that returned such a row raised inside the response and answered 500; the run
detail view of that run showed only "Error: Internal Server Error" (#2750).
"""

import json

import pytest

from tests.ui_helpers import ui_client

pytest.importorskip("fastapi")

RUN_ID = "run_overflow"
INF = float("inf")


def _strict_json(text: str):
    """Parse like a browser's JSON.parse: NaN / Infinity tokens are refused."""

    def _refuse(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=_refuse)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from soup_cli.experiment.tracker import ExperimentTracker

    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(db_path=path)
    try:
        tracker.start_run({"base": "m", "task": "sft"}, "cuda", "Tesla T4", {}, run_id=RUN_ID)
        tracker.log_metrics(RUN_ID, step=10, loss=2.31, lr=2e-4, grad_norm=1.7)
        tracker.log_metrics(RUN_ID, step=20, loss=-INF, lr=2e-4, grad_norm=INF)
        tracker.save_eval_result("out", "mmlu", INF, {}, run_id=RUN_ID)
        tracker.finish_run(
            RUN_ID, initial_loss=2.31, final_loss=INF, total_steps=20,
            duration_secs=60.0, output_dir="out",
        )
    finally:
        tracker.close()
    return path


def _client():
    from soup_cli.ui.app import create_app, get_auth_token

    client = ui_client(create_app())
    client.headers["Authorization"] = f"Bearer {get_auth_token()}"
    return client


@pytest.mark.parametrize(
    "path",
    [
        "/api/runs",
        f"/api/runs/{RUN_ID}",
        f"/api/runs/{RUN_ID}/metrics",
        f"/api/runs/{RUN_ID}/eval",
        f"/api/runs/compare?ids={RUN_ID}",
    ],
)
def test_run_routes_answer_200_with_strict_json(db, path):
    response = _client().get(path)

    assert response.status_code == 200
    _strict_json(response.text)


def test_metrics_send_infinite_values_as_null_and_keep_finite_ones(db):
    metrics = _client().get(f"/api/runs/{RUN_ID}/metrics").json()["metrics"]

    assert [(m["step"], m["loss"], m["grad_norm"]) for m in metrics] == [
        (10, 2.31, 1.7),
        (20, None, None),
    ]


def test_run_and_eval_send_infinite_values_as_null(db):
    client = _client()

    run = client.get(f"/api/runs/{RUN_ID}").json()
    assert (run["initial_loss"], run["final_loss"]) == (2.31, None)
    results = client.get(f"/api/runs/{RUN_ID}/eval").json()["eval_results"]
    assert [r["score"] for r in results] == [None]


def test_data_inspect_sample_with_nan_is_sent_as_null(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rows.jsonl").write_text(
        '{"instruction": "hi", "output": "there", "score": NaN}\n', encoding="utf-8"
    )

    response = _client().post("/api/data/inspect", json={"path": "rows.jsonl"})

    assert response.status_code == 200
    body = _strict_json(response.text)
    assert body["sample"] == [{"instruction": "hi", "output": "there", "score": None}]


def test_live_metrics_stream_sends_strict_json(db, monkeypatch):
    import soup_cli.ui.app as ui_mod

    monkeypatch.setattr(ui_mod, "_train_process", None)
    with _client().stream("GET", f"/api/train/metrics/live?run_id={RUN_ID}") as response:
        text = "".join(response.iter_text())

    # every event is parsed strictly, the closing ``event: done`` frame included
    events = [
        _strict_json(line[len("data: "):])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == {}
    rows = [event for event in events if "step" in event]
    assert [(r["step"], r["grad_norm"]) for r in rows] == [(10, 1.7), (20, None)]


@pytest.mark.parametrize(
    "value, expected",
    [
        (float("nan"), None),
        (INF, None),
        (-INF, None),
        (1.5, 1.5),
        (3, 3),
        (True, True),
        (None, None),
        ("NaN", "NaN"),
        ((INF, 2.0), [None, 2.0]),
        ([{"a": [1.5, float("nan"), (-INF, "x")]}], [{"a": [1.5, None, [None, "x"]]}]),
    ],
    ids=["nan", "inf", "-inf", "finite", "int", "bool", "none", "string", "tuple", "nested"],
)
def test_finite_json_replaces_only_non_finite_floats(value, expected):
    from soup_cli.utils.finite_json import finite_json

    result = finite_json(value)
    assert result == expected
    assert type(result) is type(expected)


def test_train_stream_frame_sends_non_finite_as_null():
    from soup_cli.utils.sse_train_stream import TrainEvent, format_sse_frame

    frame = format_sse_frame(
        TrainEvent(
            type="metric", step=20, loss=INF, grad_norm=float("nan"), val_loss=-INF, lr=2e-4
        )
    )

    assert frame.startswith("data: ") and frame.endswith("\n\n")
    body = _strict_json(frame[len("data: "):].strip())
    assert (body["loss"], body["grad_norm"], body["val_loss"]) == (None, None, None)
    assert (body["step"], body["lr"]) == (20, 2e-4)


def _chat_send(tmp_path, monkeypatch, temperature: str):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "experiments.db"))
    body = (
        '{"messages": [{"role": "user", "content": "hi"}], '
        f'"endpoint": "http://127.0.0.1:9", "temperature": {temperature}}}'
    )
    return _client().post(
        "/api/chat/send", content=body, headers={"Content-Type": "application/json"}
    )


def test_validation_error_echoing_nan_answers_422_with_strict_json(tmp_path, monkeypatch):
    # Python's JSON reader accepts NaN, and the 422 body echoes the input back.
    response = _chat_send(tmp_path, monkeypatch, "NaN")

    assert response.status_code == 422
    detail = _strict_json(response.text)["detail"]
    assert [(error["loc"], error["input"]) for error in detail] == [
        (["body", "temperature"], None)
    ]


def test_validation_error_with_finite_input_keeps_fastapis_body(tmp_path, monkeypatch):
    response = _chat_send(tmp_path, monkeypatch, '"hot"')

    assert response.status_code == 422
    (error,) = _strict_json(response.text)["detail"]
    assert (error["type"], error["loc"], error["input"]) == (
        "float_parsing", ["body", "temperature"], "hot"
    )
