"""The whole continual-learning loop, end to end, on CPU without a model.

serve -> users chat and give feedback -> an iteration harvests the feedback ->
trains -> soup ship verdict -> the adapter is loaded into the running server
and canaried on a share of keyed traffic -> the next cycle promotes it or rolls
it back -> soup loop rollback returns to the previous version.

Real pieces: the soup serve FastAPI app (routes, auth, canary router, outcome
counters, trace log, thumbs database, POST /v1/adapters/load), the loop daemon
and its pre-wired stages, loop.yaml, soup loop init / rollback. Stubbed: text
generation, the soup train and soup ship subprocesses, and PEFT itself. The
same cycle on a real model is ``tests/test_loop_end_to_end_smoke.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import soup_cli.commands.serve as serve
import soup_cli.utils.loop_stages as ls
from soup_cli.utils.loop_daemon import run_once, watch
from soup_cli.utils.loop_iteration import list_iterations, read_iteration
from soup_cli.utils.loop_state import read_state

TOKEN = "e2e-" + "t" * 28
ENDPOINT = "http://127.0.0.1:8000"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _clean(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


class _AdapterModel:
    """A PEFT-shaped stand-in: which adapter is selected, which are loaded."""

    def __init__(self) -> None:
        self.current = None
        self.loaded: set = set()

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous, self.current = self.current, None
        try:
            yield
        finally:
            self.current = previous


class _Done:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.stderr = b""


@pytest.fixture
def loop(tmp_path, monkeypatch):
    """A served app, a configured loop, and the stubbed subprocesses."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", ENDPOINT)
    monkeypatch.setenv(ls.TOOL_AUTH_TOKEN_ENV, TOKEN)
    for name in ("SOUP_LOOP_TRACE_DIR", "SOUP_LOOP_THUMBS_DB", "SOUP_LOOP_FEEDBACK_LOG"):
        monkeypatch.delenv(name, raising=False)
    for name in ("_DEPLOY_POSTER", "_LOAD_POSTER", "_TRACE_DIR_RESOLVER"):
        monkeypatch.setattr(ls, name, None)

    from soup_cli.monitoring.trace_logger import TraceLogWriter
    from soup_cli.utils import local_rl

    (tmp_path / "tasks.jsonl").write_text('{"prompt": "2+2", "expected": "4"}\n', "utf-8")
    os.makedirs(os.path.join(".soup-loops", "adapters"))
    local_rl.init_local_rl_db("thumbs.db")

    model = _AdapterModel()
    failing: set = set()
    served_by: list = []

    def generate(current_model, *args, **kwargs):
        served_by.append(current_model.current)
        if current_model.current in failing:
            raise RuntimeError("canary adapter broke")
        return f"answer from {current_model.current or 'base'}", 3, 4, "stop"

    def load(model_obj, path, name):
        assert os.path.isfile(os.path.join(path, "adapter_config.json"))
        model_obj.loaded.add(name)
        return model_obj

    monkeypatch.setattr(serve, "_generate_response", generate)
    monkeypatch.setattr(serve, "_load_runtime_adapter", load)
    app = serve._create_app(
        model_obj=model,
        tokenizer=object(),
        device="cpu",
        model_name="base",
        max_tokens_default=8,
        adapter_map={},
        peft_adapter_names=set(),
        trace_log_writer=TraceLogWriter(os.path.join(".soup-loops", "traces", "serve.jsonl")),
        record_thumbs_db="thumbs.db",
        auth_token=TOKEN,
        adapter_dir=os.path.realpath(os.path.join(".soup-loops", "adapters")),
    )
    client = TestClient(app, base_url=ENDPOINT)

    # The loop talks to the server over HTTP: route its httpx calls to the app.
    import httpx

    def post(url, headers=None, json=None, timeout=None, trust_env=None, **_):
        resp = client.post(urlparse(url).path, headers=headers or {}, json=json)
        return httpx.Response(resp.status_code, content=resp.content,
                              headers={"content-type": resp.headers.get("content-type", "")})

    monkeypatch.setattr(httpx, "post", post)

    trained_on: list = []
    trained_with: list = []

    def train(argv, **kwargs):
        config = yaml.safe_load(open(argv[argv.index("--config") + 1], encoding="utf-8"))
        assert config["base"] == "org/base"
        trained_with.append(config)
        with open(config["data"]["train"], encoding="utf-8") as fh:
            trained_on.append([json.loads(line) for line in fh if line.strip()])
        os.makedirs(config["output"], exist_ok=True)
        with open(os.path.join(config["output"], "adapter_config.json"), "w") as fh:
            json.dump({"base_model_name_or_path": config["base"]}, fh)
        return _Done()

    verdicts: list = []

    def ship(argv, **kwargs):
        verdicts.append(argv)
        with open(argv[argv.index("--output") + 1], "w", encoding="utf-8") as fh:
            json.dump({"decision": "SHIP"}, fh)
        return _Done()

    monkeypatch.setattr(ls, "_TRAIN_RUNNER", train)
    monkeypatch.setattr(ls, "_SHIP_RUNNER", ship)

    from soup_cli.commands.loop import app as loop_app

    def cli(*args):
        result = CliRunner().invoke(loop_app, list(args))
        return result, _clean(result.output)

    result, _ = cli(
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired",
        "--base-model", "org/base", "--thumbs-db", "thumbs.db", "--canary-traffic", "20%",
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))

    def cycle():
        config = ls.build_prewired_watch_config(max_iterations=1, poll_interval_sec=1.0)
        # No experiment tracker in a test: the budget estimate is 0.
        return watch(dataclasses.replace(config, cost_fn=lambda state: 0.0))

    headers = {"Authorization": f"Bearer {TOKEN}"}

    def chat(content, conversation_id=None):
        body = {"messages": [{"role": "user", "content": content}]}
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        return client.post("/v1/chat/completions", json=body)

    def users_give_feedback():
        """Two answers to one prompt; the user likes one and dislikes the other."""
        for thumb in ("up", "down"):
            answer = chat("How do I reset my router?").json()["choices"][0]["message"]["content"]
            response = client.post(
                "/v1/thumbs", headers=headers,
                json={"prompt": "How do I reset my router?",
                      "response": f"{answer} ({thumb})", "thumb": thumb},
            )
            assert response.status_code == 200, response.text

    def keyed_traffic(count=300):
        for index in range(count):
            chat("hello", conversation_id=f"user-{index}")
        app.state.canary_outcomes.flush()

    return {
        "client": client, "model": model, "failing": failing, "served_by": served_by,
        "cycle": cycle, "cli": cli, "headers": headers, "chat": chat,
        "users_give_feedback": users_give_feedback, "keyed_traffic": keyed_traffic,
        "verdicts": verdicts, "trained_on": trained_on, "trained_with": trained_with,
    }


def _same_path(left, right) -> bool:
    return os.path.normpath(left) == os.path.normpath(right)


def _records() -> list:
    return [read_iteration(iteration_id) for iteration_id in list_iterations()]


def _active_adapter(client) -> str | None:
    return client.get("/health").json()["active_adapter"]


def _start_canary(loop) -> str:
    """serve -> feedback -> one iteration -> SHIP -> loaded and canaried."""
    loop["users_give_feedback"]()
    _, iterations = loop["cycle"]()
    assert iterations == 1
    state = read_state()
    name = state.canary_active
    assert name and name.startswith("loop-train-"), state
    assert state.canary_traffic_pct == 20.0
    assert _same_path(state.canary_adapter, os.path.join(".soup-loops", "adapters", name))
    record = read_iteration(list_iterations()[-1])
    assert record.pairs_harvested == 1
    assert record.gate_verdict == "OK" and record.shipped is True
    assert f"canary {name}: 20% of keyed traffic" in record.notes
    # The new adapter is loaded in the running server but not active for everyone.
    listed = loop["client"].get("/v1/adapters", headers=loop["headers"]).json()
    assert name in {entry["name"] for entry in listed["adapters"]}
    assert name in loop["model"].loaded
    assert _active_adapter(loop["client"]) is None
    # The verdict compared the candidate against production (the base model).
    argv = loop["verdicts"][-1]
    assert "--base-adapter" not in argv and argv[argv.index("--base") + 1] == "org/base"
    return name


def test_feedback_reaches_a_canary_which_is_promoted_then_rolled_back(loop):
    name = _start_canary(loop)
    assert os.path.getsize(os.path.join(".soup-loops", "traces", "serve.jsonl")) > 0

    loop["served_by"].clear()
    loop["keyed_traffic"]()
    canary_share = loop["served_by"].count(name) / len(loop["served_by"])
    assert 0.1 < canary_share < 0.3, canary_share
    # Unkeyed traffic stays on production while the canary runs.
    loop["served_by"].clear()
    loop["chat"]("no key")
    assert loop["served_by"] == [None]

    # Next cycle: 30+ healthy canary answers -> promoted, unkeyed traffic switched.
    # Training is on, but no feedback arrived since the last run: it trains nothing and
    # no second canary starts.
    loop["cycle"]()
    state = read_state()
    assert state.canary_active is None
    assert len(loop["trained_on"]) == 1 and len(loop["verdicts"]) == 1
    assert state.production_name == name
    assert _same_path(state.production_adapter, os.path.join(".soup-loops", "adapters", name))
    assert state.previous_production_name is None
    assert _active_adapter(loop["client"]) == name
    # (Found by content: two iterations in one second sort by a random suffix.)
    promoted = [r for r in _records() if "promoted to production" in r.notes]
    assert len(promoted) == 1
    assert "not_enough_new_data: 0 new preference pairs" in promoted[0].notes

    # The person says no: back to the previous version (the served base model).
    result, text = loop["cli"]("rollback")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"rolled back: {name} -> the served model" in text
    assert read_state().production_name is None
    assert _active_adapter(loop["client"]) is None


def test_a_regressing_canary_is_rolled_back_automatically(loop):
    name = _start_canary(loop)
    loop["failing"].add(name)
    loop["keyed_traffic"]()

    loop["cycle"]()  # rolls the canary back; no new feedback, so no new training
    state = read_state()
    assert state.canary_active is None and state.canary_adapter is None
    assert state.production_name is None
    assert len(loop["trained_on"]) == 1 and len(loop["verdicts"]) == 1
    assert _active_adapter(loop["client"]) is None
    record = read_iteration(list_iterations()[-1])
    assert record.rolled_back is True
    assert f"canary {name} rolled back (MAJOR)" in record.notes


def test_a_second_iteration_compares_against_the_promoted_version(loop):
    first = _start_canary(loop)
    loop["keyed_traffic"]()
    loop["cycle"]()  # promote
    assert read_state().production_name == first

    loop["users_give_feedback"]()
    loop["cycle"]()
    state = read_state()
    second = state.canary_active
    assert second and second != first
    argv = loop["verdicts"][-1]
    assert _same_path(
        argv[argv.index("--base-adapter") + 1], os.path.join(".soup-loops", "adapters", first)
    )

    # Promote the second, then roll back: the first version comes back.
    loop["keyed_traffic"]()
    loop["cycle"]()
    assert read_state().production_name == second
    assert read_state().previous_production_name == first
    result, text = loop["cli"]("rollback")
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"rolled back: {second} -> {first}" in text
    assert _active_adapter(loop["client"]) == first


def test_no_new_iteration_while_a_canary_collects_answers(loop):
    name = _start_canary(loop)
    loop["users_give_feedback"]()
    before = len(list_iterations())
    loop["cycle"]()
    assert read_state().canary_active == name
    assert len(list_iterations()) == before


def test_feedback_on_another_prompt_retrains_on_everything_so_far(loop):
    first = _start_canary(loop)
    loop["keyed_traffic"]()
    loop["cycle"]()  # promote; nothing new, nothing trained
    assert read_state().production_name == first
    assert len(loop["trained_on"]) == 1

    # One new pair on a prompt the loop has not seen: the next run trains a fresh LoRA
    # from the base on the old pair AND the new one, and canaries it.
    for thumb, answer in (("up", "Hold the reset button."), ("down", "Unplug it.")):
        response = loop["client"].post(
            "/v1/thumbs", headers=loop["headers"],
            json={"prompt": "How do I update the firmware?", "response": answer, "thumb": thumb},
        )
        assert response.status_code == 200, response.text
    loop["cycle"]()

    second = read_state().canary_active
    assert second and second != first
    prompts = sorted(row["prompt"] for row in loop["trained_on"][-1])
    assert prompts == ["How do I reset my router?", "How do I update the firmware?"]
    (record,) = [r for r in _records() if r.run_id == second]
    assert record.pairs_harvested == 2
    # And once that is promoted too, a quiet cycle starts nothing again.
    loop["keyed_traffic"]()
    loop["cycle"]()
    assert read_state().production_name == second
    assert len(loop["trained_on"]) == 2


# ---------------------------------------------------------------------------
# Feedback in the loop's data formats: a correction trains a kto or an sft loop
# ---------------------------------------------------------------------------

_ROUTER = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "How do I reset my router?"},
]
_HELLO = [{"role": "user", "content": "hello"}]
_FIX = "Hold the reset button for ten seconds."


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def _jsonl(path: str, lines: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(line) + "\n" for line in lines))


def _answer(number: int, messages: list, response: str) -> dict:
    """A conversation record (version 1) as an application writes it."""
    return {
        "schema_version": 1, "id": _uuid(number), "conversation_id": f"chat-{number}",
        "ts": "2026-11-05T10:00:00+00:00", "model": "base", "version": 1, "adapter": None,
        "canary_bucket": None, "messages": messages, "response": response,
        "finish_reason": "stop", "latency_ms": 12.0,
        "tokens": {"prompt": 5, "completion": 3}, "source": "playground",
    }


def _feedback(number: int, kind: str, ts: str, **fields) -> dict:
    return {
        "schema_version": 1, "id": _uuid(900 + number), "response_id": _uuid(number),
        "ts": ts, "kind": kind, "source": "playground", **fields,
    }


@pytest.mark.parametrize("method", ["kto", "sft"])
def test_a_correction_given_as_feedback_trains_the_chosen_method_and_reaches_a_canary(
    loop, method
):
    feedback_log = (
        os.path.join("app-data", "feedback.jsonl")
        if method == "sft"
        else os.path.join(".soup-loops", "feedback.jsonl")
    )
    flags = ["--train-method", method]
    if method == "sft":
        flags += ["--feedback-log", feedback_log]
    result, _ = loop["cli"](
        "init", "./base", "--eval", "tasks.jsonl", "--baseline", "b", "--pre-wired",
        "--base-model", "org/base", "--thumbs-db", "thumbs.db", "--canary-traffic", "20%",
        "--force", *flags,
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    result, text = loop["cli"]("status")
    assert re.search(rf"train_method\W+{method}", text), text

    # The person corrected one answer and liked another, in the formats an application
    # (or `soup serve --feedback-log`) writes; the thumbs database stays empty.
    served = loop["chat"]("hello").json()["choices"][0]["message"]["content"]
    _jsonl(os.path.join(".soup-loops", "traces", "conversations.jsonl"), [
        _answer(1, _ROUTER, "Unplug it."), _answer(2, _HELLO, served),
    ])
    _jsonl(feedback_log, [
        _feedback(1, "edit", "2026-11-05T10:05:00+00:00", edited_text=_FIX),
        _feedback(2, "thumb", "2026-11-05T10:06:00+00:00", value=1),
    ])

    _, iterations = loop["cycle"]()
    assert iterations == 1

    (config,) = loop["trained_with"]
    (rows,) = loop["trained_on"]
    assert config["task"] == method
    assert config["data"]["val_split"] == 0
    if method == "kto":
        assert config["data"]["format"] == "kto"
        assert config["training"]["batch_size"] == 2  # TRL refuses a KTO batch of 1
        assert rows == [
            {"prompt": _ROUTER, "completion": _FIX, "label": True, "source": "feedback"},
            {"prompt": _ROUTER, "completion": "Unplug it.", "label": False,
             "source": "feedback"},
            {"prompt": _HELLO, "completion": served, "label": True, "source": "feedback"},
        ]
    else:
        assert config["data"]["format"] == "chatml"
        assert rows == [{
            "messages": [*_ROUTER, {"role": "assistant", "content": _FIX}],
            "source": "feedback-edit",
        }]
    state = read_state()
    name = state.canary_active
    assert name and name.startswith("loop-train-"), state
    assert state.harvest_cursor_ts is not None
    record = read_iteration(list_iterations()[-1])
    assert record.pairs_harvested == len(rows)
    assert record.gate_verdict == "OK" and record.shipped is True
    assert name in loop["model"].loaded

    # Every request is answered, but a feedback log exists and nobody has rated the canary's
    # answers yet: it waits for ratings (the quality verdict, below) instead of being promoted.
    loop["keyed_traffic"]()
    loop["cycle"]()
    state = read_state()
    assert state.canary_active == name and state.production_name is None
    assert len(loop["trained_on"]) == 1 and len(loop["verdicts"]) == 1


# ---------------------------------------------------------------------------
# A canary is judged by what people say about its answers, not only by errors
# ---------------------------------------------------------------------------


def _log_keyed_answers(loop, name) -> list:
    """Run keyed traffic and log each answer the way an application does (version 1).

    Returns ``(answer id, bucket)`` per answer: the canary's answers carry the canary
    adapter, the stable bucket's the base model (no adapter), both stamped after the
    rollout began.
    """
    loop["served_by"].clear()
    loop["keyed_traffic"]()
    now = datetime.now(timezone.utc).isoformat()
    logged = []
    lines = []
    for index, adapter in enumerate(loop["served_by"], start=1):
        bucket = "canary" if adapter == name else "stable"
        lines.append(dict(_answer(index, _HELLO, f"answer from {adapter or 'base'}"),
                          ts=now, adapter=adapter, canary_bucket=bucket))
        logged.append((index, bucket))
    _jsonl(os.path.join(".soup-loops", "traces", "quality.jsonl"), lines)
    return logged


def _rate(logged, *, canary, stable, limit=None, log=None):
    """Thumbs on the logged answers: ``canary`` / ``stable`` are the value (1, -1) or None."""
    path = log or os.path.join(".soup-loops", "feedback.jsonl")
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    given = {"canary": 0, "stable": 0}
    for index, bucket in logged:
        value = canary if bucket == "canary" else stable
        if value is None or (limit is not None and given[bucket] >= limit):
            continue
        given[bucket] += 1
        lines.append(_feedback(index, "thumb", now, value=value))
    _jsonl(path, lines)
    return given


def test_a_canary_people_thumb_down_is_withdrawn_though_no_request_failed(loop):
    name = _start_canary(loop)
    assert read_state().canary_started_ts is not None
    logged = _log_keyed_answers(loop, name)
    given = _rate(logged, canary=-1, stable=1)
    assert given["canary"] >= 30 and given["stable"] >= 30

    loop["cycle"]()

    state = read_state()
    assert state.canary_active is None and state.canary_adapter is None
    assert state.canary_started_ts is None
    assert state.production_name is None  # production never changed
    assert _active_adapter(loop["client"]) is None
    record = read_iteration(list_iterations()[-1])
    assert record.rolled_back is True and record.canary_verdict == "MAJOR"
    assert f"canary {name} rolled back (MAJOR)" in record.notes
    assert "canary errors: OK" in record.notes  # every request was answered
    assert (
        f"canary quality: {given['canary']}/{given['canary']} negative vs "
        f"0/{given['stable']} stable (MAJOR)"
    ) in record.notes
    assert len(loop["trained_on"]) == 1  # and nothing retrained on stale feedback


def test_a_canary_people_like_is_promoted(loop):
    name = _start_canary(loop)
    logged = _log_keyed_answers(loop, name)
    given = _rate(logged, canary=1, stable=1)

    loop["cycle"]()

    state = read_state()
    assert state.canary_active is None and state.canary_started_ts is None
    assert state.production_name == name
    assert _active_adapter(loop["client"]) == name
    promoted = [r for r in _records() if "promoted to production" in r.notes]
    assert len(promoted) == 1
    assert (
        f"canary quality: 0/{given['canary']} negative vs 0/{given['stable']} stable (OK)"
    ) in promoted[0].notes


def test_a_canary_waits_for_feedback_once_there_is_a_feedback_log(loop):
    name = _start_canary(loop)
    logged = _log_keyed_answers(loop, name)
    _rate(logged, canary=1, stable=1, limit=5)  # five thumbs per side: not enough
    before = len(list_iterations())

    loop["cycle"]()

    state = read_state()
    assert state.canary_active == name and state.production_name is None
    assert _active_adapter(loop["client"]) is None
    assert len(list_iterations()) == before  # a waiting canary starts no iteration
    # What the cycle decided, and said (a wait is not written as an iteration record):
    config = dataclasses.replace(
        ls.build_prewired_watch_config(max_iterations=1, poll_interval_sec=1.0),
        cost_fn=lambda current: 0.0,
    )
    _, waiting, decision = run_once(state, config)
    assert decision.proceed is False
    assert waiting.canary_verdict == "UNKNOWN" and waiting.rolled_back is False
    assert "canary errors: OK" in waiting.notes
    assert "canary quality: 5/30 canary, 5/30 stable rated (UNKNOWN)" in waiting.notes

    # More thumbs arrive: the next cycle decides.
    _rate(logged, canary=1, stable=1)
    loop["cycle"]()
    assert read_state().production_name == name
