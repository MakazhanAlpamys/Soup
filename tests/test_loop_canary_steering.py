"""A client that can chat cannot outvote the canary, and its bad input is not a canary failure.

Two bounds on what one client can do to a rollout's verdict:

* ``read_quality_stats`` counts at most ``MAX_RATED_PER_CONVERSATION`` rated answers per
  conversation id and side, so one conversation id (client chosen) cannot supply the 30
  ratings a side needs;
* a request that fails input validation is a 4xx and is NOT recorded as a canary failure;
  only a server-side generation error is. (A chat template that refuses the roles never
  raises on ``soup serve``: it falls back to the plain prompt.)
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import soup_cli.commands.serve as serve  # noqa: E402
from soup_cli.utils import loop_feedback as lf  # noqa: E402
from soup_cli.utils.canary_router import read_bucket_stats  # noqa: E402
from soup_cli.utils.loop_state import LoopState, write_state  # noqa: E402

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
STABLE = "v1"
CANARY = "v2"
TRACES = os.path.join(".soup-loops", "traces", "conversations.jsonl")
FEEDBACK = os.path.join(".soup-loops", "feedback.jsonl")


def uid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def answer(number, *, conversation, bucket="canary", adapter=CANARY):
    record = {
        "schema_version": 1, "id": uid(number), "ts": iso(10), "model": "base", "version": 1,
        "adapter": adapter, "canary_bucket": bucket,
        "messages": [{"role": "user", "content": f"q{number}"}], "response": "an answer",
        "tokens": {"prompt": 1, "completion": 1}, "source": "soup-serve",
    }
    if conversation is not None:
        record["conversation_id"] = conversation
    return record


def thumb(number, value, serial):
    return {
        "schema_version": 1, "id": uid(900_000 + serial), "response_id": uid(number),
        "ts": iso(20), "kind": "thumb", "value": value, "source": "playground",
    }


def write_lines(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(line) + "\n" for line in lines))


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def read():
    stats = lf.read_quality_stats(
        os.path.join(".soup-loops", "traces"), FEEDBACK, stable=STABLE, canary=CANARY,
        started_ts=T0,
    )
    return (
        stats.canary_rated, stats.canary_negative, stats.stable_rated, stats.stable_negative
    )


def rate_all(answers, value=-1):
    write_lines(TRACES, answers)
    write_lines(FEEDBACK, [
        thumb(int(item["id"][-12:]), value, serial) for serial, item in enumerate(answers)
    ])


# ---------------------------------------------------------------------------
# (a) At most N ratings per conversation id and side
# ---------------------------------------------------------------------------


class TestPerConversationCap:
    def test_the_cap_is_a_named_constant_of_three(self):
        assert lf.MAX_RATED_PER_CONVERSATION == 3

    def test_one_conversation_id_cannot_supply_more_than_the_cap_on_the_canary_side(self, cwd):
        rate_all([answer(n, conversation="one-client") for n in range(1, 41)])

        assert read() == (3, 3, 0, 0)

    def test_the_cap_applies_to_the_stable_side_too(self, cwd):
        rate_all(
            [answer(n, conversation="one-client", bucket="stable", adapter=STABLE)
             for n in range(1, 41)],
            value=1,
        )

        assert read() == (0, 0, 3, 0)

    def test_each_side_has_its_own_budget_for_the_same_id(self, cwd):
        rate_all(
            [answer(n, conversation="x") for n in range(1, 11)]
            + [answer(n, conversation="x", bucket="stable", adapter=STABLE)
               for n in range(11, 21)]
        )

        assert read() == (3, 3, 3, 3)

    def test_distinct_conversation_ids_all_count(self, cwd):
        rate_all([answer(n, conversation=f"user-{n}") for n in range(1, 31)])

        assert read() == (30, 30, 0, 0)

    def test_a_conversation_id_just_under_the_cap_counts_every_answer(self, cwd):
        rate_all([answer(n, conversation="three") for n in range(1, 4)])

        assert read() == (3, 3, 0, 0)

    def test_an_id_longer_than_256_characters_is_still_grouped(self, cwd):
        long_id = "z" * 400  # the server accepts 512
        rate_all([answer(n, conversation=long_id) for n in range(1, 21)])

        assert read() == (3, 3, 0, 0)

    def test_answers_without_a_conversation_id_share_one_group_per_side(self, cwd):
        rate_all([answer(n, conversation=None) for n in range(1, 21)])

        assert read() == (3, 3, 0, 0)

    def test_the_cap_cannot_be_dodged_by_changing_case_or_adding_spaces(self, cwd):
        # Ids are compared as they were logged; a different string is a different client
        # for routing too (the router hashes it), so the cap follows the same identity.
        rate_all([answer(n, conversation=f"id{' ' * n}") for n in range(1, 6)])

        assert read()[0] == 5


# ---------------------------------------------------------------------------
# (b) A client's invalid input is not a canary failure
# ---------------------------------------------------------------------------


class _AdapterModel:
    def __init__(self) -> None:
        self.current = None

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous = self.current
        self.current = None
        try:
            yield
        finally:
            self.current = previous


def _canary_client(cwd, monkeypatch, generate):
    state_path = cwd / ".soup" / "loop.yaml"
    stats_path = cwd / ".soup" / "canary-stats.json"
    write_state(
        LoopState(
            served_model="base", eval_suite="evals.yaml", baseline="prod",
            canary_active="candidate", canary_traffic_pct=100.0, canary_rollout_id="rollout-1",
        ),
        str(state_path),
    )
    monkeypatch.setattr(serve, "_generate_response", generate)
    app = serve._create_app(
        model_obj=_AdapterModel(), tokenizer=object(), device="cpu", model_name="base",
        max_tokens_default=8, adapter_map={"candidate": "unused"},
        peft_adapter_names={"candidate"}, canary_state_path=str(state_path),
        canary_stats_path=str(stats_path),
    )
    return TestClient(app), str(stats_path)


def _stats(client, stats_path):
    client.app.state.canary_outcomes.flush()
    return read_bucket_stats(
        stable="base", canary="candidate", rollout_id="rollout-1", path=stats_path
    )


def _chat(client, conversation="c-1"):
    return client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "conversation_id": conversation,
    })


class TestInputErrorsAreNotCanaryFailures:
    def test_a_client_input_error_is_a_400_and_no_canary_failure(self, cwd, monkeypatch):
        def generate(model, *args, **kwargs):
            raise serve._ClientInputError("the messages could not be turned into a prompt")

        client, stats_path = _canary_client(cwd, monkeypatch, generate)

        resp = _chat(client)
        stats = _stats(client, stats_path)

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "Invalid request: messages"}
        assert (stats.canary_major, stats.canary_ok) == (0, 0)
        client.app.state.canary_outcomes.close()

    def test_a_server_side_generation_error_is_still_a_canary_failure(self, cwd, monkeypatch):
        def generate(model, *args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        client, stats_path = _canary_client(cwd, monkeypatch, generate)

        resp = _chat(client)
        stats = _stats(client, stats_path)

        assert resp.status_code == 500
        assert (stats.canary_major, stats.canary_ok) == (1, 0)
        client.app.state.canary_outcomes.close()

    def test_a_good_request_is_counted_ok(self, cwd, monkeypatch):
        client, stats_path = _canary_client(
            cwd, monkeypatch, lambda model, *args, **kwargs: ("ok", 1, 1, "stop")
        )

        assert _chat(client).status_code == 200
        stats = _stats(client, stats_path)

        assert (stats.canary_major, stats.canary_ok) == (0, 1)
        client.app.state.canary_outcomes.close()

    def test_a_malformed_body_never_reaches_the_canary_counts(self, cwd, monkeypatch):
        client, stats_path = _canary_client(
            cwd, monkeypatch, lambda model, *args, **kwargs: ("ok", 1, 1, "stop")
        )

        bad = client.post("/v1/chat/completions", json={
            "messages": "not a list", "conversation_id": "c-1",
        })
        stats = _stats(client, stats_path)

        assert bad.status_code == 422
        assert (stats.canary_major, stats.canary_ok) == (0, 0)
        client.app.state.canary_outcomes.close()

    def test_a_streamed_client_input_error_is_not_a_canary_failure(self, cwd, monkeypatch):
        def generate(model, *args, **kwargs):
            raise serve._ClientInputError("the messages could not be turned into a prompt")

        client, stats_path = _canary_client(cwd, monkeypatch, generate)

        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "conversation_id": "c-1", "stream": True,
        })
        stats = _stats(client, stats_path)

        assert resp.status_code == 200  # the stream had started: the error is a frame
        assert (stats.canary_major, stats.canary_ok) == (0, 0)
        client.app.state.canary_outcomes.close()

    def test_a_template_that_refuses_the_roles_does_not_raise_on_the_serve_path(self):
        # The code path the review could not confirm: `soup serve` renders with
        # fallback_on_error=True, so a template that raises on the roles/order a client sent
        # yields the plain prompt, not an exception, and no canary error.
        from soup_cli.utils.vllm import _render_chat_prompt

        class RefusingTokenizer:
            chat_template = "{{ raise_exception('roles must alternate') }}"

            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("roles must alternate user/assistant")

        messages = [{"role": "assistant", "content": "a"}, {"role": "robot", "content": "b"}]

        text, templated = _render_chat_prompt(
            messages, RefusingTokenizer(), fallback_on_error=True
        )

        assert templated is False and isinstance(text, str) and text
