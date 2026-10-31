"""#2361 — the Transformers serve backend reports why generation stopped.

A run that uses the whole ``max_tokens`` budget without an end token was cut
off: OpenAI ``finish_reason`` is ``length`` and Anthropic ``stop_reason`` is
``max_tokens``. A run that ends on the end token, even one landing exactly on
the budget, stopped naturally: ``stop`` / ``end_turn``. Every case runs real
greedy generation on a tiny in-memory Llama, through both protocols, in the
reply and in the last stream frame.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("fastapi")

BUDGET = 8

CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hi": 3, "user": 4, "assistant": 5}
    vocab.update({f"w{i}": i for i in range(6, 64)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant"
    )
    return tokenizer


@pytest.fixture(scope="module")
def served():
    """A tiny random Llama behind the serve app, recording what it generates.

    ``/v1/messages`` does not forward ``temperature``, so that route samples;
    seeding before every call keeps each run's tokens repeatable.
    """
    import torch
    from fastapi.testclient import TestClient
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.commands.serve import _create_app

    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).eval()

    generated: list[list[int]] = []
    original = model.generate

    def recording_generate(**kwargs):
        torch.manual_seed(3)
        out = original(**kwargs)
        generated.append(out[0, kwargs["input_ids"].shape[1] :].tolist())
        return out

    model.generate = recording_generate
    app = _create_app(model, _tokenizer(), "cpu", "tiny", BUDGET)
    with TestClient(app, base_url="http://localhost") as client:
        yield client, model, generated


def _set_eos(model, eos_id: int) -> None:
    model.generation_config.eos_token_id = eos_id
    model.config.eos_token_id = eos_id


def _post(client, path: str, budget: int, stream: bool):
    response = client.post(
        path,
        json={
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": budget,
            "temperature": 0,
            "stream": stream,
        },
    )
    assert response.status_code == 200, response.text
    return response


def _reported(path: str, response, stream: bool) -> str:
    if not stream:
        body = response.json()
        if path == CHAT:
            return body["choices"][0]["finish_reason"]
        return body["stop_reason"]
    frames = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if path == CHAT:
        return frames[-1]["choices"][0]["finish_reason"]
    delta = [f for f in frames if f.get("type") == "message_delta"]
    assert len(delta) == 1
    return delta[0]["delta"]["stop_reason"]


def _first_token(client, model, generated, path: str) -> int:
    """The token greedy decoding produces first for this route's prompt."""
    _set_eos(model, 1)
    _post(client, path, BUDGET, stream=False)
    assert 1 not in generated[-1], "the end token must not appear in the cut-off run"
    return generated[-1][0]


_CUT_OFF = {CHAT: "length", MESSAGES: "max_tokens"}
_NATURAL = {CHAT: "stop", MESSAGES: "end_turn"}


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
@pytest.mark.parametrize("path", [CHAT, MESSAGES])
def test_cut_off_at_the_budget_is_reported(served, path, stream):
    client, model, generated = served
    _set_eos(model, 1)
    response = _post(client, path, BUDGET, stream)
    assert len(generated[-1]) == BUDGET
    assert 1 not in generated[-1]
    assert _reported(path, response, stream) == _CUT_OFF[path]


@pytest.mark.parametrize("budget", [BUDGET, 1], ids=["under-budget", "at-budget"])
@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
@pytest.mark.parametrize("path", [CHAT, MESSAGES])
def test_end_token_is_a_natural_stop(served, path, stream, budget):
    client, model, generated = served
    eos = _first_token(client, model, generated, path)
    _set_eos(model, eos)
    response = _post(client, path, budget, stream)
    assert generated[-1] == [eos]
    assert _reported(path, response, stream) == _NATURAL[path]


# ----- MoLE runtime: its own decode loop, the same rule -----


class _Out:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


def _constant_mole(next_token: int):
    """A MoLE runtime whose model always predicts ``next_token``."""
    import contextlib

    import torch
    from torch import nn

    from soup_cli.utils.mole_routing import (
        LoadedMole,
        MoleGatingConfig,
        build_gating_kernel,
    )

    class _Model(nn.Module):
        device = torch.device("cpu")

        def disable_adapter(self):
            return contextlib.nullcontext()

        def set_adapter(self, name):
            pass

        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            b, t = input_ids.shape
            logits = torch.zeros(b, t, 64)
            # Peaked hard enough that sampling picks it too: /v1/messages
            # does not forward temperature, so that route samples.
            logits[..., next_token] = 100.0
            hidden = [torch.ones(b, t, 4)] if output_hidden_states else None
            return _Out(logits, hidden)

    gate = build_gating_kernel(
        MoleGatingConfig(num_task_adapters=2, hidden_dim=4, temperature=1.0, top_k=1)
    )
    return LoadedMole(_Model(), _tokenizer(), gate, ["task_0", "task_1"])


@pytest.mark.parametrize(
    ("eos", "budget", "expected", "produced"),
    [
        (1, BUDGET, "length", BUDGET),
        (7, BUDGET, "stop", 1),
        (7, 1, "stop", 1),
    ],
    ids=["cut-off", "end-token-under-budget", "end-token-at-budget"],
)
def test_mole_generate_text_reports_the_stop_cause(eos, budget, expected, produced):
    mole = _constant_mole(next_token=7)
    mole.tokenizer.eos_token = mole.tokenizer.convert_ids_to_tokens(eos)
    _, _, completion_tokens, finish_reason = mole.generate_text(
        [{"role": "user", "content": "hi"}], max_tokens=budget
    )
    assert completion_tokens == produced
    assert finish_reason == expected


@pytest.mark.parametrize("stream", [False, True], ids=["reply", "stream"])
@pytest.mark.parametrize("path", [CHAT, MESSAGES])
def test_mole_cut_off_reaches_the_response(path, stream):
    from fastapi.testclient import TestClient

    from soup_cli.commands.serve import _create_app

    mole = _constant_mole(next_token=7)
    app = _create_app(mole.model, mole.tokenizer, "cpu", "tiny", BUDGET, mole_runtime=mole)
    with TestClient(app, base_url="http://localhost") as client:
        response = _post(client, path, BUDGET, stream)
    assert _reported(path, response, stream) == _CUT_OFF[path]
