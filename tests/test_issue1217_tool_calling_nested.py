"""#1217: `format: tool-calling` reads the calls where agent data puts them.

`soup agent synth` / `soup agent train` write each call inside its assistant
turn (the OpenAI shape), but the tool-calling converter only read top-level
`tools` / `tool_calls` keys, so every synthesised row was dropped and
`soup train --dry-run` still printed "Ready to train!". In rows it did accept,
the call was re-appended after the final answer and `tool_call_id` was lost.

The converter now keeps the calls on their own turns (the top-level list is
the legacy single-call form), `tools` is optional, `detect_format` classifies
`{messages, tools}` as tool-calling, and synth rows carry a `tools` schema.
The zero-row stop is covered in test_issue1217_zero_row_stop.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import DataConfig
from soup_cli.data.formats import (
    detect_format,
    format_to_messages,
    format_to_messages_with_reason,
)
from soup_cli.data.loader import load_dataset
from soup_cli.utils.agent_forge import (
    Endpoint,
    endpoint_to_rows,
    synthesise_dataset,
    write_dataset,
)
from tests.conftest import strip_ansi

_LOCAL_BASE_DIR: str | None = None


def _local_base_model_dir() -> str:
    """A tiny Llama causal LM + tokenizer built locally once per session (no
    download), standing in for hf-internal-testing/tiny-random-LlamaForCausalLM
    -- #1356 found this pulling it from the Hub for a dry-run plan and a
    tokenizer whose chat_template gets overwritten immediately after load."""
    global _LOCAL_BASE_DIR
    if _LOCAL_BASE_DIR is not None:
        return _LOCAL_BASE_DIR
    import tempfile

    from tokenizers import ByteLevelBPETokenizer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
        ["hi there friend", "hello world", "the quick brown fox"],
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<eos>"],
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<eos>", pad_token="<eos>")
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tok),
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    out_dir = tempfile.mkdtemp(prefix="soup-test-tiny-llama-")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    _LOCAL_BASE_DIR = out_dir
    return out_dir
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
_SCHEMA_TURN = {
    "role": "system",
    "content": (
        "You have access to the following tools. When a tool call is needed, "
        "respond with a function call in JSON.\n\n"
        "- get_weather: Weather for a city\n"
        '  parameters: {"type": "object", "properties": {"city": {"type": "string"}}}'
    ),
}


def _call(call_id: str, city: str, *, as_dict: bool = False) -> dict:
    arguments = {"city": city} if as_dict else json.dumps({"city": city})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }


def _normalized_call(call_id: str, city: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
    }


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_ansi(text)).strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _loaded_rows(path: str, data_format: str) -> list[dict]:
    """The rows `soup train` would train on."""
    config = DataConfig(train=path, format=data_format, val_split=0.0)
    return list(load_dataset(config)["train"])


# --- The synthesised dataset loads with the recipe `soup agent train` prints ---


_ENDPOINTS = [
    Endpoint(tool="listPets", method="get", path="/pets", description="List pets",
             parameters=(), spec_kind="openapi"),
    Endpoint(tool="getPet", method="get", path="/pets/{id}", description="Get a pet",
             parameters=("id", "fields"), spec_kind="openapi"),
]


@pytest.mark.parametrize("data_format", ["tool-calling", "auto"])
def test_every_synthesised_row_loads_with_its_tool_call(
    tmp_path, monkeypatch, data_format
):
    monkeypatch.chdir(tmp_path)
    rows = synthesise_dataset(_ENDPOINTS, examples_per_endpoint=3)
    write_dataset(rows, "agent_dataset.jsonl")

    loaded = _loaded_rows("agent_dataset.jsonl", data_format)

    assert len(loaded) == len(rows) == 6
    for source, got in zip(rows, loaded):
        calls = [m for m in got["messages"] if m.get("tool_calls")]
        assert len(calls) == 1
        assert calls[0]["tool_calls"][0]["function"]["name"] == source.tool
        assert got["messages"][0]["role"] == "system"
        assert f"- {source.tool}: " in got["messages"][0]["content"]


def test_the_printed_recipe_dry_runs_with_the_synthesised_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("openapi.yaml").write_text(
        "openapi: 3.0.0\ninfo: {title: pets, version: '1'}\npaths:\n"
        "  /pets:\n    get:\n      operationId: listPets\n      summary: List pets\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    base = Path(_local_base_model_dir()).as_posix()
    planned = runner.invoke(app, ["agent", "train", "--spec", "openapi.yaml", "--base", base])
    assert planned.exit_code == 0, planned.output
    assert "format: tool-calling" in planned.output
    Path("agent_train.yaml").write_text(
        f"base: {base}\ntask: sft\ndata:\n  train: agent_dataset.jsonl\n"
        "  format: tool-calling\n  val_split: 0.0\n"
        "training:\n  epochs: 3\n  lr: 2.0e-5\n  batch_size: auto\n"
        "output: ./agent_train_output\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["train", "--config", "agent_train.yaml", "--dry-run"])

    output = _plain(result.output)
    assert result.exit_code == 0, output
    assert "Data OK: 4 train samples" in output
    assert "dropped" not in output


def test_a_synth_row_carries_the_schema_of_its_endpoint():
    row = endpoint_to_rows(_ENDPOINTS[1], 1)[0].to_dict()

    assert row["tools"] == [{
        "type": "function",
        "function": {
            "name": "getPet",
            "description": "Get a pet",
            "parameters": {"type": "object", "properties": {"id": {}, "fields": {}}},
        },
    }]
    assert detect_format([row]) == "tool-calling"


# --- Each row shape, converted exactly --------------------------------------


def test_call_nested_in_the_assistant_turn_without_top_level_keys():
    row = {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "tool_calls": [_call("call_0", "Paris")]},
    ]}

    assert format_to_messages(row, "tool-calling") == {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_normalized_call("call_0", "Paris")]},
    ]}


def test_the_openai_fine_tuning_shape():
    row = {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [_call("call_0", "Paris")]},
            {"role": "tool", "tool_call_id": "call_0", "content": '{"temp": 21}'},
            {"role": "assistant", "content": "It is 21 C in Paris."},
        ],
        "tools": [_WEATHER_TOOL],
    }

    assert format_to_messages(row, "tool-calling") == {"messages": [
        _SCHEMA_TURN,
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_normalized_call("call_0", "Paris")]},
        {"role": "tool", "content": '{"temp": 21}', "tool_call_id": "call_0"},
        {"role": "assistant", "content": "It is 21 C in Paris."},
    ]}


def test_the_documented_top_level_shape_still_converts():
    row = {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [_WEATHER_TOOL],
        "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
    }

    assert format_to_messages(row, "tool-calling") == {"messages": [
        _SCHEMA_TURN,
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
        ]},
    ]}


def test_two_tool_rounds_keep_source_order_and_their_ids():
    messages = [
        {"role": "user", "content": "Paris, then Rome?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_0", "Paris")]},
        {"role": "tool", "tool_call_id": "call_0", "content": "21"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_1", "Rome")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "25"},
        {"role": "assistant", "content": "Paris 21 C, Rome 25 C."},
    ]
    # The top-level list repeats the calls, as the issue's documented-shape
    # row does; it must not be appended after the final answer again.
    row = {
        "messages": messages,
        "tools": [_WEATHER_TOOL],
        "tool_calls": [_call("call_0", "Paris")],
    }

    converted = format_to_messages(row, "tool-calling")["messages"]

    assert [m["role"] for m in converted] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert converted[2]["tool_calls"] == [_normalized_call("call_0", "Paris")]
    assert converted[3]["tool_call_id"] == "call_0"
    assert converted[4]["tool_calls"] == [_normalized_call("call_1", "Rome")]
    assert converted[5]["tool_call_id"] == "call_1"
    assert converted[6] == {"role": "assistant", "content": "Paris 21 C, Rome 25 C."}


def test_two_parallel_calls_in_one_assistant_turn():
    row = {
        "messages": [
            {"role": "user", "content": "Paris and Rome?"},
            {"role": "assistant", "content": "",
             "tool_calls": [_call("call_0", "Paris"), _call("call_1", "Rome")]},
            {"role": "tool", "tool_call_id": "call_0", "content": "21"},
            {"role": "tool", "tool_call_id": "call_1", "content": "25"},
            {"role": "assistant", "content": "21 and 25."},
        ],
        "tools": [_WEATHER_TOOL],
    }

    converted = format_to_messages(row, "tool-calling")["messages"]

    assert converted[2]["tool_calls"] == [
        _normalized_call("call_0", "Paris"), _normalized_call("call_1", "Rome"),
    ]
    assert [m.get("tool_call_id") for m in converted[3:5]] == ["call_0", "call_1"]
    assert converted[-1] == {"role": "assistant", "content": "21 and 25."}


@pytest.mark.parametrize("as_dict", [False, True], ids=["json-string", "dict"])
def test_arguments_as_a_json_string_or_a_dict(as_dict):
    row = {"messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c", "Paris", as_dict=as_dict)]},
    ]}

    converted = format_to_messages(row, "tool-calling")["messages"]

    assert converted[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'


@pytest.mark.parametrize("tools", [None, []], ids=["absent", "empty"])
def test_without_tools_there_is_no_schema_turn(tools):
    row = {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c", "Paris")]},
    ]}
    if tools is not None:
        row["tools"] = tools

    converted = format_to_messages(row, "tool-calling")["messages"]

    assert converted[0] == {"role": "system", "content": "Be brief."}
    assert [m["role"] for m in converted] == ["system", "user", "assistant"]


def test_with_tools_the_rows_system_prompt_is_merged_into_the_schema_turn():
    row = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        "tools": [_WEATHER_TOOL],
    }

    converted = format_to_messages(row, "tool-calling")["messages"]

    assert converted[0] == {
        "role": "system", "content": "Be brief.\n\n" + _SCHEMA_TURN["content"],
    }
    assert [m["role"] for m in converted] == ["system", "user"]


_OPENAI_ROW = {
    "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_0", "Paris")]},
        {"role": "tool", "tool_call_id": "call_0", "content": "21"},
        {"role": "assistant", "content": "21 C."},
    ],
    "tools": [_WEATHER_TOOL],
}


def test_auto_detects_the_openai_shape_and_converts_it_like_the_explicit_format(tmp_path):
    path = tmp_path / "agent.jsonl"
    _write_jsonl(path, [_OPENAI_ROW])

    assert detect_format([_OPENAI_ROW]) == "tool-calling"
    assert _loaded_rows(str(path), "auto") == _loaded_rows(str(path), "tool-calling")


def test_auto_keeps_nested_calls_when_the_row_has_no_tools(tmp_path):
    # No top-level "tools": the row is plain chatml, which keeps the messages
    # (and their calls) as written.
    row = {"messages": _OPENAI_ROW["messages"]}
    path = tmp_path / "chat.jsonl"
    _write_jsonl(path, [row])

    assert _loaded_rows(str(path), "auto") == [row]


def test_tool_calling_is_still_detected_before_audio():
    row = {**_OPENAI_ROW, "audio": "a.wav"}

    assert detect_format([row]) == "tool-calling"


def test_tools_as_a_json_string_convert_like_a_list(tmp_path):
    row = {**_OPENAI_ROW, "tools": json.dumps([_WEATHER_TOOL])}
    path = tmp_path / "agent.jsonl"
    _write_jsonl(path, [row])

    assert format_to_messages(row, "tool-calling") == format_to_messages(
        _OPENAI_ROW, "tool-calling"
    )
    assert len(_loaded_rows(str(path), "auto")) == 1


def test_tools_as_a_string_that_is_not_json_drops_the_row():
    row = {**_OPENAI_ROW, "tools": "{not json"}

    converted, why = format_to_messages_with_reason(row, "tool-calling")

    assert converted is None
    assert "'tools' is a string that is not JSON" in why


# --- Null content is only allowed on a call-only assistant turn --------------


# A call-only turn with null content converting to "" is pinned by
# test_the_openai_fine_tuning_shape above.


@pytest.mark.parametrize(
    "turn",
    [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "assistant", "content": None},
        {"role": "user", "content": None},
        {"role": "tool", "tool_call_id": "call_0", "content": None},
    ],
    ids=["assistant-empty-calls", "assistant-no-calls", "user", "tool"],
)
def test_null_content_outside_a_call_turn_drops_the_row(turn):
    # A template printing {{ message.content }} would render the literal "None".
    row = {"messages": [{"role": "user", "content": "q"}, turn]}

    converted, why = format_to_messages_with_reason(row, "tool-calling")

    assert converted is None
    assert f"tool-calling {turn['role']} content must be a string" in why


# --- Malformed nested calls are still dropped, with a reason -----------------


@pytest.mark.parametrize(
    ("assistant", "reason"),
    [
        ({"role": "assistant", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": "{bad"}}]},
         "must be JSON-parseable"),
        ({"role": "assistant", "tool_calls": {"function": {"name": "f"}}},
         "'messages.tool_calls' must be a list"),
        ({"role": "assistant", "tool_calls": [{"function": {"name": ""}}]},
         "'function.name' must be a non-empty string"),
        ({"role": "assistant", "tool_calls": [
            {"id": 7, "function": {"name": "f", "arguments": "{}"}}]},
         "'id' must be a string"),
    ],
    ids=["bad-json", "not-a-list", "empty-name", "non-string-id"],
)
def test_a_malformed_nested_call_drops_the_row(assistant, reason):
    row = {"messages": [{"role": "user", "content": "q"}, assistant]}

    converted, why = format_to_messages_with_reason(row, "tool-calling")

    assert converted is None
    assert reason in why


def test_a_non_string_tool_call_id_drops_the_row():
    row = {"messages": [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": 3, "content": "21"},
    ]}

    converted, why = format_to_messages_with_reason(row, "tool-calling")

    assert converted is None
    assert "'tool_call_id' must be a string" in why


# --- The rendered training text, through Arrow and a tool-aware template -------


# The Llama 3.x templates branch on `'tool_calls' in message`, so they see a
# key that is present with a None value as a call.
_LLAMA_STYLE_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message.role == 'tool' %}"
    "<tool id={{ message.tool_call_id }}>{{ message.content }}</tool>"
    "{%- elif 'tool_calls' in message %}"
    "{%- if message.tool_calls|length != 1 %}"
    "{{ raise_exception('one call per turn') }}"
    "{%- endif %}"
    "{% generation %}<call>{{ message.tool_calls[0].function.name }}"
    "({{ message.tool_calls[0].function.arguments }})</call>{% endgeneration %}"
    "{%- elif message.role == 'assistant' %}"
    "{% generation %}<a>{{ message.content }}</a>{% endgeneration %}"
    "{%- else %}<{{ message.role }}>{{ message.content }}</{{ message.role }}>"
    "{%- endif %}"
    "{%- endfor %}"
)


def test_the_trajectory_renders_in_source_order_through_arrow():
    from datasets import Dataset
    from transformers import AutoTokenizer

    from soup_cli.data.sft_format import build_format_row

    tokenizer = AutoTokenizer.from_pretrained(_local_base_model_dir())
    tokenizer.chat_template = _LLAMA_STYLE_TEMPLATE
    data_cfg = SimpleNamespace(
        chat_template=None,
        train_on_responses_only=True,
        train_on_messages_with_train_field=False,
        max_length=512,
    )
    row = format_to_messages({"messages": _OPENAI_ROW["messages"]}, "tool-calling")
    # The trainer maps rows through Arrow, which gives every message the
    # sparse keys it lacks as None (a user turn gets "tool_calls": None).
    arrow_rows = Dataset.from_list([row])
    assert arrow_rows[0]["messages"][0]["tool_calls"] is None

    formatted = arrow_rows.map(
        build_format_row(tokenizer, data_cfg), remove_columns=["messages"]
    )

    text = tokenizer.decode(formatted[0]["input_ids"], skip_special_tokens=True)
    assert re.sub(r"\s+", "", text) == re.sub(r"\s+", "", (
        '<user>Weather in Paris?</user><call>get_weather({"city": "Paris"})</call>'
        "<tool id=call_0>21</tool><a>21 C.</a>"
    ))
