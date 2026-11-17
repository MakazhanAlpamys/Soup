"""``parse_langsmith`` reads chat model batches and OpenAI-style ``choices`` (#2868)."""

import copy

import pytest

from soup_cli.utils.ingest_sources import parse_langsmith


def _lc(kind, content, message_type=None):
    """A LangChain message as LangSmith stores it (``dumpd``)."""
    return {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", kind],
        "kwargs": {"content": content, "type": message_type or kind[:-7].lower()},
    }


def _human(content):
    return _lc("HumanMessage", content)


def _system(content):
    return _lc("SystemMessage", content)


def _thumbs(avg):
    return {"thumbs": {"n": 1, "avg": avg}}


# The four LLM run shapes of the #2868 report (LangChain chat models and
# ``wrap_openai`` / ``@traceable(run_type="llm")``), with the text replaced.
RUNS = [
    {
        "id": "run-chat-1",
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": {"messages": [[_human("What is the capital of France?")]]},
        "outputs": {
            "generations": [[{
                "text": "Paris.",
                "type": "ChatGeneration",
                "message": _lc("AIMessage", "Paris.", "ai"),
            }]],
            "llm_output": {"model_name": "gpt-4o-mini"},
        },
        "feedback_stats": _thumbs(1.0),
    },
    {
        "id": "run-chat-2",
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": {"messages": [[
            _system("Answer in one word."),
            _human([{"type": "text", "text": "Largest planet?"}]),
        ]]},
        "outputs": {"generations": [[{"text": "Jupiter.", "type": "ChatGeneration"}]]},
        "feedback_stats": _thumbs(0.0),
    },
    {
        "id": "run-openai-1",
        "name": "ChatOpenAI",
        "run_type": "llm",
        "inputs": {
            "messages": [{"role": "user", "content": "Name another primary color."}],
            "model": "gpt-4o-mini",
        },
        "outputs": {
            "id": "chatcmpl-redacted",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Blue is also a primary color."},
                "finish_reason": "stop",
            }],
        },
    },
    {
        "id": "run-openai-2",
        "name": "summarise",
        "run_type": "llm",
        "inputs": {"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Summarise: the sky is blue."},
        ]},
        "outputs": {"choices": [{"message": {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The sky is blue."}],
        }}]},
        "feedback_stats": _thumbs(1.0),
    },
]


def _rows(runs):
    return [(row.trace_id, row.prompt, row.output, row.signal) for row in parse_langsmith(runs)]


def test_each_llm_run_gives_one_row_with_its_thumbs_signal() -> None:
    assert _rows(copy.deepcopy(RUNS)) == [
        ("run-chat-1", "What is the capital of France?", "Paris.", "thumbs_up"),
        ("run-chat-2", "Largest planet?", "Jupiter.", "thumbs_down"),
        ("run-openai-1", "Name another primary color.", "Blue is also a primary color.", "none"),
        ("run-openai-2", "Summarise: the sky is blue.", "The sky is blue.", "thumbs_up"),
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Q"}],
        [{"type": "human", "content": "Q"}],
        [{"type": "system", "content": "S"}, {"type": "human", "content": "Q"}],
    ],
    ids=["role", "type", "system-type"],
)
def test_flat_messages_give_the_same_row_as_before(messages) -> None:
    run = {"id": "r", "inputs": {"messages": messages},
           "outputs": {"generations": [[{"text": "A"}]]}, "feedback_stats": _thumbs(1)}
    assert _rows([run]) == [("r", "Q", "A", "thumbs_up")]


def test_a_batch_of_one_is_read_as_its_list() -> None:
    plain = {"id": "r", "inputs": {"messages": [[{"role": "user", "content": "Q"}]]},
             "outputs": {"generations": [[{"text": "A"}]]}}
    assert _rows([plain]) == [("r", "Q", "A", "none")]


def test_a_batch_of_several_prompts_gives_no_row_and_no_error() -> None:
    run = {"id": "r", "inputs": {"messages": [[_human("Q1")], [_human("Q2")]]},
           "outputs": {"generations": [[{"text": "A1"}], [{"text": "A2"}]]}}
    following = copy.deepcopy(RUNS[0])
    assert _rows([run, following]) == [
        ("run-chat-1", "What is the capital of France?", "Paris.", "thumbs_up"),
    ]


@pytest.mark.parametrize(
    "choice, expected",
    [
        ({"message": {"content": "A"}}, "A"),
        ({"message": {"content": [{"type": "text", "text": "A"}]}}, "A"),
        ({"text": "A"}, "A"),
    ],
    ids=["message-string", "message-parts", "text"],
)
def test_the_answer_is_read_from_choices(choice, expected) -> None:
    run = {"id": "r", "inputs": {"messages": [_human("Q")]}, "outputs": {"choices": [choice]},
           "feedback_stats": _thumbs(0)}
    assert _rows([run]) == [("r", "Q", expected, "thumbs_down")]


def test_generations_keep_priority_over_choices() -> None:
    run = {"id": "r", "inputs": {"messages": [_human("Q")]},
           "outputs": {"generations": [[{"text": "From generations"}]],
                       "choices": [{"message": {"content": "From choices"}}]}}
    assert _rows([run]) == [("r", "Q", "From generations", "none")]


@pytest.mark.parametrize(
    "choices", [[], [None], [{"message": None}], None],
    ids=["empty", "none-entry", "null-message", "null"],
)
def test_choices_without_an_answer_give_no_row_and_no_error(choices) -> None:
    run = {"id": "r", "inputs": {"messages": [_human("Q")]}, "outputs": {"choices": choices}}
    assert _rows([run]) == []
