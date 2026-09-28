"""Opt-in graph decoding reaches the real command generation wrappers."""

import json
import re
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Typer draws errors in a Rich panel; a wrapped message line carries its borders.
_BOX_RE = re.compile("[─-╿]")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output (Rich colours and boxes)."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


@pytest.mark.parametrize("args", [["infer", "--help"], ["bench", "infer", "--help"]])
def test_cuda_graph_option_is_documented(args):
    from soup_cli.cli import app

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--cuda-graphs" in _plain(result.output)


def test_chat_does_not_offer_cuda_graphs():
    """Scope pin. A chat history grows every turn and transformers sizes a static
    cache as max(this request, every earlier one), so each turn would recompile.
    Re-adding the flag to chat needs a capacity reservation first."""
    from soup_cli.cli import app

    result = CliRunner().invoke(app, ["chat", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--cuda-graphs" not in _plain(result.output)


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_uses_graph_kwargs_only_when_requested(monkeypatch, enabled):
    import torch

    from soup_cli.commands import infer
    from soup_cli.utils import cuda_graphs

    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    monkeypatch.setattr("soup_cli.utils.vllm.encode_chat_prompt", lambda *a, **kw: inputs)
    config = object()
    prepare = MagicMock(return_value={"cache_implementation": "static", "compile_config": config})
    monkeypatch.setattr(cuda_graphs, "cuda_graph_generation_kwargs", prepare)
    model = MagicMock(device=torch.device("cpu"))
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.decode.return_value = "answer"
    response = infer._generate(
        model,
        tokenizer,
        [{"role": "user", "content": "question"}],
        max_tokens=2,
        temperature=0.0,
        cuda_graphs=enabled,
    )
    assert response == ("answer", 2)
    actual = model.generate.call_args.kwargs
    assert actual["max_new_tokens"] == 2
    assert actual["do_sample"] is False
    assert torch.equal(actual["input_ids"], inputs["input_ids"])
    if enabled:
        prepare.assert_called_once_with(model)
        assert actual["cache_implementation"] == "static"
        assert actual["compile_config"] is config
    else:
        prepare.assert_not_called()
        assert "compile_config" not in actual
        assert "cache_implementation" not in actual


def _infer_env(monkeypatch, tmp_path, prompts, generate):
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (tmp_path / "prompts.txt").write_text("\n".join(prompts) + "\n")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", None))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", lambda model: {})
    monkeypatch.setattr(infer, "_load_model", MagicMock(return_value=(object(), object())))
    monkeypatch.setattr(infer, "_count_prompt_tokens", lambda tokenizer, text: len(text.split()))
    monkeypatch.setattr(infer, "_generate", generate)
    return model_dir


def _infer_args(model_dir, *extra):
    return ["infer", "--model", str(model_dir), "--input", "prompts.txt",
            "--output", "output.jsonl", *extra]


@pytest.mark.parametrize("command", ["infer", "bench"])
def test_cli_forwards_opt_in_to_every_generation(monkeypatch, tmp_path, command):
    from soup_cli.cli import app

    generate = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], generate)
    if command == "bench":
        args = ["bench", "infer", str(model_dir), "--num-prompts", "1", "--max-tokens", "64"]
    else:
        args = _infer_args(model_dir)
    result = CliRunner().invoke(app, [*args, "--cuda-graphs"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert generate.call_count == 2  # one warm-up capture, then the real request
    assert all(call.kwargs["cuda_graphs"] is True for call in generate.call_args_list)
    if command == "bench":
        assert all(call.kwargs["max_tokens"] == 64 for call in generate.call_args_list)


def test_cuda_graphs_reject_asr_before_loading(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio.jsonl").write_text('{"audio": "speech.wav"}\n')
    load = MagicMock()
    monkeypatch.setattr(infer, "_infer_asr", load)
    result = CliRunner().invoke(
        app,
        ["infer", "--model", ".", "--task", "asr", "--input", "audio.jsonl",
         "--output", "output.jsonl", "--cuda-graphs"],
    )
    assert result.exit_code == 2
    assert "text generation only" in _plain(result.output)
    assert "Omit --cuda-graphs" in _plain(result.output)
    load.assert_not_called()


def test_cuda_graphs_reject_batch_size_above_one_before_loading(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], MagicMock())
    load = MagicMock(return_value=(object(), object()))
    monkeypatch.setattr(infer, "_load_model", load)

    result = CliRunner().invoke(
        app,
        _infer_args(model_dir, "--cuda-graphs", "--batch-size", "2"),
    )

    assert result.exit_code == 2
    assert "--batch-size above 1" in _plain(result.output)
    assert "Omit --cuda-graphs" in _plain(result.output)
    load.assert_not_called()


def test_unsupported_graph_model_preserves_existing_output(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (tmp_path / "prompts.txt").write_text("question\n")
    output = tmp_path / "output.jsonl"
    output.write_text("previous results\n")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", None))
    monkeypatch.setattr(infer, "_load_model", MagicMock(return_value=(object(), object())))
    reject = MagicMock(side_effect=RuntimeError("Unsupported model for CUDA graphs"))
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", reject)
    generate = MagicMock(side_effect=RuntimeError("Should fail before generation"))
    monkeypatch.setattr(infer, "_generate", generate)
    result = CliRunner().invoke(
        app,
        ["infer", "--model", str(model_dir), "--input", "prompts.txt",
         "--output", str(output), "--cuda-graphs"],
    )
    assert result.exit_code == 1
    assert "Unsupported model for CUDA graphs" in _plain(result.output)
    assert "Omit --cuda-graphs" in _plain(result.output)
    assert output.read_text() == "previous results\n"
    generate.assert_not_called()


def test_infer_warms_up_greedily_on_the_longest_prompt_before_writing(monkeypatch, tmp_path):
    from soup_cli.cli import app

    output = tmp_path / "output.jsonl"
    output.write_text("previous results\n")
    seen = []

    def generate(model, tokenizer, messages, **kwargs):
        if not seen:
            seen.append(output.read_text())
        return "answer", 3

    spy = MagicMock(side_effect=generate)
    model_dir = _infer_env(monkeypatch, tmp_path, ["one two", "one two three four", "one"], spy)
    result = CliRunner().invoke(
        app, _infer_args(model_dir, "--temperature", "0.7", "--cuda-graphs")
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert seen == ["previous results\n"], "the capture must run before the output opens"
    warm, *rows = spy.call_args_list
    assert warm.args[2] == [{"role": "user", "content": "one two three four"}]
    assert warm.kwargs["temperature"] == 0.0
    assert warm.kwargs["cuda_graphs"] is True
    assert warm.kwargs["min_tokens"] == 4  # warm-up node, recording, replay
    assert all("min_tokens" not in call.kwargs for call in rows)
    assert [call.kwargs["temperature"] for call in rows] == [0.7, 0.7, 0.7]
    assert len(output.read_text().splitlines()) == 3


def test_a_warm_up_failure_exits_1_and_leaves_the_output_untouched(monkeypatch, tmp_path):
    from soup_cli.cli import app

    output = tmp_path / "output.jsonl"
    output.write_text("previous results\n")
    spy = MagicMock(side_effect=RuntimeError("capture exploded"))
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], spy)
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    text = _plain(result.output)
    assert "Generation failed with --cuda-graphs: capture exploded" in text
    assert "Omit --cuda-graphs to use normal generation" in text
    assert output.read_text() == "previous results\n"
    assert spy.call_count == 1


def test_a_mid_batch_failure_keeps_the_rows_already_written(monkeypatch, tmp_path):
    from soup_cli.cli import app

    outcomes = iter([("warm", 1), ("first", 2), RuntimeError("CUDA out of memory")])

    def generate(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    model_dir = _infer_env(
        monkeypatch, tmp_path, ["q1", "q2", "q3"], MagicMock(side_effect=generate)
    )
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    assert "Generation failed with --cuda-graphs: CUDA out of memory" in _plain(result.output)
    lines = (tmp_path / "output.jsonl").read_text().splitlines()
    assert [json.loads(line)["response"] for line in lines] == ["first"]


def test_a_single_prompt_costs_one_warm_up_plus_the_row(monkeypatch, tmp_path):
    from soup_cli.cli import app

    spy = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], spy)
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert spy.call_count == 2
    assert len((tmp_path / "output.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError])
def test_without_the_flag_nothing_is_warmed_and_errors_propagate_unchanged(
    monkeypatch, tmp_path, error_type
):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    spy = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(monkeypatch, tmp_path, ["q1", "q2"], spy)
    result = CliRunner().invoke(app, _infer_args(model_dir))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert spy.call_count == 2
    assert all("cuda_graphs" not in call.kwargs for call in spy.call_args_list)

    boom = error_type("plain failure")
    monkeypatch.setattr(infer, "_generate", MagicMock(side_effect=boom))
    result = CliRunner().invoke(app, _infer_args(model_dir))
    assert result.exception is boom
    assert "Generation failed with --cuda-graphs" not in _plain(result.output)


def test_warm_up_picks_the_prompt_with_the_most_tokens_not_characters(monkeypatch):
    import torch

    from soup_cli.commands import infer

    lengths = {"a much longer sentence in plain ascii": 6, "短い": 40}

    def encode(messages, tokenizer, **kwargs):
        size = lengths[messages[0]["content"]]
        return {"input_ids": torch.zeros(1, size, dtype=torch.long)}

    monkeypatch.setattr("soup_cli.utils.vllm.encode_chat_prompt", encode)
    assert infer._longest_prompt(object(), list(lengths)) == "短い"


def _bench_args(model_dir, *extra):
    return ["bench", "infer", str(model_dir), "--prompts-file", "prompts.txt",
            "--max-tokens", "64", *extra]


def test_bench_warms_on_the_longest_prompt_at_the_timed_length(monkeypatch, tmp_path):
    from soup_cli.cli import app

    spy = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(
        monkeypatch, tmp_path, ["short one", "a considerably longer prompt here"], spy
    )
    result = CliRunner().invoke(app, _bench_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    warm, *timed = spy.call_args_list
    assert warm.args[2] == [{"role": "user", "content": "a considerably longer prompt here"}]
    assert warm.kwargs["max_tokens"] == 64
    assert warm.kwargs["temperature"] == 0.0
    assert len(timed) == 2


def test_bench_without_the_flag_keeps_its_short_warm_up(monkeypatch, tmp_path):
    from soup_cli.cli import app

    spy = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(
        monkeypatch, tmp_path, ["short one", "a considerably longer prompt here"], spy
    )
    result = CliRunner().invoke(app, _bench_args(model_dir))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    warm = spy.call_args_list[0]
    assert warm.args[2] == [{"role": "user", "content": "short one"}]
    assert warm.kwargs["max_tokens"] == 32
    assert "cuda_graphs" not in warm.kwargs


def test_the_hint_is_not_repeated_when_the_refusal_already_carries_it(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    already_hinted = RuntimeError("Cannot read the PyTorch version; omit --cuda-graphs")
    reject = MagicMock(side_effect=already_hinted)
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], MagicMock())
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", reject)
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    assert _plain(result.output).lower().count("omit --cuda-graphs") == 1
    infer._generate.assert_not_called()


@pytest.mark.parametrize("error", [TypeError("cudagraphify() got an unexpected keyword"),
                                   AssertionError("cudagraph tree invariant")])
def test_any_error_in_the_warm_up_is_a_named_failure(monkeypatch, tmp_path, error):
    from soup_cli.cli import app

    output = tmp_path / "output.jsonl"
    output.write_text("previous results\n")
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], MagicMock(side_effect=error))
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    assert f"Generation failed with --cuda-graphs: {error}" in _plain(result.output)
    assert output.read_text() == "previous results\n"


def test_any_error_mid_batch_is_a_named_failure_that_keeps_rows(monkeypatch, tmp_path):
    from soup_cli.cli import app

    outcomes = iter([("warm", 1), ("first", 2), TypeError("deferred cudagraphify drifted")])

    def generate(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    model_dir = _infer_env(
        monkeypatch, tmp_path, ["q1", "q2", "q3"], MagicMock(side_effect=generate)
    )
    result = CliRunner().invoke(app, _infer_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    text = _plain(result.output)
    assert "Generation failed with --cuda-graphs: deferred cudagraphify drifted" in text
    lines = (tmp_path / "output.jsonl").read_text().splitlines()
    assert [json.loads(line)["response"] for line in lines] == ["first"]


def test_a_tiny_max_tokens_caps_the_warm_up_minimum(monkeypatch, tmp_path):
    from soup_cli.cli import app

    spy = MagicMock(return_value=("answer", 2))
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], spy)
    result = CliRunner().invoke(app, _infer_args(model_dir, "--max-tokens", "2", "--cuda-graphs"))
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert spy.call_args_list[0].kwargs["min_tokens"] == 2


def test_generate_passes_min_new_tokens_only_when_asked(monkeypatch):
    import torch

    from soup_cli.commands import infer

    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    monkeypatch.setattr("soup_cli.utils.vllm.encode_chat_prompt", lambda *a, **kw: inputs)
    model = MagicMock(device=torch.device("cpu"))
    model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.decode.return_value = "answer"
    messages = [{"role": "user", "content": "question"}]
    infer._generate(model, tokenizer, messages, max_tokens=8, temperature=0.0, min_tokens=4)
    assert model.generate.call_args.kwargs["min_new_tokens"] == 4
    infer._generate(model, tokenizer, messages, max_tokens=8, temperature=0.0)
    assert "min_new_tokens" not in model.generate.call_args.kwargs


def test_bench_refuses_the_flag_off_transformers_with_the_hint(monkeypatch, tmp_path):
    from soup_cli.cli import app

    load = MagicMock()
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], MagicMock())
    monkeypatch.setattr("soup_cli.commands.infer._load_model", load)
    result = CliRunner().invoke(
        app, ["bench", "infer", str(model_dir), "--backend", "mlx", "--cuda-graphs"]
    )
    assert result.exit_code == 2
    text = _plain(result.output)
    assert "requires --backend transformers" in text
    assert "Omit --cuda-graphs" in text
    load.assert_not_called()


def test_bench_pre_flight_refusal_carries_the_hint(monkeypatch, tmp_path):
    from soup_cli.cli import app

    generate = MagicMock()
    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], generate)
    reject = MagicMock(side_effect=RuntimeError("Unsupported model for CUDA graphs"))
    monkeypatch.setattr("soup_cli.utils.cuda_graphs.cuda_graph_generation_kwargs", reject)
    result = CliRunner().invoke(app, _bench_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    text = _plain(result.output)
    assert "Unsupported model for CUDA graphs" in text
    assert "Omit --cuda-graphs" in text
    generate.assert_not_called()


def test_bench_timed_failure_is_named_and_the_default_re_raises(monkeypatch, tmp_path):
    from soup_cli.cli import app
    from soup_cli.commands import infer

    outcomes = iter([("warm", 1), TypeError("timed row drifted")])

    def generate(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    model_dir = _infer_env(monkeypatch, tmp_path, ["question"], MagicMock(side_effect=generate))
    result = CliRunner().invoke(app, _bench_args(model_dir, "--cuda-graphs"))
    assert result.exit_code == 1
    assert "Generation failed with --cuda-graphs: timed row drifted" in _plain(result.output)

    boom = TypeError("plain bench failure")
    monkeypatch.setattr(infer, "_generate", MagicMock(side_effect=[("warm", 1), boom]))
    result = CliRunner().invoke(app, _bench_args(model_dir))
    assert result.exception is boom
