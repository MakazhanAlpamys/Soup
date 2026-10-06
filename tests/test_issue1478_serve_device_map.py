"""Regression tests for #1478: ``soup serve --device`` reaches the transformers loader.

``serve._load_model`` used to load the base model and the merged model with a hard-coded
``device_map="auto"``, so ``--device cpu`` could still place the model on a visible GPU.
It now places the model with the same rule as ``soup infer`` / ``chat`` / ``diff``
(``utils.gpu.resolve_inference_device_map_and_dtype``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from typer.testing import CliRunner

from tests.conftest import strip_ansi

runner = CliRunner()


def _collapse(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _load(device, *, is_adapter=False, kv_cache_dtype=None):
    """Run the real ``_load_model`` with transformers/peft mocked; return the loader mock."""
    from soup_cli.commands.serve import _load_model

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()
    ) as load, patch("peft.PeftModel.from_pretrained", return_value=MagicMock()):
        _load_model(
            model_path="some-org/some-model",
            base_model="some-org/base-model" if is_adapter else None,
            is_adapter=is_adapter,
            device=device,
            kv_cache_dtype=kv_cache_dtype,
        )
    return load


@pytest.mark.parametrize("is_adapter", [False, True], ids=["plain", "base+adapter"])
@pytest.mark.parametrize(
    ("device", "device_map", "dtype"),
    [
        ("cpu", {"": "cpu"}, torch.float32),
        ("mlx", {"": "cpu"}, torch.float32),
        ("mps", {"": "mps"}, torch.float16),
        ("cuda:1", {"": "cuda:1"}, torch.float16),
        ("cuda", "auto", torch.float16),
    ],
)
def test_loader_places_the_model_by_the_shared_rule(is_adapter, device, device_map, dtype):
    load = _load(device, is_adapter=is_adapter)
    assert load.call_count == 1
    kwargs = load.call_args.kwargs
    assert kwargs["device_map"] == device_map, kwargs
    assert kwargs["torch_dtype"] == dtype, kwargs


def test_no_device_behaves_as_the_helper_default():
    from soup_cli.utils.gpu import resolve_inference_device_map_and_dtype

    device_map, dtype = resolve_inference_device_map_and_dtype(None)
    kwargs = _load(None).call_args.kwargs
    assert (kwargs["device_map"], kwargs["torch_dtype"]) == (device_map, dtype)


@pytest.mark.parametrize(
    ("device", "kv_cache_dtype", "dtype"),
    [
        ("cpu", "float16", torch.float16),
        ("cpu", "bfloat16", torch.bfloat16),
        ("cuda", "float16", torch.float16),
        ("cuda", "bfloat16", torch.bfloat16),
    ],
)
def test_explicit_kv_cache_type_still_sets_the_load_dtype(device, kv_cache_dtype, dtype):
    assert _load(device, kv_cache_dtype=kv_cache_dtype).call_args.kwargs["torch_dtype"] == dtype


def test_load_model_refuses_an_unknown_device_even_when_called_directly():
    with pytest.raises(ValueError, match="cpu, cuda, cuda:<index> or mps"):
        _load("tpu")


def _serve_to_the_loader(tmp_path, *args):
    """Invoke the real CLI and stop at ``_load_model``; return (result, loader mock)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from soup_cli.cli import app

    with patch(
        "soup_cli.commands.serve._load_model", side_effect=RuntimeError("stop after the load call")
    ) as load:
        result = runner.invoke(app, ["serve", "--model", str(tmp_path), *args])
    return result, load


@pytest.mark.parametrize(
    ("given", "passed"),
    [
        ("cpu", "cpu"),
        (" CPU ", "cpu"),
        ("cuda", "cuda"),
        ("CUDA", "cuda"),
        ("cuda:1", "cuda:1"),
        ("mps", "mps"),
    ],
)
def test_cli_hands_the_normalised_device_to_the_loader(tmp_path, given, passed):
    _, load = _serve_to_the_loader(tmp_path, "--device", given)
    assert load.call_args.kwargs["device"] == passed, load.call_args


@pytest.mark.parametrize("bad", ["tpu", "gpu", "auto", "cuda:abc", "cuda:0,1", "cpu0"])
def test_cli_refuses_an_unknown_device_naming_the_accepted_values(tmp_path, bad):
    result, load = _serve_to_the_loader(tmp_path, "--device", bad)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not load.called
    text = _collapse(result.output)
    assert "cpu, cuda, cuda:<index> or mps" in text, text
    assert repr(bad) in text, text


def test_cli_refuses_a_markup_like_device_instead_of_crashing(tmp_path):
    result, _ = _serve_to_the_loader(tmp_path, "--device", "[/bold]")
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "[/bold]" in _collapse(result.output)


def test_vllm_backend_still_reaches_its_server_without_a_device_argument(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from soup_cli.cli import app

    with patch("soup_cli.utils.vllm.is_vllm_available", return_value=True), patch(
        "soup_cli.commands.serve._serve_vllm", return_value=MagicMock()
    ) as serve_vllm, patch("uvicorn.run"):
        result = runner.invoke(
            app, ["serve", "--model", str(tmp_path), "--backend", "vllm", "--device", "cuda"]
        )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "device" not in serve_vllm.call_args.kwargs


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps"])
def test_the_draft_model_is_placed_like_the_main_model(device):
    """One flag, one placement: the speculative draft must land where the main model does."""
    from soup_cli.commands.serve import _load_draft_model

    main = _load(device).call_args.kwargs
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=MagicMock()
    ) as load:
        _load_draft_model("some-org/draft", device)
    draft = load.call_args.kwargs
    assert (draft["device_map"], draft["torch_dtype"]) == (
        main["device_map"], main["torch_dtype"],
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_cli_refuses_an_unknown_device_on_every_backend(tmp_path, backend):
    """The refusal is not a transformers-only check: ``--device tpu`` stops vLLM/SGLang too.

    A mutation that validated ``--device`` only for ``--backend transformers`` survived the
    tests above, which exercise the refusal on the default backend only.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from soup_cli.cli import app

    with patch("soup_cli.utils.vllm.is_vllm_available", return_value=True), patch(
        "soup_cli.utils.sglang.check_sglang_available", return_value=True
    ), patch(
        "soup_cli.commands.serve._serve_vllm", return_value=MagicMock()
    ) as serve_vllm, patch(
        "soup_cli.commands.serve._serve_sglang", return_value=MagicMock()
    ) as serve_sglang, patch("uvicorn.run") as run:
        result = runner.invoke(
            app, ["serve", "--model", str(tmp_path), "--backend", backend, "--device", "tpu"]
        )
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert not (serve_vllm.called or serve_sglang.called or run.called)
    assert "cpu, cuda, cuda:<index> or mps" in _collapse(result.output)
