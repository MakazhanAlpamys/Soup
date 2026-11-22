"""#2574: ``UnlearnTrainerWrapper.setup()`` reads its rows before any model loads.

A ``retain_set`` that is set but yields no usable row is refused with a message
naming the path, in the form of the ``forget_set`` refusal. Both row loads come
before the policy and the NPO / RMU reference load, so either refusal costs no
model load. An absent ``retain_set`` keeps the warning it had.

The refusal tests patch ``load_model_and_tokenizer`` to fail if reached. The
tests of the unchanged paths drive the real ``setup()`` on CPU with the tiny
random Llama from ``tests/_tiny_hf_models.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from soup_cli.config.loader import load_config_from_string
from tests.conftest import strip_ansi

_METHODS = ("npo", "simnpo", "rmu")
_USABLE = json.dumps({"prompt": "hi", "completion": "there friend"}) + "\n"
_UNUSABLE = {
    "all-blank": "\n   \n\n",
    "bad-json": "{not json\n[1, 2\n",
    "no-usable-pair": "\n".join(
        json.dumps(row)
        for row in (
            {"prompt": "hi", "completion": ""},
            {"messages": [{"role": "user", "content": "hi"}]},
            {"text": ""},
            [1, 2],
        )
    )
    + "\n",
}


@pytest.fixture(autouse=True)
def train_stack() -> Iterator[None]:
    for name in ("torch", "transformers", "peft"):
        pytest.importorskip(name, reason=f"{name} requires the train extra")
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    *,
    forget: str = _USABLE,
    retain: str | None = _USABLE,
    base: str = "never-loaded-model",
) -> Any:
    import yaml

    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    monkeypatch.chdir(tmp_path)
    (tmp_path / "forget.jsonl").write_text(forget, encoding="utf-8")
    data: dict[str, Any] = {
        "train": "forget.jsonl",
        "forget_set": "forget.jsonl",
        "max_length": 64,
    }
    if retain is not None:
        (tmp_path / "retain.jsonl").write_text(retain, encoding="utf-8")
        data["retain_set"] = "retain.jsonl"
    cfg = load_config_from_string(yaml.safe_dump({
        "base": base,
        "task": "unlearn",
        "backend": "transformers",
        "data": data,
        "training": {
            "unlearn_method": method,
            "quantization": "none",
            "batch_size": 1,
            "epochs": 1,
            "warmup_ratio": 0.0,
            "seed": 7,
            "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        },
        "output": "out",
    }))
    return UnlearnTrainerWrapper(cfg, device="cpu")


@pytest.fixture
def no_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a model was loaded before the rows were checked")

    monkeypatch.setattr("soup_cli.utils.live_eval.load_model_and_tokenizer", refuse)


@pytest.mark.usefixtures("no_model_load")
@pytest.mark.parametrize("content", list(_UNUSABLE.values()), ids=list(_UNUSABLE))
@pytest.mark.parametrize("method", _METHODS)
def test_an_unusable_retain_set_is_refused_before_any_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, content: str
) -> None:
    wrapper = _wrapper(tmp_path, monkeypatch, method, retain=content)

    with pytest.raises(ValueError) as excinfo:
        wrapper.setup()

    assert str(excinfo.value) == "retain_set 'retain.jsonl' yielded no usable rows"
    assert wrapper.model is None
    assert wrapper.ref_model is None


@pytest.mark.usefixtures("no_model_load")
@pytest.mark.parametrize("content", list(_UNUSABLE.values()), ids=list(_UNUSABLE))
@pytest.mark.parametrize("method", _METHODS)
def test_an_unusable_forget_set_is_refused_before_any_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, content: str
) -> None:
    wrapper = _wrapper(tmp_path, monkeypatch, method, forget=content)

    with pytest.raises(ValueError) as excinfo:
        wrapper.setup()

    assert str(excinfo.value) == "forget_set 'forget.jsonl' yielded no usable rows"
    assert wrapper.model is None


@pytest.mark.parametrize("method", ["npo", "simnpo"])
def test_an_absent_retain_set_still_warns_and_sets_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
) -> None:
    from tests._tiny_hf_models import tiny_model_dir

    wrapper = _wrapper(tmp_path, monkeypatch, method, retain=None, base=tiny_model_dir("llama"))

    wrapper.setup()

    out = " ".join(strip_ansi(capsys.readouterr().out).split())
    assert f"method={method} has no retain_set" in out
    assert wrapper.model is not None
    assert wrapper._forget == [("hi", "there friend")]
    assert wrapper._retain == []


def test_an_absent_retain_set_still_sets_up_for_rmu_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests._tiny_hf_models import tiny_model_dir

    wrapper = _wrapper(tmp_path, monkeypatch, "rmu", retain=None, base=tiny_model_dir("llama"))

    wrapper.setup()

    assert "no retain_set" not in " ".join(strip_ansi(capsys.readouterr().out).split())
    assert wrapper._retain == []
    assert wrapper.ref_model is not None


@pytest.mark.usefixtures("no_model_load")
def test_the_method_is_validated_before_the_rows_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The config already refuses an unknown method, so set it after construction.
    wrapper = _wrapper(tmp_path, monkeypatch, "npo", forget=_UNUSABLE["all-blank"])
    wrapper.method = "bogus"

    with pytest.raises(ValueError, match="unknown unlearn method 'bogus'"):
        wrapper.setup()


@pytest.mark.parametrize("method", _METHODS)
def test_a_usable_retain_set_sets_up_with_its_rows_and_no_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    method: str,
) -> None:
    from tests._tiny_hf_models import tiny_model_dir

    # One unusable line among usable ones is still skipped, as before.
    retain = "{not json\n" + json.dumps({"text": "hello world"}) + "\n"
    wrapper = _wrapper(tmp_path, monkeypatch, method, retain=retain, base=tiny_model_dir("llama"))

    wrapper.setup()

    assert "no retain_set" not in " ".join(strip_ansi(capsys.readouterr().out).split())
    assert wrapper.model is not None
    assert wrapper._retain == [("", "hello world")]
    assert (wrapper.ref_model is not None) == (method in ("npo", "rmu"))
