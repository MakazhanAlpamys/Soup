"""`soup merge` loads the adapter's tokenizer before it loads or writes any weights.

An adapter directory copied without its tokenizer files used to fail only after
the fp16 merge had already written `config.json` and the weights to `--output`,
so the run ended with exit 1 but left a checkpoint that looks complete and has
no tokenizer. Both save formats now fail before the base model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner, Result

BASE = "org/base"


class _NoTokenizerError(OSError):
    pass


@pytest.fixture()
def merge_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE}), encoding="utf-8",
    )
    events: list[str] = []
    state = {"tokenizer_ok": False}

    def save_model(directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
        events.append("save_model")

    def load_model(base: str, **kwargs: object) -> SimpleNamespace:
        events.append("load_model")
        return SimpleNamespace(save_pretrained=save_model)

    def save_tokenizer(directory: str | Path) -> None:
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")
        events.append("save_tokenizer")

    def load_tokenizer(path: str, **kwargs: object) -> SimpleNamespace:
        events.append("load_tokenizer")
        if not state["tokenizer_ok"]:
            raise _NoTokenizerError("Couldn't instantiate the backend tokenizer")
        return SimpleNamespace(save_pretrained=save_tokenizer)

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", SimpleNamespace(
        from_pretrained=load_model,
    ))
    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(
        from_pretrained=load_tokenizer,
    ))
    monkeypatch.setattr(peft, "PeftModel", SimpleNamespace(
        from_pretrained=lambda model, adapter: SimpleNamespace(merge_and_unload=lambda: model),
    ))

    def invoke(*extra: str) -> Result:
        return CliRunner().invoke(app, [
            "--no-audit-log", "--no-telemetry", "merge", "--adapter", str(adapter),
            "--output", "merged", *extra,
        ])

    return SimpleNamespace(invoke=invoke, events=events, state=state, root=tmp_path)


def test_fp16_merge_without_a_tokenizer_writes_nothing(merge_cli: SimpleNamespace) -> None:
    result = merge_cli.invoke()

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Merge failed" in result.output
    assert merge_cli.events == ["load_tokenizer"]
    assert not (merge_cli.root / "merged").exists()


def test_4bit_merge_without_a_tokenizer_loads_no_model(
    merge_cli: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.utils import save_formats

    monkeypatch.setattr(save_formats, "merge_4bit", lambda **kwargs: pytest.fail(
        "4-bit export must not start without a tokenizer",
    ))
    result = merge_cli.invoke("--save-format", "4bit")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert merge_cli.events == ["load_tokenizer"]
    assert not (merge_cli.root / "merged").exists()


def test_fp16_merge_with_a_tokenizer_still_writes_both(merge_cli: SimpleNamespace) -> None:
    merge_cli.state["tokenizer_ok"] = True
    result = merge_cli.invoke()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert merge_cli.events == ["load_tokenizer", "load_model", "save_model", "save_tokenizer"]
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        assert (merge_cli.root / "merged" / name).is_file()
