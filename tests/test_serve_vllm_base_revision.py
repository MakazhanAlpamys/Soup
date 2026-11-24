"""A copied LoRA carries the base commit all the way to vLLM and its prompt builder."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.utils.run_provenance import write_run_provenance
from tests.conftest import strip_ansi

_BASE = "test-org/base"
_COMMIT = "a" * 40


@pytest.mark.parametrize(
    "case",
    [
        "pinned-fallback",
        "pinned-local-tokenizer",
        "pinned-equal-adapter-revision",
        "conflicting-adapter-revision",
        "override",
        "local-base",
        "legacy",
        "malformed",
        "cache-hint",
        "modelscope",
        "full-model",
    ],
)
def test_serve_vllm_uses_only_the_matching_recorded_hub_base(tmp_path, monkeypatch, case):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from soup_cli.cli import app
    from soup_cli.utils import vllm

    artifact = tmp_path / "copied"
    artifact.mkdir()
    base = tmp_path / "base"
    if case == "local-base":
        base.mkdir()
    configured_base = str(base) if case == "local-base" else _BASE
    if case != "full-model":
        adapter_config = {"base_model_name_or_path": configured_base}
        if case == "pinned-equal-adapter-revision":
            adapter_config["revision"] = _COMMIT
        elif case == "conflicting-adapter-revision":
            adapter_config["revision"] = "b" * 40
        (artifact / "adapter_config.json").write_text(
            json.dumps(adapter_config), encoding="utf-8"
        )
    if case == "malformed":
        (artifact / "soup_run.json").write_text("{", encoding="utf-8")
    elif case != "legacy":
        write_run_provenance(
            artifact,
            {
                "schema_version": 1,
                "run_id": "train-run",
                "base_model": configured_base,
                "base_revision": _COMMIT,
                "base_revision_source": "cache_path" if case == "cache-hint" else "hub_commit",
            },
        )

    command = ["serve", "--model", str(artifact), "--backend", "vllm"]
    expected_model = str(artifact) if case == "full-model" else configured_base
    if case == "override":
        expected_model = "other-org/override"
        command.extend(["--base", expected_model])
    elif case == "full-model":
        command.extend(["--base", _BASE])

    args = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    start_engine = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=args,
            AsyncLLMEngine=SimpleNamespace(from_engine_args=start_engine),
            SamplingParams=SimpleNamespace,
            envs=SimpleNamespace(VLLM_USE_MODELSCOPE=(case == "modelscope")),
        ),
    )
    monkeypatch.setattr(vllm, "is_vllm_available", lambda: True)
    monkeypatch.setattr(uvicorn, "run", Mock())
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def load_tokenizer(source, **kwargs):
        if Path(source) == artifact and case not in {"pinned-local-tokenizer", "full-model"}:
            raise OSError("no local tokenizer in this fixture")
        return SimpleNamespace(chat_template="chat template")

    tokenizer_load = Mock(side_effect=load_tokenizer)
    auto_tokenizer = SimpleNamespace(from_pretrained=tokenizer_load)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto_tokenizer))

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    engine_args = args.call_args.kwargs
    assert engine_args["model"] == expected_model
    assert engine_args.get("enable_lora", False) is (case != "full-model")
    start_engine.assert_called_once()
    uvicorn.run.assert_called_once()
    pins_expected = case.startswith("pinned-")
    output = " ".join(strip_ansi(result.output).split())
    if pins_expected:
        assert engine_args["revision"] == _COMMIT
        assert engine_args["tokenizer_revision"] == _COMMIT
        assert f"Base revision: {_COMMIT} (recorded run)" in output
    else:
        assert "revision" not in engine_args
        assert "tokenizer_revision" not in engine_args
        assert "(recorded run)" not in output
        if case != "full-model":
            assert "Base revision: unknown" in output

    first, *fallback = tokenizer_load.call_args_list
    assert first.args == (str(artifact),)
    assert "revision" not in first.kwargs
    if case in {"pinned-local-tokenizer", "full-model"}:
        assert not fallback
    else:
        assert len(fallback) == 1
        assert fallback[0].args == (expected_model,)
        if pins_expected:
            assert fallback[0].kwargs["revision"] == _COMMIT
        else:
            assert "revision" not in fallback[0].kwargs
