"""Adapter arithmetic / merge must not overwrite an input adapter (#2358)."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

_STEM = "base_model.model.layers.0.self_attn"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    return " ".join(_ANSI.sub("", text).split())


def _make_adapter(directory, a, b, extra=None):
    from safetensors.numpy import save_file

    directory.mkdir()
    tensors = {
        f"{_STEM}.q_proj.lora_A.weight": np.asarray(a, dtype=np.float32),
        f"{_STEM}.q_proj.lora_B.weight": np.asarray(b, dtype=np.float32),
    }
    tensors.update(extra or {})
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "meta/x",
                    "r": 1, "lora_alpha": 1}),
        encoding="utf-8",
    )


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


@pytest.fixture()
def adapters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    v_proj = {
        f"{_STEM}.v_proj.lora_A.weight": np.ones((1, 2), np.float32),
        f"{_STEM}.v_proj.lora_B.weight": np.ones((2, 1), np.float32),
    }
    _make_adapter(tmp_path / "A", [[1, 2]], [[3], [4]], extra=v_proj)
    _make_adapter(tmp_path / "B", [[10, 20]], [[30], [40]])
    return tmp_path


def _invoke(args):
    from typer.testing import CliRunner

    from soup_cli.commands.adapters import app

    return CliRunner().invoke(app, args)


@pytest.mark.parametrize("output", ["A", "B", "./A", "A/"])
def test_arithmetic_refuses_output_naming_an_input(adapters, output):
    before = {name: _snapshot(adapters / name) for name in ("A", "B")}
    res = _invoke(["arithmetic", "A + B", "--adapter", "A=A", "--adapter", "B=B",
                   "--allow-unscanned", "-o", output])
    assert res.exit_code == 1, (res.output, repr(res.exception))
    assert "same file" in _plain(res.output)
    assert {name: _snapshot(adapters / name) for name in ("A", "B")} == before


def test_arithmetic_distinct_output_preserves_inputs(adapters):
    before = {name: _snapshot(adapters / name) for name in ("A", "B")}
    res = _invoke(["arithmetic", "A + B", "--adapter", "A=A", "--adapter", "B=B",
                   "--allow-unscanned", "-o", "merged"])
    assert res.exit_code == 0, res.output
    assert os.path.isfile(adapters / "merged" / "adapter_model.safetensors")
    assert {name: _snapshot(adapters / name) for name in ("A", "B")} == before


@pytest.mark.parametrize("output", ["A", "B"])
def test_merge_refuses_output_naming_an_input(adapters, output):
    before = {name: _snapshot(adapters / name) for name in ("A", "B")}
    res = _invoke(["merge", "A", "B", "--allow-unscanned", "-o", output])
    assert res.exit_code == 1, (res.output, repr(res.exception))
    assert "--output" in _plain(res.output)
    assert "same file" in _plain(res.output)
    assert {name: _snapshot(adapters / name) for name in ("A", "B")} == before


def test_merge_cmaes_refuses_before_search(adapters, monkeypatch):
    from soup_cli.utils import cmaes_merge

    calls = []
    monkeypatch.setattr(
        cmaes_merge, "build_cmaes_plan", lambda **kwargs: calls.append(kwargs)
    )
    (adapters / "suite.jsonl").write_text('{"prompt": "hi"}\n', encoding="utf-8")
    before = {name: _snapshot(adapters / name) for name in ("A", "B")}
    res = _invoke(["merge", "A", "B", "--strategy", "cmaes", "--eval", "suite.jsonl",
                   "-o", "A"])
    assert res.exit_code == 1, (res.output, repr(res.exception))
    assert "same file" in _plain(res.output)
    assert calls == []
    assert {name: _snapshot(adapters / name) for name in ("A", "B")} == before
