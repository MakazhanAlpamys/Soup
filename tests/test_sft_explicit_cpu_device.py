"""SFT passes ``use_cpu=True`` to ``TrainingArguments`` when the device is cpu.

Without it a run that asks for the CPU on a CUDA host still trains on the GPU.
The tests build the real wrapper over a tiny local checkpoint and record the
keyword arguments of ``TrainingArguments`` at the point the trainer is built, so
no GPU is needed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")

from tests.test_issue1223_evaluate_val_split import _tiny_llama_dir  # noqa: E402


class _CapturedError(Exception):
    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


def _recorder(*args, **kwargs):
    raise _CapturedError(kwargs)


def _captured_kwargs(tmp_path, monkeypatch, device):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer import sft

    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    cfg = load_config_from_string(
        "\n".join(
            [
                f"base: {base}",
                "task: sft",
                "data:",
                "  train: x.jsonl",
                "  max_length: 64",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  lora:",
                "    r: 4",
                "    alpha: 8",
                "    target_modules: [q_proj, v_proj]",
                "output: ./out",
                "",
            ]
        )
    )
    wrapper = sft.SFTTrainerWrapper(cfg, device=device)
    # The tiny checkpoint always loads on the CPU, whatever device the wrapper names.
    monkeypatch.setattr(sft, "resolve_device_map", lambda _device: {"": "cpu"})
    monkeypatch.setattr("transformers.TrainingArguments", _recorder)
    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
        for _ in range(4)
    ]
    with pytest.raises(_CapturedError) as got:
        wrapper.setup({"train": rows})
    return got.value.kwargs


def test_cpu_device_sets_use_cpu(tmp_path, monkeypatch):
    kwargs = _captured_kwargs(tmp_path, monkeypatch, "cpu")
    assert kwargs.get("use_cpu") is True


def test_cuda_device_does_not_set_use_cpu(tmp_path, monkeypatch):
    kwargs = _captured_kwargs(tmp_path, monkeypatch, "cuda")
    assert "use_cpu" not in kwargs
