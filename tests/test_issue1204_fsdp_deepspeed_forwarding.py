"""#1204 — ``--fsdp`` / ``--deepspeed`` reach every trainer that stores them.

``soup train`` hands every transformers-backend wrapper the same
``fsdp_config`` / ``deepspeed_config``. ``prm`` and ``moe_lora_routing`` stored
``fsdp_config`` and never passed it to their ``TrainingArguments``; ``online_dpo``
dropped both. Each test builds the real wrapper over a tiny local checkpoint and
records the keyword arguments of the arguments class at the point the trainer
would be built, so no FSDP or DeepSpeed runtime is needed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")

from tests.test_issue1223_evaluate_val_split import _build, _tiny_llama_dir  # noqa: E402

FSDP = {"fsdp": "full_shard auto_wrap", "fsdp_config": {"min_num_params": 1234}}


class _CapturedError(Exception):
    def __init__(self, kwargs):
        super().__init__("captured")
        self.kwargs = kwargs


def _recorder(*args, **kwargs):
    raise _CapturedError(kwargs)


@pytest.mark.parametrize("task", ["prm", "moe_lora_routing"])
def test_train_hands_fsdp_to_training_arguments(tmp_path, monkeypatch, task):
    wrapper = _build(tmp_path, monkeypatch, task, n_val=0)
    wrapper.fsdp_config = dict(FSDP)
    monkeypatch.setattr("transformers.TrainingArguments", _recorder)
    with pytest.raises(_CapturedError) as got:
        wrapper.train()
    for key, value in FSDP.items():
        assert got.value.kwargs.get(key) == value, (task, key, got.value.kwargs.get(key))


def _online_dpo_wrapper(tmp_path, monkeypatch, **kwargs):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    monkeypatch.chdir(tmp_path)
    base = _tiny_llama_dir(tmp_path)
    cfg = load_config_from_string(
        f"base: {base}\n"
        "task: online_dpo\n"
        "data:\n  train: x.jsonl\n  max_length: 64\n"
        "training:\n"
        '  online_dpo_judge: "ollama://m"\n'
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  lora:\n    r: 4\n    alpha: 8\n    target_modules: [q_proj, v_proj]\n"
        "output: ./out\n"
    )
    return OnlineDPOTrainerWrapper(cfg, device="cpu", **kwargs)


def test_online_dpo_config_gets_fsdp_and_deepspeed(tmp_path, monkeypatch):
    from soup_cli.trainer import _trl_compat

    real = _trl_compat.resolve_trl_symbol

    def resolve(name, *args, **kwargs):
        return _recorder if name == "OnlineDPOConfig" else real(name, *args, **kwargs)

    monkeypatch.setattr(_trl_compat, "resolve_trl_symbol", resolve)
    wrapper = _online_dpo_wrapper(
        tmp_path, monkeypatch, fsdp_config=dict(FSDP), deepspeed_config="ds_config.json"
    )
    rows = [{"messages": [{"role": "user", "content": p}]} for p in ("hi", "hello", "yo", "hey")]
    with pytest.raises(_CapturedError) as got:
        wrapper.setup({"train": rows})
    kwargs = got.value.kwargs
    assert kwargs.get("deepspeed") == "ds_config.json"
    for key, value in FSDP.items():
        assert kwargs.get(key) == value, (key, kwargs.get(key))


def test_online_dpo_without_either_passes_no_extra_keys(tmp_path, monkeypatch):
    from soup_cli.trainer import _trl_compat

    real = _trl_compat.resolve_trl_symbol

    def resolve(name, *args, **kwargs):
        return _recorder if name == "OnlineDPOConfig" else real(name, *args, **kwargs)

    monkeypatch.setattr(_trl_compat, "resolve_trl_symbol", resolve)
    wrapper = _online_dpo_wrapper(tmp_path, monkeypatch)
    rows = [{"messages": [{"role": "user", "content": p}]} for p in ("hi", "hello")]
    with pytest.raises(_CapturedError) as got:
        wrapper.setup({"train": rows})
    assert got.value.kwargs.get("deepspeed") is None
    assert "fsdp" not in got.value.kwargs
