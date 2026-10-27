"""Boundary target qualification from #2082 by @ybai08 (#1048 item 2).

Saving and resuming now supports the formerly refused untied targets. Keep the
same literal, regex, dotted, preference and tied controls when lifting that guard.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def _config(target_modules, task="sft"):
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": "meta-llama/Llama-3.1-8B",
                "task": task,
                "data": {"train": "data.jsonl"},
                "training": {
                    "batch_size": 2,
                    "stream_layers": True,
                    "lora": {"target_modules": target_modules},
                },
            }
        )
    )


class TestTheConfigNoLongerDecides:
    @pytest.mark.parametrize(
        "target_modules",
        [["q_proj", "v_proj", "lm_head"], ["q_proj", "embed_tokens"], "lm_head"],
    )
    def test_a_boundary_target_loads(self, target_modules):
        cfg = _config(target_modules)

        assert cfg.training.lora.target_modules == target_modules


# The real streamed setup on a tiny Llama, as in test_v07204.


def _setup(tmp_path, monkeypatch, *, tie, target_modules, task="sft", device="cpu", **training):
    from tests.test_v07204 import _build_streamed_wrapper

    wrapper, _, _ = _build_streamed_wrapper(
        tmp_path,
        monkeypatch,
        task=task,
        n_layers=2,
        device=device,
        tie=tie,
        lora={"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": target_modules},
        **training,
    )
    return wrapper


def _counted_installs(monkeypatch):
    """Each call to install_streaming, which allocates the streaming buffers."""
    from soup_cli.utils import layer_stream_runtime

    installs = []
    real_install = layer_stream_runtime.install_streaming

    def counting_install(*args, **kwargs):
        installs.append(True)
        return real_install(*args, **kwargs)

    monkeypatch.setattr(layer_stream_runtime, "install_streaming", counting_install)
    return installs


class TestAnUntiedBoundaryTargetBuildsAndSaves:
    @pytest.mark.parametrize(
        ("target_modules", "named"),
        [
            (["q_proj", "v_proj", "lm_head"], "lm_head"),
            (["q_proj", "embed_tokens"], "embed_tokens"),
            (".*(q_proj|lm_head)", "lm_head"),
            (["q_proj", "embed_tokens", "lm_head"], "embed_tokens, lm_head"),
        ],
    )
    def test_every_spelling(self, tmp_path, monkeypatch, target_modules, named):
        installs = _counted_installs(monkeypatch)

        wrapper = _setup(tmp_path, monkeypatch, tie=False, target_modules=target_modules)
        try:
            wrapper.model.save_pretrained(tmp_path / "adapter")
            for target in named.split(", "):
                assert any(target in key and "lora_" in key for key in wrapper.model.state_dict())
        finally:
            wrapper._close_stream_runtime()
        assert installs == [True]

    def test_the_dotted_embedding_path_on_purpose(self, tmp_path, monkeypatch):
        # The original guard's deliberately named dotted-path regression.
        installs = _counted_installs(monkeypatch)
        wrapper = _setup(
            tmp_path, monkeypatch, tie=False, target_modules=["q_proj", "model.embed_tokens"]
        )
        try:
            wrapper.model.save_pretrained(tmp_path / "adapter")
            assert any(
                "model.embed_tokens.lora_embedding_A" in key for key in wrapper.model.state_dict()
            )
        finally:
            wrapper._close_stream_runtime()
        assert installs == [True]

    def test_a_preference_task_too(self, tmp_path, monkeypatch):
        installs = _counted_installs(monkeypatch)

        wrapper = _setup(
            tmp_path, monkeypatch, tie=False, target_modules=".*(q_proj|lm_head)", task="dpo"
        )
        try:
            wrapper.model.save_pretrained(tmp_path / "adapter")
            assert wrapper.trainer.ref_model is None
        finally:
            wrapper._close_stream_runtime()
        assert installs == [True]

    def test_without_a_boundary_target_it_builds(self, tmp_path, monkeypatch):
        installs = _counted_installs(monkeypatch)

        wrapper = _setup(tmp_path, monkeypatch, tie=False, target_modules=["q_proj", "v_proj"])

        wrapper._close_stream_runtime()
        assert installs == [True]


def _lora_parameters(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


class TestATiedBoundaryTargetTrainsAndSaves:
    """A tied checkpoint streams neither weight, so nothing refuses it: a step
    with a save writes every adapter tensor through the Trainer's default
    save, and a fresh build loads each of them unchanged."""

    @pytest.mark.parametrize("target", ["lm_head", "embed_tokens"])
    def test_one_step_a_checkpoint_and_a_reload(self, tmp_path, monkeypatch, target):
        import torch
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from safetensors.torch import load_file

        targets = ["q_proj", "v_proj", target]
        wrapper = _setup(
            tmp_path / "run",
            monkeypatch,
            tie=True,
            target_modules=targets,
            # The accelerator, as test_v07204's training tests do: the Trainer
            # picks it whatever the wrapper was given.
            device=None,
            save_steps=1,
        )
        wrapper.trainer.args.max_steps = 1
        try:
            wrapper.trainer.train()
            trained_state = {
                k: v.detach().cpu() for k, v in get_peft_model_state_dict(wrapper.model).items()
            }
            trained = _lora_parameters(wrapper.model)
        finally:
            wrapper._close_stream_runtime()

        checkpoint = tmp_path / "run" / "out" / "checkpoint-1"
        saved = load_file(str(checkpoint / "adapter_model.safetensors"))
        assert sorted(saved) == sorted(trained_state)
        for key, value in saved.items():
            assert torch.equal(trained_state[key], value), key
        boundary = [k for k in saved if target in k and "lora_" in k]
        assert len(boundary) == 2, sorted(saved)
        # Two decoder layers x (q_proj, v_proj) x (A, B), plus the boundary's A and B.
        assert len(trained) == 10, sorted(trained)

        fresh = _setup(
            tmp_path / "fresh", monkeypatch, tie=True, target_modules=targets, device="cpu"
        )
        try:
            # Off every value a dropped tensor could keep: lora_B starts at
            # zero and lora_A from the same seed, so either could match the
            # trained one without being loaded.
            torch.manual_seed(1234)
            with torch.no_grad():
                for name, param in fresh.model.named_parameters():
                    if "lora_" in name:
                        param.copy_(torch.randn_like(param) + 5.0)
            set_peft_model_state_dict(fresh.model, load_peft_weights(str(checkpoint)))
            loaded = _lora_parameters(fresh.model)
        finally:
            fresh._close_stream_runtime()

        assert sorted(loaded) == sorted(trained)
        for name, value in trained.items():
            assert torch.equal(loaded[name], value), f"{name!r} did not load unchanged"
