"""The twelve non-SFT trainer wrappers never passed a dtype to
``from_pretrained`` (follow-on to issue #339 / PR #471).

PR #471 fixes ``SFTTrainerWrapper``: a frozen (LoRA) base keeps the
checkpoint's own dtype instead of the HF default of upcasting every load to
float32. The other trainer wrappers build the identical
``model_kwargs`` shape (``trust_remote_code`` + ``device_map`` + optional
``quantization_config``) with the same gap. Pretraining now has a LISA
full-fine-tuning branch and therefore uses the full-FT-aware resolver; the
remaining wrappers always load a frozen LoRA base.

``TestResolveFrozenBaseLoadDtype`` pins the shared helper in isolation.
``TestWrapperModelKwargsCarryDtype`` proves the wiring for real: each
wrapper's own ``_setup_transformers`` is called with the model class's
``from_pretrained`` mocked to capture kwargs and stop before any real load.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import SoupConfig


class _FakeCuda:
    """Stands in for ``torch.cuda`` (mirrors tests/test_issue385_stream_dtype.py's
    fixture of the same shape)."""

    def __init__(self, available: bool, bf16: bool, emulated: bool = True):
        self._available = available
        self._bf16 = bf16
        self._emulated = emulated

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        if not including_emulation:
            return self._bf16
        return self._bf16 or self._emulated

    def get_device_capability(self, device=None):
        return (8, 0) if self._bf16 else (7, 5)


@pytest.fixture()
def fake_torch_cuda(monkeypatch):
    import torch

    def apply(available: bool, bf16: bool, emulated: bool = True) -> _FakeCuda:
        fake = _FakeCuda(available, bf16, emulated)
        monkeypatch.setattr(torch, "cuda", fake)
        return fake

    return apply


class TestResolveFrozenBaseLoadDtype:
    def test_cpu_resolves_to_auto(self):
        from soup_cli.utils.gpu import resolve_frozen_base_load_dtype

        assert resolve_frozen_base_load_dtype("cpu") == "auto"

    def test_cuda_unavailable_resolves_to_auto(self, fake_torch_cuda):
        from soup_cli.utils.gpu import resolve_frozen_base_load_dtype

        fake_torch_cuda(available=False, bf16=False)
        assert resolve_frozen_base_load_dtype("cuda") == "auto"

    def test_bf16_capable_card_resolves_to_auto(self, fake_torch_cuda):
        from soup_cli.utils.gpu import resolve_frozen_base_load_dtype

        fake_torch_cuda(available=True, bf16=True)
        assert resolve_frozen_base_load_dtype("cuda") == "auto"

    def test_pre_ampere_card_resolves_to_explicit_float16(self, fake_torch_cuda):
        import torch

        from soup_cli.utils.gpu import resolve_frozen_base_load_dtype

        fake_torch_cuda(available=True, bf16=False, emulated=True)
        assert resolve_frozen_base_load_dtype("cuda") == torch.float16


class _StopAtLoadError(Exception):
    """Raised by the mocked ``from_pretrained`` to stop the wrapper right
    after it captures the model-load kwargs, so the rest of the method
    (LoRA construction, trainer wiring) never has to run against a fake
    model."""


def _base_kwargs(task: str, **training_overrides):
    training = {"quantization": "none", **training_overrides}
    return dict(base="some-model", task=task, data={"train": "./data.jsonl"}, training=training)


WRAPPER_CASES = [
    ("soup_cli.trainer.dpo", "DPOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("dpo")),
    ("soup_cli.trainer.kto", "KTOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("kto")),
    ("soup_cli.trainer.orpo", "ORPOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("orpo")),
    (
        "soup_cli.trainer.simpo",
        "SimPOTrainerWrapper",
        "AutoModelForCausalLM",
        _base_kwargs("simpo"),
    ),
    ("soup_cli.trainer.ipo", "IPOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("ipo")),
    ("soup_cli.trainer.bco", "BCOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("bco")),
    (
        "soup_cli.trainer.online_dpo",
        "OnlineDPOTrainerWrapper",
        "AutoModelForCausalLM",
        _base_kwargs("online_dpo", online_dpo_judge="test-judge"),
    ),
    ("soup_cli.trainer.grpo", "GRPOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("grpo")),
    ("soup_cli.trainer.ppo", "PPOTrainerWrapper", "AutoModelForCausalLM", _base_kwargs("ppo")),
    (
        "soup_cli.trainer.pretrain",
        "PretrainTrainerWrapper",
        "AutoModelForCausalLM",
        _base_kwargs("pretrain"),
    ),
    (
        "soup_cli.trainer.reward_model",
        "RewardModelTrainerWrapper",
        "AutoModelForSequenceClassification",
        _base_kwargs("reward_model"),
    ),
    (
        "soup_cli.trainer.embedding",
        "EmbeddingTrainerWrapper",
        "AutoModel",
        _base_kwargs("embedding"),
    ),
]


class TestWrapperModelKwargsCarryDtype:
    @pytest.mark.parametrize(
        "module_path,class_name,model_class_name,config_kwargs",
        WRAPPER_CASES,
        ids=[c[1] for c in WRAPPER_CASES],
    )
    def test_setup_transformers_passes_resolved_dtype(
        self, module_path, class_name, model_class_name, config_kwargs
    ):
        import importlib

        module = importlib.import_module(module_path)
        wrapper_cls = getattr(module, class_name)

        cfg = SoupConfig(**config_kwargs)
        wrapper = wrapper_cls(cfg, device="cpu")

        fake_tokenizer = MagicMock()
        load_mock = MagicMock(side_effect=_StopAtLoadError())

        # A sentinel, not the literal "auto" the resolver happens to return on
        # cpu: proves the wrapper wires the resolver's *return value* through
        # to from_pretrained, not merely that it passes cpu's own default.
        # Patched in the wrapper module's own namespace (each trainer imports
        # the name directly), mirroring the real dtype-resolution call site.
        sentinel = object()
        resolver_name = (
            "resolve_base_load_dtype"
            if class_name == "PretrainTrainerWrapper"
            else "resolve_frozen_base_load_dtype"
        )
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), \
             patch(f"transformers.{model_class_name}.from_pretrained", load_mock), \
             patch(f"{module_path}.{resolver_name}", return_value=sentinel):
            with pytest.raises(_StopAtLoadError):
                wrapper._setup_transformers(cfg, cfg.training)

        assert load_mock.call_args is not None, "from_pretrained was never called"
        assert load_mock.call_args.kwargs.get("torch_dtype") is sentinel


class TestRewardModelHeadUpcastToFp32:
    """PR #492 review: a SEQ_CLS LoRA base auto-adds a ``modules_to_save``
    wrapper around ``score`` that ``get_peft_model``'s adapter autocast does
    not reach, so a bf16-resolved frozen base left the reward head training
    in pure bf16 with no fp32 master weights. Uses a real tiny model (not a
    mock) since the bug is in peft's own autocast scope, not in this
    codebase's kwargs wiring.
    """

    def test_score_head_is_fp32_even_when_base_resolves_to_bf16(self):
        import torch

        from soup_cli.trainer.reward_model import RewardModelTrainerWrapper

        cfg = SoupConfig(
            base="hf-internal-testing/tiny-random-gpt2",
            task="reward_model",
            data={"train": "./data.jsonl"},
            training={"quantization": "none"},
        )
        wrapper = RewardModelTrainerWrapper(cfg, device="cpu")

        with patch(
            "soup_cli.trainer.reward_model.resolve_frozen_base_load_dtype",
            return_value=torch.bfloat16,
        ):
            wrapper._setup_transformers(cfg, cfg.training)

        head_params = [
            p for n, p in wrapper.model.named_parameters()
            if p.requires_grad and "score" in n
        ]
        assert head_params, "no trainable score-head parameter found on the PEFT model"
        assert all(p.dtype == torch.float32 for p in head_params), (
            "reward head stayed in the frozen base's bf16 load dtype"
        )
        # The base itself really did load bf16 (proves the fixture reproduces
        # the bug's precondition, not just that everything is fp32 already).
        base_dtype = next(
            p.dtype for n, p in wrapper.model.named_parameters()
            if not p.requires_grad and "lora_" not in n and "score" not in n
        )
        assert base_dtype == torch.bfloat16
