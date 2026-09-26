"""Regression tests for #803: one LoRA-config path across every trainer."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_WRAPPERS = (
    ("dpo", "DPOTrainerWrapper"),
    ("kto", "KTOTrainerWrapper"),
    ("orpo", "ORPOTrainerWrapper"),
    ("simpo", "SimPOTrainerWrapper"),
    ("ipo", "IPOTrainerWrapper"),
    ("bco", "BCOTrainerWrapper"),
    ("grpo", "GRPOTrainerWrapper"),
    ("reward_model", "RewardModelTrainerWrapper"),
    ("pretrain", "PretrainTrainerWrapper"),
    ("sft", "SFTTrainerWrapper"),
)


def _config(task: str):
    from soup_cli.config.schema import SoupConfig

    training = {
        "quantization": "none",
        "lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "rank_pattern": {"q_proj": 4},
            "alpha_pattern": {"q_proj": 8},
        },
    }
    if task == "online_dpo":
        training["reward_model"] = "tiny-local-reward"
    return SoupConfig(
        base="tiny-local-llama",
        task=task,
        data={"train": "train.jsonl"},
        training=training,
    )


def _model(*, sequence_classification: bool = False):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=0,
    )
    cls = LlamaForSequenceClassification if sequence_classification else LlamaForCausalLM
    return cls(config).to(torch.float32)


def _tokenizer():
    return SimpleNamespace(
        pad_token=None,
        eos_token="</s>",
        chat_template=None,
    )


def _bare_wrapper(wrapper_cls, config):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper.config = config
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    return wrapper


@pytest.mark.parametrize(("module_name", "class_name"), _WRAPPERS)
def test_every_transformers_wrapper_builds_configured_per_module_rank_on_cpu(
    module_name: str,
    class_name: str,
) -> None:
    module = importlib.import_module(f"soup_cli.trainer.{module_name}")
    wrapper_cls = getattr(module, class_name)
    config = _config(module_name)
    model = _model(sequence_classification=module_name == "reward_model")
    wrapper = _bare_wrapper(wrapper_cls, config)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ),
    ):
        wrapper._setup_transformers(config, config.training)

    ranks = {
        name.rsplit(".", maxsplit=1)[-1]: module.lora_A["default"].weight.shape[0]
        for name, module in wrapper.model.named_modules()
        if name.endswith(("q_proj", "v_proj")) and hasattr(module, "lora_A")
    }
    assert ranks == {"q_proj": 4, "v_proj": 16}
    peft_config = wrapper.model.peft_config["default"]
    assert peft_config.rank_pattern == {"q_proj": 4}
    assert peft_config.alpha_pattern == {"q_proj": 8}


class _AdapterBuiltError(Exception):
    """Raised right after ``get_peft_model`` so ``setup`` stops before datasets and Trainers."""


def _stop_after_adapter(captured: dict):
    import peft

    real_get_peft_model = peft.get_peft_model

    def get_peft_model(model, lora_config, *args, **kwargs):
        captured["model"] = real_get_peft_model(model, lora_config, *args, **kwargs)
        raise _AdapterBuiltError

    return patch("peft.get_peft_model", get_peft_model)


def _assert_configured_ranks(model) -> None:
    ranks = {
        name.rsplit(".", maxsplit=1)[-1]: module.lora_A["default"].weight.shape[0]
        for name, module in model.named_modules()
        if name.endswith(("q_proj", "v_proj")) and hasattr(module, "lora_A")
    }
    assert ranks == {"q_proj": 4, "v_proj": 16}
    peft_config = model.peft_config["default"]
    assert peft_config.rank_pattern == {"q_proj": 4}
    assert peft_config.alpha_pattern == {"q_proj": 8}


def _config_with(task: str, data: dict | None = None, **training):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="tiny-local-llama",
        task=task,
        data={"train": "train.jsonl", **(data or {})},
        training={
            "quantization": "none",
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "rank_pattern": {"q_proj": 4},
                "alpha_pattern": {"q_proj": 8},
            },
            **training,
        },
    )


def _tiny_whisper():
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=64,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=16,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    return WhisperForConditionalGeneration(config).to(torch.float32)


def _drive_classifier():
    from soup_cli.trainer.classifier import ClassifierTrainerWrapper

    config = _config_with("classifier", num_labels=2, classifier_lora=True)
    wrapper = _bare_wrapper(ClassifierTrainerWrapper, config)
    with patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        return_value=_model(sequence_classification=True),
    ):
        wrapper.setup({"train": []})


def _drive_distill():
    from soup_cli.trainer.distill import DistillTrainerWrapper

    config = _config_with("distill", teacher_model="tiny-local-teacher")
    wrapper = _bare_wrapper(DistillTrainerWrapper, config)
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_model()):
        wrapper.setup({"train": []})


def _drive_embedding():
    import torch
    from transformers import LlamaModel

    from soup_cli.trainer.embedding import EmbeddingTrainerWrapper

    config = _config_with("embedding")
    wrapper = _bare_wrapper(EmbeddingTrainerWrapper, config)
    base_model = LlamaModel(_model().config).to(torch.float32)
    with patch("transformers.AutoModel.from_pretrained", return_value=base_model):
        wrapper._setup_transformers(config, config.training)


def _drive_ppo():
    from soup_cli.trainer.ppo import PPOTrainerWrapper

    config = _config_with("ppo", reward_model="tiny-local-reward")
    wrapper = _bare_wrapper(PPOTrainerWrapper, config)
    with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_model()):
        wrapper._setup_transformers(config, config.training)


def _drive_unlearn():
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    config = _config_with("unlearn", data={"forget_set": "forget.jsonl"}, unlearn_method="simnpo")
    wrapper = UnlearnTrainerWrapper(config, device="cpu")
    with patch(
        "soup_cli.utils.live_eval.load_model_and_tokenizer",
        return_value=(_model(), _tokenizer(), "cpu"),
    ):
        wrapper.setup()


def _drive_asr():
    from soup_cli.trainer.asr import AsrTrainerWrapper

    config = _config_with("asr", asr_lora=True)
    wrapper = AsrTrainerWrapper.__new__(AsrTrainerWrapper)
    wrapper.config = config
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    whisper = _tiny_whisper()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(), get_decoder_prompt_ids=lambda **_: None
    )
    with (
        patch("soup_cli.trainer.asr._load_autoconfig", return_value=whisper.config),
        patch("transformers.WhisperProcessor.from_pretrained", return_value=processor),
        patch(
            "transformers.WhisperForConditionalGeneration.from_pretrained",
            return_value=whisper,
        ),
    ):
        wrapper.setup({"train": []})


# #959: the trainers whose LoRA is built inside ``setup`` (or a load path the
# wrappers above do not share). Each is driven on CPU until ``get_peft_model``
# returns, then stopped, so the adapter is inspected without datasets, a
# teacher, a reference model or a Trainer.
_SETUP_DRIVERS = (
    ("asr", _drive_asr),
    ("classifier", _drive_classifier),
    ("distill", _drive_distill),
    ("embedding", _drive_embedding),
    ("ppo", _drive_ppo),
    ("unlearn", _drive_unlearn),
)


@pytest.mark.parametrize(("module_name", "drive"), _SETUP_DRIVERS)
def test_setup_path_trainers_build_configured_per_module_rank_on_cpu(module_name, drive) -> None:
    captured: dict = {}
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        _stop_after_adapter(captured),
        pytest.raises(_AdapterBuiltError),
    ):
        drive()

    _assert_configured_ranks(captured["model"])


def test_layer_streaming_builds_the_lora_config_with_its_patterns(monkeypatch, tmp_path) -> None:
    """#923 gave layer streaming pattern support; this holds it on CPU.

    Streaming never materialises the base model, so the check is on the
    ``LoraConfig`` handed to ``build_streamed_model``, which applies it to the
    streamed skeleton.
    """
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils.layer_stream import DEFAULT_STREAM_READ_AHEAD

    class _Wrapper(StreamingSetupMixin):
        def __init__(self):
            self.device = "cpu"
            self._trust_remote_code = False

        def _stream_budget_lines(self, *_args, **_kwargs):
            return (), None

    config = _config("sft")
    tcfg = SimpleNamespace(
        quantization="none",
        double_quant_on=True,
        stream_source="auto",
        stream_buffers=2,
        stream_disk_kind=None,
        stream_pin=None,
        seed=7,
        moe_lora=False,
        batch_size=1,
        gradient_accumulation_steps=1,
        stream_vram_probe=False,
        stream_vram_override=None,
        stream_read_ahead=DEFAULT_STREAM_READ_AHEAD,
        lora=config.training.lora,
    )
    cfg = SimpleNamespace(base="Qwen/Qwen3-0.6B", data=SimpleNamespace(max_length=64))
    model_cfg = SimpleNamespace(
        model_type="qwen3", hidden_size=64, num_hidden_layers=2, vocab_size=128
    )
    runtime = SimpleNamespace(
        stats=lambda: {
            "tier": "ram",
            "store_bytes": 0,
            "pinned": False,
            "n_layers": 2,
            "buffers": 2,
            "buffer_bytes": 8,
        }
    )
    captured: dict = {}

    def build_streamed_model(**kwargs):
        captured["lora_config"] = kwargs["lora_config"]
        return SimpleNamespace(), runtime

    fakes = {
        "transformers.AutoTokenizer.from_pretrained": lambda *_a, **_k: _tokenizer(),
        "transformers.AutoConfig.from_pretrained": lambda *_a, **_k: model_cfg,
        "soup_cli.utils.layer_shard.resolve_shard_dir": lambda *_a, **_k: str(tmp_path / "s"),
        "soup_cli.utils.layer_shard.shard_checkpoint": lambda *_a, **_k: SimpleNamespace(
            n_layers=2, total_params=100, quant="none", quant_specs={}
        ),
        "soup_cli.utils.layer_shard.source_weight_bytes": lambda *_a, **_k: 1024,
        "soup_cli.utils.spectrum_scan.resolve_model_weights": lambda *_a, **_k: str(tmp_path / "w"),
        "soup_cli.utils.layer_stream.free_ram_bytes": lambda: 1_000_000,
        "soup_cli.utils.layer_stream_runtime.build_meta_skeleton": (
            lambda *_a, **_k: SimpleNamespace()
        ),
        "soup_cli.utils.layer_stream_runtime.RamSource.layer_specs_from_shards": lambda *_a, **_k: [
            {"self_attn.q_proj.weight": ((4, 4), "float32")},
            {"self_attn.q_proj.weight": ((4, 4), "float32")},
        ],
        "soup_cli.utils.layer_stream_runtime.extras_resident_bytes": lambda *_a, **_k: 0,
        "soup_cli.utils.layer_stream_runtime.build_streamed_model": build_streamed_model,
        "soup_cli.utils.moe.detect_moe_model": lambda *_a, **_k: False,
        "soup_cli.utils.layer_stream.render_stream_panel": lambda *_a, **_k: "panel",
        "soup_cli.utils.layer_stream_runtime.expandable_segments_status": lambda: (True, ""),
    }
    for target, fake in fakes.items():
        monkeypatch.setattr(target, fake)

    _Wrapper()._setup_streaming_transformers(cfg, tcfg)

    lora_config = captured["lora_config"]
    assert lora_config.r == 16
    assert lora_config.rank_pattern == {"q_proj": 4}
    assert lora_config.alpha_pattern == {"q_proj": 8}


def test_every_build_lora_config_caller_is_covered_by_a_pattern_test() -> None:
    """A new caller of the shared builder must join one of the tables above."""
    from soup_cli import trainer

    trainer_dir = Path(trainer.__file__).parent
    callers = {
        path.stem
        for path in trainer_dir.glob("*.py")
        if "build_lora_config(" in path.read_text(encoding="utf-8")
    }
    covered = (
        {name for name, _ in _WRAPPERS}
        | {name for name, _ in _SETUP_DRIVERS}
        | {"online_dpo", "stream_setup"}
    )
    assert callers - covered == set(), f"no LoRA pattern test for: {sorted(callers - covered)}"


def test_online_dpo_preserves_patterns_in_deferred_peft_config() -> None:
    from soup_cli.trainer.online_dpo import OnlineDPOTrainerWrapper

    config = _config("online_dpo")
    wrapper = _bare_wrapper(OnlineDPOTrainerWrapper, config)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer()),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=_model()),
    ):
        wrapper._setup_transformers(config, config.training)

    assert wrapper.tokenizer.chat_template is not None
    assert wrapper.peft_config.rank_pattern == {"q_proj": 4}
    assert wrapper.peft_config.alpha_pattern == {"q_proj": 8}


def test_shared_builder_rejects_a_pattern_blind_config_double() -> None:
    from soup_cli.utils.peft_wiring import build_lora_config_kwargs

    incomplete = SimpleNamespace(
        r=16,
        alpha=32,
        dropout=0.0,
        use_dora=False,
        use_rslora=False,
    )
    with pytest.raises(AttributeError, match="rank_pattern"):
        build_lora_config_kwargs(
            incomplete,
            target_modules=["q_proj", "v_proj"],
            target_parameters=None,
            task_type="CAUSAL_LM",
        )


def test_trainers_cannot_construct_lora_config_outside_shared_builder() -> None:
    from soup_cli import trainer

    trainer_dir = Path(trainer.__file__).parent
    offenders = []
    for path in trainer_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "LoraConfig")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "LoraConfig")
            )
            for node in ast.walk(tree)
        ):
            offenders.append(path.name)
    offenders.sort()

    assert offenders == [], (
        "trainer modules must use soup_cli.utils.peft_wiring.build_lora_config; "
        f"direct LoraConfig construction found in: {offenders}"
    )
