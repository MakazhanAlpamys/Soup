"""#1197: ``moe_lora`` with ``lora.use_vera`` is refused on a fused-expert MoE.

VeRA (peft ``VeraConfig``) wraps ``nn.Linear`` modules only and has no
``target_parameters`` path, so on a transformers-5 MoE, whose routed experts are 3-D
tensors on one ``experts`` module, ``moe_lora: true`` + ``use_vera: true`` adapted the
attention projections (and a shared expert) exactly as VeRA without the flag did, while
printing the ScatterMoE line. The maintainer's ruling: refuse when VeRA would wrap no
routed expert, on every such family, Mixtral and MiniMax included. A checkpoint that
keeps one ``nn.Linear`` per expert is not refused: VeRA wraps those.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from soup_cli.utils.moe import resolve_moe_lora_targets  # noqa: E402
from soup_cli.utils.peft_wiring import build_lora_config, build_peft_config_spec  # noqa: E402

_SMALL = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
)
_FUSED = ("qwen3_moe", "mixtral", "minimax", "qwen2_moe")


def _model(arch: str):
    tf = transformers
    if arch == "qwen3_moe":
        return tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(
            **_SMALL, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16))
    if arch == "mixtral":
        return tf.MixtralForCausalLM(tf.MixtralConfig(
            **_SMALL, num_local_experts=4, num_experts_per_tok=2))
    if arch == "minimax":
        return tf.MiniMaxForCausalLM(tf.MiniMaxConfig(
            **_SMALL, num_local_experts=4, num_experts_per_tok=2))
    if arch == "qwen2_moe":
        return tf.Qwen2MoeForCausalLM(tf.Qwen2MoeConfig(
            **_SMALL, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
            shared_expert_intermediate_size=16))
    if arch == "dense":
        return tf.LlamaForCausalLM(tf.LlamaConfig(**_SMALL))
    raise KeyError(arch)


class _PerExpertLinearMoE(torch.nn.Module):
    """The transformers-4 layout: one ``nn.Linear`` per expert, which VeRA can wrap."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="mixtral", num_local_experts=2)
        experts = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "gate_proj": torch.nn.Linear(8, 16),
                "up_proj": torch.nn.Linear(8, 16),
                "down_proj": torch.nn.Linear(16, 8),
            })
            for _ in range(2)
        ])
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({
            "self_attn": torch.nn.ModuleDict(
                {name: torch.nn.Linear(8, 8) for name in ("q_proj", "v_proj")}),
            "block_sparse_moe": torch.nn.ModuleDict(
                {"gate": torch.nn.Linear(8, 2), "experts": experts}),
        })])


def _tcfg(*, moe_lora: bool = True, use_vera: bool = True, dropout: float = 0.0):
    return SimpleNamespace(
        moe_lora=moe_lora,
        lora=SimpleNamespace(
            r=4, alpha=8, dropout=dropout, use_dora=False, use_rslora=False, use_vera=use_vera,
            rank_pattern=None, alpha_pattern=None, init_strategy="random",
            target_modules="auto", target_parameters=None,
        ),
    )


def _tuner_layers(peft_model) -> list[str]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [n for n, m in peft_model.named_modules() if isinstance(m, BaseTunerLayer)]


class TestTheFactBehindTheRefusal:
    def test_vera_cannot_reach_a_fused_expert(self):
        """Measured, not assumed: VeRA over the module names ``moe_lora`` resolves
        attaches to attention only on a fused Qwen3-MoE, 0 layers on ``experts``."""
        from peft import get_peft_model

        model = _model("qwen3_moe")
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = build_lora_config(_tcfg().lora, target_modules=targets, task_type="CAUSAL_LM")
        assert type(config).__name__ == "VeraConfig"
        attached = get_peft_model(model, config)
        layers = _tuner_layers(attached)
        assert len(layers) == 8  # q/k/v/o on two layers
        assert not [n for n in layers if ".experts" in n]

    def test_the_vera_spec_has_no_target_parameters(self):
        """The only route to a fused parameter is ``target_parameters``, and peft's
        ``VeraConfig`` has none, so a carried list could not help VeRA."""
        spec = build_peft_config_spec(
            _tcfg().lora, target_modules=["q_proj"], task_type="CAUSAL_LM",
            target_parameters=["mlp.experts.gate_up_proj"],
        )
        assert spec["peft_cls"] == "VeraConfig"
        assert "target_parameters" not in spec["init_kwargs"]


class TestVeraOnFusedExpertsIsRefused:
    @pytest.mark.parametrize("arch", _FUSED)
    def test_refused_naming_both_flags_and_the_parameters(self, arch):
        with pytest.raises(ValueError) as excinfo:
            resolve_moe_lora_targets(_model(arch), _tcfg(), ["q_proj"])
        message = str(excinfo.value)
        assert "training.moe_lora=true" in message and "lora.use_vera" in message
        assert "mlp.experts.gate_up_proj" in message and "mlp.experts.down_proj" in message
        assert "nn.Linear" in message
        assert "use_vera: false" in message
        assert "remove moe_lora" in message

    def test_refused_before_the_dropout_rule_speaks(self):
        """With both problems present the VeRA one is named: fixing dropout alone
        would not make the flag do anything."""
        with pytest.raises(ValueError, match="use_vera"):
            resolve_moe_lora_targets(_model("mixtral"), _tcfg(dropout=0.05), ["q_proj"])

    def test_through_a_real_trainer_setup(self, monkeypatch):
        """The refusal reaches ``DPOTrainerWrapper._setup_transformers`` with only the
        loaders stubbed (the maintainer's measurement on #1197 used this path)."""
        pytest.importorskip("trl")
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        tokenizer = SimpleNamespace(
            pad_token=None, eos_token="</s>", pad_token_id=0, chat_template="{{ x }}"
        )
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained",
            lambda *a, **k: _model("qwen3_moe"),
        )
        cfg = load_config_from_string(
            "base: org/tiny-moe\ntask: dpo\nbackend: transformers\n"
            "data:\n  train: x.jsonl\n  format: dpo\n"
            "training:\n  quantization: none\n  moe_lora: true\n"
            "  lora:\n    r: 4\n    alpha: 8\n    dropout: 0.0\n    use_vera: true\n"
        )
        wrapper = object.__new__(DPOTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper.trust_remote_code = False
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None
        with pytest.raises(ValueError, match="lora.use_vera"):
            wrapper._setup_transformers(cfg, cfg.training)


class TestWhatIsNotRefused:
    @pytest.mark.parametrize("arch", _FUSED)
    def test_lora_on_the_same_model_still_reaches_the_experts(self, arch):
        """The control that gives the message its way out: with ``use_vera: false``
        the fused parameters travel as ``target_parameters`` (#1590)."""
        targets = resolve_moe_lora_targets(_model(arch), _tcfg(use_vera=False), ["q_proj"])
        assert targets.target_parameters == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]

    def test_vera_without_the_flag_is_untouched(self):
        assert resolve_moe_lora_targets(_model("qwen3_moe"), _tcfg(moe_lora=False), ["q_proj"]) == [
            "q_proj"
        ]

    def test_vera_on_a_per_expert_linear_layout_is_not_refused(self):
        """VeRA wraps those experts, so the flag does what it says there."""
        targets = resolve_moe_lora_targets(_PerExpertLinearMoE(), _tcfg(), ["q_proj"])
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)
        assert not targets.target_parameters

    def test_vera_on_a_dense_model_is_the_documented_no_op(self):
        assert resolve_moe_lora_targets(_model("dense"), _tcfg(), ["q_proj"]) == ["q_proj"]
