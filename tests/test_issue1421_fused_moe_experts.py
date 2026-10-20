"""#1421: fused MoE experts are found by their parameters, not by per-expert module names.

On transformers 5 an MoE layer keeps all its experts as two 3-D parameters on one
module (``mlp.experts.gate_up_proj`` / ``down_proj``). Soup's discovery looked for one
``nn.Linear`` per expert, so ``moe_lora`` adapted attention only on Mixtral, MiniMax,
Qwen2-MoE and Qwen3.5-MoE (the experts were reached only where peft's own name
conversion happened to rewrite the module names), and ``moe_expert_quant`` printed
``applied to 0 expert Linear block(s)`` as a success line.

Now ``resolve_moe_lora_targets`` hands the fused parameters to peft as
``target_parameters``, carried on the targets it returns and merged by
``build_lora_config`` (the one call every trainer makes before ``get_peft_model``), and
``moe_expert_quant`` refuses a fused-expert model by name.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from soup_cli.utils import moe as moe_mod  # noqa: E402
from soup_cli.utils.moe import (  # noqa: E402
    find_fused_expert_parameters,
    get_moe_target_modules,
    resolve_moe_lora_targets,
)
from soup_cli.utils.moe_quant import (  # noqa: E402
    apply_moe_expert_quant,
    apply_moe_expert_quant_if_configured,
)
from soup_cli.utils.peft_wiring import (  # noqa: E402
    build_lora_config,
    resolve_lora_target_modules,
)

_SMALL = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    pad_token_id=0,
)
_ROUTED = dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16)

_FUSED_FAMILIES = ("mixtral", "minimax", "qwen2_moe", "qwen3_5_moe", "qwen3_moe", "olmoe")
#: The families the issue measured at 0 routed-expert adapters on ``main``.
_WERE_ATTENTION_ONLY = ("mixtral", "minimax", "qwen2_moe", "qwen3_5_moe")


def _model(arch: str):
    tf = transformers
    if arch == "mixtral":
        return tf.MixtralForCausalLM(
            tf.MixtralConfig(**_SMALL, num_local_experts=4, num_experts_per_tok=2)
        )
    if arch == "minimax":
        return tf.MiniMaxForCausalLM(
            tf.MiniMaxConfig(**_SMALL, num_local_experts=4, num_experts_per_tok=2)
        )
    if arch == "qwen2_moe":
        return tf.Qwen2MoeForCausalLM(
            tf.Qwen2MoeConfig(**_SMALL, **_ROUTED, shared_expert_intermediate_size=16)
        )
    if arch == "qwen3_5_moe":
        return tf.Qwen3_5MoeForCausalLM(
            tf.Qwen3_5MoeTextConfig(**_SMALL, **_ROUTED, shared_expert_intermediate_size=16)
        )
    if arch == "qwen3_moe":
        return tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(**_SMALL, **_ROUTED))
    if arch == "olmoe":
        return tf.OlmoeForCausalLM(tf.OlmoeConfig(**_SMALL, **_ROUTED))
    if arch == "dense":
        return tf.LlamaForCausalLM(tf.LlamaConfig(**_SMALL))
    raise KeyError(arch)


class _PerExpertLinearMoE(torch.nn.Module):
    """The transformers-4 layout: one ``nn.Linear`` per expert under ``experts.N``."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="mixtral", num_local_experts=2)
        experts = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "gate_proj": torch.nn.Linear(8, 16),
                        "up_proj": torch.nn.Linear(8, 16),
                        "down_proj": torch.nn.Linear(16, 8),
                    }
                )
                for _ in range(2)
            ]
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "self_attn": torch.nn.ModuleDict(
                            {name: torch.nn.Linear(8, 8) for name in ("q_proj", "v_proj")}
                        ),
                        "block_sparse_moe": torch.nn.ModuleDict(
                            {"gate": torch.nn.Linear(8, 2), "experts": experts}
                        ),
                    }
                )
            ]
        )


def _tcfg(*, moe_lora: bool = True, dropout: float = 0.0, moe_expert_quant=None):
    return SimpleNamespace(
        moe_lora=moe_lora,
        moe_expert_quant=moe_expert_quant,
        lora=SimpleNamespace(
            r=4,
            alpha=8,
            dropout=dropout,
            use_dora=False,
            use_rslora=False,
            use_vera=False,
            rank_pattern=None,
            alpha_pattern=None,
            init_strategy="random",
            target_modules="auto",
            target_parameters=None,
        ),
    )


def _attach(model, tcfg):
    """The calls every MoE-wired trainer makes, in order, then peft's attach."""
    from peft import get_peft_model

    targets = resolve_lora_target_modules(model, tcfg.lora.target_modules)
    targets = resolve_moe_lora_targets(model, tcfg, targets, None)
    return get_peft_model(
        model, build_lora_config(tcfg.lora, target_modules=targets, task_type="CAUSAL_LM")
    )


def _tuner_layers(peft_model) -> list[str]:
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [n for n, m in peft_model.named_modules() if isinstance(m, BaseTunerLayer)]


def _expert_layers(peft_model) -> list[str]:
    return [n for n in _tuner_layers(peft_model) if ".experts" in n]


class TestFusedExpertsAreFound:
    @pytest.mark.parametrize("arch", _FUSED_FAMILIES)
    def test_the_fused_parameters_are_named_as_peft_suffixes(self, arch):
        found = find_fused_expert_parameters(_model(arch))
        assert found == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"], (arch, found)

    def test_the_per_expert_linear_layout_has_none(self):
        """The control: with one Linear per expert there is nothing fused, and the
        module-name discovery still finds the projections."""
        model = _PerExpertLinearMoE()
        assert find_fused_expert_parameters(model) == []
        targets = get_moe_target_modules(model)
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)

    def test_a_dense_model_has_none(self):
        assert find_fused_expert_parameters(_model("dense")) == []

    def test_a_shared_expert_is_not_a_fused_routed_expert(self):
        """Qwen2-MoE's ``mlp.shared_expert.gate_proj.weight`` is 2-D and lives on an
        ordinary Linear; only the 3-D routed tensors count."""
        found = find_fused_expert_parameters(_model("qwen2_moe"))
        assert not any("shared" in name for name in found), found

    def test_a_model_without_parameters_is_tolerated(self):
        """Several suites drive the resolver with a namespace or a mock standing in
        for the model; those have no experts and must not raise."""
        assert find_fused_expert_parameters(SimpleNamespace(config=None)) == []
        from unittest.mock import MagicMock

        assert find_fused_expert_parameters(MagicMock()) == []


class TestMoeLoraReachesTheExperts:
    @pytest.mark.parametrize("arch", _FUSED_FAMILIES)
    def test_routed_experts_get_adapters_on_every_fused_family(self, arch):
        attached = _attach(_model(arch), _tcfg())
        experts = _expert_layers(attached)
        # two parameters per layer, wrapped one inside the other, on two layers
        assert len(experts) == 4, (arch, experts)

    @pytest.mark.parametrize("arch", _WERE_ATTENTION_ONLY)
    def test_the_expert_adapters_receive_gradient(self, arch):
        """The adapters are live, not decorative: one backward pass puts a non-zero
        gradient on every expert LoRA tensor of the families that trained attention
        only before this fix."""
        attached = _attach(_model(arch), _tcfg())
        ids = torch.randint(1, 64, (1, 6))
        # no cache: a two-layer Qwen3.5 stand-in is all linear attention, and its
        # cache object cannot report a sequence length for that
        attached(input_ids=ids, labels=ids, use_cache=False).loss.backward()
        expert_lora = [
            (n, p) for n, p in attached.named_parameters()
            if p.requires_grad and ".experts" in n and "lora_" in n
        ]
        assert len(expert_lora) == 8, [n for n, _ in expert_lora]  # A and B, 2 params, 2 layers
        assert all(p.grad is not None for _, p in expert_lora)
        # lora_B starts at zero, so lora_A's gradient is zero on the first step by
        # construction; lora_B's is the one that shows the experts are in the graph.
        b_grads = [p.grad.abs().sum().item() for n, p in expert_lora if "lora_B" in n]
        assert len(b_grads) == 4 and all(g > 0 for g in b_grads), b_grads

    def test_qwen35_keeps_the_linear_attention_targets_auto_picks(self):
        """``auto`` resolves Qwen3.5 to ``q_proj, v_proj, in_proj_qkv, out_proj``;
        ``moe_lora`` used to replace that list, dropping the linear-attention
        projections. They are kept, and the experts are added."""
        attached = _attach(_model("qwen3_5_moe"), _tcfg())
        layers = _tuner_layers(attached)
        assert any(n.endswith("linear_attn.in_proj_qkv") for n in layers), layers
        assert any(n.endswith("linear_attn.out_proj") for n in layers), layers
        assert _expert_layers(attached)

    def test_without_the_flag_no_expert_is_adapted(self):
        """The control: the expert adapters are the flag's doing. Explicit module
        targets, because ``auto`` on Mixtral is peft's own (failing) default and
        not what is under test here."""
        tcfg = _tcfg(moe_lora=False)
        tcfg.lora.target_modules = ["q_proj", "v_proj"]
        attached = _attach(_model("mixtral"), tcfg)
        assert _tuner_layers(attached) and not _expert_layers(attached)

    def test_a_dense_model_is_untouched(self):
        model = _model("dense")
        targets = resolve_moe_lora_targets(model, _tcfg(), ["q_proj"])
        assert targets == ["q_proj"]
        assert getattr(targets, "target_parameters", None) in (None, [])

    def test_the_per_expert_linear_layout_still_resolves_module_names(self):
        """Control for the old layout: no ``target_parameters`` are invented, and the
        projections are still named as modules, as before this change."""
        targets = resolve_moe_lora_targets(_PerExpertLinearMoE(), _tcfg(), ["q_proj"])
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)
        assert not getattr(targets, "target_parameters", None)

    def test_the_resolved_targets_carry_the_parameters_into_the_lora_config(self):
        model = _model("mixtral")
        targets = resolve_moe_lora_targets(model, _tcfg(), ["q_proj"])
        assert targets.target_parameters == [
            "mlp.experts.down_proj", "mlp.experts.gate_up_proj"
        ]
        config = build_lora_config(_tcfg().lora, target_modules=targets, task_type="CAUSAL_LM")
        assert set(config.target_parameters) >= set(targets.target_parameters)
        assert "q_proj" in config.target_modules

    def test_explicit_target_parameters_are_merged_not_replaced(self):
        model = _model("mixtral")
        targets = resolve_moe_lora_targets(model, _tcfg(), ["q_proj"])
        config = build_lora_config(
            _tcfg().lora, target_modules=targets, task_type="CAUSAL_LM",
            target_parameters=["model.layers.0.self_attn.q_proj.weight"],
        )
        assert "model.layers.0.self_attn.q_proj.weight" in config.target_parameters
        assert "mlp.experts.gate_up_proj" in config.target_parameters

    def test_the_console_line_names_the_fused_parameters(self):
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=400, color_system=None)
        resolve_moe_lora_targets(_model("mixtral"), _tcfg(), ["q_proj"], console)
        out = buf.getvalue()
        assert "ScatterMoE LoRA" in out
        assert "mlp.experts.gate_up_proj" in out and "mlp.experts.down_proj" in out


class TestTheDropoutRefusalFollowsTheTargets:
    """peft adapts a fused parameter through ``lora.ParamWrapper``, which refuses any
    dropout. Now that Mixtral and MiniMax are targeted explicitly, they are refused
    with a non-zero dropout too; #798's test pinned the opposite while their experts
    were out of reach."""

    @pytest.mark.parametrize("arch", _FUSED_FAMILIES)
    def test_dropout_on_a_fused_family_is_refused_naming_the_flag(self, arch):
        with pytest.raises(ValueError) as excinfo:
            resolve_moe_lora_targets(_model(arch), _tcfg(dropout=0.05), ["q_proj"])
        message = str(excinfo.value)
        assert "training.moe_lora=true needs training.lora.dropout: 0.0" in message
        assert "lora.ParamWrapper does not work with lora_dropout != 0" in message

    def test_the_per_expert_linear_layout_keeps_dropout(self):
        targets = resolve_moe_lora_targets(_PerExpertLinearMoE(), _tcfg(dropout=0.05), None)
        assert targets


class TestMoeExpertQuantRefusesFusedExperts:
    def test_the_expert_scan_finds_the_fused_block(self):
        """The issue's third criterion: ``_find_expert_linears`` is not empty on a
        transformers-5 fused model. The block comes back as itself, not as a
        Linear, which is what the quant path keys its refusal on."""
        from soup_cli.utils.moe_quant import _find_expert_linears

        found = _find_expert_linears(_model("mixtral"))
        assert [name for name, _ in found] == [
            "model.layers.0.mlp.experts", "model.layers.1.mlp.experts"
        ]
        assert not any(isinstance(module, torch.nn.Linear) for _, module in found)

    def test_a_fused_model_is_refused_by_name_before_any_cuda_check(self):
        with pytest.raises(ValueError) as excinfo:
            apply_moe_expert_quant(_model("mixtral"), "nf4")
        message = str(excinfo.value)
        assert "moe_expert_quant" in message and "mixtral" in message
        assert "mlp.experts.gate_up_proj" in message
        assert "nn.Linear" in message

    def test_the_hook_never_reports_zero_as_success(self):
        """A model with no expert Linear at all (the dense control) used to print
        ``applied to 0 expert Linear block(s)``."""
        with pytest.raises(ValueError, match="matched no expert"):
            apply_moe_expert_quant_if_configured(_model("dense"), _tcfg(moe_expert_quant="nf4"))

    def test_the_hook_is_a_no_op_when_unset(self):
        apply_moe_expert_quant_if_configured(_model("mixtral"), _tcfg(moe_expert_quant=None))

    def test_the_per_expert_linear_layout_still_reaches_the_quant_path(self):
        """Control: the old layout is found by the Linear scan, and the fused-expert
        refusal does not fire on it. What happens next (the bitsandbytes / CUDA
        gate, or the real swap) is hardware-dependent and already pinned by
        ``tests/test_v07120.py``, so it is not repeated here."""
        from soup_cli.utils.moe_quant import _find_expert_linears, _refuse_fused_experts

        model = _PerExpertLinearMoE()
        assert len(_find_expert_linears(model)) == 6
        _refuse_fused_experts(model, "nf4")  # no raise: nothing fused here


class TestTheIssueScriptThroughTheSftTrainer:
    """The issue's reproduction, on its tiny Mixtral, through ``SFTTrainerWrapper``."""

    def _setup(self, tmp_path, monkeypatch, *, extra: str = ""):
        pytest.importorskip("trl")
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        words = ["<pad>", "<s>", "</s>", "<unk>", "<|user|>", "<|assistant|>", "hello", "hi"]
        raw = Tokenizer(
            models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<unk>")
        )
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=raw, bos_token="<s>", eos_token="</s>",
            pad_token="<pad>", unk_token="<unk>",
        )
        tok.chat_template = (
            "{% for m in messages %}{{ '<|' + m['role'] + '|> ' + m['content'] + ' </s> ' }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|> ' }}{% endif %}"
        )
        model = transformers.MixtralForCausalLM(
            transformers.MixtralConfig(
                vocab_size=len(words), hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
                num_local_experts=4, num_experts_per_tok=2,
            )
        )
        base = tmp_path / "tiny_moe"
        model.save_pretrained(base)
        tok.save_pretrained(base)
        monkeypatch.chdir(tmp_path)
        cfg = load_config_from_string(
            "base: ./tiny_moe\ntask: sft\n"
            "data: {train: ./unused.jsonl, format: chatml, max_length: 64}\n"
            "training:\n  epochs: 1\n  batch_size: 1\n  quantization: none\n"
            f"  moe_lora: true\n{extra}"
            "  lora: {r: 4, alpha: 8, dropout: 0.0}\n"
            "output: ./out_moe\n"
        )
        chat = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [{"messages": chat}] * 4})
        return wrapper

    def test_moe_lora_adapts_the_routed_experts(self, tmp_path, monkeypatch):
        wrapper = self._setup(tmp_path, monkeypatch)
        adapted = sorted(
            {n.split(".lora_")[0] for n, p in wrapper.model.named_parameters() if p.requires_grad}
        )
        assert any(n.endswith("mlp.experts") or ".mlp.experts." in n for n in adapted), adapted

    def test_moe_expert_quant_is_refused_instead_of_applied_to_zero(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="fused") as excinfo:
            self._setup(tmp_path, monkeypatch, extra="  moe_expert_quant: nf4\n")
        assert "mixtral" in str(excinfo.value)


class TestTheAttachPreflightSeesTheExperts:
    def test_trainer_lora_config_carries_the_fused_parameters(self):
        """``soup recipes verify`` builds the trainer's adapter config through this
        helper; it must report the experts the trainer now adapts."""
        from soup_cli.utils.attach_preflight import trainer_lora_config

        model = _model("mixtral")
        cfg = SimpleNamespace(task="sft", modality="text", training=_tcfg())
        peft_config, _targets = trainer_lora_config(model, cfg, None)
        assert "mlp.experts.gate_up_proj" in peft_config.target_parameters


class TestNothingElseMoved:
    def test_get_moe_target_modules_is_unchanged_for_fused_models(self):
        """The module-name half keeps its fallback; the fused parameters travel
        beside it, not instead of it."""
        targets = get_moe_target_modules(_model("mixtral"))
        assert targets[:4] == list(moe_mod.STANDARD_TARGET_MODULES)
