"""#798 — the MoE flags only ever worked on SFT.

``moe_lora`` picks LoRA targets that cover the expert FFNs. Only ``sft`` and
``pretrain`` (and ``tts`` through ``super()``) read it, yet 8 DPO and 7 GRPO
shipped recipes set it, so those recipes trained attention-only LoRA while
saying otherwise. ``moe_expert_quant`` and ``train_router_only`` are read by
the SFT path only, and ``moe_aux_loss_coeff`` by SFT and pretrain.

The ruling on the issue was (b): fix the recipes rather than document that they
do not work. So the five preference/RL trainers wire ``moe_lora``, and the
flags no trainer outside a named list reads are refused at config load.

The wiring tests drive each trainer's real ``_setup_transformers`` over a real
tiny ``Qwen3MoeForCausalLM`` with real peft. Only the loaders are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.loader import load_config_from_string

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

_MOE_TASKS = ("dpo", "kto", "orpo", "simpo", "grpo")

#: The per-task minimum a config needs, beyond base/data.
_TASK_YAML = {
    "dpo": ("dpo", ""),
    "kto": ("kto", ""),
    "orpo": ("dpo", ""),
    "simpo": ("dpo", ""),
    "grpo": ("alpaca", "  reward_fn: length\n"),
}


def _tiny_moe():
    """A real Qwen3-MoE, ~10k parameters. Experts are fused 3-D weights here
    (``mlp.experts.gate_up_proj``), which is what peft adapts."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            max_position_embeddings=64,
        )
    )


def _config(task: str, *, moe_lora: bool, extra: str = "") -> object:
    data_format, training_extra = _TASK_YAML[task]
    return load_config_from_string(
        f"base: org/tiny-moe\n"
        f"task: {task}\n"
        f"backend: transformers\n"
        f"data:\n  train: x.jsonl\n  format: {data_format}\n"
        f"training:\n"
        f"  quantization: none\n"
        f"  moe_lora: {'true' if moe_lora else 'false'}\n"
        f"  lora:\n    r: 4\n    alpha: 8\n"
        f"{training_extra}{extra}"
    )


_WRAPPERS = {
    "dpo": ("soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    "kto": ("soup_cli.trainer.kto", "KTOTrainerWrapper"),
    "orpo": ("soup_cli.trainer.orpo", "ORPOTrainerWrapper"),
    "simpo": ("soup_cli.trainer.simpo", "SimPOTrainerWrapper"),
    "grpo": ("soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
}


def _tiny_dense():
    """A dense Qwen3 of the same size: the control for "MoE only"."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
    )


def _attach_lora(task: str, monkeypatch, *, moe_lora: bool, dense=False, expect_failure=False):
    """Run the trainer's own ``_setup_transformers`` far enough to attach LoRA.

    The model loader and tokenizer are stubbed; ``get_peft_model`` and the
    target-module resolution are the real ones, which is the point.
    """
    import importlib

    module = importlib.import_module(_WRAPPERS[task][0])
    wrapper_cls = getattr(module, _WRAPPERS[task][1])

    tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>", pad_token_id=0)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer
    )
    build = _tiny_dense if dense else _tiny_moe
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: build()
    )

    cfg = _config(task, moe_lora=moe_lora)
    wrapper = object.__new__(wrapper_cls)
    wrapper.config = cfg
    wrapper.device = "cpu"
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    try:
        wrapper._setup_transformers(cfg, cfg.training)
    except ValueError:
        # peft's "no target_modules" refusal: the control asserts it.
        if expect_failure:
            raise
        raise
    except Exception as exc:  # noqa: BLE001 — setup continues past the LoRA attach
        if not _adapted(wrapper.model or object()):
            raise AssertionError(f"{task}: setup failed before LoRA attach: {exc!r}") from exc
    return wrapper.model


def _adapted(model) -> list[str]:
    if not hasattr(model, "named_modules"):
        return []
    return sorted(
        name.rsplit(".lora_A", 1)[0]
        for name, _ in model.named_modules()
        if name.endswith("lora_A")
    )


class TestMoeLoraReachesTheExpertFfns:
    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_moe_lora_true_adapts_the_experts(self, task, monkeypatch):
        """The 15 shipped DPO/GRPO recipes say moe_lora: true and trained
        attention-only LoRA."""
        adapted = _adapted(_attach_lora(task, monkeypatch, moe_lora=True))
        experts = [name for name in adapted if "experts" in name]
        assert experts, f"{task}: no expert module carries an adapter: {adapted}"

    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_without_the_flag_the_attach_fails_on_this_model(self, task, monkeypatch):
        """The control the ruling asked for, and it is stronger than expected: on a
        fused-expert Qwen3-MoE, ``target_modules: auto`` (what all 15 recipes use)
        resolves to None, and peft refuses. So these recipes could not attach LoRA
        at all, rather than quietly training attention-only."""
        with pytest.raises(ValueError, match="target_modules"):
            _attach_lora(task, monkeypatch, moe_lora=False, expect_failure=True)

    @pytest.mark.parametrize("task", _MOE_TASKS)
    def test_a_non_moe_base_is_untouched_by_the_flag(self, task, monkeypatch):
        """The other control: the wiring must not change a dense model's targets,
        whatever the flag says."""
        adapted_off = _adapted(_attach_lora(task, monkeypatch, moe_lora=False, dense=True))
        adapted_on = _adapted(_attach_lora(task, monkeypatch, moe_lora=True, dense=True))
        assert adapted_off == adapted_on, (adapted_off, adapted_on)
        assert adapted_on, "the dense control adapted nothing"


class TestTheFlagsNoTrainerReadsAreRefused:
    @pytest.mark.parametrize("task", ["dpo", "grpo", "kto", "pretrain"])
    @pytest.mark.parametrize(
        "knob", ["  moe_expert_quant: nf4\n", "  train_router_only: true\n"]
    )
    def test_sft_only_knobs_are_refused_off_sft(self, task, knob):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                f"base: org/m\ntask: {task}\n"
                f"data:\n  train: x.jsonl\n  format: {_TASK_YAML.get(task, ('sft', ''))[0]}\n"
                f"training:\n  moe_lora: true\n{knob}"
                + ("  reward_fn: length\n" if task == "grpo" else "")
            )
        message = str(excinfo.value)
        assert task in message
        assert "sft" in message

    @pytest.mark.parametrize("knob", ["moe_expert_quant: nf4", "train_router_only: true"])
    def test_sft_still_accepts_them(self, knob):
        cfg = load_config_from_string(
            "base: org/m\ntask: sft\ndata: {train: x.jsonl}\n"
            f"training: {{moe_lora: true, {knob}}}\n"
        )
        assert cfg.task == "sft"

    @pytest.mark.parametrize("task", ["dpo", "grpo"])
    def test_a_non_default_aux_loss_coeff_is_refused(self, task):
        with pytest.raises(ValueError, match="moe_aux_loss_coeff"):
            load_config_from_string(
                f"base: org/m\ntask: {task}\n"
                f"data:\n  train: x.jsonl\n  format: {_TASK_YAML[task][0]}\n"
                "training:\n  moe_aux_loss_coeff: 0.05\n"
                + ("  reward_fn: length\n" if task == "grpo" else "")
            )

    @pytest.mark.parametrize("task", ["dpo", "grpo"])
    def test_the_default_aux_loss_coeff_still_loads(self, task):
        """Every shipped DPO/GRPO MoE recipe writes the default 0.01 explicitly,
        and a dumped config writes it too: refusing that would break them."""
        cfg = load_config_from_string(
            f"base: org/m\ntask: {task}\n"
            f"data:\n  train: x.jsonl\n  format: {_TASK_YAML[task][0]}\n"
            "training:\n  moe_aux_loss_coeff: 0.01\n"
            + ("  reward_fn: length\n" if task == "grpo" else "")
        )
        assert cfg.training.moe_aux_loss_coeff == 0.01

    @pytest.mark.parametrize("task", ["sft", "pretrain"])
    def test_a_non_default_aux_loss_coeff_is_accepted_where_it_is_read(self, task):
        cfg = load_config_from_string(
            f"base: org/m\ntask: {task}\ndata: {{train: x.jsonl}}\n"
            "training: {moe_aux_loss_coeff: 0.05}\n"
        )
        assert cfg.training.moe_aux_loss_coeff == 0.05


class TestShippedConfigs:
    def test_every_recipe_still_loads(self):
        from soup_cli.recipes.catalog import RECIPES

        for name, recipe in RECIPES.items():
            load_config_from_string(recipe.yaml_str), name

    def test_the_moe_recipes_are_the_ones_this_issue_names(self):
        """15, not the 8 the issue estimated: 8 dpo + 7 grpo."""
        import yaml

        from soup_cli.recipes.catalog import RECIPES

        tasks = []
        for recipe in RECIPES.values():
            cfg = yaml.safe_load(recipe.yaml_str)
            training = cfg.get("training") or {}
            if training.get("moe_lora") and cfg.get("task") in _MOE_TASKS:
                tasks.append(cfg["task"])
        assert sorted(tasks) == ["dpo"] * 8 + ["grpo"] * 7, sorted(tasks)

