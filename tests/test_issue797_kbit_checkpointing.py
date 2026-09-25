"""#797: prepare_model_for_kbit_training must not force checkpointing on.

peft's `prepare_model_for_kbit_training` enables gradient checkpointing on the
model unconditionally when it is called with no `use_gradient_checkpointing`
kwarg (its own default is True). transformers.Trainer only ever *enables*
checkpointing when `args.gradient_checkpointing` is true and never disables
it, so once kbit-prep has turned it on for a 4-bit/8-bit/mxfp4 run,
`training.gradient_checkpointing: false` had no way to turn it back off.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.config.schema import SoupConfig

_TRAINER_DIR = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "trainer"

# Every file the issue names, plus sft.py's three call sites (base/vision/audio)
# and online_dpo.py's manual QLoRA-stabilization branch.
_KBIT_TRAINER_FILES = [
    "sft.py",
    "dpo.py",
    "kto.py",
    "orpo.py",
    "simpo.py",
    "ipo.py",
    "bco.py",
    "ppo.py",
    "grpo.py",
    "online_dpo.py",
    "reward_model.py",
    "pretrain.py",
    "embedding.py",
]


def _make_config(**overrides):
    base = {
        "base": "test-model",
        "data": {"train": "./data.jsonl", "format": "alpaca"},
    }
    base.update(overrides)
    return SoupConfig(**base)


# (task, module, class name): covers the base-model default trainer (sft),
# the DPO-family HF-checkpointing branch this issue was first reported against,
# and one non-streaming trainer (grpo) that never touches layer streaming at
# all, so the same should_enable_hf_gradient_checkpointing resolution has to
# hold with stream_layers structurally unset.
_KBIT_PREP_TRAINERS = [
    ("sft", "soup_cli.trainer.sft", "SFTTrainerWrapper"),
    ("dpo", "soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    ("grpo", "soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
]


class TestKbitPrepThreadsCheckpointingFlag:
    """Every transformers-backend trainer with a kbit-prep call site shares
    the same should_enable_hf_gradient_checkpointing resolution; parametrized
    over the default trainer (sft), the DPO family, and a non-streaming
    trainer (grpo) so the behavioral case isn't pinned to DPO alone.
    """

    def _run(self, monkeypatch, *, task: str, module: str, cls_name: str,
              gradient_checkpointing: bool):
        import importlib

        wrapper_cls = getattr(importlib.import_module(module), cls_name)

        cfg = _make_config(
            task=task,
            training={
                "quantization": "4bit",
                "gradient_checkpointing": gradient_checkpointing,
            },
        )

        captured_kwargs = {}

        class _Tokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

        class _Model:
            config = SimpleNamespace()

            def parameters(self):
                return []

        model = _Model()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer()),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
            AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
        )

        def _fake_kbit_prep(model_obj, **kwargs):
            captured_kwargs.update(kwargs)
            return model_obj

        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=_fake_kbit_prep,
        )

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )

        wrapper = object.__new__(wrapper_cls)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        return captured_kwargs

    @pytest.mark.parametrize("task,module,cls_name", _KBIT_PREP_TRAINERS)
    def test_gradient_checkpointing_false_reaches_kbit_prep(
        self, monkeypatch, task, module, cls_name
    ):
        # This is the exact bug: on main, kbit-prep is called with no kwarg
        # at all and peft defaults use_gradient_checkpointing to True, so a
        # config that explicitly asks for it off never gets it off.
        captured = self._run(
            monkeypatch, task=task, module=module, cls_name=cls_name,
            gradient_checkpointing=False,
        )
        assert captured.get("use_gradient_checkpointing") is False

    @pytest.mark.parametrize("task,module,cls_name", _KBIT_PREP_TRAINERS)
    def test_gradient_checkpointing_true_still_reaches_kbit_prep(
        self, monkeypatch, task, module, cls_name
    ):
        captured = self._run(
            monkeypatch, task=task, module=module, cls_name=cls_name,
            gradient_checkpointing=True,
        )
        assert captured.get("use_gradient_checkpointing") is True


class TestAllKbitCallSitesPassCheckpointingFlag:
    """Structural sweep: every prepare_model_for_kbit_training(...) call in
    every trainer must pass use_gradient_checkpointing explicitly, so a
    future call site cannot silently regress back to peft's own True
    default. AST-based (not regex) so a multi-line call is still matched.
    """

    @pytest.mark.parametrize("filename", _KBIT_TRAINER_FILES)
    def test_every_call_site_passes_the_flag(self, filename: str) -> None:
        src = (_TRAINER_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=filename)

        def _is_kbit_prep_call(node: ast.AST) -> bool:
            if not isinstance(node, ast.Call):
                return False
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            return name == "prepare_model_for_kbit_training"

        call_sites = [node for node in ast.walk(tree) if _is_kbit_prep_call(node)]
        assert call_sites, f"{filename} no longer calls prepare_model_for_kbit_training at all"

        for call in call_sites:
            kwarg_names = [kw.arg for kw in call.keywords]
            assert "use_gradient_checkpointing" in kwarg_names, (
                f"{filename}:{call.lineno} calls prepare_model_for_kbit_training "
                "without use_gradient_checkpointing, so peft's own True default "
                "applies regardless of training.gradient_checkpointing"
            )

    def test_every_call_site_covered_by_the_file_list(self) -> None:
        # Guards the file list itself: if a future trainer adds a new kbit-prep
        # call site in a file not listed above, this fails loudly instead of
        # the parametrized test above silently never running on it.
        found = set()
        for path in _TRAINER_DIR.glob("*.py"):
            if "prepare_model_for_kbit_training" in path.read_text(encoding="utf-8"):
                found.add(path.name)
        assert found == set(_KBIT_TRAINER_FILES), (
            f"trainer files calling prepare_model_for_kbit_training changed: "
            f"{found - set(_KBIT_TRAINER_FILES)} added, "
            f"{set(_KBIT_TRAINER_FILES) - found} removed: update _KBIT_TRAINER_FILES"
        )
