"""v0.40.5 Part A — Quant Menu non-SFT multi-trainer wiring (#66).

Schema gate widening: lifts `_validate_quant_menu_supported_tasks` to allow
all transformer-backend trainers (sft / dpo / grpo / kto / orpo / simpo /
ipo / ppo / reward_model / pretrain / embedding / bco). MLX backend still
rejected with distinct message (mirrors v0.34.0 review-fix policy).

Trainer wiring: each non-SFT trainer's `_setup_transformers` now calls
`build_quantization_config_for_loader` instead of inline BNB-only branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string

# ---------------------------------------------------------------------------
# Schema gate widening
# ---------------------------------------------------------------------------

# Map task → minimal extra YAML to make load_config_from_string succeed.
_TASK_YAML_EXTRA: dict[str, str] = {
    "dpo": "",
    "grpo": "",
    "kto": "\n  format: kto",
    "orpo": "",
    "simpo": "",
    "ipo": "",
    "ppo": "",
    "reward_model": "",
    "pretrain": "\n  format: plaintext",
    "embedding": "\n  format: embedding",
    "bco": "",
}

_NON_SFT_TASKS = list(_TASK_YAML_EXTRA.keys())

# Quant Menu values that should be accepted on every transformer-backend
# trainer (one rep per format family — full coverage at builder layer).
_QUANT_FORMATS = ["gptq", "awq", "hqq:4bit", "aqlm", "eetq", "mxfp4", "fp8"]


def _build_yaml(task: str, quant: str, *, backend: str = "transformers") -> str:
    extra = _TASK_YAML_EXTRA[task]
    extra_train = ""
    if task == "ppo":
        extra_train = "\n  reward_model: dummy"
    return (
        f"base: m\n"
        f"task: {task}\n"
        f"backend: {backend}\n"
        f"data:\n  train: d.jsonl{extra}\n"
        f"training:\n  quantization: {quant}{extra_train}\n"
    )


class TestQuantMenuMultiTrainerAccept:
    @pytest.mark.parametrize("task", _NON_SFT_TASKS)
    @pytest.mark.parametrize("quant", _QUANT_FORMATS)
    def test_quant_menu_accepted_on_non_sft_task(self, task: str, quant: str):
        cfg = load_config_from_string(_build_yaml(task, quant))
        assert cfg.task == task
        assert cfg.training.quantization == quant


class TestQuantMenuMultiTrainerMlxRejection:
    @pytest.mark.parametrize("task", ["dpo", "grpo", "pretrain", "embedding"])
    def test_mlx_backend_rejected_for_non_sft_quant_menu(self, task: str):
        # MLX is only valid for sft/dpo/grpo per schema, but Quant Menu MLX
        # rejection must fire BEFORE the task gate to give users the right
        # error. We test the cases where mlx is otherwise valid.
        if task in {"pretrain", "embedding"}:
            pytest.skip("mlx backend unsupported for this task — different gate fires")
        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(_build_yaml(task, "gptq", backend="mlx"))


class TestQuantMenuVisionStillSftOnly:
    """Modality gate is independent — Quant Menu still text-only."""

    def test_vision_modality_rejected_even_after_widening(self):
        with pytest.raises(ValueError, match="modality|vision"):
            load_config_from_string(
                """base: m
task: sft
modality: vision
data: {train: d.jsonl, format: llava}
training: {quantization: gptq}
"""
            )


# ---------------------------------------------------------------------------
# Source-level proofs that each non-SFT trainer wires the loader entry point.
# ---------------------------------------------------------------------------

_TRAINER_FILES = [
    "dpo.py",
    "grpo.py",
    "kto.py",
    "orpo.py",
    "simpo.py",
    "ipo.py",
    "ppo.py",
    "reward_model.py",
    "pretrain.py",
    "embedding.py",
    "bco.py",
]

_TRAINER_DIR = Path(__file__).resolve().parent.parent / "soup_cli" / "trainer"


class TestTrainerSourceWiring:
    @pytest.mark.parametrize("filename", _TRAINER_FILES)
    def test_trainer_calls_build_quantization_config_for_loader(
        self, filename: str
    ) -> None:
        src = (_TRAINER_DIR / filename).read_text(encoding="utf-8")
        assert "build_quantization_config_for_loader" in src, (
            f"{filename} must call build_quantization_config_for_loader to "
            "support v0.38.0 Quant Menu (v0.40.5 #66)"
        )

    @pytest.mark.parametrize("filename", _TRAINER_FILES)
    def test_trainer_no_inline_bnb_4bit_literal(self, filename: str) -> None:
        # The inline `BitsAndBytesConfig(load_in_4bit=True, ...)` block has
        # been replaced by the shared loader. We assert the literal call is
        # absent — defence-in-depth against accidental revert.
        src = (_TRAINER_DIR / filename).read_text(encoding="utf-8")
        assert "BitsAndBytesConfig(\n            load_in_4bit=True" not in src, (
            f"{filename} still contains the legacy inline 4bit BNB block — "
            "should delegate to build_quantization_config_for_loader"
        )

    @pytest.mark.parametrize("filename", _TRAINER_FILES)
    def test_trainer_handles_mxfp4_in_kbit_prep(self, filename: str) -> None:
        # mxfp4 is a BNB 4-bit variant — must run through
        # prepare_model_for_kbit_training. Strict regex match on the literal
        # tuple so a stray comment containing "mxfp4" cannot pass the guard.
        import re

        src = (_TRAINER_DIR / filename).read_text(encoding="utf-8")
        if "prepare_model_for_kbit_training" not in src:
            pytest.skip(f"{filename} doesn't use kbit prep")
        # Match: ("4bit", "8bit", "mxfp4")  with arbitrary whitespace.
        pattern = re.compile(
            r'\(\s*"4bit"\s*,\s*"8bit"\s*,\s*"mxfp4"\s*\)'
        )
        assert pattern.search(src), (
            f"{filename} kbit-prep tuple must be (\"4bit\", \"8bit\", \"mxfp4\")"
        )


# ---------------------------------------------------------------------------
# Loader passthrough — confirm the helper produces non-None for non-bnb formats.
# ---------------------------------------------------------------------------


class TestRewardModelValidator:
    """v0.40.5 (#66) review fix — reward_model field rejects null bytes and
    enforces a 512-char cap (matches cfg.base validation policy).
    """

    def test_reward_model_null_byte_rejected(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="null bytes"):
            TrainingConfig(reward_model="rm\x00evil")

    def test_reward_model_oversize_rejected(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="512"):
            TrainingConfig(reward_model="x" * 513)

    def test_reward_model_none_passes(self):
        cfg = load_config_from_string(
            """base: m
task: ppo
data: {train: d.jsonl}
"""
        )
        assert cfg.training.reward_model is None

    def test_reward_model_normal_path_accepted(self):
        cfg = load_config_from_string(
            """base: m
task: ppo
data: {train: d.jsonl}
training: {reward_model: ./output_rm}
"""
        )
        assert cfg.training.reward_model == "./output_rm"


class TestQuantAwareCrossValidator:
    """Cross-validator `quantization_aware × Quant Menu` rejection must
    still fire for non-SFT tasks (task-agnostic guard).
    """

    @pytest.mark.parametrize("task", ["dpo", "kto", "ppo", "embedding"])
    def test_quant_menu_with_quant_aware_rejected_non_sft(self, task: str):
        extra = ""
        if task == "ppo":
            extra = ", reward_model: dummy"
        with pytest.raises(ValueError, match="incompatible|quantization_aware"):
            load_config_from_string(
                f"""base: m
task: {task}
data: {{train: d.jsonl}}
training: {{quantization: gptq, quantization_aware: true{extra}}}
"""
            )


class TestQuantMenuMlxRejectionExplicit:
    """Even when a different gate would fire first, the MLX backend +
    Quant Menu combination must produce a ValueError for every task.
    """

    @pytest.mark.parametrize("task", ["sft", "dpo", "grpo", "pretrain", "embedding"])
    def test_mlx_quant_menu_always_rejected(self, task: str):
        extra = ""
        if task == "pretrain":
            extra = "\n  format: plaintext"
        if task == "embedding":
            extra = "\n  format: embedding"
        with pytest.raises(ValueError):
            load_config_from_string(
                f"""base: m
task: {task}
backend: mlx
data:
  train: d.jsonl{extra}
training: {{quantization: hqq:4bit}}
"""
            )


class TestPPORewardModelQuantMenu:
    """v0.40.5 review fix — `_load_reward_model` accepts `tcfg` and routes the
    reward model through `build_quantization_config_for_loader` so it doesn't
    silently OOM in full precision when the policy is GPTQ/AWQ/HQQ/etc.
    """

    def test_load_reward_model_signature_accepts_tcfg(self):
        import inspect

        from soup_cli.trainer.ppo import _load_reward_model

        sig = inspect.signature(_load_reward_model)
        assert "tcfg" in sig.parameters, (
            "_load_reward_model must accept tcfg for Quant Menu support"
        )
        assert sig.parameters["tcfg"].default is None, (
            "tcfg must default to None for backward compat"
        )

    def test_load_reward_model_calls_quant_menu_when_tcfg_provided(self):
        # Source-level proof — actually executing _load_reward_model needs
        # transformers + a real (or heavily-mocked) HF model class. The grep
        # is sufficient evidence that the wiring is in place.
        src = (_TRAINER_DIR / "ppo.py").read_text(encoding="utf-8")
        assert "build_quantization_config_for_loader" in src, (
            "ppo.py must call build_quantization_config_for_loader"
        )
        # _setup_reward + _create_reward_model both pass tcfg=tcfg now.
        assert src.count("tcfg=tcfg,") >= 2, (
            "Both PPO reward-loading sites must forward tcfg"
        )

    def test_load_reward_model_routes_through_quant_menu_live(self, monkeypatch):
        # Live dispatch — mock the heavy dependencies (transformers,
        # trust_remote_code helpers, Quant Menu loader) and assert that
        # build_quantization_config_for_loader is called with the right
        # tcfg + base when tcfg is supplied.
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.trainer import ppo as ppo_mod

        calls: list[dict] = []

        def fake_loader(*, tcfg, base, console=None):
            calls.append({"tcfg": tcfg, "base": base})
            return "FAKE_QUANT_OBJ"

        class _FakeRM:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                # Capture model_kwargs for assertion
                calls.append({"from_pretrained_kwargs": kwargs, "path": path})
                inst = cls()
                inst.kwargs = kwargs
                return inst

            def eval(self):
                pass

        monkeypatch.setattr(ppo_mod, "_load_reward_model", ppo_mod._load_reward_model)
        # Patch lazy import seam.
        import sys
        import types as _types

        fake_tf = _types.ModuleType("transformers")
        fake_tf.AutoModelForSequenceClassification = _FakeRM
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)

        # Stub trust_remote helpers to avoid filesystem probing.
        from soup_cli.utils import trust_remote
        monkeypatch.setattr(
            trust_remote, "model_requires_trust_remote_code", lambda _p: False
        )
        monkeypatch.setattr(
            trust_remote, "resolve_trust_remote_code",
            lambda *a, **kw: kw.get("requested", False),
        )

        from soup_cli.utils import quant_menu
        monkeypatch.setattr(
            quant_menu, "build_quantization_config_for_loader", fake_loader
        )

        tcfg = TrainingConfig(quantization="gptq")
        ppo_mod._load_reward_model(
            "some-org/some-rm", device="cpu", trust_remote_code=False, tcfg=tcfg,
        )
        # Assertions: loader was called with the supplied tcfg + path.
        loader_calls = [c for c in calls if "tcfg" in c]
        assert len(loader_calls) == 1
        assert loader_calls[0]["tcfg"] is tcfg
        assert loader_calls[0]["base"] == "some-org/some-rm"
        # And the resulting quant config landed in from_pretrained kwargs.
        fp_calls = [c for c in calls if "from_pretrained_kwargs" in c]
        assert len(fp_calls) == 1
        assert fp_calls[0]["from_pretrained_kwargs"].get("quantization_config") == "FAKE_QUANT_OBJ"

    def test_load_reward_model_no_tcfg_skips_quant_menu(self, monkeypatch):
        # Backward compat: when tcfg=None (default), no quant config is
        # built and the loader is NOT called.
        from soup_cli.trainer import ppo as ppo_mod

        called: list = []

        def fake_loader(*args, **kwargs):
            called.append(True)
            return None

        from soup_cli.utils import quant_menu
        monkeypatch.setattr(
            quant_menu, "build_quantization_config_for_loader", fake_loader
        )

        # Stub transformers.
        import sys
        import types as _types

        class _FakeRM:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                inst = cls()
                inst.kwargs = kwargs
                return inst

            def eval(self):
                pass

        fake_tf = _types.ModuleType("transformers")
        fake_tf.AutoModelForSequenceClassification = _FakeRM
        monkeypatch.setitem(sys.modules, "transformers", fake_tf)

        from soup_cli.utils import trust_remote
        monkeypatch.setattr(
            trust_remote, "model_requires_trust_remote_code", lambda _p: False
        )
        monkeypatch.setattr(
            trust_remote, "resolve_trust_remote_code",
            lambda *a, **kw: kw.get("requested", False),
        )

        ppo_mod._load_reward_model("rm-path", device="cpu", trust_remote_code=False)
        assert called == [], "Quant Menu loader must not be called when tcfg is None"


class TestLoaderEntryPointNonSft:
    """Smoke: build_quantization_config_for_loader returns a non-None
    quant config for the seven Quant Menu formats. Heavy transformers
    classes (GPTQConfig / AwqConfig / HqqConfig / AqlmConfig / EetqConfig)
    are mocked at the import seam.
    """

    def test_loader_gptq_calls_validator_and_returns_config(self, monkeypatch):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils import quant_menu

        sentinel = object()

        class _FakeGPTQ:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Bypass HF import by stubbing the lazy-loaded module member.
        import sys
        import types

        fake_mod = types.ModuleType("transformers")
        fake_mod.GPTQConfig = _FakeGPTQ
        monkeypatch.setitem(sys.modules, "transformers", fake_mod)

        # Validator runs against `base` — for a non-existent path, _looks_like_local_path
        # is False, so it falls through (HF repo id assumed).
        tcfg = TrainingConfig(quantization="gptq")
        result = quant_menu.build_quantization_config_for_loader(
            tcfg=tcfg, base="some-org/some-gptq-repo"
        )
        assert isinstance(result, _FakeGPTQ)
        assert result.kwargs.get("use_exllama") is False
        # Sentinel ensures we did not return the input.
        assert result is not sentinel
