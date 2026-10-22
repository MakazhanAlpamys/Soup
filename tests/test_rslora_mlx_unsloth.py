"""#1853 — ``lora.use_rslora`` is applied on ``backend: mlx`` and ``backend: unsloth``.

The flag was read in exactly one place, ``utils/peft_wiring.py`` (the PEFT
``LoraConfig`` of the transformers backend). The MLX trainer computed the LoRA
scale as ``alpha / r`` in two places (the ``adapter_config.json`` writer and
``_apply_lora``), and the Unsloth loader called ``get_peft_model`` without the
keyword, so with the default ``r: 64, alpha: 16`` an rsLoRA run on either
backend trained at scale 0.25 instead of ``alpha / sqrt(r) = 2.0`` and nothing
said so.

Now one helper gives the MLX scale to both call sites, and the Unsloth loader
forwards ``use_rslora`` from every trainer that uses it. MLX and Unsloth are not
needed: both are replaced by recording stand-ins.
"""

from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.mlx_sft import (
    MLXSFTTrainerWrapper,
    build_mlx_adapter_config,
    mlx_lora_scale,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_DIR = REPO_ROOT / "src" / "soup_cli" / "trainer"


def _lora(r: int = 64, alpha: float = 16.0, use_rslora: bool = False):
    class _L:
        dropout = 0.0
        target_modules = ["q_proj", "v_proj"]

    _L.r = r
    _L.alpha = alpha
    _L.use_rslora = use_rslora
    return _L()


# ---------------------------------------------------------------------------
# MLX: the scale
# ---------------------------------------------------------------------------


class TestMlxScale:
    def test_rslora_scales_by_the_square_root_of_the_rank(self):
        assert mlx_lora_scale(_lora(64, 16.0, use_rslora=True)) == pytest.approx(2.0)
        assert mlx_lora_scale(_lora(16, 32.0, use_rslora=True)) == pytest.approx(8.0)

    def test_plain_lora_keeps_alpha_over_rank(self):
        assert mlx_lora_scale(_lora(64, 16.0, use_rslora=False)) == pytest.approx(0.25)
        assert mlx_lora_scale(_lora(16, 32.0, use_rslora=False)) == pytest.approx(2.0)

    def test_the_two_formulas_agree_only_at_rank_one(self):
        # alpha / sqrt(r) == alpha / r iff r == 1, so a fix that confused the two
        # would fail every other case above.
        assert mlx_lora_scale(_lora(1, 8.0, True)) == mlx_lora_scale(_lora(1, 8.0, False))

    def test_a_config_without_the_flag_is_plain_lora(self):
        class _Old:  # a lora block from before the flag existed
            r = 64
            alpha = 16.0

        assert mlx_lora_scale(_Old()) == pytest.approx(0.25)

    def test_the_matching_peft_scaling(self):
        # The value PEFT computes for the transformers backend (lora_alpha / sqrt(r)),
        # so a run that moves between backends keeps its update size.
        assert mlx_lora_scale(_lora(64, 16.0, True)) == pytest.approx(16.0 / math.sqrt(64))


class TestMlxAdapterConfig:
    def test_rslora_reaches_the_adapter_config(self):
        params = build_mlx_adapter_config(_lora(64, 16.0, True), adapter_path="out")
        assert params["lora_parameters"]["scale"] == pytest.approx(2.0)
        assert params["lora_parameters"]["rank"] == 64

    def test_plain_lora_is_unchanged(self):
        params = build_mlx_adapter_config(_lora(64, 16.0, False), adapter_path="out")
        assert params["lora_parameters"]["scale"] == pytest.approx(0.25)


class TestMlxApplyLora:
    """``_apply_lora`` hands ``linear_to_lora_layers`` the same scale as the writer."""

    @staticmethod
    def _wrapper_and_recorder(monkeypatch, use_rslora: bool):
        seen: list[dict] = []

        def linear_to_lora_layers(model, num_layers, params):
            seen.append({"num_layers": num_layers, **params})

        utils = types.ModuleType("mlx_lm.tuner.utils")
        utils.linear_to_lora_layers = linear_to_lora_layers
        tuner = types.ModuleType("mlx_lm.tuner")
        tuner.utils = utils
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.tuner = tuner
        monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
        monkeypatch.setitem(sys.modules, "mlx_lm.tuner", tuner)
        monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", utils)

        flag = str(use_rslora).lower()
        cfg = load_config_from_string(
            "base: mlx-community/SmolLM2-135M-Instruct\ntask: sft\nbackend: mlx\n"
            "data:\n  train: d.jsonl\n"
            f"training:\n  lora:\n    r: 64\n    alpha: 16\n    use_rslora: {flag}\n"
        )
        return MLXSFTTrainerWrapper(cfg), seen

    class _Model:
        layers = [object(), object()]
        frozen = False

        def freeze(self):
            self.frozen = True

    def test_rslora_scale_reaches_linear_to_lora_layers(self, monkeypatch):
        wrapper, seen = self._wrapper_and_recorder(monkeypatch, use_rslora=True)
        model = self._Model()
        wrapper._apply_lora(model)
        assert model.frozen is True
        assert len(seen) == 1
        assert seen[0]["scale"] == pytest.approx(2.0)
        assert seen[0]["rank"] == 64
        assert seen[0]["num_layers"] == 2

    def test_plain_lora_scale_is_unchanged(self, monkeypatch):
        wrapper, seen = self._wrapper_and_recorder(monkeypatch, use_rslora=False)
        wrapper._apply_lora(self._Model())
        assert seen[0]["scale"] == pytest.approx(0.25)

    def test_writer_and_trainer_agree(self, monkeypatch):
        # The disagreement between the file and the trained modules was the whole
        # of #392 for the keys; the scale now goes through the same helper.
        wrapper, seen = self._wrapper_and_recorder(monkeypatch, use_rslora=True)
        wrapper._apply_lora(self._Model())
        written = build_mlx_adapter_config(wrapper.config.training.lora, adapter_path="out")
        assert written["lora_parameters"]["scale"] == seen[0]["scale"]


# ---------------------------------------------------------------------------
# Unsloth: the loader
# ---------------------------------------------------------------------------


def _fake_unsloth():
    seen: list[dict] = []
    fast = types.SimpleNamespace(
        from_pretrained=lambda **kw: (object(), object()),
        get_peft_model=lambda model, **kw: seen.append(kw) or model,
    )
    return types.SimpleNamespace(FastLanguageModel=fast), seen


class TestUnslothLoader:
    def _load(self, monkeypatch, **kwargs):
        module, seen = _fake_unsloth()
        monkeypatch.setitem(sys.modules, "unsloth", module)
        from soup_cli.utils.unsloth import load_model_and_tokenizer

        load_model_and_tokenizer("m", 512, lora_r=64, lora_alpha=16, **kwargs)
        assert len(seen) == 1
        return seen[0]

    def test_use_rslora_true_reaches_get_peft_model(self, monkeypatch):
        assert self._load(monkeypatch, use_rslora=True)["use_rslora"] is True

    def test_use_rslora_false_reaches_get_peft_model(self, monkeypatch):
        assert self._load(monkeypatch, use_rslora=False)["use_rslora"] is False

    def test_the_default_is_plain_lora(self, monkeypatch):
        # Callers that predate the parameter (the issue's reproduction) keep
        # their behaviour.
        assert self._load(monkeypatch)["use_rslora"] is False

    def test_the_other_lora_kwargs_still_arrive(self, monkeypatch):
        got = self._load(monkeypatch, use_rslora=True, lora_dropout=0.1)
        assert got["r"] == 64 and got["lora_alpha"] == 16 and got["lora_dropout"] == 0.1


# ---------------------------------------------------------------------------
# Unsloth: every trainer forwards the flag from the config
# ---------------------------------------------------------------------------

# The minimum each task needs to load (the same table as the task-gate tests).
_DATA = {"dpo": {"format": "dpo"}, "kto": {"format": "kto"}, "pretrain": {"format": "plaintext"}}

# (task, module, wrapper class): every trainer whose ``_setup_unsloth`` calls the
# shared loader. Listed literally so a trainer that stops forwarding the flag
# fails here rather than disappearing from a computed list.
WRAPPERS = [
    ("sft", "soup_cli.trainer.sft", "SFTTrainerWrapper"),
    ("dpo", "soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    ("grpo", "soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
    ("ppo", "soup_cli.trainer.ppo", "PPOTrainerWrapper"),
    ("kto", "soup_cli.trainer.kto", "KTOTrainerWrapper"),
    ("orpo", "soup_cli.trainer.orpo", "ORPOTrainerWrapper"),
    ("simpo", "soup_cli.trainer.simpo", "SimPOTrainerWrapper"),
    ("ipo", "soup_cli.trainer.ipo", "IPOTrainerWrapper"),
    ("bco", "soup_cli.trainer.bco", "BCOTrainerWrapper"),
    ("pretrain", "soup_cli.trainer.pretrain", "PretrainTrainerWrapper"),
    ("embedding", "soup_cli.trainer.embedding", "EmbeddingTrainerWrapper"),
]


def _unsloth_config(task: str, use_rslora: bool):
    raw = {
        "base": "org/m",
        "task": task,
        "backend": "unsloth",
        "data": {"train": "./x.jsonl", "max_length": 2048, **_DATA.get(task, {})},
        "training": {"lora": {"r": 64, "alpha": 16, "use_rslora": use_rslora}},
    }
    return load_config_from_string(yaml.safe_dump(raw))


@pytest.mark.parametrize("task,module,cls", WRAPPERS, ids=[w[0] for w in WRAPPERS])
@pytest.mark.parametrize("use_rslora", [True, False], ids=["rslora", "plain"])
def test_every_unsloth_trainer_forwards_the_flag_from_the_config(task, module, cls, use_rslora):
    import importlib

    wrapper_cls = getattr(importlib.import_module(module), cls)
    cfg = _unsloth_config(task, use_rslora)
    wrapper = wrapper_cls(cfg, device="cuda")

    mock_model = MagicMock()
    mock_model.get_nb_trainable_parameters.return_value = (1000, 100000)
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "pad"
    with patch(
        "soup_cli.utils.unsloth.load_model_and_tokenizer",
        return_value=(mock_model, mock_tokenizer),
    ) as mock_load:
        wrapper._setup_unsloth(cfg, cfg.training)
    mock_load.assert_called_once()
    kwargs = mock_load.call_args.kwargs
    assert kwargs["use_rslora"] is use_rslora
    assert kwargs["lora_r"] == 64 and kwargs["lora_alpha"] == 16


def test_every_loader_call_under_trainer_passes_the_flag():
    """Source guard: a trainer added later cannot call the loader without it."""
    pattern = re.compile(r"load_model_and_tokenizer\((.*?)\)", re.DOTALL)
    offenders = []
    checked = 0
    for path in sorted(TRAINER_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "from soup_cli.utils.unsloth import load_model_and_tokenizer" not in text:
            continue
        for call in pattern.finditer(text):
            if "model_name=" not in call.group(1):
                continue  # the import line or a reference, not a call with kwargs
            checked += 1
            if "use_rslora=" not in call.group(1):
                offenders.append(path.name)
    assert checked >= len(WRAPPERS), checked
    assert offenders == [], offenders
