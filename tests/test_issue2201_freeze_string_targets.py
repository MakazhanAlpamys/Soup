"""#2201 — a string ``lora.target_modules`` respects ``freeze_layers`` / ``freeze_ratio``.

peft applies ``layers_to_transform`` only to a LIST ``target_modules``, so the
combination was refused (#1432). It is now resolved instead: the string is
expanded to the module names peft would pick (``all-linear``, or a regex with
``re.fullmatch``), the names outside the trainable layers are dropped, and the
explicit list is passed with ``layers_to_transform`` unset.

Every test builds the real ``SFTTrainerWrapper`` over a tiny CPU Llama and reads
which modules actually carry a ``lora_A`` -- what the adapter did, not what the
config said.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

import yaml  # noqa: E402

from soup_cli.config.loader import load_config_from_string  # noqa: E402
from soup_cli.trainer.sft import SFTTrainerWrapper  # noqa: E402
from tests._tiny_hf_models import tiny_model_dir  # noqa: E402

# The tiny Llama has two decoder layers, so freeze_layers=1 (and
# freeze_ratio=0.5) leave exactly layer 1 trainable.
_TINY_LAYERS = 2


def _wrapper(tmp_path, monkeypatch, *, targets, freeze=None, ratio=None, extra_lora=None):
    monkeypatch.chdir(tmp_path)
    lora = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": targets}
    lora.update(extra_lora or {})
    training = {
        "epochs": 1,
        "batch_size": 1,
        "quantization": "none",
        "lora": lora,
    }
    if freeze is not None:
        training["freeze_layers"] = freeze
    if ratio is not None:
        training["freeze_ratio"] = ratio
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": tiny_model_dir("llama"),
                "task": "sft",
                "backend": "transformers",
                "data": {"train": "unused.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper._setup_transformers(cfg, cfg.training)
    return wrapper


def _adapted(wrapper) -> list[str]:
    """Module names that ended up with a LoRA adapter."""
    return sorted(n for n, m in wrapper.model.named_modules() if hasattr(m, "lora_A"))


class TestAllLinearUnderAFreezePlan:
    def test_the_adapter_lands_only_in_the_trainable_layers(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, targets="all-linear", freeze=1)
        adapted = _adapted(wrapper)
        assert adapted, "no adapter was built at all"
        assert not [n for n in adapted if ".layers.0." in n], adapted
        assert [n for n in adapted if ".layers.1." in n], adapted

    def test_the_output_head_is_not_adapted(self, tmp_path, monkeypatch):
        """``all-linear`` means every linear except the output embedding, and the
        resolved list must keep that meaning."""
        wrapper = _wrapper(tmp_path, monkeypatch, targets="all-linear", freeze=1)
        adapted = _adapted(wrapper)
        assert not [n for n in adapted if n.endswith("lm_head")], adapted

    def test_freeze_ratio_reaches_the_same_plan(self, tmp_path, monkeypatch):
        by_layers = _adapted(_wrapper(tmp_path, monkeypatch, targets="all-linear", freeze=1))
        by_ratio = _adapted(_wrapper(tmp_path, monkeypatch, targets="all-linear", ratio=0.5))
        assert by_ratio == by_layers


class TestRegexUnderAFreezePlan:
    _REGEX = r".*self_attn\.(q_proj|v_proj)"

    def test_only_the_trainable_layers_are_adapted(self, tmp_path, monkeypatch):
        wrapper = _wrapper(tmp_path, monkeypatch, targets=self._REGEX, freeze=1)
        adapted = _adapted(wrapper)
        assert adapted, "no adapter was built at all"
        assert not [n for n in adapted if ".layers.0." in n], adapted
        assert all(n.endswith(("q_proj", "v_proj")) for n in adapted), adapted
        assert len(adapted) == 2, adapted  # q_proj + v_proj, layer 1 only


class TestATargetThatReachesOnlyFrozenLayers:
    def test_it_is_refused_with_the_plan_in_the_message(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError) as err:
            _wrapper(
                tmp_path,
                monkeypatch,
                targets=r"model\.layers\.0\..*q_proj",
                freeze=1,
            )
        message = str(err.value)
        assert "leaves nothing for training.lora.target_modules" in message
        assert f"layers 1..{_TINY_LAYERS - 1}" in message


class TestTheRefusalsThatRemain:
    """The two shapes the freeze plan still cannot scope keep their refusals."""

    def test_target_parameters_is_still_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError) as err:
            _wrapper(
                tmp_path,
                monkeypatch,
                targets=["q_proj"],
                freeze=1,
                extra_lora={"target_parameters": ["model.layers.1.mlp.down_proj"]},
            )
        assert "target_parameters" in str(err.value)

    def test_vera_is_still_refused(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError) as err:
            _wrapper(tmp_path, monkeypatch, targets=["q_proj"], freeze=1,
                     extra_lora={"use_vera": True})
        assert "use_vera" in str(err.value)
