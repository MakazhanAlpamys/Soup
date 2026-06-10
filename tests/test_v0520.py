"""v0.52.0 Modality II — TTS + Distillation + BitNet + EBFT + GDPO + MoE + reasoning_effort.

Schema-only test suite. Live wiring deferred to v0.52.1. Mirrors v0.50.0 /
v0.51.0 single-file test layout (test_v0500_part_X.py was per-Part; v0.51.0
collapsed into one test_v0510.py; v0.52.0 follows v0.51.0).
"""

from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from soup_cli.config.loader import load_config_from_string

# ---------------------------------------------------------------------------
# Part A — TTS
# ---------------------------------------------------------------------------


class TestTTSUtils:
    def test_supported_families_frozenset(self):
        from soup_cli.utils.tts import SUPPORTED_TTS_FAMILIES

        assert isinstance(SUPPORTED_TTS_FAMILIES, frozenset)
        assert SUPPORTED_TTS_FAMILIES == {
            "orpheus", "sesame_csm", "llasa", "spark", "oute",
        }

    @pytest.mark.parametrize(
        "name", ["orpheus", "ORPHEUS", "Sesame_CSM", "llasa", "spark", "oute"],
    )
    def test_validate_family_canonical(self, name):
        from soup_cli.utils.tts import validate_tts_family

        assert validate_tts_family(name) == name.lower()

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (123, TypeError),
            ("", ValueError),
            ("orph\x00eus", ValueError),
            ("x" * 100, ValueError),
            ("unknown_family", ValueError),
        ],
    )
    def test_validate_family_rejects(self, bad, exc):
        from soup_cli.utils.tts import validate_tts_family

        with pytest.raises(exc):
            validate_tts_family(bad)

    def test_family_metadata_frozen(self):
        from soup_cli.utils.tts import get_tts_family_spec

        spec = get_tts_family_spec("orpheus")
        with pytest.raises(Exception):
            spec.name = "evil"  # type: ignore[misc]

    def test_supports_emotion(self):
        from soup_cli.utils.tts import family_supports_emotion

        assert family_supports_emotion("orpheus") is True
        assert family_supports_emotion("oute") is True
        assert family_supports_emotion("llasa") is False

    def test_emotion_tag_orpheus_happy(self):
        from soup_cli.utils.tts import validate_emotion_tag

        assert validate_emotion_tag("happy", family="orpheus") == "happy"
        assert validate_emotion_tag("HAPPY", family="orpheus") == "happy"

    def test_emotion_tag_orpheus_unknown(self):
        from soup_cli.utils.tts import validate_emotion_tag

        with pytest.raises(ValueError, match="orpheus allowlist"):
            validate_emotion_tag("euphoric", family="orpheus")

    def test_emotion_tag_unsupported_family(self):
        from soup_cli.utils.tts import validate_emotion_tag

        with pytest.raises(ValueError, match="does not support emotion"):
            validate_emotion_tag("happy", family="llasa")

    def test_emotion_tag_bool_rejected(self):
        from soup_cli.utils.tts import validate_emotion_tag

        with pytest.raises(TypeError):
            validate_emotion_tag(True, family="orpheus")

    def test_validate_tts_compat_happy(self):
        from soup_cli.utils.tts import validate_tts_compat

        validate_tts_compat(task="tts", modality="audio_out", backend="transformers")

    @pytest.mark.parametrize(
        "kwargs,match", [
            ({"task": "sft", "modality": "audio_out", "backend": "transformers"}, "tts"),
            ({"task": "tts", "modality": "text", "backend": "transformers"}, "audio_out"),
            ({"task": "tts", "modality": "audio_out", "backend": "mlx"}, "mlx"),
        ],
    )
    def test_validate_tts_compat_rejects(self, kwargs, match):
        from soup_cli.utils.tts import validate_tts_compat

        with pytest.raises(ValueError, match=match):
            validate_tts_compat(**kwargs)

    def test_build_tts_trainer_lifted_v07120(self):
        """v0.52.0 shipped as a NotImplementedError stub; v0.71.20 #131 lifts it
        into a real TTSTrainerWrapper factory taking ``config``. No-arg call now
        raises TypeError rather than NotImplementedError."""
        from soup_cli.utils.tts import build_tts_trainer

        with pytest.raises(TypeError):
            build_tts_trainer()


class TestTTSSchemaIntegration:
    def test_task_tts_happy(self):
        yaml = """
base: canopylabs/orpheus-tts
task: tts
modality: audio_out
data: {train: ./d.jsonl}
training:
  tts_family: orpheus
  tts_emotion: happy
"""
        cfg = load_config_from_string(yaml)
        assert cfg.task == "tts"
        assert cfg.modality == "audio_out"
        assert cfg.training.tts_family == "orpheus"
        assert cfg.training.tts_emotion == "happy"

    def test_task_tts_without_family_rejected(self):
        yaml = """
base: foo
task: tts
modality: audio_out
data: {train: ./d.jsonl}
"""
        with pytest.raises(Exception, match="tts_family"):
            load_config_from_string(yaml)

    def test_tts_family_without_task_rejected(self):
        yaml = """
base: foo
task: sft
data: {train: ./d.jsonl}
training:
  tts_family: orpheus
"""
        with pytest.raises(Exception, match="tts"):
            load_config_from_string(yaml)

    def test_tts_emotion_unsupported_family_rejected(self):
        yaml = """
base: foo
task: tts
modality: audio_out
data: {train: ./d.jsonl}
training:
  tts_family: llasa
  tts_emotion: happy
"""
        with pytest.raises(Exception, match="does not support emotion"):
            load_config_from_string(yaml)

    def test_audio_out_modality_accepted(self):
        # audio_out paired with non-TTS task still loads (modality alone
        # doesn't force task='tts'); but in practice only TTS uses it.
        # Defence-in-depth: test the Literal accepts it.
        yaml = """
base: foo
task: sft
modality: audio_out
data: {train: ./d.jsonl, format: audio}
"""
        cfg = load_config_from_string(yaml)
        assert cfg.modality == "audio_out"


# ---------------------------------------------------------------------------
# Part B — classifier / reranker / cross_encoder
# ---------------------------------------------------------------------------


class TestClassifierUtils:
    def test_classifier_tasks_frozenset(self):
        from soup_cli.utils.classifier import CLASSIFIER_TASKS

        assert CLASSIFIER_TASKS == {"classifier", "reranker", "cross_encoder"}

    @pytest.mark.parametrize("task", ["classifier", "reranker", "cross_encoder"])
    def test_is_classifier_task_true(self, task):
        from soup_cli.utils.classifier import is_classifier_task

        assert is_classifier_task(task) is True

    @pytest.mark.parametrize("task", ["sft", "dpo", "", True, 123, None])
    def test_is_classifier_task_false(self, task):
        from soup_cli.utils.classifier import is_classifier_task

        assert is_classifier_task(task) is False

    def test_get_classifier_spec_paired_input(self):
        from soup_cli.utils.classifier import get_classifier_spec

        assert get_classifier_spec("cross_encoder").paired_input is True
        assert get_classifier_spec("classifier").paired_input is False

    def test_get_classifier_spec_unknown(self):
        from soup_cli.utils.classifier import get_classifier_spec

        with pytest.raises(ValueError, match="classifier task"):
            get_classifier_spec("sft")

    @pytest.mark.parametrize(
        "value,exc", [
            (True, TypeError),
            ("3", TypeError),
            (None, TypeError),
            (0, ValueError),
            (-1, ValueError),
            (2000, ValueError),
        ],
    )
    def test_validate_num_labels_rejects(self, value, exc):
        from soup_cli.utils.classifier import validate_num_labels

        with pytest.raises(exc):
            validate_num_labels(value)

    def test_validate_num_labels_happy(self):
        from soup_cli.utils.classifier import validate_num_labels

        assert validate_num_labels(3) == 3
        assert validate_num_labels(1024) == 1024

    def test_validate_label_names_dedup(self):
        from soup_cli.utils.classifier import validate_label_names

        with pytest.raises(ValueError, match="unique"):
            validate_label_names(["a", "a", "b"])

    @pytest.mark.parametrize(
        "value,exc", [
            ("a", TypeError),
            ([True, "a"], TypeError),
            ([""], ValueError),
            (["x\x00"], ValueError),
            (["x" * 200], ValueError),
        ],
    )
    def test_validate_label_names_rejects(self, value, exc):
        from soup_cli.utils.classifier import validate_label_names

        with pytest.raises(exc):
            validate_label_names(value)

    def test_validate_label_names_defensive_copy(self):
        from soup_cli.utils.classifier import validate_label_names

        src = ["a", "b"]
        out = validate_label_names(src)
        assert out is not src

    def test_validate_classifier_compat_mlx_reject(self):
        from soup_cli.utils.classifier import validate_classifier_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_classifier_compat(
                task="classifier", backend="mlx", modality="text",
            )

    def test_validate_classifier_compat_non_text(self):
        from soup_cli.utils.classifier import validate_classifier_compat

        with pytest.raises(ValueError, match="text"):
            validate_classifier_compat(
                task="reranker", backend="transformers", modality="vision",
            )

    def test_build_classifier_trainer_lifted_in_v0532(self):
        """v0.52.0 shipped as a NotImplementedError stub; v0.53.2 #132 lifts it
        to a live factory returning ClassifierTrainerWrapper. The argless call
        now raises TypeError (missing ``config``) rather than NotImplementedError."""
        from soup_cli.utils.classifier import build_classifier_trainer

        with pytest.raises(TypeError):
            build_classifier_trainer()  # type: ignore[call-arg]


class TestClassifierSchema:
    def test_classifier_happy(self):
        cfg = load_config_from_string(
            "base: foo\ntask: classifier\ndata: {train: ./d.jsonl}\n"
            "training: {num_labels: 5, classifier_kind: single_label}\n"
        )
        assert cfg.task == "classifier"
        assert cfg.training.num_labels == 5

    @pytest.mark.parametrize("task", ["reranker", "cross_encoder"])
    def test_reranker_and_cross_encoder_happy(self, task):
        cfg = load_config_from_string(
            f"base: foo\ntask: {task}\ndata: {{train: ./d.jsonl}}\n"
            "training: {num_labels: 1}\n"
        )
        assert cfg.task == task

    def test_classifier_label_names_mismatch_rejected(self):
        yaml = (
            "base: foo\ntask: classifier\ndata: {train: ./d.jsonl}\n"
            "training:\n  num_labels: 3\n  label_names: [a, b]\n"
        )
        with pytest.raises(Exception, match="num_labels"):
            load_config_from_string(yaml)

    def test_num_labels_outside_classifier_rejected(self):
        yaml = (
            "base: foo\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {num_labels: 3}\n"
        )
        with pytest.raises(Exception, match="classifier"):
            load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# Part C — distillation
# ---------------------------------------------------------------------------


class TestDistillUtils:
    def test_divergence_canonical(self):
        from soup_cli.utils.distill import validate_divergence

        assert validate_divergence("kl") == "forward_kl"
        assert validate_divergence("KL") == "forward_kl"
        assert validate_divergence("forward_kl") == "forward_kl"
        assert validate_divergence("reverse_kl") == "reverse_kl"
        assert validate_divergence("js") == "js"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("kl\x00", ValueError),
            ("k" * 100, ValueError),
            ("unknown", ValueError),
        ],
    )
    def test_validate_divergence_rejects(self, bad, exc):
        from soup_cli.utils.distill import validate_divergence

        with pytest.raises(exc):
            validate_divergence(bad)

    def test_get_divergence_spec_symmetric(self):
        from soup_cli.utils.distill import get_divergence_spec

        assert get_divergence_spec("js").symmetric is True
        assert get_divergence_spec("forward_kl").symmetric is False

    @pytest.mark.parametrize(
        "value,exc", [
            (True, TypeError),
            ("1.0", TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (0.0, ValueError),
            (0.01, ValueError),
            (101.0, ValueError),
        ],
    )
    def test_validate_distill_temperature_rejects(self, value, exc):
        from soup_cli.utils.distill import validate_distill_temperature

        with pytest.raises(exc):
            validate_distill_temperature(value)

    def test_validate_distill_temperature_happy(self):
        from soup_cli.utils.distill import validate_distill_temperature

        assert validate_distill_temperature(2.0) == 2.0
        assert validate_distill_temperature(0.05) == 0.05

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (123, TypeError),
            ("", ValueError),
            ("x\x00", ValueError),
            ("x" * 1000, ValueError),
        ],
    )
    def test_validate_teacher_model_rejects(self, bad, exc):
        from soup_cli.utils.distill import validate_teacher_model

        with pytest.raises(exc):
            validate_teacher_model(bad)

    def test_validate_distill_compat_no_teacher(self):
        from soup_cli.utils.distill import validate_distill_compat

        with pytest.raises(ValueError, match="teacher_model"):
            validate_distill_compat(
                task="distill", backend="transformers", teacher_model=None,
            )

    def test_validate_distill_compat_mlx(self):
        from soup_cli.utils.distill import validate_distill_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_distill_compat(
                task="distill", backend="mlx", teacher_model="t/model",
            )

    def test_build_distill_trainer_lifted_in_v0532(self):
        """v0.52.0 shipped as a NotImplementedError stub; v0.53.2 #133 lifts it
        to a live factory returning DistillTrainerWrapper. The argless call
        now raises TypeError (missing ``config``) rather than NotImplementedError."""
        from soup_cli.utils.distill import build_distill_trainer

        with pytest.raises(TypeError):
            build_distill_trainer()


class TestDistillSchema:
    def test_distill_happy(self):
        yaml = (
            "base: foo\ntask: distill\ndata: {train: ./d.jsonl}\n"
            "training:\n  teacher_model: meta-llama/Llama-3.1-70B\n"
            "  distill_divergence: reverse_kl\n  distill_temperature: 2.5\n"
        )
        cfg = load_config_from_string(yaml)
        assert cfg.task == "distill"
        assert cfg.training.distill_divergence == "reverse_kl"
        assert cfg.training.distill_temperature == 2.5

    def test_kl_alias_canonicalised(self):
        yaml = (
            "base: foo\ntask: distill\ndata: {train: ./d.jsonl}\n"
            "training:\n  teacher_model: t/m\n  distill_divergence: kl\n"
        )
        cfg = load_config_from_string(yaml)
        assert cfg.training.distill_divergence == "forward_kl"

    def test_teacher_outside_distill_rejected(self):
        yaml = (
            "base: foo\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {teacher_model: t/m}\n"
        )
        with pytest.raises(Exception, match="distill"):
            load_config_from_string(yaml)

    def test_distill_temperature_nan_rejected(self):
        yaml = (
            "base: foo\ntask: distill\ndata: {train: ./d.jsonl}\n"
            "training:\n  teacher_model: t/m\n  distill_temperature: .nan\n"
        )
        with pytest.raises(Exception, match="finite"):
            load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# Part D — BitNet 1.58
# ---------------------------------------------------------------------------


class TestBitNetUtils:
    def test_bitnet_quant_formats_frozenset(self):
        from soup_cli.utils.bitnet import BITNET_EXPORT_FORMATS, BITNET_QUANT_FORMATS

        assert isinstance(BITNET_QUANT_FORMATS, frozenset)
        assert isinstance(BITNET_EXPORT_FORMATS, frozenset)
        assert BITNET_QUANT_FORMATS == {"bitnet_1.58"}
        assert BITNET_EXPORT_FORMATS == {"bitnet", "tq1_0"}

    @pytest.mark.parametrize(
        "value,expected", [
            ("bitnet_1.58", True),
            ("4bit", False),
            ("", False),
            (None, False),
            (True, False),
            (123, False),
        ],
    )
    def test_is_bitnet_quant(self, value, expected):
        from soup_cli.utils.bitnet import is_bitnet_quant

        assert is_bitnet_quant(value) is expected

    def test_get_bitnet_spec(self):
        from soup_cli.utils.bitnet import get_bitnet_spec

        spec = get_bitnet_spec("bitnet_1.58")
        assert spec.bits == 1.58
        assert spec.live_wired is False

    def test_get_bitnet_spec_unknown(self):
        from soup_cli.utils.bitnet import get_bitnet_spec

        with pytest.raises(ValueError, match="bitnet"):
            get_bitnet_spec("4bit")

    @pytest.mark.parametrize(
        "name,expected", [
            ("microsoft/bitnet-b1.58-2B", True),
            ("tiiuae/Falcon-E-1B-Instruct", True),
            ("1bitllm/foo", True),
            ("OneBitLLM/falcon-e", True),
            ("meta-llama/Llama-3.1-8B", False),
            ("", False),
            (None, False),
            (True, False),
            ("evil\x00", False),
        ],
    )
    def test_is_bitnet_model(self, name, expected):
        from soup_cli.utils.bitnet import is_bitnet_model

        assert is_bitnet_model(name) is expected

    def test_validate_bitnet_compat_mlx_reject(self):
        from soup_cli.utils.bitnet import validate_bitnet_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_bitnet_compat(task="sft", backend="mlx", modality="text")

    def test_validate_bitnet_compat_vision_reject(self):
        from soup_cli.utils.bitnet import validate_bitnet_compat

        with pytest.raises(ValueError, match="text"):
            validate_bitnet_compat(
                task="sft", backend="transformers", modality="vision",
            )

    def test_validate_bitnet_compat_grpo_reject(self):
        from soup_cli.utils.bitnet import validate_bitnet_compat

        with pytest.raises(ValueError, match="task"):
            validate_bitnet_compat(
                task="grpo", backend="transformers", modality="text",
            )

    def test_validate_bitnet_export_canonical(self):
        from soup_cli.utils.bitnet import validate_bitnet_export

        assert validate_bitnet_export("bitnet") == "bitnet"
        assert validate_bitnet_export("TQ1_0") == "tq1_0"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("foo\x00", ValueError),
            ("Q4_K_M", ValueError),
        ],
    )
    def test_validate_bitnet_export_rejects(self, bad, exc):
        from soup_cli.utils.bitnet import validate_bitnet_export

        with pytest.raises(exc):
            validate_bitnet_export(bad)

    def test_build_bitnet_trainer_lifted_v07120(self):
        """v0.52.0 stub lifted in v0.71.20 #134 to a BitNetTrainerWrapper
        factory taking ``config``. No-arg call now raises TypeError."""
        from soup_cli.utils.bitnet import build_bitnet_trainer

        with pytest.raises(TypeError):
            build_bitnet_trainer()

    def test_export_bitnet_gguf_lifted_v07120(self):
        """v0.52.0 stub lifted in v0.71.20 #134 to a real export taking
        required kwargs. No-arg call now raises TypeError."""
        from soup_cli.utils.bitnet import export_bitnet_gguf

        with pytest.raises(TypeError):
            export_bitnet_gguf()


class TestBitNetSchema:
    def test_bitnet_sft_happy(self):
        cfg = load_config_from_string(
            "base: tiiuae/Falcon-E-1B-Instruct\ntask: sft\n"
            "data: {train: ./d.jsonl}\ntraining: {quantization: bitnet_1.58}\n"
        )
        assert cfg.training.quantization == "bitnet_1.58"

    def test_bitnet_dpo_happy(self):
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata: {train: ./d.jsonl}\n"
            "training: {quantization: bitnet_1.58}\n"
        )
        assert cfg.training.quantization == "bitnet_1.58"

    def test_bitnet_grpo_rejected(self):
        yaml = (
            "base: x\ntask: grpo\ndata: {train: ./d.jsonl}\n"
            "training: {quantization: bitnet_1.58, reward_fn: accuracy, num_generations: 4}\n"
        )
        with pytest.raises(Exception, match="task"):
            load_config_from_string(yaml)

    def test_bitnet_mlx_rejected(self):
        yaml = (
            "base: x\ntask: sft\nbackend: mlx\ndata: {train: ./d.jsonl}\n"
            "training: {quantization: bitnet_1.58}\n"
        )
        with pytest.raises(Exception, match="mlx"):
            load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# Part E — EBFT + GDPO
# ---------------------------------------------------------------------------


class TestEbftGdpoUtils:
    def test_ebft_variants(self):
        from soup_cli.utils.ebft_gdpo import EBFT_VARIANTS, validate_ebft_variant

        assert EBFT_VARIANTS == {"structured", "strided"}
        assert validate_ebft_variant("structured") == "structured"
        assert validate_ebft_variant("STRIDED") == "strided"

    def test_gdpo_variants(self):
        from soup_cli.utils.ebft_gdpo import GDPO_VARIANTS, validate_gdpo_variant

        assert GDPO_VARIANTS == {"standard", "length_normalized", "margin"}
        assert validate_gdpo_variant("Margin") == "margin"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("foo\x00", ValueError),
            ("unknown", ValueError),
        ],
    )
    def test_validate_ebft_variant_rejects(self, bad, exc):
        from soup_cli.utils.ebft_gdpo import validate_ebft_variant

        with pytest.raises(exc):
            validate_ebft_variant(bad)

    def test_ebft_temperature_bounds(self):
        from soup_cli.utils.ebft_gdpo import validate_ebft_temperature

        assert validate_ebft_temperature(1.0) == 1.0
        for bad in (True, float("nan"), float("inf"), 0.0, 1000.0):
            with pytest.raises((TypeError, ValueError)):
                validate_ebft_temperature(bad)

    def test_validate_ebft_compat_dpo_rejected(self):
        from soup_cli.utils.ebft_gdpo import validate_ebft_compat

        with pytest.raises(ValueError, match="sft"):
            validate_ebft_compat(task="dpo", backend="transformers")

    def test_validate_gdpo_compat_sft_rejected(self):
        from soup_cli.utils.ebft_gdpo import validate_gdpo_compat

        with pytest.raises(ValueError, match="dpo"):
            validate_gdpo_compat(task="sft", backend="transformers")

    def test_get_ebft_spec(self):
        # v0.53.2 #135 lifted EBFT + GDPO live_wired flags from False to True
        # (kernel + attach hooks shipped).
        from soup_cli.utils.ebft_gdpo import get_ebft_spec, get_gdpo_spec

        assert get_ebft_spec("structured").live_wired is True
        assert get_gdpo_spec("margin").live_wired is True

    def test_apply_ebft_loss_lifted_in_v0532(self):
        """v0.52.0 shipped both as NotImplementedError stubs; v0.53.2 #135
        lifts them to live tensor kernels. The argless invocation now raises
        TypeError (missing required args) rather than NotImplementedError."""
        from soup_cli.utils.ebft_gdpo import apply_ebft_loss, apply_gdpo_loss

        with pytest.raises(TypeError):
            apply_ebft_loss()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            apply_gdpo_loss()  # type: ignore[call-arg]


class TestEbftGdpoSchema:
    def test_ebft_happy(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {ebft_variant: structured, ebft_temperature: 1.0}\n"
        )
        assert cfg.training.ebft_variant == "structured"
        assert cfg.training.ebft_temperature == 1.0

    def test_ebft_temp_requires_variant(self):
        yaml = (
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {ebft_temperature: 2.0}\n"
        )
        with pytest.raises(Exception, match="ebft_variant"):
            load_config_from_string(yaml)

    def test_ebft_on_dpo_rejected(self):
        yaml = (
            "base: x\ntask: dpo\ndata: {train: ./d.jsonl}\n"
            "training: {ebft_variant: strided}\n"
        )
        with pytest.raises(Exception, match="sft"):
            load_config_from_string(yaml)

    def test_gdpo_dpo_happy(self):
        cfg = load_config_from_string(
            "base: x\ntask: dpo\ndata: {train: ./d.jsonl}\n"
            "training: {gdpo_variant: length_normalized}\n"
        )
        assert cfg.training.gdpo_variant == "length_normalized"

    def test_gdpo_on_sft_rejected(self):
        yaml = (
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {gdpo_variant: standard}\n"
        )
        with pytest.raises(Exception, match="dpo"):
            load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# Part F — MoE expert quant + train_router_only
# ---------------------------------------------------------------------------


class TestMoeQuantUtils:
    def test_moe_expert_quant_formats(self):
        from soup_cli.utils.moe_quant import (
            MOE_EXPERT_QUANT_FORMATS,
            validate_moe_expert_quant,
        )

        assert MOE_EXPERT_QUANT_FORMATS == {"nf4", "int8_rowwise"}
        assert validate_moe_expert_quant("NF4") == "nf4"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("foo\x00", ValueError),
            ("unknown", ValueError),
        ],
    )
    def test_validate_moe_expert_quant_rejects(self, bad, exc):
        from soup_cli.utils.moe_quant import validate_moe_expert_quant

        with pytest.raises(exc):
            validate_moe_expert_quant(bad)

    def test_get_moe_expert_quant_spec_bits(self):
        from soup_cli.utils.moe_quant import get_moe_expert_quant_spec

        assert get_moe_expert_quant_spec("nf4").bits == 4
        assert get_moe_expert_quant_spec("int8_rowwise").bits == 8

    def test_moe_expert_quant_requires_moe_lora(self):
        from soup_cli.utils.moe_quant import validate_moe_expert_quant_compat

        with pytest.raises(ValueError, match="moe_lora"):
            validate_moe_expert_quant_compat(
                backend="transformers", moe_lora=False,
            )

    def test_train_router_only_requires_moe_lora(self):
        from soup_cli.utils.moe_quant import validate_train_router_only_compat

        with pytest.raises(ValueError, match="moe_lora"):
            validate_train_router_only_compat(
                backend="transformers", moe_lora=False,
            )

    def test_validate_moe_expert_quant_compat_mlx(self):
        from soup_cli.utils.moe_quant import validate_moe_expert_quant_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_moe_expert_quant_compat(backend="mlx", moe_lora=True)

    def test_apply_moe_expert_quant_lifted_v07120(self):
        """v0.52.0 stub lifted in v0.71.20 #136 to take (model, quant_format).
        No-arg call now raises TypeError."""
        from soup_cli.utils.moe_quant import apply_moe_expert_quant

        with pytest.raises(TypeError):
            apply_moe_expert_quant()


class TestMoeQuantSchema:
    def test_moe_expert_quant_happy(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {moe_lora: true, moe_expert_quant: nf4}\n"
        )
        assert cfg.training.moe_expert_quant == "nf4"

    def test_moe_expert_quant_without_moe_lora_rejected(self):
        yaml = (
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {moe_expert_quant: nf4}\n"
        )
        with pytest.raises(Exception, match="moe_lora"):
            load_config_from_string(yaml)

    def test_train_router_only_happy(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {moe_lora: true, train_router_only: true}\n"
        )
        assert cfg.training.train_router_only is True

    def test_train_router_only_without_moe_lora_rejected(self):
        yaml = (
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {train_router_only: true}\n"
        )
        with pytest.raises(Exception, match="moe_lora"):
            load_config_from_string(yaml)


# ---------------------------------------------------------------------------
# Part G — reasoning_effort + train_on_eot
# ---------------------------------------------------------------------------


class TestReasoningEffortUtils:
    def test_levels(self):
        from soup_cli.utils.reasoning_effort import (
            REASONING_EFFORT_LEVELS,
            validate_reasoning_effort,
        )

        assert REASONING_EFFORT_LEVELS == {"low", "medium", "high"}
        assert validate_reasoning_effort("LOW") == "low"
        assert validate_reasoning_effort("medium") == "medium"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("x\x00", ValueError),
            ("ultra", ValueError),
        ],
    )
    def test_validate_reasoning_effort_rejects(self, bad, exc):
        from soup_cli.utils.reasoning_effort import validate_reasoning_effort

        with pytest.raises(exc):
            validate_reasoning_effort(bad)


class TestReasoningEffortSchema:
    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_reasoning_effort_happy(self, level):
        cfg = load_config_from_string(
            f"base: openai/gpt-oss-20b\ntask: sft\ndata: {{train: ./d.jsonl}}\n"
            f"training: {{reasoning_effort: {level}}}\n"
        )
        assert cfg.training.reasoning_effort == level

    def test_train_on_eot_default_false(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
        )
        assert cfg.training.train_on_eot is False

    def test_train_on_eot_true(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {train_on_eot: true}\n"
        )
        assert cfg.training.train_on_eot is True


# ---------------------------------------------------------------------------
# Cross-cutting: immutability + module surface
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_tts_metadata_mapping_proxy(self):
        from soup_cli.utils.tts import _TTS_FAMILY_METADATA  # type: ignore

        assert isinstance(_TTS_FAMILY_METADATA, MappingProxyType)

    def test_classifier_metadata_mapping_proxy(self):
        from soup_cli.utils.classifier import _CLASSIFIER_METADATA  # type: ignore

        assert isinstance(_CLASSIFIER_METADATA, MappingProxyType)

    def test_distill_metadata_mapping_proxy(self):
        from soup_cli.utils.distill import (  # type: ignore
            _DIVERGENCE_ALIASES,
            _DIVERGENCE_METADATA,
        )

        assert isinstance(_DIVERGENCE_METADATA, MappingProxyType)
        assert isinstance(_DIVERGENCE_ALIASES, MappingProxyType)

    def test_bitnet_metadata_mapping_proxy(self):
        from soup_cli.utils.bitnet import _BITNET_METADATA  # type: ignore

        assert isinstance(_BITNET_METADATA, MappingProxyType)

    def test_ebft_gdpo_metadata_mapping_proxy(self):
        from soup_cli.utils.ebft_gdpo import (  # type: ignore
            _EBFT_METADATA,
            _GDPO_METADATA,
        )

        assert isinstance(_EBFT_METADATA, MappingProxyType)
        assert isinstance(_GDPO_METADATA, MappingProxyType)

    def test_moe_quant_metadata_mapping_proxy(self):
        from soup_cli.utils.moe_quant import (  # type: ignore
            _MOE_EXPERT_QUANT_METADATA,
        )

        assert isinstance(_MOE_EXPERT_QUANT_METADATA, MappingProxyType)


class TestV0520Recipes:
    NEW_RECIPES = (
        "orpheus-tts-sft",
        "sesame-csm-tts",
        "llasa-tts",
        "spark-tts",
        "oute-tts",
        "falcon-e-bitnet-sft",
    )

    @pytest.mark.parametrize("name", NEW_RECIPES)
    def test_recipe_loads(self, name):
        from soup_cli.recipes.catalog import RECIPES

        recipe = RECIPES[name]
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.base == recipe.model

    @pytest.mark.parametrize(
        "name,expected_family", [
            ("orpheus-tts-sft", "orpheus"),
            ("sesame-csm-tts", "sesame_csm"),
            ("llasa-tts", "llasa"),
            ("spark-tts", "spark"),
            ("oute-tts", "oute"),
        ],
    )
    def test_tts_recipe_family(self, name, expected_family):
        from soup_cli.recipes.catalog import RECIPES

        cfg = load_config_from_string(RECIPES[name].yaml_str)
        assert cfg.training.tts_family == expected_family
        assert cfg.task == "tts"
        assert cfg.modality == "audio_out"

    def test_falcon_e_bitnet_quant(self):
        from soup_cli.recipes.catalog import RECIPES

        cfg = load_config_from_string(RECIPES["falcon-e-bitnet-sft"].yaml_str)
        assert cfg.training.quantization == "bitnet_1.58"

    def test_total_catalog_size_grew(self):
        from soup_cli.recipes.catalog import RECIPES

        # v0.51.0 shipped 106; v0.52.0 adds 6 (5 TTS + Falcon-E BitNet).
        assert len(RECIPES) >= 112


class TestTddReviewGaps:
    """v0.52.0 TDD-review-pass coverage of gaps surfaced after the first cut."""

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("foo\x00", ValueError),
            ("x" * 33, ValueError),
            ("unknown", ValueError),
        ],
    )
    def test_validate_ebft_variant_oversize_etc(self, bad, exc):
        from soup_cli.utils.ebft_gdpo import validate_ebft_variant

        with pytest.raises(exc):
            validate_ebft_variant(bad)

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("foo\x00", ValueError),
            ("x" * 33, ValueError),
            ("unknown", ValueError),
        ],
    )
    def test_validate_gdpo_variant_full_matrix(self, bad, exc):
        from soup_cli.utils.ebft_gdpo import validate_gdpo_variant

        with pytest.raises(exc):
            validate_gdpo_variant(bad)

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            ("1.0", TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (0.0, ValueError),
            (1000.0, ValueError),
        ],
    )
    def test_validate_ebft_temperature_explicit_exc(self, bad, exc):
        from soup_cli.utils.ebft_gdpo import validate_ebft_temperature

        with pytest.raises(exc):
            validate_ebft_temperature(bad)

    @pytest.mark.parametrize(
        "kwargs,exc", [
            ({"task": "", "modality": "audio_out", "backend": "transformers"}, ValueError),
            ({"task": "tts\x00", "modality": "audio_out", "backend": "transformers"}, ValueError),
            ({"task": True, "modality": "audio_out", "backend": "transformers"}, TypeError),
            ({"task": "tts", "modality": "", "backend": "transformers"}, ValueError),
            ({"task": "tts", "modality": "audio_out", "backend": ""}, ValueError),
            ({"task": "tts", "modality": "audio_out", "backend": False}, TypeError),
        ],
    )
    def test_validate_tts_compat_input_guards(self, kwargs, exc):
        from soup_cli.utils.tts import validate_tts_compat

        with pytest.raises(exc):
            validate_tts_compat(**kwargs)

    @pytest.mark.parametrize("task", ["grpo", "pretrain", "ppo", "embedding", "tts"])
    def test_reasoning_effort_task_gate_full_matrix(self, task):
        # ``pretrain`` is in the SFT-family allowlist so it should accept.
        # All other non-SFT-family tasks must reject.
        sft_family = {"sft", "pretrain", "distill", "classifier", "reranker", "cross_encoder"}
        # Need backend / modality / data to be valid; for tts we also need family.
        extra = ""
        modality = ""
        if task == "tts":
            modality = "modality: audio_out\n"
            extra = "  tts_family: orpheus\n"
        if task == "grpo":
            extra = "  reward_fn: accuracy\n  num_generations: 4\n"
        yaml = (
            f"base: x\ntask: {task}\n{modality}data: {{train: ./d.jsonl}}\n"
            f"training:\n  reasoning_effort: low\n{extra}"
        )
        if task in sft_family:
            cfg = load_config_from_string(yaml)
            assert cfg.training.reasoning_effort == "low"
        else:
            with pytest.raises(Exception, match="reasoning_effort"):
                load_config_from_string(yaml)

    def test_train_on_eot_int_one_is_bool(self):
        # YAML "1" parses as int; Pydantic bool field coerces 0/1.
        # This is the documented Pydantic behaviour we accept; the
        # task-gate is what protects against silent no-op.
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {train_on_eot: 1}\n"
        )
        assert cfg.training.train_on_eot is True

    def test_tts_recipe_model_id_no_null_or_whitespace(self):
        # Drift guard mirroring tests/test_v0510.py model-id safety check.
        from soup_cli.recipes.catalog import RECIPES

        new = (
            "orpheus-tts-sft", "sesame-csm-tts", "llasa-tts",
            "spark-tts", "oute-tts", "falcon-e-bitnet-sft",
        )
        for name in new:
            base = RECIPES[name].model
            assert base, f"{name}: model is empty"
            assert "\x00" not in base, f"{name}: model contains null byte"
            assert base.strip() == base, f"{name}: model has surrounding whitespace"
            for part in base.split("/"):
                assert part, f"{name}: model has empty path component"


class TestBitnetExportCli:
    """v0.52.0 Part D format allowlist — live wiring in v0.71.20 #134
    (see test_v07120.py for the live bitnet-export CLI coverage)."""

    def test_export_format_bitnet_lists_in_help(self):
        from soup_cli.commands.export import SUPPORTED_FORMATS

        assert "bitnet" in SUPPORTED_FORMATS
        assert "tq1_0" in SUPPORTED_FORMATS


class TestReviewFixes:
    """Coverage of the review fixes applied between v0.52.0 first cut + ship."""

    def test_num_labels_bool_rejected_at_schema(self):
        # Pydantic ge=1 accepts True (subclass of int); the explicit
        # field_validator(mode="before") rejects bool.
        yaml = (
            "base: x\ntask: classifier\ndata: {train: ./d.jsonl}\n"
            "training: {num_labels: true}\n"
        )
        with pytest.raises(Exception, match="num_labels"):
            load_config_from_string(yaml)

    def test_reasoning_effort_canonicalised_via_validator(self):
        cfg = load_config_from_string(
            "base: x\ntask: sft\ndata: {train: ./d.jsonl}\n"
            "training: {reasoning_effort: HIGH}\n"
        )
        assert cfg.training.reasoning_effort == "high"

    def test_reasoning_effort_task_gate(self):
        yaml = (
            "base: x\ntask: dpo\ndata: {train: ./d.jsonl}\n"
            "training: {reasoning_effort: high}\n"
        )
        with pytest.raises(Exception, match="reasoning_effort"):
            load_config_from_string(yaml)

    def test_train_on_eot_task_gate(self):
        yaml = (
            "base: x\ntask: dpo\ndata: {train: ./d.jsonl}\n"
            "training: {train_on_eot: true}\n"
        )
        with pytest.raises(Exception, match="train_on_eot"):
            load_config_from_string(yaml)

    def test_oute_emotion_allowlist(self):
        from soup_cli.utils.tts import validate_emotion_tag

        assert validate_emotion_tag("happy", family="oute") == "happy"
        with pytest.raises(ValueError, match="oute allowlist"):
            validate_emotion_tag("demonic", family="oute")

    def test_distill_divergence_literal_excludes_kl(self):
        # The Literal-stored value is always the canonical form; "kl" is
        # accepted at parse time (alias) but the field never holds "kl".
        cfg = load_config_from_string(
            "base: x\ntask: distill\ndata: {train: ./d.jsonl}\n"
            "training: {teacher_model: t/m, distill_divergence: kl}\n"
        )
        assert cfg.training.distill_divergence == "forward_kl"

    def test_divergences_derived_from_aliases(self):
        from soup_cli.utils.distill import _DIVERGENCE_ALIASES, DIVERGENCES

        # Drift guard — adding a new alias updates both surfaces.
        assert DIVERGENCES == set(_DIVERGENCE_ALIASES.keys())

    def test_validate_tts_compat_bool_rejected(self):
        from soup_cli.utils.tts import validate_tts_compat

        with pytest.raises(TypeError):
            validate_tts_compat(task=True, modality="audio_out", backend="transformers")

    def test_validate_moe_quant_bool_moe_lora_rejected(self):
        from soup_cli.utils.moe_quant import validate_moe_expert_quant_compat

        with pytest.raises(TypeError):
            validate_moe_expert_quant_compat(backend="transformers", moe_lora=1)  # type: ignore[arg-type]


def test_v0520_finite_helper():
    """Sanity guard: distill temperature must use math.isfinite (not just le)."""
    from soup_cli.utils.distill import validate_distill_temperature

    # math.isfinite is the canonical rejector; Pydantic le=100 also rejects inf
    # but only NaN slips through Field bounds incidentally.
    assert math.isfinite(2.0)
    with pytest.raises(ValueError, match="finite"):
        validate_distill_temperature(float("nan"))
