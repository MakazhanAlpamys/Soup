"""v0.70.0 Part A — Reward-hacking detector schema + math kernels.

Schema-only release: live trainer-callback wiring deferred to v0.70.1.
Tests cover: closed allowlist, frozen dataclasses, cluster-separation
index kernel, RM-ensemble divergence kernel, classification helper,
and all input rejection matrices.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

# ---------------------------------------------------------------------------
# Reward-hacking detector — public surface
# ---------------------------------------------------------------------------


class TestRewardHackingPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import reward_hacking

        assert hasattr(reward_hacking, "SUPPORTED_HACK_DETECTORS")
        assert hasattr(reward_hacking, "validate_hack_detector")
        assert hasattr(reward_hacking, "compute_cluster_separation")
        assert hasattr(reward_hacking, "compute_rm_ensemble_divergence")
        assert hasattr(reward_hacking, "classify_hack_signal")
        assert hasattr(reward_hacking, "RewardHackReport")
        assert hasattr(reward_hacking, "build_reward_hack_callback")

    def test_supported_detectors_is_frozenset(self):
        from soup_cli.utils.reward_hacking import SUPPORTED_HACK_DETECTORS

        assert isinstance(SUPPORTED_HACK_DETECTORS, frozenset)
        assert "info_rm" in SUPPORTED_HACK_DETECTORS
        assert "rm_ensemble" in SUPPORTED_HACK_DETECTORS

    def test_supported_detectors_immutable(self):
        from soup_cli.utils.reward_hacking import SUPPORTED_HACK_DETECTORS

        with pytest.raises((AttributeError, TypeError)):
            SUPPORTED_HACK_DETECTORS.add("evil")


class TestValidateHackDetector:
    def test_happy_path(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        assert validate_hack_detector("info_rm") == "info_rm"
        assert validate_hack_detector("rm_ensemble") == "rm_ensemble"

    def test_case_insensitive(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        assert validate_hack_detector("INFO_RM") == "info_rm"
        assert validate_hack_detector("Rm_Ensemble") == "rm_ensemble"

    def test_unknown_raises(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="not supported"):
            validate_hack_detector("evil")

    def test_bool_rejected(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="bool"):
            validate_hack_detector(True)

    def test_empty_rejected(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="non-empty"):
            validate_hack_detector("")

    def test_non_string_rejected(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="string"):
            validate_hack_detector(123)

    def test_null_byte_rejected(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="null byte"):
            validate_hack_detector("info_rm\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.reward_hacking import validate_hack_detector

        with pytest.raises(ValueError, match="exceeds"):
            validate_hack_detector("x" * 64)


class TestComputeClusterSeparation:
    """InfoRM Cluster-Separation Index kernel.

    Higher separation = healthier RM (clusters of (good, bad) responses
    are well-separated). Sharp drop in separation across training =
    reward-hacking signal.
    """

    def test_perfect_separation(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        good_scores = [10.0, 9.5, 11.0]
        bad_scores = [1.0, 0.5, 1.5]
        # Far-apart clusters → high separation
        value = compute_cluster_separation(good_scores, bad_scores)
        assert math.isfinite(value)
        assert value > 1.0

    def test_no_separation(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        good_scores = [5.0, 5.5, 4.5]
        bad_scores = [5.0, 5.5, 4.5]
        value = compute_cluster_separation(good_scores, bad_scores)
        assert math.isfinite(value)
        assert abs(value) < 0.1

    def test_zero_variance_groups(self):
        """Zero variance is gracefully handled (small epsilon)."""
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        good_scores = [5.0, 5.0, 5.0]
        bad_scores = [1.0, 1.0, 1.0]
        value = compute_cluster_separation(good_scores, bad_scores)
        assert math.isfinite(value)
        assert value > 0.0

    def test_empty_good_rejected(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        with pytest.raises(ValueError, match="empty"):
            compute_cluster_separation([], [1.0])

    def test_empty_bad_rejected(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        with pytest.raises(ValueError, match="empty"):
            compute_cluster_separation([1.0], [])

    def test_non_finite_rejected(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        with pytest.raises(ValueError, match="finite"):
            compute_cluster_separation([1.0, float("nan")], [0.0])
        with pytest.raises(ValueError, match="finite"):
            compute_cluster_separation([float("inf")], [0.0])

    def test_bool_in_list_rejected(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        with pytest.raises(ValueError, match="bool"):
            compute_cluster_separation([1.0, True], [0.0])

    def test_non_list_rejected(self):
        from soup_cli.utils.reward_hacking import compute_cluster_separation

        with pytest.raises(TypeError):
            compute_cluster_separation("not a list", [0.0])


class TestComputeRmEnsembleDivergence:
    """RM-ensemble divergence — measures disagreement across RMs.

    Returns mean pairwise variance across RM score lists. High variance
    = RMs disagree = reward signal is unreliable.
    """

    def test_perfect_agreement(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        rm_scores = [
            [1.0, 2.0, 3.0],  # RM 1
            [1.0, 2.0, 3.0],  # RM 2 — identical
            [1.0, 2.0, 3.0],  # RM 3 — identical
        ]
        value = compute_rm_ensemble_divergence(rm_scores)
        assert math.isfinite(value)
        assert value < 1e-6

    def test_disagreement(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        rm_scores = [
            [1.0, 2.0, 3.0],
            [5.0, 7.0, 9.0],  # Wildly different
            [-2.0, -1.0, 0.0],
        ]
        value = compute_rm_ensemble_divergence(rm_scores)
        assert math.isfinite(value)
        assert value > 1.0

    def test_single_rm_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(ValueError, match="at least 2"):
            compute_rm_ensemble_divergence([[1.0, 2.0]])

    def test_empty_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(ValueError, match="at least 2"):
            compute_rm_ensemble_divergence([])

    def test_uneven_lengths_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(ValueError, match="length"):
            compute_rm_ensemble_divergence([[1.0, 2.0], [1.0, 2.0, 3.0]])

    def test_non_finite_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(ValueError, match="finite"):
            compute_rm_ensemble_divergence([[1.0, float("nan")], [0.0, 0.0]])

    def test_non_list_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(TypeError):
            compute_rm_ensemble_divergence("not a list")

    def test_bool_inner_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        with pytest.raises(ValueError, match="bool"):
            compute_rm_ensemble_divergence([[1.0, True], [0.0, 0.0]])

    def test_too_many_rms_rejected(self):
        from soup_cli.utils.reward_hacking import compute_rm_ensemble_divergence

        too_many = [[1.0] for _ in range(100)]
        with pytest.raises(ValueError, match="too many"):
            compute_rm_ensemble_divergence(too_many)


class TestClassifyHackSignal:
    """OK / WARN / HACK taxonomy matches v0.26 Quant-Lobotomy bands.

    Lower signal = healthier. Threshold semantics:
    - drop_pct < 0.10 -> OK
    - 0.10 <= drop_pct < 0.30 -> WARN
    - drop_pct >= 0.30 -> HACK
    """

    def test_ok_threshold(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        assert classify_hack_signal(0.05) == "OK"
        assert classify_hack_signal(0.0) == "OK"

    def test_warn_threshold(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        assert classify_hack_signal(0.10) == "WARN"
        assert classify_hack_signal(0.20) == "WARN"
        assert classify_hack_signal(0.29) == "WARN"

    def test_hack_threshold(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        assert classify_hack_signal(0.30) == "HACK"
        assert classify_hack_signal(0.50) == "HACK"
        assert classify_hack_signal(1.0) == "HACK"

    def test_negative_signal_rejected(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        with pytest.raises(ValueError, match="non-negative"):
            classify_hack_signal(-0.1)

    def test_non_finite_rejected(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        with pytest.raises(ValueError, match="finite"):
            classify_hack_signal(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            classify_hack_signal(float("inf"))

    def test_bool_rejected(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        with pytest.raises(ValueError, match="bool"):
            classify_hack_signal(True)

    def test_non_number_rejected(self):
        from soup_cli.utils.reward_hacking import classify_hack_signal

        with pytest.raises(ValueError, match="number"):
            classify_hack_signal("0.5")


class TestRewardHackReport:
    def test_basic_construction(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        report = RewardHackReport(
            detector="info_rm",
            signal=0.15,
            verdict="WARN",
            step=100,
            baseline_signal=0.05,
            details=("cluster sep dropped 12%",),
        )
        assert report.detector == "info_rm"
        assert report.signal == 0.15
        assert report.verdict == "WARN"
        assert report.step == 100

    def test_frozen(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        report = RewardHackReport(
            detector="info_rm",
            signal=0.15,
            verdict="WARN",
            step=100,
            baseline_signal=0.05,
            details=(),
        )
        with pytest.raises(FrozenInstanceError):
            report.signal = 0.2  # type: ignore[misc]

    def test_invalid_detector_rejected(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(ValueError, match="not supported"):
            RewardHackReport(
                detector="evil",
                signal=0.0,
                verdict="OK",
                step=0,
                baseline_signal=0.0,
                details=(),
            )

    def test_invalid_verdict_rejected(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(ValueError, match="verdict"):
            RewardHackReport(
                detector="info_rm",
                signal=0.0,
                verdict="EVIL",
                step=0,
                baseline_signal=0.0,
                details=(),
            )

    def test_negative_signal_rejected(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(ValueError):
            RewardHackReport(
                detector="info_rm",
                signal=-0.1,
                verdict="OK",
                step=0,
                baseline_signal=0.0,
                details=(),
            )

    def test_negative_step_rejected(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(ValueError, match="step"):
            RewardHackReport(
                detector="info_rm",
                signal=0.0,
                verdict="OK",
                step=-1,
                baseline_signal=0.0,
                details=(),
            )

    def test_bool_step_rejected(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(ValueError, match="bool"):
            RewardHackReport(
                detector="info_rm",
                signal=0.0,
                verdict="OK",
                step=True,
                baseline_signal=0.0,
                details=(),
            )

    def test_details_must_be_tuple(self):
        from soup_cli.utils.reward_hacking import RewardHackReport

        with pytest.raises(TypeError, match="tuple"):
            RewardHackReport(
                detector="info_rm",
                signal=0.0,
                verdict="OK",
                step=0,
                baseline_signal=0.0,
                details=["not a tuple"],  # type: ignore[arg-type]
            )


class TestBuildRewardHackCallbackStub:
    """Live trainer-callback wiring deferred to v0.70.1.

    The factory validates inputs at construction time and raises
    NotImplementedError with explicit v0.70.1 marker (mirrors v0.50.0
    apply_variant_loss policy).
    """

    def test_invalid_detector_rejected_before_deferred(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        # Validation runs BEFORE NotImplementedError.
        with pytest.raises(ValueError, match="not supported"):
            build_reward_hack_callback(detector="evil")

    def test_deferred_v0701(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        with pytest.raises(NotImplementedError, match="v0.70.1"):
            build_reward_hack_callback(detector="info_rm")

    def test_bool_halt_on_hack_rejected(self):
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        with pytest.raises(TypeError, match="halt_on_hack"):
            build_reward_hack_callback(detector="info_rm", halt_on_hack="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema integration — TrainingConfig + SoupConfig
# ---------------------------------------------------------------------------


class TestSchemaTrainingConfig:
    def test_default_none(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.reward_hack_detector is None
        assert tcfg.reward_hack_halt is False

    def test_accept_info_rm(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(reward_hack_detector="info_rm")
        assert tcfg.reward_hack_detector == "info_rm"

    def test_accept_rm_ensemble(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(reward_hack_detector="rm_ensemble")
        assert tcfg.reward_hack_detector == "rm_ensemble"

    def test_unknown_detector_rejected(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(reward_hack_detector="evil")

    def test_halt_must_be_bool(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        # Pydantic wraps the TypeError raised by the field validator into
        # a ValidationError at construction time.
        with pytest.raises((ValidationError, TypeError)):
            TrainingConfig(reward_hack_halt="yes")  # type: ignore[arg-type]


class TestSchemaSoupConfigTaskGate:
    """reward_hack_detector + reward_hack_halt only meaningful on RL tasks
    (grpo / ppo). Rejected on SFT / DPO / etc. with friendly message.
    """

    def _yaml(self, task: str, detector: str = "info_rm", halt: bool = False) -> str:
        return f"""
base: meta-llama/Llama-3.1-8B
task: {task}
data:
  train: ./data/train.jsonl
  format: chatml
training:
  reward_hack_detector: {detector}
  reward_hack_halt: {str(halt).lower()}
"""

    def test_grpo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("grpo"))
        assert cfg.training.reward_hack_detector == "info_rm"

    def test_ppo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("ppo"))
        assert cfg.training.reward_hack_detector == "info_rm"

    def test_sft_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack"):
            load_config_from_string(self._yaml("sft"))

    def test_dpo_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack"):
            load_config_from_string(self._yaml("dpo"))

    def test_halt_without_detector_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="reward_hack_detector"):
            load_config_from_string(
                """
base: meta-llama/Llama-3.1-8B
task: grpo
data:
  train: ./data/train.jsonl
  format: chatml
training:
  reward_hack_halt: true
"""
            )

    def test_mlx_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError):
            load_config_from_string(
                """
base: mlx-community/Llama-3.1-8B
task: grpo
backend: mlx
data:
  train: ./data/train.jsonl
  format: chatml
training:
  reward_hack_detector: info_rm
"""
            )


# ---------------------------------------------------------------------------
# Source-grep wiring guards
# ---------------------------------------------------------------------------


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        """Lazy-import policy — torch imported inside functions only."""
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "reward_hacking.py"
        )
        body = src.read_text(encoding="utf-8")
        # Bare module-level torch import would be a perf regression.
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body

    def test_version_bumped(self):
        import soup_cli

        # We do not freeze here — checked via floor.
        major_minor = tuple(int(x) for x in soup_cli.__version__.split(".")[:2])
        # 0.70 floor.
        assert major_minor >= (0, 70)
