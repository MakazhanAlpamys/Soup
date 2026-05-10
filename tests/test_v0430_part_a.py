"""Tests for v0.43.0 Part A — Tracker integrations + PostHog opt-out."""
from __future__ import annotations

import pytest

from soup_cli.utils.trackers import (
    NEW_TRACKERS_V0_43,
    SUPPORTED_TRACKERS,
    build_telemetry_payload,
    is_new_v0_43_tracker,
    is_telemetry_enabled,
    required_tracker_package,
    resolve_report_to,
    validate_tracker_name,
)


class TestSupportedTrackers:
    def test_includes_legacy(self):
        assert "wandb" in SUPPORTED_TRACKERS
        assert "tensorboard" in SUPPORTED_TRACKERS
        assert "none" in SUPPORTED_TRACKERS

    def test_includes_v043_additions(self):
        for name in ("mlflow", "swanlab", "trackio"):
            assert name in SUPPORTED_TRACKERS

    def test_immutable(self):
        # frozenset cannot be mutated
        with pytest.raises(AttributeError):
            SUPPORTED_TRACKERS.add("evil")  # type: ignore[attr-defined]


class TestValidateTrackerName:
    @pytest.mark.parametrize(
        "name", ["mlflow", "swanlab", "trackio", "wandb", "tensorboard", "none"]
    )
    def test_happy(self, name):
        assert validate_tracker_name(name) == name

    def test_case_insensitive(self):
        assert validate_tracker_name("MLflow") == "mlflow"
        assert validate_tracker_name("SWANLAB") == "swanlab"

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown tracker"):
            validate_tracker_name("comet")

    def test_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_tracker_name(123)  # type: ignore[arg-type]

    def test_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_tracker_name("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            validate_tracker_name("mlflow\x00")

    def test_oversize(self):
        with pytest.raises(ValueError, match="exceeds max"):
            validate_tracker_name("a" * 64)


class TestRequiredTrackerPackage:
    def test_known_packages(self):
        assert required_tracker_package("mlflow") == "mlflow"
        assert required_tracker_package("swanlab") == "swanlab"
        assert required_tracker_package("trackio") == "trackio"
        assert required_tracker_package("wandb") == "wandb"

    def test_none_means_no_install(self):
        assert required_tracker_package("none") is None

    def test_unknown_returns_none(self):
        assert required_tracker_package("comet") is None

    def test_non_string_returns_none(self):
        assert required_tracker_package(None) is None  # type: ignore[arg-type]
        assert required_tracker_package(123) is None  # type: ignore[arg-type]


class TestIsNewV043Tracker:
    @pytest.mark.parametrize("name", list(NEW_TRACKERS_V0_43))
    def test_true_for_v043(self, name):
        assert is_new_v0_43_tracker(name) is True

    @pytest.mark.parametrize("name", ["wandb", "tensorboard", "none"])
    def test_false_for_legacy(self, name):
        assert is_new_v0_43_tracker(name) is False

    def test_false_for_unknown(self):
        assert is_new_v0_43_tracker("comet") is False

    def test_non_string_safe(self):
        assert is_new_v0_43_tracker(None) is False  # type: ignore[arg-type]
        assert is_new_v0_43_tracker(123) is False  # type: ignore[arg-type]

    def test_case_insensitive(self):
        assert is_new_v0_43_tracker("MLflow") is True


class TestIsTelemetryEnabled:
    def test_default_off(self):
        assert is_telemetry_enabled({}) is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes "])
    def test_explicit_on(self, val):
        assert is_telemetry_enabled({"SOUP_TELEMETRY": val}) is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "garbage"])
    def test_explicit_off_or_garbage(self, val):
        assert is_telemetry_enabled({"SOUP_TELEMETRY": val}) is False


class TestBuildTelemetryPayload:
    def test_required_fields(self):
        payload = build_telemetry_payload(
            soup_version="0.43.0", command="train"
        )
        assert payload["soup_version"] == "0.43.0"
        assert payload["command"] == "train"
        assert "python" in payload
        assert "os" in payload
        assert "arch" in payload
        assert payload["duration_seconds"] is None

    def test_with_duration(self):
        payload = build_telemetry_payload(
            soup_version="0.43.0", command="train", duration_seconds=12.5
        )
        assert payload["duration_seconds"] == 12.5

    def test_int_duration_coerced_to_float(self):
        payload = build_telemetry_payload(
            soup_version="0.43.0", command="train", duration_seconds=10
        )
        assert payload["duration_seconds"] == 10.0
        assert isinstance(payload["duration_seconds"], float)

    def test_no_user_data_in_payload(self):
        # Schema invariant: no model name / dataset path / user identifier.
        payload = build_telemetry_payload(
            soup_version="0.43.0", command="train"
        )
        # Closed key set:
        assert set(payload.keys()) == {
            "soup_version",
            "command",
            "python",
            "os",
            "arch",
            "duration_seconds",
        }

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            build_telemetry_payload(soup_version="", command="train")
        with pytest.raises(ValueError):
            build_telemetry_payload(soup_version=None, command="train")  # type: ignore[arg-type]

    def test_null_byte_version(self):
        with pytest.raises(ValueError, match="null"):
            build_telemetry_payload(soup_version="1.0\x00", command="train")

    def test_invalid_command(self):
        with pytest.raises(ValueError):
            build_telemetry_payload(soup_version="0.43.0", command="")

    def test_null_byte_command(self):
        with pytest.raises(ValueError, match="null"):
            build_telemetry_payload(soup_version="0.43.0", command="train\x00")

    def test_bool_duration_rejected(self):
        with pytest.raises(ValueError):
            build_telemetry_payload(
                soup_version="0.43.0", command="train", duration_seconds=True  # type: ignore[arg-type]
            )

    def test_nonfinite_duration_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            build_telemetry_payload(
                soup_version="0.43.0", command="train", duration_seconds=float("inf")
            )
        with pytest.raises(ValueError, match="finite"):
            build_telemetry_payload(
                soup_version="0.43.0",
                command="train",
                duration_seconds=float("nan"),
            )

    def test_negative_duration_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            build_telemetry_payload(
                soup_version="0.43.0", command="train", duration_seconds=-1
            )

    def test_python_is_major_minor_only(self):
        payload = build_telemetry_payload(
            soup_version="0.43.0", command="train"
        )
        # Should be like "3.11" not "3.11.7"
        assert payload["python"].count(".") == 1


class TestResolveReportTo:
    def test_default_none(self):
        assert resolve_report_to() == "none"

    def test_wandb(self):
        assert resolve_report_to(wandb=True) == "wandb"

    def test_tensorboard(self):
        assert resolve_report_to(tensorboard=True) == "tensorboard"

    def test_tracker_mlflow(self):
        assert resolve_report_to(tracker="mlflow") == "mlflow"

    def test_tracker_swanlab(self):
        assert resolve_report_to(tracker="swanlab") == "swanlab"

    def test_tracker_trackio(self):
        assert resolve_report_to(tracker="trackio") == "trackio"

    def test_tracker_unknown(self):
        with pytest.raises(ValueError, match="unknown tracker"):
            resolve_report_to(tracker="comet")

    def test_mutually_exclusive_wandb_tensorboard(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_report_to(wandb=True, tensorboard=True)

    def test_mutually_exclusive_wandb_tracker(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_report_to(wandb=True, tracker="mlflow")

    def test_mutually_exclusive_tensorboard_tracker(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_report_to(tensorboard=True, tracker="swanlab")

    def test_empty_tracker_treated_as_unset(self):
        assert resolve_report_to(tracker="") == "none"
        assert resolve_report_to(tracker=None) == "none"

    def test_tracker_canonicalised(self):
        assert resolve_report_to(tracker="MLflow") == "mlflow"

    def test_tracker_none_string(self):
        # The literal "none" is a valid value of the allowlist.
        assert resolve_report_to(tracker="none") == "none"


class TestRegistryImmutability:
    def test_report_to_backends_immutable(self):
        from soup_cli.utils.trackers import _REPORT_TO_BACKENDS
        with pytest.raises(TypeError):
            _REPORT_TO_BACKENDS["evil"] = "evil"  # type: ignore[index]

    def test_telemetry_none_env_uses_os_environ(self, monkeypatch):
        monkeypatch.setenv("SOUP_TELEMETRY", "1")
        assert is_telemetry_enabled(None) is True
        monkeypatch.delenv("SOUP_TELEMETRY", raising=False)
        assert is_telemetry_enabled(None) is False
