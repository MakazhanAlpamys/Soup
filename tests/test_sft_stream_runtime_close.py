"""Regression tests for SFT streaming runtime cleanup when setup() fails (#1464)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.trainer.sft import SFTTrainerWrapper
from tests._tiny_hf_models import build_tokenizer, tiny_model_dir


class _MockStreamRuntime:
    """Mock layer-streaming runtime with call tracking."""

    def __init__(self) -> None:
        self.close = MagicMock()
        self.total_params = 1000
        self.stats = MagicMock(
            return_value={
                "n_layers": 2,
                "buffers": 2,
                "buffer_bytes": 1024,
                "large_buffer_bytes": 0,
                "store_bytes": 2048,
                "pinned": False,
            }
        )


class _MockModel:
    """Mock model exposing trainable-parameter counts for SFT setup."""

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return (100, 1000)

    def parameters(self) -> list[Any]:
        return []


def _make_stream_sft_config(base_dir: str, *, stream_layers: bool = True) -> Any:
    """Build a minimal valid SFT configuration for testing."""
    training_dict: dict[str, Any] = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "stream_layers": stream_layers,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    if stream_layers:
        training_dict["stream_source"] = "disk"

    config_dict = {
        "base": base_dir,
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {
            "format": "chatml",
            "train": "train.jsonl",
            "max_length": 64,
        },
        "training": training_dict,
    }
    return load_config_from_string(yaml.safe_dump(config_dict))


def _install_mock_streaming(
    monkeypatch: pytest.MonkeyPatch, mock_runtime: _MockStreamRuntime
) -> None:
    """Patch _setup_streaming_transformers to attach mock runtime and model."""
    mock_model = _MockModel()
    mock_tokenizer = build_tokenizer()

    def _fake_setup_streaming(self: SFTTrainerWrapper, cfg: Any, tcfg: Any) -> None:
        self.model = mock_model  # type: ignore[assignment]
        self.tokenizer = mock_tokenizer  # type: ignore[assignment]
        self._stream_runtime = mock_runtime  # type: ignore[assignment]

    monkeypatch.setattr(
        SFTTrainerWrapper, "_setup_streaming_transformers", _fake_setup_streaming
    )


def test_training_dataset_mapping_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 1: SFT setup failure during train mapping closes the streaming runtime."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    _install_mock_streaming(monkeypatch, mock_runtime)

    close_calls: list[bool] = []
    real_close = SFTTrainerWrapper._close_stream_runtime

    def _spy_close(self: SFTTrainerWrapper) -> None:
        close_calls.append(True)
        real_close(self)

    monkeypatch.setattr(SFTTrainerWrapper, "_close_stream_runtime", _spy_close)

    def _failing_map_text_sft_rows(*args: Any, **kwargs: Any) -> None:
        raise ValueError("train row 1: no tokens remain for loss computation")

    monkeypatch.setattr(
        "soup_cli.trainer.sft._map_text_sft_rows", _failing_map_text_sft_rows
    )

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(
        ValueError, match="train row 1: no tokens remain for loss computation"
    ):
        wrapper.setup({"train": [{"messages": []}]})

    assert mock_runtime.close.call_count == 1
    assert len(close_calls) == 1


def test_evaluation_dataset_mapping_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 2: SFT setup failure during validation mapping closes the streaming runtime."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    _install_mock_streaming(monkeypatch, mock_runtime)

    close_calls: list[bool] = []
    real_close = SFTTrainerWrapper._close_stream_runtime

    def _spy_close(self: SFTTrainerWrapper) -> None:
        close_calls.append(True)
        real_close(self)

    monkeypatch.setattr(SFTTrainerWrapper, "_close_stream_runtime", _spy_close)

    def _selective_map(
        rows: list[dict], *, format_row: Any, split: str, max_length: int
    ) -> Any:
        if split == "validation":
            raise ValueError("validation row 1: loss mask is empty")
        return MagicMock()

    monkeypatch.setattr("soup_cli.trainer.sft._map_text_sft_rows", _selective_map)

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(ValueError, match="validation row 1: loss mask is empty"):
        wrapper.setup({"train": [{"messages": []}], "val": [{"messages": []}]})

    assert mock_runtime.close.call_count == 1
    assert len(close_calls) == 1


def test_setup_keyboard_interrupt(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 3: KeyboardInterrupt propagates and runtime closes exactly once."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    _install_mock_streaming(monkeypatch, mock_runtime)

    close_calls: list[bool] = []
    real_close = SFTTrainerWrapper._close_stream_runtime

    def _spy_close(self: SFTTrainerWrapper) -> None:
        close_calls.append(True)
        real_close(self)

    monkeypatch.setattr(SFTTrainerWrapper, "_close_stream_runtime", _spy_close)

    def _interrupting_map(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt("Interrupted by operator")

    monkeypatch.setattr("soup_cli.trainer.sft._map_text_sft_rows", _interrupting_map)

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(KeyboardInterrupt, match="Interrupted by operator"):
        wrapper.setup({"train": [{"messages": []}]})

    assert mock_runtime.close.call_count == 1
    assert len(close_calls) == 1


def test_once_only_cleanup_guard(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 4: _close_stream_runtime runs at most once per runtime and handles replacement."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    first_runtime = _MockStreamRuntime()
    wrapper._stream_runtime = first_runtime  # type: ignore[assignment]

    wrapper._close_stream_runtime()
    assert first_runtime.close.call_count == 1
    assert wrapper._stream_runtime is first_runtime

    # Second call on same runtime must not close again
    wrapper._close_stream_runtime()
    assert first_runtime.close.call_count == 1
    assert wrapper._stream_runtime is first_runtime

    # Assign a new runtime; it must close exactly once
    second_runtime = _MockStreamRuntime()
    wrapper._stream_runtime = second_runtime  # type: ignore[assignment]

    wrapper._close_stream_runtime()
    assert second_runtime.close.call_count == 1
    assert wrapper._stream_runtime is second_runtime

    wrapper._close_stream_runtime()
    assert second_runtime.close.call_count == 1
    assert first_runtime.close.call_count == 1
    assert wrapper._stream_runtime is second_runtime


def test_inner_cleanup_followed_by_setup_failure(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 5: An inner cleanup followed by setup failure does not double-close."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    mock_model = _MockModel()
    mock_tokenizer = build_tokenizer()

    def _inner_cleanup_setup(self: SFTTrainerWrapper, cfg: Any, tcfg: Any) -> None:
        self.model = mock_model  # type: ignore[assignment]
        self.tokenizer = mock_tokenizer  # type: ignore[assignment]
        self._stream_runtime = mock_runtime  # type: ignore[assignment]
        # Simulate an inner error that already invoked cleanup
        self._close_stream_runtime()
        raise RuntimeError("Inner VRAM probe failed")

    monkeypatch.setattr(
        SFTTrainerWrapper, "_setup_streaming_transformers", _inner_cleanup_setup
    )

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(RuntimeError, match="Inner VRAM probe failed"):
        wrapper.setup({"train": [{"messages": []}]})

    assert mock_runtime.close.call_count == 1


def test_non_streaming_control(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 6: Non-streaming setup failure propagates without runtime cleanup errors."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    def _failing_transformers(self: SFTTrainerWrapper, cfg: Any, tcfg: Any) -> None:
        raise ValueError("Model weights not found on disk")

    monkeypatch.setattr(
        SFTTrainerWrapper, "_setup_transformers", _failing_transformers
    )

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=False)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(ValueError, match="Model weights not found on disk"):
        wrapper.setup({"train": [{"messages": []}]})

    assert getattr(wrapper, "_stream_runtime", None) is None


def test_existing_training_context_cleanup(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 7: _training_context closes runtime on exit and respects idempotent guard."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    mock_runtime = _MockStreamRuntime()
    wrapper._stream_runtime = mock_runtime  # type: ignore[assignment]

    with wrapper._training_context():
        assert mock_runtime.close.call_count == 0

    assert mock_runtime.close.call_count == 1

    # Re-entering training context or calling cleanup must not close again
    with wrapper._training_context():
        pass

    assert mock_runtime.close.call_count == 1


def test_cleanup_itself_raises(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 8: Cleanup failure preserves the original setup exception."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    mock_runtime.close.side_effect = OSError("Underlying disk device disconnected")
    _install_mock_streaming(monkeypatch, mock_runtime)

    def _failing_map(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Original dataset validation failed")

    monkeypatch.setattr("soup_cli.trainer.sft._map_text_sft_rows", _failing_map)

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")

    with pytest.raises(ValueError, match="Original dataset validation failed") as exc_info:
        wrapper.setup({"train": [{"messages": []}]})

    assert "Original dataset validation failed" in str(exc_info.value)
    assert mock_runtime.close.call_count == 1


def test_successful_setup_does_not_close_runtime(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 9: Successful setup keeps the streaming runtime open for training."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    mock_runtime = _MockStreamRuntime()
    _install_mock_streaming(monkeypatch, mock_runtime)

    monkeypatch.setattr(
        "soup_cli.trainer.sft._map_text_sft_rows",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "trl.SFTTrainer.__init__",
        lambda *args, **kwargs: None,
    )

    base_dir = tiny_model_dir("llama")
    cfg = _make_stream_sft_config(base_dir, stream_layers=True)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"messages": []}]})

    assert mock_runtime.close.call_count == 0
    assert wrapper._stream_runtime is mock_runtime
