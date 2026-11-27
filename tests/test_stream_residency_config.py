"""Opt-in residency must validate early and reach the real streaming setup."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def _payload(mode=0):
    return {
        "base": "local-model",
        "task": "sft",
        "data": {"train": "train.jsonl", "format": "alpaca"},
        "training": {
            "batch_size": 1,
            "quantization": "none",
            "stream_layers": True,
            "stream_resident_layers": mode,
        },
    }


@pytest.mark.parametrize("mode", [0, 1, 4, "auto"])
def test_residency_mode_round_trips(mode):
    config = SoupConfig.model_validate(_payload(mode))
    assert config.training.stream_resident_layers == mode
    restored = SoupConfig.model_validate_json(config.model_dump_json())
    assert restored.training.stream_resident_layers == mode


@pytest.mark.parametrize("mode", [True, False, -1, 1.0, "4", "AUTO", None, []])
def test_residency_mode_is_strict(mode):
    with pytest.raises(ValidationError, match="stream_resident_layers"):
        SoupConfig.model_validate(_payload(mode))


@pytest.mark.parametrize("mode", [1, "auto"])
def test_residency_without_streaming_refuses(mode):
    payload = _payload(mode)
    payload["training"]["stream_layers"] = False
    with pytest.raises(ValidationError, match="stream_resident_layers"):
        SoupConfig.model_validate(payload)


@pytest.mark.parametrize("task", ["dpo", "orpo", "simpo", "kto"])
def test_residency_refuses_unmeasured_tasks(task):
    payload = _payload("auto")
    payload["task"] = task
    payload["training"]["batch_size"] = 2
    with pytest.raises(ValidationError, match="stream_resident_layers"):
        SoupConfig.model_validate(payload)


@pytest.mark.parametrize("mode", [2, "auto"])
def test_disk_accepts_residency_with_or_without_ram_hotset(mode):
    payload = _payload(mode)
    payload["training"]["stream_source"] = "disk"
    assert SoupConfig.model_validate(payload).training.stream_source == "disk"
    payload["training"]["stream_ram_cache_mb"] = 4096
    config = SoupConfig.model_validate(payload)
    assert config.training.stream_resident_layers == mode
    assert config.training.stream_ram_cache_mb == 4096


@pytest.mark.parametrize("mode", [2, "auto"])
def test_residency_refuses_unbudgeted_optimizer(mode):
    payload = _payload(mode)
    payload["training"]["optimizer"] = "sgd"
    with pytest.raises(ValidationError, match="requires an AdamW optimizer"):
        SoupConfig.model_validate(payload)


def _text(excinfo):
    return " ".join(str(excinfo.value).split())


@pytest.mark.parametrize("mode", [True, False, -1, 1.0, "4", "AUTO", None, []])
def test_residency_mode_error_names_the_accepted_values(mode):
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(_payload(mode))
    if isinstance(mode, bool):
        expected = "stream_resident_layers must be an int, not a bool"
    else:
        expected = (
            "training.stream_resident_layers must be 0, a positive int or 'auto'; "
            f"got {type(mode).__name__} {mode!r}"
        )
    assert [(error["loc"], error["msg"]) for error in excinfo.value.errors()] == [
        (("training", "stream_resident_layers"), f"Value error, {expected}")
    ]


@pytest.mark.parametrize("mode", [1, "auto"])
def test_residency_without_streaming_names_the_gate(mode):
    payload = _payload(mode)
    payload["training"]["stream_layers"] = False
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    assert "training.stream_resident_layers set but stream_layers is false" in _text(excinfo)


@pytest.mark.parametrize("recompute", ["off", "auto"])
def test_residency_refuses_a_non_default_recompute(recompute):
    payload = _payload("auto")
    payload["training"]["stream_recompute"] = recompute
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    assert (
        "training.stream_resident_layers cannot be combined with "
        f"training.stream_recompute='{recompute}'"
    ) in _text(excinfo)


def test_residency_refuses_activation_offloading():
    payload = _payload(2)
    payload["training"]["activation_offloading"] = "cpu"
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    assert (
        "training.stream_resident_layers cannot be combined with "
        "training.activation_offloading='cpu'"
    ) in _text(excinfo)


@pytest.mark.parametrize("task", ["dpo", "orpo"])
def test_residency_task_refusal_names_the_task(task):
    payload = _payload("auto")
    payload["task"] = task
    payload["training"]["batch_size"] = 2
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    assert (
        f"training.stream_resident_layers supports task='sft' only; got task='{task}'"
        in _text(excinfo)
    )


@pytest.mark.parametrize("backend", ["unsloth", "mlx"])
def test_residency_is_refused_where_streaming_is(backend):
    payload = _payload("auto")
    payload["backend"] = backend
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(payload)
    assert "training.stream_layers requires backend='transformers'" in _text(excinfo)


def test_non_streamed_default_is_unchanged():
    payload = _payload()
    payload["training"].pop("stream_resident_layers")
    payload["training"]["stream_layers"] = False
    assert SoupConfig.model_validate(payload).training.stream_resident_layers == 0


@pytest.mark.parametrize("mode", [2, "auto"])
def test_real_setup_passes_configured_worst_shape(tmp_path, monkeypatch, mode):
    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    calls = []

    def configure(model, runtime, selection, **kwargs):
        calls.append((model, runtime, selection, kwargs))
        return SimpleNamespace(indices=())

    monkeypatch.setattr(residency, "configure_stream_residency", configure)
    spelling = "auto" if mode == "auto" else str(mode)
    _drive_wiring(
        tmp_path,
        monkeypatch,
        stream_pin_yaml="",
        extra_training_yaml=f"  stream_resident_layers: {spelling}\n",
    )
    assert len(calls) == 1
    _model, _runtime, selection, kwargs = calls[0]
    assert selection == mode
    assert kwargs["rows"] == 1
    assert kwargs["seq_len"] == SoupConfig.model_validate(_payload()).data.max_length
    assert kwargs["vocab_size"] == 64
    assert callable(kwargs["notify"])


def test_disabled_setup_does_not_probe(tmp_path, monkeypatch):
    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    def fail(*_args, **_kwargs):
        pytest.fail("default residency must not measure or allocate")

    monkeypatch.setattr(residency, "configure_stream_residency", fail)
    _drive_wiring(tmp_path, monkeypatch, stream_pin_yaml="")


def test_residency_failure_closes_created_runtime(tmp_path, monkeypatch):
    import soup_cli.utils.stream_residency as residency
    from tests.test_issue623_stream_pin_allocation import _drive_wiring

    seen = []

    def fail(_model, runtime, *_args, **_kwargs):
        seen.append(runtime)
        raise RuntimeError("ownership failure")

    monkeypatch.setattr(residency, "configure_stream_residency", fail)
    with pytest.raises(RuntimeError, match="ownership failure"):
        _drive_wiring(
            tmp_path,
            monkeypatch,
            stream_pin_yaml="",
            extra_training_yaml="  stream_resident_layers: auto\n",
        )
    assert len(seen) == 1
    seen[0].close.assert_called_once()
