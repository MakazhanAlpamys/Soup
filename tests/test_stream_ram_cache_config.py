"""An opt-in RAM budget must reach the SSD source and be visible to users."""

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def payload(value=0):
    return {
        "base": "local-model",
        "task": "sft",
        "data": {"train": "train.jsonl", "format": "alpaca"},
        "training": {
            "stream_layers": True,
            "batch_size": 1,
            "quantization": "none",
            "stream_ram_cache_mb": value,
        },
    }


@pytest.mark.parametrize("value", [0, 1, 14336])
def test_cache_budget_round_trips(value):
    config = SoupConfig.model_validate(payload(value))
    assert config.training.stream_ram_cache_mb == value
    assert (
        SoupConfig.model_validate_json(config.model_dump_json()).training.stream_ram_cache_mb
        == value
    )


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "8", "auto", None, []])
def test_cache_budget_is_a_strict_nonnegative_integer(value):
    with pytest.raises(ValidationError, match="stream_ram_cache_mb"):
        SoupConfig.model_validate(payload(value))


def test_cache_disabled_by_default():
    data = payload()
    del data["training"]["stream_ram_cache_mb"]
    assert SoupConfig.model_validate(data).training.stream_ram_cache_mb == 0


def _text(excinfo):
    return " ".join(str(excinfo.value).split())


def test_cache_requires_streaming_and_names_the_gate():
    data = payload(1024)
    data["training"]["stream_layers"] = False
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(data)
    assert "training.stream_ram_cache_mb set but stream_layers is false" in _text(excinfo)


def test_explicit_full_ram_tier_rejects_redundant_partial_cache():
    data = payload(1024)
    data["training"]["stream_source"] = "ram"
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig.model_validate(data)
    assert (
        "training.stream_ram_cache_mb=1024 has no effect with training.stream_source='ram'"
        in _text(excinfo)
    )


@pytest.mark.parametrize("source", ["auto", "disk"])
def test_cache_is_accepted_with_auto_and_disk_tiers(source):
    data = payload(1024)
    data["training"]["stream_source"] = source
    assert SoupConfig.model_validate(data).training.stream_ram_cache_mb == 1024


def test_cache_budget_reaches_the_source_after_setup(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import tests.test_issue623_stream_pin_allocation as wiring

    runtimes = []

    class Recording(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runtimes.append(self)

    monkeypatch.setattr(wiring, "MagicMock", Recording)
    kwargs, _text = wiring._drive_wiring(
        tmp_path,
        monkeypatch,
        stream_pin_yaml="",
        extra_training_yaml="  stream_source: disk\n  stream_ram_cache_mb: 1024\n",
    )
    assert "ram_cache_bytes" not in kwargs
    # The first MagicMock the wiring creates is the runtime its spy returns.
    runtimes[0].source.configure_ram_cache.assert_called_once_with(1024 * 2**20)


def test_cache_failure_closes_the_created_runtime(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import tests.test_issue623_stream_pin_allocation as wiring

    runtimes = []

    class Recording(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Record first: reading ``self.source`` below builds a child mock through this
            # very constructor, so a "first instance" test must already be false by then.
            first = not runtimes
            runtimes.append(self)
            if first:
                self.source.configure_ram_cache.side_effect = OSError("cache read sentinel")

    monkeypatch.setattr(wiring, "MagicMock", Recording)
    with pytest.raises(OSError, match="cache read sentinel"):
        wiring._drive_wiring(
            tmp_path,
            monkeypatch,
            stream_pin_yaml="",
            extra_training_yaml="  stream_source: disk\n  stream_ram_cache_mb: 1024\n",
        )
    runtimes[0].close.assert_called_once()


def test_default_setup_never_touches_the_cache(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import tests.test_issue623_stream_pin_allocation as wiring

    runtimes = []

    class Recording(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runtimes.append(self)

    monkeypatch.setattr(wiring, "MagicMock", Recording)
    wiring._drive_wiring(
        tmp_path, monkeypatch, stream_pin_yaml="", extra_training_yaml="  stream_source: disk\n"
    )
    runtimes[0].source.configure_ram_cache.assert_not_called()


def test_configure_rejects_a_non_integer_ceiling(tmp_path, monkeypatch):
    from tests.test_stream_ram_cache import source

    _folder, _specs, ours = source(tmp_path, monkeypatch, budget=0)
    try:
        for bad in (True, -1, 1.5, "1"):
            with pytest.raises(ValueError, match="RAM cache byte ceiling"):
                ours.configure_ram_cache(bad)
    finally:
        ours.close()


def test_operator_line_distinguishes_cache_from_staging():
    from soup_cli.trainer.stream_setup import _stream_source_line

    text = _stream_source_line(
        {
            "tier": "disk",
            "pinned": True,
            "pinned_bytes": 6 * 2**30,
            "disk_bytes": 20 * 2**30,
            "store_bytes": 2 * 2**30,
            "read_ahead": 2,
            "ram_cache_allocated_bytes": 4 * 2**30,
            "ram_cache_indices": [3, 4],
        }
    )
    assert "host staging" in text and "RAM cache" in text and "2 layers" in text
