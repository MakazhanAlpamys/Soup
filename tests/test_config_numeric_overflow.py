"""Oversized numeric config values produce field-specific validation errors."""

import io

import pytest
import yaml
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config import loader
from soup_cli.config.schema import SoupConfig, TrainingConfig
from tests.conftest import strip_ansi

SITES = [
    "grpo_beta",
    "mole_temperature",
    "citation_recall_threshold",
    "distill_temperature",
    "ebft_temperature",
    "mod_capacity_factor",
    "unlearn_alpha",
    "lr_groups",
    "interleave",
]
STAGED_SITES = {"citation_recall_threshold", "lr_groups"}


def _config(site: str, value: object) -> dict:
    raw = {"base": "some-model", "data": {"train": "data.jsonl"}, "training": {}}
    if site == "interleave":
        raw["data"] = {
            "train": ["one.jsonl", "two.jsonl"],
            "interleave": {"strategy": "probs", "probs": [value, 0.5]},
        }
    elif site == "lr_groups":
        raw["training"][site] = [{"pattern": "q_proj", "lr": value}]
    else:
        raw["training"][site] = value
    return raw


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _assert_named_refusal(message: str, site: str) -> None:
    assert site in message
    if site in STAGED_SITES:
        detail = message.removeprefix("Config validation error: ")
        assert detail.startswith(f"training.{site} is accepted but read by nothing")
    else:
        assert "must be finite" in message


@pytest.fixture
def printed(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(loader, "console", Console(
        file=stream, width=180, force_terminal=True, color_system="standard",
    ))
    return stream


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_config_mapping_names_the_field_instead_of_raising_overflow(
    printed: io.StringIO, site: str, sign: int,
) -> None:
    value = sign * 10**400
    with pytest.raises(SystemExit) as exc:
        loader.validate_config_mapping(_config(site, value))
    assert exc.value.code == 1
    output = _plain(printed.getvalue())
    assert "Config validation error:" in output
    _assert_named_refusal(output, site)
    assert str(value) not in output
    assert "OverflowError" not in output


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_yaml_api_names_the_field_instead_of_raising_overflow(site: str, sign: int) -> None:
    value = sign * 10**400
    text = yaml.safe_dump(_config(site, value))
    with pytest.raises(ValueError) as exc:
        loader.load_config_from_string(text)
    message = str(exc.value)
    _assert_named_refusal(message, site)
    assert str(value) not in message


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_schema_converts_overflow_to_a_named_validation_error(site: str, sign: int) -> None:
    value = sign * 10**400
    with pytest.raises(ValidationError) as exc:
        SoupConfig(**_config(site, value))
    errors = exc.value.errors(include_input=False)
    assert len(errors) == 1
    error = errors[0]
    field = "capacity_factor" if site == "mod_capacity_factor" else site
    assert field in str(error["loc"]) + error["msg"]
    assert "must be finite" in error["msg"]
    assert str(value) not in error["msg"]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")],
                         ids=["inf", "negative-inf", "nan"])
def test_existing_nonfinite_config_values_remain_named_refusals(
    printed: io.StringIO, site: str, value: float,
) -> None:
    with pytest.raises(SystemExit) as exc:
        loader.validate_config_mapping(_config(site, value))
    assert exc.value.code == 1
    output = _plain(printed.getvalue())
    assert "Config validation error:" in output
    _assert_named_refusal(output, site)


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("value", [True, False, "not-a-number", [], {}],
                         ids=["true", "false", "text", "list", "mapping"])
def test_existing_type_refusals_keep_the_field_and_exit_code(
    printed: io.StringIO, site: str, value: object,
) -> None:
    with pytest.raises(SystemExit) as exc:
        loader.validate_config_mapping(_config(site, value))
    assert exc.value.code == 1
    output = _plain(printed.getvalue())
    assert "Config validation error:" in output
    assert site in output
    if site in STAGED_SITES:
        assert f"training.{site} is accepted but read by nothing" in output


@pytest.mark.parametrize("site", SITES)
def test_finite_values_keep_the_schema_value(site: str) -> None:
    value = 0.5
    if site == "interleave":
        config = SoupConfig(**_config(site, value))
        assert config.data.interleave["probs"] == [value, 0.5]
    else:
        training = TrainingConfig(**_config(site, value)["training"])
        if site == "lr_groups":
            assert training.lr_groups == [{"pattern": "q_proj", "lr": value}]
        else:
            assert getattr(training, site) == value
