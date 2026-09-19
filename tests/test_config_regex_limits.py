"""Regexes in soup.yaml are refused when too complex to match safely."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]

BYPASSES = ["((.+))+z", "(.+){2,}z", "(?:.|.)+z", "(a+)+b"]


def _sft_yaml(extra_training: str) -> str:
    return textwrap.dedent(
        """\
        base: HuggingFaceTB/SmolLM2-135M
        task: sft
        data:
          train: data.jsonl
        training:
          quantization: none
        """
    ) + textwrap.indent(extra_training, "  ")


@pytest.mark.parametrize("pattern", BYPASSES)
def test_unfrozen_parameters_refused(pattern):
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match="too complex"):
        load_config_from_string(_sft_yaml(f"unfrozen_parameters: ['{pattern}']\n"))


def test_unfrozen_parameters_literal_prefix_still_loads():
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(
        _sft_yaml(
            "unfrozen_parameters: "
            "['model.layers.0.mlp.down_proj', 'model\\.layers\\.\\d+\\.mlp']\n"
        )
    )
    assert cfg.training.unfrozen_parameters == [
        "model.layers.0.mlp.down_proj",
        r"model\.layers\.\d+\.mlp",
    ]


@pytest.mark.parametrize("pattern", BYPASSES)
def test_lr_groups_refused(pattern):
    from soup_cli.utils.lr_groups import parse_lr_groups

    with pytest.raises(ValueError, match="too complex"):
        parse_lr_groups([{"pattern": pattern, "lr": 1e-4}])


def test_lr_groups_refused_through_the_config_loader():
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=r"training\.lr_groups: pattern .* too complex"):
        load_config_from_string(
            _sft_yaml("lr_groups:\n  - pattern: '(.+){2,}z'\n    lr: 0.0001\n")
        )


def test_lr_groups_literal_pattern_still_parses():
    from soup_cli.utils.lr_groups import parse_lr_groups

    groups = parse_lr_groups({"q_proj": 1e-4, r"layers\.\d+\.mlp": 5e-5})
    assert [g.pattern for g in groups] == ["q_proj", r"layers\.\d+\.mlp"]


def test_lr_groups_has_no_runtime_probe():
    import inspect

    from soup_cli.utils import lr_groups

    assert '"a" * 128' not in inspect.getsource(lr_groups)


def test_lr_groups_refusal_returns_promptly_in_a_subprocess():
    code = (
        "from soup_cli.utils.lr_groups import parse_lr_groups\n"
        "try:\n"
        "    parse_lr_groups([{'pattern': '(a+)+b', 'lr': 1e-4}])\n"
        "except ValueError as exc:\n"
        "    print('refused', exc)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ,
    )
    assert "refused" in res.stdout, (res.stdout, res.stderr)


def _lora_yaml(field: str, key: str) -> str:
    return textwrap.dedent(
        f"""\
        base: HuggingFaceTB/SmolLM2-135M
        task: sft
        data:
          train: data.jsonl
        training:
          lora:
            {field}:
              "{key}": 8
        """
    )


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_lora_pattern_keys_refused(field):
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=rf"lora\.{field}: pattern .* too complex"):
        load_config_from_string(_lora_yaml(field, "(.+){2,}z"))


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_lora_pattern_keys_invalid_regex_refused(field):
    from soup_cli.config.loader import load_config_from_string

    with pytest.raises(ValueError, match=rf"lora\.{field}: invalid regex"):
        load_config_from_string(_lora_yaml(field, "(unclosed"))


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_lora_pattern_literal_keys_still_load(field):
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(_lora_yaml(field, "q_proj"))
    assert getattr(cfg.training.lora, field) == {"q_proj": 8}


# --- one key's LENGTH, not the number of keys -------------------------------


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_lora_pattern_key_at_the_cap_still_loads(field):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import _MAX_LORA_PATTERN_KEY_LEN

    key = "q" * _MAX_LORA_PATTERN_KEY_LEN
    cfg = load_config_from_string(_lora_yaml(field, key))
    assert getattr(cfg.training.lora, field) == {key: 8}


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
def test_lora_pattern_key_one_over_the_cap_is_refused(field):
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import _MAX_LORA_PATTERN_KEY_LEN

    key = "q" * (_MAX_LORA_PATTERN_KEY_LEN + 1)
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_lora_yaml(field, key))
    message = str(excinfo.value)
    assert f"lora.{field}" in message, message
    assert str(_MAX_LORA_PATTERN_KEY_LEN) in message, message


def test_lora_pattern_key_refusal_does_not_echo_the_whole_key():
    """An over-long key belongs in the config, not in the terminal refusing it."""
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.config.schema import _MAX_LORA_PATTERN_KEY_SHOWN

    # 1000 and not more: PyYAML refuses a *simple* mapping key past 1024
    # characters before the schema ever sees it.
    key = "q" * 1000
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_lora_yaml("rank_pattern", key))
    message = str(excinfo.value)
    assert key not in message, len(message)
    # Exactly the documented prefix: 80 characters, not 81.
    assert "q" * _MAX_LORA_PATTERN_KEY_SHOWN in message, message
    assert "q" * (_MAX_LORA_PATTERN_KEY_SHOWN + 1) not in message, message


# --- shipped configs -------------------------------------------------------


def _patterns_in(doc: object) -> list[str]:
    if not isinstance(doc, dict):
        return []
    training = doc.get("training")
    if not isinstance(training, dict):
        return []
    found: list[str] = []
    found.extend(p for p in training.get("unfrozen_parameters") or [] if isinstance(p, str))
    groups = training.get("lr_groups") or []
    if isinstance(groups, dict):
        found.extend(k for k in groups if isinstance(k, str))
    else:
        for entry in groups:
            if isinstance(entry, dict) and isinstance(entry.get("pattern"), str):
                found.append(entry["pattern"])
            elif isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], str):
                found.append(entry[0])
    for lora in (training.get("lora"), doc.get("lora")):
        if isinstance(lora, dict):
            for field in ("rank_pattern", "alpha_pattern"):
                mapping = lora.get(field)
                if isinstance(mapping, dict):
                    found.extend(k for k in mapping if isinstance(k, str))
    return found


def _shipped_config_texts() -> list[tuple[str, str]]:
    from soup_cli.recipes.catalog import RECIPES

    texts = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    for directory in (_ROOT / "src" / "soup_cli" / "templates", _ROOT / "examples" / "configs"):
        assert directory.is_dir(), directory
        for path in sorted(directory.glob("**/*.yaml")):
            texts.append((str(path.relative_to(_ROOT)), path.read_text(encoding="utf-8")))
    return texts


def test_shipped_configs_use_only_simple_patterns():
    from soup_cli.utils.safe_regex import regex_complexity_problem

    texts = _shipped_config_texts()
    assert len(texts) > 100, len(texts)
    problems = []
    for name, text in texts:
        for pattern in _patterns_in(yaml.safe_load(text)):
            reason = regex_complexity_problem(pattern)
            if reason is not None:
                problems.append((name, pattern, reason))
    assert problems == []
