"""#1731 — ``quantization: fp8`` is refused below transformers 5.17.0, with the reason.

Soup declares ``transformers>=5.16.1``. On 5.16.1 the FP8 quantizer's
``update_tp_plan`` reads ``FP8Experts._impl_tp_layer_overrides.get(impl)`` with
no default, so a load of any base without MegaMoE experts raised
``AttributeError: 'NoneType' object has no attribute 'get'`` from inside
transformers; 5.17.0 added the ``{}`` default. ``build_quantization_config_for_loader``
handed every ``fp8`` run that config, so the run died there. It now stops where
the config is built, naming the installed version, the one it needs and why.

The installed version is stubbed: nothing is loaded here. The failing load itself
is in the issue, run on 5.16.1, 5.17.0 and 5.18.0.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import quant_menu
from soup_cli.utils.quant_menu import build_quantization_config_for_loader

DOCS = Path(__file__).resolve().parent.parent / "docs" / "performance-and-quantization.md"


def _tcfg(quantization: str):
    raw = {
        "base": "org/m", "task": "sft", "data": {"train": "./x.jsonl"},
        "training": {"quantization": quantization},
    }
    return load_config_from_string(yaml.safe_dump(raw)).training


def _installed(monkeypatch, version: str) -> None:
    import transformers

    monkeypatch.setattr(transformers, "__version__", version)


def _build(quantization: str = "fp8"):
    return build_quantization_config_for_loader(tcfg=_tcfg(quantization), base="org/m")


def test_the_minimum_is_the_release_that_fixed_the_quantizer():
    assert quant_menu.FP8_MIN_TRANSFORMERS == "5.17.0"


class TestRefusedBelowTheMinimum:
    def test_the_floor_gets_the_whole_message(self, monkeypatch):
        _installed(monkeypatch, "5.16.1")

        with pytest.raises(RuntimeError) as info:
            _build()

        assert str(info.value) == (
            "quantization: fp8 needs transformers >= 5.17.0, and 5.16.1 is installed: "
            "its FP8 quantizer raises AttributeError in update_tp_plan when it loads a "
            "dense base (fixed in 5.17.0). Upgrade with "
            "pip install -U 'transformers>=5.17.0', or pick another quantization."
        )

    @pytest.mark.parametrize(
        "version", ["5.16.1", "5.16.2", "5.16.10", "5.9.0", "4.57.1", "5.17.0rc1", "5.17.0.dev0"]
    )
    def test_every_older_version_is_refused_by_name(self, monkeypatch, version):
        """Compared as versions, not as text (5.9.0 and 5.16.10 are both older).
        The two pre-releases are refused because they sort below 5.17.0; which
        of them first carried the upstream fix was not checked."""
        _installed(monkeypatch, version)

        with pytest.raises(RuntimeError, match=re.escape(f"and {version} is installed")):
            _build()

    def test_nothing_is_printed_before_the_refusal(self, monkeypatch):
        """The "dequantize-on-load" line would say the config was built."""
        _installed(monkeypatch, "5.16.1")
        printed = []
        console = types.SimpleNamespace(print=printed.append)

        with pytest.raises(RuntimeError):
            build_quantization_config_for_loader(tcfg=_tcfg("fp8"), base="org/m", console=console)

        assert printed == []


class TestBuiltFromTheMinimumUp:
    @pytest.mark.parametrize("version", ["5.17.0", "5.17.1", "5.18.0", "5.100.0", "6.0.0"])
    def test_the_dequantizing_config_is_built(self, monkeypatch, version):
        import transformers

        fp8_config = transformers.FineGrainedFP8Config
        _installed(monkeypatch, version)

        config = _build()

        assert isinstance(config, fp8_config)
        assert config.dequantize is True

    def test_the_installed_transformers_is_not_refused(self):
        """No stub: the version this suite runs on is a supported one for fp8,
        or the refusal is what it gets."""
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.17.0"):
            with pytest.raises(RuntimeError, match="needs transformers >= 5.17.0"):
                _build()
        else:
            assert _build().dequantize is True


class TestTheOtherModesOnTheFloor:
    """The refusal is about fp8: on 5.16.1 every other mode builds what it built."""

    def test_none_is_still_no_config(self, monkeypatch):
        _installed(monkeypatch, "5.16.1")

        assert _build("none") is None

    @pytest.mark.parametrize(
        ("quantization", "flag"), [("4bit", "load_in_4bit"), ("8bit", "load_in_8bit")]
    )
    def test_bitsandbytes(self, monkeypatch, quantization, flag):
        _installed(monkeypatch, "5.16.1")

        assert getattr(_build(quantization), flag) is True

    def test_eetq(self, monkeypatch):
        import transformers

        _installed(monkeypatch, "5.16.1")

        assert isinstance(_build("eetq"), transformers.EetqConfig)


def test_a_transformers_without_the_class_names_the_same_minimum(monkeypatch):
    """The old message said ">= 4.45", which no supported install can be below."""
    fake = types.ModuleType("transformers")
    fake.__version__ = "5.17.0"
    monkeypatch.setitem(sys.modules, "transformers", fake)

    with pytest.raises(RuntimeError) as info:
        _build()

    message = str(info.value)
    assert "does not expose an FP8 config" in message
    assert "transformers >= 5.17.0" in message
    assert "4.45" not in message


def test_the_quant_menu_row_names_the_same_minimum():
    rows = [
        line for line in DOCS.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `fp8` |")
    ]

    assert len(rows) == 1
    assert rows[0].rstrip().endswith(f"| transformers ≥ {quant_menu.FP8_MIN_TRANSFORMERS} |")
