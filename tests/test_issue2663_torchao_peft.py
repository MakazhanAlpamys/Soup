"""soup doctor reports a torchao that makes peft refuse to load (#2663)."""

import sys
import types

from soup_cli.cli import app
from soup_cli.utils.torchao_compat import TORCHAO_MIN_VERSION
from tests.conftest import strip_ansi
from tests.test_doctor import _mock_clean_doctor_env, runner


def _fake_peft(monkeypatch, error=None):
    """Replace peft.import_utils with a fake whose is_torchao_available()
    raises `error` (like peft with an old torchao) or returns False."""
    peft_pkg = types.ModuleType("peft")
    import_utils = types.ModuleType("peft.import_utils")

    def is_torchao_available():
        if error is not None:
            raise error
        return False

    import_utils.is_torchao_available = is_torchao_available
    peft_pkg.import_utils = import_utils
    monkeypatch.setitem(sys.modules, "peft", peft_pkg)
    monkeypatch.setitem(sys.modules, "peft.import_utils", import_utils)


def test_doctor_old_torchao_both_findings_agree(monkeypatch):
    """torchao 0.10.0: the existing 'Upgrade torchao' line and the new finding both
    print and point to the same upgrade target, so the advice never conflicts (#2663)."""
    _mock_clean_doctor_env(monkeypatch)
    fake_torchao = types.ModuleType("torchao")
    fake_torchao.__version__ = "0.10.0"
    monkeypatch.setitem(sys.modules, "torchao", fake_torchao)
    _fake_peft(monkeypatch, ImportError("Found an incompatible version of torchao: 0.10.0"))

    result = runner.invoke(app, ["doctor"])
    out = strip_ansi(result.output)

    assert result.exit_code == 0, out
    assert "Upgrade torchao" in out
    assert "stops peft from loading" in out
    upgrade = f'pip install "torchao>={TORCHAO_MIN_VERSION}"'
    assert out.count(upgrade) >= 2  # same target in both findings


def test_doctor_reports_torchao_that_peft_refuses(monkeypatch):
    """peft refuses the installed torchao -> a finding with both fixes, uninstall first."""
    _mock_clean_doctor_env(monkeypatch)
    _fake_peft(monkeypatch, ImportError("Found an incompatible version of torchao: 0.10.0"))

    result = runner.invoke(app, ["doctor"])
    out = strip_ansi(result.output)

    assert result.exit_code == 0, out  # advisory only: exit code unchanged
    assert "pip uninstall -y torchao" in out
    assert out.index("pip uninstall -y torchao") < out.index('pip install "torchao>=')


def test_doctor_silent_when_peft_accepts_torchao(monkeypatch):
    """peft loads with the installed torchao -> no torchao finding (#2663)."""
    _mock_clean_doctor_env(monkeypatch)
    _fake_peft(monkeypatch)

    result = runner.invoke(app, ["doctor"])
    out = strip_ansi(result.output)

    assert result.exit_code == 0, out  # advisory only: exit code unchanged
    assert "pip uninstall -y torchao" not in out


def test_doctor_silent_when_peft_not_installed(monkeypatch):
    """peft is not installed -> nothing to check, no torchao finding (#2663)."""
    _mock_clean_doctor_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "peft", None)
    monkeypatch.setitem(sys.modules, "peft.import_utils", None)

    result = runner.invoke(app, ["doctor"])
    out = strip_ansi(result.output)

    assert result.exit_code == 0, out  # advisory only: exit code unchanged
    assert "pip uninstall -y torchao" not in out
