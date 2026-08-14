"""Tests for GH #389 — peft's is_torchao_available() raises ImportError on an
old preinstalled torchao (Colab ships 0.10.0; peft demands >0.16.0), and the
raw message names torchao while giving no fix, so the failure looks like a
Soup bug nine frames inside get_peft_model.

Fix lives in soup_cli.utils.errors.ERROR_MAP, following the existing
pattern-based string-match mechanism exactly (no new machinery): the mapper
never imports torchao or peft, it only matches the exception text that
already happened.
"""

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from soup_cli.utils.errors import format_friendly_error

# The exact message peft's is_torchao_available() raises (peft/import_utils.py),
# reproduced verbatim from the Colab traceback in GH #389.
_REAL_PEFT_TORCHAO_MESSAGE = (
    "Found an incompatible version of torchao.\n"
    "Found version 0.10.0, but only versions above 0.16.0 are supported"
)


def test_torchao_version_error_is_mapped_to_actionable_fix():
    """The real peft message text is mapped to a friendly, actionable message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ImportError(_REAL_PEFT_TORCHAO_MESSAGE)
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()

    # Names the actionable fix command.
    assert "pip uninstall -y torchao" in output
    # Explains Soup does not need it unless QAT is configured.
    assert "quantization_aware" in output
    # Names the likely source (hosted notebook preinstall).
    assert "Colab" in output or "Kaggle" in output
    # Must NOT fall through to the generic "unknown error" path.
    assert "Run with --verbose for the full traceback" not in output


def test_torchao_version_error_mapping_does_not_import_torchao_or_peft():
    """The mapping is a pure string match — it must not import torchao/peft
    to decide, since the whole point is deciding without triggering the
    version probe that raised in the first place.

    Asserted as "imports NOTHING NEW", not as "peft is absent from
    sys.modules". The first version of this test asserted absence and passed
    when the file was run alone, then failed the moment the full suite ran it
    after anything that imports peft — which is most of this repo. What the
    mapping must not do is trigger the import itself; whether some earlier test
    already did is none of its business.
    """
    import sys

    before = set(sys.modules)

    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ImportError(_REAL_PEFT_TORCHAO_MESSAGE)
        format_friendly_error(exc, verbose=False)

    newly_imported = set(sys.modules) - before
    offenders = {m for m in newly_imported if m.split(".")[0] in {"torchao", "peft"}}
    assert not offenders, f"the mapping imported {offenders} to decide"


def test_unrelated_import_error_is_not_rewritten_as_torchao():
    """CONTROL: an unrelated ImportError must not be caught by the new
    torchao pattern (and, absent any other matching pattern, falls through
    to the generic unknown-error path)."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ImportError("cannot import name 'foo' from 'bar'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()

    assert "pip uninstall -y torchao" not in output
    assert "torchao" not in output
    # Falls through to the generic unknown-error branch.
    assert "ImportError" in output
    assert "cannot import name 'foo' from 'bar'" in output


def test_unrelated_torchao_mention_without_version_shape_is_not_rewritten():
    """CONTROL: merely mentioning torchao (e.g. a plain missing-module error)
    must not trigger the version-incompatibility fix message — only the
    specific 'incompatible version of torchao' shape peft raises should."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError("No module named 'torchao'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()

    assert "pip uninstall -y torchao" not in output
