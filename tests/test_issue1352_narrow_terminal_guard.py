"""The #808 staged-field guard must not depend on terminal width (#1352).

`test_issue808_staged_config_fields.py`'s two negative assertions in
`TestLoaderStagedFieldIntegration.test_default_equivalent_spellings_load_silently`
and `test_load_config_file_default_equivalent_spelling_loads_silently` are:

    assert "read by nothing" not in capsys.readouterr().out

the only check in the `warn` parametrisations. The warning is printed
through a module-level ``Console()`` (`soup_cli/config/loader.py`), which
wraps at the render width. If the line break falls inside "read by nothing",
the raw substring is absent even though the warning was printed, so a
future regression that makes a default-equivalent spelling warn would pass
these tests unnoticed under a narrow terminal (COLUMNS=50 reproduces it;
CI's implicit 80 columns does not).

This test proves the guard is discriminating regardless of width: it
simulates that regression (forcing the staged-field detection to treat any
value as staged, as `if val != default_val` -> `if True` would) and confirms
the guard's own tests fail under it at COLUMNS=50.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD_TEST_FILE = "tests/test_issue808_staged_config_fields.py"
STAGED_FIELDS_SRC = REPO_ROOT / "src" / "soup_cli" / "config" / "staged_fields.py"
MUTATION_NEEDLE = "if val != default_val:"
MUTATION_REPLACEMENT = "if True:  # simulated regression (#1352)"


def _run_guard_tests_under_simulated_regression(columns: str) -> subprocess.CompletedProcess:
    """Run the #808 guard's warn-case tests with a simulated regression
    (every staged field always reported as staged, matching or not) at the
    given terminal width, and return the completed pytest run."""
    original = STAGED_FIELDS_SRC.read_text()
    assert MUTATION_NEEDLE in original, "mutation target line not found in staged_fields.py"
    STAGED_FIELDS_SRC.write_text(original.replace(MUTATION_NEEDLE, MUTATION_REPLACEMENT, 1))
    try:
        env = dict(os.environ)
        env["COLUMNS"] = columns
        return subprocess.run(
            [sys.executable, "-m", "pytest", GUARD_TEST_FILE, "-k", "warn", "-q", "--no-cov"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        STAGED_FIELDS_SRC.write_text(original)


def test_narrow_terminal_does_not_mask_a_staged_field_regression() -> None:
    result = _run_guard_tests_under_simulated_regression(columns="50")
    assert result.returncode != 0, (
        "the #808 guard's warn-case tests passed under a simulated staged-field "
        "regression at COLUMNS=50: the wrapped 'read by nothing' phrase was not "
        "detected\n--- pytest output ---\n" + result.stdout[-4000:]
    )
