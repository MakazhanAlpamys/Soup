"""The light-CLI invariant: ``import soup_cli.cli`` must not load PyTorch.

This replaces the family of per-module ``test_no_top_level_torch`` AST guards for
the purpose of protecting CLI startup. Those guards prove a *syntactic* property
(this file has no top-level ``import torch``); the requirement is a *runtime* one
(torch is not in ``sys.modules``). An AST guard cannot see through a transitive
import, nor through a lazily-written factory that is *called* at module scope --
which is exactly how the v0.71.41 regression got in with every guard green.

One runtime assertion covers every module transitively and cannot drift.

Everything here runs in a FRESH subprocess: the pytest process has already
imported torch via other test modules, so an in-process assertion would be
vacuous (it would fail on a clean tree and pass on nothing).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Heavy deps that the documented "light core" (v0.71.0 deps-split) excludes.
# These live behind the [train] extra; the CLI must import without them.
HEAVY = ("torch", "transformers", "accelerate", "peft", "trl", "datasets", "bitsandbytes")

_PROBE = (
    "import sys\n"
    "import soup_cli.cli\n"
    "print(','.join(sorted(m for m in {heavy!r} if m in sys.modules)))\n"
)


def _loaded_heavy_deps() -> list[str]:
    """Import the CLI in a clean interpreter; return heavy deps it pulled in."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(heavy=HEAVY)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return [m for m in proc.stdout.strip().split(",") if m]


def test_cli_import_does_not_load_torch():
    """The headline invariant. A violation is a ~7x CLI startup regression."""
    leaked = _loaded_heavy_deps()
    assert not leaked, (
        f"`import soup_cli.cli` pulled in {leaked}. The light core must stay "
        "torch-free -- run scripts/find_import_leak.py to get the call site. "
        "A lazy import that is CALLED at module scope is not lazy."
    )


def test_probe_would_catch_a_regression():
    """Control: the probe must actually detect torch when it IS loaded.

    Without this, a probe that silently returned ``[]`` -- wrong module name,
    swallowed error, changed output format -- would make the test above pass
    forever while proving nothing.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; import soup_cli.cli; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip("torch not installed (light-core-only environment)")
    assert proc.stdout.strip() == "True", (proc.stdout, proc.stderr)


@pytest.mark.parametrize(
    "module",
    [
        # Modules CLAUDE.md documents as "NO top-level torch". Each is reachable
        # from a light command or is a pure planner/verdict half.
        "soup_cli.utils.reward_stress",
        "soup_cli.utils.reward_synth",
        "soup_cli.utils.ship_verdict",
        "soup_cli.utils.layer_stream",
        # Light cores that a command body imports lazily, so neither the startup
        # assertion nor a `--help` invocation reaches them. `soup mcp serve`
        # blocks (it is a stdio server), so its registry is covered here rather
        # than as an invocation.
        "soup_cli.mcp_server.registry",
        "soup_cli.utils.advise",
        "soup_cli.eval.gate_suites",
        "soup_cli.recipes.catalog",
    ],
)
def test_documented_light_module_stays_light(module: str):
    """Per-module runtime check for the modules that claim to be torch-free.

    Same reasoning as above: the claim is about ``sys.modules``, so the test has
    to be about ``sys.modules``.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, {module}; "
            "print(','.join(sorted(m for m in "
            f"{HEAVY!r} if m in sys.modules)))",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    leaked = [m for m in proc.stdout.strip().split(",") if m]
    assert not leaked, f"{module} pulled in {leaked} at import time"


def test_every_command_module_is_imported_at_startup():
    """Pins WHY the single ``import soup_cli.cli`` assertion is sufficient.

    ``cli.py`` registers all 88 command modules eagerly, so a top-level leak in
    any of them is caught by ``test_cli_import_does_not_load_torch``. If command
    registration is ever made lazy, this test goes red and the guard above stops
    being a whole-surface check -- at which point the per-invocation test below
    becomes the only coverage and needs extending.
    """
    probe = (
        "import pathlib, sys\n"
        "import soup_cli.cli\n"
        "loaded = {m for m in sys.modules if m.startswith('soup_cli.commands.')}\n"
        "root = pathlib.Path(soup_cli.cli.__file__).parent / 'commands'\n"
        "disk = {'soup_cli.commands.' + p.stem for p in root.glob('*.py')"
        " if p.stem != '__init__'}\n"
        "print(len(disk), len(disk - loaded))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    total, unloaded = (int(x) for x in proc.stdout.split())
    assert total > 50, f"only found {total} command modules -- probe is wrong"
    assert unloaded == 0, f"{unloaded}/{total} command modules are NOT imported by cli.py"


# Light commands: a user runs these to look something up, not to touch a model.
# Each is probed as a real invocation, because a command body can import a heavy
# module lazily at call time -- which the startup assertion cannot see.
LIGHT_INVOCATIONS = [
    ["version"],
    ["--help"],
    ["recipes", "list"],
    ["recipes", "search", "llama"],
    ["data", "--help"],
    ["mcp", "--help"],
    ["ship", "--help"],
    ["advise", "--help"],
    ["doctor", "--help"],
    ["reward", "--help"],
    ["draft", "list"],
]


@pytest.mark.parametrize("argv", LIGHT_INVOCATIONS, ids=lambda a: " ".join(a))
def test_light_command_invocation_stays_light(argv: list[str]):
    """Running a light command must not pull the training stack either.

    ``soup mcp serve`` is the motivating shape: ``commands/mcp.py`` is imported at
    startup, but ``mcp_server/registry.py`` is not loaded until the command body
    runs. A heavy import added there would be invisible to the startup guard.
    """
    # Marker-prefixed single line. A two-line format would collapse under
    # .strip() when the heavy-dep list is empty (i.e. on every passing run), and
    # a bare split would trip over anything the command writes to real stdout.
    probe = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from soup_cli.cli import app\n"
        f"r = CliRunner().invoke(app, {argv!r})\n"
        "print('SOUPPROBE|%s|%s' % (r.exit_code, ','.join(sorted(m for m in "
        f"{HEAVY!r} if m in sys.modules))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    marked = [ln for ln in proc.stdout.splitlines() if ln.startswith("SOUPPROBE|")]
    assert len(marked) == 1, (proc.stdout, proc.stderr)
    _, exit_code, leaked_line = marked[0].split("|", 2)
    assert exit_code == "0", f"`soup {' '.join(argv)}` exited {exit_code}"
    leaked = [m for m in leaked_line.split(",") if m]
    assert not leaked, f"`soup {' '.join(argv)}` pulled in {leaked}"


def test_stress_sentinel_matches_the_controller():
    """``reward_stress`` copies the sentinel rather than importing it.

    The copy exists so the light CLI path does not pay for the controller's
    transformers import. Pin the two values equal, so the duplication cannot
    silently drift into two different sentinels.
    """
    from soup_cli.utils import reward_hack_control, reward_stress

    assert reward_stress.DEFAULT_SENTINEL == reward_hack_control._DEFAULT_SENTINEL
