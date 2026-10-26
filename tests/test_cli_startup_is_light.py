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

from tests.conftest import strip_ansi


def _plain(text: str) -> str:
    """Whitespace-collapsed, ANSI-stripped output."""
    return " ".join(strip_ansi(text).split())

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
        "soup_cli.utils.cuda_graphs",
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
    ["--version"],
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


@pytest.mark.parametrize(
    "module",
    [
        "soup_cli.utils.reward_hack_control",
        "soup_cli.utils.reward_hacking",
        "soup_cli.utils.echo_trap",
        "soup_cli.utils.minillm",
        "soup_cli.utils.rl_checkpoint",
        "soup_cli.utils.relora",
        "soup_cli.utils.lisa",
        "soup_cli.utils.dpo_variants",
        "soup_cli.monitoring.hf_push",
        "soup_cli.monitoring.curriculum_callback",
        "soup_cli.monitoring.grpo_stability_callback",
        "soup_cli.monitoring.callback",
    ],
)
def test_callback_module_imports_without_transformers(module: str):
    """Each TrainerCallback module must be importable without pulling transformers.

    These modules used to resolve the TrainerCallback base at class-definition
    time, eagerly importing transformers (and therefore torch) at module scope.
    Issue #320 deferred class construction via PEP 562 ``__getattr__``.
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


def test_stress_sentinel_matches_the_controller():
    """``reward_stress`` copies the sentinel rather than importing it.

    The copy exists so the light CLI path does not pay for the controller's
    transformers import. Pin the two values equal, so the duplication cannot
    silently drift into two different sentinels.
    """
    from soup_cli.utils import reward_hack_control, reward_stress

    assert reward_stress.DEFAULT_SENTINEL == reward_hack_control._DEFAULT_SENTINEL


def _invoke_with_probe(argv: list[str], monkey_patch_code: str = "") -> tuple[int, list[str], str]:
    """Run an invocation in a clean subprocess, returning (exit_code, leaked_deps, output)."""
    probe = (
        "import sys\n"
        f"{monkey_patch_code}\n"
        "from typer.testing import CliRunner\n"
        "from soup_cli.cli import app\n"
        f"r = CliRunner().invoke(app, {argv!r})\n"
        "leaked = ','.join(sorted(m for m in " + repr(HEAVY) + " if m in sys.modules))\n"
        "print('---SOUPPROBE---')\n"
        "print(r.exit_code)\n"
        "print(leaked)\n"
        "print(r.output)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    parts = proc.stdout.split("---SOUPPROBE---\n", 1)[1]
    exit_code_str, leaked_str, output = parts.split("\n", 2)
    leaked = [m for m in leaked_str.strip().split(",") if m]
    return int(exit_code_str), leaked, output


@pytest.mark.parametrize(("steps", "warmup"), [("2", "5"), ("2", "2")])
def test_bench_train_warmup_refusal_is_light(tmp_path, steps, warmup):
    """bench train --steps <= --warmup must refuse before loading torch (#1817)."""
    cfg_file = tmp_path / "soup.yaml"
    cfg_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  format: auto\n"
        "training:\n"
        "  epochs: 1\n"
        "  lr: 2e-4\n",
        encoding="utf-8",
    )
    exit_code, leaked, output = _invoke_with_probe(
        ["bench", "train", "-c", str(cfg_file), "--steps", steps, "--warmup", warmup]
    )
    assert exit_code == 1
    assert (
        f"--steps {steps} leaves nothing after --warmup {warmup}: raise "
        f"--steps or lower --warmup."
    ) in _plain(output)
    assert not leaked, f"bench train early refusal pulled in {leaked}"


def test_recipes_verify_missing_config_refusal_is_light():
    """recipes verify with non-existent config must exit 3 before loading torch (#1817)."""
    exit_code, leaked, output = _invoke_with_probe(
        ["recipes", "verify", "-c", "nonexistent_recipe_file.yaml"]
    )
    assert exit_code == 3
    assert "Config not found: nonexistent_recipe_file.yaml" in _plain(output)
    assert not leaked, f"recipes verify missing config pulled in {leaked}"


def test_recipes_verify_unparseable_config_refusal_is_light(tmp_path):
    """recipes verify with unparseable config must exit 3 before loading torch (#1817)."""
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("invalid: yaml: [", encoding="utf-8")
    exit_code, leaked, output = _invoke_with_probe(
        ["recipes", "verify", "-c", str(bad_cfg)]
    )
    assert exit_code == 3
    assert "Config does not parse:" in _plain(output)
    assert not leaked, f"recipes verify unparseable config pulled in {leaked}"


def test_cuda_graphs_cpu_refusal_is_light(tmp_path):
    """infer and bench infer with --cuda-graphs on CPU must exit before loading torch (#1817)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"prompt": "hello"}\n', encoding="utf-8")
    out_file = tmp_path / "out.jsonl"

    expected_msg = (
        "CUDA graphs require every model tensor on a single CUDA device. Omit --cuda-graphs."
    )

    # bench infer on CPU (force CPU via CUDA_VISIBLE_DEVICES="" probe environment)
    env_code = "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''"
    exit_code, leaked, output = _invoke_with_probe(
        ["bench", "infer", str(model_dir), "--cuda-graphs"],
        monkey_patch_code=env_code,
    )
    assert exit_code == 1
    assert expected_msg in _plain(output)
    assert not leaked, f"bench infer --cuda-graphs pulled in {leaked}"

    # infer on CPU without --device
    exit_code, leaked, output = _invoke_with_probe(
        [
            "infer",
            "-m",
            str(model_dir),
            "-i",
            str(prompt_file),
            "-o",
            str(out_file),
            "--cuda-graphs",
        ],
        monkey_patch_code=env_code,
    )
    assert exit_code == 1
    assert expected_msg in _plain(output)
    assert not leaked, f"infer --cuda-graphs pulled in {leaked}"

    # infer on explicit --device cpu
    exit_code, leaked, output = _invoke_with_probe(
        [
            "infer",
            "-m",
            str(model_dir),
            "-i",
            str(prompt_file),
            "-o",
            str(out_file),
            "--device",
            "cpu",
            "--cuda-graphs",
        ],
    )
    assert exit_code == 1
    assert expected_msg in _plain(output)
    assert not leaked, f"infer --device cpu --cuda-graphs pulled in {leaked}"


def test_check_steps_warmup_logic():
    """check_steps_warmup helper catches steps <= warmup without pulling in torch."""
    from soup_cli.bench.train_run import check_steps_warmup

    assert check_steps_warmup(2, 5) == (
        "--steps 2 leaves nothing after --warmup 5: raise --steps or lower --warmup."
    )
    assert check_steps_warmup(2, 2) == (
        "--steps 2 leaves nothing after --warmup 2: raise --steps or lower --warmup."
    )
    assert check_steps_warmup(3, 2) is None

