"""Every YAML document read under ``src/soup_cli`` goes through the limited loader.

Two halves. The source scan keeps ``utils/yaml_limits.py`` the only module that
can parse YAML: no other module may hold PyYAML except to write YAML or to
catch its error class, and each one that does is listed here with exactly the
attributes it uses. The reader table then drives each existing reader with a
document past the node allowance and one past the nesting limit, and checks
that the refusal comes out in that reader's own contract.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.utils.yaml_limits import (
    ALIAS_EXPANSION_SLACK,
    MAX_YAML_NESTING,
    YamlLimitError,
    load_yaml_limited,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

#: The one module that parses YAML.
LOADER_MODULE = "utils/yaml_limits.py"

#: What another module may take from PyYAML: writing, and the error class.
_NOT_PARSING = frozenset({"safe_dump", "dump", "YAMLError"})

#: Every other module that imports PyYAML, with the attributes it uses. A new
#: entry needs one of ``_NOT_PARSING``; reading YAML goes through LOADER_MODULE.
YAML_USERS = {
    "autopilot/generate_config.py": ("safe_dump",),
    "cans/pack.py": ("safe_dump",),
    "cloud/runpod.py": ("YAMLError",),
    "commands/adapters.py": ("YAMLError",),
    "commands/deploy.py": ("YAMLError",),
    "commands/push.py": ("YAMLError",),
    "commands/ship.py": ("YAMLError",),
    "commands/spectrum.py": ("safe_dump",),
    "commands/sweep.py": ("YAMLError",),
    "config/loader.py": ("YAMLError",),
    "migrate/common.py": ("dump",),
    "ui/app.py": ("dump",),
    "utils/adapter_branch.py": ("safe_dump",),
    "utils/build_dag.py": ("YAMLError",),
    "utils/checklist_dsl.py": ("YAMLError",),
    "utils/compile_tools.py": ("safe_dump",),
    "utils/expectations.py": ("YAMLError",),
    "utils/iterative_dpo.py": ("safe_dump",),
    "utils/local_rl.py": ("safe_dump",),
    "utils/loop_stages.py": ("safe_dump",),
    "utils/mix_proxy.py": ("safe_dump",),
    "utils/onboarding.py": ("safe_dump",),
    "utils/ra_dit_run.py": ("safe_dump",),
    "utils/recipe_dag.py": ("YAMLError",),
    "utils/terraform_plan.py": ("YAMLError", "safe_dump"),
}

_YAML_PACKAGES = ("yaml", "ruamel")


def _is_yaml_package(name: object) -> bool:
    return isinstance(name, str) and name.split(".")[0] in _YAML_PACKAGES


def _yaml_uses(source: str) -> tuple[set[str], list[tuple[int, str]]]:
    """``(attributes, problems)`` for one module's use of a YAML package.

    ``attributes`` are the ``yaml.<name>`` accesses made through a plain
    ``import yaml`` (or ``import yaml as y``). ``problems`` is every other way
    of reaching the package, each of which could hide a parse: a from-import, a
    submodule import, ``ruamel``, a dynamic import by literal name, and any use
    of the module object that is not an attribute access (``y = yaml``,
    ``getattr(yaml, ...)``, passing it on).
    """
    tree = ast.parse(source)
    names: set[str] = set()
    problems: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "yaml":
                    names.add(alias.asname or "yaml")
                elif _is_yaml_package(alias.name):
                    problems.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and _is_yaml_package(node.module):
            problems.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Call):
            target = node.func
            called = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            literal = node.args[0].value if node.args and isinstance(
                node.args[0], ast.Constant
            ) else None
            if called in ("import_module", "__import__") and _is_yaml_package(literal):
                problems.append((node.lineno, f"{called}({literal!r})"))

    attributes: set[str] = set()
    through_attribute: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
        ):
            attributes.add(node.attr)
            through_attribute.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names and id(node) not in through_attribute:
            problems.append((node.lineno, f"bare use of {node.id}"))
    return attributes, sorted(problems)


@pytest.mark.parametrize(
    ("source", "attribute"),
    [
        ("import yaml\nx = yaml.safe_load(t)\n", "safe_load"),
        ("import yaml\nx = yaml.load(t, Loader=None)\n", "load"),
        ("import yaml as y\nx = y.safe_load(t)\n", "safe_load"),
        ("import yaml\nfor d in yaml.safe_load_all(t):\n    pass\n", "safe_load_all"),
        ("import yaml\nx = yaml.full_load(t)\n", "full_load"),
        ("import yaml\nx = yaml.unsafe_load(t)\n", "unsafe_load"),
        ("import yaml\nx = yaml.compose(t)\n", "compose"),
        ("import yaml\nfor e in yaml.parse(t):\n    pass\n", "parse"),
        ("import yaml\nfor e in yaml.scan(t):\n    pass\n", "scan"),
        ("import yaml\nx = yaml.SafeLoader(t).get_single_data()\n", "SafeLoader"),
        ("def f(t):\n    import yaml\n    return yaml.safe_load(t)\n", "safe_load"),
        ("import yaml\nloader = yaml.safe_load\nx = loader(t)\n", "safe_load"),
    ],
)
def test_scan_reports_each_parsing_attribute(source, attribute):
    attributes, problems = _yaml_uses(source)
    assert attributes == {attribute}
    assert not attributes <= _NOT_PARSING
    assert problems == []


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        ("from yaml import safe_load\nx = safe_load(t)\n", "from yaml import ..."),
        ("from yaml import safe_load as sl\nx = sl(t)\n", "from yaml import ..."),
        ("from yaml import *\nx = safe_load(t)\n", "from yaml import ..."),
        ("from yaml import SafeLoader\n", "from yaml import ..."),
        ("from yaml.loader import SafeLoader\n", "from yaml.loader import ..."),
        ("import yaml.loader\n", "import yaml.loader"),
        ("import yaml\ny = yaml\nx = y.safe_load(t)\n", "bare use of yaml"),
        ('import yaml\nx = getattr(yaml, "safe_load")(t)\n', "bare use of yaml"),
        ("import yaml\nx = helper(yaml, t)\n", "bare use of yaml"),
        ('import importlib\ny = importlib.import_module("yaml")\n', "import_module('yaml')"),
        (
            'from importlib import import_module\ny = import_module("yaml")\n',
            "import_module('yaml')",
        ),
        ('y = __import__("yaml")\n', "__import__('yaml')"),
        ('y = __import__("yaml.loader")\n', "__import__('yaml.loader')"),
        ("import ruamel.yaml\n", "import ruamel.yaml"),
        ("from ruamel.yaml import YAML\n", "from ruamel.yaml import ..."),
        (
            'import importlib\ny = importlib.import_module("ruamel.yaml")\n',
            "import_module('ruamel.yaml')",
        ),
    ],
)
def test_scan_reports_each_other_way_of_reaching_a_yaml_package(source, problem):
    assert [found for _, found in _yaml_uses(source)[1]] == [problem]


@pytest.mark.parametrize(
    "source",
    [
        "import yaml\ntext = yaml.safe_dump(data)\n",
        "import yaml\ntext = yaml.dump(data)\n",
        "import yaml\ntry:\n    pass\nexcept yaml.YAMLError:\n    pass\n",
        "import yaml\ntry:\n    pass\nexcept (yaml.YAMLError, OSError):\n    pass\n",
    ],
)
def test_scan_accepts_writing_yaml_and_catching_its_error(source):
    attributes, problems = _yaml_uses(source)
    assert attributes <= _NOT_PARSING
    assert problems == []


@pytest.mark.parametrize(
    "source",
    [
        "import json\nx = json.load(fh)\n",
        "x = config.load(t)\n",
        'overlay_yaml = "a: 1"\nsize = len(overlay_yaml.encode())\n',
        '"""Uses ``yaml.safe_load`` semantics."""\n',
        'y = importlib.import_module("json")\n',
        "y = importlib.import_module(name)\n",
    ],
)
def test_scan_ignores_what_is_not_a_yaml_package(source):
    assert _yaml_uses(source) == (set(), [])


def test_no_module_but_the_limited_loader_can_parse_yaml():
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 300, "the scan did not find the package"
    found: dict[str, tuple[str, ...]] = {}
    offenders: list[str] = []
    for file in files:
        rel = file.relative_to(SRC).as_posix()
        if rel == LOADER_MODULE:
            continue
        attributes, problems = _yaml_uses(file.read_text(encoding="utf-8"))
        offenders.extend(f"{rel}:{line} {problem}" for line, problem in problems)
        offenders.extend(
            f"{rel}: yaml.{name}" for name in sorted(attributes - _NOT_PARSING)
        )
        if attributes:
            found[rel] = tuple(sorted(attributes))
    assert offenders == [], (
        "read YAML with soup_cli.utils.yaml_limits.load_yaml_limited / "
        "load_yaml_file_limited instead:\n" + "\n".join(offenders)
    )
    assert found == YAML_USERS, "YAML_USERS must list each module with the attributes it uses"


def test_the_limited_loader_is_where_the_scan_expects_it():
    attributes, _ = _yaml_uses((SRC / LOADER_MODULE).read_text(encoding="utf-8"))
    assert "SafeLoader" in attributes


# ---------------------------------------------------------------------------
# Each reader, driven with a document past a limit
# ---------------------------------------------------------------------------


def _alias_fan_out(levels: int) -> str:
    """Lists that each reference the previous one ten times."""
    text = "l0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for index in range(1, levels + 1):
        text += f"l{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]\n"
    return text


def _fan_out_nodes(levels: int) -> int:
    """Nodes in a tree walk of ``_alias_fan_out(levels)``, counted without walking it."""
    list_nodes = 11  # l0: itself and its ten items
    total = 1 + (levels + 1) + list_nodes  # the root mapping, its keys, and l0
    for _ in range(levels):
        list_nodes = 1 + 10 * list_nodes
        total += list_nodes
    return total


def _fan_out_levels_just_over() -> int:
    """The fewest fan-out levels whose tree walk is past the node allowance."""
    for levels in range(1, 12):
        size = len(_alias_fan_out(levels).encode())
        if _fan_out_nodes(levels) > size + ALIAS_EXPANSION_SLACK:
            return levels
    raise AssertionError("no fan-out passes the allowance: did ALIAS_EXPANSION_SLACK grow?")


def _alias_fan_out_just_over() -> str:
    """A fan-out document that is only just past the node allowance.

    Nine levels is what this used to build: 570 bytes that PyYAML loads in an instant (the
    references stay shared) and that expand to about 1.2e10 nodes as soon as anything walks
    them. On 2026-10-14 a loader that had slipped back to a bare ``yaml.safe_load`` turned these
    tests into a run of more than ten minutes and about 31 GB. The shortest fan-out past the
    allowance is refused with the same message and costs next to nothing if no one refuses it.
    """
    return _alias_fan_out(_fan_out_levels_just_over())


def _nested_past_the_limit() -> str:
    inner = MAX_YAML_NESTING  # plus the root mapping: one level past the limit
    return "deep: " + "[" * inner + "]" * inner + "\n"


_OVER_LIMIT = {
    "node-allowance": (_alias_fan_out_just_over, "expands to more than"),
    "nesting": (
        _nested_past_the_limit,
        f"nests collections more than {MAX_YAML_NESTING} levels deep",
    ),
}


def _write(name: str, text: str) -> str:
    Path(name).write_bytes(text.encode("utf-8"))  # no newline translation
    return name


def _plan(text):
    from soup_cli.commands.plan import _load_yaml_config

    return _load_yaml_config(_write("soup.yaml", text))


def _gate_suite(text):
    from soup_cli.eval.gate import load_suite

    return load_suite(_write("gate.yaml", text))


def _rubric(text):
    from soup_cli.eval.judge import load_rubric

    return load_rubric(Path(_write("rubric.yaml", text)))


def _axolotl(text):
    from soup_cli.migrate.axolotl import migrate_axolotl

    return migrate_axolotl(Path(_write("ax.yaml", text)))


def _llamafactory(text):
    from soup_cli.migrate.llamafactory import migrate_llamafactory

    return migrate_llamafactory(Path(_write("lf.yaml", text)))


def _agent_spec(text):
    from soup_cli.utils.agent_forge import load_spec_file

    return load_spec_file(_write("spec.yaml", text))


def _build(text):
    from soup_cli.utils.build_dag import parse_build_yaml

    return parse_build_yaml(text)


def _expectations(text):
    from soup_cli.utils.expectations import parse_suite_yaml

    return parse_suite_yaml(text)


def _recipe_dag(text):
    from soup_cli.utils.recipe_dag import parse_recipe_yaml

    return parse_recipe_yaml(text)


def _checklist(text):
    from soup_cli.utils.checklist_dsl import load_checklist_spec

    return load_checklist_spec(_write("checklist.yaml", text))


def _mix_recipe(text):
    from soup_cli.utils.data_mix import load_mix_recipe

    return load_mix_recipe(_write("recipe.yaml", text))


def _mix_proxy(text):
    from soup_cli.utils.mix_proxy import _render_overlay_yaml

    return _render_overlay_yaml(text, ("a.jsonl",), (1.0,))


def _ra_dit(text):
    from soup_cli.utils.ra_dit_run import _load_yaml_config

    return _load_yaml_config(_write("ra.yaml", text))


def _quant_config(text):
    from soup_cli.utils.save_formats import load_quant_config

    return load_quant_config(_write("quant.yaml", text))


def _runpod(text):
    from soup_cli.cloud.runpod import render_runpod_stub

    return render_runpod_stub(text, gpu="a100", output_dir="out", soup_version="0.76.0")


def _sweep(text):
    from soup_cli.utils.sweep_config import parse_sweep_yaml

    return parse_sweep_yaml(text)


#: reader id -> (driver, how the refusal names the document)
_RAISING_READERS = {
    "cloud/runpod.py": (_runpod, "RunPod config"),
    "commands/plan.py": (_plan, "config 'soup.yaml'"),
    "eval/gate.py": (_gate_suite, "eval-gate suite 'gate.yaml'"),
    "eval/judge.py": (_rubric, "Rubric file rubric.yaml"),
    "migrate/axolotl.py": (_axolotl, "Config file ax.yaml"),
    "migrate/llamafactory.py": (_llamafactory, "Config file lf.yaml"),
    "utils/agent_forge.py": (_agent_spec, "spec file"),
    "utils/build_dag.py": (_build, "build text"),
    "utils/expectations.py": (_expectations, "suite text"),
    "utils/recipe_dag.py": (_recipe_dag, "recipe text"),
    "utils/checklist_dsl.py": (_checklist, "spec file"),
    "utils/data_mix.py": (_mix_recipe, "recipe"),
    "utils/mix_proxy.py": (_mix_proxy, "base YAML"),
    "utils/ra_dit_run.py": (_ra_dit, "config YAML"),
    "utils/save_formats.py": (_quant_config, "quant_config"),
    "utils/sweep_config.py": (_sweep, "sweep yaml"),
}


# A reader is driven in a child process, under a timeout and (on Linux) a memory cap, so that a
# limited loader that has regressed to a bare ``yaml.safe_load`` fails these tests in seconds.
# In process, that regression ran for more than ten minutes and reached about 31 GB
# (2026-10-14): nothing in the process could stop a reader from walking a document built to be
# walked for ever. The documents are now only just past their limits, so the child has little to
# do even then; the child and its timeout are for the day one of them is not.

#: How long a child gets to drive one reader with one document. The limited loader refuses in
#: milliseconds and the rest is interpreter start-up, so a child still running after this has
#: not refused anything.
_CHILD_TIMEOUT_SECONDS = 30

#: Address-space cap for the child on Linux, where it is enforced. Windows and macOS rely on
#: the timeout alone.
_CHILD_MEMORY_BYTES = 4 * 1024**3

_CHILD = """
import importlib, json, sys
reader, limit, memory = sys.argv[1], sys.argv[2], int(sys.argv[3])
if sys.platform.startswith("linux"):
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    except (ValueError, OSError):
        pass
tests = importlib.import_module("tests.test_yaml_reads_are_limited")
drive = tests._RAISING_READERS[reader][0]
try:
    drive(tests._OVER_LIMIT[limit][0]())
except tests.YamlLimitError as exc:
    refusal = str(exc)
else:
    refusal = None
print("REFUSED|" + json.dumps(refusal))
"""


def _child_environment() -> dict[str, str]:
    """The environment of a child that imports the same ``soup_cli`` this process imported."""
    import soup_cli

    entries = [
        str(Path(__file__).resolve().parents[1]),  # so that ``tests`` imports
        str(Path(soup_cli.__file__).resolve().parents[1]),
    ]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        entries.append(inherited)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entries), "PYTHONIOENCODING": "utf-8"}


def _refusal_in_subprocess(tmp_path: Path, reader: str, limit: str) -> str | None:
    """The ``YamlLimitError`` message from driving ``reader`` with the ``limit`` document.

    ``None`` when it raised nothing. The child runs in ``tmp_path`` (the drivers write their
    files to the working directory) and is killed after ``_CHILD_TIMEOUT_SECONDS``.
    """
    argv = [sys.executable, "-W", "ignore", "-c", _CHILD, reader, limit, str(_CHILD_MEMORY_BYTES)]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_CHILD_TIMEOUT_SECONDS,
            cwd=tmp_path,
            env=_child_environment(),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{reader} neither refused the {limit} document nor failed within "
            f"{_CHILD_TIMEOUT_SECONDS} s (child killed): is the YAML loader still the limited one?",
            pytrace=False,
        )
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-2000:])
    marked = [line for line in done.stdout.splitlines() if line.startswith("REFUSED|")]
    assert len(marked) == 1, (done.stdout[-2000:], done.stderr[-2000:])
    return json.loads(marked[0].removeprefix("REFUSED|"))


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
@pytest.mark.parametrize("reader", sorted(_RAISING_READERS))
def test_reader_refuses_a_document_past_the_limit(tmp_path, reader, limit):
    what = _RAISING_READERS[reader][1]
    message = _OVER_LIMIT[limit][1]
    refusal = _refusal_in_subprocess(tmp_path, reader, limit)
    assert refusal is not None, f"{reader} raised no YamlLimitError for the {limit} document"
    assert refusal.startswith(f"{what} {message}"), refusal


def test_the_fan_out_document_is_only_just_past_the_node_allowance():
    levels = _fan_out_levels_just_over()
    # The document every reader test is given, not only the level count it is built from. A
    # string comparison, so it fails before anything parses a document that is too large.
    assert _alias_fan_out_just_over() == _alias_fan_out(levels)
    with pytest.raises(YamlLimitError, match=r"^doc expands to more than"):
        load_yaml_limited(_alias_fan_out(levels), "doc", max_bytes=10**6)
    load_yaml_limited(_alias_fan_out(levels - 1), "doc", max_bytes=10**6)  # one level less loads


# Readers whose contract is a fallback value rather than an exception: each is
# shown returning the real value first, so the fallback is the refusal at work.


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_deploy_template_detection_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.commands.deploy import _auto_detect_template

    monkeypatch.chdir(tmp_path)
    base = "data:\n  format: chatml\n"
    _write("soup.yaml", base)
    assert _auto_detect_template() is not None
    _write("soup.yaml", base + _OVER_LIMIT[limit][0]())
    assert _auto_detect_template() is None


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_push_training_config_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.commands.push import _load_training_config

    monkeypatch.chdir(tmp_path)
    base = "base: some/model\n"
    _write("training_config.yaml", base)
    assert _load_training_config(tmp_path) == {"base": "some/model"}
    _write("training_config.yaml", base + _OVER_LIMIT[limit][0]())
    assert _load_training_config(tmp_path) == {}


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_license_frontmatter_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.utils.license_matrix import _license_from_frontmatter

    monkeypatch.chdir(tmp_path)
    readme = str(tmp_path / "README.md")
    _write("README.md", "---\nlicense: mit\n---\nbody\n")
    assert _license_from_frontmatter(readme) == "mit"
    _write("README.md", "---\nlicense: mit\n" + _OVER_LIMIT[limit][0]() + "---\nbody\n")
    assert _license_from_frontmatter(readme) is None


def test_every_routed_reader_has_a_case():
    """The table above and the modules that import the limited loader agree."""
    importers = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == LOADER_MODULE:
            continue
        text = path.read_text(encoding="utf-8")
        if any(name in text for name in _LIMITED_READS):
            importers.add(rel)
    covered_elsewhere = {
        "config/loader.py",  # tests/test_config_loader_limits.py
        "cans/unpack.py",  # tests/test_yaml_expanded_limit.py
        "commands/adapters.py",  # tests/test_config_loader_limits.py
        "commands/recipes.py",  # tests/test_config_loader_limits.py
        "commands/deploy.py",
        "commands/push.py",
        "utils/license_matrix.py",
    }
    assert importers == set(_RAISING_READERS) | covered_elsewhere


# ---------------------------------------------------------------------------
# Each reader's own size cap: one byte over it is refused in that reader's words
# ---------------------------------------------------------------------------

_LIMITED_READS = ("load_yaml_limited", "load_yaml_file_limited", "read_text_limited")


def _comment_of(size: int) -> str:
    """A YAML document that is one comment, exactly ``size`` bytes long."""
    return "#" + "x" * (size - 2) + "\n"


def _cap(module: str, name: str) -> int:
    import importlib

    return getattr(importlib.import_module(f"soup_cli.{module}"), name)


#: reader id -> (module, name of its cap, how it words a file one byte over the cap)
_READER_CAPS = {
    "cloud/runpod.py": (
        "cloud.runpod", "_MAX_CONFIG_BYTES",
        "RunPod config is larger than {cap} bytes; refusing",
    ),
    "commands/plan.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "config 'soup.yaml' is larger than {cap} bytes; refusing",
    ),
    "eval/gate.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "eval-gate suite 'gate.yaml' is larger than {cap} bytes; refusing",
    ),
    "eval/judge.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "Rubric file rubric.yaml is larger than {cap} bytes; refusing",
    ),
    "migrate/axolotl.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "Config file ax.yaml is larger than {cap} bytes; refusing",
    ),
    "migrate/llamafactory.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "Config file lf.yaml is larger than {cap} bytes; refusing",
    ),
    "utils/agent_forge.py": (
        "utils.agent_forge", "_MAX_SPEC_BYTES", "spec file exceeds {cap} bytes",
    ),
    "utils/build_dag.py": ("utils.build_dag", "_MAX_FILE_BYTES", "build text exceeds {cap} bytes"),
    "utils/expectations.py": (
        "utils.expectations", "_MAX_FILE_BYTES", "suite text exceeds {cap} bytes",
    ),
    "utils/recipe_dag.py": (
        "utils.recipe_dag", "_MAX_FILE_BYTES", "recipe text exceeds {cap} bytes",
    ),
    "utils/checklist_dsl.py": (
        "utils.checklist_dsl", "_MAX_FILE_BYTES", "spec file too large ({over} > {cap})",
    ),
    "utils/data_mix.py": (
        "utils.data_mix", "_MAX_RECIPE_BYTES", "recipe exceeds {cap} bytes cap",
    ),
    "utils/mix_proxy.py": (
        "utils.yaml_limits", "MAX_CONFIG_BYTES",
        "base YAML is larger than {cap} bytes; refusing",
    ),
    "utils/ra_dit_run.py": (
        "utils.ra_dit_run", "_MAX_YAML_BYTES", "config YAML exceeds {kib}KB cap",
    ),
    "utils/save_formats.py": (
        "utils.save_formats", "_MAX_QUANT_CONFIG_BYTES",
        "quant_config too large ({over} bytes; cap {cap})",
    ),
    "utils/sweep_config.py": (
        "utils.sweep_config", "_MAX_FILE_BYTES", "sweep yaml exceeds {cap} bytes",
    ),
}

_SIZE_WORDS = re.compile(r"larger than|exceeds|too large")


def test_every_raising_reader_has_a_cap_case():
    assert set(_READER_CAPS) == set(_RAISING_READERS)


@pytest.mark.parametrize("reader", sorted(_READER_CAPS))
def test_reader_refuses_a_document_one_byte_over_its_cap(tmp_path, monkeypatch, reader):
    monkeypatch.chdir(tmp_path)
    module, name, wording = _READER_CAPS[reader]
    cap = _cap(module, name)
    expected = wording.format(cap=cap, over=cap + 1, kib=cap // 1024)
    with pytest.raises(ValueError) as raised:
        _RAISING_READERS[reader][0](_comment_of(cap + 1))
    assert expected in str(raised.value), str(raised.value)


@pytest.mark.parametrize("reader", sorted(_READER_CAPS))
def test_reader_does_not_refuse_a_document_exactly_at_its_cap_for_size(
    tmp_path, monkeypatch, reader
):
    monkeypatch.chdir(tmp_path)
    module, name, _ = _READER_CAPS[reader]
    try:
        _RAISING_READERS[reader][0](_comment_of(_cap(module, name)))
    except (ValueError, TypeError) as exc:  # an empty document is not a valid one
        assert not _SIZE_WORDS.search(str(exc)), str(exc)


def test_deploy_template_detection_falls_back_one_byte_over_its_cap(tmp_path, monkeypatch):
    from soup_cli.commands.deploy import _auto_detect_template
    from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES

    monkeypatch.chdir(tmp_path)
    base = "data:\n  format: chatml\n"
    _write("soup.yaml", base + _comment_of(MAX_CONFIG_BYTES - len(base)))
    assert _auto_detect_template() is not None
    _write("soup.yaml", base + _comment_of(MAX_CONFIG_BYTES - len(base) + 1))
    assert _auto_detect_template() is None


def test_push_training_config_falls_back_one_byte_over_its_cap(tmp_path, monkeypatch):
    from soup_cli.commands.push import _load_training_config
    from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES

    monkeypatch.chdir(tmp_path)
    base = "base: some/model\n"
    _write("training_config.yaml", base + _comment_of(MAX_CONFIG_BYTES - len(base)))
    assert _load_training_config(tmp_path) == {"base": "some/model"}
    _write("training_config.yaml", base + _comment_of(MAX_CONFIG_BYTES - len(base) + 1))
    assert _load_training_config(tmp_path) == {}


def test_license_frontmatter_falls_back_one_byte_over_its_cap(tmp_path, monkeypatch):
    from soup_cli.utils import license_matrix

    monkeypatch.chdir(tmp_path)
    cap = license_matrix._MAX_FRONTMATTER_BYTES
    readme = str(tmp_path / "README.md")
    head, tail = "---\nlicense: mit\n", "---\nbody\n"
    _write("README.md", head + _comment_of(cap - len(head) - len(tail)) + tail)
    assert license_matrix._license_from_frontmatter(readme) == "mit"
    _write("README.md", head + _comment_of(cap - len(head) - len(tail) + 1) + tail)
    assert license_matrix._license_from_frontmatter(readme) is None


#: Every call of the limited loader names its cap: a module constant, never a number.
_CAP_NAMES = {
    "cans/unpack.py": {"MAX_CONFIG_BYTES", "MAX_MANIFEST_BYTES"},
    "cloud/runpod.py": {"_MAX_CONFIG_BYTES"},
    "commands/adapters.py": {"MAX_CONFIG_BYTES"},
    "commands/deploy.py": {"MAX_CONFIG_BYTES"},
    "commands/plan.py": {"MAX_CONFIG_BYTES"},
    "commands/push.py": {"MAX_CONFIG_BYTES"},
    "commands/recipes.py": {"MAX_CONFIG_BYTES"},
    "config/loader.py": {"MAX_CONFIG_BYTES", "limit"},
    "eval/gate.py": {"MAX_CONFIG_BYTES"},
    "eval/judge.py": {"MAX_CONFIG_BYTES"},
    "migrate/axolotl.py": {"MAX_CONFIG_BYTES"},
    "migrate/llamafactory.py": {"MAX_CONFIG_BYTES"},
    "utils/agent_forge.py": {"_MAX_SPEC_BYTES"},
    "utils/build_dag.py": {"_MAX_FILE_BYTES"},
    "utils/checklist_dsl.py": {"_MAX_FILE_BYTES"},
    "utils/data_mix.py": {"_MAX_RECIPE_BYTES"},
    "utils/expectations.py": {"_MAX_FILE_BYTES"},
    "utils/license_matrix.py": {"_MAX_FRONTMATTER_BYTES"},
    "utils/mix_proxy.py": {"MAX_CONFIG_BYTES"},
    "utils/ra_dit_run.py": {"_MAX_YAML_BYTES"},
    "utils/recipe_dag.py": {"_MAX_FILE_BYTES"},
    "utils/save_formats.py": {"_MAX_QUANT_CONFIG_BYTES"},
    "utils/sweep_config.py": {"_MAX_FILE_BYTES"},
}


def test_every_limited_read_names_its_cap():
    found: dict[str, set[str]] = {}
    for file in sorted(SRC.rglob("*.py")):
        rel = file.relative_to(SRC).as_posix()
        if rel == LOADER_MODULE:
            continue
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if called not in _LIMITED_READS:
                continue
            caps = [keyword.value for keyword in node.keywords if keyword.arg == "max_bytes"]
            assert len(caps) == 1, f"{rel}:{node.lineno} must pass max_bytes by keyword"
            assert isinstance(caps[0], ast.Name), f"{rel}:{node.lineno} max_bytes must be a name"
            found.setdefault(rel, set()).add(caps[0].id)
    assert found == _CAP_NAMES


# ---------------------------------------------------------------------------
# A reader's message quotes the path it was given: it is printed escaped
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_MARKED = "[bold]x.yaml"  # an opening tag Rich would otherwise consume

_VALID_CONFIG = (
    "base: meta-llama/Llama-3.1-8B-Instruct\ntask: sft\n"
    "data:\n  train: ./data.jsonl\n  format: alpaca\noutput: ./output\n"
)


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "gate", "--suite", _MARKED],
        ["eval", "judge", "--rubric", _MARKED, "--target", "t.jsonl", "--model", "m"],
        ["train", "--config", "soup.yaml", "--gate", _MARKED, "--dry-run"],
        ["export", "--model", "./merged", "--format", "torchao", "--quant-config", _MARKED],
    ],
    ids=["eval-gate", "eval-judge", "train-gate", "export-quant-config"],
)
def test_command_prints_a_path_with_markup_as_typed(tmp_path, monkeypatch, argv):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    _write("soup.yaml", _VALID_CONFIG)
    _write("t.jsonl", '{"prompt": "p", "response": "r"}\n')
    Path("merged").mkdir()
    result = CliRunner().invoke(app, argv)
    output = " ".join(_ANSI_RE.sub("", result.output).split())
    assert result.exit_code != 0, output
    assert _MARKED in output, (output, repr(result.exception))
