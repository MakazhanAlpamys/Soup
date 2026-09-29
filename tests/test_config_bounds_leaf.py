"""``utils/config_bounds.py`` must stay a leaf, and keep the streaming runtime off
the ``soup version`` import path (#780).

``config/schema.py`` takes its stream-buffer, read-ahead and noise-floor bounds
from the leaf. If the leaf grows an import, or the schema goes back to importing
a runtime module, six modules (``layer_stream``, ``layer_shard``,
``async_disk_source``, ``safetensors_reader``, ``queue``, ``_queue``) creep back
onto CLI startup with no other test noticing.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import soup_cli.config.schema as schema
import soup_cli.utils.async_disk_source as async_disk_source
import soup_cli.utils.config_bounds as config_bounds
import soup_cli.utils.layer_stream as layer_stream
import soup_cli.utils.ship_verdict as ship_verdict

# queue/_queue are left out: a third-party dependency may import them on some
# Python or platform, which would make the assertion flaky for non-Soup reasons.
_OFF_THE_CLI_PATH = (
    "soup_cli.utils.layer_stream",
    "soup_cli.utils.layer_shard",
    "soup_cli.utils.async_disk_source",
    "soup_cli.utils.safetensors_reader",
)


def _imported_modules(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_leaf_imports_nothing():
    imported = [m for m in _imported_modules(Path(config_bounds.__file__)) if m != "__future__"]
    assert not imported, (
        f"config_bounds imports {imported}; it must stay a leaf or the schema drags "
        "that module's subtree onto `soup version` (or `import soup_cli.config.schema`)."
    )


def _loaded_after(code: str, modules: tuple[str, ...]) -> list[str]:
    probe = f"import sys\n{code}\nprint(','.join(m for m in {modules!r} if m in sys.modules))"
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return [m for m in proc.stdout.strip().split(",") if m]


def test_cli_import_loads_none_of_the_streaming_runtime():
    loaded = _loaded_after("import soup_cli.cli", _OFF_THE_CLI_PATH)
    assert not loaded, (
        f"`import soup_cli.cli` loaded {loaded}. Something on the CLI import path "
        "imports the streaming runtime eagerly; config bounds belong in utils/config_bounds."
    )


def test_probe_reports_every_module_when_loaded():
    """Control: each name is reported when it IS loaded."""
    imports = "\n".join(f"import {m}" for m in _OFF_THE_CLI_PATH)
    assert _loaded_after(f"import soup_cli.cli\n{imports}", _OFF_THE_CLI_PATH) == list(
        _OFF_THE_CLI_PATH
    )


@pytest.mark.parametrize(
    ("module", "name", "leaf_name"),
    [
        (layer_stream, "MIN_STREAM_BUFFERS", "MIN_STREAM_BUFFERS"),
        (layer_stream, "MAX_STREAM_BUFFERS", "MAX_STREAM_BUFFERS"),
        (layer_stream, "DEFAULT_STREAM_BUFFERS", "DEFAULT_STREAM_BUFFERS"),
        (layer_stream, "MIN_STREAM_READ_AHEAD", "MIN_STREAM_READ_AHEAD"),
        (layer_stream, "MAX_STREAM_READ_AHEAD", "MAX_STREAM_READ_AHEAD"),
        (layer_stream, "DEFAULT_STREAM_READ_AHEAD", "DEFAULT_STREAM_READ_AHEAD"),
        (layer_stream, "SUPPORTED_STREAM_TASKS", "SUPPORTED_STREAM_TASKS"),
        (layer_stream, "ROLLOUT_STREAM_TASKS", "ROLLOUT_STREAM_TASKS"),
        (async_disk_source, "MIN_STREAM_READ_AHEAD", "MIN_STREAM_READ_AHEAD"),
        (async_disk_source, "MAX_STREAM_READ_AHEAD", "MAX_STREAM_READ_AHEAD"),
        (async_disk_source, "DEFAULT_STREAM_READ_AHEAD", "DEFAULT_STREAM_READ_AHEAD"),
        (ship_verdict, "MIN_NOISE_FLOOR_RUNS", "MIN_NOISE_FLOOR_RUNS"),
        (ship_verdict, "MAX_NOISE_FLOOR_RUNS", "MAX_NOISE_FLOOR_RUNS"),
        (schema, "MIN_STREAM_BUFFERS", "MIN_STREAM_BUFFERS"),
        (schema, "MAX_STREAM_BUFFERS", "MAX_STREAM_BUFFERS"),
        (schema, "DEFAULT_STREAM_BUFFERS", "DEFAULT_STREAM_BUFFERS"),
        (schema, "MIN_STREAM_READ_AHEAD", "MIN_STREAM_READ_AHEAD"),
        (schema, "MAX_STREAM_READ_AHEAD", "MAX_STREAM_READ_AHEAD"),
        (schema, "DEFAULT_STREAM_READ_AHEAD", "DEFAULT_STREAM_READ_AHEAD"),
        (schema, "_STREAM_SUPPORTED_TASKS", "SUPPORTED_STREAM_TASKS"),
        (schema, "_STREAM_ROLLOUT_TASKS", "ROLLOUT_STREAM_TASKS"),
        (schema, "MIN_NOISE_FLOOR_RUNS", "MIN_NOISE_FLOOR_RUNS"),
        (schema, "MAX_NOISE_FLOOR_RUNS", "MAX_NOISE_FLOOR_RUNS"),
    ],
    ids=lambda v: v.__name__.rsplit(".", 1)[-1] if hasattr(v, "__name__") else v,
)
def test_bound_comes_from_the_leaf_and_is_never_redeclared(module, name, leaf_name):
    """Re-exported, not copied: a redeclared value could drift from the schema bound.

    Checked on the source, not with ``is`` alone: CPython caches small ints, so a
    copied ``MIN_STREAM_BUFFERS = 2`` would still be identical to the leaf's object.
    Every ``Name`` in store context counts, so tuple unpacking, ``+=``, ``for``
    targets and the walrus operator are caught as well as a plain assignment.
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    from_leaf = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "soup_cli.utils.config_bounds"
        for alias in node.names
    }
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert name in from_leaf, f"{module.__name__} must import {name} from config_bounds"
    assert name not in assigned, f"{module.__name__} redeclares {name}; re-export it instead"
    assert getattr(module, name) is getattr(config_bounds, leaf_name)
