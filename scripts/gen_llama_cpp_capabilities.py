"""Generate offline capabilities from the exact pinned llama.cpp converter AST.

Supply the unmodified ``convert_hf_to_gguf.py`` entry script from the recorded
commit, plus its ``conversion/`` package for the multi-file layout (b12423 and
newer); the pre-refactor monolith needs only the entry script. This script
never imports the converter or fetches source at runtime. See
docs/serving-and-export.md.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src/soup_cli/utils/llama_cpp_capabilities.json"
ENTRY_NAME = "convert_hf_to_gguf.py"


def _kinds_from_model_maps(tree: ast.Module) -> dict[str, set[str]] | None:
    """Return the b12423 ``conversion/__init__.py`` model maps, or None.

    ``print_registered_models`` lists exactly the keys of these static dicts;
    dotted remote-code aliases are skipped because the probe listing regex
    only matches bare identifier names.
    """
    maps: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = {node.target.id}, node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        for bucket, map_name in (("TEXT", "TEXT_MODEL_MAP"), ("VISION", "MMPROJ_MODEL_MAP")):
            if map_name in targets:
                entries = ast.literal_eval(value)
                maps[bucket] = {name for name in entries
                                if isinstance(name, str) and name and "." not in name}
    if not maps:
        return None
    if set(maps) != {"TEXT", "VISION"}:
        raise ValueError("conversion/__init__.py must define TEXT_MODEL_MAP and MMPROJ_MODEL_MAP")
    return maps


def extract_capabilities(source: bytes) -> dict[str, list[str]]:
    """Resolve the same class inheritance/model_arch rule as ModelBase.register."""
    return extract_capabilities_from_sources({ENTRY_NAME: source})


def extract_capabilities_from_sources(sources: dict[str, bytes]) -> dict[str, list[str]]:
    """AST-parse every converter source into one shared class namespace.

    ``sources`` maps a label to file bytes; labels only order duplicate class
    definitions, where later modules win as at runtime. When a source defines
    ``TEXT_MODEL_MAP`` / ``MMPROJ_MODEL_MAP`` (the b12423 conversion package,
    whose printed listing is those maps' keys), they are authoritative over
    decorator registrations: mmproj-registered names still count as TEXT
    listings, and dotted remote-code aliases (``modeling_pkg.Class``) are
    skipped like the ``--print-supported-models`` listing does.
    """
    classes: dict[str, ast.ClassDef] = {}
    map_kinds: dict[str, set[str]] | None = None
    for name in sorted(sources):
        tree = ast.parse(sources[name])
        found = _kinds_from_model_maps(tree)
        if found is not None:
            if map_kinds is not None:
                raise ValueError("multiple sources define converter model maps")
            map_kinds = found
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
    if map_kinds is not None:
        if not map_kinds["TEXT"]:
            raise ValueError("converter TEXT_MODEL_MAP lists no architectures")
        return {kind: sorted(names) for kind, names in map_kinds.items()}

    def model_arch(name: str, visited: frozenset[str] = frozenset()) -> str | None:
        if name in visited:
            raise ValueError(f"cyclic converter class inheritance: {name}")
        node = classes[name]
        for statement in node.body:
            if (isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "model_arch"
                            for target in statement.targets)):
                if not isinstance(statement.value, ast.Attribute):
                    raise ValueError(f"unrecognized model_arch on {name}")
                return statement.value.attr
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                value = model_arch(base.id, visited | {name})
                if value is not None:
                    return value
        return None

    kinds: dict[str, set[str]] = {"TEXT": set(), "VISION": set()}
    for node in classes.values():
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "ModelBase"
                    and decorator.func.attr == "register"):
                arch = model_arch(node.name)
                if arch is None or not decorator.args or decorator.keywords:
                    raise ValueError(f"unrecognized registration on {node.name}")
                # b12423 marks mmproj-only models with MODEL_ARCH.MMPROJ
                # (conversion/base.py routes them to ModelType.MMPROJ); the
                # pre-refactor monolith used MODEL_ARCH.CLIP_VISION instead.
                kind = "VISION" if arch in ("CLIP_VISION", "MMPROJ") else "TEXT"
                for argument in decorator.args:
                    name = ast.literal_eval(argument)
                    if not isinstance(name, str) or not name:
                        raise ValueError(f"non-string architecture on {node.name}")
                    kinds[kind].add(name)
    if not kinds["TEXT"]:
        raise ValueError("converter has no recognized text registrations")
    return {kind: sorted(names) for kind, names in kinds.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, required=True,
                        help=f"the {ENTRY_NAME} entry script of the pinned commit")
    parser.add_argument("--package-dir", type=Path, default=None,
                        help="the conversion/ package directory beside the entry "
                             "script; omit for the pre-refactor single-file layout")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    export_tree = ast.parse((ROOT / "src/soup_cli/commands/export.py").read_text(encoding="utf-8"))
    pin = next(ast.literal_eval(node.value) for node in export_tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "LLAMA_CPP_TAG"
                       for target in node.targets))
    if arguments.tag != pin:
        parser.error(f"--tag must match LLAMA_CPP_TAG={pin}")
    if not re.fullmatch(r"[0-9a-f]{40}", arguments.commit):
        parser.error("--commit must be a full immutable Git commit SHA")
    source = arguments.converter.read_bytes()
    sources = {ENTRY_NAME: source}
    if arguments.package_dir is not None:
        for module in sorted(arguments.package_dir.glob("*.py")):
            sources[f"conversion/{module.name}"] = module.read_bytes()
        if len(sources) == 1:
            parser.error(f"--package-dir {arguments.package_dir} contains no *.py modules")
    # Record both the file hash and Git's blob identity, not a reformatted copy.
    def _provenance(data: bytes) -> dict[str, str]:
        blob = f"blob {len(data)}\0".encode() + data
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob": hashlib.sha1(blob).hexdigest(),
        }

    manifest = {
        "schema_version": 1,
        "tag": arguments.tag,
        "commit": arguments.commit,
        "source_url": (
            f"https://github.com/ggml-org/llama.cpp/blob/{arguments.commit}/{ENTRY_NAME}"
        ),
        "source_sha256": _provenance(source)["sha256"],
        "source_git_blob": _provenance(source)["git_blob"],
        "architectures": extract_capabilities_from_sources(sources),
        # Per-file provenance for EVERY file the data was read from.
        "sources": {relpath: _provenance(data) for relpath, data in sorted(sources.items())},
    }
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
