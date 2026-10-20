"""Generate offline capabilities from the exact pinned llama.cpp converter AST.

Supply the unmodified convert_hf_to_gguf.py from the recorded commit. This script
never imports the converter or fetches source at runtime. See docs/serving-and-export.md.
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


def extract_capabilities(source: bytes) -> dict[str, list[str]]:
    """Resolve the same class inheritance/model_arch rule as ModelBase.register."""
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

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
                kind = "VISION" if arch == "CLIP_VISION" else "TEXT"
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
    parser.add_argument("--converter", type=Path, required=True)
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
    # Record both the file hash and Git's blob identity, not a reformatted copy.
    blob = f"blob {len(source)}\0".encode() + source
    manifest = {
        "schema_version": 1,
        "tag": arguments.tag,
        "commit": arguments.commit,
        "source_url": (
            f"https://github.com/ggml-org/llama.cpp/blob/{arguments.commit}/convert_hf_to_gguf.py"
        ),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_git_blob": hashlib.sha1(blob).hexdigest(),
        "architectures": extract_capabilities(source),
    }
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
