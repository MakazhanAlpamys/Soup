"""v0.44.0 Part D — `soup llama <subcommand>` proxy for llama.cpp binaries.

Validates the subcommand against a closed allowlist + builds the argv list
(no shell). Live subprocess invocation is owned by the CLI command in
`commands/llama.py`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from types import MappingProxyType
from typing import List, Mapping, Optional

# Closed allowlist — any subcommand outside this set is rejected.
_SUBCOMMAND_TO_BINARY: Mapping[str, str] = MappingProxyType(
    {
        "cli": "llama-cli",
        "mtmd-cli": "llama-mtmd-cli",
        "gguf-split": "llama-gguf-split",
        "server": "llama-server",
        "quantize": "llama-quantize",
    }
)

_MAX_ARGS = 64
_MAX_ARG_LEN = 1024


@dataclass(frozen=True)
class LlamaInvocation:
    """Resolved llama.cpp invocation."""

    subcommand: str
    binary: str
    binary_path: str
    args: List[str]


def known_subcommands() -> Mapping[str, str]:
    return _SUBCOMMAND_TO_BINARY


def _validate_arg(arg: str) -> str:
    if not isinstance(arg, str):
        raise TypeError("each llama arg must be str")
    if "\x00" in arg or "\n" in arg or "\r" in arg:
        raise ValueError("arg contains control character")
    if len(arg) > _MAX_ARG_LEN:
        raise ValueError(f"arg exceeds {_MAX_ARG_LEN} chars")
    return arg


def resolve(
    subcommand: str,
    args: Optional[List[str]] = None,
    *,
    binary_search_path: Optional[str] = None,
) -> LlamaInvocation:
    """Resolve `(subcommand, args)` into an executable plan.

    Raises:
      ValueError on unknown subcommand or invalid arg
      FileNotFoundError when the binary is not on PATH
    """
    if subcommand not in _SUBCOMMAND_TO_BINARY:
        raise ValueError(
            f"unknown llama subcommand {subcommand!r}; "
            f"expected one of {sorted(_SUBCOMMAND_TO_BINARY)}"
        )
    arg_list = list(args or [])
    if len(arg_list) > _MAX_ARGS:
        raise ValueError(f"too many args (>{_MAX_ARGS})")
    cleaned = [_validate_arg(arg) for arg in arg_list]
    binary = _SUBCOMMAND_TO_BINARY[subcommand]
    binary_path = shutil.which(binary, path=binary_search_path)
    if binary_path is None:
        raise FileNotFoundError(
            f"{binary} not found on PATH; install llama.cpp or set LLAMA_CPP_HOME"
        )
    return LlamaInvocation(
        subcommand=subcommand,
        binary=binary,
        binary_path=os.path.realpath(binary_path),
        args=cleaned,
    )


def build_argv(invocation: LlamaInvocation) -> List[str]:
    """Build the final argv list to pass to subprocess.run / Popen."""
    return [invocation.binary_path, *invocation.args]
