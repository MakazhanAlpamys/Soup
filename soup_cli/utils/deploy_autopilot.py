"""v0.46.0 Part A — On-Device Deploy Autopilot.

Closed allowlist of deploy-target profiles. Each profile maps a hardware /
runtime target (mac-m3, rtx-4090-24gb, iphone-16, ollama-local, ...) to a
PEFT + quantisation + speculative-decoding combo plus an output recipe and
a deploy shell script.

Live wiring into the v0.26.0 Quant-Lobotomy Checker (so the autopilot
actually *measures* OK/MINOR/MAJOR before picking a quant) is deferred to
v0.46.1; this release ships the schema + the canonical combo table + the
recipe-yaml / deploy-script writers so users have a reproducible artifact
the moment they run ``soup deploy autopilot --target mac-m3``.

The catalog is frozen at import time (``MappingProxyType``); name lookup is
case-insensitive over a strict kebab-case regex (matches v0.45.0 Part A
``register_plugin`` policy).
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from soup_cli.utils.paths import is_under_cwd

_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,31}$")
_MAX_BASE_LEN = 200
_MAX_OUTPUT_PATH_LEN = 4096

# Allowlist for the quant field — matches v0.38.0 / v0.40.5 Quant Menu names
# plus the lightweight ``none`` / ``4bit`` / ``8bit`` legacy values. We keep
# this small on purpose: a profile that picks an unknown quant is a bug, not
# a free-form string.
_ALLOWED_QUANT = frozenset(
    {"none", "4bit", "8bit", "gptq", "awq", "fp8", "mxfp4", "hqq:4bit", "hqq:8bit"}
)
_ALLOWED_PEFT = frozenset({"lora", "dora", "qlora", "full"})
_ALLOWED_RUNTIME = frozenset(
    {"transformers", "vllm", "sglang", "mlx", "ollama", "lm-studio", "executorch"}
)


@dataclass(frozen=True)
class DeployProfile:
    """One canonical deploy-target combo entry."""

    name: str
    description: str
    runtime: str
    quant: str
    peft: str
    spec_decoding: bool
    recommended_max_length: int
    notes: str


def _make(
    name: str,
    description: str,
    runtime: str,
    quant: str,
    peft: str,
    spec_decoding: bool,
    recommended_max_length: int,
    notes: str = "",
) -> DeployProfile:
    if not isinstance(name, str) or not _PROFILE_NAME_RE.match(name):
        raise ValueError(
            "profile name must be kebab-case ([a-z0-9][a-z0-9-]{0,31})"
        )
    if runtime not in _ALLOWED_RUNTIME:
        raise ValueError(f"runtime {runtime!r} not in allowlist")
    if quant not in _ALLOWED_QUANT:
        raise ValueError(f"quant {quant!r} not in allowlist")
    if peft not in _ALLOWED_PEFT:
        raise ValueError(f"peft {peft!r} not in allowlist")
    if not isinstance(spec_decoding, bool):
        raise TypeError("spec_decoding must be bool")
    if isinstance(recommended_max_length, bool) or not isinstance(
        recommended_max_length, int
    ):
        raise TypeError("recommended_max_length must be int (not bool)")
    if not (64 <= recommended_max_length <= 1_048_576):
        raise ValueError("recommended_max_length must be in [64, 1048576]")
    for label, txt in (("description", description), ("notes", notes)):
        if not isinstance(txt, str) or "\x00" in txt:
            raise ValueError(f"{label} must be NUL-free string")
        if len(txt) > 512:
            raise ValueError(f"{label} exceeds 512 chars")
    return DeployProfile(
        name=name,
        description=description,
        runtime=runtime,
        quant=quant,
        peft=peft,
        spec_decoding=spec_decoding,
        recommended_max_length=recommended_max_length,
        notes=notes,
    )


_BUILTIN: Mapping[str, DeployProfile] = MappingProxyType(
    {
        "mac-m3": _make(
            "mac-m3", "Apple Silicon M3 / M3 Pro / M3 Max — MLX inference",
            "mlx", "4bit", "lora", False, 8192,
            "MLX backend handles quantisation; LoRA stays as adapter",
        ),
        "mac-m4-pro": _make(
            "mac-m4-pro", "Apple Silicon M4 / M4 Pro — MLX inference",
            "mlx", "4bit", "lora", False, 16384,
            "Higher memory band; doubles default context window",
        ),
        "rtx-3060-12gb": _make(
            "rtx-3060-12gb", "Consumer NVIDIA 12GB — 4bit + speculative decoding",
            "transformers", "4bit", "qlora", True, 4096,
        ),
        "rtx-4090-24gb": _make(
            "rtx-4090-24gb", "Consumer NVIDIA 24GB — AWQ + vLLM",
            "vllm", "awq", "lora", True, 8192,
        ),
        "iphone-16": _make(
            "iphone-16", "iPhone 16 / 16 Pro — ExecuTorch on-device",
            "executorch", "4bit", "qlora", False, 2048,
            "Plan-only; ExecuTorch packaging lands in v0.54.0 Part D",
        ),
        "pixel-9": _make(
            "pixel-9", "Pixel 9 — ExecuTorch / AICore on-device",
            "executorch", "4bit", "qlora", False, 2048,
            "Plan-only; export pipeline lands in v0.54.0",
        ),
        "ollama-local": _make(
            "ollama-local", "Local Ollama / llama.cpp via GGUF",
            "ollama", "4bit", "lora", False, 4096,
        ),
        "lm-studio": _make(
            "lm-studio", "LM Studio desktop app — GGUF model",
            "lm-studio", "4bit", "lora", False, 4096,
        ),
        "runpod-a100": _make(
            "runpod-a100", "RunPod A100 40GB — bf16 vLLM with speculative",
            "vllm", "none", "lora", True, 32768,
        ),
        "hf-jobs-h100": _make(
            "hf-jobs-h100", "HF Jobs H100 80GB — FP8 + vLLM prefix cache",
            "vllm", "fp8", "lora", True, 65536,
        ),
    }
)


def list_profiles() -> Mapping[str, DeployProfile]:
    """Return an immutable view of the deploy profile registry."""
    return _BUILTIN


def get_profile(name: str) -> DeployProfile:
    """Return the profile for ``name`` (case-insensitive)."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    canonical = name.strip().lower()
    if not canonical or "\x00" in canonical:
        raise ValueError("name must be a non-empty NUL-free string")
    if canonical not in _BUILTIN:
        raise KeyError(canonical)
    return _BUILTIN[canonical]


def has_profile(name: str) -> bool:
    """True when ``name`` resolves to a known profile."""
    if not isinstance(name, str):
        return False
    return name.strip().lower() in _BUILTIN


def _validate_base(base: str) -> str:
    if not isinstance(base, str):
        raise TypeError("base must be a string")
    if not base or "\x00" in base or "\n" in base or "\r" in base:
        raise ValueError("base must be a non-empty single-line NUL-free string")
    if len(base) > _MAX_BASE_LEN:
        raise ValueError(f"base exceeds {_MAX_BASE_LEN} chars")
    return base


def render_recipe_yaml(profile: DeployProfile, base: str, output_dir: str) -> str:
    """Render a ready-to-train ``soup.yaml`` for ``profile`` + ``base``.

    Returns the YAML *text*; callers handle write-out with their own
    containment checks. The output_dir is validated as a non-empty
    NUL/newline-free string but is NOT path-realpath'd here so the recipe
    can be re-used from a different working dir later.
    """
    if not isinstance(profile, DeployProfile):
        raise TypeError("profile must be a DeployProfile")
    base = _validate_base(base)
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be a string")
    if (
        not output_dir
        or "\x00" in output_dir
        or "\n" in output_dir
        or "\r" in output_dir
    ):
        raise ValueError("output_dir must be a non-empty single-line NUL-free string")
    if len(output_dir) > _MAX_OUTPUT_PATH_LEN:
        raise ValueError(f"output_dir exceeds {_MAX_OUTPUT_PATH_LEN} chars")
    peft_section = ""
    if profile.peft in ("lora", "qlora", "dora"):
        peft_section = (
            "  lora:\n"
            "    r: 16\n"
            "    alpha: 32\n"
            "    dropout: 0.05\n"
        )
        if profile.peft == "dora":
            peft_section += "    use_dora: true\n"
    lines = [
        f"# Soup deploy autopilot — profile: {profile.name}",
        f"# {profile.description}",
        f"base: {base}",
        "task: sft",
        f"backend: {'mlx' if profile.runtime == 'mlx' else 'transformers'}",
        "data:",
        "  train: ./data.jsonl",
        "  format: auto",
        f"  max_length: {profile.recommended_max_length}",
        "training:",
        "  epochs: 3",
        "  lr: 2.0e-5",
        "  batch_size: auto",
        f"  quantization: {profile.quant}",
    ]
    if peft_section:
        lines.append(peft_section.rstrip("\n"))
    lines.append(f"output: {output_dir}")
    return "\n".join(lines) + "\n"


def render_deploy_script(profile: DeployProfile, model_path: str) -> str:
    """Render the shell script that takes the trained model to the target.

    The script is intentionally a stub: it prints the planned ``soup serve``
    or ``soup deploy`` command rather than executing it, so the user keeps
    the final approval gate. ``model_path`` is shell-escaped via
    ``shlex.quote`` so a crafted path cannot inject shell syntax.
    """
    if not isinstance(profile, DeployProfile):
        raise TypeError("profile must be a DeployProfile")
    if not isinstance(model_path, str):
        raise TypeError("model_path must be a string")
    if (
        not model_path
        or "\x00" in model_path
        or "\n" in model_path
        or "\r" in model_path
    ):
        raise ValueError("model_path must be non-empty single-line NUL-free string")
    if len(model_path) > _MAX_OUTPUT_PATH_LEN:
        raise ValueError(f"model_path exceeds {_MAX_OUTPUT_PATH_LEN} chars")
    quoted = shlex.quote(model_path)
    header = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# Soup deploy autopilot — profile: {profile.name}\n"
        f"# {profile.description}\n"
        f"MODEL={quoted}\n"
    )
    if profile.runtime == "ollama":
        body = (
            'NAME="soup-${USER:-local}-$(basename \"$MODEL\")"\n'
            'echo "Planned: soup deploy ollama --model $MODEL --name $NAME"\n'
        )
    elif profile.runtime == "lm-studio":
        body = 'echo "Planned: copy GGUF to ~/.cache/lm-studio/models/ then load in UI"\n'
    elif profile.runtime == "executorch":
        body = (
            'echo "Plan-only: ExecuTorch packaging not yet implemented (v0.54.0)"\n'
            'echo "Target: $MODEL"\n'
        )
    elif profile.runtime == "mlx":
        body = 'echo "Planned: soup serve --backend mlx --model $MODEL"\n'
    else:
        spec_flag = " --auto-spec" if profile.spec_decoding else ""
        body = (
            'echo "Planned: soup serve --backend '
            f'{profile.runtime} --model $MODEL{spec_flag}"\n'
        )
    return header + body


def _reject_symlink_target(path: str, label: str) -> None:
    """Reject if ``path`` already exists and is a symlink (TOCTOU defence).

    Matches v0.33.0 #22 / v0.43.0 Part C / v0.44.0 Part B policy.
    """
    import stat as _stat

    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(
            f"{label} must not be a symlink: {os.path.basename(path)}"
        )


def write_recipe(
    profile: DeployProfile,
    base: str,
    output_dir: str,
    recipe_path: str,
) -> str:
    """Write the recipe YAML under cwd; returns the realpath written."""
    if not isinstance(recipe_path, str):
        raise TypeError("recipe_path must be a string")
    if not recipe_path or "\x00" in recipe_path:
        raise ValueError("recipe_path must be non-empty NUL-free string")
    if len(recipe_path) > _MAX_OUTPUT_PATH_LEN:
        raise ValueError(f"recipe_path exceeds {_MAX_OUTPUT_PATH_LEN} chars")
    if not is_under_cwd(recipe_path):
        raise ValueError(
            f"recipe_path must stay under cwd: {os.path.basename(recipe_path)}"
        )
    _reject_symlink_target(recipe_path, "recipe_path")
    text = render_recipe_yaml(profile, base=base, output_dir=output_dir)
    real = os.path.realpath(recipe_path)
    os.makedirs(os.path.dirname(real) or ".", exist_ok=True)
    with open(real, "w", encoding="utf-8") as fh:
        fh.write(text)
    return real


def write_deploy_script(
    profile: DeployProfile, model_path: str, script_path: str
) -> str:
    """Write the deploy bash script under cwd; returns the realpath written."""
    if not isinstance(script_path, str):
        raise TypeError("script_path must be a string")
    if not script_path or "\x00" in script_path:
        raise ValueError("script_path must be non-empty NUL-free string")
    if len(script_path) > _MAX_OUTPUT_PATH_LEN:
        raise ValueError(f"script_path exceeds {_MAX_OUTPUT_PATH_LEN} chars")
    if not is_under_cwd(script_path):
        raise ValueError(
            f"script_path must stay under cwd: {os.path.basename(script_path)}"
        )
    _reject_symlink_target(script_path, "script_path")
    text = render_deploy_script(profile, model_path=model_path)
    real = os.path.realpath(script_path)
    os.makedirs(os.path.dirname(real) or ".", exist_ok=True)
    with open(real, "w", encoding="utf-8") as fh:
        fh.write(text)
    if os.name != "nt":
        try:
            os.chmod(real, 0o755)
        except OSError:
            pass
    return real


def autopilot_artifacts(
    profile_name: str,
    base: str,
    output_dir: str,
    model_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (recipe_yaml_text, deploy_script_text) for a profile.

    Helper for callers that want both artifacts in-memory without writing
    them out. ``model_path`` defaults to ``output_dir`` when omitted.
    """
    profile = get_profile(profile_name)
    recipe = render_recipe_yaml(profile, base=base, output_dir=output_dir)
    script = render_deploy_script(profile, model_path=model_path or output_dir)
    return recipe, script


__all__ = [
    "DeployProfile",
    "list_profiles",
    "get_profile",
    "has_profile",
    "render_recipe_yaml",
    "render_deploy_script",
    "write_recipe",
    "write_deploy_script",
    "autopilot_artifacts",
]
