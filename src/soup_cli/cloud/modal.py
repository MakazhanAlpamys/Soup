"""Modal.com cloud-training backend for ``soup train --cloud modal`` (#16).

Many users have no local GPU. `Modal.com <https://modal.com>`_ offers
serverless GPU training with per-second billing. ``--cloud modal`` renders a
self-contained Modal app from the user's ``soup.yaml`` (the config YAML is
base64-embedded — no code interpolation, no secrets) that:

1. builds an image with ``soup-cli[train]`` pinned to the running version,
2. writes the embedded config to ``/root/soup.yaml`` inside the container,
3. runs ``soup train --config /root/soup.yaml --yes`` on the chosen GPU.

Default behaviour is **plan-only**: write the stub + print the planned
``modal run`` command (matching the ``soup quantize`` / ``soup agent train``
"print the command" design). ``--cloud-submit`` attempts a live submit,
gated on a Modal token (``modal setup`` or ``MODAL_TOKEN_ID`` /
``MODAL_TOKEN_SECRET``); a mockable seam (``_MODAL_SUBMIT_OVERRIDE``) keeps
the submit path testable without an account.

Security:
- The config YAML is base64-embedded as DATA; the rendered stub never evals
  user strings. ``gpu`` / ``output_dir`` are validated against closed
  allowlists / shape rules before embedding (no YAML / arg injection).
- Never embeds API keys. Modal auth is via Modal's own ``modal setup``; HF /
  WANDB tokens flow through the container env, not the rendered stub.
- ``--config`` is cwd-contained + symlink-rejected. Stub written atomically
  under cwd.
- No top-level ``import modal`` — lazy import inside ``submit_modal_run``.
"""

from __future__ import annotations

import base64
import os
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

_MAX_NAME_LEN = 32
_MAX_PATH_LEN = 4096
_MAX_VERSION_LEN = 64
_MAX_CONFIG_BYTES = 1_000_000  # 1 MiB cap on the embedded soup.yaml
# PEP 440-ish version shape — defence-in-depth so a crafted soup_version can't
# break out of the embedded ``pip_install("soup-cli[train]==<ver>")`` string.
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")

SUPPORTED_CLOUDS: frozenset[str] = frozenset({"modal"})

# Modal GPU types (https://modal.com/docs/reference/modal.gpu). Canonical
# lower-case keys; the stub emits Modal's expected upper-case name.
_GPU_MODAL_NAME: Mapping[str, str] = types.MappingProxyType({
    "t4": "T4",
    "l4": "L4",
    "a10g": "A10G",
    "a100": "A100",
    "a100-80gb": "A100-80GB",
    "l40s": "L40S",
    "h100": "H100",
})
SUPPORTED_GPUS: frozenset[str] = frozenset(_GPU_MODAL_NAME)

# Test / advanced-operator seam — replaces the live submit. Signature mirrors
# :func:`submit_modal_run` body -> returns an int exit code.
_MODAL_SUBMIT_OVERRIDE: Optional[Callable[["CloudPlan"], int]] = None


def validate_cloud(name: object) -> str:
    """Validate + normalise a ``--cloud`` provider name (closed allowlist)."""
    if isinstance(name, bool):
        raise ValueError("cloud must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(f"cloud must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("cloud must be a non-empty string")
    if "\x00" in name:
        raise ValueError("cloud must not contain null bytes")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"cloud exceeds {_MAX_NAME_LEN} chars")
    normalised = name.lower()
    if normalised not in SUPPORTED_CLOUDS:
        raise ValueError(
            f"cloud={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_CLOUDS)}"
        )
    return normalised


def validate_gpu(gpu: object) -> str:
    """Validate + normalise a ``--gpu`` type against the Modal allowlist."""
    if isinstance(gpu, bool):
        raise ValueError("gpu must be a string, got bool")
    if not isinstance(gpu, str):
        raise ValueError(f"gpu must be a string, got {type(gpu).__name__}")
    if not gpu:
        raise ValueError("gpu must be a non-empty string")
    if "\x00" in gpu:
        raise ValueError("gpu must not contain null bytes")
    if len(gpu) > _MAX_NAME_LEN:
        raise ValueError(f"gpu exceeds {_MAX_NAME_LEN} chars")
    normalised = gpu.lower()
    if normalised not in SUPPORTED_GPUS:
        raise ValueError(
            f"gpu={gpu!r} is not supported. Valid: {sorted(SUPPORTED_GPUS)}"
        )
    return normalised


def _validate_path_shape(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must not contain NUL / newline")
    if len(value) > _MAX_PATH_LEN:
        raise ValueError(f"{field} exceeds {_MAX_PATH_LEN} chars")
    return value


@dataclass(frozen=True)
class CloudPlan:
    """A rendered cloud-training plan (plan-only by default)."""

    cloud: str
    gpu: str
    output_dir: str
    stub_path: str
    stub_text: str
    run_command: str


def render_modal_stub(
    config_yaml: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
) -> str:
    """Render the Modal app stub for ``config_yaml`` (v0.71.18 #16).

    ``config_yaml`` is base64-embedded as data (no interpolation). ``gpu``
    is validated + mapped to Modal's name. Returns the stub source text.
    """
    if not isinstance(config_yaml, str):
        raise TypeError("config_yaml must be a string")
    encoded = config_yaml.encode("utf-8")
    if len(encoded) > _MAX_CONFIG_BYTES:
        raise ValueError(
            f"config exceeds {_MAX_CONFIG_BYTES} bytes "
            "(too large to embed in the Modal stub)"
        )
    gpu_key = validate_gpu(gpu)
    modal_gpu = _GPU_MODAL_NAME[gpu_key]
    _validate_path_shape(output_dir, "output_dir")
    if not isinstance(soup_version, str) or "\x00" in soup_version:
        raise ValueError("soup_version must be a NUL-free string")
    if len(soup_version) > _MAX_VERSION_LEN or not _VERSION_RE.match(soup_version):
        raise ValueError(
            f"soup_version must match {_VERSION_RE.pattern} "
            f"and be <= {_MAX_VERSION_LEN} chars"
        )
    cfg_b64 = base64.b64encode(encoded).decode("ascii")
    # repr()-embed so a stray quote / backslash cannot break out of the literal
    # (defence-in-depth on top of the _VERSION_RE allowlist above).
    pip_spec = f"soup-cli[train]=={soup_version}"

    # Built by concatenation so there is no triple-quote / brace escaping; the
    # only embedded user-derived value is the base64 blob (injection-free).
    return (
        '"""Auto-generated by `soup train --cloud modal` (v0.71.18 #16).\n'
        "Run with: modal run soup_modal_app.py\n"
        '(authenticate once with `modal setup`).\n"""\n'
        "import base64\n"
        "import pathlib\n"
        "import subprocess\n"
        "\n"
        "import modal\n"
        "\n"
        f'_CONFIG_B64 = "{cfg_b64}"\n'
        f'_LOCAL_OUTPUT = {output_dir!r}\n'
        "\n"
        'app = modal.App("soup-train")\n'
        "image = modal.Image.debian_slim().pip_install(\n"
        f"    {pip_spec!r}\n"
        ")\n"
        "\n"
        f'@app.function(image=image, gpu="{modal_gpu}", timeout=86400)\n'
        "def train() -> None:\n"
        '    cfg = base64.b64decode(_CONFIG_B64).decode("utf-8")\n'
        '    pathlib.Path("/root/soup.yaml").write_text(cfg)\n'
        "    subprocess.run(\n"
        '        ["soup", "train", "--config", "/root/soup.yaml", "--yes"],\n'
        "        check=True,\n"
        "    )\n"
        "\n"
        "\n"
        "@app.local_entrypoint()\n"
        "def main() -> None:\n"
        "    train.remote()\n"
        # Reference the already-repr()-embedded ``_LOCAL_OUTPUT`` via a runtime
        # f-string in the GENERATED code — no user value is interpolated into
        # the stub source here. (Interpolating raw ``{output_dir}`` was a code-
        # injection hole; ``{output_dir!r}`` alone would still break for a path
        # containing a quote because the repr is nested inside a "..." literal.)
        '    print(f"Training submitted; download checkpoints to {_LOCAL_OUTPUT}")\n'
    )


def plan_modal_run(
    config_path: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    stub_path: str = "soup_modal_app.py",
) -> CloudPlan:
    """Build a :class:`CloudPlan` from a cwd-contained ``soup.yaml``.

    Reads the config (cwd-containment + symlink rejection), renders the
    Modal stub, and returns the plan (stub text + planned ``modal run``
    command). Does NOT write the stub to disk — the caller decides.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(config_path, "--config")
    with open(config_path, encoding="utf-8") as fh:
        config_yaml = fh.read(_MAX_CONFIG_BYTES + 1)
    if len(config_yaml.encode("utf-8")) > _MAX_CONFIG_BYTES:
        raise ValueError(f"config exceeds {_MAX_CONFIG_BYTES} bytes")
    gpu_key = validate_gpu(gpu)
    _validate_path_shape(output_dir, "output_dir")
    _validate_path_shape(stub_path, "stub_path")
    stub_text = render_modal_stub(
        config_yaml,
        gpu=gpu_key,
        output_dir=output_dir,
        soup_version=soup_version,
    )
    run_command = f"modal run {stub_path}"
    return CloudPlan(
        cloud="modal",
        gpu=gpu_key,
        output_dir=output_dir,
        stub_path=stub_path,
        stub_text=stub_text,
        run_command=run_command,
    )


def write_stub(plan: CloudPlan) -> str:
    """Write the plan's stub atomically under cwd; return the realpath."""
    from soup_cli.utils.paths import atomic_write_text

    return atomic_write_text(plan.stub_text, plan.stub_path, field="stub_path")


def submit_modal_run(plan: CloudPlan, *, env: Optional[Mapping] = None) -> int:
    """Live-submit the Modal run (gated on a Modal token; mockable).

    Returns the ``modal run`` subprocess exit code. Raises ``RuntimeError``
    with a friendly message when the Modal token is missing or the Modal SDK
    is not installed. A ``_MODAL_SUBMIT_OVERRIDE`` seam replaces the live
    path for tests.
    """
    if not isinstance(plan, CloudPlan):
        raise TypeError(f"plan must be a CloudPlan, got {type(plan).__name__}")
    if _MODAL_SUBMIT_OVERRIDE is not None:
        return _MODAL_SUBMIT_OVERRIDE(plan)
    environ = env if env is not None else os.environ
    has_token = bool(
        environ.get("MODAL_TOKEN_ID") and environ.get("MODAL_TOKEN_SECRET")
    )
    has_config = os.path.exists(os.path.expanduser("~/.modal.toml"))
    if not (has_token or has_config):
        raise RuntimeError(
            "Modal not authenticated. Run `modal setup`, or set "
            "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET, then re-run with "
            "--cloud-submit."
        )
    try:
        import modal  # noqa: F401 — presence check only
    except ImportError as exc:
        raise RuntimeError(
            "Modal SDK not installed. Run `pip install \"soup-cli[modal]\"`."
        ) from exc
    import subprocess

    proc = subprocess.run(  # noqa: S603 — argv list, no shell
        ["modal", "run", plan.stub_path],
        check=False,
    )
    return proc.returncode
