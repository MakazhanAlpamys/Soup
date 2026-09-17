"""Modal.com cloud-training backend for ``soup train --cloud modal`` (#16).

Many users have no local GPU. `Modal.com <https://modal.com>`_ offers
serverless GPU training with per-second billing. ``--cloud modal`` renders a
self-contained Modal app from the user's ``soup.yaml`` (the config YAML is
base64-embedded — no code interpolation, no secrets) that:

1. builds an image with ``soup-cli[train]`` pinned to the running version,
2. writes the embedded config to ``/root/soup.yaml`` inside the container,
3. runs ``soup train --config /root/soup.yaml --yes`` on the chosen GPU, with
   its working directory on the named ``soup-outputs`` Modal volume
   (``/outputs/<run name>``), so the config's relative ``output`` survives the
   container; the volume is committed even when training fails,
4. downloads every file under that run directory into the local output
   directory when the run ends (also after a failed run) and prints the
   ``modal volume get`` command that retries the download.

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
import secrets
import types
from collections.abc import Callable, Mapping
from typing import Optional

from soup_cli.cloud._common import (
    _MAX_CONFIG_BYTES,
    _MAX_VERSION_LEN,
    _VERSION_RE,
    CloudPlan,
    validate_choice,
    write_cloud_stub,
)
from soup_cli.cloud._common import (
    validate_path_shape as _validate_path_shape,
)

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

# Named Modal volume the rendered app writes run outputs to. Each run gets its
# own top-level directory, so concurrent runs never overwrite each other.
MODAL_OUTPUT_VOLUME = "soup-outputs"
# Run names become a volume directory AND a repr()-embedded literal in the stub,
# so they are restricted to a closed alphabet.
_RUN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")

# Test / advanced-operator seam — replaces the live submit. Signature mirrors
# :func:`submit_modal_run` body -> returns an int exit code.
_MODAL_SUBMIT_OVERRIDE: Optional[Callable[["CloudPlan"], int]] = None


def validate_cloud(name: object) -> str:
    """Validate + normalise a ``--cloud`` provider name (closed allowlist)."""
    return validate_choice(name, "cloud", SUPPORTED_CLOUDS)


def validate_gpu(gpu: object) -> str:
    """Validate + normalise a ``--gpu`` type against the Modal allowlist."""
    return validate_choice(gpu, "gpu", SUPPORTED_GPUS)


def render_modal_stub(
    config_yaml: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    run_name: str | None = None,
) -> str:
    """Render the Modal app stub for ``config_yaml`` (v0.71.18 #16).

    ``config_yaml`` is base64-embedded as data (no interpolation). ``gpu``
    is validated + mapped to Modal's name. ``run_name`` names the run's
    directory on the ``soup-outputs`` volume (default: ``soup-<12 hex>``) and
    must match ``[a-z0-9][a-z0-9-]{0,62}``. Returns the stub source text.
    """
    if run_name is None:
        run_name = f"soup-{secrets.token_hex(6)}"
    if not isinstance(run_name, str) or not _RUN_NAME_RE.fullmatch(run_name):
        raise ValueError(
            f"run_name must match {_RUN_NAME_RE.pattern} (lower-case letters, "
            "digits and '-', at most 63 chars)"
        )
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

    # Built by concatenation so there is no triple-quote / brace escaping. The
    # embedded values are the base64 blob and repr()-embedded literals only
    # (output_dir, run_name from a closed alphabet, the volume constant).
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
        f"_LOCAL_OUTPUT = {output_dir!r}\n"
        f"_RUN_NAME = {run_name!r}\n"
        f"_VOLUME_NAME = {MODAL_OUTPUT_VOLUME!r}\n"
        '_REMOTE_ROOT = "/outputs"\n'
        '_CONFIG_PATH = "/root/soup.yaml"\n'
        "\n"
        'app = modal.App("soup-train")\n'
        "image = modal.Image.debian_slim().pip_install(\n"
        f"    {pip_spec!r}\n"
        ")\n"
        "outputs = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)\n"
        "\n"
        "\n"
        "@app.function(\n"
        f'    image=image, gpu="{modal_gpu}", timeout=86400,'
        " volumes={_REMOTE_ROOT: outputs}\n"
        ")\n"
        "def train() -> None:\n"
        "    # The config's relative `output` resolves against this directory,\n"
        "    # which lives on the volume and therefore outlives the container.\n"
        "    run_dir = pathlib.Path(_REMOTE_ROOT) / _RUN_NAME\n"
        "    run_dir.mkdir(parents=True, exist_ok=True)\n"
        '    cfg = base64.b64decode(_CONFIG_B64).decode("utf-8")\n'
        "    pathlib.Path(_CONFIG_PATH).write_text(cfg)\n"
        "    try:\n"
        "        subprocess.run(\n"
        '            ["soup", "train", "--config", _CONFIG_PATH, "--yes"],\n'
        "            check=True,\n"
        "            cwd=str(run_dir),\n"
        "        )\n"
        "    finally:\n"
        "        outputs.commit()\n"
        "\n"
        "\n"
        "def _is_file(entry) -> bool:\n"
        "    return entry.type == modal.volume.FileEntryType.FILE\n"
        "\n"
        "\n"
        "def _download(prefix: str, local_root: str) -> int:\n"
        "    root = pathlib.Path(local_root).resolve()\n"
        "    count = 0\n"
        "    for entry in outputs.listdir(prefix, recursive=True):\n"
        "        if not _is_file(entry):\n"
        "            continue\n"
        "        # Volume paths may or may not carry a leading '/'; the entry's\n"
        "        # path below the run directory is kept, '..' included, and the\n"
        "        # RESOLVED destination must stay inside the local output root.\n"
        '        rel = pathlib.PurePosixPath(entry.path.lstrip("/")).relative_to(\n'
        '            prefix.strip("/")\n'
        "        )\n"
        "        dest = root.joinpath(*rel.parts).resolve()\n"
        "        if root not in dest.parents:\n"
        '            raise RuntimeError(f"refusing to write outside {root}: {entry.path}")\n'
        "        dest.parent.mkdir(parents=True, exist_ok=True)\n"
        '        with dest.open("wb") as handle:\n'
        "            for chunk in outputs.read_file(entry.path):\n"
        "                handle.write(chunk)\n"
        "        count += 1\n"
        "    return count\n"
        "\n"
        "\n"
        "@app.local_entrypoint()\n"
        "def main() -> None:\n"
        "    try:\n"
        "        train.remote()\n"
        "    finally:\n"
        # Only already-repr()-embedded names are referenced, through runtime
        # f-strings in the GENERATED code — no user value is interpolated into
        # the stub source here. (Interpolating raw ``{output_dir}`` was a code-
        # injection hole; ``{output_dir!r}`` alone would still break for a path
        # containing a quote because the repr is nested inside a "..." literal.)
        '        count = _download(f"/{_RUN_NAME}", _LOCAL_OUTPUT)\n'
        "        print(\n"
        '            f"Downloaded {count} file(s) from volume "\n'
        '            f"{_VOLUME_NAME}/{_RUN_NAME} to {_LOCAL_OUTPUT}"\n'
        "        )\n"
        "        print(\n"
        '            "Retry the download with: "\n'
        '            f"modal volume get {_VOLUME_NAME} /{_RUN_NAME} {_LOCAL_OUTPUT} --force"\n'
        '            f" (files land under {_LOCAL_OUTPUT}/{_RUN_NAME})"\n'
        "        )\n"
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
    return write_cloud_stub(plan)


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
