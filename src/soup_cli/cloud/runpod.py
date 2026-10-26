"""RunPod plans and credential-gated submission; no provider SDK is imported."""

from __future__ import annotations

import base64
import os
import re
import shlex
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from soup_cli.cloud._common import (
    _MAX_CONFIG_BYTES,
    _MAX_VERSION_LEN,
    _VERSION_RE,
    CloudPlan,
    validate_choice,
    validate_path_shape,
    write_cloud_stub,
)

SUPPORTED_CLOUDS = frozenset({"runpod"})
# REST v1 gpuTypeIds, verified against the provider's published OpenAPI schema.
_GPU_IDS = types.MappingProxyType(
    {
        "a100": "NVIDIA A100-SXM4-80GB",
        "h100": "NVIDIA H100 80GB HBM3",
        "a40": "NVIDIA A40",
        "rtx3090": "NVIDIA GeForce RTX 3090",
        "rtx4090": "NVIDIA GeForce RTX 4090",
    }
)
SUPPORTED_GPUS = frozenset(_GPU_IDS)
_RUNPOD_SUBMIT_OVERRIDE: Callable[[CloudPlan], int] | None = None
_CREDENTIAL_FIELDS = frozenset(
    {
        "api_key",
        "token",
        "access_token",
        "auth_token",
        "hf_token",
        "password",
        "private_key",
        "secret",
        "wandb_api_key",
        "runpod_api_key",
        "aws_secret_access_key",
    }
)


def validate_cloud(name: object) -> str:
    return validate_choice(name, "cloud", SUPPORTED_CLOUDS)


def validate_gpu(gpu: object) -> str:
    return validate_choice(gpu, "gpu", SUPPORTED_GPUS)


def _validate_config(config_yaml: str) -> None:
    import yaml

    from soup_cli.utils.yaml_limits import load_yaml_limited

    if not isinstance(config_yaml, str):
        raise ValueError("config must be a string")
    key = os.environ.get("RUNPOD_API_KEY")
    if key and key in config_yaml:
        raise ValueError("config must not embed provider credentials")
    try:
        content = load_yaml_limited(config_yaml, "RunPod config", max_bytes=_MAX_CONFIG_BYTES)
    except yaml.YAMLError:
        raise ValueError("config must be valid YAML; credentials must stay local") from None

    def walk(value: object, active: set[int]) -> None:
        if not isinstance(value, (dict, list)):
            return
        if id(value) in active:
            raise ValueError("recursive config aliases are not supported")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                for name, item in value.items():
                    if str(name).lower() in _CREDENTIAL_FIELDS and item not in (None, ""):
                        raise ValueError("inline config credentials are not supported")
                    walk(item, active)
            else:
                for item in value:
                    walk(item, active)
        finally:
            active.remove(id(value))

    walk(content, set())


def render_runpod_stub(
    config_yaml: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
) -> str:
    """Render a self-contained local controller; config is embedded only as data."""
    gpu = validate_gpu(gpu)
    validate_path_shape(output_dir, "output_dir")
    output = PurePosixPath(output_dir)
    if (
        output.is_absolute()
        or ".." in output.parts
        or any(part.startswith("-") for part in output.parts)
        or str(output) == "."
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", output_dir)
    ):
        raise ValueError("output_dir must be a simple relative path under cwd")
    if (
        not isinstance(soup_version, str)
        or len(soup_version) > _MAX_VERSION_LEN
        or not _VERSION_RE.fullmatch(soup_version)
    ):
        raise ValueError("soup_version must be a safe package version")
    _validate_config(config_yaml)
    encoded = base64.b64encode(config_yaml.encode("utf-8")).decode("ascii")
    constants = (
        f"_GPU_ID = {_GPU_IDS[gpu]!r}\n_CONFIG_B64 = {encoded!r}\n"
        f"_OUTPUT = {str(output)!r}\n_SOUP_VERSION = {soup_version!r}\n"
    )
    controller = Path(__file__).with_name("_runpod_controller.py").read_text(encoding="utf-8")
    return constants + controller


def plan_runpod_run(
    config_path: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    stub_path: str = "soup_runpod_app.py",
) -> CloudPlan:
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    for value, field in (
        (config_path, "--config"),
        (output_dir, "output_dir"),
        (stub_path, "stub_path"),
    ):
        validate_path_shape(value, field)
        enforce_under_cwd_and_no_symlink(value, field)
    with open(config_path, encoding="utf-8") as source:
        config_yaml = source.read(_MAX_CONFIG_BYTES + 1)
    if os.path.lexists(output_dir):
        raise ValueError("output_dir already exists; choose a fresh directory for retrieval")
    stub = render_runpod_stub(
        config_yaml,
        gpu=gpu,
        output_dir=output_dir,
        soup_version=soup_version,
    )
    return CloudPlan(
        "runpod", validate_gpu(gpu), output_dir, stub_path, stub,
        f"python -I {shlex.quote(stub_path)}",
    )


def write_stub(plan: CloudPlan) -> str:
    return write_cloud_stub(plan)


_CONTROLLER_ENV_NAMES: frozenset[str] = frozenset({
    "RUNPOD_API_KEY", "RUNPOD_SSH_PRIVATE_KEY", "PATH", "HOME", "USERPROFILE", "SYSTEMROOT",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "TEMP", "TMP", "TMPDIR", "SYSTEMDRIVE", "WINDIR", "SSH_AUTH_SOCK",
    # Windows program search: the extensions tried, and the setting that keeps
    # the current directory out of it.
    "PATHEXT", "NoDefaultCurrentDirectoryInExePath",
})


def _controller_environment(
    environ: Mapping[str, str], *, case_insensitive: bool | None = None,
) -> dict[str, str]:
    """Keep the controller's credentials and OS/network settings, with OS name rules."""
    if case_insensitive is None:
        case_insensitive = os.name == "nt"
    allowed = ({name.upper() for name in _CONTROLLER_ENV_NAMES}
               if case_insensitive else _CONTROLLER_ENV_NAMES)
    return {
        name: value for name, value in environ.items()
        if (name.upper() if case_insensitive else name) in allowed
    }


def submit_runpod_run(plan: CloudPlan, *, env: Mapping | None = None) -> int:
    """Keep waiting through interrupts so the controller can finish its cleanup."""
    if not isinstance(plan, CloudPlan) or plan.cloud != "runpod":
        raise TypeError("plan must be a RunPod CloudPlan")
    if _RUNPOD_SUBMIT_OVERRIDE is not None:
        return _RUNPOD_SUBMIT_OVERRIDE(plan)
    environ = dict(env if env is not None else os.environ)
    missing = [
        name for name in ("RUNPOD_API_KEY", "RUNPOD_SSH_PRIVATE_KEY") if not environ.get(name)
    ]
    if missing:
        raise RuntimeError("RunPod not configured. Set " + ", ".join(missing))
    environ = _controller_environment(environ)
    import subprocess

    proc = None
    while True:
        try:
            if proc is None:
                proc = subprocess.Popen([sys.executable, "-I", plan.stub_path], env=environ)
            return proc.wait()
        except BaseException as exc:
            if proc is None or not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                sys.stderr.write("Interrupted: waiting for RunPod cleanup to finish.\n")
                sys.stderr.flush()
            except BaseException:
                pass
