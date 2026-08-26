"""Lambda Cloud controller for ``soup train --cloud lambda`` (#264).

The rendered controller launches an instance with secret-free cloud-init
``user_data``, waits for training over SSH, copies the configured output back
to the caller, and terminates the instance from the caller in a ``finally``
block.  The Lambda API key never enters the instance or its boot logs.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Optional

from soup_cli.cloud._common import (
    _MAX_VERSION_LEN,
    _VERSION_RE,
    CloudPlan,
    validate_choice,
    write_cloud_stub,
)
from soup_cli.cloud._common import (
    validate_path_shape as _validate_path_shape,
)

SUPPORTED_CLOUDS: frozenset[str] = frozenset({"lambda"})

# Lambda instance type names from the public Cloud API catalogue.
_GPU_LAMBDA_NAME: Mapping[str, str] = types.MappingProxyType({
    "a10": "gpu_1x_a10",
    "a100": "gpu_1x_a100_sxm4",
    "h100": "gpu_1x_h100_pcie",
    "a6000": "gpu_1x_a6000",
})
SUPPORTED_GPUS: frozenset[str] = frozenset(_GPU_LAMBDA_NAME)

_REMOTE_OUTPUT_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# Lambda caps user_data at 1 MB. The YAML is base64-encoded inside a small
# shell script, so keep enough headroom for the 4/3 expansion and bootstrap.
_MAX_LAMBDA_CONFIG_BYTES = 700_000
_LAMBDA_SUBMIT_OVERRIDE: Optional[Callable[["CloudPlan"], int]] = None


def validate_cloud(name: object) -> str:
    """Validate and normalize the Lambda provider name."""
    return validate_choice(name, "cloud", SUPPORTED_CLOUDS)


def validate_gpu(gpu: object) -> str:
    """Validate and normalize a Lambda GPU type."""
    return validate_choice(gpu, "gpu", SUPPORTED_GPUS)


def _validate_remote_output(value: object) -> str:
    output = _validate_path_shape(value, "output_dir")
    path = PurePosixPath(output)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Lambda output_dir must be a relative path without '..'")
    if not _REMOTE_OUTPUT_RE.fullmatch(output):
        raise ValueError(
            "Lambda output_dir may contain only letters, digits, '.', '_', '-', and '/'"
        )
    return str(path)


def render_lambda_stub(
    config_yaml: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
) -> str:
    """Render a local Lambda lifecycle controller with secret-free user data."""
    if not isinstance(config_yaml, str):
        raise TypeError("config_yaml must be a string")
    encoded = config_yaml.encode("utf-8")
    if len(encoded) > _MAX_LAMBDA_CONFIG_BYTES:
        raise ValueError(
            f"config exceeds {_MAX_LAMBDA_CONFIG_BYTES} bytes "
            "(too large for Lambda user_data after base64 encoding)"
        )
    gpu_key = validate_gpu(gpu)
    lambda_gpu = _GPU_LAMBDA_NAME[gpu_key]
    remote_output = _validate_remote_output(output_dir)
    if not isinstance(soup_version, str) or "\x00" in soup_version:
        raise ValueError("soup_version must be a NUL-free string")
    if len(soup_version) > _MAX_VERSION_LEN or not _VERSION_RE.match(soup_version):
        raise ValueError(
            f"soup_version must match {_VERSION_RE.pattern} "
            f"and be <= {_MAX_VERSION_LEN} chars"
        )

    cfg_b64 = base64.b64encode(encoded).decode("ascii")
    pip_spec = f"soup-cli[train]=={soup_version}"
    user_data = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p /home/ubuntu/soup\n"
        "finish() {\n"
        "  rc=$?\n"
        "  printf '%s\\n' \"$rc\" > /home/ubuntu/soup.exit\n"
        "  chmod -R a+rX /home/ubuntu/soup /home/ubuntu/soup.exit\n"
        "  trap - EXIT\n"
        "  exit \"$rc\"\n"
        "}\n"
        "trap finish EXIT\n"
        "apt-get update\n"
        "apt-get install -y python3-pip\n"
        f"python3 -m pip install {pip_spec!r}\n"
        "cd /home/ubuntu/soup\n"
        f"printf '%s' {cfg_b64!r} | base64 -d > soup.yaml\n"
        "soup train --config soup.yaml --yes\n"
    )

    lines = [
        '"""Generated Lambda Cloud training controller; keep it running until completion."""',
        "import base64",
        "import json",
        "import os",
        "import pathlib",
        "import re",
        "import subprocess",
        "import sys",
        "import time",
        "import urllib.error",
        "import urllib.request",
        "",
        f"_INSTANCE_TYPE = {lambda_gpu!r}",
        f"_USER_DATA = {user_data!r}",
        f"_LOCAL_OUTPUT = {output_dir!r}",
        f"_REMOTE_OUTPUT = {remote_output!r}",
        '_API_BASE = "https://cloud.lambdalabs.com/api/v1"',
        '_REGION_RE = re.compile(r"^[a-z0-9-]{1,64}$")',
        "",
        "def _request_json(api_key, path, *, method='GET', payload=None):",
        "    data = None if payload is None else json.dumps(payload).encode('utf-8')",
        "    request = urllib.request.Request(_API_BASE + path, data=data, method=method)",
        "    auth = base64.b64encode(f'{api_key}:'.encode('ascii')).decode('ascii')",
        "    request.add_header('Authorization', f'Basic {auth}')",
        "    request.add_header('Content-Type', 'application/json')",
        "    try:",
        "        with urllib.request.urlopen(request, timeout=60) as response:",
        "            return json.loads(response.read().decode('utf-8'))",
        "    except urllib.error.HTTPError as exc:",
        "        detail = exc.read().decode('utf-8', errors='replace')",
        "        raise RuntimeError(",
        "            f'Lambda API {method} {path} failed with HTTP {exc.code}: {detail}'",
        "        ) from exc",
        "",
        "def _launch_payload(region, ssh_key_name):",
        "    return {",
        "        'region_name': region,",
        "        'instance_type_name': _INSTANCE_TYPE,",
        "        'ssh_key_names': [ssh_key_name],",
        "        'user_data': _USER_DATA,",
        "    }",
        "",
        "def _wait_for_ip(api_key, instance_id):",
        "    deadline = time.monotonic() + 900",
        "    while time.monotonic() < deadline:",
        "        record = _request_json(api_key, f'/instances/{instance_id}').get('data', {})",
        "        ip = record.get('ip')",
        "        if ip:",
        "            return str(ip)",
        "        if record.get('status') in {'terminated', 'unhealthy'}:",
        "            raise RuntimeError(f'Lambda instance entered {record.get(\"status\")} state')",
        "        time.sleep(5)",
        "    raise RuntimeError('Lambda instance did not expose an IP within 15 minutes')",
        "",
        "def _wait_for_termination(api_key, instance_id):",
        "    deadline = time.monotonic() + 300",
        "    while time.monotonic() < deadline:",
        "        try:",
        "            record = _request_json(",
        "                api_key, f'/instances/{instance_id}',",
        "            ).get('data', {})",
        "        except RuntimeError as exc:",
        "            if 'HTTP 404' in str(exc):",
        "                return",
        "            raise",
        "        if record.get('status') == 'terminated':",
        "            return",
        "        time.sleep(5)",
        "    raise RuntimeError('Lambda did not confirm termination within 5 minutes')",
        "",
        "def _train_and_copy(ip, private_key):",
        "    common = [",
        "        '-i', private_key, '-o', 'BatchMode=yes',",
        "        '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15',",
        "    ]",
        "    wait = subprocess.run(",
        "        ['ssh', *common, f'ubuntu@{ip}',",
        "         'cloud-init status --wait >/dev/null 2>&1; cat /home/ubuntu/soup.exit'],",
        "        check=False, capture_output=True, text=True, timeout=90000,",
        "    )",
        "    if wait.returncode != 0:",
        "        raise RuntimeError(f'remote training status failed: {wait.stderr.strip()}')",
        "    try:",
        "        training_rc = int(wait.stdout.strip().splitlines()[-1])",
        "    except (IndexError, ValueError) as exc:",
        "        raise RuntimeError('remote training did not publish an exit status') from exc",
        "    if training_rc != 0:",
        "        raise RuntimeError(f'remote soup train exited {training_rc}')",
        "    destination = pathlib.Path(_LOCAL_OUTPUT).parent",
        "    destination.mkdir(parents=True, exist_ok=True)",
        "    source = f'ubuntu@{ip}:/home/ubuntu/soup/{_REMOTE_OUTPUT}'",
        "    copied = subprocess.run(",
        "        ['scp', *common, '-r', source, str(destination)], check=False,",
        "    )",
        "    if copied.returncode != 0:",
        "        raise RuntimeError(f'checkpoint copy failed with exit {copied.returncode}')",
        "",
        "def main():",
        "    api_key = os.environ.get('LAMBDA_API_KEY')",
        "    ssh_key_name = os.environ.get('LAMBDA_SSH_KEY_NAME')",
        "    private_key = os.path.expanduser(os.environ.get('LAMBDA_SSH_PRIVATE_KEY', ''))",
        "    region = os.environ.get('LAMBDA_REGION', 'us-tx-1').lower()",
        "    if not api_key or not ssh_key_name or not private_key:",
        "        print('Set LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME, and LAMBDA_SSH_PRIVATE_KEY.')",
        "        return 2",
        "    if not _REGION_RE.fullmatch(region):",
        "        print('LAMBDA_REGION must contain only lowercase letters, digits, and hyphens.')",
        "        return 2",
        "    if not pathlib.Path(private_key).is_file():",
        "        print('LAMBDA_SSH_PRIVATE_KEY does not name a readable file.')",
        "        return 2",
        "    instance_id = None",
        "    rc = 1",
        "    try:",
        "        response = _request_json(",
        "            api_key, '/instance-operations/launch', method='POST',",
        "            payload=_launch_payload(region, ssh_key_name),",
        "        )",
        "        instance_ids = response.get('data', {}).get('instance_ids', [])",
        "        if len(instance_ids) != 1:",
        "            raise RuntimeError(f'Lambda launch returned {instance_ids!r}')",
        "        instance_id = str(instance_ids[0])",
        "        print(f'Lambda instance launched: {instance_id}')",
        "        ip = _wait_for_ip(api_key, instance_id)",
        "        _train_and_copy(ip, private_key)",
        "        print(f'Training completed; checkpoints copied to {_LOCAL_OUTPUT}')",
        "        rc = 0",
        "    except Exception as exc:",
        "        print(f'Lambda training failed: {exc}')",
        "    finally:",
        "        if instance_id is not None:",
        "            try:",
        "                _request_json(",
        "                    api_key, '/instance-operations/terminate', method='POST',",
        "                    payload={'instance_ids': [instance_id]},",
        "                )",
        "                _wait_for_termination(api_key, instance_id)",
        "                print(f'Lambda instance termination verified: {instance_id}')",
        "            except Exception as exc:",
        "                print(f'CRITICAL: Lambda termination failed for {instance_id}: {exc}')",
        "                rc = 1",
        "    return rc",
        "",
        "if __name__ == '__main__':",
        "    sys.exit(main())",
        "",
    ]
    return "\n".join(lines)


def plan_lambda_run(
    config_path: str,
    *,
    gpu: str,
    output_dir: str,
    soup_version: str,
    stub_path: str = "soup_lambda_app.py",
) -> CloudPlan:
    """Build a Lambda controller plan from a cwd-contained config."""
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(config_path, "--config")
    with open(config_path, encoding="utf-8") as fh:
        config_yaml = fh.read(_MAX_LAMBDA_CONFIG_BYTES + 1)
    if len(config_yaml.encode("utf-8")) > _MAX_LAMBDA_CONFIG_BYTES:
        raise ValueError(f"config exceeds {_MAX_LAMBDA_CONFIG_BYTES} bytes")
    gpu_key = validate_gpu(gpu)
    _validate_remote_output(output_dir)
    _validate_path_shape(stub_path, "stub_path")
    stub_text = render_lambda_stub(
        config_yaml,
        gpu=gpu_key,
        output_dir=output_dir,
        soup_version=soup_version,
    )
    return CloudPlan(
        cloud="lambda",
        gpu=gpu_key,
        output_dir=output_dir,
        stub_path=stub_path,
        stub_text=stub_text,
        run_command=f"python {stub_path}",
    )


def write_stub(plan: CloudPlan) -> str:
    """Write a Lambda controller atomically under the current directory."""
    return write_cloud_stub(plan)


def submit_lambda_run(plan: CloudPlan, *, env: Optional[Mapping] = None) -> int:
    """Run the local lifecycle controller after validating its credentials."""
    if not isinstance(plan, CloudPlan):
        raise TypeError(f"plan must be a CloudPlan, got {type(plan).__name__}")
    if _LAMBDA_SUBMIT_OVERRIDE is not None:
        return _LAMBDA_SUBMIT_OVERRIDE(plan)
    environ = env if env is not None else os.environ
    missing = [
        name
        for name in (
            "LAMBDA_API_KEY",
            "LAMBDA_SSH_KEY_NAME",
            "LAMBDA_SSH_PRIVATE_KEY",
        )
        if not environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Lambda Cloud not authenticated/configured. Set "
            + ", ".join(missing)
            + ", then re-run with --cloud-submit."
        )
    import subprocess

    proc = subprocess.run(  # noqa: S603 — argv list, no shell
        [sys.executable, plan.stub_path],
        check=False,
        env=dict(environ),
    )
    return proc.returncode
