"""Standalone controller template appended after validated plan constants.

REST contract: https://rest.runpod.io/v1/openapi.json
SSH contract: https://docs.runpod.io/pods/configuration/use-ssh
Only main() performs network operations. This module is also copied into plans.
"""

import http.client
import ipaddress
import json
import os
import pathlib
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

# Isolated Python ignores PYTHONIOENCODING/PYTHONUTF8. A redirected Windows
# stream may not encode a log character; print a replacement instead of failing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

_GPU_ID = globals().get("_GPU_ID", "")
_CONFIG_B64 = globals().get("_CONFIG_B64", "")
_OUTPUT = globals().get("_OUTPUT", "")
_SOUP_VERSION = globals().get("_SOUP_VERSION", "")
# ssh and scp are found and given their environment the way soup runs every
# external program: render_runpod_stub puts the source of find_tool and
# tool_environment (soup_cli/utils/tool_paths.py) ahead of this text.
_find_tool = globals().get("find_tool")
_environment_for_tools = globals().get("tool_environment")
_MISSING_TOOLS = "OpenSSH ssh and scp must be installed and on PATH before RunPod submission."

_API_BASE = "https://rest.runpod.io/v1"
_POD_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_MAX_RESPONSE = 1024 * 1024
_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
_BOOT = (
    "set -eu; if [ ! -x /usr/sbin/sshd ]; then apt-get update; "
    "apt-get install -y openssh-server; fi; "
    "mkdir -p /root/.ssh /run/sshd; chmod 700 /root/.ssh; "
    "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys; "
    "chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e"
)


def _notice(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


class _ControllerError(RuntimeError):
    """Only fixed, controller-owned diagnostics are safe to show to the user."""


class _APIError(_ControllerError):
    def __init__(self, method: str, status: int) -> None:
        super().__init__(f"RunPod API {method} failed: HTTP {status}")
        # A server/request timeout can still have allocated a Pod. Reconcile it.
        self.create_rejected = method == "POST" and 400 <= status < 500 and status != 408


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request_json(api_key, path, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(_API_BASE + path, data=data, method=method)
    request.add_header("Authorization", "Bearer " + api_key)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            raise _ControllerError("RunPod API response exceeds 1 MB")
        if not raw and method == "DELETE":
            return None
        parsed = json.loads(raw)
        if not isinstance(parsed, (dict, list)):
            raise _ControllerError("RunPod API returned an invalid JSON shape")
        return parsed
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and method in {"GET", "DELETE"} and path.startswith("/pods/"):
            return None
        raise _APIError(method, exc.code) from None
    except (OSError, ValueError, http.client.HTTPException):
        raise _ControllerError("RunPod API transport or JSON response failed") from None


def _launch_payload(public_key, name):
    return {
        "name": name,
        "imageName": _IMAGE,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "gpuTypeIds": [_GPU_ID],
        "gpuCount": 1,
        "containerDiskInGb": 80,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},
        "dockerEntrypoint": ["/bin/bash", "-lc"],
        "dockerStartCmd": [_BOOT],
    }


def _valid_id(value):
    if not isinstance(value, str) or not _POD_ID.fullmatch(value):
        raise _ControllerError("RunPod returned an invalid Pod ID; check the console")
    return value


def _wait_for_ssh(api_key, pod_id):
    path = "/pods/" + _valid_id(pod_id)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        record = _request_json(api_key, path)
        if not isinstance(record, dict) or record.get("desiredStatus") in {"EXITED", "TERMINATED"}:
            raise _ControllerError("RunPod is unavailable before SSH startup")
        ip = record.get("publicIp")
        port = (record.get("portMappings") or {}).get("22")
        if ip and isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
            address = ipaddress.ip_address(ip)
            # The API chooses the destination; refuse local/metadata addresses.
            if address.version != 4 or not address.is_global:
                raise _ControllerError("RunPod did not return a public SSH address")
            return str(address), port
        time.sleep(5)
    raise _ControllerError("RunPod did not expose SSH within 15 minutes")


def _tool(name):
    path = _find_tool(name)
    if path is None:
        raise _ControllerError(_MISSING_TOOLS)
    return path


def _tool_environment():
    """The environment for ssh and scp: the controller's, without the RunPod API key,
    which they do not use; on Windows the program search skips the current directory."""
    return _environment_for_tools(drop=("RUNPOD_API_KEY",))


def _train_and_copy(ip, port, private_key):
    with tempfile.TemporaryDirectory(prefix="soup-runpod-ssh-") as directory:
        known_hosts = pathlib.Path(directory) / "known_hosts"
        _run_ssh_session(ip, port, private_key, known_hosts)


def _run_ssh_session(ip: str, port: int, private_key: str, known_hosts: pathlib.Path) -> None:
    common = [
        "-i",
        private_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f'UserKnownHostsFile="{known_hosts.as_posix()}"',
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
    ]
    script = (
        "import base64, pathlib, subprocess, sys\n"
        "root = pathlib.Path('/workspace/soup'); root.mkdir(parents=True, exist_ok=True)\n"
        "import os; os.chdir(root)\n"
        f"pathlib.Path('soup.yaml').write_bytes(base64.b64decode({_CONFIG_B64!r}))\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', "
        f"'soup-cli[train]=={_SOUP_VERSION}'], "
        "check=True, stdout=sys.stderr)\n"
        "with open('train.log', 'w') as log:\n"
        "    result = subprocess.run(['soup', 'train', '--config', 'soup.yaml', '--yes'], "
        "stdout=log, stderr=subprocess.STDOUT)\n"
        "sys.exit(result.returncode)\n"
    )
    ssh, scp = _tool("ssh"), _tool("scp")
    env = _tool_environment()
    # A readiness mapping can precede sshd. Retry a harmless readiness command,
    # never the training command: reconnecting must not start training twice.
    deadline = time.monotonic() + 300
    while True:
        ready = subprocess.run(
            [ssh, *common, "-p", str(port), f"root@{ip}", "true"],
            capture_output=True, timeout=30, env=env,
        )
        if ready.returncode == 0:
            break
        if time.monotonic() >= deadline:
            raise _ControllerError("RunPod SSH did not become ready within five minutes")
        time.sleep(5)
    result = subprocess.run(
        [ssh, *common, "-p", str(port), f"root@{ip}", "python3 -"],
        input=script,
        text=True,
        capture_output=True,
        timeout=25 * 3600,
        env=env,
    )
    if result.returncode != 0:
        raise _ControllerError("RunPod remote training failed; no successful artifact copy claimed")
    target = _local_destination()
    target.parent.mkdir(parents=True, exist_ok=True)
    # The copy goes into a directory created for it, and only the one expected
    # entry is then moved to the output path.
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".soup-runpod-copy-", dir=str(target.parent)))
    try:
        copied = subprocess.run(
            [
                scp,
                *common,
                "-P",
                str(port),
                "-r",
                "--",
                f"root@{ip}:/workspace/soup/{_OUTPUT}",
                str(staging),
            ],
            capture_output=True,
            timeout=900,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise _ControllerError(_failed_copy("RunPod artifact copy timed out", staging)) from None
    if copied.returncode != 0:
        raise _ControllerError(_failed_copy("RunPod artifact copy failed", staging))
    _move_copied_output(staging, target)


def _failed_copy(reason, staging):
    """``reason`` for a copy that did not finish. The copy directory ``staging``
    (and nothing else) is removed when the copy left it empty; otherwise it is kept
    and the returned text names it."""
    try:
        os.rmdir(staging)
    except FileNotFoundError:
        pass
    except OSError:
        return f"{reason}; the copy directory is left in place: {staging}"
    return reason


def _move_copied_output(staging, target):
    """Move the output directory the copy wrote into ``staging`` to ``target``.

    ``staging`` must hold exactly that one entry, made of directories and regular
    files only; otherwise nothing is moved and ``staging`` is left for inspection.
    """
    if sorted(os.listdir(staging)) != [target.name]:
        raise _ControllerError(
            "RunPod artifact copy held unexpected entries; nothing was moved to the output path"
        )
    for directory, dirnames, filenames in os.walk(staging):
        for name in dirnames + filenames:
            info = os.lstat(os.path.join(directory, name))
            regular = stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
            if not regular or getattr(info, "st_file_attributes", 0) & 0x400:
                raise _ControllerError(
                    "RunPod artifact copy held a link or special file; nothing was moved "
                    "to the output path"
                )
    _local_destination()
    os.replace(staging / target.name, target)
    os.rmdir(staging)


def _local_destination():
    """Recheck the download path before allocation and immediately before SCP;
    return it."""
    root = os.path.realpath(os.getcwd())
    target = pathlib.Path(_OUTPUT)
    if (target.is_absolute() or ".." in target.parts or str(target) == "."
            or any(part.startswith("-") for part in target.parts)):
        raise _ControllerError("RunPod output must stay under cwd without option-like components")
    candidate = pathlib.Path.cwd()
    for part in target.parts:
        candidate = candidate / part
        if os.path.lexists(candidate):
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise _ControllerError("RunPod output must not traverse symlinks or junctions")
    if os.path.commonpath([root, os.path.realpath(target)]) != root:
        raise _ControllerError("RunPod output must stay under cwd")
    if os.path.lexists(target):
        raise _ControllerError("RunPod output already exists; choose a fresh output directory")
    return target


def _terminate(api_key, pod_id):
    path = "/pods/" + _valid_id(pod_id)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            _request_json(api_key, path, method="DELETE")
            # A successful DELETE alone is not evidence of termination.
            if _request_json(api_key, path) is None:
                return
        except RuntimeError:
            pass
        time.sleep(5)
    raise _ControllerError("RunPod deletion was not confirmed within five minutes")


def _recover_created(api_key, name):
    """A lost POST response must not cause another POST or delete unrelated Pods."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        records = _request_json(api_key, "/pods")
        if not isinstance(records, list):
            raise _ControllerError("RunPod could not reconcile the launch; check the console")
        found = [
            item["id"] for item in records if isinstance(item, dict) and item.get("name") == name
        ]
        if found:
            return [_valid_id(item) for item in found]
        time.sleep(5)
    raise _ControllerError("RunPod launch result is unknown; check the console")


def main():
    if not all((_GPU_ID, _CONFIG_B64, _OUTPUT, _SOUP_VERSION)):
        _notice("Generate a RunPod plan with soup train --cloud runpod first.")
        return 2
    if not all(_find_tool(command) for command in ("ssh", "scp")):
        _notice(_MISSING_TOOLS)
        return 2
    try:
        _local_destination()
    except (OSError, RuntimeError):
        _notice("RunPod needs a fresh local output path under cwd, without symlinks.")
        return 2
    api_key = os.environ.get("RUNPOD_API_KEY")
    private_key = os.path.expanduser(os.environ.get("RUNPOD_SSH_PRIVATE_KEY", ""))
    if not api_key or not private_key or not pathlib.Path(private_key).is_file():
        _notice("Set RUNPOD_API_KEY and RUNPOD_SSH_PRIVATE_KEY to a readable key file.")
        return 2
    try:
        public_key = pathlib.Path(private_key + ".pub").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"ssh-(?:ed25519|rsa) [A-Za-z0-9+/=]+(?: [^\r\n]+)?", public_key):
            raise ValueError("invalid public key")
    except (OSError, ValueError):
        _notice("A matching .pub file with one OpenSSH public key is required.")
        return 2
    name = "soup-" + uuid.uuid4().hex
    pod_id = None
    launched = False
    rc = 1
    try:
        launched = True
        response = _request_json(
            api_key, "/pods", method="POST", payload=_launch_payload(public_key, name)
        )
        pod_id = _valid_id(response.get("id") if isinstance(response, dict) else None)
        _notice("RunPod Pod launched: " + pod_id)
        ip, port = _wait_for_ssh(api_key, pod_id)
        _train_and_copy(ip, port, private_key)
        rc = 0
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, _APIError) and exc.create_rejected:
            launched = False
        _notice(str(exc) if isinstance(exc, _ControllerError) else (
            "RunPod training/controller failed; cleaning up the Pod."
        ))
    finally:
        # Repeated Ctrl+C must not abandon cleanup of a billable resource.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            ids = [pod_id] if pod_id else (_recover_created(api_key, name) if launched else [])
            for owned_id in ids:
                _terminate(api_key, owned_id)
            _notice("RunPod termination verified." if ids else (
                "RunPod create request rejected; no Pod to clean up."
            ))
        except Exception:
            _notice("CRITICAL: RunPod cleanup unconfirmed; check the console for " + name)
            rc = 1
        finally:
            signal.signal(signal.SIGINT, previous)
    return rc


if __name__ == "__main__":
    sys.exit(main())
