"""Standalone controller template appended after validated plan constants.

REST contract (API v2): https://docs.runpod.io/api-reference-v2/openapi.json
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
import urllib.parse
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

_API_BASE = "https://api.runpod.io/v2"
_POD_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_MAX_RESPONSE = 1024 * 1024
# A page of the Pod list: v2 pages by cursor, newest first, and a full-size page can
# approach the response cap above, so ask for a small one and bound the walk.
_LIST_PAGE = 100
_MAX_LIST_PAGES = 20
# RFC 9457 error bodies: how much is read, and how much of it is ever shown.
_PROBLEM_TYPE = "application/problem+json"
_MAX_ERROR_BODY = 64 * 1024
_PROBLEM_CHARS = 400
_LOG_TAIL_LINES = 50
_LOG_TAIL_BYTES = 50_000
# Bound scheme matching so long non-URL log lines do not cause quadratic work.
_URL_CREDENTIALS = re.compile(r"(\b[a-z][a-z0-9+.-]{0,31}://)[^/\s?#]*@", re.IGNORECASE)
# The same token shapes as utils/crash.py, without importing the installed CLI.
_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"api[_-]?key[\"' :=]+[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}"),
)
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
    def __init__(self, method: str, status: int, detail: str = "") -> None:
        text = f"RunPod API {method} failed: HTTP {status}"
        super().__init__(f"{text} ({detail})" if detail else text)
        # A server/request timeout can still have allocated a Pod. Reconcile it.
        self.create_rejected = method == "POST" and 400 <= status < 500 and status != 408


# An answer that says "this route is not served": the Pod collection is not found, the
# route is gone, or the request is redirected (redirects are never followed).
_NOT_SERVED_STATUSES = frozenset({301, 302, 303, 307, 308, 404, 410})
_RETIRED_NOTICE = (
    "RunPod answered HTTP {status} where the REST API v2 (https://api.runpod.io/v2) that "
    "this soup-cli version uses for `soup train --cloud runpod --cloud-submit` should "
    "answer: RunPod has probably retired or moved that API. Upgrade with "
    "`pip install -U soup-cli`, or start the run yourself on a Pod from the RunPod "
    "console."
)


class _ApiRetiredError(_APIError):
    """RunPod says the route this controller calls is not served. Nothing was created."""

    def __init__(self, method: str, status: int) -> None:
        _ControllerError.__init__(self, _RETIRED_NOTICE.format(status=status))
        self.create_rejected = method == "POST"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _clean_text(text):
    """``text`` without terminal controls and with credentials redacted.

    render_runpod_stub serializes the canonical utils/terminal.py table. Controls are
    stripped before matching credentials, and redaction comes before any truncation:
    cutting a password first can remove the URL prefix that identifies it.
    """
    safe = text.translate(globals()["_TERMINAL_STRIP_TABLE"])
    safe = _URL_CREDENTIALS.sub(r"\1<redacted>@", safe)
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("<redacted>", safe)
    return safe


def _problem_detail(exc, api_key):
    """The title, detail and first three ``errors`` of an RFC 9457 error body, cleaned for
    the terminal, or "" when the response is not an application/problem+json object.

    Nothing else RunPod sends is ever shown: the key is removed, controls are stripped,
    and the text is capped, because it is provider-controlled and reaches a terminal.
    """
    try:
        if not str(exc.headers.get("Content-Type", "")).lower().startswith(_PROBLEM_TYPE):
            return ""
        body = json.loads(exc.read(_MAX_ERROR_BODY))
        errors = body.get("errors")
        parts = [body.get("title"), body.get("detail")]
        parts += errors[:3] if isinstance(errors, list) else []
    except Exception:
        return ""
    shown = []
    for part in parts:
        if isinstance(part, str):
            part = part.replace(api_key, "<redacted>") if api_key else part
            part = " ".join(_clean_text(part).split())
            if part:
                shown.append(part)
    return "; ".join(shown)[:_PROBLEM_CHARS]


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
        if exc.code in _NOT_SERVED_STATUSES:
            raise _ApiRetiredError(method, exc.code) from None
        raise _APIError(method, exc.code, _problem_detail(exc, api_key)) from None
    except (OSError, ValueError, http.client.HTTPException):
        raise _ControllerError("RunPod API transport or JSON response failed") from None


def _launch_payload(public_key, name):
    # The v2 Create Pod body: exactly one of gpu/cpu, nested; unknown fields are a 422.
    # v2 has no default ports (22/tcp stays explicit) and attaches no volume unless asked.
    return {
        "name": name,
        "image": _IMAGE,
        "cloud": "SECURE",
        "gpu": {"id": _GPU_ID, "count": 1},
        "disk": 80,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": public_key},
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": [_BOOT],
    }


def _valid_id(value):
    if not isinstance(value, str) or not _POD_ID.fullmatch(value):
        raise _ControllerError("RunPod returned an invalid Pod ID; check the console")
    return value


def _ssh_endpoints(record):
    """The (host, port) pairs a Pod record publishes for its SSH port: ``ssh.direct``
    first, then the ``runtime.ports`` entry for 22/tcp."""
    ssh = record.get("ssh")
    direct = ssh.get("direct") if isinstance(ssh, dict) else None
    if isinstance(direct, dict):
        yield direct.get("host"), direct.get("port")
    runtime = record.get("runtime")
    ports = runtime.get("ports") if isinstance(runtime, dict) else None
    for entry in ports if isinstance(ports, list) else ():
        if isinstance(entry, dict) and entry.get("private") == 22 and entry.get("type") == "tcp":
            yield entry.get("ip"), entry.get("public")


def _ssh_address(record):
    """The public (ip, port) of the Pod's SSH port, or None while none is published."""
    unusable_host = False
    for host, port in _ssh_endpoints(record):
        if host is None or isinstance(port, bool) or not isinstance(port, int):
            continue
        if not 1 <= port <= 65535:
            continue
        try:
            address = ipaddress.ip_address(host) if isinstance(host, str) else None
        except ValueError:
            address = None
        if address is None:
            # Only an IP literal is ever connected to; a name would need a lookup.
            unusable_host = True
            continue
        # The API chooses the destination; refuse local/metadata addresses.
        if address.version != 4 or not address.is_global:
            raise _ControllerError("RunPod did not return a public SSH address")
        return str(address), port
    if unusable_host:
        raise _ControllerError("RunPod did not return a public SSH address")
    return None


def _wait_for_ssh(api_key, pod_id):
    path = "/pods/" + _valid_id(pod_id)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        record = _request_json(api_key, path)
        if not isinstance(record, dict):
            raise _ControllerError("RunPod is unavailable before SSH startup")
        status = record.get("status")
        if status == "ERROR":
            raise _ControllerError("RunPod reported the Pod in ERROR state before SSH startup")
        if status in {"EXITED", "TERMINATED"}:
            raise _ControllerError("RunPod is unavailable before SSH startup")
        if status in {"STARTING", "RUNNING"}:
            address = _ssh_address(record)
            if address:
                return address
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


def _print_remote_stderr_tail(output: str | None) -> None:
    """Print a sanitized, redacted failure tail as literal stderr."""
    if not output:
        return
    safe = _clean_text(output)
    encoded = safe.encode("utf-8", errors="replace")[-_LOG_TAIL_BYTES:]
    lines = encoded.decode("utf-8", errors="ignore").splitlines()[-_LOG_TAIL_LINES:]
    tail = "\n".join(lines)
    if tail:
        _notice(f"RunPod remote stderr tail (last {_LOG_TAIL_LINES} lines):\n{tail}")


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
        "subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', "
        "'/workspace/venv'], check=True, stdout=sys.stderr)\n"
        "subprocess.run(['/workspace/venv/bin/python', '-m', 'pip', 'install', "
        f"'soup-cli[train]=={_SOUP_VERSION}'], "
        "check=True, stdout=sys.stderr)\n"
        "with open('train.log', 'w') as log:\n"
        "    result = subprocess.run(['/workspace/venv/bin/soup', 'train', "
        "'--config', 'soup.yaml', '--yes'], "
        "stdout=log, stderr=subprocess.STDOUT)\n"
        "if result.returncode:\n"
        "    with open('train.log', 'rb') as log:\n"
        "        log.seek(0, os.SEEK_END)\n"
        f"        log.seek(max(0, log.tell() - {_LOG_TAIL_BYTES + 1}))\n"
        f"        tail = log.read({_LOG_TAIL_BYTES + 1})\n"
        # Do not forward a partial first line with its credential prefix missing.
        f"        if len(tail) > {_LOG_TAIL_BYTES}:\n"
        "            tail = tail.partition(b'\\n')[2]\n"
        f"        tail = b'\\n'.join(tail.splitlines()[-{_LOG_TAIL_LINES}:])\n"
        "    sys.stderr.buffer.write(tail)\n"
        "sys.exit(result.returncode)\n"
    )
    ssh, scp = _tool("ssh"), _tool("scp")
    env = _tool_environment()
    # A readiness mapping can precede sshd. Retry a harmless readiness command,
    # never the training command: reconnecting must not start training twice.
    deadline = time.monotonic() + 300
    while True:
        try:
            ready = subprocess.run(
                [ssh, *common, "-p", str(port), f"root@{ip}", "true"],
                capture_output=True,
                timeout=30,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise _ControllerError(
                "RunPod SSH readiness probe timed out after 30 seconds"
            ) from None
        if ready.returncode == 0:
            break
        if time.monotonic() >= deadline:
            raise _ControllerError("RunPod SSH did not become ready within five minutes")
        time.sleep(5)
    try:
        result = subprocess.run(
            [ssh, *common, "-p", str(port), f"root@{ip}", "python3 -"],
            input=script,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=25 * 3600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise _ControllerError("RunPod remote training command timed out after 25 hours") from None
    if result.returncode != 0:
        _print_remote_stderr_tail(result.stderr)
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
            record = _request_json(api_key, path)
            # v2: a deleted Pod answers 404, or is still readable in status TERMINATED.
            if record is None or (
                isinstance(record, dict) and record.get("status") == "TERMINATED"
            ):
                return
        except RuntimeError:
            pass
        time.sleep(5)
    raise _ControllerError("RunPod deletion was not confirmed within five minutes")


def _pod_pages(api_key):
    """The pages of the account's Pod list, newest first, following the cursor.

    The walk is bounded, and any page that is not the documented v2 shape is refused.
    """
    refused = "RunPod could not reconcile the launch; check the console"
    path = f"/pods?limit={_LIST_PAGE}"
    for _ in range(_MAX_LIST_PAGES):
        page = _request_json(api_key, path)
        pods = page.get("pods") if isinstance(page, dict) else None
        paging = page.get("pagination") if isinstance(page, dict) else None
        if not isinstance(pods, list) or not isinstance(paging, dict):
            raise _ControllerError(refused)
        yield pods
        if not paging.get("hasNextPage"):
            return
        cursor = paging.get("nextCursor")
        if not isinstance(cursor, str) or not cursor:
            raise _ControllerError(refused)
        path = f"/pods?limit={_LIST_PAGE}&cursor=" + urllib.parse.quote(cursor, safe="")
    raise _ControllerError(refused)


def _recover_created(api_key, name):
    """A lost POST response must not cause another POST or delete unrelated Pods."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        for pods in _pod_pages(api_key):
            found = [
                item.get("id")
                for item in pods
                if isinstance(item, dict) and item.get("name") == name
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
