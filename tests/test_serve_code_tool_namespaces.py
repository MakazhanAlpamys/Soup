"""``/v1/tools/python`` and ``/v1/tools/bash`` run in their own network namespace on Linux.

Every other test of these routes fakes ``os.unshare`` or the sandbox. These start the
real child on a Linux host with Python 3.12+ (``os.unshare``) and read the network
interfaces it sees from ``/proc/self/net/dev``, which follows the network namespace of
the reading process. In a fresh namespace the only interface is ``lo``; the host's own
interfaces show up only when the child runs in the host's namespace.

A host or container that refuses unprivileged user and network namespaces makes the
routes answer 501 without running anything; these tests then skip and say why. With
``SOUP_TEST_REQUIRE_SANDBOX_NAMESPACES=1`` in the environment the skip becomes a
failure, for a CI job whose runner is known to allow the namespaces.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "unshare"),
    reason="the namespace sandbox needs Linux and Python 3.12+ (os.unshare)",
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

REQUIRE_ENV = "SOUP_TEST_REQUIRE_SANDBOX_NAMESPACES"
TOKEN = "s3cret-tool-token"

# Each child prints /proc/self/net/dev as one process; _interfaces() reads the names.
PRINT_NET_DEV_PY = "print(open('/proc/self/net/dev').read(), end='')"
PRINT_NET_DEV_SH = "cat /proc/self/net/dev"


def _client() -> TestClient:
    from soup_cli.commands.serve import _create_app

    app = _create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name="test-model",
        max_tokens_default=16,
        auth_token=TOKEN,
    )
    return TestClient(
        app, base_url="http://127.0.0.1:8000", headers={"Authorization": f"Bearer {TOKEN}"}
    )


def _unavailable(reason: str) -> None:
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1)")
    pytest.skip(reason)


def _interfaces(net_dev: str) -> list[str]:
    """The interface names in a /proc/net/dev text: two header lines, then "name: ..."."""
    return sorted(row.split(":")[0].strip() for row in net_dev.splitlines()[2:] if ":" in row)


def _host_interfaces() -> list[str]:
    with open("/proc/self/net/dev", encoding="utf-8") as handle:
        return _interfaces(handle.read())


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/v1/tools/python", {"code": PRINT_NET_DEV_PY}),
        ("/v1/tools/bash", {"command": PRINT_NET_DEV_SH}),
    ],
    ids=["python", "bash"],
)
def test_the_child_sees_only_the_loopback_interface(route, body, monkeypatch):
    from soup_cli.trainer import rewards

    host = _host_interfaces()
    if host == ["lo"]:
        # Nothing to tell apart: the child would print "lo" in either namespace.
        pytest.skip("this host has no network interface besides lo")
    # Detect the strategy afresh, whatever an earlier test cached.
    monkeypatch.setattr(rewards, "_ISOLATION_STRATEGY_CACHE", None)

    resp = _client().post(route, json=body)
    if resp.status_code == 501:
        _unavailable(f"this host refuses the sandbox namespaces: {resp.json()['detail']}")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["exit_code"] == 0, data
    assert _interfaces(data["stdout"]) == ["lo"], (data, "host interfaces:", host)
