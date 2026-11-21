"""Tests for Lambda Cloud controller launch response loss recovery (#2355).

Verifies that the rendered Lambda controller:
1. Sends a unique `name` with the launch request for reconciliation.
2. Distinguishes definite rejection (4xx except 408) from ambiguous failures.
3. On ambiguous launch, reconciles via `GET /instances` matching the `name`.
4. Terminates only matching instances and verifies termination.
5. Prints CRITICAL warning when reconciliation fails.
6. Keeps the known-ID path unchanged: terminate and verify the returned ID.
"""

from __future__ import annotations

import subprocess
import types
import urllib.error
from pathlib import Path

from soup_cli.cloud.lambda_labs import render_lambda_stub


class _FakeLambdaAPIError(RuntimeError):
    """Test double for the controller's `_LambdaAPIError`.

    The controller defines its own class inside the generated stub, so the
    tests carry the same three attributes the generated `main()` reads.
    """

    def __init__(self, status: int, method: str = "POST") -> None:
        super().__init__(f"Lambda API {method} failed with HTTP {status}")
        self.status = status
        self.method = method
        self.create_rejected = method == "POST" and 400 <= status < 500 and status != 408


_SOUP_YAML = (
    "base: hf-internal-testing/tiny-random-gpt2\n"
    "task: sft\n"
    "data:\n  train: data.jsonl\n  format: chatml\n"
    "output: ./out\n"
)


def _setup_env(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    key = tmp_path / "lambda-key"
    key.write_text("test-only", encoding="utf-8")
    monkeypatch.setenv("LAMBDA_API_KEY", "secret")
    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "my-key")
    monkeypatch.setenv("LAMBDA_SSH_PRIVATE_KEY", str(key))
    monkeypatch.setenv("LAMBDA_REGION", "us-tx-1")
    stub = render_lambda_stub(
        _SOUP_YAML, gpu="a100", output_dir="./out", soup_version="0.77.1"
    )
    return stub, key


def _make_fake_run():
    """Create a fake_run that simulates successful training (poll returns 0 immediately)."""

    def fake_run(argv, **kw):
        if argv[0] == "ssh":
            cmd_str = argv[-1] if argv else ""
            if "tail" in cmd_str:
                return subprocess.CompletedProcess(argv, 0, "error tail\n", "")
            return subprocess.CompletedProcess(argv, 0, "0\n", "")
        if argv[0] == "scp":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return fake_run


def _exec_controller(
    stub: str,
    fake_run_fn,
    fake_request_fn,
    captured_prints: list[str] | None = None,
    terminations: list[tuple] | None = None,
    lookups: list[tuple] | None = None,
    termination_checks: list[str] | None = None,
) -> tuple[int, list, list]:
    namespace: dict[str, object] = {"__name__": "controller"}
    exec(stub, namespace)
    namespace["_find_tool"] = lambda name: name

    if terminations is None:
        terminations = []
    if lookups is None:
        lookups = []
    if termination_checks is None:
        termination_checks = []
    clock = {"now": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    namespace["_request_json"] = lambda api_key, path, *, method="GET", payload=None: (
        fake_request_fn(api_key, path, method=method, payload=payload)
    )
    namespace["_wait_for_ip"] = lambda *_: "192.0.2.1"
    namespace["_wait_for_termination"] = lambda _api_key, instance_id: (
        termination_checks.append(instance_id)
    )
    namespace["time"] = types.SimpleNamespace(
        sleep=fake_sleep,
        monotonic=lambda: clock["now"],
    )

    if captured_prints is not None:
        namespace["print"] = lambda *args, **kw: captured_prints.append(
            " ".join(str(a) for a in args)
        )

    namespace["subprocess"] = types.SimpleNamespace(
        run=fake_run_fn,
        TimeoutExpired=subprocess.TimeoutExpired,
    )

    rc = namespace["main"]()
    return rc, terminations, lookups


class TestLambdaLaunchRecovery:
    """Tests for the three launch outcome cases."""

    def test_known_id_cleanup_unchanged(self, tmp_path: Path, monkeypatch) -> None:
        """Known-ID path: launch returns an ID, controller terminates and verifies it."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        checks: list[str] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                return {"data": {"instance_ids": ["instance-known"]}}
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                return {"data": []}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
            termination_checks=checks,
        )

        assert rc == 0
        assert terms == [(terms[0][0], "/instance-operations/terminate",
                          {"instance_ids": ["instance-known"]})]
        assert lkups == []
        assert checks == ["instance-known"]
        assert any("Lambda instance launched: instance-known" in p for p in prints)
        assert any(
            "Lambda instance termination verified: instance-known" in p
            for p in prints
        )

    def test_definite_rejection_no_cleanup(self, tmp_path: Path, monkeypatch) -> None:
        """Definite rejection: 4xx (except 408) from launch POST means no instance created."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                raise _FakeLambdaAPIError(400)
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                return {"data": []}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms == []
        assert lkups == []
        assert any(
            "Lambda create request rejected; no instance to clean up" in p
            for p in prints
        )

    def test_definite_rejection_404_no_cleanup(self, tmp_path: Path, monkeypatch) -> None:
        """Definite rejection: 404 from launch POST means no instance created."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                raise _FakeLambdaAPIError(404)
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                return {"data": []}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms == []
        assert lkups == []
        assert any(
            "Lambda create request rejected; no instance to clean up" in p
            for p in prints
        )

    def test_lost_launch_response_reconciles_and_reports_failure(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The issue's own case: `URLError` does not escape as a traceback."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        checks: list[str] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise urllib.error.URLError("simulated response lost")
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                return {"data": [
                    {"id": "instance-lost", "name": launch_payloads[0]["name"]},
                ]}
            return {"data": {}}

        rc, _, _ = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, termination_checks=checks,
        )

        assert rc == 1
        assert terminations == [(
            terminations[0][0], "/instance-operations/terminate",
            {"instance_ids": ["instance-lost"]},
        )]
        assert checks == ["instance-lost"]
        assert any(
            "Lambda training failed: <urlopen error simulated response lost>" in p
            for p in prints
        )

    def test_ambiguous_transport_error_reconciles_and_terminates(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Ambiguous failure: transport error triggers reconciliation and termination."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        checks: list[str] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise RuntimeError("simulated response lost")
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                name = launch_payloads[0].get("name") if launch_payloads else "unknown"
                return {"data": [{"id": "instance-recovered", "name": name}]}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
            termination_checks=checks,
        )

        assert rc == 1
        assert len(launch_payloads) == 1
        assert "name" in launch_payloads[0]
        assert launch_payloads[0]["name"].startswith("soup-")
        assert terms[0][2] == {"instance_ids": ["instance-recovered"]}
        assert len(lkups) == 1
        assert checks == ["instance-recovered"]
        assert any(
            "Lambda termination verified for ambiguous launch" in p
            for p in prints
        )

    def test_ambiguous_two_ids_reconciles_and_terminates(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A launch answer with two ids is ambiguous: reconcile rather than trust one."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        checks: list[str] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                return {"data": {"instance_ids": ["instance-a", "instance-b"]}}
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                name = launch_payloads[0].get("name") if launch_payloads else "unknown"
                return {"data": [{"id": "instance-owned", "name": name}]}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
            termination_checks=checks,
        )

        assert rc == 1
        assert terms[0][2] == {"instance_ids": ["instance-owned"]}
        assert len(lkups) == 1
        assert checks == ["instance-owned"]
        assert any("Lambda launch returned" in p for p in prints)

    def test_ambiguous_malformed_response_reconciles(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Ambiguous failure: a launch answer with no ids triggers reconciliation."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                return {"data": {"instance_ids": []}}
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                name = launch_payloads[0].get("name") if launch_payloads else "unknown"
                return {"data": [{"id": "instance-recovered-2", "name": name}]}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms[0][2] == {"instance_ids": ["instance-recovered-2"]}
        assert len(lkups) == 1
        assert any(
            "Lambda termination verified for ambiguous launch" in p
            for p in prints
        )

    def test_ambiguous_reconciliation_fails_prints_critical(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Ambiguous failure: no matching instance for the window prints CRITICAL."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise RuntimeError("service unavailable")
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                return {"data": []}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms == []
        assert len(lkups) >= 1
        name = launch_payloads[0].get("name") if launch_payloads else "unknown"
        assert any(
            f"CRITICAL: Lambda cleanup unconfirmed; check the console for {name}" in p
            for p in prints
        )

    def test_failed_lookup_prints_critical(self, tmp_path: Path, monkeypatch) -> None:
        """A failing `GET /instances` is reported as unconfirmed, not a traceback."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise RuntimeError(
                    "Lambda API POST /instance-operations/launch failed "
                    "with HTTP 502: bad gateway"
                )
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                raise RuntimeError("Lambda API GET /instances failed with HTTP 500: down")
            return {"data": {}}

        rc, _, _ = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations,
        )

        assert rc == 1
        assert terminations == []
        name = launch_payloads[0]["name"]
        assert any(
            f"CRITICAL: Lambda cleanup unconfirmed; check the console for {name}" in p
            for p in prints
        )

    def test_unrelated_instances_not_terminated(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Unrelated instances (different name) are not terminated during reconciliation."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise RuntimeError("simulated response lost")
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                name = launch_payloads[0].get("name") if launch_payloads else "unknown"
                return {"data": [
                    {"id": "instance-matching", "name": name},
                    {"id": "instance-unrelated", "name": "other-user-instance"},
                ]}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms[0][2] == {"instance_ids": ["instance-matching"]}
        assert any(
            "Lambda termination verified for ambiguous launch" in p
            for p in prints
        )

    def test_launch_payload_contains_name(self, tmp_path: Path, monkeypatch) -> None:
        """The launch payload includes a unique name field."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                return {"data": {"instance_ids": ["instance-1"]}}
            return {"data": {}}

        _exec_controller(stub, _make_fake_run(), fake_request)

        assert len(launch_payloads) == 1
        assert "name" in launch_payloads[0]
        assert launch_payloads[0]["name"].startswith("soup-")
        assert len(launch_payloads[0]["name"]) <= 64

    def test_408_request_timeout_is_ambiguous(self, tmp_path: Path, monkeypatch) -> None:
        """408 Request Timeout is treated as ambiguous, not definite rejection."""
        stub, _ = _setup_env(tmp_path, monkeypatch)
        prints: list[str] = []
        terminations: list[tuple] = []
        lookups: list[tuple] = []
        launch_payloads: list[dict] = []

        def fake_request(api_key, path, *, method="GET", payload=None, **_):
            if path.endswith("/launch"):
                launch_payloads.append(payload)
                raise RuntimeError("request timeout")
            if path.endswith("/terminate"):
                terminations.append((api_key, path, payload))
                return {"data": {}}
            if path == "/instances":
                lookups.append((api_key, path))
                name = launch_payloads[0].get("name") if launch_payloads else "unknown"
                return {"data": [{"id": "instance-408", "name": name}]}
            return {"data": {}}

        rc, terms, lkups = _exec_controller(
            stub, _make_fake_run(), fake_request, captured_prints=prints,
            terminations=terminations, lookups=lookups,
        )

        assert rc == 1
        assert terms[0][2] == {"instance_ids": ["instance-408"]}
        assert len(lkups) == 1
        assert any(
            "Lambda termination verified for ambiguous launch" in p
            for p in prints
        )
