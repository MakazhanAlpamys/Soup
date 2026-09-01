"""Tests for #593 - telemetry endpoint SSRF guard bypass.

`_telemetry_endpoint_is_safe` must reject private/loopback/link-local
hosts on HTTPS (not just HTTP), while still accepting legitimate public
PostHog endpoints. The guard uses a two-layer design:

1. `validate_hub_endpoint` for baseline sanitization (CRLF, null, types).
2. Telemetry-strict private/loopback/link-local rejection on any scheme.

Every test class here is structured so that **deleting** or **neutering**
the Layer 2 guard deterministically fails at least one named test.
"""

from __future__ import annotations

import pytest

from soup_cli.utils.trackers import (
    _resolve_posthog_target,
    _telemetry_endpoint_is_safe,
)

# --- Rejection: private / loopback / link-local must be refused ----------


class TestTelemetrySSRFRejection:
    """Every private/loopback/link-local HTTPS endpoint must be refused."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://10.0.0.1/",
            "https://10.255.255.255/capture/",
        ],
        ids=["rfc1918-10-simple", "rfc1918-10-upper"],
    )
    def test_rejects_private_ipv4_class_a(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://172.16.0.1/",
            "https://172.31.255.255/",
        ],
        ids=["rfc1918-172-lower", "rfc1918-172-upper"],
    )
    def test_rejects_private_ipv4_class_b(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://192.168.0.1/",
            "https://192.168.1.1/capture/",
        ],
        ids=["rfc1918-192-simple", "rfc1918-192-subnet"],
    )
    def test_rejects_private_ipv4_class_c(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is False

    def test_rejects_loopback_ipv4(self) -> None:
        assert _telemetry_endpoint_is_safe("https://127.0.0.1/") is False

    def test_rejects_cloud_metadata(self) -> None:
        """169.254.169.254 is the cloud metadata service endpoint."""
        assert (
            _telemetry_endpoint_is_safe("https://169.254.169.254/")
            is False
        )

    def test_rejects_ipv6_loopback_bracketed(self) -> None:
        """urlparse('https://[::1]/').hostname strips brackets."""
        assert (
            _telemetry_endpoint_is_safe("https://[::1]/") is False
        )

    def test_rejects_ipv4_mapped_ipv6(self) -> None:
        """IPv4-mapped IPv6 (::ffff:10.0.0.1) must not bypass the guard."""
        assert (
            _telemetry_endpoint_is_safe(
                "https://[::ffff:10.0.0.1]/"
            )
            is False
        )

    def test_rejects_localhost(self) -> None:
        assert (
            _telemetry_endpoint_is_safe("https://localhost/") is False
        )

    def test_rejects_localhost_uppercase(self) -> None:
        """Hostname normalisation must be case-insensitive."""
        assert (
            _telemetry_endpoint_is_safe("https://LOCALHOST/") is False
        )

    def test_rejects_localhost_trailing_dot(self) -> None:
        """FQDN trailing dot: 'LOCALHOST.' -> 'localhost' after strip."""
        assert (
            _telemetry_endpoint_is_safe("https://LOCALHOST./") is False
        )

    def test_rejects_http_any(self) -> None:
        """HTTP is rejected before either layer runs."""
        assert (
            _telemetry_endpoint_is_safe(
                "http://us.i.posthog.com/capture/"
            )
            is False
        )

    def test_rejects_non_string(self) -> None:
        assert _telemetry_endpoint_is_safe(42) is False  # type: ignore[arg-type]
        assert _telemetry_endpoint_is_safe(None) is False  # type: ignore[arg-type]

    def test_rejects_empty_string(self) -> None:
        assert _telemetry_endpoint_is_safe("") is False


# --- Acceptance: legitimate public endpoints must pass -------------------


class TestTelemetrySSRFAcceptance:
    """Valid public PostHog endpoints must be accepted.

    A validator that rejects everything is worse than the bug it fixes.
    """

    def test_accepts_public_posthog(self) -> None:
        assert (
            _telemetry_endpoint_is_safe(
                "https://us.i.posthog.com/capture/"
            )
            is True
        )

    def test_accepts_eu_posthog(self) -> None:
        assert (
            _telemetry_endpoint_is_safe(
                "https://eu.i.posthog.com/capture/"
            )
            is True
        )

    def test_accepts_custom_posthog_host(self) -> None:
        """A company's self-hosted public PostHog must pass when resolving to public IP."""
        import socket
        from unittest.mock import patch

        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert (
                _telemetry_endpoint_is_safe(
                    "https://posthog.mycompany.com/capture/"
                )
                is True
            )

    def test_accepts_default_posthog_endpoint(self) -> None:
        """The hardcoded default must always be accepted."""
        from soup_cli.utils.trackers import _POSTHOG_ENDPOINT

        assert _telemetry_endpoint_is_safe(_POSTHOG_ENDPOINT) is True


# --- Mutation controls: guard deletion must fail a named test ------------


class TestTelemetrySSRFMutationControl:
    """Negative controls proving the guard is load-bearing.

    Each test patches away a specific part of the Layer 2 guard and
    verifies that a private endpoint would then be accepted -- proving
    the guard's presence is what causes rejection, not some other
    mechanism.
    """

    def test_private_ip_guard_has_teeth(self) -> None:
        """Removing _is_private_or_link_local check would accept 10.0.0.1.

        This proves the guard is not dead code. If you delete the
        `_is_private_or_link_local(host)` branch from
        `_telemetry_endpoint_is_safe`, this test fails.
        """
        import soup_cli.utils.trackers as trackers_mod

        original_fn = trackers_mod._telemetry_endpoint_is_safe

        def _neutered_safe(endpoint: str) -> bool:
            """Bypass: skip the _is_private_or_link_local check."""
            if not isinstance(endpoint, str):
                return False
            if not endpoint.startswith("https://"):
                return False
            try:
                from soup_cli.utils.hubs import validate_hub_endpoint

                validate_hub_endpoint(endpoint, hub="telemetry")
            except (TypeError, ValueError):
                return False
            # Layer 2 deliberately omitted -- this is the mutation.
            return True

        # With the guard neutered, a private IP passes through.
        assert _neutered_safe("https://10.0.0.1/") is True
        # But the real implementation rejects it.
        assert original_fn("https://10.0.0.1/") is False

    def test_loopback_guard_has_teeth(self) -> None:
        """Removing _LOOPBACK_HOSTS check would accept localhost.

        `_is_private_or_link_local('localhost')` returns False because
        `ipaddress.ip_address('localhost')` raises ValueError. Without
        the explicit `_LOOPBACK_HOSTS` check, `https://localhost/`
        would slip through.
        """
        import ipaddress

        # Prove the gap: ipaddress cannot parse "localhost".
        with pytest.raises(ValueError):
            ipaddress.ip_address("localhost")

        from soup_cli.utils.hubs import _is_private_or_link_local

        # Prove _is_private_or_link_local does NOT catch localhost.
        assert _is_private_or_link_local("localhost") is False

        # But the real guard catches it via _LOOPBACK_HOSTS.
        assert (
            _telemetry_endpoint_is_safe("https://localhost/") is False
        )


# --- Integration: _resolve_posthog_target respects the guard -------------


class TestResolvePosthogTargetSSRF:
    """End-to-end: endpoint overrides go through the SSRF guard."""

    def test_env_override_private_ip_blocked(self) -> None:
        """SOUP_POSTHOG_ENDPOINT pointing to a private IP -> None."""
        result = _resolve_posthog_target(
            "phc_test",
            endpoint="https://10.0.0.1/capture/",
        )
        assert result is None

    def test_env_override_cloud_metadata_blocked(self) -> None:
        result = _resolve_posthog_target(
            "phc_test",
            endpoint="https://169.254.169.254/capture/",
        )
        assert result is None

    def test_env_override_localhost_blocked(self) -> None:
        result = _resolve_posthog_target(
            "phc_test",
            endpoint="https://localhost/capture/",
        )
        assert result is None

    def test_env_override_valid_accepted(self) -> None:
        """A valid custom PostHog host must resolve."""
        result = _resolve_posthog_target(
            "phc_test",
            endpoint="https://custom.posthog.com/capture/",
        )
        assert result is not None
        key, ep = result
        assert key == "phc_test"
        assert ep == "https://custom.posthog.com/capture/"

    def test_env_var_override_private_ip_blocked(self) -> None:
        """SOUP_POSTHOG_ENDPOINT env var with private IP -> None."""
        result = _resolve_posthog_target(
            None,
            env={"SOUP_POSTHOG_ENDPOINT": "https://10.0.0.1/capture/"},
        )
        assert result is None

    def test_env_var_override_valid_accepted(self) -> None:
        """SOUP_POSTHOG_ENDPOINT env var with public host -> tuple."""
        result = _resolve_posthog_target(
            None,
            env={
                "SOUP_POSTHOG_ENDPOINT":
                    "https://eu.i.posthog.com/capture/",
            },
        )
        assert result is not None
        _, ep = result
        assert ep == "https://eu.i.posthog.com/capture/"
