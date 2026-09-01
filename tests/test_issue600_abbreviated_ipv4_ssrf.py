"""#600 — abbreviated/decimal/hex/octal IPv4 forms bypass SSRF guards.

Verifies that:
1. All three copies of ``_is_private_or_link_local`` (hubs.py, hf.py, webhooks.py)
   reject non-canonical IPv4 forms and trailing-dot FQDN variants.
2. High-level callers (trackers._telemetry_endpoint_is_safe,
   trackers._resolve_posthog_target, webhooks.validate_webhook_url) reject
   non-canonical IPv4 endpoint overrides end-to-end.
"""
from __future__ import annotations

import pytest

# =====================================================================
# Unit: hubs.py — _is_private_or_link_local
# =====================================================================


class TestHubsPrivateOrLinkLocalAbbreviated:
    """socket.inet_aton fallback in hubs._is_private_or_link_local."""

    @pytest.mark.parametrize(
        "host",
        [
            # Canonical forms:
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",
            "::1",
            "fe80::1",
            "::ffff:10.0.0.1",
            # Abbreviated / non-canonical forms (#600 bypass vectors):
            "127.1",           # abbreviated loopback
            "127.0.1",         # abbreviated loopback (3-octet)
            "2130706433",      # decimal 127.0.0.1
            "0x7f000001",      # hex 127.0.0.1
            "0177.0.0.1",      # octal 127.0.0.1
            "10.1",            # abbreviated 10.0.0.1
            "167772161",       # decimal 10.0.0.1
            "2852039166",      # decimal 169.254.169.254 (cloud metadata)
            # Trailing-dot forms:
            "127.0.0.1.",
            "169.254.169.254.",
            "127.1.",
            "2852039166.",
        ],
    )
    def test_rejects_private_ip(self, host: str) -> None:
        from soup_cli.utils.hubs import _is_private_or_link_local

        assert _is_private_or_link_local(host) is True, (
            f"_is_private_or_link_local({host!r}) must return True"
        )

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "1.1.1.1",
            "us.i.posthog.com",
            "eu.i.posthog.com",
            "example.com",
            "8.8.8.8.",
        ],
    )
    def test_allows_public_host(self, host: str) -> None:
        from soup_cli.utils.hubs import _is_private_or_link_local

        assert _is_private_or_link_local(host) is False, (
            f"_is_private_or_link_local({host!r}) must return False"
        )


# =====================================================================
# Unit: hf.py — _is_private_or_link_local
# =====================================================================


class TestHfPrivateOrLinkLocalAbbreviated:
    """socket.inet_aton fallback in hf._is_private_or_link_local."""

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.1",
            "2130706433",
            "0x7f000001",
            "0177.0.0.1",
            "10.1",
            "167772161",
            "2852039166",
            "169.254.169.254.",
        ],
    )
    def test_rejects_private_ip(self, host: str) -> None:
        from soup_cli.utils.hf import _is_private_or_link_local

        assert _is_private_or_link_local(host) is True, (
            f"hf._is_private_or_link_local({host!r}) must return True"
        )

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "us.i.posthog.com",
            "example.com",
        ],
    )
    def test_allows_public_host(self, host: str) -> None:
        from soup_cli.utils.hf import _is_private_or_link_local

        assert _is_private_or_link_local(host) is False, (
            f"hf._is_private_or_link_local({host!r}) must return False"
        )


# =====================================================================
# Unit: webhooks.py — _is_private_or_link_local
# =====================================================================


class TestWebhooksPrivateOrLinkLocalAbbreviated:
    """socket.inet_aton fallback in webhooks._is_private_or_link_local."""

    @pytest.mark.parametrize(
        "host",
        [
            "10.0.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "127.1",
            "2130706433",
            "0x7f000001",
            "10.1",
            "167772161",
            "2852039166",
            "169.254.169.254.",
            "2852039166.",
        ],
    )
    def test_rejects_private_ip(self, host: str) -> None:
        from soup_cli.utils.webhooks import _is_private_or_link_local

        assert _is_private_or_link_local(host) is True, (
            f"webhooks._is_private_or_link_local({host!r}) must return True"
        )

    @pytest.mark.parametrize(
        "host",
        [
            "8.8.8.8",
            "1.1.1.1",
            "us.i.posthog.com",
            "example.com",
        ],
    )
    def test_allows_public_host(self, host: str) -> None:
        from soup_cli.utils.webhooks import _is_private_or_link_local

        assert _is_private_or_link_local(host) is False, (
            f"webhooks._is_private_or_link_local({host!r}) must return False"
        )


# =====================================================================
# End-to-End: Telemetry SSRF Guard Pinning (#600)
# =====================================================================


class TestTelemetrySSRFEndToEndPinning:
    """Pins #600 at the telemetry caller layer (trackers.py)."""

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://127.1/",
            "https://127.0.1/capture/",
            "https://2130706433/",
            "https://0x7f000001/",
            "https://0177.0.0.1/",
            "https://10.1/",
            "https://167772161/",
            "https://2852039166/capture/",
            "https://169.254.169.254./capture/",
            "https://127.1./",
        ],
    )
    def test_telemetry_endpoint_is_safe_rejects_abbreviated_forms(
        self, endpoint: str
    ) -> None:
        from soup_cli.utils.trackers import _telemetry_endpoint_is_safe

        assert _telemetry_endpoint_is_safe(endpoint) is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://us.i.posthog.com/",
            "https://eu.i.posthog.com/i/v0/e/",
            "https://app.posthog.com/capture/",
        ],
    )
    def test_telemetry_endpoint_is_safe_allows_public(
        self, endpoint: str
    ) -> None:
        from soup_cli.utils.trackers import _telemetry_endpoint_is_safe

        assert _telemetry_endpoint_is_safe(endpoint) is True

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://127.1/",
            "https://2130706433/",
            "https://2852039166/",
            "https://10.1/",
        ],
    )
    def test_resolve_posthog_target_rejects_abbreviated_arg(
        self, endpoint: str
    ) -> None:
        from soup_cli.utils.trackers import _resolve_posthog_target

        assert _resolve_posthog_target("phc_key", endpoint=endpoint) is None

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://127.1/",
            "https://2130706433/",
            "https://2852039166/",
            "https://167772161/",
        ],
    )
    def test_resolve_posthog_target_rejects_abbreviated_env(
        self, endpoint: str
    ) -> None:
        from soup_cli.utils.trackers import _resolve_posthog_target

        env = {"SOUP_POSTHOG_ENDPOINT": endpoint}
        assert _resolve_posthog_target(None, env=env) is None


# =====================================================================
# End-to-End: Webhooks SSRF Guard Pinning (#600)
# =====================================================================


class TestWebhookSSRFEndToEndPinning:
    """Pins #600 at the webhook caller layer (webhooks.py)."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://2852039166/hook",
            "https://10.1/hook",
            "https://167772161/hook",
            "https://169.254.169.254./hook",
            "https://2852039166./hook",
            "http://10.1/hook",
        ],
    )
    def test_validate_webhook_url_rejects_abbreviated_private_ips(
        self, url: str
    ) -> None:
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(ValueError):
            validate_webhook_url(url)

    def test_validate_webhook_url_permits_explicit_allow_private(self) -> None:
        from soup_cli.utils.webhooks import validate_webhook_url

        url = "https://10.1/hook"
        assert (
            validate_webhook_url(url, allow_private_hosts=True)
            == "https://10.1/hook"
        )

    def test_validate_webhook_url_allows_public(self) -> None:
        from soup_cli.utils.webhooks import validate_webhook_url

        url = "https://hooks.slack.com/services/T/B/x"
        assert validate_webhook_url(url) == url
