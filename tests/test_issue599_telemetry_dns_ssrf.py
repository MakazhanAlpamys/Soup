"""#599 — Telemetry SSRF: Hostname DNS resolution & tiered validation.

Verifies:
1. Tier 1 (Primary Control): Trusted PostHog domain allowlist (*.posthog.com)
   accepts immediately with zero DNS lookups.
2. Tier 2 (Static Guards): Rejects loopback names, literal private IPs, and
   internal TLD suffixes (.local, .internal, .lan, .corp, etc.) without DNS.
3. Tier 3 (Defence in Depth): Resolves custom non-default FQDNs with bounded
   timeout, rejecting if any resolved IP is private/loopback/link-local, and
   failing closed on resolver errors/timeouts.
"""
from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from soup_cli.utils.trackers import (
    _INTERNAL_TLD_SUFFIXES,
    _is_trusted_posthog_domain,
    _resolve_host_ips,
    _resolve_posthog_target,
    _telemetry_endpoint_is_safe,
)

# =====================================================================
# Tier 1: Primary Control (Trusted Domain Allowlist)
# =====================================================================


class TestTier1TrustedAllowlist:
    """Primary control: trusted PostHog domains accept with zero DNS queries."""

    @pytest.mark.parametrize(
        "host",
        [
            "posthog.com",
            "us.i.posthog.com",
            "eu.i.posthog.com",
            "app.posthog.com",
            "custom.subdomain.posthog.com",
        ],
    )
    def test_is_trusted_posthog_domain_helper(self, host: str) -> None:
        assert _is_trusted_posthog_domain(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "notposthog.com",
            "fake-posthog.com",
            "posthog.com.attacker.com",
            "example.com",
            "10.0.0.1.nip.io",
        ],
    )
    def test_untrusted_domains_rejected_by_tier1(self, host: str) -> None:
        assert _is_trusted_posthog_domain(host) is False

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://us.i.posthog.com/",
            "https://eu.i.posthog.com/i/v0/e/",
            "https://app.posthog.com/capture/",
            "https://custom.sub.posthog.com/capture",
        ],
    )
    def test_tier1_allows_trusted_posthog_endpoints(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is True

    def test_tier1_never_performs_dns_resolution(self) -> None:
        """Default/trusted endpoints must execute in sub-ms with 0 network calls."""
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.side_effect = AssertionError("DNS resolution must not be called for Tier 1")
            assert _telemetry_endpoint_is_safe("https://us.i.posthog.com/") is True
            assert _telemetry_endpoint_is_safe("https://eu.i.posthog.com/i/v0/e/") is True
            assert mock_gai.call_count == 0


# =====================================================================
# Tier 2: Static Syntactic & Internal TLD Rejection
# =====================================================================


class TestTier2StaticRejection:
    """Static guards: reject literal private IPs and internal TLDs without DNS."""

    @pytest.mark.parametrize(
        "suffix",
        [
            ".local",
            ".internal",
            ".localhost",
            ".lan",
            ".home",
            ".corp",
            ".intranet",
        ],
    )
    def test_all_expected_internal_tlds_covered(self, suffix: str) -> None:
        assert suffix in _INTERNAL_TLD_SUFFIXES

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://metadata.internal/capture/",
            "https://service.local/capture/",
            "https://myhost.lan/capture/",
            "https://gateway.home/capture/",
            "https://portal.corp/capture/",
            "https://secure.intranet/capture/",
            "https://test.localhost/capture/",
        ],
    )
    def test_tier2_rejects_internal_tld_endpoints(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is False

    def test_tier2_internal_tld_never_performs_dns(self) -> None:
        """Internal TLDs must be rejected as a pure string decision without DNS."""
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.side_effect = AssertionError("DNS resolution must not be called for Tier 2")
            assert _telemetry_endpoint_is_safe("https://metadata.internal/capture/") is False
            assert _telemetry_endpoint_is_safe("https://router.local/capture/") is False
            assert mock_gai.call_count == 0

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://10.0.0.1/capture/",
            "https://127.0.0.1/capture/",
            "https://169.254.169.254/capture/",
            "https://192.168.1.1/capture/",
            "https://127.1/capture/",
            "https://2130706433/capture/",
            "https://[::1]/capture/",
            "https://[fe80::1]/capture/",
            "https://localhost/capture/",
        ],
    )
    def test_tier2_rejects_literal_private_and_loopback_ips(self, endpoint: str) -> None:
        assert _telemetry_endpoint_is_safe(endpoint) is False


# =====================================================================
# Tier 3: Defence-in-Depth DNS Resolution for Custom FQDNs
# =====================================================================


class TestTier3DNSDefenceInDepth:
    """DNS resolution checks custom hostnames to raise the bar against indirection."""

    def test_tier3_rejects_hostname_resolving_to_private_rfc1918(self) -> None:
        """10.0.0.1.nip.io or custom domain pointing to private IP is rejected."""
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert (
                _telemetry_endpoint_is_safe("https://custom-analytics.example.com/capture")
                is False
            )

    def test_tier3_rejects_hostname_resolving_to_loopback(self) -> None:
        """localtest.me or domain pointing to 127.0.0.1 is rejected."""
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert _telemetry_endpoint_is_safe("https://localtest.me/capture/") is False

    def test_tier3_rejects_hostname_resolving_to_cloud_metadata(self) -> None:
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert _telemetry_endpoint_is_safe("https://cloud-meta.custom.org/capture") is False

    def test_tier3_rejects_dual_stack_multi_ip_when_any_ip_is_private(self) -> None:
        """Dual-stack (public IPv4 + loopback IPv6) must reject."""
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert _telemetry_endpoint_is_safe("https://dual-stack.example.com/capture") is False

    def test_tier3_allows_custom_self_hosted_domain_resolving_to_public_ip(self) -> None:
        """Custom self-hosted PostHog on a legitimate public IP is permitted."""
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            assert _telemetry_endpoint_is_safe("https://telemetry.mycorp.org/capture/") is True

    def test_tier3_fails_closed_on_resolution_error(self) -> None:
        gai_err = socket.gaierror(-2, "Name or service not known")
        with patch("socket.getaddrinfo", side_effect=gai_err):
            assert _telemetry_endpoint_is_safe("https://unresolvable.invalid/capture") is False

    def test_tier3_fails_closed_on_resolution_timeout(self) -> None:
        with patch("soup_cli.utils.trackers._resolve_host_ips", return_value=None):
            assert _telemetry_endpoint_is_safe("https://hanging-dns.example.com/capture") is False

    def test_resolve_host_ips_helper_timeout_handling(self) -> None:
        """_resolve_host_ips returns None when resolution exceeds timeout deadline."""
        def slow_getaddrinfo(*args: object, **kwargs: object) -> list[object]:
            import time
            time.sleep(0.5)
            return []

        with patch("socket.getaddrinfo", side_effect=slow_getaddrinfo):
            # 0.05s timeout should trigger timeout return None
            assert _resolve_host_ips("example.com", timeout=0.05) is None


# =====================================================================
# End-to-End: _resolve_posthog_target Integration
# =====================================================================


class TestResolvePosthogTargetTieredIntegration:
    """Integration tests for target resolution with environment variables."""

    def test_resolve_default_posthog_target_succeeds(self) -> None:
        resolved = _resolve_posthog_target(None, env={})
        assert resolved is not None
        key, endpoint = resolved
        assert "posthog.com" in endpoint

    def test_resolve_rejects_hostname_ssrf_via_env(self) -> None:
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            env = {"SOUP_POSTHOG_ENDPOINT": "https://10.0.0.1.nip.io/capture/"}
            assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_rejects_internal_tld_via_env(self) -> None:
        env = {"SOUP_POSTHOG_ENDPOINT": "https://metadata.internal/capture/"}
        assert _resolve_posthog_target(None, env=env) is None

    def test_resolve_accepts_valid_custom_endpoint_via_env(self) -> None:
        fake_addrinfo = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            env = {"SOUP_POSTHOG_ENDPOINT": "https://telemetry.selfhosted.io/capture/"}
            resolved = _resolve_posthog_target("phc_custom", env=env)
            assert resolved is not None
            key, endpoint = resolved
            assert key == "phc_custom"
            assert endpoint == "https://telemetry.selfhosted.io/capture/"
