"""Ratchet direct/aliased httpx.post/get calls and review post_capped callers.

The raw-read check covers only httpx.post/get, with explicit legacy exceptions.
It does not detect reads on arbitrary response objects, httpx.stream or Client
instances. The sink ledger separately requires endpoint review for post_capped;
this module does not establish a byte limit for every outbound response.
"""

from pathlib import Path

import pytest

from tests.test_issue1547_outbound_http_sinks import (
    DECLARED_SINKS,
    scan_src_sinks,
    unlisted_sinks,
)

LEGACY_REQUESTS = {
    (
        "data/providers/ollama.py",
        "detect_ollama",
    ): "Local Ollama discovery/version probes; separate discovery contract.",
    (
        "utils/data_forge.py",
        "make_judge_provider_fn._anthropic_judge",
    ): "Separate forge judge adapter with its own failure contract; fixed Anthropic host.",
    (
        "utils/data_forge.py",
        "make_judge_provider_fn._ollama_judge",
    ): "Separate forge judge adapter; local endpoint, follow-up streaming migration.",
    (
        "utils/data_forge.py",
        "make_judge_provider_fn._vllm_judge",
    ): "Separate forge judge adapter; follow-up streaming migration.",
    (
        "utils/sglang.py",
        "generate_with_runtime",
    ): "In-process local SGLang runtime protocol; separate lifecycle contract.",
    (
        "utils/loop_stages.py",
        "_post_activate",
    ): "Activation acknowledgment only; response body is not parsed.",
    (
        "utils/webhooks.py",
        "post_webhook",
    ): "Webhook acknowledgment only; response body is not parsed.",
}


def _unapproved_reads(hits: list[tuple[str, str, int, str]]) -> list[tuple[str, str, int, str]]:
    return [
        hit for hit in hits
        if hit[3] in ("httpx.post", "httpx.get") and hit[:2] not in LEGACY_REQUESTS
    ]


def test_no_new_unbounded_httpx_reads() -> None:
    hits = scan_src_sinks()
    reads = {
        (path, name)
        for path, name, _, call in hits
        if call in ("httpx.post", "httpx.get")
    }
    assert _unapproved_reads(hits) == []
    assert reads == set(LEGACY_REQUESTS), (
        f"Unapproved reads: {sorted(reads - set(LEGACY_REQUESTS))}; "
        f"stale exceptions: {sorted(set(LEGACY_REQUESTS) - reads)}"
    )
    assert all(LEGACY_REQUESTS.values())


@pytest.mark.parametrize(
    "source",
    [
        "import httpx\ndef rogue(url):\n    return httpx.post(url).json()\n",
        "from httpx import get as fetch\ndef rogue(url):\n"
        "    response = fetch(url)\n    return response.json()\n",
    ],
)
def test_ratchet_detects_new_and_aliased_reads(tmp_path: Path, source: str) -> None:
    (tmp_path / "rogue.py").write_text(source, encoding="utf-8")
    hits = _unapproved_reads(scan_src_sinks(tmp_path))
    assert len(hits) == 1
    assert hits[0][:2] == ("rogue.py", "rogue")
    assert hits[0][3] in ("httpx.post", "httpx.get")


@pytest.mark.parametrize("imports, call", [
    ("from soup_cli.utils.http_response import post_capped", "post_capped"),
    ("from soup_cli.utils.http_response import post_capped as send", "send"),
    ("import soup_cli.utils.http_response as transport", "transport.post_capped"),
    ("from soup_cli.utils import http_response as transport", "transport.post_capped"),
    ("import soup_cli.utils.http_response", "soup_cli.utils.http_response.post_capped"),
])
def test_new_capped_callers_still_require_endpoint_review(
    tmp_path: Path, imports: str, call: str,
) -> None:
    source = f"{imports}\ndef rogue(url):\n    return {call}(url).json()\n"
    (tmp_path / "rogue.py").write_text(source, encoding="utf-8")
    hits = scan_src_sinks(tmp_path)
    assert len(hits) == 1
    assert hits[0][:2] == ("rogue.py", "rogue")
    assert hits[0][3] == "soup_cli.utils.http_response.post_capped"
    errors = unlisted_sinks(hits, DECLARED_SINKS)
    assert len(errors) == 1
    assert "rogue.py" in errors[0]
    assert "soup_cli.utils.http_response.post_capped" in errors[0]
