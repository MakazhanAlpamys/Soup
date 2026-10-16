"""Outbound endpoint validators refuse non-public IP literals on every scheme.

Each validator below sits in front of an outbound HTTP request whose URL comes
from a flag, a config file or a request body. They already refused plain HTTP
to a non-loopback host. They must also refuse a URL whose host is an IP literal
in a non-public range, in every spelling ``net_guard.parse_ip_literal`` accepts
(abbreviated, decimal, hex and octal IPv4, IPv4-mapped IPv6), while loopback
literals and hostnames keep working. Hostnames are not resolved, here or
anywhere else in the repo, so this narrows what a URL can name directly; it
does not make an internal service unreachable by name.

No test in this file touches the network: ``httpx.post`` and ``httpx.stream``
are replaced by a recorder that raises instead of connecting. Every refused
case asserts the recorder was never reached, and every allowed case asserts it
WAS reached, so the first assertion cannot pass vacuously.
"""

from __future__ import annotations

import io
import re
from types import SimpleNamespace

import pytest

_REFUSED = "private/link-local/reserved IP hosts are not allowed"
# #1549: the refusal says what to do instead.
_REMEDY = "address the server by its hostname"
_UNSPECIFIED_HINT = "0.0.0.0 is ambiguous; use 127.0.0.1 or localhost"

# The host part of the URL, as written in its authority.
_NON_PUBLIC_HOSTS = [
    pytest.param("10.0.0.1", id="10/8"),
    pytest.param("172.16.0.1", id="172.16/12"),
    pytest.param("192.168.1.10:8443", id="192.168/16-with-port"),
    pytest.param("169.254.169.254", id="link-local-v4"),
    pytest.param("[fd00::1]", id="unique-local-v6"),
    pytest.param("[fe80::1]", id="link-local-v6"),
    pytest.param("[::ffff:10.0.0.1]", id="v4-mapped-v6"),
    pytest.param("[64:ff9b::a9fe:a9fe]", id="nat64-v6"),
    pytest.param("10.1", id="abbreviated-v4"),
    pytest.param("167772161", id="decimal-v4"),
    pytest.param("0x0a000001", id="hex-v4"),
    pytest.param("012.0.0.1", id="octal-v4"),
    pytest.param("169.254.169.254.", id="trailing-dot"),
    pytest.param("0.0.0.0", id="unspecified-v4"),
    pytest.param("[::]", id="unspecified-v6"),
    pytest.param("224.0.0.1", id="multicast-v4"),
    # RFC 6598 shared address space: neither is_private nor is_global.
    pytest.param("100.64.0.1", id="shared-space"),
    pytest.param("100.127.255.254", id="shared-space-top"),
    pytest.param("[::ffff:100.64.0.1]", id="shared-space-v4-mapped"),
    # Deprecated IPv6 site-local, which ipaddress still reports as global.
    pytest.param("[fec0::1]", id="site-local-v6"),
    # A client IDNA-encodes a non-ASCII host before it connects, folding these
    # separators and digits to ASCII: httpx dials 10.0.0.1 for the first three.
    pytest.param("10\u30020\u30020\u30021", id="ideographic-full-stops"),
    pytest.param("10\uff0e0\uff0e0\uff0e1", id="fullwidth-full-stops"),
    pytest.param("10\uff610\uff610\uff611", id="halfwidth-ideographic-full-stops"),
    pytest.param("169\uff0e254\uff0e169\uff0e254", id="link-local-fullwidth-stops"),
    pytest.param("0xa\u30020\u30020\u30021", id="hex-with-ideographic-stops"),
    pytest.param("10\ufe520\ufe520\ufe521", id="small-full-stops"),
    pytest.param("\uff11\uff10.0.0.1", id="fullwidth-digits"),
    pytest.param("1\u00ad0.0.0.1", id="soft-hyphen"),
]

# Loopback is the one non-public range these endpoints exist to reach.
_ALLOWED_HOSTS = [
    pytest.param("api.example.com", id="hostname"),
    pytest.param("8.8.8.8", id="public-v4"),
    pytest.param("100.63.255.255", id="just-below-shared-space"),
    pytest.param("100.128.0.1", id="just-above-shared-space"),
    pytest.param("127.0.0.1:8443", id="loopback-v4"),
    pytest.param("127.0.0.5", id="loopback-range"),
    pytest.param("127.1", id="abbreviated-loopback"),
    pytest.param("[::1]:8443", id="loopback-v6"),
    pytest.param("[::ffff:127.0.0.1]", id="v4-mapped-loopback"),
    pytest.param("127\u30020\u30020\u30021", id="loopback-ideographic-stops"),
    pytest.param("8\uff0e8\uff0e8\uff0e8", id="public-fullwidth-stops"),
    pytest.param("m\u00fcnchen.example", id="idn-hostname"),
]


class _RequestAttemptedError(Exception):
    """Raised by the recorder in place of a request."""


@pytest.fixture
def sent(monkeypatch) -> list[tuple[str, dict]]:
    """``httpx.post`` / ``httpx.stream`` replaced by a recorder that never connects."""
    import httpx

    calls: list[tuple[str, dict]] = []

    def _post(url, *args, **kwargs):
        calls.append((str(url), dict(kwargs.get("headers") or {})))
        raise _RequestAttemptedError(str(url))

    def _stream(method, url, *args, **kwargs):
        calls.append((str(url), dict(kwargs.get("headers") or {})))
        raise _RequestAttemptedError(str(url))

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(httpx, "stream", _stream)
    return calls


@pytest.fixture(autouse=True)
def _online_dpo_without_trl(monkeypatch):
    """Keep the online-DPO judge path off trl (its version probe imports it)."""
    import soup_cli.trainer.online_dpo as od

    monkeypatch.setattr(od, "_trl_has_judges", lambda: False)
    monkeypatch.setattr(od, "_ONLINE_DPO_JUDGE_OVERRIDE", None)


# --- one adapter per gate; each takes the URL's scheme://authority ----------


def _vllm_url(base: str) -> None:
    from soup_cli.data.providers.vllm import validate_vllm_url

    validate_vllm_url(base)


def _vllm_generate(base: str) -> None:
    from soup_cli.data.providers.vllm import generate_vllm

    generate_vllm(
        prompt="p", count=1, fmt="alpaca", model_name="m", base_url=base,
        temperature=0.0, generation_prompt="g",
    )


def _generate_openai(base: str) -> None:
    from soup_cli.commands.generate import _generate_openai as run

    run(
        prompt="p", count=1, fmt="alpaca", model_name="m", api_key="sk-test",
        api_base=base, temperature=0.0, seed_examples=[], generation_prompt="g",
    )


def _generate_server(base: str) -> None:
    from soup_cli.commands.generate import _generate_server as run

    run(
        prompt="p", count=1, fmt="alpaca", model_name="m", api_base=base,
        temperature=0.0, seed_examples=[], generation_prompt="g",
    )


def _judge_api_base(base: str) -> None:
    from soup_cli.eval.judge import validate_judge_api_base

    validate_judge_api_base(base)


def _judge_evaluator(base: str) -> None:
    from soup_cli.eval.judge import JudgeEvaluator

    JudgeEvaluator(provider="server", model="m", api_base=base)._call_llm("p")


def _gate_suite_task(base: str) -> None:
    from soup_cli.eval.gate import GateTask

    GateTask(
        type="judge", name="t", threshold=0.5, prompts="p.jsonl", judge_model=f"{base}/m"
    )


def _online_dpo_field(base: str) -> None:
    from soup_cli.config.schema import TrainingConfig

    TrainingConfig(online_dpo_judge=f"{base}/m")


def _online_dpo_trainer(base: str) -> None:
    """Trainer setup, reached with a config the schema never validated."""
    import soup_cli.trainer.online_dpo as od

    wrapper = object.__new__(od.OnlineDPOTrainerWrapper)
    wrapper._build_judge_or_reward(
        SimpleNamespace(online_dpo_judge=f"{base}/m", reward_model=None)
    )


def _ship_judge_model(base: str) -> None:
    from soup_cli.commands.ship import _validate_judge_model_url

    _validate_judge_model_url(f"{base}/Qwen2.5")


_VALIDATORS = [
    pytest.param(_vllm_url, id="vllm-url"),
    pytest.param(_vllm_generate, id="vllm-generate"),
    pytest.param(_generate_openai, id="generate-openai"),
    pytest.param(_generate_server, id="generate-server"),
    pytest.param(_judge_api_base, id="judge-api-base"),
    pytest.param(_judge_evaluator, id="judge-evaluator"),
    pytest.param(_gate_suite_task, id="gate-suite-judge-model"),
    pytest.param(_online_dpo_field, id="online-dpo-judge-field"),
    pytest.param(_online_dpo_trainer, id="online-dpo-trainer"),
]
# The adapters that go on to make a request once their gate passes.
_SINKS = frozenset({_vllm_generate, _generate_openai, _generate_server, _judge_evaluator})

# The setting each gate names in its refusal.
_LABELS = {
    _vllm_url: "vLLM URL",
    _vllm_generate: "vLLM URL",
    _generate_openai: "api_base",
    _generate_server: "api_base",
    _judge_api_base: "judge URL",
    _judge_evaluator: "judge URL",
    _gate_suite_task: "judge_model URL",
    _online_dpo_field: "online_dpo_judge",
    _online_dpo_trainer: "judge URL",
}

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    """CLI output without colour codes and with Rich's line wrapping undone."""
    return " ".join(_ANSI.sub("", text).split())


class TestNonPublicLiteralsAreRefused:
    @pytest.mark.parametrize("host", _NON_PUBLIC_HOSTS)
    @pytest.mark.parametrize("validator", _VALIDATORS)
    def test_refused_before_any_request(self, validator, host, sent):
        with pytest.raises(ValueError, match=re.escape(_REFUSED)):
            validator(f"https://{host}")
        assert sent == [], f"a request was attempted for {host!r}: {sent}"

    @pytest.mark.parametrize("validator", _VALIDATORS)
    def test_refusal_names_the_setting(self, validator, sent):
        with pytest.raises(ValueError) as info:
            validator("https://10.0.0.1")
        assert f"{_LABELS[validator]}: {_REFUSED}" in str(info.value)
        assert sent == []

    @pytest.mark.parametrize("validator", _VALIDATORS)
    def test_unspecified_address_is_not_local_over_http(self, validator, sent):
        with pytest.raises(ValueError):
            validator("http://0.0.0.0:8000")
        assert sent == []


class TestPublicAndLoopbackStillPass:
    @pytest.mark.parametrize("host", _ALLOWED_HOSTS)
    @pytest.mark.parametrize("validator", _VALIDATORS)
    def test_allowed(self, validator, host, sent):
        base = f"https://{host}"
        try:
            validator(base)
        except _RequestAttemptedError:
            pass
        urls = [url for url, _headers in sent]
        if validator in _SINKS:
            assert len(urls) == 1 and urls[0].startswith(base + "/"), urls
        else:
            assert urls == []

    @pytest.mark.parametrize("base", ["http://localhost:8000", "http://127.0.0.1:8000"])
    @pytest.mark.parametrize("validator", _VALIDATORS)
    def test_loopback_over_http(self, validator, base, sent):
        try:
            validator(base)
        except _RequestAttemptedError:
            pass
        assert len(sent) == (1 if validator in _SINKS else 0)

    @pytest.mark.parametrize(
        "validator",
        [
            _vllm_url, _vllm_generate, _generate_openai, _generate_server, _judge_api_base,
            _judge_evaluator, _online_dpo_field, _gate_suite_task, _ship_judge_model,
            _online_dpo_trainer,
        ],
    )
    def test_ipv6_loopback_over_http(self, validator, sent):
        """``::1`` is loopback for every outbound gate: all of them accept it over
        http through the shared ``LOOPBACK_HOSTS`` set (#1548)."""
        try:
            validator("http://[::1]:8000")
        except _RequestAttemptedError:
            pass
        assert len(sent) == (1 if validator in _SINKS else 0)

    def test_default_openai_base_still_carries_the_key(self, sent):
        from soup_cli.commands.generate import _generate_openai as run

        with pytest.raises(_RequestAttemptedError):
            run(
                prompt="p", count=1, fmt="alpaca", model_name="m", api_key="sk-test",
                api_base=None, temperature=0.0, seed_examples=[], generation_prompt="g",
            )
        [(url, headers)] = sent
        assert url == "https://api.openai.com/v1/chat/completions"
        assert headers.get("Authorization") == "Bearer sk-test"


class TestRemoteHttpKeepsItsMessage:
    """The scheme check still runs first, so the existing message is unchanged."""

    @pytest.mark.parametrize(
        ("validator", "message"),
        [
            (_vllm_url, "HTTPS for remote"),
            (_generate_openai, "HTTPS for remote"),
            (_generate_server, "HTTPS for remote"),
            (_judge_api_base, "Use HTTPS for remote"),
            (_gate_suite_task, "disallowed scheme"),
        ],
    )
    def test_remote_http(self, validator, message, sent):
        with pytest.raises(ValueError, match=message):
            validator("http://10.0.0.1:8000")
        assert sent == []

    @pytest.mark.parametrize(
        ("validator", "message"),
        [
            (_vllm_url, "HTTPS for remote"),
            (_generate_openai, "HTTPS for remote"),
            (_generate_server, "HTTPS for remote"),
            (_judge_api_base, "Use HTTPS for remote"),
        ],
    )
    def test_localhost_lookalike_is_remote_over_http(self, validator, message, sent):
        with pytest.raises(ValueError, match=message):
            validator("http://localhost.attacker.example:8000")
        assert sent == []

    @pytest.mark.parametrize("validator", [_generate_openai, _generate_server])
    def test_unspecified_address_over_http_names_the_fix(self, validator, sent):
        """0.0.0.0 is the bind-any wildcard; #1549 says to use loopback instead."""
        with pytest.raises(ValueError) as info:
            validator("http://0.0.0.0:8000")
        assert str(info.value) == f"api_base {_UNSPECIFIED_HINT}"
        assert sent == []


class TestShipJudgeModelFlag:
    @pytest.fixture
    def out(self, monkeypatch) -> io.StringIO:
        from rich.console import Console

        import soup_cli.commands.ship as ship

        buf = io.StringIO()
        monkeypatch.setattr(ship, "console", Console(file=buf, width=400, color_system=None))
        return buf

    @pytest.mark.parametrize("host", _NON_PUBLIC_HOSTS)
    def test_refused_as_a_usage_error(self, host, out):
        import click

        from soup_cli.commands.ship import _validate_judge_model_url

        with pytest.raises(click.exceptions.Exit) as info:
            _validate_judge_model_url(f"https://{host}/m")
        assert info.value.exit_code == 3
        assert f"--judge-model: {_REFUSED}" in out.getvalue()

    @pytest.mark.parametrize("host", _ALLOWED_HOSTS)
    def test_allowed(self, host, out):
        from soup_cli.commands.ship import _validate_judge_model_url

        _validate_judge_model_url(f"https://{host}/m")
        assert out.getvalue() == ""

    def test_refused_before_any_model_loads(self, monkeypatch, tmp_path):
        """Through the real command: the judge URL is a usage error before the base
        and tuned models are built, not after."""
        import json

        from typer.testing import CliRunner

        from soup_cli.commands import ship as ship_cmd
        from soup_cli.utils import live_eval

        loads: list[str] = []

        def _factory(model_id, **_kwargs):
            loads.append(str(model_id))
            return lambda prompt: ""

        monkeypatch.setattr(live_eval, "make_generator", _factory)
        monkeypatch.chdir(tmp_path)
        row = {"prompt": "p", "expected": "x", "scoring": "contains"}
        (tmp_path / "tasks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        res = CliRunner().invoke(
            ship_cmd.app,
            [
                "--base", "fake-base", "--adapter", "fake-adapter",
                "--task-eval", "tasks.jsonl", "--task-mode", "judge_score",
                "--judge-model", "https://10.0.0.1/m",
            ],
        )
        assert res.exit_code == 3, (res.output, repr(res.exception))
        assert f"--judge-model: {_REFUSED}" in _plain(res.output)
        assert loads == []


class TestChatProxy:
    @pytest.fixture
    def post(self, sent):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.ui.app import create_app, get_auth_token

        client = TestClient(create_app())
        headers = {"Authorization": f"Bearer {get_auth_token()}"}

        def _post(endpoint: str):
            body = {"messages": [{"role": "user", "content": "hi"}], "endpoint": endpoint}
            return client.post("/api/chat/send", json=body, headers=headers)

        return _post

    @pytest.mark.parametrize("host", _NON_PUBLIC_HOSTS)
    def test_refused_with_400_before_dispatch(self, host, post, sent):
        resp = post(f"https://{host}")
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert detail == f"endpoint: {_REFUSED} (SSRF protection); {_REMEDY}"
        assert sent == []

    @pytest.mark.parametrize("host", _ALLOWED_HOSTS)
    def test_allowed_endpoint_is_dispatched(self, host, post, sent):
        resp = post(f"https://{host}")
        assert resp.status_code == 200, resp.text
        assert [url for url, _ in sent] == [f"https://{host}/v1/chat/completions"]

    def test_unspecified_address_over_http_is_refused(self, post, sent):
        """0.0.0.0 is the bind-any wildcard; #1549 says to use loopback instead."""
        resp = post("http://0.0.0.0:8000")
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == f"endpoint {_UNSPECIFIED_HINT}"
        assert sent == []

    def test_loopback_range_over_http_is_dispatched(self, post, sent):
        resp = post("http://127.0.0.5:8000")
        assert resp.status_code == 200, resp.text
        assert [url for url, _ in sent] == ["http://127.0.0.5:8000/v1/chat/completions"]

    @pytest.mark.parametrize(
        "base", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"]
    )
    def test_loopback_spellings_over_http_are_dispatched(self, base, post, sent):
        resp = post(base)
        assert resp.status_code == 200, resp.text
        assert [url for url, _ in sent] == [f"{base}/v1/chat/completions"]

    def test_localhost_lookalike_over_http_is_refused(self, post, sent):
        resp = post("http://localhost.attacker.example:8000")
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "HTTP only allowed for localhost endpoints"
        assert sent == []

    @pytest.mark.parametrize("endpoint", ["https://[fd00::1/m", "https://[10.0.0.1]/m"])
    def test_malformed_endpoint_is_a_bad_request(self, endpoint, post, sent):
        """An unbalanced or non-IPv6 bracketed host is the client's error (400), not
        the server's (500)."""
        resp = post(endpoint)
        assert resp.status_code == 400, resp.text
        assert sent == []


_ODPO_YAML = (
    "base: sshleifer/tiny-gpt2\ntask: online_dpo\ndata:\n  train: x.jsonl\n"
    'training:\n  online_dpo_judge: "{judge}"\n'
)


class TestConfigLoad:
    """A shared soup.yaml is refused at load, before any model download."""

    @pytest.mark.parametrize(
        "judge",
        [
            "https://10.0.0.1/m",
            "https://[fd00::1]:8443/m",
            "https://167772161/m",
            "http://169.254.169.254/m",
        ],
    )
    def test_non_public_judge_literal_is_refused(self, judge):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError) as info:
            load_config_from_string(_ODPO_YAML.format(judge=judge))
        assert "online_dpo_judge" in str(info.value)
        assert _REFUSED in str(info.value)

    @pytest.mark.parametrize(
        "judge",
        [
            "https://judge.example.com/m",
            "https://127.0.0.1:8443/m",
            "http://localhost:8000/m",
            "ollama://llama3.1",
        ],
    )
    def test_other_judges_still_load(self, judge):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_ODPO_YAML.format(judge=judge))
        assert cfg.training.online_dpo_judge == judge

    def test_an_unparsable_judge_url_names_the_setting(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="online_dpo_judge is not a valid URL"):
            TrainingConfig(online_dpo_judge="https://[::1/m")


class TestCliEntryPoints:
    """The flags reach the same checks through the real commands."""

    def test_data_generate_refuses_a_private_api_base(self, sent, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app,
            [
                "data", "generate", "--prompt", "x", "--provider", "server",
                "--api-base", "https://10.0.0.1/v1", "--count", "1", "--output", "out.jsonl",
            ],
        )
        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert f"api_base: {_REFUSED}" in _plain(res.output)
        assert sent == []
        assert not (tmp_path / "out.jsonl").exists()

    def test_eval_judge_refuses_a_private_api_base(self, sent, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "t.jsonl").write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")
        res = CliRunner().invoke(
            app, ["eval", "judge", "--target", "t.jsonl", "--api-base", "https://10.0.0.1/v1"]
        )
        assert res.exit_code == 1, (res.output, repr(res.exception))
        assert f"judge URL: {_REFUSED}" in _plain(res.output)
        assert sent == []


class TestTheSharedHelper:
    @pytest.mark.parametrize(
        "host",
        [
            None, "", "localhost", "localhost.", "127.0.0.1", "::1", "[::1]", "127.1",
            "2130706433", "::ffff:127.0.0.1", "api.example.com", "8.8.8.8",
            "2606:4700::1111",
        ],
    )
    def test_passes(self, host):
        from soup_cli.utils.net_guard import refuse_private_ip_literal

        refuse_private_ip_literal(host, label="x")

    @pytest.mark.parametrize(
        "host",
        [
            "10.0.0.1", " 10.0.0.1", "[fd00::1]", "FE80::1", "fe80::1%eth0", "169.254.169.254.",
            "0", "::",
        ],
    )
    def test_refuses_with_the_label_first(self, host):
        from soup_cli.utils.net_guard import refuse_private_ip_literal

        with pytest.raises(ValueError) as info:
            refuse_private_ip_literal(host, label="vLLM URL")
        assert str(info.value) == f"vLLM URL: {_REFUSED} (SSRF protection); {_REMEDY}"

    def test_the_unspecified_hint_is_one_constant(self, monkeypatch):
        """#1549: every 0.0.0.0 refusal reuses the one hint in ``net_guard``."""
        from soup_cli.utils import hf
        from soup_cli.utils.hubs import validate_hub_endpoint
        from soup_cli.utils.net_guard import UNSPECIFIED_HOST_HINT
        from soup_cli.utils.webhooks import validate_webhook_url

        assert UNSPECIFIED_HOST_HINT == _UNSPECIFIED_HINT
        monkeypatch.setenv("HF_ENDPOINT", "http://0.0.0.0:8080")
        with pytest.raises(ValueError) as info:
            hf.resolve_endpoint()
        assert str(info.value) == f"HF_ENDPOINT {UNSPECIFIED_HOST_HINT}"
        with pytest.raises(ValueError) as info:
            validate_webhook_url("http://0.0.0.0:8080/hook")
        assert str(info.value) == f"webhook URL {UNSPECIFIED_HOST_HINT}"
        with pytest.raises(ValueError) as info:
            validate_hub_endpoint("http://0.0.0.0:8080", hub="modelscope")
        assert str(info.value) == f"modelscope {UNSPECIFIED_HOST_HINT}"

    def test_mapped_loopback_does_not_depend_on_the_interpreter(self, monkeypatch):
        """Whether ``IPv6Address.is_loopback`` looks through an IPv4-mapped address
        depends on the interpreter's patch release, so the helper unwraps it
        itself. Simulate an interpreter that does not."""
        import ipaddress

        from soup_cli.utils.net_guard import refuse_private_ip_literal

        monkeypatch.setattr(
            ipaddress.IPv6Address, "is_loopback", property(lambda self: self._ip == 1)
        )
        assert not ipaddress.ip_address("::ffff:127.0.0.1").is_loopback
        refuse_private_ip_literal("::ffff:127.0.0.1", label="x")
        with pytest.raises(ValueError, match=re.escape(_REFUSED)):
            refuse_private_ip_literal("::ffff:10.0.0.1", label="x")

    @pytest.mark.parametrize(
        "host",
        ["\u00e9" * 64 + ".example", "\u00e9..example"],
        ids=["label-too-long", "empty-label"],
    )
    def test_a_non_ascii_host_that_does_not_encode_is_a_hostname(self, host):
        """No client can dial a host that does not IDNA-encode, and the check must not
        raise on one either: it is treated as a hostname."""
        from soup_cli.utils.net_guard import is_private_or_link_local, refuse_private_ip_literal

        refuse_private_ip_literal(host, label="x")
        assert is_private_or_link_local(host) is False


class TestTheSharedPredicate:
    """The predicate also backs webhooks, OTLP, telemetry, ``soup ingest --pull``
    and the hub endpoints, so the two ranges it missed close for them too."""

    @pytest.mark.parametrize(
        "host",
        [
            "100.64.0.0", "100.100.100.200", "100.127.255.255", "::ffff:100.64.0.1",
            "fec0::1", "feff:ffff::1",
        ],
    )
    def test_non_public(self, host):
        from soup_cli.utils.net_guard import is_private_or_link_local

        assert is_private_or_link_local(host) is True

    @pytest.mark.parametrize(
        "host", ["100.63.255.255", "100.128.0.0", "8.8.8.8", "2606:4700::1111", "2001:4860::8888"]
    )
    def test_public(self, host):
        from soup_cli.utils.net_guard import is_private_or_link_local

        assert is_private_or_link_local(host) is False

    def test_webhook_url(self):
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="private/link-local/reserved"):
            validate_webhook_url("https://100.64.0.1/hook")

    def test_otlp_endpoint(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="private"):
            validate_otlp_endpoint("https://100.64.0.1:4317")

    def test_telemetry_endpoint(self):
        from soup_cli.utils.trackers import _telemetry_endpoint_is_safe

        assert _telemetry_endpoint_is_safe("https://100.64.0.1/i/v0/e/") is False

    def test_ingest_pull_needs_the_private_host_opt_in(self):
        from soup_cli.utils.ingest_pull import _is_private

        assert _is_private("https://100.64.0.1") is True

    @pytest.mark.parametrize(
        "host", ["10\u30020\u30020\u30021", "\uff11\uff10.0.0.1", "1\u00ad0.0.0.1"]
    )
    def test_non_ascii_spellings_are_classified(self, host):
        from soup_cli.utils.net_guard import is_private_or_link_local

        assert is_private_or_link_local(host) is True

    @pytest.mark.parametrize("host", ["8\u30028\u30028\u30028", "m\u00fcnchen.example"])
    def test_non_ascii_public_hosts_stay_public(self, host):
        from soup_cli.utils.net_guard import is_private_or_link_local

        assert is_private_or_link_local(host) is False

    def test_webhook_url_with_ideographic_stops(self):
        from soup_cli.utils.webhooks import validate_webhook_url

        with pytest.raises(ValueError, match="private/link-local/reserved"):
            validate_webhook_url("https://10\u30020\u30020\u30021/hook")

    def test_otlp_endpoint_with_fullwidth_stops(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="private"):
            validate_otlp_endpoint("https://10\uff0e0\uff0e0\uff0e1:4317")

    def test_hub_endpoint_counts_shared_space_as_private(self):
        from soup_cli.utils.hubs import validate_hub_endpoint

        with pytest.raises(ValueError, match="private/link-local hosts require HTTPS"):
            validate_hub_endpoint("http://100.64.0.1", hub="modelscope")

    def test_hf_endpoint_counts_shared_space_as_private(self, monkeypatch):
        from soup_cli.utils import hf

        monkeypatch.setenv("HF_ENDPOINT", "http://100.64.0.1")
        with pytest.raises(ValueError, match="private/link-local hosts require HTTPS"):
            hf.resolve_endpoint()

    def test_ingest_pull_with_ideographic_stops(self):
        from soup_cli.utils.ingest_pull import _is_private

        assert _is_private("https://10\u30020\u30020\u30021") is True


class TestWhatTheClientConnectsTo:
    """The checks read ``urlparse(url).hostname``; httpx connects to
    ``httpx.URL(url).host``, which it IDNA-encodes first. Whenever the host
    httpx would dial is a non-public, non-loopback IP literal, the check must
    refuse the URL, and when it is public or loopback, the check must not."""

    @pytest.mark.parametrize("separator", [".", "\u3002", "\uff0e", "\uff61"])
    @pytest.mark.parametrize(
        "address",
        [
            "10.0.0.1", "192.168.1.10", "169.254.169.254", "100.64.0.1", "0xa.0.0.1",
            "10.1", "127.0.0.1", "127.0.0.5", "8.8.8.8",
        ],
    )
    def test_the_check_agrees_with_the_client(self, separator, address):
        httpx = pytest.importorskip("httpx")
        from urllib.parse import urlparse

        from soup_cli.utils.net_guard import (
            is_private_or_link_local,
            parse_ip_literal,
            refuse_private_ip_literal,
        )

        url = f"https://{address.replace('.', separator)}/v1"
        try:
            dialled = httpx.URL(url).host
        except httpx.InvalidURL:
            pytest.skip("this httpx refuses the spelling outright, so it cannot dial it")
        assert dialled.isascii(), dialled
        target = parse_ip_literal(dialled)
        assert target is not None, dialled
        hostname = urlparse(url).hostname
        if is_private_or_link_local(dialled) and not target.is_loopback:
            with pytest.raises(ValueError, match=re.escape(_REFUSED)):
                refuse_private_ip_literal(hostname, label="x")
        else:
            refuse_private_ip_literal(hostname, label="x")
