"""Data-parallel layer streaming over one TCP socket (B1): the transport, the launch, the runs.

B1 is for machines that cannot reach each other's addresses (vast.ai containers behind NAT):
each node runs a plain, NON-distributed ``soup train`` with ``SOUP_STREAM_DP_*`` set, HF is not
distributed, Soup shards the rows by rank, and ``utils/adapter_grad_transport.py`` reduces the
LoRA gradients through a star of TCP connections to rank 0. What is under test:

* ``TestB1Env`` / ``TestDecision`` / ``TestTrainCommand``: the environment is validated at startup
  and every refusal names the variable or the blocking setting (no process group, no model).
* ``TestShardRows``: each rank's rows, cut to equal lengths, and the union per step.
* ``TestHandshake`` / ``TestObjects`` / ``TestWireBounds`` / ``TestAllReduce`` /
  ``TestBroadcast``: the group on 127.0.0.1 in one process (a thread per rank). Handshake
  refusals by name, JSON objects, every declared length checked before it is read, and the
  weighted all-reduce against its formula ``sum_r(w_r g_r) / sum_r(w_r)``, bitwise equal on every
  rank and in rank order.
* ``TestPeerDeath``: real processes, no torch: a crashed peer in both directions, a silent peer
  cut off at the shortened timeout.
* ``TestGatesUnit``: the pieces around the group, in process: the token weight the trainer hands
  the group, the timeout and the metrics file of ``_finish_distributed_stream_setup``, the rank
  gates of the callback, the tracker, the tokenizer and the saves.
* ``TestEquivalence`` and friends (T1, T1a, T1b, T1c for B1): REAL processes, one per rank, on
  127.0.0.1, each running the production setup (``SFTTrainerWrapper.setup`` -> the group, the
  sharding, the init check, the SFT trainer class on a tiny Llama) and then ``wrapper.train``
  with transformers' own loop, against ONE process with twice the per-device batch on the union
  of the rows. Five SGD steps, fp32, ``lora_dropout=0``, a sequential sampler; the adapters are
  compared after every step. The two ranks' adapters are BITWISE equal (one reduced gradient,
  a deterministic optimizer); they equal the one-rank adapters within ``atol=1e-6, rtol=1e-5``.
  SGD and not Adam, because Adam divides by its own running gradient scale, so a gradient that is
  too large would yield nearly the same update.
* ``TestRankZeroOnly`` / ``TestPeerDeathInARun``: which rank wrote what, and a rank that dies or
  goes silent in the middle of a run.

Why a plain mean would be WRONG here, unlike under ``torch.distributed``: there HF gathers
``num_items_in_batch`` over the ranks and multiplies the loss by the number of processes, so
the mean of the rank gradients is the union gradient. Under B1 each non-distributed HF divides by
its OWN window token count ``N_r``, so the union gradient is ``sum_r(N_r g_r) / sum_r(N_r)``;
``TestRaggedRanks`` gives the ranks different token counts, which is where the two differ.

The multi-process tests need torch, transformers, peft, trl, datasets, accelerate and safetensors
and skip without them (and on macOS, where two-process runs can time out on loaded runners). Each
process has its own shard cache and tracker database and ``CUDA_VISIBLE_DEVICES=-1``; the worker
is ``_stream_dp_tcp_worker.py``. CLI output is compared through an ANSI-strip + whitespace-collapse
helper.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from soup_cli.utils import adapter_grad_transport as transport
from soup_cli.utils.adapter_grad_transport import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    ENV_ADDR,
    ENV_BIND_PORT,
    ENV_RANK,
    ENV_TOKEN,
    ENV_WORLD_SIZE,
    MAX_OBJECT_BYTES,
    B1ConfigError,
    GradGroup,
    GradTransportError,
    HandshakeError,
    PeerClosedError,
    PeerTimeoutError,
    ProtocolError,
    TcpStarGroup,
    TorchDistributedGroup,
    address_is_public,
    b1_rank_or_none,
    read_b1_env,
    refuse_b1_without_streaming,
    require_token_for_public_address,
    resolve_group,
)
from soup_cli.utils.stream_dp_runtime import shard_rows

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

_WORKER = Path(__file__).resolve().with_name("_stream_dp_tcp_worker.py")
_TIMEOUT_SECONDS = 420
_STEPS = 5
_ATOL = 1e-6
_RTOL = 1e-5
_GROUP_TIMEOUT = 123
_TAGS = ("b0", "b1")


def _plain(text: str | None) -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(_ANSI.sub("", text or "").replace("│", " ").split())


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --------------------------------------------------------------------------
# the launch: SOUP_STREAM_DP_* is validated at startup, every refusal by name
# --------------------------------------------------------------------------

_TOKEN = "shared-secret-0123456789"
_RANK_0 = {ENV_RANK: "0", ENV_WORLD_SIZE: "2", ENV_BIND_PORT: "41000"}
# 203.0.113.7 is a public address (the documentation range is not private), so rank 1 carries
# the token the launch then requires; _RANK_1_NO_TOKEN is the same launch without it
_RANK_1 = {
    ENV_RANK: "1",
    ENV_WORLD_SIZE: "2",
    ENV_ADDR: "203.0.113.7:41123",
    ENV_TOKEN: _TOKEN,
}
_RANK_1_NO_TOKEN = {name: value for name, value in _RANK_1.items() if name != ENV_TOKEN}


class TestB1Env:
    def test_without_any_variable_b1_is_off(self):
        assert read_b1_env({}) is None
        # the torchrun variables are not B1's
        assert read_b1_env({"WORLD_SIZE": "2", "RANK": "1", "MASTER_ADDR": "x"}) is None

    def test_rank_0_listens_on_its_bind_port(self):
        env = read_b1_env(_RANK_0)
        assert (env.rank, env.world_size, env.bind_port) == (0, 2, 41000)
        assert (env.host, env.port, env.token) == (None, None, None)
        assert env.bind_host == "0.0.0.0", "a vast port mapping reaches the container's address"

    def test_a_peer_reaches_rank_0_at_the_address(self):
        env = read_b1_env(_RANK_1)
        assert (env.rank, env.world_size) == (1, 2)
        assert (env.host, env.port, env.bind_port) == ("203.0.113.7", 41123, None)
        assert env.token == _TOKEN

    def test_the_other_variable_is_optional_but_checked_when_present(self):
        env = read_b1_env({**_RANK_0, ENV_ADDR: "node0.example.com:41123", ENV_TOKEN: _TOKEN})
        assert (env.host, env.port, env.bind_port) == ("node0.example.com", 41123, 41000)
        env = read_b1_env({**_RANK_1, ENV_BIND_PORT: "41000"})
        assert env.bind_port == 41000
        with pytest.raises(B1ConfigError, match=re.escape(f"{ENV_BIND_PORT}='abc' is not valid")):
            read_b1_env({**_RANK_1, ENV_BIND_PORT: "abc"})

    @pytest.mark.parametrize(
        ("address", "host", "port"),
        [
            ("localhost:1", "localhost", 1),
            ("10.0.0.1:65535", "10.0.0.1", 65535),
            ("node-0.region.example.com:41123", "node-0.region.example.com", 41123),
            ("[::1]:5000", "[::1]", 5000),
            ("[2001:db8::7]:41123", "[2001:db8::7]", 41123),
        ],
    )
    def test_every_form_of_an_address_that_is_valid(self, address, host, port):
        env = read_b1_env({**_RANK_1, ENV_ADDR: address})
        assert (env.host, env.port) == (host, port)

    def test_the_largest_world_is_accepted_and_the_next_is_not(self):
        assert read_b1_env({**_RANK_0, ENV_WORLD_SIZE: "64"}).world_size == 64
        with pytest.raises(B1ConfigError, match=r"SOUP_STREAM_DP_WORLD_SIZE=65 is out of range"):
            read_b1_env({**_RANK_0, ENV_WORLD_SIZE: "65"})

    def test_a_token_is_carried(self):
        token = "t" * 16
        assert read_b1_env({**_RANK_0, ENV_TOKEN: token}).token == token

    def test_it_is_a_value_error_so_every_caller_that_catches_refusals_sees_it(self):
        assert issubclass(B1ConfigError, ValueError)

    @pytest.mark.parametrize(
        ("env", "fragment"),
        [
            (
                {ENV_RANK: "0", ENV_BIND_PORT: "41000"},
                f"{ENV_WORLD_SIZE} is not set, but {ENV_BIND_PORT}, {ENV_RANK} are set",
            ),
            (
                {ENV_WORLD_SIZE: "2", ENV_ADDR: "h:1"},
                f"{ENV_RANK} is not set, but {ENV_ADDR}, {ENV_WORLD_SIZE} are set",
            ),
            (
                {ENV_ADDR: "h:1"},
                f"{ENV_RANK}, {ENV_WORLD_SIZE} are not set, but {ENV_ADDR} is set",
            ),
            (
                {ENV_TOKEN: "t" * 16},
                f"{ENV_RANK}, {ENV_WORLD_SIZE} are not set, but {ENV_TOKEN} is set",
            ),
            ({**_RANK_1, ENV_ADDR: None}, f"{ENV_ADDR} is not set on rank 1"),
            ({**_RANK_0, ENV_BIND_PORT: None}, f"{ENV_BIND_PORT} is not set on rank 0"),
            (
                {**_RANK_0, ENV_RANK: "2"},
                f"{ENV_RANK}=2 is not below {ENV_WORLD_SIZE}=2: ranks are numbered from 0 to 1",
            ),
            ({**_RANK_0, ENV_WORLD_SIZE: "1"}, f"{ENV_WORLD_SIZE}=1 is out of range"),
            ({**_RANK_0, ENV_WORLD_SIZE: "0"}, f"{ENV_WORLD_SIZE}=0 is out of range"),
            ({**_RANK_0, ENV_BIND_PORT: "0"}, f"{ENV_BIND_PORT}=0 is out of range"),
            ({**_RANK_0, ENV_BIND_PORT: "65536"}, f"{ENV_BIND_PORT}=65536 is out of range"),
        ],
        ids=[
            "no-world",
            "no-rank",
            "only-addr",
            "only-token",
            "peer-without-addr",
            "root-without-bind-port",
            "rank-not-below-world",
            "world-1",
            "world-0",
            "bind-port-0",
            "bind-port-65536",
        ],
    )
    def test_each_problem_is_refused_by_name(self, env, fragment):
        env = {name: value for name, value in env.items() if value is not None}
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env(env)
        assert fragment in str(excinfo.value), str(excinfo.value)

    @pytest.mark.parametrize("value", ["abc", "-1", "1.0", " 1", "1 ", "", "+1", "１", "0x1"])
    @pytest.mark.parametrize("name", [ENV_RANK, ENV_WORLD_SIZE, ENV_BIND_PORT])
    def test_a_value_that_is_not_a_plain_decimal_integer_is_refused(self, name, value):
        env = {**_RANK_0, name: value}
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env(env)
        assert f"{name}={value!r} is not valid: it must be a plain decimal integer" in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        "address",
        [
            "nocolon",
            ":123",
            "host:",
            "host:abc",
            "http://host:123",
            "ho st:123",
            "host:123\n",
            "host\x1b[31m:123",
            "[::1:123",
            "ho_st:1",
            "héllo:1",
            "a" * 300 + ":1",
            "",
        ],
    )
    def test_an_address_that_is_not_host_and_port_is_refused(self, address):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1, ENV_ADDR: address})
        message = str(excinfo.value)
        assert f"{ENV_ADDR}=" in message or "the port of " in message, message
        assert "is not valid" in message
        assert "\x1b" not in message, "the text of the variable reached the message unfiltered"

    @pytest.mark.parametrize("port", ["0", "65536", "99999"])
    def test_a_port_of_the_address_out_of_range_is_refused(self, port):
        pattern = r"the port of SOUP_STREAM_DP_ADDR=\d+ is out of range"
        with pytest.raises(B1ConfigError, match=pattern):
            read_b1_env({**_RANK_1, ENV_ADDR: f"host:{port}"})

    @pytest.mark.parametrize("token", ["", "short", "x" * 15, "x" * 257, "a" * 16 + "\n"])
    def test_a_token_of_the_wrong_length_is_refused_without_showing_it(self, token):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_0, ENV_TOKEN: token})
        assert f"{ENV_TOKEN} must be 16 to 256 characters" in str(excinfo.value)
        assert "the value is not shown" in str(excinfo.value)
        assert token.strip() == "" or token.strip() not in str(excinfo.value)


def _needs_token(what: str) -> str:
    """The refusal of a public address without SOUP_STREAM_DP_TOKEN, spelled out in full."""
    return (
        f"SOUP_STREAM_DP_TOKEN is required when {what} is a public internet address and is not "
        "set: anyone who can reach the port could join the group. Set the same "
        "SOUP_STREAM_DP_TOKEN (16 to 256 characters) on every node. It is optional only for "
        "loopback, private (RFC 1918, fc00::/7) and link-local addresses and for localhost."
    )


_PUBLIC_HOSTS = [
    "203.0.113.7",  # the documentation range is not private
    "8.8.8.8",
    "11.0.0.1",  # just above 10/8
    "172.15.255.255",  # just below 172.16/12
    "172.32.0.1",  # just above 172.16/12
    "192.169.0.1",  # just above 192.168/16
    "169.253.0.1",  # just below 169.254/16
    "100.64.0.1",  # shared (carrier-grade NAT) space is not on the list
    "0.0.0.0",
    "[2001:db8::7]",
    "[2606:4700::1111]",
    "[fec0::1]",  # site-local, deprecated, not unique-local
    "[::]",
    "[::ffff:8.8.8.8]",  # an IPv4-mapped address is judged by its IPv4 half
    "node0.example.com",  # a name that is not an IP literal is public: it is not resolved
    "localhost.example.com",
    "my-localhost",
]
_PRIVATE_HOSTS = [
    "127.0.0.1",
    "127.255.255.254",
    "10.0.0.1",
    "10.255.255.255",
    "172.16.0.1",
    "172.31.255.255",
    "192.168.0.1",
    "192.168.255.254",
    "169.254.0.1",
    "169.254.169.254",
    "[::1]",
    "[fc00::1]",
    "[fd12:3456:789a::1]",
    "[fe80::1]",
    "[::ffff:10.0.0.1]",
    "[::ffff:127.0.0.1]",
    "localhost",
    "LocalHost",
]


class TestPublicAddressNeedsAToken:
    """SOUP_STREAM_DP_TOKEN is required on a public address and optional on a private one."""

    @pytest.mark.parametrize("host", _PUBLIC_HOSTS)
    def test_a_public_address_is_public(self, host):
        assert address_is_public(host) is True

    @pytest.mark.parametrize("host", _PRIVATE_HOSTS)
    def test_a_loopback_private_or_link_local_address_is_not(self, host):
        assert address_is_public(host) is False

    @pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "localhost.", "[::1]", "::1"])
    def test_localhost_and_a_bare_or_bracketed_ipv6_literal_are_read_the_same(self, host):
        assert address_is_public(host) is False

    def test_a_bare_ipv6_literal_is_judged_like_the_bracketed_one(self):
        assert address_is_public("2001:db8::7") is True
        assert address_is_public("[2001:db8::7]") is True
        assert address_is_public("fd12::1") is False

    @pytest.mark.parametrize("host", _PUBLIC_HOSTS)
    def test_a_peer_with_a_public_address_and_no_token_is_refused_by_name(self, host):
        address = f"{host}:41123"
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: address})
        # the address reaches the message through for_terminal, which also escapes what Rich
        # would read as a markup tag: "[fec0::1]" is shown as "\[fec0::1]"
        shown = address.replace("[fec0", "\\[fec0")
        assert str(excinfo.value) == _needs_token(f"{ENV_ADDR}={shown}")

    def test_both_families_of_a_public_literal_are_refused_with_the_address_in_the_text(self):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: "198.51.100.9:7"})
        assert str(excinfo.value) == _needs_token("SOUP_STREAM_DP_ADDR=198.51.100.9:7")
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: "[2001:db8::7]:7"})
        assert str(excinfo.value) == _needs_token("SOUP_STREAM_DP_ADDR=[2001:db8::7]:7")

    def test_rank_0_with_a_public_address_and_no_token_is_refused_too(self):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_0, ENV_ADDR: "203.0.113.7:41123"})
        assert str(excinfo.value) == _needs_token("SOUP_STREAM_DP_ADDR=203.0.113.7:41123")

    @pytest.mark.parametrize("host", _PRIVATE_HOSTS)
    def test_a_private_address_works_without_a_token(self, host):
        env = read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: f"{host}:41123"})
        assert env.token is None
        assert env.host == host

    @pytest.mark.parametrize("host", _PUBLIC_HOSTS)
    def test_a_public_address_with_a_token_passes_the_gate(self, host):
        env = read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: f"{host}:41123", ENV_TOKEN: _TOKEN})
        assert (env.host, env.token) == (host, _TOKEN)

    def test_rank_0_without_an_address_listens_on_every_interface_without_a_gate(self):
        # nothing names an address to judge: the documented case, and the token stays optional
        env = read_b1_env(_RANK_0)
        assert (env.host, env.token, env.bind_host) == (None, None, "0.0.0.0")

    def test_a_malformed_token_is_refused_for_what_it_is_before_the_address_is_judged(self):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1_NO_TOKEN, ENV_TOKEN: "short"})
        assert str(excinfo.value).startswith(f"{ENV_TOKEN} must be 16 to 256 characters")

    def test_a_malformed_address_is_still_refused_as_malformed(self):
        with pytest.raises(B1ConfigError) as excinfo:
            read_b1_env({**_RANK_1_NO_TOKEN, ENV_ADDR: "http://203.0.113.7:1"})
        assert "is not valid" in str(excinfo.value)

    def test_the_refusal_does_not_echo_terminal_escapes(self):
        with pytest.raises(B1ConfigError) as excinfo:
            TcpStarGroup(rank=1, world_size=2, host="evil\x1b[31m.example.com", port=1)
        assert "evil" in str(excinfo.value)
        assert "\x1b" not in str(excinfo.value)

    def test_the_helper_judges_only_the_address_and_treats_an_empty_token_as_none(self):
        require_token_for_public_address("203.0.113.7", _TOKEN, what="x")
        require_token_for_public_address("10.0.0.1", None, what="x")
        with pytest.raises(B1ConfigError) as excinfo:
            require_token_for_public_address("203.0.113.7", None, what="x")
        assert str(excinfo.value) == _needs_token("x")
        with pytest.raises(B1ConfigError) as excinfo:
            require_token_for_public_address("203.0.113.7", "", what="x")
        assert str(excinfo.value) == _needs_token("x")


class TestPublicAddressNeedsATokenInTheGroup:
    """The same gate in ``TcpStarGroup``, so a direct caller cannot skip it. Both fire before
    any socket is opened: the two methods that open one are replaced by a tripwire."""

    @pytest.fixture
    def no_socket(self, monkeypatch):
        opened = []

        def tripwire(self, *args, **kwargs):
            opened.append(args)
            raise AssertionError("a socket was opened before the token gate")

        monkeypatch.setattr(TcpStarGroup, "_connect_to_root", tripwire)
        monkeypatch.setattr(TcpStarGroup, "_accept_peers", tripwire)
        return opened

    @pytest.fixture
    def sockets_are_stubbed(self, monkeypatch):
        reached = []

        def note(self, *args, **kwargs):
            reached.append(args)

        monkeypatch.setattr(TcpStarGroup, "_connect_to_root", note)
        monkeypatch.setattr(TcpStarGroup, "_accept_peers", note)
        return reached

    @pytest.mark.parametrize("host", ["203.0.113.7", "[2001:db8::7]", "node0.example.com"])
    def test_a_peer_aimed_at_a_public_address_without_a_token_is_refused_first(
        self, host, no_socket
    ):
        with pytest.raises(B1ConfigError) as excinfo:
            TcpStarGroup(rank=1, world_size=2, host=host, port=41123)
        assert str(excinfo.value) == _needs_token(f"host {host}")
        assert no_socket == []

    def test_an_empty_token_is_no_token(self, no_socket):
        with pytest.raises(B1ConfigError) as excinfo:
            TcpStarGroup(rank=1, world_size=2, host="203.0.113.7", port=1, token="")
        assert str(excinfo.value) == _needs_token("host 203.0.113.7")
        assert no_socket == []

    def test_rank_0_bound_to_one_public_address_without_a_token_is_refused_first(
        self, no_socket
    ):
        with pytest.raises(B1ConfigError) as excinfo:
            TcpStarGroup(rank=0, world_size=2, bind_host="203.0.113.5", bind_port=41000)
        assert str(excinfo.value) == _needs_token("bind_host 203.0.113.5")
        assert no_socket == []

    @pytest.mark.parametrize("bind_host", ["0.0.0.0", "::", "127.0.0.1", "10.1.2.3", "::1"])
    def test_rank_0_on_every_interface_or_a_private_one_needs_no_token(
        self, bind_host, sockets_are_stubbed
    ):
        TcpStarGroup(rank=0, world_size=2, bind_host=bind_host, bind_port=41000)
        assert len(sockets_are_stubbed) == 1

    @pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.1", "192.168.1.5", "[::1]", "localhost"])
    def test_a_peer_aimed_at_a_private_address_needs_no_token(self, host, sockets_are_stubbed):
        TcpStarGroup(rank=1, world_size=2, host=host, port=41123)
        assert len(sockets_are_stubbed) == 1

    def test_a_public_address_with_a_token_reaches_the_socket(self, sockets_are_stubbed):
        TcpStarGroup(rank=1, world_size=2, host="203.0.113.7", port=41123, token=_TOKEN)
        TcpStarGroup(rank=0, world_size=2, bind_host="203.0.113.5", bind_port=41000, token=_TOKEN)
        assert len(sockets_are_stubbed) == 2


class TestStreamingIsRequired:
    def test_a_clean_environment_is_fine(self):
        refuse_b1_without_streaming({})

    @pytest.mark.parametrize("name", transport.B1_ENV_NAMES)
    def test_any_variable_without_stream_layers_is_refused_by_name(self, name):
        with pytest.raises(B1ConfigError) as excinfo:
            refuse_b1_without_streaming({name: "1"})
        message = str(excinfo.value)
        assert message.startswith(f"{name} is set, but training.stream_layers is not true"), message
        assert "exists only for layer streaming" in message

    def test_several_variables_are_all_named(self):
        with pytest.raises(B1ConfigError, match=rf"{ENV_RANK}, {ENV_WORLD_SIZE} are set, but"):
            refuse_b1_without_streaming({ENV_RANK: "0", ENV_WORLD_SIZE: "2"})

    def test_the_lenient_reader_names_the_chief_without_ever_raising(self):
        assert b1_rank_or_none({}) is None
        assert b1_rank_or_none({ENV_RANK: "0"}) == 0
        assert b1_rank_or_none({ENV_RANK: "3"}) == 3
        for bad in ("", "x", "-1", "1.5", "9" * 20):
            assert b1_rank_or_none({ENV_RANK: bad}) is None


# --------------------------------------------------------------------------
# the decision: the same blockers as the C-form, plus four, no LOCAL_WORLD_SIZE
# --------------------------------------------------------------------------


def _sft_cfg(tmp_path, *, training=None, data=None, task="sft"):
    """A valid streaming config; ``training`` / ``data`` are applied WITHOUT validation."""
    from soup_cli.config.loader import load_config_from_string

    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": str(tmp_path / "never-downloaded-base"),
                "task": task,
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": {
                    "batch_size": 2 if task == "kto" else 1,
                    "gradient_accumulation_steps": 1,
                    "quantization": "none",
                    "epochs": 1,
                    "stream_layers": True,
                    "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    updates = {}
    if training:
        updates["training"] = cfg.training.model_copy(update=training)
    if data:
        updates["data"] = cfg.data.model_copy(update=data)
    return cfg.model_copy(update=updates) if updates else cfg


@pytest.fixture
def b1_launch(monkeypatch):
    """Rank 0 of a two-node B1 launch: no WORLD_SIZE, no LOCAL_WORLD_SIZE."""
    for name in (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
        *transport.B1_ENV_NAMES,
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in _RANK_0.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


_B1_MESSAGE = (
    "training.stream_layers=true with SOUP_STREAM_DP_* set (data-parallel over TCP, rank 0 of 2): "
    "distributed layer streaming is not supported yet with {items}, see #2585. It is BETA for "
    "task='sft', without --deepspeed, --fsdp, --resume, --hf-resume, --push-as, an eval gate "
    "(--gate / training.eval_gate), training.multipack, training.moe_lora, training.packing, "
    "training.loss_watchdog, training.loss_spike_recovery, training.grad_accum_auto_tune or "
    "data.format='raft'. Remove the blocking setting, or unset the SOUP_STREAM_DP_* variables "
    "to train on one node."
)


class TestDecision:
    def test_the_allowed_slice_streams_with_no_launcher_variable(self, tmp_path, b1_launch):
        from soup_cli.trainer.stream_setup import (
            distributed_stream_blockers,
            refuse_if_distributed_launch,
        )

        cfg = _sft_cfg(tmp_path)
        assert distributed_stream_blockers(cfg, b1=True) == []
        assert refuse_if_distributed_launch(cfg) is True

    def test_without_b1_the_missing_launcher_variable_is_still_a_blocker(self, tmp_path, b1_launch):
        from soup_cli.trainer.stream_setup import distributed_stream_blockers

        assert distributed_stream_blockers(_sft_cfg(tmp_path)) == [
            "LOCAL_WORLD_SIZE unset or invalid (one rank per host cannot be confirmed)"
        ]

    def test_a_malformed_variable_is_refused_before_any_blocker(self, tmp_path, b1_launch):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        b1_launch.setenv(ENV_WORLD_SIZE, "two")
        with pytest.raises(ValueError, match=r"SOUP_STREAM_DP_WORLD_SIZE='two' is not valid"):
            refuse_if_distributed_launch(_sft_cfg(tmp_path, task="dpo"))

    @pytest.mark.parametrize("size", ["2", "8"])
    def test_b1_together_with_a_launcher_is_refused_naming_both(self, tmp_path, b1_launch, size):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        b1_launch.setenv("WORLD_SIZE", size)
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path))
        message = _plain(str(excinfo.value))
        assert message.startswith(
            f"{ENV_RANK}, {ENV_WORLD_SIZE}, {ENV_BIND_PORT} set together with WORLD_SIZE={size}:"
        ), message
        assert "two launches that must not be mixed" in message

    @pytest.mark.parametrize("value", ["1", "0", "", "abc"])
    def test_a_world_size_that_is_not_a_launch_does_not_conflict(self, tmp_path, b1_launch, value):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        b1_launch.setenv("WORLD_SIZE", value)
        assert refuse_if_distributed_launch(_sft_cfg(tmp_path)) is True

    @pytest.mark.parametrize(
        ("training", "item"),
        [
            ({"packing": True}, "training.packing"),
            ({"loss_watchdog": True}, "training.loss_watchdog"),
            (
                {"loss_watchdog": True, "loss_spike_recovery": True},
                "training.loss_watchdog, training.loss_spike_recovery",
            ),
            ({"grad_accum_auto_tune": True}, "training.grad_accum_auto_tune"),
            ({"multipack": True}, "training.multipack"),
            ({"moe_lora": True}, "training.moe_lora"),
        ],
    )
    def test_each_setting_that_would_make_the_nodes_disagree_is_named(
        self, tmp_path, b1_launch, training, item
    ):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path, training=training))
        assert _plain(str(excinfo.value)) == _B1_MESSAGE.format(items=item)

    def test_the_options_of_the_command_are_named_too(self, tmp_path, b1_launch):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        cfg = _sft_cfg(tmp_path)
        for kwargs, item in (
            ({"resume": "auto"}, "--resume"),
            ({"hf_resume": True}, "--hf-resume"),
            ({"push_as": "u/r"}, "--push-as"),
            ({"gate": "evals/gate.yaml"}, "--gate"),
            ({"deepspeed": "zero2"}, "--deepspeed"),
            ({"fsdp": "full_shard"}, "--fsdp"),
        ):
            with pytest.raises(ValueError) as excinfo:
                refuse_if_distributed_launch(cfg, **kwargs)
            assert _plain(str(excinfo.value)) == _B1_MESSAGE.format(items=item)

    def test_another_task_and_raft_are_refused_as_in_the_c_form(self, tmp_path, b1_launch):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path, task="dpo"))
        assert _plain(str(excinfo.value)) == _B1_MESSAGE.format(items="task='dpo'")
        raft = _sft_cfg(tmp_path, data={"format": "raft"})
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(raft)
        assert _plain(str(excinfo.value)) == _B1_MESSAGE.format(items="data.format='raft'")

    def test_the_c_form_message_is_unchanged_and_never_mentions_b1(self, tmp_path, monkeypatch):
        from soup_cli.trainer.stream_setup import refuse_if_distributed_launch

        for name in transport.B1_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        with pytest.raises(ValueError) as excinfo:
            refuse_if_distributed_launch(_sft_cfg(tmp_path), resume="auto")
        message = _plain(str(excinfo.value))
        assert "SOUP_STREAM_DP" not in message and "packing" not in message
        assert "not supported yet with --resume, see #2585" in message


# --------------------------------------------------------------------------
# the rows: each rank's share, cut to equal lengths
# --------------------------------------------------------------------------


class TestShardRows:
    def test_rank_r_takes_every_world_th_row_from_r(self):
        rows = list(range(12))
        assert shard_rows(rows, 0, 3)[0] == [0, 3, 6, 9]
        assert shard_rows(rows, 1, 3)[0] == [1, 4, 7, 10]
        assert shard_rows(rows, 2, 3)[0] == [2, 5, 8, 11]

    @pytest.mark.parametrize(("total", "world"), [(7, 2), (21, 2), (10, 3), (11, 4), (5, 5)])
    def test_every_rank_is_cut_to_the_shortest_so_all_run_the_same_steps(self, total, world):
        """Mutation target (the cut): without it rank 0 of 7 rows has 4 and rank 1 has 3."""
        rows = list(range(total))
        per_rank = total // world
        shards = [shard_rows(rows, rank, world) for rank in range(world)]
        assert [len(mine) for mine, _n, _d in shards] == [per_rank] * world
        assert {n for _m, n, _d in shards} == {per_rank}
        assert {d for _m, _n, d in shards} == {total - per_rank * world}

    def test_the_dropped_rows_are_the_trailing_ones_and_each_is_counted(self):
        rows = list(range(21))
        kept = sorted(row for rank in range(2) for row in shard_rows(rows, rank, 2)[0])
        assert kept == list(range(20)), "the one dropped row is the last"
        assert shard_rows(rows, 0, 2)[2] == 1

    def test_nothing_is_dropped_when_the_rows_divide(self):
        assert shard_rows(list(range(20)), 1, 2)[2] == 0

    @pytest.mark.parametrize(("world", "batch"), [(2, 1), (2, 3), (3, 2), (4, 1)])
    def test_with_a_sequential_sampler_a_step_holds_the_union_batch_of_one_rank(self, world, batch):
        """Step ``k`` of the ranks together holds rows ``W*k*b .. W*(k+1)*b``: the batch of
        ``W*b`` rows that ONE rank takes, which is what the equivalence tests compare with."""
        rows = list(range(world * batch * 4 + 1))
        shards = [shard_rows(rows, rank, world)[0] for rank in range(world)]
        for step in range(len(shards[0]) // batch):
            union = sorted(
                row for mine in shards for row in mine[step * batch : (step + 1) * batch]
            )
            assert union == rows[world * step * batch : world * (step + 1) * batch], step

    def test_too_few_rows_are_refused_by_name(self):
        with pytest.raises(ValueError, match=r"1 training row\(s\) cannot be sharded over 2 ranks"):
            shard_rows([1], 0, 2)
        with pytest.raises(ValueError, match=r"every rank needs at least one row"):
            shard_rows([], 0, 2)

    @pytest.mark.parametrize(("rank", "world"), [(2, 2), (-1, 2), (0, 0), (True, 2), (0, 2.0)])
    def test_a_rank_outside_the_world_is_a_bug_of_the_caller(self, rank, world):
        with pytest.raises(ValueError):
            shard_rows([1, 2, 3, 4], rank, world)

    def test_a_dataset_is_sharded_like_a_list(self):
        datasets = pytest.importorskip("datasets")
        rows = [{"id": index} for index in range(21)]
        dataset = datasets.Dataset.from_list(rows)
        for rank in range(2):
            mine, per_rank, dropped = shard_rows(dataset, rank, 2)
            assert (per_rank, dropped) == (10, 1)
            as_list = shard_rows(rows, rank, 2)[0]
            assert [row["id"] for row in mine] == [row["id"] for row in as_list]


# --------------------------------------------------------------------------
# the group in process: a thread per rank on 127.0.0.1
# --------------------------------------------------------------------------


def run_ranks(world, work, *, per_rank=None, join_seconds=90.0, **common):
    """Form a group of ``world`` ranks (a thread each) and run ``work(group, rank)`` on every one.

    Returns ``SimpleNamespace(results, errors, groups)``, each keyed by rank. A construction
    failure is in ``errors`` too. ``per_rank[rank]`` overrides any constructor argument of that
    rank, ``common`` is given to all. A hang fails the test.
    """
    listening = threading.Event()
    box = {}
    results, errors, groups = {}, {}, {}

    def body(rank):
        kwargs = {"rank": rank, "world_size": world, **common, **(per_rank or {}).get(rank, {})}
        group = None
        try:
            if rank == 0:

                def heard(port):
                    box["port"] = port
                    listening.set()

                try:
                    group = TcpStarGroup(bind_port=0, on_listening=heard, **kwargs)
                finally:
                    listening.set()
            else:
                if not listening.wait(30) or "port" not in box:
                    raise RuntimeError("rank 0 never listened")
                group = TcpStarGroup(host="127.0.0.1", port=box["port"], **kwargs)
            groups[rank] = group
            results[rank] = work(group, rank)
        except BaseException as exc:  # noqa: BLE001 - the tests read it
            errors[rank] = exc
        finally:
            if group is not None:
                group.close()

    threads = [threading.Thread(target=body, args=(rank,), daemon=True) for rank in range(world)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + join_seconds
    for thread in threads:
        thread.join(max(0.1, deadline - time.monotonic()))
    hung = [index for index, thread in enumerate(threads) if thread.is_alive()]
    if hung:
        for group in list(groups.values()):
            group.close()
        pytest.fail(f"rank(s) {hung} hung for {join_seconds:g} s: {errors}", pytrace=False)
    return SimpleNamespace(results=results, errors=errors, groups=groups, port=box.get("port"))


def _ok(outcome):
    assert not outcome.errors, outcome.errors
    return outcome.results


def _read(sock, size):
    data = bytearray()
    while len(data) < size:
        piece = sock.recv(size - len(data))
        if not piece:
            raise EOFError(f"closed after {len(data)} of {size} bytes")
        data += piece
    return bytes(data)


def raw_join(port, *, rank=1, world=2, version=None, magic=None, payload=None, token=None):
    """The handshake of a peer by hand; returns ``(socket, status, reason)``."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=20)
    _magic, _version, _needs, nonce = transport._GREETING.unpack(
        _read(sock, transport._GREETING.size)
    )
    prefix = transport._HELLO_PREFIX.pack(
        transport._MAGIC if magic is None else magic,
        transport.PROTOCOL_VERSION if version is None else version,
        rank,
        world,
        DEFAULT_MAX_PAYLOAD_BYTES if payload is None else payload,
    )
    digest = (
        hmac.new(token.encode(), nonce + prefix, hashlib.sha256).digest()
        if token
        else bytes(transport._DIGEST_BYTES)
    )
    sock.sendall(prefix + digest)
    try:
        status, length = transport._VERDICT.unpack(_read(sock, transport._VERDICT.size))
        reason = _read(sock, length).decode() if length else ""
    except EOFError:
        status, reason = None, ""
    return sock, status, reason


def forge(frame_op, seq, nbytes, weight=0.0):
    return transport._FRAME.pack(frame_op, seq, nbytes, weight)


class _Root:
    """Rank 0 in a thread, to be driven against a hand-made peer in the test thread."""

    def __init__(self, work, *, world=2, **kwargs):
        self.port = None
        self.error = None
        self.result = None
        self.group = None
        self._ready = threading.Event()

        def heard(port):
            self.port = port
            self._ready.set()

        def run():
            try:
                self.group = TcpStarGroup(
                    rank=0, world_size=world, bind_port=0, on_listening=heard, **kwargs
                )
                self.result = work(self.group)
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
            finally:
                self._ready.set()
                if self.group is not None:
                    self.group.close()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert self._ready.wait(30) and self.port, "rank 0 never listened"

    def finish(self, seconds=60.0):
        self._thread.join(seconds)
        assert not self._thread.is_alive(), "rank 0 hung"
        return self


class TestHandshake:
    def test_a_group_forms_and_every_rank_knows_its_place(self):
        outcome = run_ranks(3, lambda group, rank: (group.rank, group.world_size, group.peers))
        assert _ok(outcome) == {
            0: (0, 3, [1, 2]),
            1: (1, 3, [0]),
            2: (2, 3, [0]),
        }

    def test_the_first_wait_is_the_setup_budget_of_1800_seconds(self):
        outcome = run_ranks(2, lambda group, rank: group.timeout_seconds)
        assert _ok(outcome) == {0: 1800.0, 1: 1800.0}

    def test_set_timeout_changes_it_for_every_rank(self):
        def work(group, rank):
            label = group.set_timeout(300)
            return label, group.timeout_seconds

        assert _ok(run_ranks(2, work)) == {
            0: ("TcpStarGroup.set_timeout", 300.0),
            1: ("TcpStarGroup.set_timeout", 300.0),
        }

    @pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, "5", None])
    def test_a_bad_timeout_is_a_bug_of_the_caller(self, seconds):
        def work(group, rank):
            with pytest.raises(ValueError, match=r"the group timeout must be"):
                group.set_timeout(seconds)

        _ok(run_ranks(2, work))

    def test_a_peer_with_another_world_size_is_refused_by_name_on_both_sides(self):
        outcome = run_ranks(2, lambda group, rank: None, per_rank={1: {"world_size": 3}})
        assert isinstance(outcome.errors[1], HandshakeError)
        assert str(outcome.errors[1]) == (
            "rank 0 refused rank 1: world size 3 on the peer, 2 on rank 0 "
            "(SOUP_STREAM_DP_WORLD_SIZE must be equal on every node)"
        )
        assert isinstance(outcome.errors[0], HandshakeError)
        assert "rank 0 refused a peer: world size 3 on the peer, 2 on rank 0" in str(
            outcome.errors[0]
        )

    def test_a_peer_with_another_payload_length_is_refused_by_name(self):
        """Mutation target (the handshake length check): ranks that would split the gradient
        buffer differently must not form a group."""
        outcome = run_ranks(
            2,
            lambda group, rank: None,
            max_payload_bytes=2048,
            per_rank={1: {"max_payload_bytes": 1024}},
        )
        assert isinstance(outcome.errors[1], HandshakeError)
        assert str(outcome.errors[1]).startswith(
            "rank 0 refused rank 1: payload length 1,024 bytes on the peer, 2,048 on rank 0:"
        ), outcome.errors[1]
        assert "payload length 1,024 bytes on the peer, 2,048 on rank 0" in str(outcome.errors[0])

    def test_two_nodes_with_the_same_rank_are_refused_by_name(self):
        outcome = run_ranks(3, lambda group, rank: None, per_rank={2: {"rank": 1}})
        refused = [rank for rank in (1, 2) if isinstance(outcome.errors.get(rank), HandshakeError)]
        assert refused, outcome.errors
        assert "rank 1 is already connected: two nodes set SOUP_STREAM_DP_RANK=1" in str(
            outcome.errors[refused[0]]
        )
        assert isinstance(outcome.errors[0], HandshakeError)

    def test_a_peer_that_claims_a_rank_outside_the_world_is_refused(self):
        root = _Root(lambda group: None)
        sock, status, reason = raw_join(root.port, rank=5)
        sock.close()
        root.finish()
        assert status == 1
        assert reason == (
            "the peer says it is rank 5, but only ranks 1 to 1 connect to rank 0 "
            "(SOUP_STREAM_DP_RANK)"
        )
        assert isinstance(root.error, HandshakeError)
        assert "rank 0 refused a peer: the peer says it is rank 5" in str(root.error)

    def test_a_peer_that_claims_rank_0_is_refused(self):
        root = _Root(lambda group: None)
        sock, status, reason = raw_join(root.port, rank=0)
        sock.close()
        root.finish()
        assert status == 1 and "the peer says it is rank 0" in reason

    def test_a_protocol_version_that_differs_is_refused_by_name(self):
        root = _Root(lambda group: None)
        sock, status, reason = raw_join(root.port, version=9)
        sock.close()
        root.finish()
        assert status == 1
        assert reason == (
            "protocol version 9 on the peer, 1 on rank 0: run the same Soup version on every node"
        )
        assert isinstance(root.error, HandshakeError)

    def test_a_stranger_is_dropped_and_the_group_still_forms(self):
        announced = []
        root = _Root(lambda group: group.peers, announce=announced.append, connect_timeout=30)
        sock, status, _reason = raw_join(root.port, magic=b"NOTSOUP!")
        sock.close()
        assert status is None, "a stranger gets no verdict at all"
        with socket.create_connection(("127.0.0.1", root.port), timeout=5) as junk:
            junk.sendall(b"GET / HTTP/1.1\r\n\r\n")
        peer = TcpStarGroup(rank=1, world_size=2, host="127.0.0.1", port=root.port)
        peer.close()
        root.finish()
        assert root.error is None and root.result == [1]
        assert any("ignored a connection" in line for line in announced), announced

    def test_a_service_that_is_not_rank_0_is_named(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def serve():
            conn, _ = listener.accept()
            conn.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n" + b"x" * 40)
            time.sleep(1)
            conn.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with pytest.raises(HandshakeError) as excinfo:
            TcpStarGroup(
                rank=1,
                world_size=2,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                connect_timeout=20,
            )
        listener.close()
        assert str(excinfo.value) == (
            "rank 1: the service at rank 0's address did not answer with the Soup B1 greeting "
            "(is another program using that port?)"
        )

    def test_nobody_listening_is_named_after_the_connect_budget(self):
        started = time.monotonic()
        with pytest.raises(HandshakeError) as excinfo:
            TcpStarGroup(
                rank=1, world_size=2, host="127.0.0.1", port=_free_port(), connect_timeout=1
            )
        elapsed = time.monotonic() - started
        message = str(excinfo.value)
        assert re.search(r"rank 1 could not reach rank 0 at 127\.0\.0\.1:\d+ within 1 s", message)
        assert "Is rank 0 running, and is SOUP_STREAM_DP_ADDR its external address" in message
        assert 0.9 <= elapsed < 15, elapsed

    def test_a_rank_that_never_comes_is_named_by_rank_0(self):
        root = _Root(lambda group: None, world=3, connect_timeout=3)
        peer = TcpStarGroup(rank=1, world_size=3, host="127.0.0.1", port=root.port)
        root.finish(30)
        peer.close()
        error = root.error
        assert isinstance(error, HandshakeError)
        assert re.search(r"rank\(s\) 2 did not connect to port \d+ in time", str(error)), error
        assert "same SOUP_STREAM_DP_WORLD_SIZE" in str(error)

    def test_a_matching_token_authenticates(self):
        token = "secret-token-0123456789"
        assert _ok(run_ranks(2, lambda group, rank: group.peers, token=token)) == {0: [1], 1: [0]}

    def test_a_peer_with_another_token_is_refused_and_rank_0_keeps_waiting(self):
        announced = []
        outcome = run_ranks(
            2,
            lambda group, rank: None,
            per_rank={
                0: {
                    "token": "right-token-0123456789",
                    "connect_timeout": 3,
                    "announce": announced.append,
                },
                1: {"token": "WRONG-token-0123456789"},
            },
        )
        assert str(outcome.errors[1]) == (
            "rank 0 refused rank 1: authentication failed: check that every node has the same "
            "SOUP_STREAM_DP_TOKEN"
        )
        assert re.search(
            r"did not connect to port \d+ in time \(1 connection", str(outcome.errors[0])
        )
        assert any("failed authentication" in line for line in announced)

    def test_a_peer_without_the_token_rank_0_requires_is_told_so_before_it_sends_anything(self):
        outcome = run_ranks(
            2,
            lambda group, rank: None,
            per_rank={0: {"token": "right-token-0123456789", "connect_timeout": 2}},
        )
        assert str(outcome.errors[1]) == (
            "rank 0 requires SOUP_STREAM_DP_TOKEN, which is not set on rank 1"
        )
        assert isinstance(outcome.errors[0], HandshakeError)

    def test_a_peer_with_a_token_that_rank_0_does_not_use_is_refused_by_name(self):
        outcome = run_ranks(2, lambda group, rank: None, per_rank={1: {"token": "t" * 20}})
        assert str(outcome.errors[1]) == (
            "rank 0 refused rank 1: this peer has SOUP_STREAM_DP_TOKEN set but rank 0 has none: "
            "set it on every node or on none"
        )
        assert isinstance(outcome.errors[0], HandshakeError)

    def test_a_forged_digest_does_not_pass_even_with_the_right_shape(self):
        root = _Root(lambda group: None, token="right-token-0123456789", connect_timeout=2)
        sock, status, reason = raw_join(root.port, token="wrong-token-0123456789")
        sock.close()
        root.finish(20)
        assert status == 1 and reason.startswith("authentication failed")
        assert isinstance(root.error, HandshakeError), "rank 0 waited and gave up"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"world_size": 1}, "world_size must be from 2 to 64, got 1"),
            ({"world_size": 65}, "world_size must be from 2 to 64, got 65"),
            ({"world_size": True}, "world_size must be an int"),
            ({"rank": 2}, "rank must be an int from 0 to 1, got 2"),
            ({"rank": -1}, "rank must be an int from 0 to 1, got -1"),
            ({"rank": True}, "rank must be an int from 0 to 1"),
            ({"max_payload_bytes": 3}, "max_payload_bytes must be an int from 4 to"),
            ({"max_payload_bytes": 2**40}, "max_payload_bytes must be an int from 4 to"),
            ({"timeout_seconds": 0}, "timeout_seconds must be a positive number of seconds"),
            ({"connect_timeout": -1}, "connect_timeout must be a positive number of seconds"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_by_name(self, kwargs, message):
        arguments = {"rank": 1, "world_size": 2, "host": "127.0.0.1", "port": 9, **kwargs}
        with pytest.raises(ValueError, match=re.escape(message)):
            TcpStarGroup(**arguments)

    def test_rank_0_needs_a_port_to_listen_on_and_a_peer_needs_an_address(self):
        with pytest.raises(ValueError, match=r"rank 0 needs bind_port"):
            TcpStarGroup(rank=0, world_size=2)
        with pytest.raises(ValueError, match=r"rank 1 needs host and port"):
            TcpStarGroup(rank=1, world_size=2)

    def test_a_port_that_is_taken_is_named(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            port = taken.getsockname()[1]
            with pytest.raises(HandshakeError) as excinfo:
                TcpStarGroup(rank=0, world_size=2, bind_port=port)
        assert f"rank 0 could not listen on 127.0.0.1:{port}" in str(excinfo.value)
        assert "Is the port taken (see SOUP_STREAM_DP_BIND_PORT)?" in str(excinfo.value)


class _RecordingSocket:
    """A stand-in for a socket: records the options asked for, answers for the buffers."""

    def __init__(self, granted=None, fail_getsockopt=False):
        self.asked = []
        self.granted = granted
        self.fail_getsockopt = fail_getsockopt

    def setsockopt(self, level, option, value):
        self.asked.append((level, option, value))

    def getsockopt(self, level, option):
        if self.fail_getsockopt:
            raise OSError("no such option")
        if self.granted is not None:
            return self.granted
        asked = {(lvl, opt): val for lvl, opt, val in self.asked}
        return asked[(level, option)] * 2  # Linux reports twice what it keeps


class TestSocketBufferGrant:
    """The kernel clamps a socket-buffer request silently (L2: 8 MiB asked, 425,984 granted):
    the transport logs what it got, once, at debug level, and changes nothing else."""

    LOGGER = "soup_cli.utils.adapter_grad_transport"

    @pytest.fixture(autouse=True)
    def _fresh_flag(self, monkeypatch):
        monkeypatch.setattr(transport, "_buffer_grant_logged", False)

    def _records(self, caplog):
        return [
            record
            for record in caplog.records
            if record.name == self.LOGGER and "socket buffers" in record.message
        ]

    def test_a_clamped_grant_is_logged_with_both_sizes(self, caplog):
        caplog.set_level("DEBUG", logger=self.LOGGER)
        transport._tune_socket(_RecordingSocket(granted=425_984))
        (record,) = self._records(caplog)
        assert record.levelname == "DEBUG"
        assert "requested 8388608 bytes each" in record.message
        assert "granted receive=425984 send=425984" in record.message
        assert "the kernel clamped it" in record.message
        assert "net.core.rmem_max" in record.message

    def test_an_unclamped_grant_is_logged_without_the_clamp_note(self, caplog):
        caplog.set_level("DEBUG", logger=self.LOGGER)
        transport._tune_socket(_RecordingSocket())  # Linux reports twice the request
        (record,) = self._records(caplog)
        assert "granted receive=16777216 send=16777216" in record.message
        assert "clamped" not in record.message

    def test_it_is_logged_once_per_process(self, caplog):
        caplog.set_level("DEBUG", logger=self.LOGGER)
        for _ in range(4):
            transport._tune_socket(_RecordingSocket(granted=425_984))
        assert len(self._records(caplog)) == 1

    def test_nothing_is_logged_above_debug(self, caplog):
        caplog.set_level("INFO", logger=self.LOGGER)
        transport._tune_socket(_RecordingSocket(granted=425_984))
        assert self._records(caplog) == []

    def test_a_socket_that_cannot_be_read_back_is_not_an_error(self, caplog):
        caplog.set_level("DEBUG", logger=self.LOGGER)
        transport._tune_socket(_RecordingSocket(fail_getsockopt=True))
        assert self._records(caplog) == []

    def test_the_request_itself_is_unchanged(self):
        """Behaviour pin: still 8 MiB on both buffers, Nagle off, keepalive on."""
        sock = _RecordingSocket(granted=425_984)
        transport._tune_socket(sock)
        asked = {(level, option): value for level, option, value in sock.asked}
        assert asked[(socket.SOL_SOCKET, socket.SO_SNDBUF)] == 8 * 1024 * 1024
        assert asked[(socket.SOL_SOCKET, socket.SO_RCVBUF)] == 8 * 1024 * 1024
        assert asked[(socket.IPPROTO_TCP, socket.TCP_NODELAY)] == 1
        assert asked[(socket.SOL_SOCKET, socket.SO_KEEPALIVE)] == 1

    def test_a_real_group_logs_it_once_on_loopback(self, caplog):
        caplog.set_level("DEBUG", logger=self.LOGGER)
        _ok(run_ranks(2, lambda group, rank: rank))
        assert len(self._records(caplog)) == 1


class TestObjects:
    def test_every_rank_gets_the_same_list_in_rank_order(self):
        def work(group, rank):
            return group.all_gather_object({"rank": rank, "tuple": (1, 2), "text": f"r{rank}"})

        results = _ok(run_ranks(3, work))
        expected = [{"rank": rank, "tuple": [1, 2], "text": f"r{rank}"} for rank in range(3)]
        assert results[0] == results[1] == results[2] == expected

    def test_a_long_sequence_of_collectives_stays_in_step(self):
        def work(group, rank):
            seen = []
            for round_no in range(20):
                seen.append(group.all_gather_object([rank, round_no]))
                group.barrier()
            return seen

        results = _ok(run_ranks(2, work))
        assert results[0] == results[1]
        assert results[0][19] == [[0, 19], [1, 19]]

    def test_a_barrier_holds_every_rank_until_the_last_arrives(self):
        order = []

        def work(group, rank):
            if rank == 1:
                time.sleep(1.0)
            order.append(("in", rank, time.monotonic()))
            group.barrier()
            order.append(("out", rank, time.monotonic()))

        _ok(run_ranks(2, work))
        entered = max(stamp for kind, _r, stamp in order if kind == "in")
        left = min(stamp for kind, _r, stamp in order if kind == "out")
        assert left >= entered - 0.05, "a rank left the barrier before the last one entered"

    def test_json_never_pickle(self):
        class Evil:
            def __reduce__(self):  # pragma: no cover - must never be called
                return (os.system, ("echo pwned",))

        def work(group, rank):
            with pytest.raises(TypeError):
                group.all_gather_object(Evil())
            with pytest.raises(TypeError):
                group.all_gather_object({1, 2})  # a set is not JSON
            return True

        assert _ok(run_ranks(2, work, join_seconds=30)) == {0: True, 1: True}

    def test_nan_is_not_an_object_the_group_carries(self):
        def work(group, rank):
            with pytest.raises(ValueError):
                group.all_gather_object({"x": float("nan")})
            return True

        _ok(run_ranks(2, work, join_seconds=30))

    def test_an_object_over_one_mebibyte_is_refused_before_it_is_sent(self):
        def work(group, rank):
            with pytest.raises(ValueError) as excinfo:
                group.all_gather_object("x" * (MAX_OBJECT_BYTES + 1))
            return str(excinfo.value)

        results = _ok(run_ranks(2, work, join_seconds=30))
        assert "over the 1,048,576-byte limit" in results[0]

    def test_an_object_of_exactly_the_limit_travels(self):
        size = MAX_OBJECT_BYTES - 2  # the two quotes of the JSON string

        def work(group, rank):
            return [len(item) for item in group.all_gather_object("y" * size)]

        assert _ok(run_ranks(2, work)) == {0: [size, size], 1: [size, size]}


class TestWireBounds:
    """What rank 0 and a peer do with a frame that does not say what it should. The frame is
    forged by hand; every declared length is checked BEFORE a byte of the payload is read."""

    @staticmethod
    def _joined(root):
        sock, status, _ = raw_join(root.port)
        assert status == 0
        return sock

    def test_a_declared_object_length_over_the_cap_is_refused_unread(self):
        root = _Root(lambda group: group.all_gather_object("ok"))
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_GATHER, 1, MAX_OBJECT_BYTES + 1))
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert str(root.error) == (
            "rank 1 declared a payload of 1,048,577 bytes for an object gather, over the "
            "1,048,576-byte limit"
        )

    def test_a_declared_length_of_a_terabyte_is_refused_without_allocating_it(self):
        pytest.importorskip("torch")
        import torch

        root = _Root(lambda group: group.all_reduce_weighted(torch.zeros(4), 1.0))
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_ALL_REDUCE, 1, 1 << 40, 1.0))
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert str(root.error) == (
            "rank 1 declared a payload of 1,099,511,627,776 bytes, but rank 0 expects 16 bytes "
            "for an all-reduce: the ranks do not hold the same adapter layout"
        )

    def test_the_peer_that_declares_another_length_is_told_why_by_rank_0(self):
        pytest.importorskip("torch")
        import torch

        def work(group, rank):
            try:
                group.all_reduce_weighted(torch.zeros(8), 1.0)
            except GradTransportError as exc:
                return str(exc)

        results = _ok(run_ranks(2, work, per_rank={}))
        # both ranks use 8 elements here: this is the control, and passes
        assert results == {0: None, 1: None}

        def mismatched(group, rank):
            try:
                group.all_reduce_weighted(torch.zeros(8 if rank == 0 else 6), 1.0)
            except GradTransportError as exc:
                return type(exc).__name__, str(exc)

        results = _ok(run_ranks(2, mismatched))
        name0, text0 = results[0]
        name1, text1 = results[1]
        assert name0 == "ProtocolError"
        assert text0 == (
            "rank 1 declared a payload of 24 bytes, but rank 0 expects 32 bytes for an "
            "all-reduce: the ranks do not hold the same adapter layout"
        )
        assert name1 == "GradTransportError"
        assert text1.startswith("rank 0 stopped the run: rank 0: ProtocolError: rank 1 declared")

    def test_a_peer_out_of_step_is_named_with_both_operations(self):
        root = _Root(lambda group: group.all_gather_object("x"))
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_BARRIER, 1, 0))
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert str(root.error) == (
            "rank 1 sent a barrier while rank 0 is in an object gather: the ranks run different "
            "collective calls (are they training the same config?)"
        )

    def test_a_peer_that_skipped_a_step_is_named(self):
        root = _Root(lambda group: group.barrier())
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_BARRIER, 2, 0))
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert str(root.error).startswith(
            "rank 1 is at collective #2 while rank 0 is at #1: the ranks are out of step"
        )

    def test_an_operation_nobody_knows_is_named_as_such(self):
        root = _Root(lambda group: group.barrier())
        sock = self._joined(root)
        sock.sendall(forge(77, 1, 0))
        root.finish()
        sock.close()
        assert "rank 1 sent unknown operation 77 while rank 0 is in a barrier" in str(root.error)

    @pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
    def test_a_weight_that_is_not_a_finite_non_negative_number_is_refused(self, weight):
        pytest.importorskip("torch")
        import torch

        root = _Root(lambda group: group.all_reduce_weighted(torch.zeros(4), 1.0))
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_ALL_REDUCE, 1, 16, weight) + bytes(16))
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert re.fullmatch(
            r"rank 1 sent the weight (nan|inf|-1\.0), which is not a finite number of at least 0",
            str(root.error),
        ), root.error

    def test_a_gathered_object_that_is_not_json_is_refused(self):
        root = _Root(lambda group: group.all_gather_object("x"))
        sock = self._joined(root)
        sock.sendall(forge(transport.OP_GATHER, 1, 9) + b"{not json")
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)
        assert str(root.error).startswith("rank 1 sent an object that is not valid JSON")

    def test_a_pickle_stream_is_not_json_either(self):
        import pickle

        root = _Root(lambda group: group.all_gather_object("x"))
        sock = self._joined(root)
        payload = pickle.dumps({"a": 1})
        sock.sendall(forge(transport.OP_GATHER, 1, len(payload)) + payload)
        root.finish()
        sock.close()
        assert isinstance(root.error, ProtocolError)

    def test_rank_0_that_answers_with_the_wrong_length_is_refused_by_the_peer(self):
        """The peer side of the same check: a RESULT frame that declares another length."""
        pytest.importorskip("torch")
        import torch

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def fake_root():
            conn, _ = listener.accept()
            nonce = bytes(16)
            conn.sendall(
                transport._GREETING.pack(transport._MAGIC, transport.PROTOCOL_VERSION, 0, nonce)
            )
            _read(conn, transport._HELLO_PREFIX.size + transport._DIGEST_BYTES)
            conn.sendall(transport._VERDICT.pack(0, 0))
            _read(conn, transport._FRAME.size + 16)  # the peer's all-reduce frame and payload
            conn.sendall(forge(transport.OP_RESULT, 1, 4096))
            time.sleep(2)
            conn.close()

        thread = threading.Thread(target=fake_root, daemon=True)
        thread.start()
        peer = TcpStarGroup(rank=1, world_size=2, host="127.0.0.1", port=port, connect_timeout=20)
        with pytest.raises(ProtocolError) as excinfo:
            peer.all_reduce_weighted(torch.zeros(4), 1.0)
        listener.close()
        assert str(excinfo.value) == (
            "rank 0 declared a payload of 4,096 bytes, but rank 1 expects 16 bytes for an "
            "all-reduce result: the ranks do not hold the same adapter layout"
        )

    def test_after_a_failure_the_group_is_broken_and_says_why(self):
        def work(group, rank):
            if rank == 0:
                with pytest.raises(ProtocolError):
                    group.all_gather_object("x")
                with pytest.raises(GradTransportError) as excinfo:
                    group.barrier()
                return str(excinfo.value)
            with pytest.raises(GradTransportError):
                group.barrier()  # out of step with the gather on rank 0

        results = _ok(run_ranks(2, work))
        assert results[0].startswith("the data-parallel group is broken: rank 0: ProtocolError:")

    def test_a_closed_group_refuses_every_call(self):
        def work(group, rank):
            group.close()
            group.close()  # idempotent
            with pytest.raises(GradTransportError, match=r"the data-parallel group is closed"):
                group.barrier()

        outcome = run_ranks(2, work)
        # the peer that was left alone saw its socket close (neither hangs)
        assert set(outcome.results) | set(outcome.errors) == {0, 1}


def _weighted_reference(grads, weights):
    """``sum_r(w_r * g_r) / sum_r(w_r)`` in float64: the formula of the design note."""
    total = sum(weights)
    return sum(w * g.double() for w, g in zip(weights, grads)) / total


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


class TestAllReduce:
    def test_it_equals_the_weighted_mean_and_the_ranks_end_bitwise_equal(self, torch_module):
        torch = torch_module
        generator = torch.Generator().manual_seed(11)
        grads = [torch.randn(1000, generator=generator) for _ in range(2)]
        weights = [37.0, 91.0]

        def work(group, rank):
            flat = grads[rank].clone()
            group.all_reduce_weighted(flat, weights[rank])
            return flat

        results = _ok(run_ranks(2, work))
        assert torch.equal(results[0], results[1]), "the ranks must hold the same bytes"
        expected = _weighted_reference(grads, weights)
        assert torch.allclose(results[0].double(), expected, atol=_ATOL, rtol=_RTOL), (
            (results[0].double() - expected).abs().max()
        )

    def test_it_is_not_the_plain_mean(self, torch_module):
        """Mutation target: a plain mean, which is right for the C-form and wrong here."""
        torch = torch_module
        grads = [torch.full((64,), 1.0), torch.full((64,), 5.0)]
        weights = [1.0, 3.0]

        def work(group, rank):
            flat = grads[rank].clone()
            group.all_reduce_weighted(flat, weights[rank])
            return flat

        result = _ok(run_ranks(2, work))[0]
        assert torch.allclose(result, torch.full((64,), 4.0), atol=1e-6), result[:3]
        assert not torch.allclose(result, torch.full((64,), 3.0), atol=1e-3), "a plain mean"

    def test_equal_weights_give_the_plain_mean(self, torch_module):
        torch = torch_module
        grads = [torch.arange(10, dtype=torch.float32), torch.ones(10) * 4]

        def work(group, rank):
            flat = grads[rank].clone()
            group.all_reduce_weighted(flat, 6.0)
            return flat

        result = _ok(run_ranks(2, work))[0]
        assert torch.allclose(result, (grads[0] + grads[1]) / 2, atol=1e-6)

    def test_the_sum_runs_in_rank_order(self, torch_module):
        """Mutation target (the rank order of the sum): fp32 addition is not associative.
        ``(1e8 + 1) + (-1e8)`` is 0 (the 1 is lost), but ``(1e8 + -1e8) + 1`` is 1."""
        torch = torch_module
        values = [1e8, 1.0, -1e8]

        def work(group, rank):
            flat = torch.full((16,), values[rank])
            group.all_reduce_weighted(flat, 1.0)
            return flat

        results = _ok(run_ranks(3, work))
        for rank in range(3):
            assert results[rank].tolist() == [0.0] * 16, (rank, results[rank][:2])

    def test_a_rank_with_weight_zero_contributes_nothing(self, torch_module):
        torch = torch_module
        grads = [torch.full((8,), 100.0), torch.arange(8, dtype=torch.float32)]

        def work(group, rank):
            flat = grads[rank].clone()
            group.all_reduce_weighted(flat, 0.0 if rank == 0 else 5.0)
            return flat

        result = _ok(run_ranks(2, work))[1]
        assert torch.allclose(result, grads[1], atol=1e-6)

    def test_every_weight_zero_is_refused_on_both_ranks(self, torch_module):
        torch = torch_module

        def work(group, rank):
            try:
                group.all_reduce_weighted(torch.ones(4), 0.0)
            except GradTransportError as exc:
                return type(exc).__name__, str(exc)

        results = _ok(run_ranks(2, work))
        assert results[0][0] == "ProtocolError"
        assert "every rank reported a weight of 0 (no token in any rank's window)" in results[0][1]
        assert results[1][0] == "GradTransportError"
        assert "rank 0 stopped the run" in results[1][1]

    @pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True, "3", None])
    def test_a_bad_weight_is_refused_before_anything_is_sent(self, torch_module, weight):
        torch = torch_module

        def work(group, rank):
            with pytest.raises(ValueError, match=r"weight must be"):
                group.all_reduce_weighted(torch.ones(4), weight)
            return True

        assert _ok(run_ranks(2, work, join_seconds=30)) == {0: True, 1: True}

    def test_a_tensor_the_wire_cannot_carry_is_refused_by_name(self, torch_module):
        torch = torch_module

        def work(group, rank):
            messages = []
            for tensor in (
                torch.ones(4, dtype=torch.float64),
                torch.ones(2, 2),
                torch.ones(8)[::2],
                torch.ones(0),
                [1.0, 2.0],
            ):
                with pytest.raises(ValueError) as excinfo:
                    group.all_reduce_weighted(tensor, 1.0)
                messages.append(str(excinfo.value))
            return messages

        messages = _ok(run_ranks(2, work, join_seconds=30))[0]
        assert (
            messages[0]
            == messages[1]
            == messages[4]
            == ("the group moves one-dimensional fp32 tensors")
        )
        assert messages[2] == "the tensor handed to the group must be contiguous"
        assert messages[3] == "the tensor handed to the group is empty"

    def test_a_payload_over_the_limit_of_the_group_is_refused(self, torch_module):
        torch = torch_module

        def work(group, rank):
            with pytest.raises(ValueError) as excinfo:
                group.all_reduce_weighted(torch.ones(17), 1.0)
            return str(excinfo.value)

        results = _ok(run_ranks(2, work, max_payload_bytes=64, join_seconds=30))
        assert results[0] == (
            "a payload of 68 bytes is over this group's 64-byte limit per operation: "
            "split it into chunks"
        )

    def test_a_view_of_a_larger_tensor_is_reduced_in_place_and_the_rest_is_left_alone(
        self, torch_module
    ):
        torch = torch_module

        def work(group, rank):
            big = torch.full((12,), float(rank + 1))
            group.all_reduce_weighted(big[4:8], 1.0)
            return big

        big = _ok(run_ranks(2, work))[0]
        assert big.tolist() == [1.0] * 4 + [1.5] * 4 + [1.0] * 4

    def test_many_rounds_reuse_the_buffers_and_stay_exact(self, torch_module):
        torch = torch_module

        def work(group, rank):
            outputs = []
            for round_no in range(6):
                flat = torch.full((300 + 50 * (round_no % 3),), float(rank + round_no))
                group.all_reduce_weighted(flat, float(rank + 1))
                outputs.append(flat[0].item())
            return outputs

        results = _ok(run_ranks(2, work))
        expected = [(1.0 * step + 2.0 * (step + 1)) / 3.0 for step in range(6)]
        assert results[0] == results[1]
        assert results[0] == pytest.approx(expected, abs=1e-5)

    def test_adapter_grad_sync_chunks_through_the_group_with_the_given_weight(self, torch_module):
        torch = torch_module
        from soup_cli.utils.adapter_grad_sync import AdapterGradSync

        weights = [2.0, 6.0]

        def work(group, rank):
            module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
            index = 0
            for param in module.parameters():
                param.grad = torch.full_like(param, float(rank + 1) + 0.25 * index)
                index += 1
            sync = AdapterGradSync(module, chunk_bytes=40)  # 10 fp32 elements a chunk
            stats = sync.sync(group, weight=weights[rank])
            return stats, torch.cat([param.grad.reshape(-1) for param in module.parameters()])

        results = _ok(run_ranks(2, work, max_payload_bytes=40))
        stats, flat = results[0]
        assert (stats.bytes, stats.n_params, stats.n_chunks) == (4 * 26, 4, 3)
        assert torch.equal(flat, results[1][1])
        expected = []
        for count, index in ((12, 0), (4, 1), (8, 2), (2, 3)):
            expected += [(2.0 * (1 + 0.25 * index) + 6.0 * (2 + 0.25 * index)) / 8.0] * count
        assert torch.allclose(flat, torch.tensor(expected), atol=1e-6)

    def test_the_torch_group_accepts_only_equal_weights(self, torch_module):
        torch = torch_module
        group = TorchDistributedGroup()
        with pytest.raises(ValueError, match=r"averages with equal weights \(weight=1\)"):
            group.all_reduce_weighted(torch.ones(2), 2.0)

    def test_the_interface_names_who_weights(self):
        assert TcpStarGroup.weighted is True
        assert TorchDistributedGroup.weighted is False
        assert issubclass(TcpStarGroup, GradGroup) and issubclass(TorchDistributedGroup, GradGroup)

    def test_a_group_resolves_to_itself_and_no_process_group_to_none(self, torch_module):
        outcome = run_ranks(2, lambda group, rank: resolve_group(group) is group)
        assert _ok(outcome) == {0: True, 1: True}
        assert resolve_group(None) is None, "this process has no torch process group"


class TestBroadcast:
    def test_rank_0_values_reach_every_rank_bit_for_bit(self, torch_module):
        torch = torch_module
        generator = torch.Generator().manual_seed(5)
        source = torch.randn(500, generator=generator)

        def work(group, rank):
            flat = source.clone() if rank == 0 else torch.zeros(500)
            group.broadcast_(flat)
            return flat

        results = _ok(run_ranks(3, work))
        assert all(torch.equal(results[rank], source) for rank in range(3))

    def test_only_rank_0_can_be_the_source(self, torch_module):
        torch = torch_module

        def work(group, rank):
            with pytest.raises(ValueError, match=r"broadcasts from rank 0 only"):
                group.broadcast_(torch.zeros(4), src=1)
            return True

        assert _ok(run_ranks(2, work, join_seconds=30)) == {0: True, 1: True}

    def test_a_peer_with_another_length_is_named(self, torch_module):
        torch = torch_module

        def work(group, rank):
            try:
                group.broadcast_(torch.zeros(4 if rank == 0 else 6))
            except GradTransportError as exc:
                return type(exc).__name__, str(exc)

        results = _ok(run_ranks(2, work))
        assert results[1] is not None
        assert results[1][0] == "ProtocolError"
        assert "rank 0 declared a payload of 16 bytes, but rank 1 expects 24 bytes" in results[1][1]


# --------------------------------------------------------------------------
# peer death and a silent peer: real processes, no torch
# --------------------------------------------------------------------------


def _worker_env() -> dict:
    import soup_cli

    source_root = str(Path(soup_cli.__file__).resolve().parent.parent)
    env = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "ACCELERATE_USE_CPU",
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_USE_FSDP",
        "FORCE_COLOR",
        "NO_COLOR",
        *transport.B1_ENV_NAMES,
    ):
        env.pop(name, None)  # a variable left in the shell must not reach a rank
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        PYTHONPATH=os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH", "")])),
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TQDM_DISABLE="1",
    )
    return env


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return "<no output>"
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _transport_pair(tmp_path, extra, *, deadline=60.0, wait_for=(0, 1)):
    """Two ``transport`` workers (rank 0 listens, rank 1 connects); ``extra[rank]`` are their
    arguments. Returns ``{rank: SimpleNamespace(code, out, elapsed)}``; what still runs when
    the ``wait_for`` ranks have exited, or at the deadline, is killed and has ``code None``."""
    port = _free_port()
    processes, handles, logs, finished = {}, [], {}, {}
    start = time.monotonic()
    try:
        for rank in (0, 1):
            logs[rank] = tmp_path / f"rank{rank}.log"
            handle = logs[rank].open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                str(_WORKER),
                "transport",
                "--rank",
                str(rank),
                "--port",
                str(port),
                *[str(part) for part in extra.get(rank, [])],
            ]
            processes[rank] = subprocess.Popen(
                command, env=_worker_env(), stdout=handle, stderr=subprocess.STDOUT
            )
        while (
            not all(rank in finished for rank in wait_for) and time.monotonic() - start < deadline
        ):
            for rank, process in processes.items():
                if rank not in finished and process.poll() is not None:
                    finished[rank] = time.monotonic() - start
            time.sleep(0.1)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    return {
        rank: SimpleNamespace(
            code=processes[rank].returncode if rank in finished else None,
            out=_plain(logs[rank].read_text(encoding="utf-8", errors="replace")),
            elapsed=finished.get(rank),
        )
        for rank in (0, 1)
    }


class TestPeerDeath:
    def test_a_clean_run_of_real_processes_finishes_on_both_ranks(self, tmp_path):
        runs = _transport_pair(tmp_path, {0: ["--rounds", 3], 1: ["--rounds", 3]})
        for rank in (0, 1):
            assert runs[rank].code == 0, (rank, runs[rank].out)
            assert '"event": "done"' in runs[rank].out

    def test_a_crashed_peer_ends_rank_0_at_once_and_names_it(self, tmp_path):
        runs = _transport_pair(tmp_path, {0: ["--rounds", 5], 1: ["--rounds", 5, "--die-after", 1]})
        assert runs[1].code == 1, runs[1].out
        assert runs[0].code == 3, runs[0].out
        assert re.search(
            r"ERROR PeerClosedError: rank 1 closed the connection after 0 of 21 bytes of a frame "
            r"header \(it exited or was lost\)",
            runs[0].out,
        ), runs[0].out
        # the worker's own timeout is 60 s: ending sooner is the closed socket, not the clock
        assert runs[0].elapsed < 30, runs[0].elapsed

    def test_a_crashed_rank_0_ends_the_peer_at_once_and_names_it(self, tmp_path):
        runs = _transport_pair(tmp_path, {0: ["--rounds", 5, "--die-after", 1], 1: ["--rounds", 5]})
        assert runs[0].code == 1, runs[0].out
        assert runs[1].code == 3, runs[1].out
        assert re.search(
            r"ERROR PeerClosedError: rank 0 closed the connection (after 0 of 21 bytes of a frame "
            r"header|while rank 1 was sending) \(it exited or was lost\)",
            runs[1].out,
        ), runs[1].out
        assert runs[1].elapsed < 30, runs[1].elapsed

    def test_a_silent_peer_is_cut_off_at_the_shortened_timeout_and_named(self, tmp_path):
        """The workers join with 60 s and are given 3 s after the join, as the setup does with
        300 s after it. Without ``set_timeout`` rank 0 would still be waiting when the deadline
        kills it."""
        runs = _transport_pair(
            tmp_path,
            {
                0: ["--rounds", 5, "--timeout", 3],
                1: ["--rounds", 5, "--timeout", 3, "--hang-after", 1],
            },
            deadline=45,
            wait_for=(0,),
        )
        assert runs[0].code == 3, runs[0].out
        assert '"timeout": 3.0' in runs[0].out and '"event": "joined"' in runs[0].out
        assert re.search(
            r"ERROR PeerTimeoutError: rank 1 sent nothing for 3 s while a frame header was "
            r"expected \(0 of 21 bytes\)",
            runs[0].out,
        ), runs[0].out
        assert 3 <= runs[0].elapsed < 40, runs[0].elapsed
        assert runs[1].code is None, "the silent rank was still asleep when it was killed"

    def test_with_three_ranks_the_one_that_died_is_the_one_named(self):
        def work(group, rank):
            if rank == 2:
                group.close()
                return "left"
            time.sleep(0.5)
            with pytest.raises(GradTransportError) as excinfo:
                group.barrier()
            return str(excinfo.value)

        results = _ok(run_ranks(3, work, join_seconds=60))
        assert "rank 2 closed the connection" in results[0], results[0]
        assert "rank 2 closed the connection" in results[1], results[1]

    def test_a_peer_that_goes_quiet_is_named_by_the_threaded_group_too(self):
        def work(group, rank):
            group.set_timeout(1.5)
            if rank == 1:
                time.sleep(6)
                return "slept"
            started = time.monotonic()
            with pytest.raises(PeerTimeoutError) as excinfo:
                group.barrier()
            return str(excinfo.value), time.monotonic() - started

        outcome = run_ranks(2, work, join_seconds=60)
        message, waited = outcome.results[0]
        assert message == (
            "rank 1 sent nothing for 1.5 s while a frame header was expected (0 of 21 bytes)"
        )
        assert 1.4 <= waited < 5, waited

    def test_a_peer_that_vanishes_mid_payload_is_named_with_the_bytes_it_sent(self, torch_module):
        torch = torch_module
        root = _Root(lambda group: group.all_reduce_weighted(torch.zeros(1000), 1.0))
        sock, status, _ = raw_join(root.port)
        assert status == 0
        sock.sendall(forge(transport.OP_ALL_REDUCE, 1, 4000, 1.0) + bytes(1500))
        sock.close()
        root.finish()
        assert isinstance(root.error, PeerClosedError)
        assert str(root.error) == (
            "rank 1 closed the connection after 1,500 of 4,000 bytes of its gradient payload "
            "(it exited or was lost)"
        )


# --------------------------------------------------------------------------
# the pieces around the group, in process
# --------------------------------------------------------------------------


class _Recorder(GradGroup):
    """A weighted group that records what it is asked and reduces nothing."""

    name = "recorder"

    def __init__(self, rank=0, world_size=2, weighted=True, log=None):
        self.rank = rank
        self.world_size = world_size
        self.weighted = weighted
        self.calls = []
        self.timeouts = []
        self.log = log if log is not None else []
        self.closed = False

    def all_reduce_weighted(self, flat, weight=1.0):
        self.calls.append((flat.numel(), weight))

    def set_timeout(self, seconds):
        self.timeouts.append(seconds)
        self.log.append(("timeout", seconds))
        return "Recorder.set_timeout"

    def close(self):
        self.closed = True


class _FakeHfTrainer:
    """Stands in for HF's ``Trainer``: one backward per ``training_step``."""

    def __init__(self):
        self.accelerator = SimpleNamespace(sync_gradients=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        return "the-loss"


def _trainer_with(group, torch):
    from soup_cli.utils.adapter_sync_trainer import make_adapter_sync_trainer_class

    trainer = make_adapter_sync_trainer_class(_FakeHfTrainer)()
    trainer._dp_group = group
    model = torch.nn.Linear(2, 2)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    return trainer, model


class TestTokenWeight:
    def test_the_weight_of_a_rank_is_its_windows_token_count(self, torch_module):
        """Mutation target (plain mean instead of weighted): the group must get N_r, not 1."""
        torch = torch_module
        group = _Recorder()
        trainer, model = _trainer_with(group, torch)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model, {}, torch.tensor(7))
        assert group.calls == [(6, 7.0)]

    def test_it_is_read_from_the_keyword_or_the_position(self, torch_module):
        torch = torch_module
        group = _Recorder()
        trainer, model = _trainer_with(group, torch)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model=model, inputs={}, num_items_in_batch=5)
        trainer.training_step(model, {}, 11)
        trainer.training_step(model, {}, num_items_in_batch=torch.tensor(13))
        assert [weight for _n, weight in group.calls] == [5.0, 11.0, 13.0]

    def test_only_the_last_micro_batch_of_a_window_syncs(self, torch_module):
        torch = torch_module
        group = _Recorder()
        trainer, model = _trainer_with(group, torch)
        for last in (False, False, True):
            trainer.accelerator.sync_gradients = last
            trainer.training_step(model, {}, torch.tensor(9))
        assert group.calls == [(6, 9.0)]
        assert trainer.sync_calls == 1

    def test_no_token_count_is_refused_instead_of_weighting_equally(self, torch_module):
        torch = torch_module
        trainer, model = _trainer_with(_Recorder(), torch)
        trainer.accelerator.sync_gradients = True
        with pytest.raises(RuntimeError) as excinfo:
            trainer.training_step(model, {})
        assert str(excinfo.value).startswith(
            "transformers passed no num_items_in_batch to training_step, so this rank cannot "
            "weight its gradient by its token count"
        )

    def test_an_unweighted_group_gets_weight_one_whatever_the_count(self, torch_module):
        torch = torch_module
        group = _Recorder(weighted=False)
        trainer, model = _trainer_with(group, torch)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model, {}, torch.tensor(7))
        trainer.training_step(model, {})
        assert [weight for _n, weight in group.calls] == [1.0, 1.0]

    def test_without_a_group_the_torch_default_is_called_as_before(self, torch_module, monkeypatch):
        torch = torch_module
        from soup_cli.utils.adapter_grad_sync import AdapterGradSync

        seen = []
        monkeypatch.setattr(AdapterGradSync, "sync", lambda self, *a, **k: seen.append((a, k)))
        trainer, model = _trainer_with(None, torch)
        trainer.accelerator.sync_gradients = True
        trainer.training_step(model, {}, torch.tensor(7))
        assert seen == [((), {})]

    def test_a_model_whose_forward_takes_no_loss_kwargs_is_refused_by_name(self):
        from soup_cli.utils.adapter_sync_trainer import check_token_weighting

        weighted, equal = _Recorder(), _Recorder(weighted=False)
        accepting = SimpleNamespace(model_accepts_loss_kwargs=True)
        refusing = SimpleNamespace(model_accepts_loss_kwargs=False)
        check_token_weighting(accepting, weighted)
        check_token_weighting(refusing, equal)  # the C-form does not weight
        with pytest.raises(ValueError) as excinfo:
            check_token_weighting(refusing, weighted)
        assert str(excinfo.value).startswith(
            "data-parallel streaming over TCP (SOUP_STREAM_DP_*) weights each node's gradient by "
            "its token count, but this model's forward takes no loss kwargs"
        )


class TestFinishSetupThroughTheGroup:
    """``_finish_distributed_stream_setup`` for B1: C5, then the timeout (C8), then C7."""

    @pytest.fixture
    def host(self, tmp_path, monkeypatch):
        log = []
        group = _Recorder(log=log)
        report = SimpleNamespace(
            world_size=2,
            rank=0,
            n_tensors=8,
            n_elements=1024,
            differing_ranks=(),
            broadcast_bytes=4096,
            seconds=0.5,
        )

        def verify(model, base_items, *, group=None):
            log.append(("init", model, base_items, group))
            return report

        def attach(trainer, output_dir):
            log.append(("metrics", trainer, output_dir))

        monkeypatch.setattr("soup_cli.utils.distributed_init_check.verify_and_sync_init", verify)
        monkeypatch.setattr("soup_cli.utils.stream_dp_runtime.attach_stream_dp_metrics", attach)
        trainer = SimpleNamespace(model_accepts_loss_kwargs=True)
        return SimpleNamespace(
            self=SimpleNamespace(
                model="the-model",
                trainer=trainer,
                _output_dir=str(tmp_path / "out"),
                _stream_base_items={"item": "value"},
                _dp_group=group,
            ),
            log=log,
            group=group,
            trainer=trainer,
        )

    @staticmethod
    def _run(host):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        StreamingSetupMixin._finish_distributed_stream_setup(host.self)

    def test_the_init_check_runs_through_the_group_then_the_timeout_then_the_metrics(
        self, host, capsys
    ):
        self._run(host)
        assert [entry[0] for entry in host.log] == ["init", "timeout", "metrics"]
        assert host.log[0][1:] == ("the-model", {"item": "value"}, host.group)
        assert host.log[2][1:] == (host.trainer, host.self._output_dir)
        output = _plain(capsys.readouterr().out)
        assert "Group timeout set to 300 s (Recorder.set_timeout)" in output
        assert "a peer that stops answering ends this run within that time" in output

    def test_the_timeout_after_the_setup_is_300_seconds(self, host):
        """Mutation target (drop the 300 s timeout after setup): the sockets would keep the
        1800 s of the setup, and a silent peer would be waited for half an hour."""
        self._run(host)
        assert host.group.timeouts == [300]

    def test_the_value_comes_from_the_module_constant(self, host, monkeypatch):
        import soup_cli.trainer.stream_setup as stream_setup

        monkeypatch.setattr(stream_setup, "DISTRIBUTED_STREAM_TIMEOUT_SECONDS", 7)
        self._run(host)
        assert host.group.timeouts == [7]

    def test_a_model_without_loss_kwargs_is_refused_before_any_collective(self, host):
        host.trainer.model_accepts_loss_kwargs = False
        with pytest.raises(ValueError, match=r"takes no loss kwargs"):
            self._run(host)
        assert host.log == [] and host.group.timeouts == []


class TestRankGates:
    def test_the_callback_gate_reads_soups_rank_not_hfs(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback

        # HF is not distributed under B1: every node believes it is the chief
        state = SimpleNamespace(global_step=3, max_steps=10, epoch=1.0, is_world_process_zero=True)
        logs = {"loss": 1.0, "learning_rate": 1e-4, "grad_norm": 0.3}
        for soup_rank, draws in ((0, True), (1, False), (2, False)):
            display, tracker = MagicMock(), MagicMock()
            callback = SoupTrainerCallback(display=display, tracker=tracker, run_id="r")
            callback.rank_gated, callback.soup_rank = True, soup_rank
            callback.on_train_begin(MagicMock(), state, MagicMock())
            callback.on_log(MagicMock(), state, MagicMock(), logs=dict(logs))
            callback.on_train_end(MagicMock(), state, MagicMock())
            assert display.start.called is draws, soup_rank
            assert display.stop.called is draws, soup_rank
            assert tracker.log_metrics.called is draws, soup_rank

    def test_rank_0_draws_even_if_hf_says_it_is_not_the_chief(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback

        state = SimpleNamespace(global_step=3, max_steps=10, epoch=1.0, is_world_process_zero=False)
        display = MagicMock()
        callback = SoupTrainerCallback(display=display, tracker=MagicMock(), run_id="r")
        callback.rank_gated, callback.soup_rank = True, 0
        callback.on_train_begin(MagicMock(), state, MagicMock())
        display.start.assert_called_once()

    def test_the_builder_passes_the_rank_on(self):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.monitoring.callback import build_soup_trainer_callback

        cfg = SoupConfig(base="m", data={"train": "d.jsonl"})
        built = build_soup_trainer_callback(MagicMock(), config=cfg, rank_gated=True, soup_rank=1)
        assert (built.rank_gated, built.soup_rank) == (True, 1)
        plain = build_soup_trainer_callback(MagicMock(), config=cfg)
        assert (plain.rank_gated, plain.soup_rank) == (False, None)

    def test_the_tracker_and_the_diagnose_gate_follow_the_b1_rank(self, tmp_path, monkeypatch):
        from soup_cli.commands.train import _open_run_tracker, _should_run_diagnose_gate_on_rank
        from soup_cli.experiment.tracker import ExperimentTracker, NullTracker

        database = tmp_path / "runs.db"
        monkeypatch.setenv("SOUP_DB_PATH", str(database))
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.setenv("RANK", "0")  # a leftover torchrun variable must not win
        monkeypatch.setenv(ENV_RANK, "1")
        assert _should_run_diagnose_gate_on_rank() is False
        tracker = _open_run_tracker(distributed_stream=True)
        assert type(tracker) is NullTracker and tracker.records_runs is False
        assert not database.exists()
        monkeypatch.setenv(ENV_RANK, "0")
        monkeypatch.setenv("RANK", "1")
        assert _should_run_diagnose_gate_on_rank() is True
        assert type(_open_run_tracker(distributed_stream=True)) is ExperimentTracker

    def test_a_rank_other_than_0_neither_saves_nor_writes_a_tokenizer(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        for rank, writes in ((0, True), (1, False), (3, False)):
            wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
            wrapper._distributed_stream = True
            wrapper._dp_group = _Recorder(rank=rank)
            # HF's should_save is True on every B1 node (it believes each one is rank 0)
            wrapper.trainer = SimpleNamespace(args=SimpleNamespace(should_save=True))
            assert wrapper._writes_artifacts() is writes, rank
            assert wrapper._is_dp_follower() is (rank != 0), rank

    def test_closing_the_runtime_closes_the_sockets(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper._dp_group = _Recorder()
        wrapper._close_stream_runtime()
        assert wrapper._dp_group.closed is True
        wrapper._dp_group = None
        wrapper._close_stream_runtime()  # nothing to close, nothing to raise

    def test_the_rewind_recorder_is_off_for_b1_though_world_size_is_unset(self, tmp_path):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper._distributed_stream = True
        calls = []

        class _Base:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        trainer = wrapper._build_rewind_trainer(
            _Base, {"a": 1}, rows=[], output_dir=tmp_path, batch_size=1, grad_accum=1
        )
        assert isinstance(trainer, _Base) and calls == [{"a": 1}]
        assert wrapper._rewind_notice is True

    def test_the_metrics_file_and_record_carry_the_groups_rank(self, tmp_path):
        pytest.importorskip("transformers")
        from soup_cli.utils.adapter_grad_sync import SyncStats
        from soup_cli.utils.stream_dp_runtime import build_stream_dp_metrics_callback

        trainer = SimpleNamespace(
            _dp_group=SimpleNamespace(rank=1, world_size=3), sync_calls=0, last_sync_stats=None
        )
        callback = build_stream_dp_metrics_callback(trainer, str(tmp_path))
        args = SimpleNamespace(process_index=0, world_size=1)  # HF's view under B1
        state, control = SimpleNamespace(global_step=1), SimpleNamespace(should_log=False)
        callback.on_train_begin(args, state, control)
        callback.on_step_begin(args, state, control)
        trainer.sync_calls = 1
        trainer.last_sync_stats = SyncStats(bytes=96, seconds=0.5, n_params=2, n_chunks=1)
        callback.on_step_end(args, state, control)
        callback.on_train_end(args, state, control)
        lines = (tmp_path / "stream_dp_metrics.rank1.jsonl").read_text("utf-8").splitlines()
        record = json.loads(lines[0])
        assert (record["rank"], record["world_size"], record["sync_bytes"]) == (1, 3, 96)
        assert not (tmp_path / "stream_dp_metrics.rank0.jsonl").exists()


# --------------------------------------------------------------------------
# soup train as one rank of a B1 launch
# --------------------------------------------------------------------------


class _StopAtTrainerError(Exception):
    """Raised by the stub trainer: the command got as far as building one."""


def _invoke_train(tmp_path, monkeypatch, *, rank, extra_env=None, streaming=True, args=()):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ]
        }
        for index in range(3)
    ]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "soup.yaml").write_text(
        "base: smollm2-135m\n"
        "task: sft\n"
        "backend: transformers\n"
        "data:\n"
        "  train: train.jsonl\n"
        "training:\n"
        f"  stream_layers: {'true' if streaming else 'false'}\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", *transport.B1_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    env = {**(_RANK_0 if rank == 0 else _RANK_1), ENV_RANK: str(rank), **(extra_env or {})}
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    class _Wrapper:
        def setup(self, dataset):
            raise _StopAtTrainerError

    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", lambda cfg, **kw: _Wrapper())
    command = ["train", "--config", str(tmp_path / "soup.yaml"), "--yes", *args]
    return CliRunner().invoke(app, command)


class TestTrainCommand:
    def test_rank_0_records_the_run(self, tmp_path, monkeypatch):
        from soup_cli.experiment.tracker import ExperimentTracker

        result = _invoke_train(tmp_path, monkeypatch, rank=0)
        assert isinstance(result.exception, _StopAtTrainerError), (
            result.output,
            repr(result.exception),
        )
        assert "Run ID:" in _plain(result.output)
        assert len(ExperimentTracker(tmp_path / "runs.db").list_runs()) == 1

    def test_every_other_rank_records_nothing_and_says_so(self, tmp_path, monkeypatch):
        result = _invoke_train(tmp_path, monkeypatch, rank=1)
        assert isinstance(result.exception, _StopAtTrainerError), (
            result.output,
            repr(result.exception),
        )
        output = _plain(result.output)
        assert "Run record: rank 0 keeps it; this rank records nothing." in output
        assert "Run ID:" not in output
        assert not (tmp_path / "runs.db").exists()

    def test_the_variables_without_stream_layers_are_refused_by_name(self, tmp_path, monkeypatch):
        result = _invoke_train(tmp_path, monkeypatch, rank=0, streaming=False)
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert (
            f"{ENV_RANK}, {ENV_WORLD_SIZE}, {ENV_BIND_PORT} are set, but training.stream_layers "
            "is not true" in output
        ), output
        assert not isinstance(result.exception, _StopAtTrainerError)
        assert "Run ID" not in output

    def test_a_malformed_variable_is_refused_by_name_before_any_work(self, tmp_path, monkeypatch):
        result = _invoke_train(tmp_path, monkeypatch, rank=0, extra_env={ENV_WORLD_SIZE: "two"})
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "SOUP_STREAM_DP_WORLD_SIZE='two' is not valid" in output
        assert not isinstance(result.exception, _StopAtTrainerError)
        assert "Run ID" not in output and "Loaded:" not in output

    def test_b1_together_with_a_launcher_is_refused(self, tmp_path, monkeypatch):
        result = _invoke_train(tmp_path, monkeypatch, rank=0, extra_env={"WORLD_SIZE": "2"})
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert "set together with WORLD_SIZE=2" in output and "must not be mixed" in output

    @pytest.mark.parametrize(
        ("args", "item"),
        [(["--resume", "auto"], "--resume"), (["--push-as", "user/model"], "--push-as")],
    )
    def test_an_option_that_cannot_run_across_nodes_is_named(
        self, tmp_path, monkeypatch, args, item
    ):
        result = _invoke_train(tmp_path, monkeypatch, rank=0, args=args)
        output = _plain(result.output)
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert _B1_MESSAGE.format(items=item) in output, output
        assert not isinstance(result.exception, _StopAtTrainerError)


# --------------------------------------------------------------------------
# real ranks against one rank: T1, T1a, T1b, T1c for B1
# --------------------------------------------------------------------------

_ROWS = 20


def _letters(seed: int, count: int) -> str:
    return "".join(chr(97 + (seed * 7 + index * 5) % 26) for index in range(count))


def _row(index: int, answer_length: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"item {index:02d} " + _letters(index, 6)},
            {"role": "assistant", "content": _letters(index + 3, answer_length)},
        ]
    }


def _equal_rows(count: int = _ROWS) -> list:
    return [_row(index, 24) for index in range(count)]


def _ragged_rows() -> list:
    """Even rows are long, odd rows short. Rank 0 of two takes the even ones, rank 1 the odd
    ones (``rows[r::2]``), so the two ranks hold very different token counts in every step."""
    return [_row(index, 40 if index % 2 == 0 else 2) for index in range(_ROWS)]


def _need_runs():
    if sys.platform == "darwin":
        pytest.skip("multi-process runs are skipped on macOS (runner timeouts)")
    for module in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "safetensors"):
        pytest.importorskip(module)


def _b1_env(env: dict, rank: int, port: int) -> dict:
    env = dict(env)
    env[ENV_RANK] = str(rank)
    env[ENV_WORLD_SIZE] = "2"
    if rank == 0:
        env[ENV_BIND_PORT] = str(port)
    else:
        env[ENV_ADDR] = f"127.0.0.1:{port}"
    return env


def _read_run(out_dir: Path, tag: str):
    from safetensors.torch import load_file

    return SimpleNamespace(
        tag=tag,
        facts=json.loads((out_dir / f"{tag}.json").read_text("utf-8")),
        tensors=load_file(str(out_dir / f"{tag}.safetensors")),
        log=_plain((out_dir / f"{tag}.log").read_text("utf-8", errors="replace")),
    )


def _command(tag, rank, world, out_dir, base_dir, rows_file, batch, accum, clip, extra):
    command = [sys.executable, str(_WORKER), "train"]
    for flag, value in (
        ("--rank", rank),
        ("--world-size", world),
        ("--tag", tag),
        ("--out-dir", out_dir),
        ("--base-dir", base_dir),
        ("--rows-file", rows_file),
        ("--batch-size", batch),
        ("--grad-accum", accum),
        ("--max-grad-norm", clip),
    ):
        command += [flag, str(value)]
    return command + [str(part) for part in extra]


def _run_group(
    base_dir,
    out_dir: Path,
    *,
    rows,
    ref_rows=None,
    per_device_batch,
    grad_accum,
    max_grad_norm,
    extra=None,
):
    """Two B1 ranks (batch ``b``) and one reference process (batch ``2b``), all at once.

    ``extra[tag]`` are further worker arguments. Nothing launcher-like is set: no
    ``WORLD_SIZE``, no ``MASTER_ADDR``; only ``SOUP_STREAM_DP_*`` on the two ranks.
    """
    _need_runs()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = out_dir / "rows.json"
    rows_file.write_text(json.dumps(rows), encoding="utf-8")
    ref_file = out_dir / "rows_ref.json"
    ref_file.write_text(json.dumps(rows if ref_rows is None else ref_rows), encoding="utf-8")
    port = _free_port()
    extra = extra or {}
    plans = [
        ("b0", 0, per_device_batch, rows_file),
        ("b1", 1, per_device_batch, rows_file),
        ("ref", None, 2 * per_device_batch, ref_file),
    ]
    handles, processes, logs = [], [], {}
    failure = None
    try:
        for tag, rank, batch, rows_path in plans:
            env = _worker_env()
            if rank is not None:
                env = _b1_env(env, rank, port)
            logs[tag] = out_dir / f"{tag}.log"
            handle = logs[tag].open("w", encoding="utf-8")
            handles.append(handle)
            more = list(extra.get(tag, []))
            if rank is None:
                more.append("--no-rewind-log")
            command = _command(
                tag,
                rank or 0,
                2 if rank is not None else 1,
                out_dir,
                base_dir,
                rows_path,
                batch,
                grad_accum,
                max_grad_norm,
                more,
            )
            processes.append(
                subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            )
        deadline = time.monotonic() + _TIMEOUT_SECONDS
        while failure is None and any(process.poll() is None for process in processes):
            for (tag, *_rest), process in zip(plans, processes):
                if process.poll() not in (None, 0):
                    failure = f"{tag} exited with {process.poll()}"
            if failure is None and time.monotonic() > deadline:
                failure = f"still running after {_TIMEOUT_SECONDS} s (a hang)"
            time.sleep(0.25)
        for (tag, *_rest), process in zip(plans, processes):
            if failure is None and process.returncode != 0:
                failure = f"{tag} exited with {process.returncode}"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    if failure is not None:
        report = "\n".join(
            f"--- {tag} (last 40 lines) ---\n{_tail(logs[tag])}" for tag, *_rest in plans
        )
        pytest.fail(f"B1 run failed: {failure}\n{report}", pytrace=False)
    dp0, dp1, ref = (_read_run(out_dir, tag) for tag, *_rest in plans)
    return SimpleNamespace(dp=(dp0, dp1), ref=ref, rows=rows, out_dir=out_dir)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny Llama with its tokenizer, built ONCE with a fixed seed and shared read-only
    (each process shards it into its own cache)."""
    for module in ("torch", "transformers", "tokenizers"):
        pytest.importorskip(module)
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests._tiny_hf_models import build_tokenizer

    target = tmp_path_factory.mktemp("base") / "model"
    torch.manual_seed(7)
    tokenizer = build_tokenizer()
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=256,
    )
    LlamaForCausalLM(config).to(torch.float32).save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)


@pytest.fixture(scope="module")
def group_plain(base_dir, tmp_path_factory):
    """T1: 21 rows (one is cut), per-device batch 2 (reference 4), no accumulation, no clipping.
    The ranks start from different seeds and a 123 s group timeout is asked for; checkpoints
    every 2 steps show which rank writes them."""
    rows = _equal_rows(_ROWS + 1)
    common = ["--save-steps", 2, "--group-timeout", _GROUP_TIMEOUT]
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("plain"),
        rows=rows,
        ref_rows=rows[:_ROWS],
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=1000.0,
        extra={
            "b0": [*common, "--seed", 42],
            "b1": [*common, "--seed", 43],
            "ref": ["--seed", 42],
        },
    )


@pytest.fixture(scope="module")
def group_clip(base_dir, group_plain, tmp_path_factory):
    """T1a: the same run, with ``max_grad_norm`` below every step's gradient norm."""
    norms = group_plain.ref.facts["grad_norms"]
    assert len(norms) == _STEPS and min(norms) > 0, norms
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("clip"),
        rows=_equal_rows(),
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=0.3 * min(norms),
    )


@pytest.fixture(scope="module")
def group_ragged(base_dir, tmp_path_factory):
    """T1b: rank 0 holds long rows, rank 1 short ones (unequal token counts per step)."""
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("ragged"),
        rows=_ragged_rows(),
        per_device_batch=2,
        grad_accum=1,
        max_grad_norm=1000.0,
    )


@pytest.fixture(scope="module")
def group_accum(base_dir, tmp_path_factory):
    """T1c: accumulation 2 with ragged rows, per-device batch 1 (reference batch 2)."""
    return _run_group(
        base_dir,
        tmp_path_factory.mktemp("accum"),
        rows=_ragged_rows(),
        per_device_batch=1,
        grad_accum=2,
        max_grad_norm=1000.0,
    )


def _lora_keys(run):
    return sorted(key for key in run.tensors if key != "per_step")


def _worst(left, right) -> float:
    return float((left.double() - right.double()).abs().max())


def _assert_took_the_b1_path(group):
    """The facts that make the comparison mean what it says."""
    for run in group.dp:
        facts = run.facts
        # HF is NOT distributed under B1: every node believes it is the only process
        assert (facts["world_size"], facts["process_index"]) == (1, 0), facts
        assert facts["distributed_stream"] is True, facts
        assert (facts["group_class"], facts["group_world"]) == ("TcpStarGroup", 2), facts
        assert facts["trainer_class"].startswith("AdapterSync"), facts
        assert facts["has_adapter_sync"] is True
        assert facts["accepts_loss_kwargs"] is True, "HF no longer counts the window's tokens"
        assert facts["global_step"] == _STEPS, facts
        assert len(facts["window_tokens"]) == _STEPS, facts
    assert [run.facts["group_rank"] for run in group.dp] == [0, 1]
    ref = group.ref.facts
    assert ref["world_size"] == 1 and ref["distributed_stream"] is False, ref
    assert ref["group_class"] is None and not ref["trainer_class"].startswith("AdapterSync"), ref
    assert ref["sync_calls"] is None and ref["global_step"] == _STEPS, ref


def _assert_not_vacuous(group):
    ref = group.ref
    assert ref.facts["initial_sha"] == group.dp[0].facts["initial_sha"], "unequal starts"
    assert group.dp[0].facts["initial_sha"] == group.dp[1].facts["initial_sha"]
    assert ref.facts["final_sha"] != ref.facts["initial_sha"], "the adapters never moved"
    per_step = ref.tensors["per_step"]
    assert per_step.shape[0] == _STEPS
    for step in range(1, _STEPS):
        assert not (per_step[step] == per_step[step - 1]).all(), f"step {step} changed nothing"
    assert min(ref.facts["grad_norms"]) > 0


def _assert_equivalent(group):
    """Both ranks equal each other bit for bit, and the one-rank run within tolerance."""
    import torch

    zero, one = group.dp
    ref = group.ref
    assert _lora_keys(zero) == _lora_keys(one) == _lora_keys(ref)
    assert len(_lora_keys(ref)) == 8  # 2 layers x (q_proj, v_proj) x (lora_A, lora_B)
    assert zero.facts["final_sha"] == one.facts["final_sha"], "the ranks ended different"
    for key in (*_lora_keys(ref), "per_step"):
        assert torch.equal(zero.tensors[key], one.tensors[key]), f"ranks differ at {key}"
        assert torch.allclose(zero.tensors[key], ref.tensors[key], atol=_ATOL, rtol=_RTOL), (
            f"{key}: 2 ranks vs 1 rank differ by {_worst(zero.tensors[key], ref.tensors[key])}"
        )
    assert min(zero.facts["grad_norms"]) > 0


def _assert_one_sync_per_optimizer_step(group):
    for run in group.dp:
        facts = run.facts
        assert facts["sync_calls"] == _STEPS, facts
        assert facts["sync_bytes"] == 4 * facts["n_trainable"] > 0, facts
        assert facts["sync_chunks"] == 1 and facts["sync_seconds"] > 0, facts


def _assert_union_loss(group):
    """HF logs each rank's window loss ``L_r = sum of its token losses / N_r``. The loss of the
    union batch, which the reference logs, is therefore ``sum_r(N_r L_r) / sum_r(N_r)``."""
    import torch

    zero, one = group.dp
    for name in ("losses", "window_tokens"):
        assert len(zero.facts[name]) == len(one.facts[name]) == _STEPS, name
    tokens = torch.tensor(
        [zero.facts["window_tokens"], one.facts["window_tokens"]], dtype=torch.float64
    )
    losses = torch.tensor([zero.facts["losses"], one.facts["losses"]], dtype=torch.float64)
    union = (tokens * losses).sum(0) / tokens.sum(0)
    reference = torch.tensor(group.ref.facts["losses"], dtype=torch.float64)
    assert torch.allclose(union, reference, atol=3e-4), (union, reference)
    ref_tokens = torch.tensor(group.ref.facts["window_tokens"], dtype=torch.float64)
    assert torch.equal(tokens.sum(0), ref_tokens), "the ranks together did not see the union batch"


class TestEquivalence:
    """T1 (design note, section 6) for B1: 2 ranks with batch ``b`` == 1 rank with batch ``2b``."""

    def test_the_run_took_the_production_b1_path(self, group_plain):
        _assert_took_the_b1_path(group_plain)

    def test_the_comparison_is_not_vacuous(self, group_plain):
        _assert_not_vacuous(group_plain)

    def test_two_ranks_equal_one_rank_on_the_union_batch(self, group_plain):
        """Mutation target: without the sync the ranks drift apart and away from the one-rank
        run. The ranks started from different seeds, so this also shows that the init
        broadcast (C5) went through the TCP group."""
        _assert_equivalent(group_plain)

    def test_one_sync_per_optimizer_step_of_exactly_the_adapter_bytes(self, group_plain):
        _assert_one_sync_per_optimizer_step(group_plain)

    def test_the_union_loss_is_the_token_weighted_mean_of_the_rank_losses(self, group_plain):
        _assert_union_loss(group_plain)

    def test_each_rank_cut_its_shard_to_the_same_length_and_says_so(self, group_plain):
        for rank, run in enumerate(group_plain.dp):
            assert run.facts["n_train_rows"] == _ROWS // 2, run.facts
            text = run.log
            assert (
                f"Data parallel: 21 training rows over 2 ranks, 10 each (this is rank {rank})."
                in text
            ), text[-2500:]
            assert (
                "Data parallel: 1 trailing row(s) dropped so that every rank runs the same" in text
            )
        assert group_plain.ref.facts["n_train_rows"] == _ROWS

    def test_the_banner_the_group_and_the_init_check_are_said_once_per_rank(self, group_plain):
        for rank, run in enumerate(group_plain.dp):
            log = run.log
            assert log.count("Distributed layer streaming is BETA (B1, TCP):") == 1, log[-2500:]
            assert f"this is rank {rank} (task=sft" in log
            assert log.count("Data-parallel group ready: 2 ranks connected") == 1
            assert log.count("Distributed init:") == 1, log[-2500:]
            assert "2 ranks agree on the trainable layout (8 tensors" in log
            assert "adapters broadcast from rank 0" in log
            assert "Rewind log off: the recorder is single-process" in log
        for run in group_plain.dp:
            assert (
                "rank(s) 1 started from different adapter values (a different seed?) and took "
                "rank 0's" in run.log
            ), run.log[-2500:]
        assert "Distributed" not in group_plain.ref.log
        assert "Data-parallel group" not in group_plain.ref.log


class TestClipping:
    """T1a: clipping sees the AVERAGED gradient, because the sync precedes it."""

    def test_the_clipping_really_fires_on_every_step(self, group_plain, group_clip):
        limit = 0.3 * min(group_plain.ref.facts["grad_norms"])
        for run in (*group_clip.dp, group_clip.ref):
            norms = run.facts["grad_norms"]  # HF logs the norm BEFORE clipping
            assert len(norms) == _STEPS and all(norm > limit for norm in norms), (norms, limit)

    def test_two_ranks_equal_one_rank_when_every_step_is_clipped(self, group_clip):
        _assert_took_the_b1_path(group_clip)
        _assert_not_vacuous(group_clip)
        _assert_equivalent(group_clip)
        _assert_one_sync_per_optimizer_step(group_clip)

    def test_the_clipped_run_is_not_the_unclipped_one(self, group_plain, group_clip):
        import torch

        assert not torch.allclose(
            group_clip.ref.tensors["per_step"],
            group_plain.ref.tensors["per_step"],
            atol=_ATOL,
            rtol=_RTOL,
        ), "clipping changed nothing: the threshold is not exercising anything"


class TestRaggedRanks:
    """T1b: the ranks hold different token counts in the same step. This is where the token
    weight matters: a plain mean of the two gradients is NOT the gradient of the union."""

    def test_the_ranks_really_hold_different_amounts_of_text(self, group_ragged):
        zero, one = (run.facts["window_tokens"] for run in group_ragged.dp)
        assert len(zero) == len(one) == _STEPS
        for step, (long_rank, short_rank) in enumerate(zip(zero, one)):
            assert long_rank > 2 * short_rank, (step, long_rank, short_rank)

    def test_two_ranks_equal_one_rank_however_the_tokens_are_spread(self, group_ragged):
        """Mutation target (plain mean instead of weighted): HF divides each rank's loss by its
        OWN token count, so equal weights over-weight the short rank's tokens."""
        _assert_took_the_b1_path(group_ragged)
        _assert_not_vacuous(group_ragged)
        _assert_equivalent(group_ragged)
        _assert_one_sync_per_optimizer_step(group_ragged)

    def test_the_union_loss_is_the_token_weighted_mean_of_the_rank_losses(self, group_ragged):
        _assert_union_loss(group_ragged)


class TestAccumulation:
    """T1c: the sync fires once per OPTIMIZER step, on the last micro-batch of a window, and a
    window's weight is the token count of the WHOLE window."""

    @pytest.mark.parametrize("window", [1, 2])
    def test_equal_to_the_one_rank_run_and_one_sync_per_step(self, request, window):
        group = request.getfixturevalue("group_plain" if window == 1 else "group_accum")
        _assert_took_the_b1_path(group)
        _assert_not_vacuous(group)
        _assert_equivalent(group)
        _assert_one_sync_per_optimizer_step(group)

    def test_a_window_of_two_means_half_as_many_syncs_as_micro_batches(self, group_accum):
        for run in group_accum.dp:
            assert run.facts["sync_calls"] == _STEPS == _ROWS // 2 // 2

    def test_the_union_loss_is_still_the_weighted_mean_over_windows(self, group_accum):
        _assert_union_loss(group_accum)


class TestRankZeroOnly:
    """Which rank wrote what. Each rank has its own output directory (two hosts in life)."""

    @staticmethod
    def _files(group, rank):
        return group.dp[rank].facts["files"]

    def test_only_rank_0_wrote_the_adapter_and_its_checkpoints(self, group_plain):
        zero, one = self._files(group_plain, 0), self._files(group_plain, 1)
        assert "adapter_model.safetensors" in zero and "adapter_config.json" in zero
        for step in (2, 4):
            assert f"checkpoint-{step}/adapter_model.safetensors" in zero, zero
        assert not [name for name in one if "adapter_" in name], one
        assert not [name for name in one if name.startswith("checkpoint-")], one

    def test_the_tokenizer_is_saved_where_the_model_is_and_nowhere_else(self, group_plain):
        zero, one = self._files(group_plain, 0), self._files(group_plain, 1)
        assert any(name.startswith("tokenizer") and "/" not in name for name in zero), zero
        assert not [name for name in one if "tokenizer" in name], one

    def test_each_rank_wrote_its_own_metrics_file_with_its_own_rank_in_it(self, group_plain):
        zero, one = self._files(group_plain, 0), self._files(group_plain, 1)
        assert "stream_dp_metrics.rank0.jsonl" in zero
        assert "stream_dp_metrics.rank1.jsonl" not in zero
        assert "stream_dp_metrics.rank1.jsonl" in one
        assert "stream_dp_metrics.rank0.jsonl" not in one
        for rank, run in enumerate(group_plain.dp):
            path = Path(run.facts["output_dir"]) / f"stream_dp_metrics.rank{rank}.jsonl"
            lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            assert [line["step"] for line in lines] == list(range(1, _STEPS + 1)), lines
            for line in lines:
                assert line["sync_bytes"] == 4 * run.facts["n_trainable"] > 0, line
                assert line["sync_chunks"] == 1 and line["t_sync"] > 0, line
                assert (line["world_size"], line["rank"]) == (2, rank), line
                assert line["grad_norm"] is not None and line["grad_norm"] > 0, line
            assert [line["loss"] for line in lines] == run.facts["losses"]

    def test_the_live_panel_was_drawn_on_rank_0_alone(self, group_plain):
        zero, one = (run.facts for run in group_plain.dp)
        assert zero["display_calls"]["start"] == 1 and zero["display_calls"]["stop"] == 1
        assert zero["display_calls"]["update"] >= _STEPS
        assert one["display_calls"] == {"start": 0, "update": 0, "stop": 0}

    def test_tracker_rows_were_written_for_rank_0_alone(self, group_plain):
        """Both ranks hand the callback a REAL tracker (each its own database), so only the
        callback's rank gate, which reads Soup's rank and not HF's, keeps rank 1 from writing."""
        zero, one = (run.facts for run in group_plain.dp)
        assert zero["metric_rows"] >= _STEPS, zero
        assert one["metric_rows"] == 0, one

    def test_the_group_timeout_was_set_once_after_the_setup(self, group_plain):
        """Mutation target (drop the shortened timeout): the sockets would keep 1800 s."""
        for run in group_plain.dp:
            assert run.facts["group_timeout"] == float(_GROUP_TIMEOUT), run.facts
            assert (
                run.log.count(f"Group timeout set to {_GROUP_TIMEOUT} s (TcpStarGroup.set_timeout)")
                == 1
            ), run.log[-2500:]


# --------------------------------------------------------------------------
# a rank that dies or goes silent in the middle of a run
# --------------------------------------------------------------------------

_SLACK = 60
_SILENT_TIMEOUT = 8


def _fault_run(base_dir, out_dir, *, fault_rank, fault, timeout, wait_for):
    """Two B1 ranks and a fault on one: ``exit`` (a crash) or ``hang`` (silence)."""
    _need_runs()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = out_dir / "rows.json"
    rows_file.write_text(json.dumps(_equal_rows()), encoding="utf-8")
    port = _free_port()
    common = [
        "--group-timeout",
        timeout,
        "--fault",
        fault,
        "--fault-step",
        2,
        "--fault-rank",
        fault_rank,
    ]
    processes, handles, logs = {}, [], {}
    exited_at, marker_at, codes = {}, {}, {}
    hung = False
    start = time.monotonic()
    try:
        for rank, tag in enumerate(_TAGS):
            env = _b1_env(_worker_env(), rank, port)
            logs[tag] = out_dir / f"{tag}.log"
            handle = logs[tag].open("w", encoding="utf-8")
            handles.append(handle)
            command = _command(tag, rank, 2, out_dir, base_dir, rows_file, 2, 1, 1000.0, common)
            processes[tag] = subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
        deadline = max(timeout, 30) + _SLACK + 240
        while True:
            now = time.monotonic() - start
            for tag, process in processes.items():
                if tag not in exited_at and process.poll() is not None:
                    exited_at[tag] = now
                    codes[tag] = process.returncode
                if tag not in marker_at:
                    try:
                        text = logs[tag].read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        text = ""
                    if "fault -- " in text:
                        marker_at[tag] = now
            if all(tag in exited_at for tag in wait_for):
                break
            if now > deadline:
                hung = True
                break
            time.sleep(0.25)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        for handle in handles:
            handle.close()
    runs = {
        tag: SimpleNamespace(
            tag=tag,
            code=codes.get(tag),
            exited_at=exited_at.get(tag),
            marker_at=marker_at.get(tag),
            log=_plain(logs[tag].read_text(encoding="utf-8", errors="replace")),
            raw=logs[tag],
        )
        for tag in _TAGS
    }
    return SimpleNamespace(runs=runs, hung=hung)


def _report(outcome) -> str:
    return "\n".join(
        f"--- {tag} (exit {run.code}, last 40 lines) ---\n{_tail(run.raw)}"
        for tag, run in outcome.runs.items()
    )


@pytest.fixture(scope="module")
def rank_1_exits(base_dir, tmp_path_factory):
    return _fault_run(
        base_dir,
        tmp_path_factory.mktemp("rank1_exits"),
        fault_rank=1,
        fault="exit",
        timeout=120,
        wait_for=_TAGS,
    )


@pytest.fixture(scope="module")
def rank_0_exits(base_dir, tmp_path_factory):
    return _fault_run(
        base_dir,
        tmp_path_factory.mktemp("rank0_exits"),
        fault_rank=0,
        fault="exit",
        timeout=120,
        wait_for=_TAGS,
    )


@pytest.fixture(scope="module")
def rank_1_goes_silent(base_dir, tmp_path_factory):
    return _fault_run(
        base_dir,
        tmp_path_factory.mktemp("rank1_silent"),
        fault_rank=1,
        fault="hang",
        timeout=_SILENT_TIMEOUT,
        wait_for=("b0",),
    )


class TestPeerDeathInARun:
    @staticmethod
    def _survivor_ends_the_run(outcome, dead_tag, timeout):
        alive_tag = _TAGS[1 - _TAGS.index(dead_tag)]
        dead, alive = outcome.runs[dead_tag], outcome.runs[alive_tag]
        assert not outcome.hung, f"the survivor hung\n{_report(outcome)}"
        assert dead.marker_at is not None, f"the fault never fired\n{_report(outcome)}"
        assert alive.code not in (None, 0), f"the survivor did not fail\n{_report(outcome)}"
        waited = alive.exited_at - dead.marker_at
        assert waited >= -1, f"the survivor ended before the fault ({waited:.1f} s)"
        assert waited <= timeout + _SLACK, f"{waited:.1f} s after the fault\n{_report(outcome)}"
        assert "Traceback" in alive.log, alive.log[-2000:]
        return alive, waited

    def test_rank_0_ends_the_run_at_once_when_rank_1_exits(self, rank_1_exits):
        alive, waited = self._survivor_ends_the_run(rank_1_exits, "b1", 120)
        assert "rank 1 closed the connection" in alive.log, alive.log[-2000:]
        assert waited < 100, "a closed socket must not wait for the 120 s timeout"
        assert rank_1_exits.runs["b1"].code == 1

    def test_rank_1_ends_the_run_at_once_when_rank_0_exits(self, rank_0_exits):
        alive, waited = self._survivor_ends_the_run(rank_0_exits, "b0", 120)
        assert "rank 0 closed the connection" in alive.log, alive.log[-2000:]
        assert waited < 100, "a closed socket must not wait for the 120 s timeout"
        assert rank_0_exits.runs["b0"].code == 1

    def test_a_peer_that_goes_silent_is_cut_off_at_the_shortened_timeout(self, rank_1_goes_silent):
        """The case the timeout exists for. Without the call after the setup the sockets keep
        1800 s and this run hits the deadline instead."""
        alive, waited = self._survivor_ends_the_run(rank_1_goes_silent, "b1", _SILENT_TIMEOUT)
        assert re.search(
            rf"PeerTimeoutError: rank 1 sent nothing for {_SILENT_TIMEOUT} s", alive.log
        ), alive.log[-2000:]
        assert waited >= _SILENT_TIMEOUT - 2, f"cut off after {waited:.1f} s, before the timeout"
