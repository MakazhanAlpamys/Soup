"""The transport under data-parallel layer streaming: one group interface, two backends (B1).

Data parallelism over the LoRA adapters needs four collectives and nothing else: a weighted
mean of the flat gradient buffer, a broadcast of the adapters from rank 0, an all-gather of
small objects (the cross-rank init check), and a barrier. :class:`GradGroup` is that
interface. Two classes implement it:

* :class:`TorchDistributedGroup` wraps today's ``torch.distributed`` calls (the C-form, a
  launch by ``torchrun`` / ``accelerate launch``). It behaves exactly as before: a SUM over
  the chunk, then a divide by the world size, i.e. a plain mean, because HF itself scales
  the loss by the GLOBAL token count there. It refuses a weight other than 1.
* :class:`TcpStarGroup` (B1) is for machines that cannot reach each other's addresses (vast.ai
  containers behind NAT): gloo and NCCL open a data connection to the peer's own interface
  address, which is refused there, but ONE TCP connection to rank 0's mapped port works.
  Every peer connects to rank 0, which sums and answers; no other socket is opened.

B1 in short:

* **Launch.** Each node runs a plain, non-distributed ``soup train`` (``WORLD_SIZE`` unset)
  with four environment variables, see :func:`read_b1_env`. HF is not distributed: Soup
  shards the rows by rank and this group reduces the gradients.
* **Weighted mean.** A non-distributed HF divides the window's summed token loss by its OWN
  token count ``N_r``, so the union-batch gradient is ``sum_r(N_r * g_r) / sum_r(N_r)``, not
  the plain mean. :meth:`TcpStarGroup.all_reduce_weighted` takes ``weight = N_r``; rank 0 sums
  ``weight_r * flat_r`` in RANK order, divides by the total weight, and sends the one result
  to every rank, so all ranks hold the same bytes.
* **Wire.** After the handshake (magic, protocol version, rank, world size, the largest
  payload one operation may carry, and optionally an HMAC over a nonce: ``SOUP_STREAM_DP_TOKEN``)
  every operation is a 21-byte frame (operation, sequence number, payload length, weight)
  followed by the payload. Every declared length is checked against the length the operation
  expects BEFORE a byte of it is read, and against a hard cap; objects are JSON (never
  pickle) and capped at 1 MiB. Payloads are received with ``recv_into`` into preallocated
  buffers; socket buffers are enlarged; Nagle is off.
* **Failure.** A closed socket, a timeout, a frame out of step or a refused handshake raises
  :class:`GradTransportError` naming the peer rank. The group is then broken: the other side
  is told (best effort) and every later call fails at once. Timeouts: 1800 s until
  :meth:`TcpStarGroup.set_timeout` lowers them once the heavy setup is behind (300 s).

This module imports only the standard library at import time; torch is imported inside the
functions that move tensors.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ENV_RANK = "SOUP_STREAM_DP_RANK"
ENV_WORLD_SIZE = "SOUP_STREAM_DP_WORLD_SIZE"
ENV_ADDR = "SOUP_STREAM_DP_ADDR"
ENV_BIND_PORT = "SOUP_STREAM_DP_BIND_PORT"
ENV_TOKEN = "SOUP_STREAM_DP_TOKEN"

#: The four variables that switch B1 on, plus the optional token.
B1_ENV_NAMES = (ENV_RANK, ENV_WORLD_SIZE, ENV_ADDR, ENV_BIND_PORT, ENV_TOKEN)

#: A star on one host's threads: more ranks than this is not what the design was measured for.
MAX_WORLD_SIZE = 64

#: Seconds until the setup is done (the slower rank may still be downloading and sharding),
#: then the shorter one ``TcpStarGroup.set_timeout`` is given (the setup passes
#: ``DISTRIBUTED_STREAM_TIMEOUT_SECONDS``).
SETUP_TIMEOUT_SECONDS = 1800.0
#: Seconds one handshake step may take (a stranger that opens a port and says nothing).
HANDSHAKE_STEP_SECONDS = 10.0
#: Seconds between two attempts of a peer to reach rank 0.
CONNECT_RETRY_SECONDS = 1.0

#: The largest payload one operation may carry (``AdapterGradSync`` chunk size by default).
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
#: Hard ceiling on ``max_payload_bytes``, whatever a caller asks for.
HARD_MAX_PAYLOAD_BYTES = 1 << 30
#: One gathered object, as JSON.
MAX_OBJECT_BYTES = 1024 * 1024
#: A failure reason sent in an ERROR frame or a refusal.
MAX_REASON_BYTES = 512

#: How long a failed group reads and drops what its peers still send before it closes.
DRAIN_SECONDS = 2.0

#: Requested socket buffer size (the kernel may grant less; Linux doubles what is asked).
SOCKET_BUFFER_BYTES = 8 * 1024 * 1024

PROTOCOL_VERSION = 1
_MAGIC = b"SOUPB1DP"

_LOG = logging.getLogger(__name__)
_buffer_grant_lock = threading.Lock()
_buffer_grant_logged = False

_GREETING = struct.Struct("!8sHB16s")  # magic, version, token required, nonce
_HELLO_PREFIX = struct.Struct("!8sHIIQ")  # magic, version, rank, world size, max payload
_DIGEST_BYTES = 32
_VERDICT = struct.Struct("!BH")  # status (0 ok, 1 refused), reason length
_FRAME = struct.Struct("!BIQd")  # operation, sequence, payload length, weight

OP_ALL_REDUCE = 1
OP_RESULT = 2
OP_BROADCAST = 3
OP_GATHER = 4
OP_GATHER_RESULT = 5
OP_BARRIER = 6
OP_RELEASE = 7
OP_ERROR = 8

_OP_NAMES = {
    OP_ALL_REDUCE: "an all-reduce",
    OP_RESULT: "an all-reduce result",
    OP_BROADCAST: "a broadcast",
    OP_GATHER: "an object gather",
    OP_GATHER_RESULT: "a gather result",
    OP_BARRIER: "a barrier",
    OP_RELEASE: "a barrier release",
    OP_ERROR: "an error report",
}

_ELEMENT_BYTES = 4
_MIN_TOKEN_CHARS = 16
_MAX_TOKEN_CHARS = 256


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class GradTransportError(RuntimeError):
    """The data-parallel group failed: a peer is gone, silent, out of step or refused."""


class HandshakeError(GradTransportError):
    """A rank could not join the group (a mismatch, a refusal, a rank that never came)."""


class PeerClosedError(GradTransportError):
    """The connection to a peer rank ended."""


class PeerTimeoutError(GradTransportError):
    """A peer rank stayed silent for longer than the group timeout."""


class ProtocolError(GradTransportError):
    """A peer sent a frame this rank did not expect (wrong operation, step or length)."""


class B1ConfigError(ValueError):
    """The ``SOUP_STREAM_DP_*`` environment is malformed or incomplete."""


def _text(value: object, limit: int = 200) -> str:
    """Text for a message: terminal-safe, and short enough to read."""
    from soup_cli.utils.terminal import for_terminal

    shown = for_terminal(str(value))
    return shown if len(shown) <= limit else shown[:limit] + "..."


# ---------------------------------------------------------------------------
# the group interface
# ---------------------------------------------------------------------------


class GradGroup:
    """What data-parallel layer streaming needs from a group of ranks.

    ``rank`` and ``world_size`` are plain attributes or properties. ``weighted`` says
    whether :meth:`all_reduce_weighted` honours ``weight`` (B1) or averages with equal
    weights because the loss was already scaled globally (the C-form).
    """

    name = "group"
    weighted = False
    rank: int
    world_size: int

    def all_reduce_weighted(self, flat: Any, weight: float = 1.0) -> None:
        """Replace ``flat`` (fp32, 1-D) in place by ``sum_r(w_r * flat_r) / sum_r(w_r)``."""
        raise NotImplementedError

    def broadcast_(self, flat: Any, src: int = 0) -> None:
        """Copy rank ``src``'s ``flat`` into every rank's, in place."""
        raise NotImplementedError

    def all_gather_object(self, obj: Any) -> List[Any]:
        """One JSON-able object per rank, in rank order (the same list on every rank)."""
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError

    def set_timeout(self, seconds: float) -> str:
        """Shorten how long a collective waits for a silent peer; returns the API's name."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the group's resources (idempotent)."""


class TorchDistributedGroup(GradGroup):
    """``torch.distributed`` behind :class:`GradGroup`: the C-form, unchanged.

    ``group`` is a torch process group or ``None`` for the default one. Every call reads the
    ``torch.distributed`` attribute at call time, so a test that fakes ``dist.all_reduce``
    sees the same calls as before the interface existed.
    """

    name = "torch.distributed"
    weighted = False

    def __init__(self, group: Optional[Any] = None) -> None:
        self._group = group

    @property
    def rank(self) -> int:  # type: ignore[override]
        import torch.distributed as dist

        return dist.get_rank(self._group)

    @property
    def world_size(self) -> int:  # type: ignore[override]
        import torch.distributed as dist

        return dist.get_world_size(self._group)

    def all_reduce_weighted(self, flat: Any, weight: float = 1.0) -> None:
        import torch.distributed as dist

        if weight != 1.0:
            raise ValueError(
                "the torch.distributed group averages with equal weights (weight=1), got "
                f"{weight!r}: HF scales the loss by the global token count under a distributed "
                "launch, so a weight here would count the tokens twice"
            )
        world_size = dist.get_world_size(self._group)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self._group)
        flat.div_(world_size)

    def broadcast_(self, flat: Any, src: int = 0) -> None:
        import torch.distributed as dist

        source = src if self._group is None else dist.get_global_rank(self._group, src)
        dist.broadcast(flat, src=source, group=self._group)

    def all_gather_object(self, obj: Any) -> List[Any]:
        import torch.distributed as dist

        gathered: List[Any] = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(gathered, obj, group=self._group)
        return gathered

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self._group)

    def set_timeout(self, seconds: float) -> str:
        raise NotImplementedError(
            "the torch.distributed group's timeout is set by stream_dp_runtime.apply_group_timeout"
        )


def resolve_group(group: Optional[Any] = None) -> Optional[GradGroup]:
    """A :class:`GradGroup` for ``group``: itself, or the torch process group's wrapper.

    ``None`` (and any torch group) means ``torch.distributed``; the result is ``None`` when
    torch has no initialised process group, which is how a single-process run looks.
    """
    if isinstance(group, GradGroup):
        return group
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return None
    return TorchDistributedGroup(group)


# ---------------------------------------------------------------------------
# B1: the environment
# ---------------------------------------------------------------------------

_HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
_HOSTNAME_RE = re.compile(rf"{_HOST_LABEL}(?:\.{_HOST_LABEL})*")
_IPV6_RE = re.compile(r"\[[0-9A-Fa-f:.]{2,45}\]")


@dataclass(frozen=True)
class B1Env:
    """The validated ``SOUP_STREAM_DP_*`` environment of one rank."""

    rank: int
    world_size: int
    #: Where peers reach rank 0 (``SOUP_STREAM_DP_ADDR``); ``None`` on rank 0 if it was left out.
    host: Optional[str]
    port: Optional[int]
    #: The port rank 0 listens on locally (``SOUP_STREAM_DP_BIND_PORT``); ``None`` on a peer.
    bind_port: Optional[int]
    token: Optional[str]
    #: All interfaces: a vast port mapping reaches the container's own address, not loopback.
    bind_host: str = "0.0.0.0"


def _plain_int(name: str, raw: str, low: int, high: int) -> int:
    if not (raw.isascii() and raw.isdigit()) or len(raw) > 9:
        raise B1ConfigError(
            f"{name}={_text(repr(raw))} is not valid: it must be a plain decimal integer "
            f"from {low} to {high}"
        )
    value = int(raw)
    if not low <= value <= high:
        raise B1ConfigError(f"{name}={raw} is out of range: it must be from {low} to {high}")
    return value


def _has_blank_or_control(text: str) -> bool:
    return any(ch.isspace() or ch < " " for ch in text)


def _split_addr(raw: str) -> Tuple[str, int]:
    shown = _text(repr(raw))  # quote first, escape second (tests/test_terminal_output_hygiene.py)
    if not raw or len(raw) > 260 or not raw.isascii() or _has_blank_or_control(raw):
        raise B1ConfigError(
            f"{ENV_ADDR}={shown} is not valid: it must be host:port, for example "
            "203.0.113.7:41123 or node0.example.com:41123"
        )
    host, separator, port_text = raw.rpartition(":")
    if not separator or not host:
        raise B1ConfigError(
            f"{ENV_ADDR}={shown} is not valid: it must be host:port with a port "
            "(the external port that maps to rank 0's listening port)"
        )
    if not (_HOSTNAME_RE.fullmatch(host) or _IPV6_RE.fullmatch(host)) or len(host) > 253:
        raise B1ConfigError(
            f"{ENV_ADDR}={shown} is not valid: '{_text(host)}' is not a host name or an "
            "IP address (no scheme, no path, no spaces; an IPv6 address goes in [brackets])"
        )
    return host, _plain_int(f"the port of {ENV_ADDR}", port_text, 1, 65535)


def _validated_token(raw: str) -> str:
    if not _MIN_TOKEN_CHARS <= len(raw) <= _MAX_TOKEN_CHARS or any(ch < " " for ch in raw):
        raise B1ConfigError(
            f"{ENV_TOKEN} must be {_MIN_TOKEN_CHARS} to {_MAX_TOKEN_CHARS} characters without "
            f"control characters (it is {len(raw)} long); the value is not shown"
        )
    return raw


def read_b1_env(environ: Optional[Mapping[str, str]] = None) -> Optional[B1Env]:
    """The B1 settings of this process, ``None`` when none of the variables is set.

    B1 is on as soon as any of ``SOUP_STREAM_DP_RANK``, ``_WORLD_SIZE``, ``_ADDR``,
    ``_BIND_PORT`` or ``_TOKEN`` is set, and then every problem is refused BY NAME
    (:class:`B1ConfigError`, a ``ValueError``): a variable that is missing, malformed or out of
    range, ``rank >= world size``, a world below 2. Rank 0 needs ``_BIND_PORT`` (the port it
    listens on); every other rank needs ``_ADDR`` (``host:port`` where rank 0 is reached, on
    vast the external mapped port). The other one is optional and checked when present.
    ``_TOKEN`` (optional, 16 to 256 characters) makes the handshake authenticate with an HMAC.
    """
    env = os.environ if environ is None else environ
    present = {name: env.get(name) for name in B1_ENV_NAMES if env.get(name) is not None}
    if not present:
        return None
    if ENV_RANK not in present or ENV_WORLD_SIZE not in present:
        missing = [name for name in (ENV_RANK, ENV_WORLD_SIZE) if name not in present]
        raise B1ConfigError(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not set, but "
            f"{', '.join(sorted(present))} {'is' if len(present) == 1 else 'are'} set: B1 "
            f"data-parallel streaming needs {ENV_RANK} and {ENV_WORLD_SIZE} on every node "
            "(and unset all SOUP_STREAM_DP_* variables for an ordinary run)"
        )
    world_size = _plain_int(ENV_WORLD_SIZE, present[ENV_WORLD_SIZE], 2, MAX_WORLD_SIZE)
    rank = _plain_int(ENV_RANK, present[ENV_RANK], 0, 1 << 30)
    if rank >= world_size:
        raise B1ConfigError(
            f"{ENV_RANK}={rank} is not below {ENV_WORLD_SIZE}={world_size}: ranks are "
            f"numbered from 0 to {world_size - 1}"
        )
    host: Optional[str] = None
    port: Optional[int] = None
    bind_port: Optional[int] = None
    if ENV_ADDR in present:
        host, port = _split_addr(present[ENV_ADDR])
    elif rank != 0:
        raise B1ConfigError(
            f"{ENV_ADDR} is not set on rank {rank}: it is host:port where this rank reaches "
            "rank 0 (on vast, rank 0's public IP and the external port mapped to its "
            f"{ENV_BIND_PORT})"
        )
    if ENV_BIND_PORT in present:
        bind_port = _plain_int(ENV_BIND_PORT, present[ENV_BIND_PORT], 1, 65535)
    elif rank == 0:
        raise B1ConfigError(
            f"{ENV_BIND_PORT} is not set on rank 0: it is the port rank 0 listens on inside its "
            f"container (peers reach it through {ENV_ADDR}, which on vast is a different, "
            "mapped port)"
        )
    token = _validated_token(present[ENV_TOKEN]) if ENV_TOKEN in present else None
    return B1Env(
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        bind_port=bind_port,
        token=token,
    )


def refuse_b1_without_streaming(environ: Optional[Mapping[str, str]] = None) -> None:
    """Refuse a run that sets ``SOUP_STREAM_DP_*`` but does not stream layers.

    B1 exists only for ``training.stream_layers: true``; a leftover variable in the shell
    must not silently train an ordinary run on a shard of the data.
    """
    env = os.environ if environ is None else environ
    names = [name for name in B1_ENV_NAMES if env.get(name) is not None]
    if names:
        raise B1ConfigError(
            f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} set, but "
            "training.stream_layers is not true: data-parallel training over TCP (B1) exists "
            "only for layer streaming. Unset the SOUP_STREAM_DP_* variables, or set "
            "training.stream_layers: true."
        )


def b1_rank_or_none(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The B1 rank of this process, read leniently; ``None`` when B1 is off or unreadable.

    For gates that must not raise (is this the node that writes the run's files?): the strict
    validation is :func:`read_b1_env`, which runs before any of them matter.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_RANK)
    if raw is not None and raw.isascii() and raw.isdigit() and len(raw) <= 9:
        return int(raw)
    return None


# ---------------------------------------------------------------------------
# B1: sockets
# ---------------------------------------------------------------------------


def _log_buffer_grant_once(sock: socket.socket) -> None:
    """Debug-log, once per process, what the kernel granted for the socket buffer request.

    Linux clamps ``SO_RCVBUF`` / ``SO_SNDBUF`` to ``net.core.rmem_max`` / ``wmem_max`` without an
    error, and a socket whose buffer was set explicitly is no longer autotuned, so a large request
    can leave a SMALLER window than the default would reach (measured 2026-11-09 on two rented
    nodes: 8 MiB asked, 425,984 bytes granted). Linux reports twice what it keeps, so a grant
    below the request is a real clamp. This reads and logs; it changes no setting.
    """
    global _buffer_grant_logged
    with _buffer_grant_lock:
        if _buffer_grant_logged:
            return
        _buffer_grant_logged = True
    try:
        received = int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
        sent = int(sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
    except (OSError, TypeError, ValueError):
        return
    clamped = min(received, sent) < SOCKET_BUFFER_BYTES
    _LOG.debug(
        "B1 socket buffers: requested %d bytes each, the kernel granted receive=%d send=%d%s",
        SOCKET_BUFFER_BYTES,
        received,
        sent,
        " (below the request: the kernel clamped it, see net.core.rmem_max and wmem_max, "
        "and no longer autotunes the TCP window of this socket)"
        if clamped
        else "",
    )


def _tune_socket(sock: socket.socket) -> None:
    """Large buffers, no Nagle, keepalive. Every option is best effort."""
    for level, option, value in (
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_BYTES),
        (socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_BYTES),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ):
        try:
            sock.setsockopt(level, option, value)
        except OSError:
            pass
    _log_buffer_grant_once(sock)
    # A rank may sit idle for minutes while the other downloads and shards: keep a NAT's
    # mapping alive. Names differ per platform.
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 6)):
        option = getattr(socket, name, None)
        if option is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
            except OSError:
                pass
    if sys.platform == "win32" and hasattr(sock, "ioctl"):  # pragma: no cover - platform
        try:
            sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60_000, 15_000))  # type: ignore[attr-defined]
        except (OSError, ValueError, AttributeError):
            pass


def _size_buffers(sock: socket.socket) -> None:
    for option in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, option, SOCKET_BUFFER_BYTES)
        except OSError:
            pass


def _make_listener(bind_host: str, bind_port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name != "nt":  # on Windows SO_REUSEADDR lets another process share the port
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _size_buffers(listener)  # before listen: the accepted sockets inherit it
        listener.bind((bind_host, bind_port))
        listener.listen(max(8, MAX_WORLD_SIZE))
    except OSError as exc:
        listener.close()
        raise HandshakeError(
            f"rank 0 could not listen on {_text(bind_host)}:{bind_port}: {_text(exc)}. "
            f"Is the port taken (see {ENV_BIND_PORT})?"
        ) from exc
    return listener


#: What a peer that exited looks like: an end of stream on Linux, a reset on Windows.
_GONE = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


class _Link:
    """One connection of rank 0 to a peer, or of a peer to rank 0."""

    def __init__(self, peer_rank: int, sock: socket.socket) -> None:
        self.peer_rank = peer_rank
        self.sock = sock


def _checked_weight(weight: Any) -> float:
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"weight must be a number, got {type(weight).__name__} ({weight!r})")
    value = float(weight)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"weight must be a finite number of at least 0, got {weight!r}")
    return value


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def encode_object(obj: Any) -> bytes:
    """``obj`` as compact JSON (never pickle), at most :data:`MAX_OBJECT_BYTES`."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_OBJECT_BYTES:
        raise ValueError(
            f"an object sent through the group is {len(data):,} bytes as JSON, over the "
            f"{MAX_OBJECT_BYTES:,}-byte limit"
        )
    return data


def decode_object(data: bytes, *, source: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError(
            f"{source} sent an object that is not valid JSON: {_text(exc)}"
        ) from exc


# ---------------------------------------------------------------------------
# B1: the group
# ---------------------------------------------------------------------------


class TcpStarGroup(GradGroup):
    """A star of TCP connections: rank 0 listens, every other rank connects to it (B1).

    Construction blocks until the whole group is connected (or ``connect_timeout`` seconds
    pass): rank 0 accepts ``world_size - 1`` peers, a peer retries until rank 0 listens.

    ``host`` / ``port`` are where a peer reaches rank 0; ``bind_host`` / ``bind_port`` are
    where rank 0 listens (``bind_port=0`` takes a free port, reported through
    ``on_listening(port)``). ``token`` enables the HMAC handshake. ``timeout_seconds`` is the
    per-operation wait for a silent peer; lower it with :meth:`set_timeout` after the setup.
    """

    name = "tcp-star"
    weighted = True

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        host: Optional[str] = None,
        port: Optional[int] = None,
        bind_host: str = "127.0.0.1",
        bind_port: Optional[int] = None,
        token: Optional[str] = None,
        timeout_seconds: float = SETUP_TIMEOUT_SECONDS,
        connect_timeout: float = SETUP_TIMEOUT_SECONDS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        on_listening: Optional[Callable[[int], None]] = None,
        announce: Optional[Callable[[str], None]] = None,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise ValueError(f"world_size must be an int, got {world_size!r}")
        if not 2 <= world_size <= MAX_WORLD_SIZE:
            raise ValueError(f"world_size must be from 2 to {MAX_WORLD_SIZE}, got {world_size}")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError(f"rank must be an int from 0 to {world_size - 1}, got {rank!r}")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or not _ELEMENT_BYTES <= max_payload_bytes <= HARD_MAX_PAYLOAD_BYTES
        ):
            raise ValueError(
                f"max_payload_bytes must be an int from {_ELEMENT_BYTES} to "
                f"{HARD_MAX_PAYLOAD_BYTES}, got {max_payload_bytes!r}"
            )
        self.rank = rank
        self.world_size = world_size
        self.max_payload_bytes = max_payload_bytes
        self._token = None if token is None else token.encode("utf-8")
        self._announce = announce or (lambda _message: None)
        self._timeout = self._positive(timeout_seconds, "timeout_seconds")
        self._links: Dict[int, _Link] = {}
        self._seq = 0
        self._closed = False
        self._broken: Optional[str] = None
        self._peer_buffers: Dict[int, Any] = {}
        self._connect_seconds = self._positive(connect_timeout, "connect_timeout")
        deadline = time.monotonic() + self._connect_seconds
        try:
            if rank == 0:
                if bind_port is None:
                    raise ValueError("rank 0 needs bind_port (the port it listens on)")
                self._accept_peers(bind_host, bind_port, deadline, on_listening)
            else:
                if host is None or port is None:
                    raise ValueError(f"rank {rank} needs host and port (where rank 0 is reached)")
                self._connect_to_root(host, port, deadline)
        except BaseException:
            self.close()
            raise

    # -- small helpers -------------------------------------------------------

    @staticmethod
    def _positive(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number of seconds, got {value!r}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive number of seconds, got {value}")
        return float(value)

    @property
    def peers(self) -> List[int]:
        """The ranks this rank is connected to (rank 0: all others; others: just 0)."""
        return sorted(self._links)

    def _digest(self, nonce: bytes, prefix: bytes) -> bytes:
        if self._token is None:
            return bytes(_DIGEST_BYTES)
        return hmac.new(self._token, nonce + prefix, hashlib.sha256).digest()

    # -- joining the group ---------------------------------------------------

    def _accept_peers(
        self,
        bind_host: str,
        bind_port: int,
        deadline: float,
        on_listening: Optional[Callable[[int], None]],
    ) -> None:
        listener = _make_listener(bind_host, bind_port)
        try:
            actual = listener.getsockname()[1]
            if on_listening is not None:
                on_listening(actual)
            self._announce(
                f"rank 0 listens on {bind_host}:{actual}, waiting for "
                f"{self.world_size - 1} peer(s)"
                + ("" if self._token else " (no token: anyone who can reach the port may join)")
            )
            ignored = 0
            while len(self._links) < self.world_size - 1:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        rank for rank in range(1, self.world_size) if rank not in self._links
                    ]
                    raise HandshakeError(
                        f"rank(s) {', '.join(map(str, missing))} did not connect to port "
                        f"{actual} in time ({ignored} connection(s) were ignored: not this "
                        "protocol or failed authentication). Check that every node set "
                        f"{ENV_ADDR} to this node's external address for that port and the same "
                        f"{ENV_WORLD_SIZE}."
                    )
                listener.settimeout(min(remaining, 1.0))
                try:
                    conn, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    raise HandshakeError(f"rank 0 lost its listening socket: {_text(exc)}") from exc
                if not self._admit(conn):
                    ignored += 1
                    self._announce(
                        "rank 0 ignored a connection that did not speak the protocol or failed "
                        "authentication"
                    )
        finally:
            listener.close()

    def _admit(self, conn: socket.socket) -> bool:
        """Handshake with one connection; ``False`` to drop it and keep listening.

        A stranger (wrong magic, wrong digest) is dropped quietly. A peer that proved itself
        (right magic and digest) but disagrees with this rank on a setting is told so by name
        and the group fails: the launch is inconsistent and nothing would train correctly.
        """
        try:
            _tune_socket(conn)
            conn.settimeout(HANDSHAKE_STEP_SECONDS)
            nonce = secrets.token_bytes(16)
            conn.sendall(_GREETING.pack(_MAGIC, PROTOCOL_VERSION, 1 if self._token else 0, nonce))
            raw = self._recv_exact_raw(conn, _HELLO_PREFIX.size + _DIGEST_BYTES)
        except (OSError, EOFError):
            conn.close()
            return False
        prefix, digest = raw[: _HELLO_PREFIX.size], raw[_HELLO_PREFIX.size :]
        magic, version, peer_rank, peer_world, peer_payload = _HELLO_PREFIX.unpack(prefix)
        if magic != _MAGIC:
            conn.close()
            return False
        if self._token is not None:
            if not hmac.compare_digest(digest, self._digest(nonce, prefix)):
                self._refuse(
                    conn,
                    f"authentication failed: check that every node has the same {ENV_TOKEN}",
                )
                return False
        elif digest != bytes(_DIGEST_BYTES):
            self._refuse_and_fail(
                conn,
                f"this peer has {ENV_TOKEN} set but rank 0 has none: set it on every node "
                "or on none",
            )
        problem = self._hello_problem(version, peer_rank, peer_world, peer_payload)
        if problem is not None:
            self._refuse_and_fail(conn, problem)
        try:
            conn.sendall(_VERDICT.pack(0, 0))
        except OSError:
            conn.close()
            return False
        conn.settimeout(self._timeout)
        self._links[peer_rank] = _Link(peer_rank, conn)
        return True

    def _hello_problem(
        self, version: int, peer_rank: int, peer_world: int, peer_payload: int
    ) -> Optional[str]:
        if version != PROTOCOL_VERSION:
            return (
                f"protocol version {version} on the peer, {PROTOCOL_VERSION} on rank 0: run the "
                "same Soup version on every node"
            )
        if peer_world != self.world_size:
            return (
                f"world size {peer_world} on the peer, {self.world_size} on rank 0 "
                f"({ENV_WORLD_SIZE} must be equal on every node)"
            )
        if not 1 <= peer_rank < self.world_size:
            return (
                f"the peer says it is rank {peer_rank}, but only ranks 1 to "
                f"{self.world_size - 1} connect to rank 0 ({ENV_RANK})"
            )
        if peer_rank in self._links:
            return f"rank {peer_rank} is already connected: two nodes set {ENV_RANK}={peer_rank}"
        if peer_payload != self.max_payload_bytes:
            return (
                f"payload length {peer_payload:,} bytes on the peer, {self.max_payload_bytes:,} "
                "on rank 0: the ranks would split the gradient buffer differently "
                "(run the same Soup version with the same settings on every node)"
            )
        return None

    @staticmethod
    def _refuse(conn: socket.socket, reason: str) -> None:
        data = reason.encode("utf-8")[:MAX_REASON_BYTES]
        try:
            conn.sendall(_VERDICT.pack(1, len(data)) + data)
        except OSError:
            pass
        finally:
            conn.close()

    def _refuse_and_fail(self, conn: socket.socket, reason: str) -> None:
        self._refuse(conn, reason)
        raise HandshakeError(f"rank 0 refused a peer: {reason}")

    @staticmethod
    def _recv_exact_raw(sock: socket.socket, size: int) -> bytes:
        buffer = bytearray(size)
        view = memoryview(buffer)
        got = 0
        while got < size:
            count = sock.recv_into(view[got:])
            if not count:
                raise EOFError("closed")
            got += count
        return bytes(buffer)

    def _connect_to_root(self, host: str, port: int, deadline: float) -> None:
        shown = f"{_text(host)}:{port}"
        last = "no attempt was made"
        attempts = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HandshakeError(
                    f"rank {self.rank} could not reach rank 0 at {shown} within "
                    f"{self._connect_seconds:g} s ({attempts} attempt(s); last: {last}). "
                    f"Is rank 0 running, and is {ENV_ADDR} its external address for the "
                    "port it listens on?"
                )
            attempts += 1
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(
                    socket.AF_INET6 if host.startswith("[") else socket.AF_INET, socket.SOCK_STREAM
                )
                _size_buffers(sock)  # before connect: the window scale is fixed by it
                sock.settimeout(min(HANDSHAKE_STEP_SECONDS, remaining))
                sock.connect((host.strip("[]"), port))
                _tune_socket(sock)
                sock.settimeout(HANDSHAKE_STEP_SECONDS)
                self._join(sock)
            except (OSError, EOFError) as exc:
                last = _text(exc) or type(exc).__name__
                if sock is not None:
                    sock.close()
                time.sleep(min(CONNECT_RETRY_SECONDS, max(0.0, deadline - time.monotonic())))
                continue
            except BaseException:
                if sock is not None:
                    sock.close()
                raise
            sock.settimeout(self._timeout)
            self._links[0] = _Link(0, sock)
            return

    def _join(self, sock: socket.socket) -> None:
        """The peer's side of the handshake; ``OSError`` / ``EOFError`` mean "try again"."""
        greeting = self._recv_exact_raw(sock, _GREETING.size)
        magic, version, needs_token, nonce = _GREETING.unpack(greeting)
        if magic != _MAGIC:
            raise HandshakeError(
                f"rank {self.rank}: the service at rank 0's address did not answer with the "
                "Soup B1 greeting (is another program using that port?)"
            )
        if version != PROTOCOL_VERSION:
            raise HandshakeError(
                f"rank 0 speaks protocol version {version}, this rank {PROTOCOL_VERSION}: run the "
                "same Soup version on every node"
            )
        if needs_token and self._token is None:
            raise HandshakeError(
                f"rank 0 requires {ENV_TOKEN}, which is not set on rank {self.rank}"
            )
        prefix = _HELLO_PREFIX.pack(
            _MAGIC, PROTOCOL_VERSION, self.rank, self.world_size, self.max_payload_bytes
        )
        sock.sendall(prefix + self._digest(nonce, prefix))
        status, length = _VERDICT.unpack(self._recv_exact_raw(sock, _VERDICT.size))
        if status == 0:
            return
        reason = self._recv_exact_raw(sock, min(length, MAX_REASON_BYTES)) if length else b""
        raise HandshakeError(
            f"rank 0 refused rank {self.rank}: "
            f"{_text(reason.decode('utf-8', errors='replace'), MAX_REASON_BYTES)}"
        )

    # -- the timeout, failure, closing ---------------------------------------

    def set_timeout(self, seconds: float) -> str:
        """Make every later wait for a silent peer last ``seconds``; returns the API label."""
        value = self._positive(seconds, "the group timeout")
        self._timeout = value
        for link in self._links.values():
            try:
                link.sock.settimeout(value)
            except OSError:
                pass
        return "TcpStarGroup.set_timeout"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for link in self._links.values():
            for action in (partial(link.sock.shutdown, socket.SHUT_RDWR), link.sock.close):
                try:
                    action()
                except OSError:
                    pass

    def _unblock(self, *, except_rank: int) -> None:
        """Wake every thread that waits to receive on a link but the failing one's."""
        for rank, link in self._links.items():
            if rank == except_rank:
                continue
            try:
                link.sock.shutdown(socket.SHUT_RD)
            except OSError:
                pass

    def _close_gracefully(self) -> None:
        """Close after a failure without a reset that would eat the error report.

        Closing a socket that still holds unread data makes the kernel send a RST, and a RST
        makes the other side discard what it has received but not yet read, which is the
        ERROR frame that says why. So: end the sending half (the report is followed by a FIN),
        read and drop whatever the peer is still sending for a moment, then close.
        """
        deadline = time.monotonic() + DRAIN_SECONDS
        for link in self._links.values():
            try:
                link.sock.shutdown(socket.SHUT_WR)
                while time.monotonic() < deadline:
                    link.sock.settimeout(max(0.05, deadline - time.monotonic()))
                    if not link.sock.recv(1 << 20):
                        break
            except OSError:
                pass
        self.close()

    def _tell_others(self, reason: str) -> None:
        """Best effort: say why before the sockets close, so the peers do not just see a hang-up."""
        data = reason.encode("utf-8", errors="replace")[:MAX_REASON_BYTES]
        frame = _FRAME.pack(OP_ERROR, self._seq & 0xFFFFFFFF, len(data), 0.0) + data
        for link in self._links.values():
            try:
                link.sock.settimeout(1.0)
                link.sock.sendall(frame)
            except OSError:
                pass

    def _guard(self) -> None:
        if self._broken is not None:  # first: a failed group is also closed
            raise GradTransportError(f"the data-parallel group is broken: {self._broken}")
        if self._closed:
            raise GradTransportError("the data-parallel group is closed")

    def _run(self, operation: Callable[[], Any]) -> Any:
        """Run one collective; on any failure break the group, tell the peers, and re-raise."""
        self._guard()
        try:
            return operation()
        except BaseException as exc:
            reason = f"rank {self.rank}: {type(exc).__name__}: {_text(exc, 300)}"
            self._broken = reason
            self._tell_others(reason)
            self._close_gracefully()
            raise

    # -- raw I/O with peer naming --------------------------------------------

    def _send(self, link: _Link, *parts: Any) -> None:
        try:
            for part in parts:
                link.sock.sendall(part)
        except socket.timeout as exc:
            raise PeerTimeoutError(
                f"rank {self.rank} timed out after {self._timeout:g} s sending to rank "
                f"{link.peer_rank}"
            ) from exc
        except _GONE as exc:
            raise PeerClosedError(
                f"rank {link.peer_rank} closed the connection while rank {self.rank} was sending "
                "(it exited or was lost)"
            ) from exc
        except OSError as exc:
            raise PeerClosedError(
                f"rank {self.rank} lost the connection to rank {link.peer_rank} while sending: "
                f"{_text(exc)}"
            ) from exc

    def _recv_into(self, link: _Link, view: memoryview, what: str) -> None:
        got, size = 0, len(view)
        try:
            while got < size:
                count = link.sock.recv_into(view[got:])
                if not count:
                    raise PeerClosedError(
                        f"rank {link.peer_rank} closed the connection after {got:,} of {size:,} "
                        f"bytes of {what} (it exited or was lost)"
                    )
                got += count
        except socket.timeout as exc:
            raise PeerTimeoutError(
                f"rank {link.peer_rank} sent nothing for {self._timeout:g} s while "
                f"{what} was expected ({got:,} of {size:,} bytes)"
            ) from exc
        except PeerClosedError:
            raise
        except _GONE as exc:
            raise PeerClosedError(
                f"rank {link.peer_rank} closed the connection after {got:,} of {size:,} bytes of "
                f"{what} (it exited or was lost)"
            ) from exc
        except OSError as exc:
            raise PeerClosedError(
                f"lost the connection to rank {link.peer_rank} while receiving {what}: {_text(exc)}"
            ) from exc

    def _send_frame(
        self, link: _Link, op: int, seq: int, nbytes: int, weight: float = 0.0, payload: Any = None
    ) -> None:
        header = _FRAME.pack(op, seq, nbytes, weight)
        if payload is None or nbytes == 0:
            self._send(link, header)
        else:
            self._send(link, header, payload)

    def _read_frame(self, link: _Link) -> Tuple[int, int, int, float]:
        raw = bytearray(_FRAME.size)
        self._recv_into(link, memoryview(raw), "a frame header")
        op, seq, nbytes, weight = _FRAME.unpack(raw)
        if op == OP_ERROR:
            if nbytes > MAX_REASON_BYTES:
                raise ProtocolError(
                    f"rank {link.peer_rank} sent an error report of {nbytes:,} bytes "
                    f"(over {MAX_REASON_BYTES})"
                )
            reason = bytearray(nbytes)
            if nbytes:
                self._recv_into(link, memoryview(reason), "an error report")
            raise GradTransportError(
                f"rank {link.peer_rank} stopped the run: "
                f"{_text(bytes(reason).decode('utf-8', errors='replace'), MAX_REASON_BYTES)}"
            )
        return op, seq, nbytes, weight

    def _expect(
        self,
        link: _Link,
        frame: Tuple[int, int, int, float],
        op: int,
        seq: int,
        *,
        nbytes: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> None:
        got_op, got_seq, got_bytes, _weight = frame
        peer = link.peer_rank
        if got_op != op:
            raise ProtocolError(
                f"rank {peer} sent {_OP_NAMES.get(got_op, f'unknown operation {got_op}')} while "
                f"rank {self.rank} is in {_OP_NAMES[op]}: the ranks run different collective "
                "calls (are they training the same config?)"
            )
        if got_seq != seq:
            raise ProtocolError(
                f"rank {peer} is at collective #{got_seq} while rank {self.rank} is at #{seq}: "
                "the ranks are out of step (a rank skipped or repeated an optimizer step)"
            )
        if nbytes is not None and got_bytes != nbytes:
            raise ProtocolError(
                f"rank {peer} declared a payload of {got_bytes:,} bytes, but rank {self.rank} "
                f"expects {nbytes:,} bytes for {_OP_NAMES[op]}: the ranks do not hold the same "
                "adapter layout"
            )
        if limit is not None and got_bytes > limit:
            raise ProtocolError(
                f"rank {peer} declared a payload of {got_bytes:,} bytes for {_OP_NAMES[op]}, "
                f"over the {limit:,}-byte limit"
            )

    def _fan(self, ranks: Sequence[int], work: Callable[[_Link], Any]) -> Dict[int, Any]:
        """Run ``work`` for every link of ``ranks``, in parallel when there are several.

        Results are keyed by rank; the FIRST failure is raised (the others are its echo).
        """
        results: Dict[int, Any] = {}
        errors: Dict[int, BaseException] = {}
        failed: List[int] = []
        lock = threading.Lock()

        def one(rank: int) -> None:
            try:
                results[rank] = work(self._links[rank])
            except BaseException as exc:  # noqa: BLE001 - the first one is re-raised below
                with lock:
                    errors[rank] = exc
                    failed.append(rank)
                # The other ranks' threads must not wait out the timeout for a group that has
                # already failed. Only their receiving half is shut: the report of the failure
                # still goes out on them.
                self._unblock(except_rank=rank)

        if len(ranks) == 1:
            one(ranks[0])
        else:
            threads = [threading.Thread(target=one, args=(rank,), daemon=True) for rank in ranks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if errors:
            raise errors[failed[0]]  # the first to fail is the cause, the rest are its echo
        return results

    # -- tensors -------------------------------------------------------------

    def _host_buffer(self, flat: Any) -> Tuple[Any, memoryview, bool]:
        """``flat`` as a CPU tensor, the bytes of it, and whether it is a staging copy."""
        import torch

        if not isinstance(flat, torch.Tensor) or flat.dtype != torch.float32 or flat.dim() != 1:
            raise ValueError("the group moves one-dimensional fp32 tensors")
        if not flat.is_contiguous():
            raise ValueError("the tensor handed to the group must be contiguous")
        count = flat.numel()
        if count == 0:
            raise ValueError("the tensor handed to the group is empty")
        nbytes = count * _ELEMENT_BYTES
        if nbytes > self.max_payload_bytes:
            raise ValueError(
                f"a payload of {nbytes:,} bytes is over this group's "
                f"{self.max_payload_bytes:,}-byte limit per operation: split it into chunks"
            )
        staged = flat.device.type != "cpu"
        host = flat.detach().to("cpu") if staged else flat.detach()
        return host, memoryview(host.numpy()).cast("B"), staged

    def _peer_buffer(self, rank: int, count: int) -> Any:
        import torch

        buffer = self._peer_buffers.get(rank)
        if buffer is None or buffer.numel() < count:
            buffer = torch.empty(count, dtype=torch.float32)
            self._peer_buffers[rank] = buffer
        return buffer[:count]

    # -- the collectives -----------------------------------------------------

    def all_reduce_weighted(self, flat: Any, weight: float = 1.0) -> None:
        value = _checked_weight(weight)
        host, view, staged = self._host_buffer(flat)
        count, nbytes = host.numel(), len(view)

        def operation() -> None:
            self._seq += 1
            seq = self._seq
            if self.rank == 0:
                self._reduce_on_root(host, view, count, nbytes, seq, value)
            else:
                link = self._links[0]
                self._send_frame(link, OP_ALL_REDUCE, seq, nbytes, value, view)
                frame = self._read_frame(link)
                self._expect(link, frame, OP_RESULT, seq, nbytes=nbytes)
                self._recv_into(link, view, "the all-reduce result")
            if staged:
                flat.copy_(host)

        self._run(operation)

    def _reduce_on_root(
        self, host: Any, view: memoryview, count: int, nbytes: int, seq: int, weight: float
    ) -> None:
        import torch

        weights: Dict[int, float] = {0: weight}
        buffers: Dict[int, Any] = {}

        def take(link: _Link) -> None:
            frame = self._read_frame(link)
            self._expect(link, frame, OP_ALL_REDUCE, seq, nbytes=nbytes)
            peer_weight = frame[3]
            if not math.isfinite(peer_weight) or peer_weight < 0:
                raise ProtocolError(
                    f"rank {link.peer_rank} sent the weight {peer_weight!r}, which is not a "
                    "finite number of at least 0"
                )
            buffer = self._peer_buffer(link.peer_rank, count)
            self._recv_into(link, memoryview(buffer.numpy()).cast("B"), "its gradient payload")
            weights[link.peer_rank] = peer_weight
            buffers[link.peer_rank] = buffer

        self._fan(self.peers, take)
        total = 0.0
        for rank in range(self.world_size):  # rank order: the sum is the same on every run
            total += weights[rank]
        if not total > 0.0:
            raise ProtocolError(
                "every rank reported a weight of 0 (no token in any rank's window): there is "
                "no gradient to average"
            )
        with torch.no_grad():
            host.mul_(weights[0])
            for rank in range(1, self.world_size):
                host.add_(buffers[rank], alpha=weights[rank])
            host.div_(total)
        self._fan(
            self.peers,
            lambda link: self._send_frame(link, OP_RESULT, seq, nbytes, 0.0, view),
        )

    def broadcast_(self, flat: Any, src: int = 0) -> None:
        if src != 0:
            raise ValueError("the TCP group broadcasts from rank 0 only")
        host, view, staged = self._host_buffer(flat)
        nbytes = len(view)

        def operation() -> None:
            self._seq += 1
            seq = self._seq
            if self.rank == 0:
                self._fan(
                    self.peers,
                    lambda link: self._send_frame(link, OP_BROADCAST, seq, nbytes, 0.0, view),
                )
            else:
                link = self._links[0]
                frame = self._read_frame(link)
                self._expect(link, frame, OP_BROADCAST, seq, nbytes=nbytes)
                self._recv_into(link, view, "the broadcast payload")
                if staged:
                    flat.copy_(host)

        self._run(operation)

    def all_gather_object(self, obj: Any) -> List[Any]:
        mine = encode_object(obj)

        def operation() -> List[Any]:
            self._seq += 1
            seq = self._seq
            if self.rank == 0:
                parts: Dict[int, bytes] = {0: mine}

                def take(link: _Link) -> None:
                    frame = self._read_frame(link)
                    self._expect(link, frame, OP_GATHER, seq, limit=MAX_OBJECT_BYTES)
                    data = bytearray(frame[2])
                    if data:
                        self._recv_into(link, memoryview(data), "its gathered object")
                    parts[link.peer_rank] = bytes(data)

                self._fan(self.peers, take)
                joined = b"[" + b",".join(parts[rank] for rank in range(self.world_size)) + b"]"
                for rank in range(self.world_size):  # validate before anyone else sees it
                    decode_object(parts[rank], source=f"rank {rank}")
                self._fan(
                    self.peers,
                    lambda link: self._send_frame(
                        link, OP_GATHER_RESULT, seq, len(joined), 0.0, memoryview(joined)
                    ),
                )
                decoded = decode_object(joined, source="rank 0")
            else:
                link = self._links[0]
                self._send_frame(link, OP_GATHER, seq, len(mine), 0.0, memoryview(mine))
                frame = self._read_frame(link)
                self._expect(
                    link,
                    frame,
                    OP_GATHER_RESULT,
                    seq,
                    limit=MAX_OBJECT_BYTES * self.world_size + 4096,
                )
                data = bytearray(frame[2])
                self._recv_into(link, memoryview(data), "the gathered objects")
                decoded = decode_object(bytes(data), source="rank 0")
            if not isinstance(decoded, list) or len(decoded) != self.world_size:
                raise ProtocolError("the gather result does not hold one object per rank")
            return decoded

        return self._run(operation)

    def barrier(self) -> None:
        def operation() -> None:
            self._seq += 1
            seq = self._seq
            if self.rank == 0:

                def take(link: _Link) -> None:
                    self._expect(link, self._read_frame(link), OP_BARRIER, seq, nbytes=0)

                self._fan(self.peers, take)
                self._fan(self.peers, lambda link: self._send_frame(link, OP_RELEASE, seq, 0))
            else:
                link = self._links[0]
                self._send_frame(link, OP_BARRIER, seq, 0)
                self._expect(link, self._read_frame(link), OP_RELEASE, seq, nbytes=0)

        self._run(operation)


def open_b1_group(
    env: B1Env,
    *,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    announce: Optional[Callable[[str], None]] = None,
) -> TcpStarGroup:
    """Join (rank 0: form) the group ``env`` describes. Blocks until every rank is connected."""
    return TcpStarGroup(
        rank=env.rank,
        world_size=env.world_size,
        host=env.host,
        port=env.port,
        bind_host=env.bind_host,
        bind_port=env.bind_port,
        token=env.token,
        max_payload_bytes=max_payload_bytes,
        announce=announce,
    )
