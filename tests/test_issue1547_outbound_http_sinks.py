"""#1547 — AST sink-based ratchet for outbound HTTP calls across src/soup_cli.

The #616 ratchet (tests/test_issue616_net_guard_is_shared.py) finds gates by
their syntactic shape: a ``.hostname`` attribute access occurring next to a
``"localhost"`` literal or a ``*LOOPBACK_HOSTS`` identifier within the same
function. An alternative gate written with a different idiom — such as
``urlsplit(u).netloc.split(":")[0]``, ``httpx.URL(u).host``, or a bare
outbound call like ``httpx.post(base + ...)`` with no validation at all —
passed it unchecked.

``src/soup_cli/utils/energy.py:66`` (``validate_electricity_map_endpoint``) was
a concrete instance: its own private ``_LOOPBACK`` set and an ad-hoc IP
predicate that missed abbreviated IPv4 (e.g. ``127.1``) and RFC 6598
carrier-grade NAT address space (``100.64.0.0/10``).

This file implements an inverted sink-based ratchet over the entire
``src/soup_cli/`` tree:
1. Every outbound HTTP library reference (``httpx``, ``urllib.request``, and
   ``requests``) is discovered via a two-pass AST analysis. Instead of
   matching a fixed whitelist of method names, any attribute access or
   from-import not belonging to a small inert set of exceptions / data types
   (e.g. ``HTTPError``, ``Request``, ``URL``, ``Timeout``) is treated as a sink.
2. Every discovered call site ``(relpath, qualname)`` must exist in the explicit
   declared table below, pinning its exact multiset of expected calls and
   naming the shared gate function it relies on or the reason it needs none
   (fixed host, loopback only, operator environment).
3. A newly introduced HTTP sink with no table entry fails the test immediately,
   reporting the file, function, line number, and offending call snippet.
4. An allowlisted entry that disappears, is renamed, or changes its call count
   fails the test, preventing ledger rot and silent additions to existing gates.
5. Every declared gate is verified via AST walk to ensure the named caller
   function actually calls the gate.
6. Path handling is normalized to POSIX style (``.as_posix()``) to ensure
   identical execution on Windows and Linux CI runners.

Limits of this static analysis:
- An untyped client passed in as a parameter (e.g. ``def send(client, u): client.get(u)``)
  cannot be tracked without dynamic type analysis.
- Type annotations are not scanned.
- ``__import__`` reached through builtins / importlib or with keyword arguments (``name=...``)
  is not resolved.
- This test specifically audits ``httpx``, ``urllib.request``, and ``requests``.
  Other connection vectors (``urllib3``, ``http.client``, ``aiohttp``, ``socket``,
  ``websockets``, subprocesses, or SDK-level endpoint configs) are not scanned;
  today ``src/soup_cli`` opens no outbound network connections through those.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

# Inert members of the target HTTP libraries that do not initiate network traffic
# or configure transport pipelines. Any attribute access or from-import not in
# these sets is flagged as an outbound sink reference.
INERT_NAMES: dict[str, frozenset[str]] = {
    "httpx": frozenset({
        "Auth",
        "BasicAuth",
        "ByteStream",
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "CookieConflict",
        "Cookies",
        "DecodingError",
        "DigestAuth",
        "Headers",
        "HTTPError",
        "HTTPStatusError",
        "InvalidURL",
        "Limits",
        "NetRCAuth",
        "NetworkError",
        "PoolTimeout",
        "ProtocolError",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "Request",
        "RequestError",
        "RequestNotRead",
        "Response",
        "ResponseNotRead",
        "StreamConsumed",
        "StreamError",
        "SyncByteStream",
        "Timeout",
        "TimeoutException",
        "TooManyRedirects",
        "TransportError",
        "URL",
        "UnsupportedProtocol",
        "WriteError",
        "WriteTimeout",
        "_api",
        "codes",
    }),
    "urllib.request": frozenset({
        "BaseHandler",
        "HTTPPasswordMgr",
        "HTTPPasswordMgrWithDefaultRealm",
        "HTTPPasswordMgrWithPriorAuth",
        "HTTPRedirectHandler",
        "Request",
    }),
    "requests": frozenset({
        "ConnectionError",
        "ConnectTimeout",
        "HTTPError",
        "JSONDecodeError",
        "ReadTimeout",
        "Request",
        "RequestException",
        "Response",
        "Timeout",
        "TooManyRedirects",
        "URLRequired",
        "api",
        "auth",
        "codes",
        "cookies",
        "exceptions",
        "status_codes",
        "structures",
    }),
}

# Submodules that re-export the sinks: ``httpx._api.post`` is ``httpx.post``.
PASS_THROUGH: dict[str, frozenset[str]] = {
    "httpx": frozenset({"_api"}),
    "requests": frozenset({"api"}),
}


@dataclass(frozen=True)
class DeclaredSink:
    calls: tuple[str, ...]
    gate: tuple[str, str] | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Declared Outbound Sink Ledger
# ---------------------------------------------------------------------------
# Keyed by (relpath relative to src/soup_cli with forward slashes, qualname).
# Every entry documents:
# - calls: exact multiset of expected sink calls (fail-closed if call count changes)
# - gate: (caller_func, gate_func) where caller_func is AST-verified to call gate_func
# - reason: explicit justification for exemptions (fixed host, loopback only, operator env)
DECLARED_SINKS: dict[tuple[str, str], DeclaredSink] = {
    ("commands/generate.py", "_generate_openai"): DeclaredSink(
        calls=("httpx.post",),
        gate=("_generate_openai", "refuse_private_ip_literal"),
        reason=(
            "validates api_base against LOOPBACK_HOSTS (HTTPS required for remote) "
            "and calls net_guard.refuse_private_ip_literal"
        ),
    ),
    ("commands/generate.py", "_generate_server"): DeclaredSink(
        calls=("httpx.post",),
        gate=("_generate_server", "refuse_private_ip_literal"),
        reason=(
            "validates api_base against LOOPBACK_HOSTS (HTTPS required for remote) "
            "and calls net_guard.refuse_private_ip_literal"
        ),
    ),
    ("data/providers/anthropic.py", "generate_anthropic"): DeclaredSink(
        calls=("httpx.post",),
        gate=None,
        reason="fixed host: ANTHROPIC_API_URL is hardcoded to https://api.anthropic.com/v1/messages",
    ),
    ("data/providers/ollama.py", "detect_ollama"): DeclaredSink(
        calls=("httpx.get", "httpx.get"),
        gate=None,
        reason=(
            "loopback only: callers use default http://localhost:11434; "
            "function accepts base_url parameter"
        ),
    ),
    ("data/providers/ollama.py", "generate_ollama"): DeclaredSink(
        calls=("httpx.post",),
        gate=("generate_ollama", "validate_ollama_url"),
        reason="validate_ollama_url restricts base_url to localhost/127.0.0.1/::1",
    ),
    ("data/providers/vllm.py", "generate_vllm"): DeclaredSink(
        calls=("httpx.post",),
        gate=("generate_vllm", "validate_vllm_url"),
        reason=(
            "validate_vllm_url requires HTTPS for remote and calls "
            "net_guard.refuse_private_ip_literal"
        ),
    ),
    ("eval/judge.py", "_judge_request"): DeclaredSink(
        calls=("httpx.post",),
        gate=("validate_judge_api_base", "refuse_private_ip_literal"),
        reason=(
            "JudgeEvaluator.__init__ enforces policy via validate_judge_api_base, "
            "which calls net_guard.refuse_private_ip_literal"
        ),
    ),
    ("ui/app.py", "create_app.chat_send._stream_chat"): DeclaredSink(
        calls=("httpx.stream",),
        gate=("chat_send", "refuse_private_ip_literal"),
        reason=(
            "chat_send validates endpoint (loopback-only HTTP or HTTPS) and "
            "calls net_guard.refuse_private_ip_literal before initiating stream"
        ),
    ),
    ("utils/data_forge.py", "make_judge_provider_fn._anthropic_judge"): DeclaredSink(
        calls=("httpx.post",),
        gate=None,
        reason=(
            "fixed host: _ANTHROPIC_MESSAGES_URL is hardcoded to "
            "https://api.anthropic.com/v1/messages"
        ),
    ),
    ("utils/data_forge.py", "make_judge_provider_fn._ollama_judge"): DeclaredSink(
        calls=("httpx.post",),
        gate=("make_judge_provider_fn", "validate_ollama_url"),
        reason="make_judge_provider_fn enforces validate_ollama_url before returning judge closure",
    ),
    ("utils/data_forge.py", "make_judge_provider_fn._vllm_judge"): DeclaredSink(
        calls=("httpx.post",),
        gate=("make_judge_provider_fn", "validate_vllm_url"),
        reason="make_judge_provider_fn enforces validate_vllm_url before returning judge closure",
    ),
    ("utils/ingest_pull.py", "_urllib_transport"): DeclaredSink(
        calls=("urllib.request.build_opener",),
        gate=("resolve_langfuse_host", "validate_webhook_url"),
        reason=(
            "resolve_langfuse_host routes host through validate_webhook_url "
            "(HTTPS only, rejects private hosts unless explicitly opted in)"
        ),
    ),
    ("utils/loop_stages.py", "_post_activate"): DeclaredSink(
        calls=("httpx.post",),
        gate=("deploy_to_canary", "validate_webhook_url"),
        reason=(
            "operator environment: deploy_to_canary validates endpoint via "
            "validate_webhook_url and restricts target to local/LAN via _endpoint_is_local"
        ),
    ),
    ("utils/magpie.py", "make_magpie_generate_fn._ollama_generate"): DeclaredSink(
        calls=("httpx.post",),
        gate=("make_magpie_generate_fn", "validate_ollama_url"),
        reason="make_magpie_generate_fn enforces validate_ollama_url (loopback only)",
    ),
    ("utils/magpie.py", "make_magpie_generate_fn._vllm_generate"): DeclaredSink(
        calls=("httpx.post",),
        gate=("make_magpie_generate_fn", "validate_vllm_url"),
        reason=(
            "make_magpie_generate_fn enforces validate_vllm_url (calls "
            "net_guard.refuse_private_ip_literal)"
        ),
    ),
    ("utils/sglang.py", "generate_with_runtime"): DeclaredSink(
        calls=("httpx.post",),
        gate=None,
        reason="loopback only: target is the in-process local SGLang Runtime server (runtime.url)",
    ),
    ("utils/trackers.py", "send_telemetry_payload"): DeclaredSink(
        calls=("urllib.request.urlopen",),
        gate=("_resolve_posthog_target", "_telemetry_endpoint_is_safe"),
        reason=(
            "_resolve_posthog_target enforces _telemetry_endpoint_is_safe "
            "(HTTPS only, pre- and post-DNS private-IP refusal)"
        ),
    ),
    ("utils/webhooks.py", "post_webhook"): DeclaredSink(
        calls=("httpx.post",),
        gate=("post_webhook", "validate_webhook_url"),
        reason=(
            "post_webhook validates URL via validate_webhook_url (HTTPS for remote, "
            "calls net_guard.is_private_or_link_local)"
        ),
    ),
}


def _resolve_call_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "import_module" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in ("httpx", "requests", "urllib.request"):
                        return arg.value
                    if arg.value == "urllib":
                        return "urllib"
        elif isinstance(node.func, ast.Name):
            if node.func.id in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in ("httpx", "requests", "urllib.request"):
                        return arg.value
                    if arg.value == "urllib":
                        return "urllib"
    return None


# ---------------------------------------------------------------------------
# Two-Pass Inverted AST Outbound Sink Scanner
# ---------------------------------------------------------------------------
class _ModuleCollector(ast.NodeVisitor):
    """Pass 1: Collect all module and from-import bindings across the file."""

    def __init__(self) -> None:
        self.module_bindings: dict[str, str] = {}
        self.from_bindings: dict[str, tuple[str, str]] = {}
        self.star_imports: list[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "httpx" or alias.name.startswith("httpx."):
                if alias.asname:
                    self.module_bindings[alias.asname] = "httpx"
                else:
                    self.module_bindings["httpx"] = "httpx"
            elif alias.name == "requests" or alias.name.startswith("requests."):
                if alias.asname:
                    self.module_bindings[alias.asname] = "requests"
                else:
                    self.module_bindings["requests"] = "requests"
            elif alias.name == "urllib.request":
                if alias.asname:
                    self.module_bindings[alias.asname] = "urllib.request"
                else:
                    self.module_bindings["urllib"] = "urllib"
            elif alias.name == "urllib":
                self.module_bindings[alias.asname or "urllib"] = "urllib"
            elif alias.name.startswith("urllib.") and not alias.asname:
                self.module_bindings["urllib"] = "urllib"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        root, _, sub = mod.partition(".")
        sub = sub.split(".")[0]
        if root in PASS_THROUGH and sub in INERT_NAMES[root] and sub not in PASS_THROUGH[root]:
            return
        if mod == "httpx" or mod.startswith("httpx."):
            for alias in node.names:
                if alias.name == "*":
                    self.star_imports.append((node.lineno, "httpx"))
                else:
                    target = alias.asname or alias.name
                    if alias.name not in INERT_NAMES["httpx"]:
                        self.from_bindings[target] = ("httpx", alias.name)
        elif mod == "requests" or mod.startswith("requests."):
            for alias in node.names:
                if alias.name == "*":
                    self.star_imports.append((node.lineno, "requests"))
                else:
                    target = alias.asname or alias.name
                    if alias.name not in INERT_NAMES["requests"]:
                        self.from_bindings[target] = ("requests", alias.name)
        elif mod == "urllib":
            for alias in node.names:
                target = alias.asname or alias.name
                if alias.name == "request":
                    self.module_bindings[target] = "urllib.request"
        elif mod == "urllib.request":
            for alias in node.names:
                if alias.name == "*":
                    self.star_imports.append((node.lineno, "urllib.request"))
                else:
                    target = alias.asname or alias.name
                    if alias.name not in INERT_NAMES["urllib.request"]:
                        self.from_bindings[target] = ("urllib.request", alias.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        mod = _resolve_call_module(node.value)
        if mod:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.module_bindings[target.id] = mod
        self.generic_visit(node)


class _SinkVisitor(ast.NodeVisitor):
    """Pass 2: Traverse AST with qualname stack and report any non-inert references."""

    def __init__(self, bindings: _ModuleCollector) -> None:
        self.b = bindings
        self.scope_stack: list[str] = []
        self.shadowed_stack: list[set[str]] = []
        self.sinks: list[tuple[str, int, str]] = []
        self._qualified: set[int] = set()

    def _scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"

    def _is_shadowed(self, name: str) -> bool:
        return any(name in shadowed for shadowed in self.shadowed_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # bases, keywords and decorators are evaluated in the enclosing scope
        for outer in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(outer)
        self.scope_stack.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def _enter_function(self, name: str, args: ast.arguments) -> None:
        self.scope_stack.append(name)
        params = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)
        self.shadowed_stack.append(params)

    def _exit_function(self) -> None:
        self.shadowed_stack.pop()
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        self._enter_function(node.name, node.args)
        for stmt in node.body:
            self.visit(stmt)
        self._exit_function()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        self._enter_function(node.name, node.args)
        for stmt in node.body:
            self.visit(stmt)
        self._exit_function()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def _resolve_module(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if self._is_shadowed(node.id):
                return None
            self._qualified.add(id(node))
            return self.b.module_bindings.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = self._resolve_module(node.value)
            if parent == "urllib" and node.attr == "request":
                return "urllib.request"
            if parent in ("httpx", "requests") and node.attr in PASS_THROUGH[parent]:
                return parent
        call_mod = _resolve_call_module(node)
        if call_mod:
            return call_mod
        return None

    def _resolve_attr(self, node: ast.AST) -> tuple[str | None, str | None]:
        if isinstance(node, ast.Attribute):
            mod = self._resolve_module(node.value)
            if mod:
                return mod, node.attr
        return None, None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        mod, attr = self._resolve_attr(node)
        if mod and mod != "urllib":
            if attr not in INERT_NAMES.get(mod, frozenset()):
                self.sinks.append((self._scope(), node.lineno, f"{mod}.{attr}"))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load) or self._is_shadowed(node.id):
            return
        if node.id in self.b.from_bindings:
            mod, attr = self.b.from_bindings[node.id]
            self.sinks.append((self._scope(), node.lineno, f"{mod}.{attr}"))
        elif node.id in self.b.module_bindings and id(node) not in self._qualified:
            mod = self.b.module_bindings[node.id]
            self.sinks.append((self._scope(), node.lineno, f"{mod} (module object)"))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        is_getattr = (
            (isinstance(node.func, ast.Name) and node.func.id == "getattr")
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
                and node.func.attr == "getattr"
            )
        )
        if is_getattr and len(node.args) >= 2:
            mod = self._resolve_module(node.args[0])
            if mod and mod != "urllib":
                arg1 = node.args[1]
                if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
                    if arg1.value not in INERT_NAMES.get(mod, frozenset()):
                        self.sinks.append((self._scope(), node.lineno, f"{mod}.{arg1.value}"))
                else:
                    self.sinks.append((self._scope(), node.lineno, f"{mod}.<getattr>"))

        call_mod = _resolve_call_module(node)
        if call_mod and call_mod != "urllib":
            self.sinks.append((self._scope(), node.lineno, f"{call_mod}.import_module"))

        self.generic_visit(node)


def find_outbound_sinks(source: str) -> list[tuple[str, int, str]]:
    """Return ``[(qualname, lineno, call_repr)]`` for all outbound sinks in source."""
    tree = ast.parse(source)
    collector = _ModuleCollector()
    collector.visit(tree)
    visitor = _SinkVisitor(collector)
    for lineno, mod in collector.star_imports:
        visitor.sinks.append((visitor._scope(), lineno, f"from {mod} import *"))
    visitor.visit(tree)
    return visitor.sinks


def scan_src_sinks(src_dir: Path = SRC) -> list[tuple[str, str, int, str]]:
    """Scan all python files under ``src_dir``.

    Returns a list of ``(relpath, qualname, line, call)`` tuples.
    """
    hits: list[tuple[str, str, int, str]] = []
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(src_dir).as_posix()
        source = path.read_text(encoding="utf-8")
        for qualname, lineno, call_repr in find_outbound_sinks(source):
            hits.append((rel, qualname, lineno, call_repr))
    return hits


# ---------------------------------------------------------------------------
# Ratchet Test Helpers
# ---------------------------------------------------------------------------
def unlisted_sinks(
    hits: list[tuple[str, str, int, str]],
    table: dict[tuple[str, str], DeclaredSink],
) -> list[str]:
    """Return error strings for call sites that are not declared or whose calls mismatch."""
    errors: list[str] = []
    hits_by_key: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for rel, qualname, lineno, call in hits:
        hits_by_key.setdefault((rel, qualname), []).append((lineno, call))

    for (rel, qualname), calls_in_func in hits_by_key.items():
        if (rel, qualname) not in table:
            for lineno, call in calls_in_func:
                errors.append(f"{rel}:{lineno} in {qualname}() calling {call}")
        else:
            declared_calls = table[(rel, qualname)].calls
            actual_calls = tuple(call for _, call in calls_in_func)
            if Counter(actual_calls) != Counter(declared_calls):
                first_line = calls_in_func[0][0]
                errors.append(
                    f"{rel}:{first_line} in {qualname}() calls mismatch: "
                    f"expected {sorted(declared_calls)}, found {sorted(actual_calls)}"
                )
    return sorted(errors)


def dead_entries(
    hits: list[tuple[str, str, int, str]],
    table: dict[tuple[str, str], DeclaredSink],
) -> list[str]:
    """Return error strings for table entries that have no live sinks in code."""
    live_keys = {(rel, qualname) for rel, qualname, _lineno, _call in hits}
    dead: list[str] = []
    for (rel, qualname), entry in sorted(table.items()):
        if (rel, qualname) not in live_keys:
            reason = entry.reason or (f"gate={entry.gate}" if entry.gate else "")
            dead.append(f"{rel}::{qualname} -> {reason}")
    return dead


# ---------------------------------------------------------------------------
# Ratchet Test Suite & Invariants
# ---------------------------------------------------------------------------
class TestOutboundHttpSinksAreDeclared:
    def test_every_outbound_http_sink_is_declared(self) -> None:
        """Every httpx, urllib.request and requests sink call must match DECLARED_SINKS."""
        hits = scan_src_sinks()
        unlisted = unlisted_sinks(hits, DECLARED_SINKS)
        assert not unlisted, (
            "Found outbound HTTP sinks in src/soup_cli that are not declared in DECLARED_SINKS, "
            "or whose call counts mismatch:\n  " + "\n  ".join(unlisted)
        )

    def test_every_declared_sink_is_still_present(self) -> None:
        """Every entry in DECLARED_SINKS must match live call sites in code."""
        hits = scan_src_sinks()
        dead = dead_entries(hits, DECLARED_SINKS)
        assert not dead, (
            "These entries in DECLARED_SINKS no longer have any matching outbound HTTP sinks "
            "in src/soup_cli. Remove or update them so the ledger cannot rot:\n  "
            + "\n  ".join(dead)
        )

    def test_every_declared_gate_exists_and_is_called(self) -> None:
        """Every entry with a gate tuple must name a real function that calls the gate."""
        for (rel, qualname), entry in sorted(DECLARED_SINKS.items()):
            if entry.gate is None:
                assert entry.reason, f"{rel}::{qualname} has no gate and no reason"
                assert entry.reason.startswith(
                    ("fixed host:", "loopback only:", "operator environment:")
                ), (
                    f"{rel}::{qualname} reason must start with a recognized category: "
                    f"{entry.reason}"
                )
                continue

            caller_name, gate_name = entry.gate
            path = SRC / rel
            tree = ast.parse(path.read_text(encoding="utf-8"))
            matching_callers = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == caller_name
            ]
            assert matching_callers, f"Caller function {caller_name!r} not found in {rel}"

            gate_called = False
            for caller_node in matching_callers:
                for child in ast.walk(caller_node):
                    if isinstance(child, ast.Call):
                        called = None
                        if isinstance(child.func, ast.Name):
                            called = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            called = child.func.attr
                        if called == gate_name:
                            gate_called = True
                            break
                if gate_called:
                    break

            assert gate_called, (
                f"In {rel}, function {caller_name}() was declared as the gate for {qualname}(), "
                f"but it does not call {gate_name}()"
            )

    def test_scan_actually_covers_src(self) -> None:
        """Breadth control: verify the scanner actually walks src/soup_cli and finds modules."""
        scanned = list(SRC.rglob("*.py"))
        assert len(scanned) >= 400, f"only {len(scanned)} modules scanned in {SRC}"
        hits = scan_src_sinks()
        discovered_keys = {(rel, qualname) for rel, qualname, _lineno, _call in hits}
        assert discovered_keys == set(DECLARED_SINKS)


# ---------------------------------------------------------------------------
# Energy Endpoint SSRF Hardening Parity & Pin Tests
# ---------------------------------------------------------------------------
class TestEnergyEndpointSSRFHardening:
    """Verifies validate_electricity_map_endpoint uses net_guard and shared policy."""

    def test_energy_does_not_declare_private_loopback_set(self) -> None:
        """Pins that energy.py does not define its own _LOOPBACK set."""
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        assigned = [
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        assert "_LOOPBACK" not in assigned, "energy.py must use net_guard.LOOPBACK_HOSTS"

    def test_energy_validator_calls_the_shared_refusal(self) -> None:
        """Pins that validate_electricity_map_endpoint calls net_guard.refuse_private_ip_literal."""
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        funcs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_electricity_map_endpoint"
        ]
        assert len(funcs) == 1
        called = {
            node.func.id
            for node in ast.walk(funcs[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_private_ip_literal" in called

    def test_energy_imports_shared_loopback_hosts(self) -> None:
        """Pins that energy.py imports LOOPBACK_HOSTS from net_guard."""
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        imported = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "soup_cli.utils.net_guard"
            and any(alias.name == "LOOPBACK_HOSTS" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imported, "energy.py must import LOOPBACK_HOSTS from soup_cli.utils.net_guard"

    def test_energy_has_no_rebuilt_loopback_literals(self) -> None:
        """Pins that energy.py does not define collection literals with 2+ loopback strings."""
        loopback_strings = frozenset({"localhost", "127.0.0.1", "::1"})
        energy_path = SRC / "utils" / "energy.py"
        tree = ast.parse(energy_path.read_text(encoding="utf-8"))
        rebuilt = []
        for node in ast.walk(tree):
            elts = None
            if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
                elts = node.elts
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("frozenset", "tuple", "set", "list")
                and node.args
                and isinstance(node.args[0], (ast.Tuple, ast.Set, ast.List))
            ):
                elts = node.args[0].elts
            if elts is None:
                continue
            hosts = {
                elt.value
                for elt in elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
            if len(hosts & loopback_strings) >= 2:
                rebuilt.append(node.lineno)
        assert rebuilt == [], f"energy.py rebuilds loopback set at lines: {rebuilt}"

    def test_validate_electricity_map_endpoint_loopback_and_remote(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        assert (
            validate_electricity_map_endpoint("http://localhost:8080/co2")
            == "http://localhost:8080/co2"
        )
        assert (
            validate_electricity_map_endpoint("http://127.0.0.1:8080/co2")
            == "http://127.0.0.1:8080/co2"
        )
        assert (
            validate_electricity_map_endpoint("http://[::1]:8080/co2")
            == "http://[::1]:8080/co2"
        )
        # 127.0.0.2 and IPv4-mapped IPv6 loopback were refused by the old predicate
        # on HTTPS; they are now accepted under the shared loopback policy
        assert (
            validate_electricity_map_endpoint("https://127.0.0.2/co2")
            == "https://127.0.0.2/co2"
        )
        assert (
            validate_electricity_map_endpoint("https://[::ffff:127.0.0.1]/co2")
            == "https://[::ffff:127.0.0.1]/co2"
        )
        assert (
            validate_electricity_map_endpoint(
                "https://api.electricitymap.org/v3/carbon-intensity/latest"
            )
            == "https://api.electricitymap.org/v3/carbon-intensity/latest"
        )

    def test_validate_electricity_map_endpoint_rejects_zero_address(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        with pytest.raises(ValueError, match="0.0.0.0 endpoints are rejected"):
            validate_electricity_map_endpoint("http://0.0.0.0:8080/co2")
        with pytest.raises(ValueError, match="0.0.0.0 endpoints are rejected"):
            validate_electricity_map_endpoint("https://0.0.0.0/co2")

    def test_validate_electricity_map_endpoint_rejects_abbreviated_ipv4(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        # Abbreviated IPv4 (127.1, 2130706433) are rejected on plain HTTP
        # because they are not in LOOPBACK_HOSTS (loopback-only plain HTTP)
        with pytest.raises(ValueError, match="http:// only permitted for loopback hosts"):
            validate_electricity_map_endpoint("http://127.1/co2")
        with pytest.raises(ValueError, match="http:// only permitted for loopback hosts"):
            validate_electricity_map_endpoint("http://2130706433/co2")

        # Abbreviated private IPv4 (0x0a000001 -> 10.0.0.1) on HTTPS is rejected by
        # net_guard.refuse_private_ip_literal
        with pytest.raises(
            ValueError, match="private/link-local/reserved IP hosts are not allowed"
        ):
            validate_electricity_map_endpoint("https://0x0a000001/co2")

    def test_validate_electricity_map_endpoint_rejects_rfc6598_cgnat(self) -> None:
        from soup_cli.utils.energy import validate_electricity_map_endpoint

        # 100.64.0.0/10 carrier-grade NAT space was accepted by the old ip.is_private check;
        # net_guard.refuse_private_ip_literal rejects it
        with pytest.raises(ValueError, match="http:// only permitted for loopback hosts"):
            validate_electricity_map_endpoint("http://100.64.0.1/co2")
        with pytest.raises(
            ValueError, match="private/link-local/reserved IP hosts are not allowed"
        ):
            validate_electricity_map_endpoint("https://100.64.0.1/co2")


# ---------------------------------------------------------------------------
# Mutations Caught In Addition To The #616 Guard
# ---------------------------------------------------------------------------
class TestMutationsCaughtInAdditionToTheOldGuard:
    """Demonstrates, not just asserts: the shapes from #1547 caught in addition to the #616
    shape-based scan."""

    def test_shape1_energy_custom_loopback_set_is_caught(self, tmp_path: Path) -> None:
        """Old energy.py used a private _LOOPBACK set instead of *LOOPBACK_HOSTS."""
        code = (
            "import httpx\n"
            "from urllib.parse import urlsplit\n"
            "_LOOPBACK = frozenset({'localhost', '127.0.0.1', '::1'})\n\n"
            "def validate_and_fetch(endpoint):\n"
            "    parts = urlsplit(endpoint)\n"
            "    if parts.hostname in _LOOPBACK:\n"
            "        return httpx.get(endpoint)\n"
        )
        module = tmp_path / "rogue_energy_shape.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_energy_shape.py", "validate_and_fetch", 8, "httpx.get")]

    def test_shape2_netloc_split_gate_is_caught(self, tmp_path: Path) -> None:
        """urlsplit(u).netloc.split(':')[0] parsed host without accessing .hostname."""
        code = (
            "import httpx\n"
            "from urllib.parse import urlsplit\n\n"
            "def netloc_gate(url):\n"
            "    host = urlsplit(url).netloc.split(':')[0]\n"
            "    if host == 'localhost':\n"
            "        return httpx.post(url, json={})\n"
        )
        module = tmp_path / "rogue_netloc.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_netloc.py", "netloc_gate", 7, "httpx.post")]

    def test_shape3_httpx_url_host_is_caught(self, tmp_path: Path) -> None:
        """httpx.URL(u).host extracted host without urllib .hostname access."""
        code = (
            "import httpx\n\n"
            "def httpx_url_gate(url):\n"
            "    if httpx.URL(url).host == 'localhost':\n"
            "        return httpx.post(url, json={})\n"
        )
        module = tmp_path / "rogue_httpx_url.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_httpx_url.py", "httpx_url_gate", 5, "httpx.post")]

    def test_shape4_module_allowlist_under_another_name_is_caught(self, tmp_path: Path) -> None:
        """A module-level allowlist under another name like _ALLOWED_HOSTS."""
        code = (
            "import httpx\n"
            "_ALLOWED_HOSTS = frozenset({'127.0.0.1'})\n\n"
            "def custom_allowlist_gate(url):\n"
            "    from urllib.parse import urlparse\n"
            "    if urlparse(url).hostname in _ALLOWED_HOSTS:\n"
            "        return httpx.post(url)\n"
        )
        module = tmp_path / "rogue_allowlist.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_allowlist.py", "custom_allowlist_gate", 7, "httpx.post")]

    def test_shape5_bare_outbound_call_with_no_check_is_caught(self, tmp_path: Path) -> None:
        """A bare httpx.post(...) with no validation check at all."""
        code = (
            "import httpx\n\n"
            "def send_webhook_unvalidated(base, payload):\n"
            "    httpx.post(base + '/events', json=payload)\n"
        )
        module = tmp_path / "rogue_bare.py"
        module.write_text(code, encoding="utf-8")
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_bare.py", "send_webhook_unvalidated", 4, "httpx.post")]


# ---------------------------------------------------------------------------
# Negative Self-Tests: Prove the Scanner Catches Undeclared Calls & Aliases
# ---------------------------------------------------------------------------
class TestTheScannerCanActuallyFail:
    def test_an_undeclared_httpx_post_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue.py"
        module.write_text(
            "import httpx\n\n"
            "def send_data(url, data):\n"
            "    httpx.post(url, json=data)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue.py", "send_data", 4, "httpx.post")]

    def test_an_undeclared_urllib_urlopen_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_urllib.py"
        module.write_text(
            "import urllib.request\n\n"
            "def fetch(url):\n"
            "    return urllib.request.urlopen(url)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_urllib.py", "fetch", 4, "urllib.request.urlopen")]

    def test_an_undeclared_httpx_client_is_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_client.py"
        module.write_text(
            "import httpx\n\n"
            "def get_client():\n"
            "    return httpx.Client()\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        assert sinks == [("rogue_client.py", "get_client", 4, "httpx.Client")]

    def test_import_aliases_are_caught(self, tmp_path: Path) -> None:
        module = tmp_path / "rogue_alias.py"
        module.write_text(
            "import httpx as h\n"
            "from httpx import post as my_post\n"
            "from urllib import request as req\n"
            "from urllib.request import urlopen as my_urlopen\n"
            "from requests import get as req_get\n\n"
            "def call_aliased(url):\n"
            "    h.get(url)\n"
            "    my_post(url)\n"
            "    req.urlopen(url)\n"
            "    my_urlopen(url)\n"
            "    req_get(url)\n",
            encoding="utf-8",
        )
        sinks = scan_src_sinks(tmp_path)
        calls = [(func, call) for _rel, func, _line, call in sinks]
        assert calls == [
            ("call_aliased", "httpx.get"),
            ("call_aliased", "httpx.post"),
            ("call_aliased", "urllib.request.urlopen"),
            ("call_aliased", "urllib.request.urlopen"),
            ("call_aliased", "requests.get"),
        ]

    def test_dead_entry_fails_ratchet_assertion(self) -> None:
        dummy_ledger = {
            ("fake/module.py", "fake_func"): DeclaredSink(
                calls=("httpx.post",), reason="justification"
            )
        }
        mock_hits = [("real/module.py", "real_func", 1, "httpx.post")]
        dead = dead_entries(mock_hits, dummy_ledger)
        assert dead == ["fake/module.py::fake_func -> justification"]

    def test_unlisted_sink_fails_ratchet_assertion(self) -> None:
        dummy_ledger = {
            ("real/module.py", "real_func"): DeclaredSink(calls=("httpx.post",), reason="ok")
        }
        mock_hits = [("unlisted/module.py", "rogue_func", 42, "httpx.post")]
        unlisted = unlisted_sinks(mock_hits, dummy_ledger)
        assert unlisted == ["unlisted/module.py:42 in rogue_func() calling httpx.post"]

    def test_call_multiset_mismatch_fails_ratchet_assertion(self) -> None:
        dummy_ledger = {
            ("mod.py", "f"): DeclaredSink(calls=("httpx.post",), reason="ok")
        }
        mock_hits = [
            ("mod.py", "f", 10, "httpx.post"),
            ("mod.py", "f", 15, "httpx.post"),
        ]
        unlisted = unlisted_sinks(mock_hits, dummy_ledger)
        assert unlisted == [
            "mod.py:10 in f() calls mismatch: "
            "expected ['httpx.post'], found ['httpx.post', 'httpx.post']"
        ]


# ---------------------------------------------------------------------------
# Review of #1646: spellings of the same outbound call the scanner must also see
# ---------------------------------------------------------------------------
_OTHER_SPELLINGS = {
    "httpx: import below the function that uses it": (
        "def f(u):\n    httpx.post(u)\n\nimport httpx\n"
    ),
    "httpx: import in a sibling function (global)": (
        "def f(u):\n    return httpx.post(u)\n\ndef _load():\n    global httpx\n    import httpx\n"
    ),
    "httpx: module-level alias": "import httpx\n_post = httpx.post\ndef f(u):\n    _post(u)\n",
    "httpx: local alias": "import httpx\ndef f(u):\n    send = httpx.post\n    send(u)\n",
    "httpx: getattr with a literal": "import httpx\ndef f(u):\n    getattr(httpx, 'post')(u)\n",
    "httpx: getattr with a variable": "import httpx\ndef f(u, m):\n    getattr(httpx, m)(u)\n",
    "httpx: functools.partial": (
        "import functools\nimport httpx\ndef f(u):\n"
        "    functools.partial(httpx.post, timeout=5)(u)\n"
    ),
    "httpx: default argument": "import httpx\ndef f(u, poster=httpx.post):\n    poster(u)\n",
    "httpx: dict dispatch": (
        "import httpx\ndef f(u, m):\n    {'post': httpx.post, 'get': httpx.get}[m](u)\n"
    ),
    "httpx: importlib.import_module": (
        "import importlib\ndef f(u):\n    importlib.import_module('httpx').post(u)\n"
    ),
    "httpx: __import__": "def f(u):\n    __import__('httpx').post(u)\n",
    "httpx: subclass of Client": (
        "import httpx\nclass C(httpx.Client):\n    pass\ndef f(u):\n    C().post(u)\n"
    ),
    "httpx: unbound Client.post": "import httpx\ndef f(c, u):\n    httpx.Client.post(c, u)\n",
    "httpx: star import": "from httpx import *\ndef f(u):\n    post(u)\n",
    "httpx: private submodule": "import httpx._api\ndef f(u):\n    httpx._api.post(u)\n",
    "urllib: package alias": (
        "import urllib as ul\nimport urllib.request\ndef f(u):\n    ul.request.urlopen(u)\n"
    ),
    "urllib: OpenerDirector": (
        "import urllib.request\ndef f(u):\n    o = urllib.request.OpenerDirector()\n    o.open(u)\n"
    ),
    "urllib: urlretrieve": (
        "import urllib.request\ndef f(u):\n    urllib.request.urlretrieve(u, 'x')\n"
    ),
    "urllib: from-import urlretrieve": (
        "from urllib.request import urlretrieve\ndef f(u):\n    urlretrieve(u, 'x')\n"
    ),
    "requests: Session()": "import requests\ndef f(u):\n    requests.Session().get(u)\n",
    "requests: session()": "import requests\ndef f(u):\n    requests.session().get(u)\n",
    "requests: from-import Session": (
        "from requests import Session\ndef f(u):\n    Session().get(u)\n"
    ),
    "requests: requests.api.get": "import requests\ndef f(u):\n    requests.api.get(u)\n",
    "requests: getattr": "import requests\ndef f(u):\n    getattr(requests, 'get')(u)\n",
}


@pytest.mark.parametrize("spelling", sorted(_OTHER_SPELLINGS))
def test_every_spelling_of_an_outbound_call_is_seen(spelling: str) -> None:
    assert find_outbound_sinks(_OTHER_SPELLINGS[spelling]), spelling


_BRANCHES_NO_TEST_REACHES = {
    "httpx.request": ("import httpx\ndef f(u):\n    httpx.request('GET', u)\n", "httpx.request"),
    "httpx.AsyncClient": (
        "import httpx\nasync def f(u):\n"
        "    async with httpx.AsyncClient() as c:\n        await c.get(u)\n",
        "httpx.AsyncClient",
    ),
    "requests.post through the module": (
        "import requests\ndef f(u):\n    requests.post(u)\n",
        "requests.post",
    ),
    "build_opener under another variable name": (
        "import urllib.request\ndef f(u):\n    director = urllib.request.build_opener()\n"
        "    return director.open(u)\n",
        "urllib.request.build_opener",
    ),
}


@pytest.mark.parametrize("case", sorted(_BRANCHES_NO_TEST_REACHES))
def test_each_branch_of_the_scanner_is_reached(case: str) -> None:
    code, expected = _BRANCHES_NO_TEST_REACHES[case]
    found = find_outbound_sinks(code)
    assert ("f", expected) in [(func, call) for func, _line, call in found], found


def test_two_functions_with_one_name_are_two_keys(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "import httpx\n"
        "class A:\n    def send(self, u):\n        httpx.post(u)\n"
        "class B:\n    def send(self, u):\n        httpx.post(u)\n",
        encoding="utf-8",
    )
    keys = {(rel, func) for rel, func, _line, _call in scan_src_sinks(tmp_path)}
    assert len(keys) == 2, keys


def test_a_file_handle_named_opener_is_not_a_sink() -> None:
    code = (
        "import zipfile\ndef f(p):\n"
        "    opener = zipfile.ZipFile(p)\n    return opener.open('a.txt')\n"
    )
    assert find_outbound_sinks(code) == []


def test_a_file_that_does_not_parse_is_an_error_not_an_empty_result() -> None:
    with pytest.raises(SyntaxError):
        find_outbound_sinks("import httpx\ndef f(:\n    httpx.post(u)\n")


def test_assigned_importlib_call_is_caught() -> None:
    code = (
        "import importlib\n"
        "def f(u):\n"
        "    mod = importlib.import_module('httpx')\n"
        "    mod.post(u)\n"
    )
    sinks = find_outbound_sinks(code)
    calls = [call for _func, _line, call in sinks]
    assert "httpx.post" in calls or "httpx.import_module" in calls, sinks


def test_standalone_importlib_call_is_caught() -> None:
    code = (
        "import importlib\n"
        "def f():\n"
        "    importlib.import_module('httpx')\n"
    )
    sinks = find_outbound_sinks(code)
    assert any(call == "httpx.import_module" for _func, _line, call in sinks)


def test_type_annotations_are_not_flagged_as_sinks() -> None:
    code = (
        "import httpx\n"
        "def f(client: httpx.Client, u: str) -> None:\n"
        "    pass\n"
    )
    assert find_outbound_sinks(code) == []


def test_shadowed_parameter_name_is_not_flagged() -> None:
    code = (
        "from httpx import post\n"
        "def f(post: str) -> str:\n"
        "    return post\n"
    )
    assert find_outbound_sinks(code) == []


# ---------------------------------------------------------------------------
# Review of #1646, round 2
# ---------------------------------------------------------------------------
_ROUND2_SPELLINGS = {
    "urllib: a handler called directly": (
        "import urllib.request\ndef f(u):\n    req = urllib.request.Request(u)\n"
        "    return urllib.request.HTTPSHandler().https_open(req)\n"
    ),
    "urllib: the file imports only urllib.parse": (
        "import urllib.parse\ndef f(u):\n    return urllib.request.urlopen(u)\n"
    ),
    "httpx: module rebound to another name": (
        "import httpx\ndef f(u):\n    lib = httpx\n    return lib.post(u)\n"
    ),
    "httpx: module passed as an argument": (
        "import httpx\ndef g(lib, u):\n    return lib.post(u)\n"
        "def f(u):\n    return g(httpx, u)\n"
    ),
    "httpx: helper returns the module": (
        "def _hx():\n    import httpx\n    return httpx\ndef f(u):\n    return _hx().post(u)\n"
    ),
}


@pytest.mark.parametrize("spelling", sorted(_ROUND2_SPELLINGS))
def test_round2_spellings_are_seen(spelling: str) -> None:
    assert find_outbound_sinks(_ROUND2_SPELLINGS[spelling]), spelling


# value: (code, functions that do hold a real reference and may be reported)
_NOT_SINKS = {
    "httpx.TimeoutException": (
        "import httpx\ndef f():\n    try:\n        pass\n"
        "    except httpx.TimeoutException:\n        pass\n",
        set(),
    ),
    "requests.JSONDecodeError": (
        "import requests\ndef f():\n    try:\n        pass\n"
        "    except requests.JSONDecodeError:\n        pass\n",
        set(),
    ),
    "httpx.codes.OK": (
        "import httpx\ndef f(r):\n    return r.status_code == httpx.codes.OK\n",
        set(),
    ),
    "requests.exceptions.SSLError": (
        "import requests\ndef f():\n    try:\n        pass\n"
        "    except requests.exceptions.SSLError:\n        pass\n",
        set(),
    ),
    "from requests.exceptions import SSLError": (
        "from requests.exceptions import SSLError\ndef f():\n    try:\n        pass\n"
        "    except SSLError:\n        pass\n",
        set(),
    ),
    "a parameter named request next to `from urllib import request`": (
        "def a(u):\n    from urllib import request\n    return request.urlopen(u)\n"
        "def b(request):\n    return request.url\n",
        {"a"},
    ),
    "a name that is only assigned": (
        "def a(u):\n    from httpx import post\n    return post(u)\n"
        "def b(blog):\n    post = blog.latest()\n",
        {"a"},
    ),
}


@pytest.mark.parametrize("case", sorted(_NOT_SINKS))
def test_these_are_not_reported(case: str) -> None:
    code, real = _NOT_SINKS[case]
    assert [hit for hit in find_outbound_sinks(code) if hit[0] not in real] == []


def test_a_class_base_is_reported_once_in_the_enclosing_scope() -> None:
    code = (
        "import httpx\n"
        "class C(httpx.Client):\n"
        "    def get(self, u):\n"
        "        pass\n"
        "    def post(self, u):\n"
        "        pass\n"
    )
    hits = find_outbound_sinks(code)
    assert hits == [("<module>", 2, "httpx.Client")]


# ---------------------------------------------------------------------------
# Branch coverage pins for Round 2 review
# ---------------------------------------------------------------------------
def test_from_private_submodule_import_is_caught() -> None:
    code = "from httpx._api import post\ndef f(u):\n    post(u)\n"
    assert ("f", "httpx.post") in [(func, call) for func, _line, call in find_outbound_sinks(code)]


def test_builtins_getattr_is_caught() -> None:
    code = "import builtins\nimport httpx\ndef f(u):\n    builtins.getattr(httpx, 'post')(u)\n"
    calls = [(func, call) for func, _line, call in find_outbound_sinks(code)]
    assert ("f", "httpx.post") in calls


def test_decorator_referencing_sink_is_caught() -> None:
    code = "import httpx\ndef dec(fn):\n    return fn\n@dec(httpx.post)\ndef f():\n    pass\n"
    calls = [(func, call) for func, _line, call in find_outbound_sinks(code)]
    assert ("<module>", "httpx.post") in calls


def test_variable_annotation_is_not_flagged() -> None:
    code = "import httpx\ndef f():\n    client: httpx.Client\n"
    assert find_outbound_sinks(code) == []


def test_inert_from_import_is_not_flagged() -> None:
    code = (
        "from httpx import HTTPError\n"
        "def f():\n"
        "    try:\n"
        "        pass\n"
        "    except HTTPError:\n"
        "        pass\n"
    )
    assert find_outbound_sinks(code) == []


def test_assigned_importlib_call_resolves_module_sinks() -> None:
    code = (
        "import importlib\n"
        "def f(u):\n"
        "    mod = importlib.import_module('httpx')\n"
        "    mod.post(u)\n"
    )
    calls = [(func, call) for func, _line, call in find_outbound_sinks(code)]
    assert ("f", "httpx.post") in calls


def test_call_multiset_mismatch_with_equal_call_count() -> None:
    dummy_ledger = {
        ("mod.py", "f"): DeclaredSink(calls=("httpx.get", "httpx.post"), reason="ok")
    }
    mock_hits = [
        ("mod.py", "f", 10, "httpx.post"),
        ("mod.py", "f", 15, "httpx.post"),
    ]
    unlisted = unlisted_sinks(mock_hits, dummy_ledger)
    assert unlisted == [
        "mod.py:10 in f() calls mismatch: "
        "expected ['httpx.get', 'httpx.post'], found ['httpx.post', 'httpx.post']"
    ]


# ---------------------------------------------------------------------------
# Optional pins: each class in urllib.request that opens a connection (or
# configures the pipeline that does) is reported under its own name, so a
# later edit of INERT_NAMES["urllib.request"] cannot take one back out quietly.
# ---------------------------------------------------------------------------
_CONNECTION_HANDLERS = {
    "HTTPHandler": "http_open",
    "HTTPSHandler": "https_open",
    "FTPHandler": "ftp_open",
    "CacheFTPHandler": "ftp_open",
}


@pytest.mark.parametrize("name", sorted(_CONNECTION_HANDLERS))
def test_a_handler_that_opens_a_connection_is_reported_under_its_own_name(name: str) -> None:
    method = _CONNECTION_HANDLERS[name]
    code = f"import urllib.request\ndef f(req):\n    return urllib.request.{name}().{method}(req)\n"
    calls = [call for _scope, _line, call in find_outbound_sinks(code)]
    assert f"urllib.request.{name}" in calls, (name, calls)


def test_a_proxy_handler_is_reported_under_its_own_name() -> None:
    code = "import urllib.request\ndef f(p):\n    return urllib.request.ProxyHandler(p)\n"
    calls = [call for _scope, _line, call in find_outbound_sinks(code)]
    assert "urllib.request.ProxyHandler" in calls, calls


def test_class_decorators_and_keywords_are_read_in_the_enclosing_scope() -> None:
    decorated = find_outbound_sinks("import httpx\n@httpx.post\nclass C:\n    pass\n")
    assert decorated == [("<module>", 2, "httpx.post")]
    keyword = find_outbound_sinks("import httpx\nclass C(metaclass=httpx.Client):\n    pass\n")
    assert keyword == [("<module>", 2, "httpx.Client")]

