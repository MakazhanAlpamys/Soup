"""``POST /v1/adapters/load`` adds a LoRA adapter to a running ``soup serve``.

A continual-learning loop trains a fresh adapter while the server runs and needs the
server to take it in before it can canary it. The route loads files into the serving
process, so it is off unless ``--adapter-dir`` names the one directory adapters may come
from, it takes the bearer token on every bind (loopback too), and the check order below
is part of the contract: an unauthenticated caller learns nothing, not even whether the
feature is enabled.

Everything here runs on fake model objects; ``serve._load_runtime_adapter`` (the one
function that calls PEFT) is replaced, except in the last section, which runs the real
helper on a two-layer torch module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import soup_cli.commands.serve as serve_cmd  # noqa: E402
from soup_cli.cli import app as cli_app  # noqa: E402
from soup_cli.utils.loop_state import LoopState, write_state  # noqa: E402

TOKEN = "tool_token_" + "a1B2c3D4" * 4  # 43 chars of the alphabet --tool-auth-token accepts
AUTH = {"Authorization": f"Bearer {TOKEN}"}
LAN_HOST = "192.168.1.5"
LOAD = "/v1/adapters/load"
runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output, safe to substring-match."""
    return " ".join(_ANSI.sub("", text or "").split())


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stands in for the served model; ``current`` is the adapter a request ran under."""

    def __init__(self, label: str = "base") -> None:
        self.label = label
        self.current = None

    def set_adapter(self, name: str) -> None:
        self.current = name

    @contextmanager
    def disable_adapter(self):
        previous = self.current
        self.current = None
        try:
            yield
        finally:
            self.current = previous


class _FakeLoader:
    """Replaces ``serve._load_runtime_adapter``: records calls, returns a new model object."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.returned = _FakeModel("peft")
        self.error: Exception | None = None
        self.entered = threading.Event()
        self.hold: threading.Event | None = None  # when set, a call waits for it

    def __call__(self, model_obj, path, name):
        self.calls.append((model_obj, path, name))
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(10), "the test never released the loader"
        if self.error is not None:
            raise self.error
        return self.returned


@pytest.fixture
def loader(monkeypatch) -> _FakeLoader:
    fake = _FakeLoader()
    monkeypatch.setattr(serve_cmd, "_load_runtime_adapter", fake)
    return fake


@pytest.fixture
def seen(monkeypatch) -> list:
    """What each generation ran on: ``(model object, adapter selected at that moment)``."""
    runs: list = []

    def generate(model, *args, **kwargs):
        runs.append((model, getattr(model, "current", None)))
        return "ok", 1, 1, "stop"

    monkeypatch.setattr(serve_cmd, "_generate_response", generate)
    return runs


@pytest.fixture
def root(tmp_path):
    directory = tmp_path / "adapters"
    directory.mkdir()
    return directory


def _adapter(parent, name: str):
    """A directory ``soup serve`` would accept as a LoRA adapter (config only; no PEFT here)."""
    directory = parent / name
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    return directory


class _Served:
    """A running app plus the objects it was built from, so tests can look inside."""

    def __init__(self, client, model, adapter_map, peft_names):
        self.client = client
        self.model = model
        self.adapter_map = adapter_map
        self.peft_names = peft_names

    def load(self, name, path, headers=AUTH):
        return self.client.post(LOAD, json={"name": name, "path": str(path)}, headers=headers)

    def chat(self, **extra):
        body = {"messages": [{"role": "user", "content": "hi"}], **extra}
        return self.client.post("/v1/chat/completions", json=body)

    def listed(self):
        resp = self.client.get("/v1/adapters", headers=AUTH)
        assert resp.status_code == 200, resp.text
        return [entry["name"] for entry in resp.json()["adapters"]]


def _serve(
    root,
    *,
    host="127.0.0.1",
    token=TOKEN,
    adapter_dir="root",
    startup=None,
    **create_app_kwargs,
) -> _Served:
    model = _FakeModel()
    adapter_map = {name: f"./{name}" for name in (startup or [])}
    peft_names = set(adapter_map)
    kwargs = dict(
        model_obj=model,
        tokenizer=object(),
        device="cpu",
        model_name="test",
        max_tokens_default=8,
        adapter_map=adapter_map or None,
        peft_adapter_names=peft_names,
        auth_token=token,
        host=host,
    )
    if adapter_dir == "root":
        kwargs["adapter_dir"] = str(root)
    elif adapter_dir is not None:
        kwargs["adapter_dir"] = adapter_dir
    kwargs.update(create_app_kwargs)
    application = serve_cmd._create_app(**kwargs)
    # The Host header names the bind, so the Host/Origin guard lets the request through.
    client = TestClient(application, base_url=f"http://{host}")
    return _Served(client, model, adapter_map, peft_names)


@pytest.fixture
def served(root, loader):
    return _serve(root)


# ---------------------------------------------------------------------------
# a. the token comes first, on every bind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", LAN_HOST], ids=["loopback", "lan"])
def test_without_a_configured_token_the_route_is_401_on_every_bind(root, loader, host):
    served = _serve(root, host=host, token=None)

    resp = served.load("v2", _adapter(root, "v2"), headers={})

    assert resp.status_code == 401, resp.text
    assert "--tool-auth-token" in resp.json()["detail"]
    assert "SOUP_TOOL_AUTH_TOKEN" in resp.json()["detail"]
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    assert loader.calls == []


def test_a_bearer_token_is_ignored_when_the_server_has_none(root, loader):
    # The loopback default of the other adapter routes ("no token configured: open") must
    # not leak in: with no configured token nothing a client sends opens this route.
    served = _serve(root, token=None)

    resp = served.load("v2", _adapter(root, "v2"), headers=AUTH)

    assert resp.status_code == 401, resp.text
    assert "--tool-auth-token" in resp.json()["detail"]
    assert loader.calls == []


@pytest.mark.parametrize("host", ["127.0.0.1", LAN_HOST], ids=["loopback", "lan"])
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": TOKEN},  # the scheme is missing
        {"Authorization": f"bearer {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN} "},
    ],
    ids=["missing", "wrong", "no-scheme", "lowercase-scheme", "trailing-space"],
)
def test_a_missing_or_wrong_token_is_401(root, loader, host, headers):
    served = _serve(root, host=host)

    resp = served.load("v2", _adapter(root, "v2"), headers=headers)

    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing bearer token"
    assert resp.headers["WWW-Authenticate"] == "Bearer"
    assert loader.calls == []


def test_the_bearer_comparison_uses_compare_digest(root, loader, monkeypatch):
    import secrets

    calls, real = [], secrets.compare_digest
    monkeypatch.setattr(secrets, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    served = _serve(root)

    assert served.load("v2", _adapter(root, "v2")).status_code == 200
    assert calls


def test_auth_runs_before_the_enabled_check(root, loader):
    # No --adapter-dir: a caller without the token must not learn that from a 404.
    served = _serve(root, adapter_dir=None)

    assert served.load("v2", root, headers={}).status_code == 401
    assert served.load("v2", root, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert served.load("v2", root).status_code == 404


def test_a_server_with_no_token_and_no_adapter_dir_answers_the_token_message(root, loader):
    served = _serve(root, adapter_dir=None, token=None)

    resp = served.load("v2", root, headers={})

    assert resp.status_code == 401
    assert "--tool-auth-token" in resp.json()["detail"]


def test_auth_runs_before_the_body_is_read(served):
    garbage = {"Content-Type": "application/json"}

    resp = served.client.post(LOAD, content=b"{not json", headers=garbage)

    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Invalid or missing bearer token"


def test_the_host_guard_runs_before_the_token_check(root, loader):
    served = _serve(root, host=LAN_HOST)

    # No token: a token check that ran first would answer 401.
    resp = served.client.post(LOAD, json={}, headers={"Host": "evil.example"})

    assert resp.status_code == 421
    assert loader.calls == []


# ---------------------------------------------------------------------------
# b. the feature is off without --adapter-dir
# ---------------------------------------------------------------------------


def test_the_route_is_404_without_an_adapter_dir(root, loader):
    served = _serve(root, adapter_dir=None)

    resp = served.load("v2", _adapter(root, "v2"))

    assert resp.status_code == 404, resp.text
    assert resp.json()["detail"] == (
        "runtime adapter loading is not enabled; start soup serve with --adapter-dir"
    )
    assert loader.calls == []


def test_the_enabled_check_runs_before_body_validation(root, loader):
    served = _serve(root, adapter_dir=None)

    resp = served.client.post(LOAD, content=b"{not json", headers=AUTH)

    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# c. the body
# ---------------------------------------------------------------------------

_VALID_NAME = "v2"


def _invalid_bodies():
    json_type = {"Content-Type": "application/json"}
    cases = {
        "not-json": {"content": b"{not json", "headers": json_type},
        "empty-body": {"content": b"", "headers": json_type},
        "invalid-utf8": {"content": b'{"name": "\xff"}', "headers": json_type},
        "list": {"json": []},
        "string": {"json": "v2"},
        "null": {"json": None},
        "number": {"json": 7},
        "empty-object": {"json": {}},
        "missing-name": {"json": {"path": "x"}},
        "missing-path": {"json": {"name": "v2"}},
        "name-int": {"json": {"name": 2, "path": "x"}},
        "name-null": {"json": {"name": None, "path": "x"}},
        "name-list": {"json": {"name": ["v2"], "path": "x"}},
        "name-bool": {"json": {"name": True, "path": "x"}},
        "path-int": {"json": {"name": "v2", "path": 2}},
        "path-null": {"json": {"name": "v2", "path": None}},
        "path-list": {"json": {"name": "v2", "path": ["x"]}},
        "name-empty": {"json": {"name": "", "path": "x"}},
        "name-leading-hyphen": {"json": {"name": "-v2", "path": "x"}},
        "name-underscore": {"json": {"name": "v_2", "path": "x"}},
        "name-space": {"json": {"name": "v 2", "path": "x"}},
        "name-slash": {"json": {"name": "a/b", "path": "x"}},
        "name-dotdot": {"json": {"name": "..", "path": "x"}},
        "name-trailing-newline": {"json": {"name": "v2\n", "path": "x"}},
        "name-non-ascii": {"json": {"name": "vé", "path": "x"}},
        "name-65-chars": {"json": {"name": "a" * 65, "path": "x"}},
        "path-empty": {"json": {"name": "v2", "path": ""}},
        "path-nul": {"json": {"name": "v2", "path": "a\x00b"}},
        "path-4097-chars": {"json": {"name": "v2", "path": "a" * 4097}},
        "deeply-nested": {"content": b"[" * 60_000, "headers": json_type},
        "body-over-64k": {"json": {"name": "v2", "path": "x", "padding": "p" * 70_000}},
    }
    return [pytest.param(kwargs, id=case_id) for case_id, kwargs in cases.items()]


@pytest.mark.parametrize("post", _invalid_bodies())
def test_an_invalid_body_is_400_invalid_request(served, loader, post):
    post = dict(post)
    headers = {**AUTH, **post.pop("headers", {})}

    resp = served.client.post(LOAD, headers=headers, **post)

    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert resp.json()["detail"] == "Invalid request"
    assert loader.calls == []


def test_a_64_character_name_is_the_longest_accepted(root, served, loader):
    resp = served.load("a" * 64, _adapter(root, "v2"))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"name": "a" * 64, "status": "loaded"}


def test_a_4096_character_path_passes_validation_and_stops_at_containment(served, loader):
    resp = served.load("v2", "a" * 4096)

    assert resp.status_code == 403, resp.text
    assert loader.calls == []


def test_names_that_the_activate_route_accepts_are_accepted(root, served, loader):
    for index, name in enumerate(["A", "9", "chat-v2", "x--y", "Zz9-"]):
        assert served.load(name, _adapter(root, f"d{index}")).status_code == 200, name


def _body_of_exactly(size: int, path) -> bytes:
    """A valid JSON body of ``size`` bytes (the padding key carries the difference)."""
    base = json.dumps({"name": "v2", "path": str(path), "pad": ""}).encode()
    body = json.dumps({"name": "v2", "path": str(path), "pad": "p" * (size - len(base))}).encode()
    assert len(body) == size
    return body


def test_a_body_of_exactly_64k_is_read_and_one_byte_more_is_not(root, served, loader):
    path = _adapter(root, "v2")
    headers = {**AUTH, "Content-Type": "application/json"}

    at_the_limit = served.client.post(LOAD, content=_body_of_exactly(65536, path), headers=headers)
    over = served.client.post(LOAD, content=_body_of_exactly(65537, path), headers=headers)

    assert at_the_limit.status_code == 200, at_the_limit.text
    assert over.status_code == 400, over.text
    assert over.json()["detail"] == "Invalid request"
    assert len(loader.calls) == 1


def test_any_content_type_is_accepted_so_plain_curl_works(root, served, loader):
    # `curl -d '{...}'` labels the body application/x-www-form-urlencoded.
    body = json.dumps({"name": "v2", "path": str(_adapter(root, "v2"))})

    resp = served.client.post(
        LOAD,
        content=body,
        headers={**AUTH, "Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 200, resp.text


def test_extra_keys_are_ignored(root, served, loader):
    body = {"name": "v2", "path": str(_adapter(root, "v2")), "activate": True, "x": [1]}

    resp = served.client.post(LOAD, json=body, headers=AUTH)

    assert resp.status_code == 200, resp.text
    assert served.client.get("/health").json()["active_adapter"] is None


# ---------------------------------------------------------------------------
# d. containment
# ---------------------------------------------------------------------------


@pytest.fixture
def outside(tmp_path):
    """A valid adapter directory next to, not inside, the adapter dir."""
    return _adapter(tmp_path, "outside")


def test_an_absolute_path_outside_the_adapter_dir_is_403(served, loader, outside):
    resp = served.load("v2", outside)

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "adapter path is outside --adapter-dir"
    assert loader.calls == []


def test_dot_dot_traversal_out_of_the_adapter_dir_is_403(root, served, loader, outside):
    _adapter(root, "v2")

    resp = served.load("v2", root / "v2" / ".." / ".." / "outside")

    assert resp.status_code == 403, resp.text
    assert loader.calls == []


def test_dot_dot_traversal_that_stays_inside_is_accepted(root, served, loader):
    _adapter(root, "v2")

    resp = served.load("v2", root / "other" / ".." / "v2")

    assert resp.status_code == 200, resp.text
    assert loader.calls[0][1] == os.path.realpath(root / "v2")


def test_a_sibling_sharing_the_directory_name_prefix_is_403(tmp_path, root, served, loader):
    sibling = _adapter(tmp_path, "adapters-evil")

    resp = served.load("v2", sibling)

    assert resp.status_code == 403, resp.text
    assert loader.calls == []


def test_the_adapter_dir_itself_may_be_the_adapter(root, served, loader):
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")

    resp = served.load("v2", root)

    assert resp.status_code == 200, resp.text
    assert loader.calls[0][1] == os.path.realpath(root)


@pytest.mark.requires_symlink
def test_a_symlink_inside_pointing_outside_is_403(root, served, loader, outside):
    link = root / "innocent"
    os.symlink(outside, link, target_is_directory=True)

    resp = served.load("v2", link)

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "adapter path is outside --adapter-dir"
    assert loader.calls == []


@pytest.mark.requires_symlink
def test_a_symlink_to_a_directory_inside_is_loaded_from_its_target(root, served, loader):
    real = _adapter(root, "real")
    os.symlink(real, root / "alias", target_is_directory=True)

    resp = served.load("v2", root / "alias")

    assert resp.status_code == 200, resp.text
    assert loader.calls[0][1] == os.path.realpath(real)


@pytest.mark.requires_symlink
def test_a_symlinked_adapter_dir_is_resolved_once_at_startup(tmp_path, loader, outside):
    # The allowed root is the REAL directory behind --adapter-dir, fixed when the app is built:
    # re-pointing the link afterwards must not move the root to somewhere else.
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    adapter = _adapter(real_root, "v2")
    link = tmp_path / "link-root"
    os.symlink(real_root, link, target_is_directory=True)
    served = _serve(tmp_path, adapter_dir=str(link))

    assert served.load("v2", link / "v2").status_code == 200  # through the link: inside
    link.unlink()
    os.symlink(outside.parent, link, target_is_directory=True)

    assert served.load("v3", adapter).status_code == 200  # the real path still counts
    assert served.load("v4", link / "outside").status_code == 403  # the link now leaves the root


def test_a_relative_path_is_resolved_against_the_servers_cwd(tmp_path, root, loader, monkeypatch):
    _adapter(root, "v2")
    _adapter(tmp_path, "outside")
    served = _serve(root)
    monkeypatch.chdir(tmp_path)

    inside = served.client.post(LOAD, json={"name": "v2", "path": "adapters/v2"}, headers=AUTH)
    leaving = served.client.post(LOAD, json={"name": "v3", "path": "outside"}, headers=AUTH)

    assert inside.status_code == 200, inside.text
    assert loader.calls[0][1] == os.path.realpath(root / "v2")
    assert os.path.isabs(loader.calls[0][1])
    assert leaving.status_code == 403, leaving.text


def test_containment_is_checked_before_the_adapter_config(served, loader, tmp_path):
    # A path outside the adapter dir gets the same 403 whether or not it exists or holds an
    # adapter: nothing outside the directory can be probed through this route.
    empty = tmp_path / "empty"
    empty.mkdir()

    for path in (empty, tmp_path / "does-not-exist"):
        resp = served.load("v2", path)
        assert resp.status_code == 403, (path, resp.text)
    assert loader.calls == []


@pytest.mark.parametrize("error", [OSError("bad path"), ValueError("embedded null byte")])
def test_a_path_that_cannot_be_resolved_is_outside(monkeypatch, root, error):
    def refuse(_path):
        raise error

    monkeypatch.setattr(os.path, "realpath", refuse)

    assert serve_cmd._resolve_inside("anything", str(root)) is None


def test_a_refusal_does_not_echo_the_path(served, loader, outside):
    resp = served.load("v2", outside)

    assert str(outside) not in resp.text
    assert os.path.realpath(outside) not in resp.text


# ---------------------------------------------------------------------------
# e. the directory must be a LoRA adapter
# ---------------------------------------------------------------------------


def test_a_directory_without_adapter_config_is_400(root, served, loader):
    (root / "plain").mkdir()

    resp = served.load("v2", root / "plain")

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "not a LoRA adapter directory"
    assert loader.calls == []


def test_a_missing_path_inside_the_adapter_dir_is_400(root, served, loader):
    resp = served.load("v2", root / "nope")

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "not a LoRA adapter directory"


def test_a_file_is_not_an_adapter_directory(root, served, loader):
    (root / "afile").write_text("{}", encoding="utf-8")

    resp = served.load("v2", root / "afile")

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "not a LoRA adapter directory"
    assert loader.calls == []


def test_an_adapter_config_that_is_a_directory_does_not_count(root, served, loader):
    (root / "weird").mkdir()
    (root / "weird" / "adapter_config.json").mkdir()

    resp = served.load("v2", root / "weird")

    assert resp.status_code == 400, resp.text
    assert loader.calls == []


# ---------------------------------------------------------------------------
# f. names and the cap
# ---------------------------------------------------------------------------


def test_a_name_loaded_at_startup_is_409(root, loader):
    served = _serve(root, startup=["chat"])

    resp = served.load("chat", _adapter(root, "again"))

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "adapter name already loaded"
    assert loader.calls == []


def test_a_name_loaded_at_runtime_is_409(root, served, loader):
    first = _adapter(root, "first")
    assert served.load("v2", first).status_code == 200

    resp = served.load("v2", _adapter(root, "second"))

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "adapter name already loaded"
    assert len(loader.calls) == 1


def test_a_name_the_served_model_already_carries_is_409(root, loader):
    # `soup serve --model ./adapter` serves a PeftModel whose adapter is called "default":
    # loading under that name would overwrite the served weights in place.
    served = _serve(root)
    served.model.peft_config = {"default": object()}

    resp = served.load("default", _adapter(root, "v2"))

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "adapter name already loaded"
    assert loader.calls == []


@pytest.mark.parametrize("forget", ["adapter_map", "peft_names"])
def test_a_name_known_to_either_registry_is_409(root, loader, forget):
    # The map and the PEFT name set are filled together everywhere today; the route refuses
    # a name either one holds, so a future caller that fills only one is still covered.
    served = _serve(root, startup=["chat"])
    getattr(served, forget).clear()

    resp = served.load("chat", _adapter(root, "again"))

    assert resp.status_code == 409, resp.text
    assert loader.calls == []


def test_the_duplicate_check_leaves_the_map_alone(root, loader):
    served = _serve(root, startup=["chat"])
    before = dict(served.adapter_map)

    served.load("chat", _adapter(root, "again"))

    assert served.adapter_map == before


def test_sixteen_runtime_adapters_load_and_the_seventeenth_is_409(root, loader):
    served = _serve(root, startup=["chat"])  # a startup adapter does not use up a slot

    for index in range(16):
        resp = served.load(f"v{index}", _adapter(root, f"d{index}"))
        assert resp.status_code == 200, (index, resp.text)

    resp = served.load("v16", _adapter(root, "d16"))

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "runtime adapter limit reached (16)"
    assert len(loader.calls) == 16
    assert "v16" not in served.listed()
    assert len(served.listed()) == 17  # chat + 16


def test_failed_loads_do_not_use_up_the_cap(root, loader):
    served = _serve(root)
    loader.error = RuntimeError("bad weights")
    for index in range(20):
        assert served.load(f"bad{index}", _adapter(root, f"b{index}")).status_code == 400
    loader.error = None

    for index in range(16):
        resp = served.load(f"v{index}", _adapter(root, f"d{index}"))
        assert resp.status_code == 200, (index, resp.text)


def test_the_cap_is_per_process_not_per_name(root, loader):
    first = _serve(root)
    second = _serve(root)

    for index in range(16):
        assert first.load(f"v{index}", _adapter(root, f"d{index}")).status_code == 200

    # A different app (a different process in real life) has its own count.
    assert second.load("v0", root / "d0").status_code == 200


# ---------------------------------------------------------------------------
# g. MoLE / vector bank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "feature"),
    [({"mole_runtime": MagicMock()}, "--mole"), ({"loaded_bank": MagicMock()}, "--bank")],
    ids=["mole", "bank"],
)
def test_mole_and_vector_bank_refuse_runtime_loading(root, loader, kwargs, feature):
    served = _serve(root, **kwargs)

    resp = served.load("v2", _adapter(root, "v2"))

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == f"runtime adapter loading is not supported with {feature}"
    assert loader.calls == []
    assert served.listed() == []


# ---------------------------------------------------------------------------
# h. the load itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("boom"),
        ValueError("boom"),
        OSError("boom"),
        KeyError("boom"),
        ImportError("boom"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_a_loader_exception_is_400_and_leaves_no_trace(root, loader, seen, caplog, error):
    served = _serve(root, startup=["chat"])
    served.client.post("/v1/adapters/activate/chat", headers=AUTH)
    path = _adapter(root, "v2")
    loader.error = error

    with caplog.at_level(logging.ERROR, logger="soup_cli.commands.serve"):
        resp = served.load("v2", path)

    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "adapter could not be loaded"}
    assert "boom" not in resp.text and "Traceback" not in resp.text
    assert str(path) not in resp.text
    logged = [rec for rec in caplog.records if rec.exc_info and rec.levelno >= logging.ERROR]
    assert logged, "the failure must reach the log with its traceback"
    assert "v2" in logged[0].getMessage()
    # nothing changed: the map, the active adapter, the model every request runs on
    assert served.listed() == ["chat"]
    assert served.adapter_map == {"chat": "./chat"}
    assert served.peft_names == {"chat"}
    assert served.client.get("/health").json()["active_adapter"] == "chat"
    assert served.client.post("/v1/adapters/activate/v2", headers=AUTH).status_code == 404
    assert served.chat(adapter="v2").status_code == 404
    assert served.chat().status_code == 200
    assert seen[-1][0] is served.model


def test_a_failed_load_can_be_retried_under_the_same_name(root, loader, seen):
    served = _serve(root)
    path = _adapter(root, "v2")
    loader.error = RuntimeError("transient")
    assert served.load("v2", path).status_code == 400

    loader.error = None
    resp = served.load("v2", path)

    assert resp.status_code == 200, resp.text
    # the failed attempt did not rebind the model: the retry got the original object
    assert [call[0] for call in loader.calls] == [served.model, served.model]
    assert served.chat().status_code == 200
    assert seen[-1][0] is loader.returned


def test_the_load_runs_under_the_generation_lock(root, loader, monkeypatch):
    served = _serve(root, startup=["chat"])  # adapters present: generation takes the lock
    in_generation, release = threading.Event(), threading.Event()

    def slow_generate(model, *args, **kwargs):
        in_generation.set()
        assert release.wait(10), "the load never let the generation finish"
        return "ok", 1, 1, "stop"

    monkeypatch.setattr(serve_cmd, "_generate_response", slow_generate)
    path = _adapter(root, "v2")
    results: dict = {}

    def run_chat():
        results["chat"] = served.chat(adapter="chat")

    def run_load():
        other = TestClient(served.client.app, base_url="http://127.0.0.1")
        results["load"] = other.post(
            LOAD, json={"name": "v2", "path": str(path)}, headers=AUTH
        )

    chat_thread = threading.Thread(target=run_chat)
    chat_thread.start()
    assert in_generation.wait(10)
    load_thread = threading.Thread(target=run_load)
    load_thread.start()
    try:
        # The generation holds the lock, so the load has to wait for it.
        assert not loader.entered.wait(0.5), "the adapter loaded while a generation was running"
    finally:
        release.set()
        chat_thread.join(10)
        load_thread.join(10)

    assert loader.entered.is_set()
    assert results["chat"].status_code == 200
    assert results["load"].status_code == 200, results["load"].text


def test_two_loads_of_one_name_at_once_load_it_once(root, served, loader):
    loader.hold = threading.Event()
    path = _adapter(root, "v2")
    codes: list[int] = []

    def run_load():
        other = TestClient(served.client.app, base_url="http://127.0.0.1")
        body = {"name": "v2", "path": str(path)}
        codes.append(other.post(LOAD, json=body, headers=AUTH).status_code)

    first = threading.Thread(target=run_load)
    first.start()
    assert loader.entered.wait(10)
    second = threading.Thread(target=run_load)
    second.start()
    second.join(0.5)  # the second request has had time to pass every check it can
    loader.hold.set()
    first.join(10)
    second.join(10)

    assert sorted(codes) == [200, 409], codes
    assert len(loader.calls) == 1
    assert served.listed() == ["v2"]


def test_a_load_waiting_for_the_lock_does_not_block_the_event_loop(root, loader, monkeypatch):
    served = _serve(root, startup=["chat"])
    in_generation, release = threading.Event(), threading.Event()

    def slow_generate(model, *args, **kwargs):
        in_generation.set()
        release.wait(10)
        return "ok", 1, 1, "stop"

    monkeypatch.setattr(serve_cmd, "_generate_response", slow_generate)
    path = _adapter(root, "v2")
    answered: dict = {}

    # One client as a context manager runs every request on one shared event loop, so a
    # handler that blocks the loop stalls all of them.
    with TestClient(served.client.app, base_url="http://127.0.0.1") as shared:

        def run_chat():
            shared.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "adapter": "chat"},
            )

        def run_load():
            answered["load"] = shared.post(
                LOAD, json={"name": "v2", "path": str(path)}, headers=AUTH
            )

        def run_health():
            answered["health"] = shared.get("/health")

        chat_thread = threading.Thread(target=run_chat)
        chat_thread.start()
        assert in_generation.wait(10)
        load_thread = threading.Thread(target=run_load)
        load_thread.start()
        health_thread = threading.Thread(target=run_health)
        try:
            assert not loader.entered.wait(0.5), "the load did not wait for the generation"
            health_thread.start()
            health_thread.join(3)
            assert "health" in answered, "the waiting load froze the event loop"
        finally:
            release.set()
            chat_thread.join(10)
            load_thread.join(10)
            health_thread.join(10)

    assert answered["health"].status_code == 200
    assert answered["load"].status_code == 200, answered["load"].text


def test_the_loader_gets_the_resolved_path_and_the_current_model(root, served, loader):
    path = _adapter(root, "v2")

    resp = served.load("v2", path)

    assert resp.status_code == 200
    assert loader.calls == [(served.model, os.path.realpath(path), "v2")]


# ---------------------------------------------------------------------------
# i. what a successful load changes
# ---------------------------------------------------------------------------


def test_a_load_reports_the_name_and_status(root, served, loader):
    resp = served.load("v2", _adapter(root, "v2"))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"name": "v2", "status": "loaded"}


def test_a_loaded_adapter_is_listed_but_not_active(root, loader):
    served = _serve(root, startup=["chat"])
    served.client.post("/v1/adapters/activate/chat", headers=AUTH)

    assert served.load("v2", _adapter(root, "v2")).status_code == 200

    listing = served.client.get("/v1/adapters", headers=AUTH).json()
    assert listing == {
        "adapters": [{"name": "chat", "active": True}, {"name": "v2", "active": False}],
        "active": "chat",
    }
    assert served.client.get("/health").json()["active_adapter"] == "chat"


def test_loading_the_first_adapter_does_not_activate_it(root, served, loader, seen):
    served.load("v2", _adapter(root, "v2"))

    assert served.client.get("/health").json()["active_adapter"] is None
    served.chat()
    assert seen[-1][1] is None  # no adapter selected: the base model


def test_the_maps_are_updated_in_place(root, loader):
    served = _serve(root, startup=["chat"])
    path = _adapter(root, "v2")

    # Spelled with a `..`, so the stored value can only be the resolved path, not the request's.
    served.load("v2", root / "elsewhere" / ".." / "v2")

    assert served.adapter_map["v2"] == os.path.realpath(path)
    assert ".." not in served.adapter_map["v2"]
    assert served.peft_names == {"chat", "v2"}


def test_a_loaded_adapter_can_be_activated(root, served, loader, seen):
    served.load("v2", _adapter(root, "v2"))

    resp = served.client.post("/v1/adapters/activate/v2", headers=AUTH)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"active": "v2", "status": "ok"}
    assert served.chat().status_code == 200
    model, adapter = seen[-1]
    assert model is loader.returned
    assert adapter == "v2"


def test_a_loaded_adapter_can_be_picked_per_request(root, served, loader, seen):
    served.load("v2", _adapter(root, "v2"))

    assert served.chat(adapter="v2").status_code == 200
    assert seen[-1] == (loader.returned, "v2")
    assert served.chat().status_code == 200
    assert seen[-1] == (loader.returned, None)  # no adapter: the base, via disable_adapter


def test_an_unknown_adapter_is_still_unknown_after_a_load(root, served, loader, seen):
    served.load("v2", _adapter(root, "v2"))

    assert served.chat(adapter="v3").status_code == 404
    assert served.client.post("/v1/adapters/activate/v3", headers=AUTH).status_code == 404


def test_the_chat_handler_runs_on_the_model_the_loader_returned(root, served, loader, seen):
    original = served.model
    served.chat()
    assert seen[-1][0] is original

    served.load("v2", _adapter(root, "v2"))
    served.chat()

    assert seen[-1][0] is loader.returned
    assert seen[-1][0] is not original


def test_a_streaming_request_runs_on_the_model_the_loader_returned(
    root, served, loader, monkeypatch
):
    models = []

    def stream(model, *args, **kwargs):
        models.append(model)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(serve_cmd, "_stream_response", stream)
    served.load("v2", _adapter(root, "v2"))

    resp = served.chat(stream=True)

    assert resp.status_code == 200, resp.text
    assert models == [loader.returned]


def test_the_streaming_handler_sees_the_new_adapter_names(root, served, loader, monkeypatch):
    names = []

    def stream(model, *args, adapter_names=None, **kwargs):
        names.append(set(adapter_names))
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(serve_cmd, "_stream_response", stream)
    served.load("v2", _adapter(root, "v2"))

    assert served.chat(stream=True, adapter="v2").status_code == 200
    assert names == [{"v2"}]


def test_a_second_load_gets_the_model_the_first_one_returned(root, served, loader):
    served.load("v1", _adapter(root, "d1"))
    original = served.model

    served.load("v2", _adapter(root, "d2"))

    assert loader.calls[0][0] is original
    assert loader.calls[1][0] is loader.returned


def test_loading_works_on_a_lan_bind_with_the_token(root, loader):
    served = _serve(root, host=LAN_HOST)

    assert served.load("v2", _adapter(root, "v2")).status_code == 200


def test_the_canary_router_sees_a_runtime_loaded_adapter(tmp_path, root, loader, seen, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = tmp_path / ".soup" / "loop.yaml"
    write_state(
        LoopState(
            served_model="base",
            eval_suite="evals.yaml",
            baseline="prod",
            canary_active="v2",
            canary_traffic_pct=50.0,
            canary_rollout_id="rollout-1",
        ),
        str(state_path),
    )
    served = _serve(
        root,
        canary_state_path=str(state_path),
        canary_stats_path=str(tmp_path / ".soup" / "canary-stats.json"),
    )

    def run(count):
        for index in range(count):
            resp = served.chat(conversation_id=f"conversation-{index}")
            assert resp.status_code == 200, resp.text

    run(60)
    assert all(adapter is None for _, adapter in seen), "nothing is loaded yet"
    seen.clear()

    assert served.load("v2", _adapter(root, "v2")).status_code == 200
    run(60)

    routed = [adapter for _, adapter in seen]
    assert 10 <= routed.count("v2") <= 50, routed
    assert all(model is loader.returned for model, _ in seen)
    served.client.app.state.canary_outcomes.close()


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------


def _cli(*args: str):
    # The model path does not exist, so a command that gets past the flag checks stops there.
    return runner.invoke(cli_app, ["serve", "--model", "no-such-model-dir", *args])


def test_help_lists_adapter_dir(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")

    result = runner.invoke(cli_app, ["serve", "--help"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _plain(result.output.replace("│", " "))
    assert "--adapter-dir" in out
    assert (
        "Directory runtime adapters may be loaded from through POST /v1/adapters/load "
        "(needs --tool-auth-token). Not set: the route is off."
    ) in out


def test_a_missing_adapter_dir_is_refused_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    missing = tmp_path / "no-such-adapters"

    result = _cli("--adapter-dir", str(missing), "--tool-auth-token", TOKEN)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert "--adapter-dir" in out
    assert "is not an existing directory" in out
    assert "Model path not found" not in out


def test_an_adapter_dir_that_is_a_file_is_refused_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    a_file = tmp_path / "file.txt"
    a_file.write_text("x", encoding="utf-8")

    result = _cli("--adapter-dir", str(a_file), "--tool-auth-token", TOKEN)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "is not an existing directory" in _plain(result.output)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "mii", "VLLM"])
def test_adapter_dir_needs_the_transformers_backend(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("COLUMNS", "200")

    result = _cli("--adapter-dir", str(tmp_path), "--backend", backend, "--tool-auth-token", TOKEN)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--adapter-dir requires --backend transformers" in _plain(result.output)


def test_adapter_dir_cannot_be_combined_with_mole(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mole-out").mkdir()
    (tmp_path / "adapters").mkdir()

    result = _cli("--mole", "mole-out", "--adapter-dir", "adapters", "--tool-auth-token", TOKEN)

    assert result.exit_code == 2, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert "--mole cannot be combined with:" in out
    assert "--adapter-dir" in out


def test_an_existing_adapter_dir_passes_the_flag_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")

    result = _cli("--adapter-dir", str(tmp_path), "--tool-auth-token", TOKEN)

    # Past every --adapter-dir check; it is the missing model that stops the command.
    assert result.exit_code == 1, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert "Model path not found" in out
    assert "--adapter-dir" not in out


def _run_command(tmp_path, *extra: str):
    pytest.importorskip("uvicorn")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    recorded: dict = {}

    def record(**kwargs):
        recorded.update(kwargs)
        return MagicMock()

    with (
        patch.object(serve_cmd, "_load_model", return_value=(MagicMock(), MagicMock())),
        patch.object(serve_cmd, "_create_app", side_effect=record),
        patch("uvicorn.run"),
    ):
        result = runner.invoke(
            cli_app, ["serve", "--model", str(model_dir), "--device", "cpu", *extra]
        )
    return result, recorded


def test_the_command_hands_the_resolved_adapter_dir_to_the_app(tmp_path, monkeypatch):
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    monkeypatch.chdir(tmp_path)

    # Given relative: the app gets the absolute, resolved directory, fixed at startup.
    result, recorded = _run_command(
        tmp_path, "--adapter-dir", "adapters", "--tool-auth-token", TOKEN
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert recorded["adapter_dir"] == os.path.realpath(adapters)
    assert os.path.isabs(recorded["adapter_dir"])
    assert "POST /v1/adapters/load" in _plain(result.output)


def test_without_the_flag_the_app_gets_no_adapter_dir(tmp_path):
    result, recorded = _run_command(tmp_path, "--tool-auth-token", TOKEN)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert recorded["adapter_dir"] is None
    assert "/v1/adapters/load" not in _plain(result.output)


def test_a_server_built_without_adapter_dir_has_the_route_off(root, loader):
    # `_create_app` called the way every existing caller calls it.
    application = serve_cmd._create_app(
        model_obj=_FakeModel(),
        tokenizer=object(),
        device="cpu",
        model_name="test",
        max_tokens_default=8,
        auth_token=TOKEN,
    )
    client = TestClient(application, base_url="http://127.0.0.1")

    resp = client.post(LOAD, json={"name": "v2", "path": str(root)}, headers=AUTH)

    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# the real PEFT helper, on a two-layer torch module
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, tensor):
            return self.fc(tensor)

    def save(path):
        config = peft.LoraConfig(r=2, target_modules=["fc"])
        peft.get_peft_model(Tiny(), config).save_pretrained(str(path))
        return path

    return Tiny, save, peft


def test_the_helper_wraps_a_plain_model_then_adds_to_the_peft_model(tmp_path, tiny):
    module, save, peft = tiny
    first, second = save(tmp_path / "a"), save(tmp_path / "b")
    base = module()

    wrapped = serve_cmd._load_runtime_adapter(base, str(first), "v1")

    assert isinstance(wrapped, peft.PeftModel)
    assert wrapped is not base
    assert list(wrapped.peft_config) == ["v1"]
    assert wrapped.training is False

    again = serve_cmd._load_runtime_adapter(wrapped, str(second), "v2")

    assert again is wrapped
    assert list(wrapped.peft_config) == ["v1", "v2"]
    assert wrapped.training is False


def test_the_helper_puts_the_model_in_eval_mode(tmp_path, tiny):
    # PEFT 0.20 already leaves a loaded adapter in eval mode, so a real PeftModel cannot show
    # the call; a PeftModel whose load_adapter leaves the training flag alone can.
    torch = pytest.importorskip("torch")
    _, _, peft = tiny

    class _Stub(peft.PeftModel):
        peft_config: dict = {}

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.loaded: list = []

        def load_adapter(self, path, adapter_name):
            self.loaded.append((path, adapter_name))

    stub = _Stub()
    assert stub.training is True

    result = serve_cmd._load_runtime_adapter(stub, str(tmp_path), "v2")

    assert result is stub
    assert stub.loaded == [(str(tmp_path), "v2")]
    assert stub.training is False


def test_the_helper_refuses_pickle_weights_before_peft_sees_them(tmp_path, tiny, monkeypatch):
    module, save, peft = tiny
    unsafe = save(tmp_path / "unsafe")
    (unsafe / "adapter_model.safetensors").unlink()
    (unsafe / "adapter_model.bin").write_bytes(b"\x80\x04 pretend this is a pickle")
    calls = []
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *a, **k: calls.append(a))

    with pytest.raises(ValueError, match="unsafe weight file"):
        serve_cmd._load_runtime_adapter(module(), str(unsafe), "v1")

    assert calls == []


def test_a_failed_add_leaves_no_half_registered_adapter(tmp_path, tiny):
    module, save, _ = tiny
    wrapped = serve_cmd._load_runtime_adapter(module(), str(save(tmp_path / "a")), "v1")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "adapter_config.json").write_text(
        (tmp_path / "a" / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )  # a config and no weights: PEFT registers the adapter, then fails to fill it

    with pytest.raises(Exception):  # noqa: B017 — which one PEFT raises is its business
        serve_cmd._load_runtime_adapter(wrapped, str(broken), "v2")

    assert list(wrapped.peft_config) == ["v1"]
    # and the name is free for a good adapter
    serve_cmd._load_runtime_adapter(wrapped, str(save(tmp_path / "good")), "v2")
    assert list(wrapped.peft_config) == ["v1", "v2"]


def test_a_failed_load_never_removes_an_adapter_that_was_already_there(tmp_path, tiny):
    module, save, _ = tiny
    wrapped = serve_cmd._load_runtime_adapter(module(), str(save(tmp_path / "a")), "v1")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017
        serve_cmd._load_runtime_adapter(wrapped, str(empty), "v1")

    assert list(wrapped.peft_config) == ["v1"]
