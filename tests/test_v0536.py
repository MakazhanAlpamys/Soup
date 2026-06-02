"""v0.53.6 — Plugin + Agent + Anthropic API.

Covers:
- #101: SoupPluginCallback + attach_plugin_callback wired into 13 trainers.
- #102: Anthropic /v1/messages route on transformers backend.
- #104: n-gram speculative decoding wiring (prompt_lookup_num_tokens).
- #103: server-side tool endpoints — deferred-live stubs returning 501.
- #105: utils/trainer_plugins.instantiate_trainer_plugins — stub.
- #106: utils/recipe_run.run_recipe — stub + `soup data recipe --execute`.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# ANSI escape stripper — CI terminals render Typer/Rich help text with style
# spans (`-` and `-execute` end up in separate `\x1b[...]m` runs), so naive
# substring checks against `result.output` fail. Mirrors the v0.53.5 CI fix
# pattern (test_v0535.py `--live` substring guard).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

# ----------------------------------------------------------------------
# #101 — SoupPluginCallback + attach_plugin_callback
# ----------------------------------------------------------------------


class _RecordingPlugin:
    def __init__(self) -> None:
        self.events: list[str] = []

    def pre_train(self, _ctx):  # noqa: D401
        self.events.append("pre_train")

    def post_step(self, _ctx):  # noqa: D401
        self.events.append("post_step")


class _RaisingPlugin:
    def pre_train(self, _ctx):  # noqa: D401
        raise RuntimeError("plugin boom")


@pytest.fixture
def clear_plugins_fixture():
    from soup_cli.plugins import clear_plugins

    clear_plugins()
    yield
    clear_plugins()


def test_build_plugin_callback_none_when_no_plugins(clear_plugins_fixture):
    """No registered plugins → build returns None (no-op short-circuit)."""
    from soup_cli.monitoring.plugin_callback import build_plugin_callback

    assert build_plugin_callback() is None


def test_build_plugin_callback_skips_disabled(clear_plugins_fixture):
    """Disabled plugin → build returns None (no enabled hooks)."""
    from soup_cli.monitoring.plugin_callback import build_plugin_callback
    from soup_cli.plugins import disable_plugin, register_plugin

    plugin = _RecordingPlugin()
    register_plugin(name="rec-a", version="0.1.0", plugin=plugin)
    disable_plugin("rec-a")
    assert build_plugin_callback() is None


def test_build_plugin_callback_returns_callback_when_enabled(
    clear_plugins_fixture,
):
    """Enabled plugin with at least one hook → real TrainerCallback."""
    transformers = pytest.importorskip("transformers")
    from soup_cli.monitoring.plugin_callback import build_plugin_callback
    from soup_cli.plugins import register_plugin

    plugin = _RecordingPlugin()
    register_plugin(name="rec-b", version="0.1.0", plugin=plugin)
    callback = build_plugin_callback()
    assert callback is not None
    assert isinstance(callback, transformers.TrainerCallback)


def test_plugin_callback_dispatches_to_implemented_hooks(
    clear_plugins_fixture,
):
    """Trainer event → only implemented hooks fire."""
    pytest.importorskip("transformers")
    from soup_cli.monitoring.plugin_callback import build_plugin_callback
    from soup_cli.plugins import register_plugin

    plugin = _RecordingPlugin()
    register_plugin(name="rec-c", version="0.1.0", plugin=plugin)
    callback = build_plugin_callback()
    assert callback is not None
    args = MagicMock()
    state = MagicMock()
    control = MagicMock()

    callback.on_train_begin(args, state, control)
    callback.on_step_end(args, state, control)
    callback.on_step_begin(args, state, control)  # plugin has no pre_step
    callback.on_train_end(args, state, control)  # plugin has no post_train
    assert plugin.events == ["pre_train", "post_step"]


def test_plugin_callback_swallows_hook_exceptions(
    clear_plugins_fixture, caplog
):
    """One misbehaving plugin must not crash training."""
    pytest.importorskip("transformers")
    from soup_cli.monitoring.plugin_callback import build_plugin_callback
    from soup_cli.plugins import register_plugin

    register_plugin(name="bad-plugin", version="0.1.0", plugin=_RaisingPlugin())
    callback = build_plugin_callback()
    assert callback is not None
    caplog.clear()  # review fix — defend against accumulated log records
    with caplog.at_level("WARNING"):
        callback.on_train_begin(MagicMock(), MagicMock(), MagicMock())
    # Did not raise; recorded a WARNING for this specific plugin.
    matching = [
        rec
        for rec in caplog.records
        if rec.levelname == "WARNING" and "bad-plugin" in rec.message
    ]
    assert matching, "expected WARNING for bad-plugin hook failure"


def test_attach_plugin_callback_no_plugins_returns_false(
    clear_plugins_fixture,
):
    """No plugins registered → helper short-circuits to False (no trainer touch)."""
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    trainer = MagicMock()
    attached = attach_plugin_callback(trainer)
    assert attached is False
    trainer.add_callback.assert_not_called()


def test_attach_plugin_callback_attaches_when_enabled(
    clear_plugins_fixture,
):
    """Enabled plugin → trainer.add_callback called once with the callback."""
    pytest.importorskip("transformers")
    from soup_cli.plugins import register_plugin
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    register_plugin(name="rec-attach", version="0.1.0", plugin=_RecordingPlugin())
    trainer = MagicMock()
    attached = attach_plugin_callback(trainer)
    assert attached is True
    trainer.add_callback.assert_called_once()


def test_attach_plugin_callback_swallows_add_callback_failure(
    clear_plugins_fixture,
):
    """trainer.add_callback raising → helper returns False, no crash."""
    pytest.importorskip("transformers")
    from soup_cli.plugins import register_plugin
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    register_plugin(name="rec-fail", version="0.1.0", plugin=_RecordingPlugin())
    trainer = MagicMock()
    trainer.add_callback.side_effect = RuntimeError("add_callback broken")
    attached = attach_plugin_callback(trainer)
    assert attached is False


def test_attach_plugin_callback_console_advisory(clear_plugins_fixture):
    """Optional `console` argument prints an advisory; never crashes."""
    pytest.importorskip("transformers")
    from soup_cli.plugins import register_plugin
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    register_plugin(name="rec-console", version="0.1.0", plugin=_RecordingPlugin())
    trainer = MagicMock()
    console = MagicMock()
    attached = attach_plugin_callback(trainer, console)
    assert attached is True
    console.print.assert_called_once()
    msg = console.print.call_args.args[0]
    assert "1 enabled" in msg


def test_attach_plugin_callback_console_print_failure_swallowed(
    clear_plugins_fixture,
):
    """`console.print` raising must not crash the helper."""
    pytest.importorskip("transformers")
    from soup_cli.plugins import register_plugin
    from soup_cli.utils.peft_wiring import attach_plugin_callback

    register_plugin(name="rec-console2", version="0.1.0", plugin=_RecordingPlugin())
    trainer = MagicMock()
    console = MagicMock()
    console.print.side_effect = RuntimeError("console broken")
    attached = attach_plugin_callback(trainer, console)
    assert attached is True  # callback still attached even though print failed


@pytest.mark.parametrize(
    "trainer_file",
    [
        "src/soup_cli/trainer/sft.py",
        "src/soup_cli/trainer/dpo.py",
        "src/soup_cli/trainer/grpo.py",
        "src/soup_cli/trainer/kto.py",
        "src/soup_cli/trainer/orpo.py",
        "src/soup_cli/trainer/simpo.py",
        "src/soup_cli/trainer/ipo.py",
        "src/soup_cli/trainer/bco.py",
        "src/soup_cli/trainer/ppo.py",
        "src/soup_cli/trainer/pretrain.py",
        "src/soup_cli/trainer/reward_model.py",
        "src/soup_cli/trainer/embedding.py",
        "src/soup_cli/trainer/distill.py",
    ],
)
def test_every_trainer_wires_attach_plugin_callback(trainer_file: str):
    """Source-level invariant: every transformer-backend trainer wires the helper."""
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / trainer_file).read_text(encoding="utf-8")
    assert "attach_plugin_callback" in src, (
        f"{trainer_file} is missing attach_plugin_callback wiring"
    )
    # Direct import from canonical peft_wiring module (no re-export shim).
    assert "attach_plugin_callback," in src or (
        "from soup_cli.utils.peft_wiring import" in src
        and "attach_plugin_callback" in src
    ), f"{trainer_file} must import attach_plugin_callback from peft_wiring"
    # Sanity: imported AND called on self.trainer.
    assert "attach_plugin_callback(self.trainer" in src, (
        f"{trainer_file} imports but never invokes attach_plugin_callback"
    )


# ----------------------------------------------------------------------
# #102 — Anthropic /v1/messages route
# ----------------------------------------------------------------------


def _fake_model_and_tokenizer():
    """Build minimal mocks so _create_app can construct without HF deps."""
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return model, tokenizer


def _build_app(**overrides):
    pytest.importorskip("fastapi")
    from soup_cli.commands.serve import _create_app

    model, tokenizer = _fake_model_and_tokenizer()
    kwargs = dict(
        model_obj=model,
        tokenizer=tokenizer,
        device="cpu",
        model_name="test-model",
        max_tokens_default=64,
    )
    kwargs.update(overrides)
    return _create_app(**kwargs)


def test_anthropic_messages_route_registered():
    """POST /v1/messages must be present on the FastAPI app."""
    app = _build_app()
    routes = {(r.path, tuple(sorted(r.methods))) for r in app.routes if hasattr(r, "methods")}
    assert ("/v1/messages", ("POST",)) in routes


def test_anthropic_messages_rejects_malformed_payload(monkeypatch):
    """Schema validation propagates to HTTP 400 — no internal crash."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = _build_app()
    client = TestClient(app)
    # Missing required `messages` field.
    response = client.post("/v1/messages", json={"model": "x", "max_tokens": 16})
    assert response.status_code == 400


def test_anthropic_messages_rejects_streaming():
    """v0.53.7: `stream=True` now returns 200 SSE; the 501 stub is gone.

    Coverage of the SSE shape itself lives in test_v0537.py; here we just
    assert the v0.53.6 stub is no longer in effect.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = _build_app()
    client = TestClient(app)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": True,
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code != 501


def test_anthropic_messages_happy_path(monkeypatch):
    """End-to-end: payload → from_anthropic → chat_completions mock → Anthropic shape."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # Patch _generate_response so we don't need a real model.
    monkeypatch.setattr(
        "soup_cli.commands.serve._generate_response",
        lambda *a, **kw: ("hello world", 3, 2),
    )
    app = _build_app()
    client = TestClient(app)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello world"}]
    assert body["model"] == "test-model"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 3
    assert body["usage"]["output_tokens"] == 2


# ----------------------------------------------------------------------
# #104 — n-gram speculative decoding wiring
# ----------------------------------------------------------------------


def test_ngram_config_threaded_into_generate_response(monkeypatch):
    """`_create_app(ngram_config=...)` forwards through to `_generate_response`."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from soup_cli.utils.ngram_spec import NgramSpecConfig

    received_kwargs: dict = {}

    def _capture(*args, **kwargs):
        received_kwargs.update(kwargs)
        return "ok", 1, 1

    monkeypatch.setattr("soup_cli.commands.serve._generate_response", _capture)
    cfg = NgramSpecConfig(n=3, num_draft_tokens=4, prompt_lookup_max=0)
    app = _build_app(ngram_config=cfg)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200, response.text
    assert received_kwargs.get("ngram_config") is cfg


def test_ngram_kwarg_emits_prompt_lookup_num_tokens():
    """Direct unit test on `_generate_response` proves the kwarg makes it
    into ``model.generate``."""
    pytest.importorskip("torch")
    from soup_cli.commands.serve import _generate_response
    from soup_cli.utils.ngram_spec import NgramSpecConfig

    captured: dict = {}

    class _FakeOutput:
        def __getitem__(self, _):
            class _T:
                shape = (0, 5)

                def __getitem__(self, _):
                    return []

            return _T()

    class _FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured.update(kwargs)

            class _Tensor:
                def __getitem__(self, _):
                    return []

            return [[0, 1, 2, 3, 4]]

    class _FakeTokenizer:
        pad_token_id = 0
        chat_template = None  # forces "Assistant:" fallback path

        def __call__(self, _text, return_tensors=None):
            import torch as _torch

            ids = _torch.tensor([[1, 2, 3]])
            return {"input_ids": ids, "attention_mask": _torch.ones_like(ids)}

        def decode(self, _tokens, skip_special_tokens=True):
            return "hello"

    cfg = NgramSpecConfig(n=3, num_draft_tokens=7, prompt_lookup_max=0)
    _generate_response(
        _FakeModel(),
        _FakeTokenizer(),
        [{"role": "user", "content": "hi"}],
        max_tokens=4,
        ngram_config=cfg,
    )
    assert captured.get("prompt_lookup_num_tokens") == 7


def test_ngram_kwarg_skipped_when_draft_model_set():
    """A real `assistant_model` wins over n-gram — keys mutually exclusive."""
    pytest.importorskip("torch")
    from soup_cli.commands.serve import _generate_response
    from soup_cli.utils.ngram_spec import NgramSpecConfig

    captured: dict = {}

    class _FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[0, 1, 2, 3]]

    class _FakeTokenizer:
        pad_token_id = 0
        chat_template = None

        def __call__(self, _text, return_tensors=None):
            import torch as _torch

            ids = _torch.tensor([[1, 2]])
            return {"input_ids": ids, "attention_mask": _torch.ones_like(ids)}

        def decode(self, _tokens, skip_special_tokens=True):
            return ""

    cfg = NgramSpecConfig(n=3, num_draft_tokens=4)
    _generate_response(
        _FakeModel(),
        _FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        max_tokens=2,
        assistant_model=object(),  # truthy
        ngram_config=cfg,
    )
    assert "prompt_lookup_num_tokens" not in captured
    assert captured.get("assistant_model") is not None


# ----------------------------------------------------------------------
# #103 — Server-side tool endpoints (stub: 501)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/v1/tools/python", "/v1/tools/web_search"],
)
def test_tool_endpoint_returns_501(path: str):
    """v0.53.7: python + web_search are live; the 501 stub is gone.

    Per-endpoint live coverage lives in ``test_v0537.py``. The ``bash``
    endpoint was reverted to 501 by the v0.53.7 review (C1) — its 501
    surface is asserted in ``test_v0537.py::TestReviewFixesC1BashStub``.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = _build_app()
    client = TestClient(app)
    response = client.post(path, json={"code": "print(1)"})
    assert response.status_code != 501


# ----------------------------------------------------------------------
# #106 — recipe_run stub + `soup data recipe --execute`
# ----------------------------------------------------------------------


def test_run_recipe_rejects_non_dag():
    from soup_cli.utils.recipe_run import run_recipe

    with pytest.raises(TypeError, match="RecipeDAG"):
        run_recipe("not-a-dag", output_dir="out")  # type: ignore[arg-type]


def test_run_recipe_rejects_bad_kwargs():
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises(TypeError, match="output_dir"):
        run_recipe(dag, output_dir=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resume"):
        run_recipe(dag, output_dir="out", resume="yes")  # type: ignore[arg-type]


def test_run_recipe_raises_not_implemented():
    """v0.53.7: live runner replaces the NotImplementedError stub.

    A bare `seed` node without a `path` config raises ValueError; coverage of
    the live happy-path lives in test_v0537.py.
    """
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises((ValueError, NotImplementedError)):
        run_recipe(dag, output_dir="out")


def test_data_recipe_execute_flag_present(tmp_path: Path):
    """`soup data recipe --help` lists `--execute`."""
    from soup_cli.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["data", "recipe", "--help"])
    assert result.exit_code == 0, result.output
    # Strip ANSI: CI terminals split `--execute` across Rich style spans.
    assert "--execute" in _strip_ansi(result.output)


def test_data_recipe_execute_requires_output(tmp_path: Path, monkeypatch):
    """`--execute` without `--output` exits 2 with a clear message."""
    from soup_cli.cli import app as cli_app

    monkeypatch.chdir(tmp_path)
    Path("recipe.yaml").write_text(
        "nodes:\n"
        "  - name: a\n"
        "    kind: seed\n"
        "edges: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["data", "recipe", "recipe.yaml", "--execute"],
    )
    assert result.exit_code == 2, result.output
    assert "--output" in _strip_ansi(result.output)


def test_data_recipe_execute_surfaces_v0537_marker(tmp_path: Path, monkeypatch):
    """v0.53.7: `--execute --output <dir>` invokes the live runner.

    A bare `seed` node without a `path` config now fails fast at runtime
    (live runner does real I/O). The v0.53.7 marker check from the v0.53.6
    stub era is no longer applicable; live coverage lives in test_v0537.py.
    """
    from soup_cli.cli import app as cli_app

    monkeypatch.chdir(tmp_path)
    Path("recipe.yaml").write_text(
        "nodes:\n"
        "  - name: a\n"
        "    kind: seed\n"
        "edges: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        [
            "data",
            "recipe",
            "recipe.yaml",
            "--execute",
            "--output",
            "out",
        ],
    )
    # Bare seed node without `path` fails at runtime — exit code non-zero.
    assert result.exit_code != 0


# ----------------------------------------------------------------------
# #105 — instantiate_trainer_plugins stub
# ----------------------------------------------------------------------


def test_instantiate_trainer_plugins_validates_then_raises():
    """v0.53.7: live instantiation replaces the NotImplementedError stub.

    `grokfast` requires the upstream `grokfast` package; in a CI environment
    without it the call surfaces an ImportError (or runs through to advisory
    when present). Live coverage lives in test_v0537.py.
    """
    from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins

    with pytest.raises((NotImplementedError, ImportError, ModuleNotFoundError)):
        instantiate_trainer_plugins(["grokfast"])


def test_instantiate_trainer_plugins_validation_runs_first():
    """Unknown plugin name → ValueError BEFORE NotImplementedError."""
    from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins

    with pytest.raises(ValueError, match="unknown trainer plugin"):
        instantiate_trainer_plugins(["definitely-not-a-real-plugin"])


def test_instantiate_trainer_plugins_rejects_non_sequence():
    from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins

    with pytest.raises(TypeError):
        instantiate_trainer_plugins("grokfast")  # type: ignore[arg-type]


def test_instantiate_trainer_plugins_in_dunder_all():
    """Public API surface includes the new stub."""
    import soup_cli.utils.trainer_plugins as mod

    assert "instantiate_trainer_plugins" in mod.__all__


def test_instantiate_trainer_plugins_empty_list_passes_validation():
    """v0.53.7: empty list now returns empty tuple (live runner).

    The v0.53.6 stub raised NotImplementedError on empty input; the live
    runner short-circuits to an empty tuple.
    """
    from soup_cli.utils.trainer_plugins import instantiate_trainer_plugins

    result = instantiate_trainer_plugins([])
    assert result == ()


# ----------------------------------------------------------------------
# Review-fix coverage: regression guards + missing boundaries
# ----------------------------------------------------------------------


def test_ngram_kwarg_omitted_when_config_is_none():
    """Default `ngram_config=None` → `prompt_lookup_num_tokens` NOT emitted.

    Regression guard: a future refactor that always emits the kwarg
    would break the legacy free-form generation path.
    """
    pytest.importorskip("torch")
    from soup_cli.commands.serve import _generate_response

    captured: dict = {}

    class _FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[0, 1, 2]]

    class _FakeTokenizer:
        pad_token_id = 0
        chat_template = None

        def __call__(self, _text, return_tensors=None):
            import torch as _torch

            ids = _torch.tensor([[1]])
            return {"input_ids": ids, "attention_mask": _torch.ones_like(ids)}

        def decode(self, _tokens, skip_special_tokens=True):
            return ""

    _generate_response(
        _FakeModel(),
        _FakeTokenizer(),
        [{"role": "user", "content": "x"}],
        max_tokens=1,
    )
    assert "prompt_lookup_num_tokens" not in captured


def test_run_recipe_rejects_empty_output_dir():
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises(TypeError, match="output_dir"):
        run_recipe(dag, output_dir="")


def test_run_recipe_rejects_non_str_judge_provider():
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises(TypeError, match="judge_provider"):
        run_recipe(dag, output_dir="out", judge_provider=123)  # type: ignore[arg-type]


def test_run_recipe_rejects_non_str_judge_model():
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises(TypeError, match="judge_model"):
        run_recipe(dag, output_dir="out", judge_model=b"bytes")  # type: ignore[arg-type]


def test_run_recipe_rejects_non_str_checkpoint_dir():
    from soup_cli.utils.recipe_dag import RecipeDAG, RecipeNode
    from soup_cli.utils.recipe_run import run_recipe

    dag = RecipeDAG(
        nodes=(RecipeNode(name="a", kind="seed", config={}),),
        edges=(),
        topo_order=("a",),
    )
    with pytest.raises(TypeError, match="checkpoint_dir"):
        run_recipe(dag, output_dir="out", checkpoint_dir=99)  # type: ignore[arg-type]


def test_anthropic_messages_validation_detail_redacted(monkeypatch):
    """Validation errors must surface as generic 'Invalid request' — no
    internal validator detail in the HTTP body. Security review M1."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = _build_app()
    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={"model": "x", "messages": [], "max_tokens": 16},  # empty msgs
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"] == "Invalid request"


def test_anthropic_messages_rejects_oversize_max_tokens():
    """`max_tokens` above the v0.30.0 16384 cap is rejected by the
    underlying validator (defence-in-depth — also covered upstream)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = _build_app()
    client = TestClient(app)
    response = client.post(
        "/v1/messages",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999_999,
        },
    )
    assert response.status_code == 400


def test_data_recipe_execute_rejects_empty_output(tmp_path: Path, monkeypatch):
    """`--output ""` is rejected at the CLI boundary."""
    from soup_cli.cli import app as cli_app

    monkeypatch.chdir(tmp_path)
    Path("recipe.yaml").write_text(
        "nodes:\n  - name: a\n    kind: seed\nedges: []\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["data", "recipe", "recipe.yaml", "--execute", "--output", ""],
    )
    assert result.exit_code == 2, result.output
    assert "must be a non-empty path" in _strip_ansi(result.output)


def test_data_recipe_execute_rejects_outside_cwd(tmp_path: Path, monkeypatch):
    """`--output` outside cwd is rejected before the stub runs."""
    import sys

    from soup_cli.cli import app as cli_app

    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "proj"
    sub.mkdir()
    monkeypatch.chdir(sub)
    Path("recipe.yaml").write_text(
        "nodes:\n  - name: a\n    kind: seed\nedges: []\n",
        encoding="utf-8",
    )
    outside = str(tmp_path / "outside")
    # On Windows, an absolute path under tmp_path.parent (different
    # drive in some CI envs) is still not under cwd=tmp_path/proj.
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["data", "recipe", "recipe.yaml", "--execute", "--output", outside],
    )
    # Either outside-cwd (preferred) or some platform-specific reject —
    # the live runner should NEVER fire.
    assert result.exit_code == 2, (result.output, sys.platform)
    assert "v0.53.7" not in _strip_ansi(result.output), (
        "outside-cwd output must NOT reach the live runner stub"
    )
