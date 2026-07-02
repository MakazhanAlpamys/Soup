"""Regression tests for the CRITICAL findings in CODE_REVIEW.md.

Each test asserts the *specific* previously-broken behavior is now correct, and
does so end-to-end where feasible. The bugs shipped because the unit tests
exercised helpers in isolation and never hit the real call site / wiring.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import soup_cli
from soup_cli.config.loader import load_config_from_string

# ─────────────────────────── CRITICAL 1: RLVR verifiable rewards ───────────────

_GRPO_YAML = """
base: hf-internal-testing/tiny-random-gpt2
task: grpo
data:
  train: train.jsonl
  format: chatml
training:
  reward_fn: verifiable
  verifiable_domain: math
  num_generations: 2
"""

_PPO_YAML = """
base: hf-internal-testing/tiny-random-gpt2
task: ppo
data:
  train: train.jsonl
  format: chatml
training:
  reward_fn: verifiable
  verifiable_domain: math
"""


def test_load_reward_fn_requires_domain_and_config_carries_it():
    """The verifiable_domain is load-bearing; the config carries it."""
    from soup_cli.trainer.rewards import load_reward_fn, math_verify_reward

    cfg = load_config_from_string(_GRPO_YAML)
    # This is exactly what the fixed grpo.setup / ppo._setup_reward now do.
    fn = load_reward_fn(
        cfg.training.reward_fn, verifiable_domain=cfg.training.verifiable_domain
    )
    assert fn is math_verify_reward
    # The old (broken) call site dropped the domain, raising ValueError.
    with pytest.raises(ValueError):
        load_reward_fn(cfg.training.reward_fn)


def test_ppo_setup_reward_loads_verifiable_end_to_end():
    """PPO's real reward-setup path must load a verifiable reward, not crash."""
    from soup_cli.trainer.ppo import PPOTrainerWrapper
    from soup_cli.trainer.rewards import math_verify_reward

    cfg = load_config_from_string(_PPO_YAML)
    # Bypass __init__ (which resolves trust_remote_code) — we only exercise
    # the reward-setup method that carried the bug.
    wrapper = object.__new__(PPOTrainerWrapper)
    wrapper.reward_model_instance = None
    wrapper.reward_fn = None
    wrapper.device = "cpu"
    wrapper.trust_remote_code = False
    # Pre-fix this raised ValueError("reward_fn='verifiable' requires ...").
    wrapper._setup_reward(cfg, cfg.training)
    assert wrapper.reward_fn is math_verify_reward


def test_grpo_setup_passes_verifiable_domain(monkeypatch):
    """GRPO's real setup() must forward verifiable_domain to load_reward_fn."""
    pytest.importorskip("trl")
    pytest.importorskip("datasets")
    import soup_cli.trainer.grpo as grpo_mod

    captured: dict = {}

    class _StopError(Exception):
        pass

    def _spy(spec, verifiable_domain=None):
        captured["spec"] = spec
        captured["domain"] = verifiable_domain
        raise _StopError()

    # grpo.setup() does `from soup_cli.trainer.rewards import load_reward_fn`
    # at call time, so patching the source module attribute is effective.
    monkeypatch.setattr("soup_cli.trainer.rewards.load_reward_fn", _spy)

    cfg = load_config_from_string(_GRPO_YAML)
    wrapper = grpo_mod.GRPOTrainerWrapper(cfg, device="cpu")
    with pytest.raises(_StopError):
        wrapper.setup({})
    assert captured["spec"] == "verifiable"
    assert captured["domain"] == "math"


# ─────────────────────────── CRITICAL 2: serve auth / host default ─────────────


def test_serve_host_defaults_to_loopback():
    from soup_cli.commands.serve import serve

    host_param = inspect.signature(serve).parameters["host"].default
    # typer.Option(...) returns an OptionInfo whose .default holds the value.
    assert getattr(host_param, "default", host_param) == "127.0.0.1"


def test_serve_exposes_tool_auth_token_option():
    from soup_cli.commands.serve import serve

    assert "tool_auth_token" in inspect.signature(serve).parameters


def _build_tool_app(auth_token):
    from soup_cli.commands.serve import _create_app

    return _create_app(
        model_obj=None,
        tokenizer=None,
        device="cpu",
        model_name="t",
        max_tokens_default=64,
        auth_token=auth_token,
    )


def test_tool_python_endpoint_requires_bearer_when_token_set():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(_build_tool_app("secret"))
    # Auth is checked before body inspection, so an empty body still 401s.
    assert client.post("/v1/tools/python", json={}).status_code == 401
    assert (
        client.post(
            "/v1/tools/python", json={}, headers={"Authorization": "Bearer nope"}
        ).status_code
        == 401
    )


def test_tool_python_gate_is_noop_without_token():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    client = TestClient(_build_tool_app(None))
    # No token configured -> auth passes -> empty body fails the code check
    # with 400 (NOT 401). Proves the gate is opt-in, and reachable.
    assert client.post("/v1/tools/python", json={}).status_code == 400


# ─────────────────────────── CRITICAL 3: eval benchmark injection ──────────────


def test_eval_benchmark_rejects_trust_remote_code_injection(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.commands.eval import app

    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "attacker/m,trust_remote_code=True"}',
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["benchmark", "--model", "adapter"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "trust_remote_code" in result.output or "delimiters" in result.output


# ─────────────────────────── CRITICAL 4: webhook SSRF ──────────────────────────


def test_webhook_rejects_https_to_internal_ips():
    from soup_cli.utils.webhooks import validate_webhook_url

    for url in (
        "https://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://10.0.0.1/hook",
        "https://192.168.1.10/hook",
        "https://172.16.0.5/hook",
        "http://169.254.169.254/",  # already rejected pre-fix, keep covered
    ):
        with pytest.raises(ValueError):
            validate_webhook_url(url)


def test_webhook_allows_public_https_and_loopback():
    from soup_cli.utils.webhooks import validate_webhook_url

    assert (
        validate_webhook_url("https://hooks.slack.com/services/T/B/x")
        == "https://hooks.slack.com/services/T/B/x"
    )
    assert validate_webhook_url("http://localhost:8000/hook") == "http://localhost:8000/hook"
    assert validate_webhook_url("https://localhost/hook") == "https://localhost/hook"


# ─────────────────────────── CRITICAL 5: modal stub injection ──────────────────


def test_modal_stub_cannot_inject_via_output_dir():
    from soup_cli.cloud.modal import render_modal_stub

    payload = '"); import os; os.system("evil") #'
    stub = render_modal_stub(
        "task: sft\nbase: x\n", gpu="a100", output_dir=payload, soup_version="0.0.1"
    )
    tree = ast.parse(stub)  # must parse — no breakout from the string literal

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "os" not in imported, "output_dir injected an `import os` statement"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "system", "injected os.system() call present"

    # The payload survives only as inert data inside a repr'd string literal.
    literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert any(payload in s for s in literals)


# ─────────────────────────── CRITICAL 6: eval-gate wiring ──────────────────────


class _StubDisplay:
    def start(self, total_steps=0):
        self.total_steps = total_steps


class _FakeArgs:
    pass


class _FakeState:
    def __init__(self, epoch=0, max_steps=1):
        self.epoch = epoch
        self.max_steps = max_steps
        self.global_step = 0


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False


def _write_failing_gate(tmp_path):
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt": "2+2?", "expected": "4", "scoring": "exact"}\n', encoding="utf-8"
    )
    (tmp_path / "gate.yaml").write_text(
        "suite: t\n"
        "tasks:\n"
        "  - type: custom\n"
        "    name: math\n"
        "    threshold: 0.9\n"
        "    tasks: tasks.jsonl\n"
        "    scorer: exact\n",
        encoding="utf-8",
    )


def test_eval_gate_halts_training_end_to_end(tmp_path, monkeypatch):
    """A configured gate must load its suite at train-begin and actually stop."""
    monkeypatch.chdir(tmp_path)
    _write_failing_gate(tmp_path)

    from soup_cli.config.schema import EvalGateConfig
    from soup_cli.monitoring.callback import SoupTrainerCallback

    cfg = EvalGateConfig(
        enabled=True,
        suite="gate.yaml",
        on_regression="stop",
        every_n_epochs=1,
        regression_threshold=0.05,
    )
    cb = SoupTrainerCallback(_StubDisplay(), eval_gate_config=cfg)
    # Inject a wrong-answer generator so we don't need a real model. The suite
    # itself is loaded from config by on_train_begin — the wiring under test.
    cb._gate_generate_fn = lambda prompt: "wrong"

    args, state, control = _FakeArgs(), _FakeState(epoch=0, max_steps=1), _FakeControl()
    cb.on_train_begin(args, state, control)
    assert cb._gate_suite is not None  # previously ALWAYS None -> gate was dead

    state.epoch = 1
    cb.on_epoch_end(args, state, control)
    assert control.should_training_stop is True


def test_no_eval_gate_config_never_halts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.monitoring.callback import SoupTrainerCallback

    cb = SoupTrainerCallback(_StubDisplay())  # eval_gate_config=None
    args, state, control = _FakeArgs(), _FakeState(epoch=1, max_steps=1), _FakeControl()
    cb.on_train_begin(args, state, control)
    cb.on_epoch_end(args, state, control)
    assert control.should_training_stop is False


_TRAINERS_WITH_CALLBACK = [
    "sft", "dpo", "grpo", "ppo", "kto", "orpo", "simpo", "ipo",
    "bco", "pretrain", "reward_model", "distill", "embedding", "classifier",
]


@pytest.mark.parametrize("name", _TRAINERS_WITH_CALLBACK)
def test_trainer_wires_eval_gate_config(name):
    """Every trainer that builds SoupTrainerCallback must pass eval_gate_config."""
    src = (Path(soup_cli.__file__).parent / "trainer" / f"{name}.py").read_text(
        encoding="utf-8"
    )
    assert "eval_gate_config=" in src, f"{name}.py does not wire eval_gate_config"
