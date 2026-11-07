"""Execute each resident trainer's actual HF call expression with strict receivers.

This inventories the load boundary rather than trusting one representative
trainer: removing the revision argument at any base call must fail independently.
Model computation is covered separately using locally constructed HF models.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from soup_cli.utils import run_provenance as provenance

BASE = "owner/base"
REVISION = "a" * 40
ROOT = Path(__file__).resolve().parents[1]
_BASE_ARGUMENTS = {
    "cfg.base", "config.base", "self.config.base", "base", "base_model", "model_id", "model_name",
}


def _load_calls():
    paths = sorted((ROOT / "src/soup_cli/trainer").glob("*.py"))
    paths += [ROOT / f"src/soup_cli/utils/{name}.py" for name in ("cut_ce", "liger")]
    cases = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for call in ast.walk(ast.parse(text)):
            if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
                    or call.func.attr != "from_pretrained" or not call.args):
                continue
            receiver = ast.get_source_segment(text, call.func.value)
            if receiver == "PeftModel":
                continue  # Its first argument is a loaded model, not a Hub repository.
            if ast.get_source_segment(text, call.args[0]) in _BASE_ARGUMENTS:
                cases.append(pytest.param(call, receiver, id=f"{path.stem}:{call.lineno}"))
    assert len(cases) >= 40, "the inventory must cover the whole resident base-loader surface"
    return cases


class _Values:
    def __getattr__(self, name):
        if name == "base":
            return BASE
        return self

    def __call__(self, *args, **kwargs):
        return None


class _Environment(dict):
    def __missing__(self, name):
        if name.endswith("kwargs"):
            return {}
        return _Values()


@pytest.mark.parametrize("call,receiver", _load_calls())
def test_every_actual_base_loader_call_receives_the_bound_commit(call, receiver, monkeypatch):
    seen = []

    def load(model_id, *args, revision, **kwargs):
        assert model_id == BASE
        assert revision == REVISION
        seen.append((model_id, revision))

    env = _Environment(
        base=BASE, base_model=BASE, model_id=BASE, model_name=BASE,
        cfg=_Values(), config=_Values(), self=_Values(),
        base_revision_kwargs=provenance.base_revision_kwargs,
    )
    env[receiver] = SimpleNamespace(from_pretrained=load)
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha=REVISION),
    ))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    expression = compile(ast.Expression(body=call), "<actual base load>", "eval")
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        eval(expression, env)
    assert seen == [(BASE, REVISION)]


def _stream_setup_tree():
    path = ROOT / "src/soup_cli/trainer/stream_setup.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    model_code = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_STREAM_NATIVE_MODEL_CODE"
            for target in node.targets
        )
    )
    assert model_code is False
    return tree, model_code


def _from_pretrained_calls(tree, receiver):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver
            and node.func.attr == "from_pretrained"]


def _bind_revision(monkeypatch):
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha=REVISION),
    ))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)


@pytest.mark.parametrize("kimi_k2", [False, True], ids=["autoconfig", "kimi-k2-raw-read"])
def test_streamed_config_load_preserves_native_model_policy_and_bound_revision(
    kimi_k2, monkeypatch,
):
    """Execute the merged call through the real loader: tokenizer trust must not enable remote
    model code, and every read the loader makes carries the bound commit.

    #2107 moved the streamed model's config read from ``AutoConfig.from_pretrained`` in
    stream_setup into ``utils/kimi_k2.load_stream_model_config``, which reads ``AutoConfig``
    and, for a Kimi K2 file ``AutoConfig`` refuses as custom code, the raw ``config.json``.
    Both reads are strict receivers here: a dropped revision or a True policy fails."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, PretrainedConfig

    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from tests.test_kimi_k2_streaming import K2_BASE_CONFIG

    tree, model_code = _stream_setup_tree()
    # No second, unguarded config read beside the loader.
    assert _from_pretrained_calls(tree, "AutoConfig") == []
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "load_stream_model_config"]
    assert len(calls) == 1
    seen = []
    native = object()

    def auto_config(model_id, *, trust_remote_code, revision):
        assert model_id == BASE and revision == REVISION
        assert trust_remote_code is False
        seen.append("AutoConfig")
        if kimi_k2:
            raise ValueError("The repository contains custom code")
        return native

    def raw_read(model_id, *, revision, name_or_path):
        assert model_id == name_or_path == BASE and revision == REVISION
        seen.append("get_config_dict")
        return dict(K2_BASE_CONFIG), {}

    monkeypatch.setattr(AutoConfig, "from_pretrained", auto_config)
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", raw_read)
    env = {
        "cfg": SimpleNamespace(base=BASE),
        "self": SimpleNamespace(_trust_remote_code=True, _STREAM_NATIVE_MODEL_CODE=model_code),
        "load_stream_model_config": load_stream_model_config,
        "base_revision_kwargs": provenance.base_revision_kwargs,
    }
    _bind_revision(monkeypatch)
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        config = eval(
            compile(ast.Expression(body=calls[0]), "<streamed config load>", "eval"), env,
        )
    if kimi_k2:
        assert seen == ["AutoConfig", "get_config_dict"]
        assert config.model_type == "deepseek_v3"
    else:
        assert seen == ["AutoConfig"]
        assert config is native


def test_streamed_tokenizer_load_keeps_user_trust_bound_revision_and_the_native_config(
    monkeypatch,
):
    """The tokenizer gets the user's trust flag, the bound commit, and the model config already
    loaded: without ``config=``, ``AutoTokenizer`` reads the config again with the user's flag
    and, for a repository with an ``auto_map`` (Kimi K2, Moonlight), imports its code."""
    tree, model_code = _stream_setup_tree()
    calls = _from_pretrained_calls(tree, "AutoTokenizer")
    assert len(calls) == 1
    seen = []
    model_config = object()

    def load(model_id, *, config, trust_remote_code, revision):
        assert model_id == BASE and revision == REVISION
        assert config is model_config
        assert trust_remote_code is True
        seen.append(model_id)

    env = {
        "cfg": SimpleNamespace(base=BASE),
        "self": SimpleNamespace(_trust_remote_code=True, _STREAM_NATIVE_MODEL_CODE=model_code),
        "AutoTokenizer": SimpleNamespace(from_pretrained=load),
        "model_config": model_config,
        "base_revision_kwargs": provenance.base_revision_kwargs,
    }
    _bind_revision(monkeypatch)
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        eval(compile(ast.Expression(body=calls[0]), "<streamed tokenizer load>", "eval"), env)
    assert seen == [BASE]
