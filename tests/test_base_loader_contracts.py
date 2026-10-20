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


@pytest.mark.parametrize("receiver,remote_code", [("AutoConfig", False), ("AutoTokenizer", True)])
def test_streamed_loads_preserve_native_model_policy_and_bound_revision(
    receiver, remote_code, monkeypatch,
):
    """Execute the merged calls: tokenizer trust must not enable remote model code."""
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
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver
             and node.func.attr == "from_pretrained"]
    assert len(calls) == 1
    seen = []

    def load(model_id, *, trust_remote_code, revision):
        assert model_id == BASE and revision == REVISION
        assert trust_remote_code is remote_code
        seen.append(model_id)

    env = {
        "cfg": SimpleNamespace(base=BASE),
        "self": SimpleNamespace(_trust_remote_code=True, _STREAM_NATIVE_MODEL_CODE=model_code),
        receiver: SimpleNamespace(from_pretrained=load),
        "base_revision_kwargs": provenance.base_revision_kwargs,
    }
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha=REVISION),
    ))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with provenance.base_revision_scope(provenance.BaseRevision(BASE)):
        eval(compile(ast.Expression(body=calls[0]), "<streamed base load>", "eval"), env)
    assert seen == [BASE]
