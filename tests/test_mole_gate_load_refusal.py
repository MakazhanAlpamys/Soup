"""A ``mole_gate.pt`` that ``weights_only=True`` refuses is refused by name, not as a traceback.

Two readers load the MoLE gate: ``utils.mole_routing.load_mole_for_serve`` (``soup serve
--mole``) and the MoLE trainer's resume (``_load_from_checkpoint``). Both already loaded it with
``weights_only=True``, so a file that needs code to load never ran anything; but the refusal
surfaced as a raw ``UnpicklingError`` / ``RuntimeError`` traceback. Both now go through
``load_gate_state``, which raises a ``ValueError`` that names the file and says what to do (the
type the rest of the CLI maps to a clean exit).

The "payload" only creates a marker file under ``tmp_path``.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest


def _run_payload(marker: str) -> str:
    Path(marker).write_text("payload ran", encoding="utf-8")
    return "ran"


class _Payload:
    """A pickle whose load calls ``_run_payload``; ``weights_only=True`` refuses it."""

    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self):
        return (_run_payload, (self.marker,))


def _gate_state(hidden: int = 4, experts: int = 2) -> dict:
    import torch

    weight = torch.arange(hidden * experts, dtype=torch.float32).reshape(experts, hidden)
    return {"gate.weight": weight}


def _write(path: Path, kind: str, marker: Path) -> None:
    import torch

    if kind == "valid":
        torch.save(_gate_state(), path)
    elif kind == "needs-code":
        torch.save({"gate.weight": _Payload(marker)}, path)
    elif kind == "truncated":
        torch.save(_gate_state(), path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
    else:  # pragma: no cover
        raise AssertionError(kind)


# --------------------------------------------------------------------------- the helper


def test_the_helper_reads_a_plain_gate_file(tmp_path):
    import torch

    from soup_cli.utils.mole_routing import load_gate_state

    path = tmp_path / "mole_gate.pt"
    _write(path, "valid", tmp_path / "marker")

    state = load_gate_state(str(path))

    assert torch.equal(state["gate.weight"], _gate_state()["gate.weight"])


@pytest.mark.parametrize("kind", ["needs-code", "truncated"])
def test_the_helper_refuses_by_name_and_never_runs_the_file(tmp_path, kind):
    from soup_cli.utils.mole_routing import load_gate_state

    marker = tmp_path / "marker"
    path = tmp_path / "mole_gate.pt"
    _write(path, kind, marker)

    with pytest.raises(ValueError) as excinfo:
        load_gate_state(str(path))

    text = str(excinfo.value)
    assert "mole_gate.pt" in text and "was refused" in text, text
    assert "Delete this file" in text and "written by soup" in text, text
    assert not marker.exists()


def test_the_payload_fixture_is_live_under_the_unsafe_load_and_inert_under_the_helper(tmp_path):
    """Control: without it, "the marker is absent" above could pass vacuously."""
    import torch

    from soup_cli.utils.mole_routing import load_gate_state

    marker = tmp_path / "marker"
    path = tmp_path / "mole_gate.pt"
    _write(path, "needs-code", marker)

    with pytest.raises(ValueError):
        load_gate_state(str(path))
    assert not marker.exists(), "the helper ran the file"

    torch.load(path, map_location="cpu", weights_only=False)

    assert marker.read_text(encoding="utf-8") == "payload ran", "the fixture is not live"


# --------------------------------------------------------------------------- the trainer resume


def _resume_target():
    import torch

    class _Gate(torch.nn.Module):
        """Shaped like the real gate: one ``gate`` linear, so keys are ``gate.weight``."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Linear(4, 2, bias=False)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.mole_gate = _Gate()

    return _Model()


def _trainer():
    from soup_cli.trainer.mole_routing import make_mole_trainer_class

    class _StubTrainer:
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            return None

    return object.__new__(make_mole_trainer_class(_StubTrainer))


def test_resume_loads_a_plain_gate_file(tmp_path):
    import torch

    ckpt = tmp_path / "checkpoint-2"
    ckpt.mkdir()
    _write(ckpt / "mole_gate.pt", "valid", tmp_path / "marker")
    model = _resume_target()

    _trainer()._load_from_checkpoint(str(ckpt), model=model)

    assert torch.equal(model.mole_gate.gate.weight, _gate_state()["gate.weight"])


@pytest.mark.parametrize("kind", ["needs-code", "truncated"])
def test_resume_refuses_a_bad_gate_file_by_name(tmp_path, kind):
    import torch

    marker = tmp_path / "marker"
    ckpt = tmp_path / "checkpoint-2"
    ckpt.mkdir()
    _write(ckpt / "mole_gate.pt", kind, marker)
    model = _resume_target()
    before = model.mole_gate.gate.weight.detach().clone()

    with pytest.raises(ValueError) as excinfo:
        _trainer()._load_from_checkpoint(str(ckpt), model=model)

    text = str(excinfo.value)
    assert "mole_gate.pt" in text and "was refused" in text, text
    assert not marker.exists()
    assert torch.equal(model.mole_gate.gate.weight, before)


# --------------------------------------------------------------------------- serve


def _stub_the_heavy_loaders(monkeypatch, hidden: int):
    """``load_mole_for_serve`` loads a base model, a tokenizer and PEFT adapters first."""
    import peft
    import transformers

    class _Peft:
        def load_adapter(self, path, adapter_name=None):
            return None

        def parameters(self):
            return []

        def eval(self):
            return self

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *a, **k: types.SimpleNamespace(pad_token="x", eos_token="x")),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(
            lambda *a, **k: types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=hidden))
        ),
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", staticmethod(lambda *a, **k: _Peft()))


@pytest.mark.parametrize("kind", ["needs-code", "truncated"])
def test_serve_refuses_a_bad_gate_file_by_name(tmp_path, monkeypatch, kind):
    from soup_cli.utils.mole_routing import (
        MoleServeManifest,
        load_mole_for_serve,
        write_mole_manifest,
    )

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "marker"
    run = tmp_path / "run"
    write_mole_manifest(
        MoleServeManifest(
            base="stub-base",
            adapters=("adapter-a", "adapter-b"),
            num_task_adapters=2,
            hidden_dim=64,
            top_k=2,
            temperature=1.0,
        ),
        str(run),
    )
    _write(run / "mole_gate.pt", kind, marker)
    _stub_the_heavy_loaders(monkeypatch, hidden=64)

    with pytest.raises(ValueError) as excinfo:
        load_mole_for_serve("run", device="cpu")

    text = str(excinfo.value)
    assert "mole_gate.pt" in text and "was refused" in text, text
    assert not marker.exists()
