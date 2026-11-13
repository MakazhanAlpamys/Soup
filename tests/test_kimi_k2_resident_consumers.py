"""Native Kimi K2 loading for resident-trained adapters across CLI consumers."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest


def _patch_adapter_load(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from soup_cli.utils import gpu, kimi_k2, trust_remote

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "moonshotai/Kimi-K2-Base"}),
        encoding="utf-8",
    )
    model = Mock()
    model.to.return_value = model
    tokenizer = Mock(pad_token=None, eos_token="<eos>")
    load_resident = Mock(return_value=model)
    peft_load = Mock(return_value=model)
    monkeypatch.setattr(kimi_k2, "load_resident_model", load_resident)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_load)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        Mock(side_effect=AssertionError("Kimi K2 must use the resident loader")),
    )
    monkeypatch.setattr(
        gpu,
        "resolve_inference_device_map_and_dtype",
        lambda _device: ("cpu", pytest.importorskip("torch").float32),
    )
    monkeypatch.setattr(
        trust_remote, "model_requires_trust_remote_code", lambda _path: False
    )
    monkeypatch.setattr(
        trust_remote,
        "resolve_trust_remote_code",
        lambda _target, *, requested, **_kwargs: requested,
    )
    return adapter, model, load_resident, peft_load


@pytest.mark.parametrize("consumer", ["infer", "chat", "diff", "serve"])
def test_adapter_command_loaders_use_the_resident_route(consumer, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    adapter, model, load_resident, peft_load = _patch_adapter_load(monkeypatch, tmp_path)
    base = "moonshotai/Kimi-K2-Base"

    if consumer == "infer":
        from soup_cli.commands.infer import _load_model

        result = _load_model(str(adapter), None, "cpu")
    elif consumer == "chat":
        from soup_cli.commands.chat import _load_model

        result = _load_model(str(adapter), base, True, "cpu")
    elif consumer == "diff":
        from soup_cli.commands.diff import _load_model

        result = _load_model(str(adapter), None, "cpu")
    else:
        from soup_cli.commands.serve import _load_model

        result = _load_model(str(adapter), base, True, "cpu")

    assert result[0] is model
    load_resident.assert_called_once()
    args, kwargs = load_resident.call_args
    assert args == (base,)
    assert kwargs["model_kwargs"]["trust_remote_code"] is False
    assert kwargs["model_kwargs"]["device_map"] == "cpu"
    assert kwargs["model_kwargs"]["dtype"] is pytest.importorskip("torch").float32
    peft_load.assert_called_once_with(model, str(adapter))
    AutoTokenizer.from_pretrained.assert_called_once()


def test_live_eval_adapter_loads_k2_base_through_the_resident_route(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    adapter, model, load_resident, peft_load = _patch_adapter_load(monkeypatch, tmp_path)
    from soup_cli.utils import run_provenance
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    monkeypatch.setattr(run_provenance, "base_revision_kwargs", lambda _base: {})
    monkeypatch.setattr(run_provenance, "recorded_base_kwargs", lambda _adapter, _base: {})

    loaded, _tokenizer, device = load_model_and_tokenizer(
        "moonshotai/Kimi-K2-Base", adapter=str(adapter), device="cpu"
    )

    assert loaded is model
    assert str(device) == "cpu"
    load_resident.assert_called_once_with(
        "moonshotai/Kimi-K2-Base",
        model_kwargs={"trust_remote_code": False},
    )
    peft_load.assert_called_once_with(model, str(adapter))
    AutoTokenizer.from_pretrained.assert_called_once()


def test_export_merge_loads_k2_base_through_the_resident_route(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from soup_cli.commands import export as export_module
    from soup_cli.utils import kimi_k2, run_provenance, trust_remote

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    merged_dir = tmp_path / "merged"
    base = "moonshotai/Kimi-K2-Base"
    model = Mock()
    model.save_pretrained = Mock(
        side_effect=lambda path: (Path(path) / "config.json").write_text("{}", encoding="utf-8")
    )
    peft_model = Mock()
    peft_model.merge_and_unload.return_value = model
    peft_load = Mock(return_value=peft_model)
    resident_load = Mock(return_value=Mock())
    tokenizer = Mock()
    monkeypatch.setattr(kimi_k2, "load_resident_model", resident_load)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_load)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        Mock(side_effect=AssertionError("Kimi K2 must use the resident loader")),
    )
    monkeypatch.setattr(trust_remote, "model_requires_trust_remote_code", lambda _path: False)
    monkeypatch.setattr(
        trust_remote,
        "resolve_trust_remote_code",
        lambda _target, *, requested, **_kwargs: requested,
    )
    monkeypatch.setattr(run_provenance, "recorded_base_kwargs", lambda _adapter, _base: {})

    export_module._merge_adapter(str(adapter), base, str(merged_dir))

    resident_load.assert_called_once_with(
        base,
        model_kwargs={
            "dtype": torch.float16,
            "trust_remote_code": False,
            "device_map": "cpu",
        },
        console=export_module.console,
    )
    peft_load.assert_called_once_with(resident_load.return_value, str(adapter))
    model.save_pretrained.assert_called_once_with(str(merged_dir))
    tokenizer.save_pretrained.assert_called_once_with(str(merged_dir))


def test_merge_command_loads_k2_base_through_the_resident_route(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from soup_cli.commands.merge import merge
    from soup_cli.utils import run_provenance

    adapter, model, load_resident, _peft_load = _patch_adapter_load(monkeypatch, tmp_path)
    merged = Mock()

    def save_merged(path):
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    merged.save_pretrained = Mock(side_effect=save_merged)
    model.merge_and_unload.return_value = merged
    monkeypatch.setattr(run_provenance, "recorded_base_kwargs", lambda _adapter, _base: {})
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=Mock())
    )
    monkeypatch.chdir(tmp_path)

    merge(
        ctx=None,
        adapter=str(adapter),
        backend="transformers",
        dequantize=False,
        base="moonshotai/Kimi-K2-Base",
        output="merged",
        dtype="float16",
        trust_remote_code=False,
        save_format="fp16",
        double_quant=True,
        low_memory=False,
        hub="hf",
    )

    load_resident.assert_called_once()
    assert load_resident.call_args.args == ("moonshotai/Kimi-K2-Base",)
    assert load_resident.call_args.kwargs["model_kwargs"] == {
        "dtype": torch.float16,
        "trust_remote_code": False,
        "device_map": "cpu",
    }
    merged.save_pretrained.assert_called_once_with("merged")
    assert (tmp_path / "merged" / "config.json").is_file()


def test_low_memory_merge_refuses_k2_fp8_before_reading_weights(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("transformers")
    import typer

    from soup_cli.commands.merge import _merge_low_memory
    from soup_cli.utils import lowmem_merge

    base = tmp_path / "k2"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps({"model_type": "kimi_k2"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    forbidden = Mock(side_effect=AssertionError("low-memory merge read K2 safetensors"))
    monkeypatch.setattr(lowmem_merge, "merge_low_memory", forbidden)

    with pytest.raises(typer.Exit) as caught:
        _merge_low_memory(adapter, str(base), tmp_path / "merged", "float16", False)

    assert caught.value.exit_code == 1
    output = " ".join(capsys.readouterr().out.split())
    assert "stored safetensors directly" in output
    forbidden.assert_not_called()


def test_mole_serve_loads_k2_base_through_the_resident_route(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    from types import SimpleNamespace

    from soup_cli.utils import kimi_k2, mole_routing

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "mole-run"
    run_dir.mkdir()
    (run_dir / "mole_gate.pt").write_bytes(b"")
    manifest = SimpleNamespace(
        base="moonshotai/Kimi-K2-Base",
        adapters=("adapter-0", "adapter-1"),
        hidden_dim=64,
        num_task_adapters=2,
        temperature=1.0,
        top_k=1,
    )
    monkeypatch.setattr(mole_routing, "load_mole_manifest", lambda _path: manifest)

    base_model = Mock(config=SimpleNamespace(hidden_size=64))
    load_resident = Mock(return_value=base_model)
    monkeypatch.setattr(kimi_k2, "load_resident_model", load_resident)
    direct_load = Mock(side_effect=AssertionError("bypassed resident K2 loader"))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", direct_load
    )
    tokenizer = Mock(pad_token="<pad>", eos_token="<eos>")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    model = Mock()
    model.parameters.return_value = ()
    peft_load = Mock(return_value=model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", peft_load)
    gate = Mock()
    monkeypatch.setattr(
        mole_routing, "build_gating_kernel", Mock(return_value=gate)
    )
    monkeypatch.setattr(torch, "load", Mock(return_value={}))

    console = Mock()
    runtime = mole_routing.load_mole_for_serve(
        "mole-run", device="cpu", trust_remote_code=True, console=console
    )

    load_resident.assert_called_once_with(
        "moonshotai/Kimi-K2-Base",
        model_kwargs={
            "trust_remote_code": True,
            "device_map": None,
            "dtype": torch.float32,
        },
        console=console,
    )
    direct_load.assert_not_called()
    assert runtime.model is model
