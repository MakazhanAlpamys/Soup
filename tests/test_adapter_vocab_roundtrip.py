"""#3351: consume real Soup SFT vocabulary-expanded saves without losing rows or IDs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests._tiny_hf_models import build_llama, build_tokenizer

    root = tmp_path_factory.mktemp("vocab-artifacts")
    outputs = {}
    with torch.random.fork_rng(devices=[]), pytest.MonkeyPatch.context() as patcher:
        patcher.chdir(root)
        patcher.setenv("HF_HUB_OFFLINE", "1")
        patcher.setenv("TRANSFORMERS_OFFLINE", "1")
        original_post_init = transformers.TrainingArguments.__post_init__

        def cpu_post_init(arguments):
            arguments.use_cpu = True
            original_post_init(arguments)

        patcher.setattr(transformers.TrainingArguments, "__post_init__", cpu_post_init)
        for tied, kind in [(False, "plain"), (False, "regular"), (False, "special"),
                           (True, "regular"), (True, "special")]:
            torch.manual_seed(3351)
            label = f"{kind}-{'tied' if tied else 'untied'}"
            base = root / f"base-{label}"
            tokenizer = build_tokenizer()
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] + ': ' "
                "+ message['content'] + eos_token }}{% endfor %}"
            )
            base_model = build_llama(tokenizer)
            base_model.config.tie_word_embeddings = tied
            if tied:
                base_model.tie_weights()
            base_model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            token = "<|soup_new_token|>"
            extra = {
                "plain": "", "regular": f"  add_new_tokens: ['{token}']\n",
                "special": f"  new_special_tokens: ['{token}']\n",
            }[kind]
            config = load_config_from_string(
                f"base: {base.as_posix()}\nbackend: transformers\ntask: sft\n"
                "data:\n  train: unused.jsonl\n  format: chatml\n  max_length: 64\n"
                f"  train_on_responses_only: false\n{extra}"
                "training:\n  epochs: 1\n  batch_size: 1\n  gradient_accumulation_steps: 1\n"
                "  quantization: none\n  gradient_checkpointing: false\n  warmup_ratio: 0.0\n"
                "  lr: 0.01\n  save_steps: 1000\n  logging_steps: 1\n"
                "  lora: {r: 2, alpha: 4, dropout: 0.0, target_modules: [q_proj]}\n"
                f"output: {label}\n"
            )
            wrapper = SFTTrainerWrapper(config, device="cpu")
            wrapper.setup({"train": [{"messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": f"hi {token}"},
            ]}] * 2})
            assert next(wrapper.model.parameters()).device.type == "cpu"
            before = {name: parameter.detach().clone()
                      for name, parameter in wrapper.model.named_parameters() if "lora_B" in name}
            wrapper.train()
            assert wrapper.trainer.state.global_step == 2
            assert any(not torch.equal(parameter, before[name])
                       for name, parameter in wrapper.model.named_parameters() if name in before)
            tokenizer = wrapper.tokenizer
            inputs = tokenizer(f"hello {token}", return_tensors="pt")
            wrapper.model.eval()
            with torch.no_grad():
                logits = wrapper.model(**inputs).logits.clone()
            outputs[label] = SimpleNamespace(
                base=base, adapter=root / label, tokenizer=tokenizer, inputs=inputs, logits=logits,
                embedding=wrapper.model.get_input_embeddings().weight.detach().clone(),
                head=wrapper.model.get_output_embeddings().weight.detach().clone(), token=token,
                tied=tied, kind=kind,
            )
    return outputs


@pytest.fixture
def artifact(artifacts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    original = artifacts["regular-untied"]
    adapter = tmp_path / "adapter"
    shutil.copytree(original.adapter, adapter)
    return SimpleNamespace(**{**vars(original), "adapter": adapter})


@pytest.mark.parametrize("label", [
    "plain-untied", "regular-untied", "special-untied", "regular-tied", "special-tied",
])
@pytest.mark.parametrize("consumer", ["evaluation", "merge"])
def test_real_sft_save_roundtrips(artifacts, tmp_path, monkeypatch, label, consumer):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.cli import app
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    saved = artifacts[label]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    if consumer == "evaluation":
        model, tokenizer, device = load_model_and_tokenizer(
            str(saved.base), adapter=str(saved.adapter), device="cpu", dtype="float32",
        )
        assert device == "cpu"
    else:
        result = CliRunner().invoke(app, [
            "merge", "--adapter", str(saved.adapter), "--output", "merged", "--dtype", "float32",
        ])
        assert result.exit_code == 0, result.output
        model = AutoModelForCausalLM.from_pretrained("merged").eval()
        tokenizer = AutoTokenizer.from_pretrained("merged")
    assert tokenizer.get_vocab() == saved.tokenizer.get_vocab()
    assert tokenizer.all_special_tokens == saved.tokenizer.all_special_tokens
    assert tokenizer(saved.token, add_special_tokens=False) == saved.tokenizer(
        saved.token, add_special_tokens=False,
    )
    assert torch.equal(model.get_input_embeddings().weight, saved.embedding)
    assert torch.equal(model.get_output_embeddings().weight, saved.head)
    if saved.tied:
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    with torch.no_grad():
        torch.testing.assert_close(model(**saved.inputs).logits, saved.logits)


@pytest.mark.parametrize("consumer", ["evaluation", "merge"])
@pytest.mark.parametrize("damage,diagnostic", [
    ("no-tokenizer", "tokenizer"),
    ("no-embedding", "embedding"),
    ("no-head", "head"),
    ("no-rows", "embedding"),
    ("empty-weights", "embedding"),
    ("bad-shape", "shape"),
    ("bad-rank", "shape"),
    ("empty-embedding", "shape"),
    ("duplicate-shape", "shapes disagree"),
    ("bad-row-count", "shapes disagree"),
    ("tokenizer-too-large", "does not fit"),
    ("remapped", "token ID"),
])
def test_incomplete_expanded_artifacts_fail_before_output(artifact, consumer, damage, diagnostic):
    from safetensors.torch import load_file, save_file

    from soup_cli.cli import app
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    weights_path = artifact.adapter / "adapter_model.safetensors"
    weights = load_file(str(weights_path))
    if damage == "no-tokenizer":
        for path in artifact.adapter.glob("*token*"):
            path.unlink()
    elif damage == "remapped":
        path = artifact.adapter / "tokenizer.json"
        data = json.loads(path.read_text())
        merges = data["model"]["vocab"]
        first, second = [word for word in merges if word != "<eos>"][:2]
        merges[first], merges[second] = merges[second], merges[first]
        path.write_text(json.dumps(data))
    elif damage == "tokenizer-too-large":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(artifact.adapter)
        tokenizer.add_tokens(["<|extra_missing_row|>"])
        tokenizer.save_pretrained(artifact.adapter)
    else:
        key = next(name for name in weights if (
            "lm_head" in name if damage == "no-head" else "embed_tokens" in name
        ))
        if damage == "bad-shape":
            weights[key] = weights[key][:, :-1].contiguous()
        elif damage == "bad-rank":
            weights[key] = weights[key].flatten()
        elif damage == "empty-embedding":
            weights[key] = weights[key][:0].contiguous()
        elif damage == "duplicate-shape":
            weights[key.removesuffix("weight") + "base_layer.weight"] = (
                weights[key][:-1].clone()
            )
        elif damage == "bad-row-count":
            weights[key] = weights[key][:-1].contiguous()
        elif damage == "no-rows":
            weights = {name: value for name, value in weights.items()
                       if "embed_tokens" not in name and "lm_head" not in name}
        elif damage == "empty-weights":
            weights = {}
        else:
            weights.pop(key)
        save_file(weights, str(weights_path))
    if consumer == "evaluation":
        with pytest.raises(ValueError, match=diagnostic):
            load_model_and_tokenizer(
                str(artifact.base), adapter=str(artifact.adapter), device="cpu",
            )
    else:
        result = CliRunner().invoke(app, [
            "merge", "--adapter", str(artifact.adapter), "--output", "merged", "--dtype", "float32",
        ])
        assert result.exit_code == 1
        assert diagnostic in _plain(result.output)
        assert not Path("merged").exists()


def test_legacy_adapter_without_tokenizer_keeps_base_ids(artifacts, tmp_path, monkeypatch):
    import torch

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    saved = artifacts["plain-untied"]
    adapter = tmp_path / "legacy"
    shutil.copytree(saved.adapter, adapter)
    for path in adapter.glob("*token*"):
        path.unlink()
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda *args, **kwargs: pytest.fail(
        "local adapter reconstruction must not query the Hub",
    ))
    model, tokenizer, _ = load_model_and_tokenizer(
        str(saved.base), adapter=str(adapter), device="cpu",
    )
    assert tokenizer.get_vocab() == saved.tokenizer.get_vocab()
    with torch.no_grad():
        torch.testing.assert_close(model(**saved.inputs).logits, saved.logits)


@pytest.mark.parametrize("consumer", ["evaluation", "merge"])
def test_legacy_bin_serialization_uses_restricted_loading(artifact, monkeypatch, consumer):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from soup_cli.cli import app
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    path = artifact.adapter / "adapter_model.safetensors"
    torch.save(load_file(str(path)), artifact.adapter / "adapter_model.bin")
    path.unlink()
    original_load = torch.load
    calls = []

    def restricted_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", restricted_load)
    if consumer == "evaluation":
        model, tokenizer, _ = load_model_and_tokenizer(
            str(artifact.base), adapter=str(artifact.adapter), device="cpu",
        )
        assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    else:
        result = CliRunner().invoke(app, [
            "merge", "--adapter", str(artifact.adapter), "--output", "merged", "--dtype", "float32",
        ])
        assert result.exit_code == 0, result.output
        model = AutoModelForCausalLM.from_pretrained("merged").eval()
    assert calls
    assert torch.equal(model.get_input_embeddings().weight, artifact.embedding)
    assert torch.equal(model.get_output_embeddings().weight, artifact.head)
    with torch.no_grad():
        torch.testing.assert_close(model(**artifact.inputs).logits, artifact.logits)


@pytest.mark.parametrize("weights", [[1], {"not-a-tensor": 1}])
def test_bin_rejects_non_tensor_containers(artifact, weights):
    import torch

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    (artifact.adapter / "adapter_model.safetensors").unlink()
    torch.save(weights, artifact.adapter / "adapter_model.bin")
    with pytest.raises(ValueError, match="tensor state dictionary"):
        load_model_and_tokenizer(str(artifact.base), adapter=str(artifact.adapter), device="cpu")


@pytest.mark.parametrize("quantization", ["4bit", "8bit", "prequantized"])
def test_expanded_quantized_base_is_refused_before_resize(artifact, monkeypatch, quantization):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.adapter_vocab import prepare_adapter_vocab

    model = AutoModelForCausalLM.from_pretrained(artifact.base)
    if quantization == "prequantized":
        model.is_quantized = True
        quantization = None
    monkeypatch.setattr(model, "resize_token_embeddings", lambda *args, **kwargs: pytest.fail(
        "quantized reconstruction must refuse before resizing",
    ))
    with pytest.raises(ValueError, match="requires an unquantized base"):
        prepare_adapter_vocab(
            model, str(artifact.adapter), str(artifact.base),
            base_tokenizer=AutoTokenizer.from_pretrained(artifact.base), quantization=quantization,
        )


def test_remote_adapter_files_use_hub_resolution_without_base_revision(artifact, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.adapter_vocab import prepare_adapter_vocab

    original_tokenizer = AutoTokenizer.from_pretrained
    base_tokenizer = original_tokenizer(artifact.base)
    remote = "owner/vocab-adapter"
    files = []

    def cached_file(repo, filename, **kwargs):
        assert repo == remote
        assert kwargs == {"_raise_exceptions_for_missing_entries": False}
        files.append(filename)
        path = artifact.adapter / filename
        return str(path) if path.exists() else None

    def load_tokenizer(repo, **kwargs):
        assert repo == remote and kwargs == {"trust_remote_code": False}
        return original_tokenizer(artifact.adapter)

    monkeypatch.setattr("transformers.utils.hub.cached_file", cached_file)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(artifact.base)
    tokenizer = prepare_adapter_vocab(
        model, remote, str(artifact.base), base_tokenizer=base_tokenizer,
        revision_kwargs={"revision": "a" * 40},
    )
    assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    assert files == ["tokenizer_config.json", "adapter_model.safetensors"]
    assert model.get_input_embeddings().weight.shape == artifact.embedding.shape


def test_local_roundtrip_makes_no_dns_requests(artifact, monkeypatch):
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: pytest.fail(
        "local adapter reconstruction must not resolve DNS",
    ))
    model, tokenizer, _ = load_model_and_tokenizer(
        str(artifact.base), adapter=str(artifact.adapter), device="cpu",
    )
    assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    assert model.get_input_embeddings().weight.shape == artifact.embedding.shape


@pytest.mark.parametrize("implicit", [False, True])
def test_expanded_reload_validates_the_recorded_base(artifact, monkeypatch, implicit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    remote_base = "owner/recorded-base"
    revision = "a" * 40
    config_path = artifact.adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = remote_base
    config_path.write_text(json.dumps(config))
    (artifact.adapter / "soup_run.json").write_text(json.dumps({
        "schema_version": 1, "run_id": "expanded", "base_model": remote_base,
        "base_revision": revision, "base_revision_source": "hub_commit",
    }))
    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    reads = []

    def load_model(base, **kwargs):
        assert base == remote_base and kwargs.pop("revision") == revision
        reads.append("model")
        return original_model(artifact.base, **kwargs)

    def load_tokenizer(base, **kwargs):
        if base == remote_base:
            assert kwargs.pop("revision") == revision
            reads.append("base tokenizer")
            base = artifact.base
        else:
            assert "revision" not in kwargs
        return original_tokenizer(base, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    selected_base = str(artifact.adapter) if implicit else remote_base
    options = {} if implicit else {"adapter": str(artifact.adapter)}
    model, tokenizer, _ = load_model_and_tokenizer(selected_base, device="cpu", **options)
    assert sorted(reads) == ["base tokenizer", "model"]
    assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    with torch.no_grad():
        torch.testing.assert_close(model(**artifact.inputs).logits, artifact.logits)


@pytest.mark.parametrize("storage", ["lora", "modules_to_save"])
def test_embedding_storage_preserves_trained_values(artifact, tmp_path, storage):
    import copy

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from soup_cli.cli import app
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    base = AutoModelForCausalLM.from_pretrained(artifact.base)
    base.resize_token_embeddings(len(artifact.tokenizer))
    options = (
        {"target_modules": ["embed_tokens", "lm_head", "q_proj"]} if storage == "lora"
        else {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens", "lm_head"]}
    )
    trained = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, task_type="CAUSAL_LM", **options,
    ))
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.1)
    trained(**artifact.inputs, labels=artifact.inputs["input_ids"]).loss.backward()
    optimizer.step()
    trained.eval()
    with torch.no_grad():
        logits = trained(**artifact.inputs).logits.clone()
    adapter = tmp_path / "embedding-lora"
    trained.save_pretrained(adapter)
    artifact.tokenizer.save_pretrained(adapter)
    loaded, tokenizer, _ = load_model_and_tokenizer(
        str(artifact.base), adapter=str(adapter), device="cpu",
    )
    assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    with torch.no_grad():
        torch.testing.assert_close(loaded(**artifact.inputs).logits, logits)
    expected_merge = copy.deepcopy(trained).merge_and_unload()
    result = CliRunner().invoke(app, [
        "merge", "--adapter", str(adapter), "--output", "merged", "--dtype", "float32",
    ])
    assert result.exit_code == 0, result.output
    merged = AutoModelForCausalLM.from_pretrained("merged").eval()
    assert torch.equal(merged.get_input_embeddings().weight,
                       expected_merge.get_input_embeddings().weight)
    assert torch.equal(merged.get_output_embeddings().weight,
                       expected_merge.get_output_embeddings().weight)
    with torch.no_grad():
        torch.testing.assert_close(merged(**artifact.inputs).logits, logits)


@pytest.mark.parametrize("padding", [0, 8])
def test_saved_rows_override_base_padding(artifact, tmp_path, padding):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    padded = tmp_path / "padded-base"
    base = AutoModelForCausalLM.from_pretrained(artifact.base)
    base.resize_token_embeddings(len(artifact.tokenizer) + padding)
    base.save_pretrained(padded)
    AutoTokenizer.from_pretrained(artifact.base).save_pretrained(padded)
    model, tokenizer, _ = load_model_and_tokenizer(
        str(padded), adapter=str(artifact.adapter), device="cpu",
    )
    assert tokenizer.get_vocab() == artifact.tokenizer.get_vocab()
    assert torch.equal(model.get_input_embeddings().weight, artifact.embedding)
    with torch.no_grad():
        torch.testing.assert_close(model(**artifact.inputs).logits, artifact.logits)


def test_padding_cannot_hide_missing_expanded_rows(artifact):
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.adapter_vocab import prepare_adapter_vocab

    model = AutoModelForCausalLM.from_pretrained(artifact.base)
    model.resize_token_embeddings(len(artifact.tokenizer))
    path = artifact.adapter / "adapter_model.safetensors"
    saved = {name: value for name, value in load_file(str(path)).items() if "lm_head" not in name}
    save_file(saved, str(path))
    with pytest.raises(ValueError, match="missing its saved head"):
        prepare_adapter_vocab(
            model, str(artifact.adapter), str(artifact.base),
            base_tokenizer=AutoTokenizer.from_pretrained(artifact.base),
        )


@pytest.mark.parametrize("adapter", ["", " ", 5, True])
def test_invalid_adapter_is_rejected_before_loading(monkeypatch, adapter):
    import sys

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="adapter must be a non-empty string"):
        load_model_and_tokenizer("owner/base", adapter=adapter, device="cpu")


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_evaluation_forwards_quantized_refusal_before_resize(artifact, monkeypatch, quantization):
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    original_model = AutoModelForCausalLM.from_pretrained

    def load_model(*args, **kwargs):
        config = kwargs.pop("quantization_config")
        assert config.load_in_4bit if quantization == "4bit" else config.load_in_8bit
        model = original_model(*args, **kwargs)
        monkeypatch.setattr(model, "resize_token_embeddings", lambda *args, **kwargs: pytest.fail(
            "evaluation must forward the quantized setting before a resize",
        ))
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    with pytest.raises(ValueError, match="requires an unquantized base"):
        load_model_and_tokenizer(
            str(artifact.base), adapter=str(artifact.adapter), device="cpu",
            quantization=quantization,
        )


def test_sparse_added_id_must_fit_saved_rows(artifact, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.adapter_vocab import prepare_adapter_vocab

    tokenizer = AutoTokenizer.from_pretrained(artifact.adapter)
    vocabulary = tokenizer.get_vocab()
    vocabulary[artifact.token] = len(tokenizer) + 3
    monkeypatch.setattr(tokenizer, "get_vocab", lambda: vocabulary)
    with pytest.raises(ValueError, match="does not fit"):
        prepare_adapter_vocab(
            AutoModelForCausalLM.from_pretrained(artifact.base), str(artifact.adapter),
            str(artifact.base), tokenizer=tokenizer,
        )


def test_adapter_without_pad_token_keeps_eos_fallback(artifact):
    from transformers import AutoTokenizer

    from soup_cli.utils.live_eval import load_model_and_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact.adapter)
    tokenizer.pad_token = None
    tokenizer.save_pretrained(artifact.adapter)
    _, restored, _ = load_model_and_tokenizer(
        str(artifact.base), adapter=str(artifact.adapter), device="cpu",
    )
    assert restored.pad_token_id == restored.eos_token_id
    assert restored.get_vocab() == artifact.tokenizer.get_vocab()
