"""#2609: k-bit preparation keeps the non-quantised weights in bf16 on bf16 cards.

PEFT's ``prepare_model_for_kbit_training`` casts every fp16/bf16 parameter that is not a
``Params4bit`` to fp32 (embeddings, norms, the output head). On a resident 4-bit
Qwen2.5-7B SFT step on an RTX 4090 that fp32 residual stream cost about 25% of the step
and 3.8 GiB of the peak (``benchmarks/gate-2609-kbit-fp32-ab.md``). ``prepare_kbit_model``
(``soup_cli/utils/kbit_prepare.py``) calls PEFT and casts back what PEFT upcast, when the
run computes in bf16; a card without bf16 keeps PEFT's fp32.

The model is a tiny Llama on the CPU with the 4-bit flag set by hand, so no bitsandbytes
and no GPU: the dtype decision is exactly what PEFT does to the dtypes, which is
what these tests observe. The scanner at the bottom is the guard that a new trainer
cannot call PEFT's function directly and quietly bring the fp32 cast back (the #328/#336
shape: a helper wired into some wrappers and forgotten in the next one).
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli"
_HELPER = _SRC / "utils" / "kbit_prepare.py"


def _tiny_model(dtype):
    """A 2-layer Llama (untied head) with the 4-bit flag set by hand."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(dtype)
    model.is_loaded_in_4bit = True
    return model


def _dtypes(model) -> dict:
    return {name: param.dtype for name, param in model.named_parameters()}


def _group(model, fragment: str) -> list:
    return [dtype for name, dtype in _dtypes(model).items() if fragment in name]


class TestBf16Device:
    """compute dtype bf16: what PEFT upcast to fp32 goes back to bf16."""

    def _prepared(self, **overrides):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        kwargs = {"use_gradient_checkpointing": True, "compute_dtype": torch.bfloat16}
        kwargs.update(overrides)
        return prepare_kbit_model(model, **kwargs)

    def test_embeddings_norms_and_head_keep_bf16(self):
        model = self._prepared()
        assert _group(model, "embed_tokens") == [torch.bfloat16]
        assert _group(model, "lm_head") == [torch.bfloat16]
        norms = _group(model, "norm")
        assert len(norms) == 5  # input + post-attention per layer, and the final norm
        assert set(norms) == {torch.bfloat16}

    def test_every_parameter_that_loaded_bf16_is_bf16(self):
        # Not a hand-picked list: anything PEFT upcast comes back.
        assert set(_dtypes(self._prepared()).values()) == {torch.bfloat16}

    def test_base_parameters_are_frozen(self):
        model = self._prepared()
        assert [name for name, p in model.named_parameters() if p.requires_grad] == []

    def test_input_grads_and_checkpointing_are_enabled(self):
        model = self._prepared()
        assert model.is_gradient_checkpointing is True
        # Frozen embeddings still hand the checkpointed blocks an input that
        # requires grad; that is what enable_input_require_grads() installs.
        ids = torch.tensor([[1, 2, 3]])
        assert model.get_input_embeddings()(ids).requires_grad is True

    def test_checkpointing_off_stays_off(self):
        model = self._prepared(use_gradient_checkpointing=False)
        assert model.is_gradient_checkpointing is False
        assert set(_dtypes(model).values()) == {torch.bfloat16}

    def test_it_returns_the_prepared_model(self):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        assert prepare_kbit_model(
            model, use_gradient_checkpointing=False, compute_dtype=torch.bfloat16
        ) is model

    def test_a_forward_and_backward_runs_with_a_bf16_stream(self):
        # The point of the change: embeddings out, norms and head in bf16 end to end.
        model = self._prepared()
        ids = torch.tensor([[1, 2, 3, 4]])
        out = model(input_ids=ids, labels=ids)
        assert out.logits.dtype is torch.bfloat16
        out.loss.backward()  # reaches the checkpointed blocks without a grad-less error


class TestWhatPeftLeavesAlone:
    """Only what PEFT upcast comes back; nothing else is touched."""

    def test_adapter_parameters_keep_what_peft_gave_them(self):
        # An adapter that already existed (the #1503 / #1573 resume paths load
        # trainable adapters in their own dtype) is not the helper's to cast back.
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        proj = model.model.layers[0].self_attn.q_proj
        proj.register_parameter(
            "lora_A", torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
        )
        prepare_kbit_model(
            model, use_gradient_checkpointing=False, compute_dtype=torch.bfloat16
        )
        assert proj.lora_A.dtype is torch.float32  # PEFT's cast, unchanged by the helper
        assert model.model.layers[0].self_attn.q_proj.weight.dtype is torch.bfloat16

    def test_a_parameter_loaded_in_fp32_stays_fp32(self):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        model.model.norm.weight.data = model.model.norm.weight.data.float()
        prepare_kbit_model(
            model, use_gradient_checkpointing=False, compute_dtype=torch.bfloat16
        )
        assert model.model.norm.weight.dtype is torch.float32
        assert model.lm_head.weight.dtype is torch.bfloat16

    def test_a_parameter_loaded_in_fp16_keeps_peft_fp32_on_a_bf16_card(self):
        # Not the compute dtype, so not restored: the unmeasured case keeps today's numerics.
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        model.model.norm.weight.data = model.model.norm.weight.data.half()
        prepare_kbit_model(
            model, use_gradient_checkpointing=False, compute_dtype=torch.bfloat16
        )
        assert model.model.norm.weight.dtype is torch.float32

    def test_a_packed_4bit_parameter_is_not_touched(self):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        class Params4bit(torch.nn.Parameter):  # PEFT skips it by class name
            pass

        model = _tiny_model(torch.bfloat16)
        packed = Params4bit(torch.zeros(4, 4, dtype=torch.bfloat16), requires_grad=False)
        model.model.layers[0].self_attn.q_proj.weight = packed
        prepare_kbit_model(
            model, use_gradient_checkpointing=False, compute_dtype=torch.bfloat16
        )
        assert model.model.layers[0].self_attn.q_proj.weight.dtype is torch.bfloat16


class TestCardsWithoutBf16:
    """An fp16-only card, a run that trains in fp16, and the CPU keep PEFT's fp32."""

    @pytest.mark.parametrize(
        ("compute_dtype", "loaded"),
        [
            (torch.float16, torch.float16),  # a T4: the loader gives fp16 there
            (torch.float16, torch.bfloat16),  # grpo_fp16 or an auto pick of fp16 on a bf16 card
            (torch.float32, torch.bfloat16),  # the CPU
        ],
        ids=["fp16-only-card", "fp16-run-on-a-bf16-card", "cpu"],
    )
    def test_everything_stays_fp32(self, compute_dtype, loaded):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(loaded)
        prepare_kbit_model(
            model, use_gradient_checkpointing=True, compute_dtype=compute_dtype
        )
        assert set(_dtypes(model).values()) == {torch.float32}
        assert [n for n, p in model.named_parameters() if p.requires_grad] == []
        assert model.is_gradient_checkpointing is True


class TestPeftCall:
    """The arguments reach PEFT unchanged."""

    @pytest.fixture
    def captured(self, monkeypatch):
        import peft

        calls = []

        def fake(model, **kwargs):
            calls.append(kwargs)
            return model

        monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fake)
        return calls

    def test_use_gradient_checkpointing_is_passed_explicitly(self, captured):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        for flag in (True, False):
            prepare_kbit_model(
                _tiny_model(torch.bfloat16),
                use_gradient_checkpointing=flag,
                compute_dtype=torch.float32,
            )
        assert [call["use_gradient_checkpointing"] for call in captured] == [True, False]

    def test_checkpointing_kwargs_are_forwarded_only_when_given(self, captured):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        model = _tiny_model(torch.bfloat16)
        prepare_kbit_model(
            model, use_gradient_checkpointing=True, compute_dtype=torch.float32
        )
        prepare_kbit_model(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            compute_dtype=torch.float32,
        )
        assert "gradient_checkpointing_kwargs" not in captured[0]
        assert captured[1]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    def test_the_compute_dtype_must_be_a_torch_dtype(self, captured):
        from soup_cli.utils.kbit_prepare import prepare_kbit_model

        for bad in ("bf16", None, 16):
            with pytest.raises(TypeError, match="compute_dtype must be a torch.dtype"):
                prepare_kbit_model(
                    _tiny_model(torch.bfloat16),
                    use_gradient_checkpointing=True,
                    compute_dtype=bad,
                )
        assert captured == []  # refused before PEFT touched the model


class TestKbitComputeDtype:
    @pytest.fixture
    def flags(self, monkeypatch):
        import soup_cli.utils.gpu as gpu

        def _set(bf16, fp16):
            monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda device, **kw: (bf16, fp16))

        return _set

    def test_bf16_card(self, flags):
        from soup_cli.utils.kbit_prepare import kbit_compute_dtype

        flags(True, False)
        assert kbit_compute_dtype("cuda") is torch.bfloat16

    def test_fp16_only_card(self, flags):
        from soup_cli.utils.kbit_prepare import kbit_compute_dtype

        flags(False, True)
        assert kbit_compute_dtype("cuda") is torch.float16

    def test_no_mixed_precision_device(self, flags):
        from soup_cli.utils.kbit_prepare import kbit_compute_dtype

        flags(False, False)
        assert kbit_compute_dtype("cpu") is torch.float32

    def test_a_run_that_asks_for_fp16_on_a_bf16_card_is_fp16(self, flags):
        # grpo_fp16 and auto_mixed_precision can train in fp16 on a bf16 card.
        from soup_cli.utils.kbit_prepare import kbit_compute_dtype

        flags(True, False)
        assert kbit_compute_dtype("cuda", prefer_fp16=True) is torch.float16

    def test_the_cpu_answer_is_not_bf16(self):
        from soup_cli.utils.kbit_prepare import kbit_compute_dtype

        assert kbit_compute_dtype("cpu") is torch.float32


# --- the trainers: the dtype reaches the helper -------------------------------------


def _setup_trainer(monkeypatch, *, task, module, cls_name, flags, training=None):
    """Run a trainer's ``_setup_transformers`` on a tiny bf16 Llama with the real PEFT
    preparation, the card's bf16/fp16 answer set to ``flags``, and return the model."""
    import importlib
    import types
    from types import SimpleNamespace

    import peft

    from soup_cli.config.schema import SoupConfig

    wrapper_cls = getattr(importlib.import_module(module), cls_name)
    cfg = SoupConfig(
        base="test-model",
        data={"train": "./data.jsonl", "format": "alpaca"},
        task=task,
        training={"quantization": "4bit", "gradient_checkpointing": True, **(training or {})},
    )
    model = _tiny_model(torch.bfloat16)

    class _Tokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
    )
    fake_peft = types.SimpleNamespace(
        LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=lambda model_obj, _cfg: model_obj,
        prepare_model_for_kbit_training=peft.prepare_model_for_kbit_training,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(
        "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("soup_cli.utils.gpu.bf16_fp16_flags", lambda device, **kw: flags)

    wrapper = object.__new__(wrapper_cls)
    wrapper.config = cfg
    wrapper.device = "cpu"  # the card's answer is the patched bf16_fp16_flags above
    wrapper._trust_remote_code = False
    wrapper.model = None
    wrapper.tokenizer = None
    if hasattr(wrapper_cls, "_resolve_mixed_precision"):
        # An auto_mixed_precision pick of fp16 on a bf16 card (qwen2 and phi-3 families).
        wrapper._resolve_mixed_precision = lambda tcfg, base: (False, True)
    wrapper._setup_transformers(cfg, cfg.training)
    return wrapper.model


_TRAINERS = [
    ("sft", "soup_cli.trainer.sft", "SFTTrainerWrapper"),
    ("dpo", "soup_cli.trainer.dpo", "DPOTrainerWrapper"),
    ("grpo", "soup_cli.trainer.grpo", "GRPOTrainerWrapper"),
]


@pytest.mark.parametrize("task,module,cls_name", _TRAINERS)
class TestTrainersReachTheHelper:
    def test_a_bf16_card_keeps_bf16(self, monkeypatch, task, module, cls_name):
        model = _setup_trainer(
            monkeypatch, task=task, module=module, cls_name=cls_name, flags=(True, False)
        )
        assert set(_dtypes(model).values()) == {torch.bfloat16}
        assert [n for n, p in model.named_parameters() if p.requires_grad] == []

    def test_an_fp16_only_card_keeps_fp32(self, monkeypatch, task, module, cls_name):
        model = _setup_trainer(
            monkeypatch, task=task, module=module, cls_name=cls_name, flags=(False, True)
        )
        assert set(_dtypes(model).values()) == {torch.float32}


def test_grpo_fp16_keeps_fp32_on_a_bf16_card(monkeypatch):
    model = _setup_trainer(
        monkeypatch,
        task="grpo",
        module="soup_cli.trainer.grpo",
        cls_name="GRPOTrainerWrapper",
        flags=(True, False),
        training={"grpo_fp16": True},
    )
    assert set(_dtypes(model).values()) == {torch.float32}


def test_an_auto_mixed_precision_pick_of_fp16_keeps_fp32_on_a_bf16_card(monkeypatch):
    model = _setup_trainer(
        monkeypatch,
        task="sft",
        module="soup_cli.trainer.sft",
        cls_name="SFTTrainerWrapper",
        flags=(True, False),
        training={"auto_mixed_precision": True},
    )
    assert set(_dtypes(model).values()) == {torch.float32}


# --- the scanner ---------------------------------------------------------------------

_PEFT_FUNCTION = "prepare_model_for_kbit_training"
_HELPER_NAME = "prepare_kbit_model"


def _source_files():
    return sorted(p for p in _SRC.rglob("*.py") if p != _HELPER)


def _mentions_peft_function(tree: ast.AST) -> list[int]:
    """Lines where the PEFT function is imported, named, attributed or passed by name."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == _PEFT_FUNCTION for alias in node.names
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == _PEFT_FUNCTION:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == _PEFT_FUNCTION:
            lines.append(node.lineno)
        elif isinstance(node, ast.Constant) and node.value == _PEFT_FUNCTION:
            lines.append(node.lineno)  # getattr(peft, "prepare_model_for_kbit_training")
    return lines


def test_no_module_but_the_helper_calls_peft_kbit_prep_directly():
    """Every direct reference to PEFT's function is the fp32 cast coming back (#2609).

    Scan, don't hand-write a list: a trainer added tomorrow would otherwise ship with the
    fp32 residual stream and nothing would say so.
    """
    found = {
        f"{path.relative_to(_SRC).as_posix()}:{line}"
        for path in _source_files()
        for line in _mentions_peft_function(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert not found, (
        f"{_PEFT_FUNCTION} is referenced outside utils/kbit_prepare.py: {sorted(found)}; "
        f"call {_HELPER_NAME}(...) instead"
    )


def test_the_helper_is_the_one_place_that_calls_peft():
    tree = ast.parse(_HELPER.read_text(encoding="utf-8"))
    assert len(_mentions_peft_function(tree)) >= 2  # the import and the call


def _helper_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == _HELPER_NAME)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == _HELPER_NAME)
        )
    ]


def test_every_call_passes_the_checkpointing_flag_and_a_compute_dtype():
    calls = {}
    for path in _source_files():
        for call in _helper_calls(ast.parse(path.read_text(encoding="utf-8"))):
            calls.setdefault(path.relative_to(_SRC).as_posix(), []).append(call)
    for rel, found in calls.items():
        for call in found:
            keywords = {kw.arg for kw in call.keywords}
            assert {"use_gradient_checkpointing", "compute_dtype"} <= keywords, (
                f"{rel}:{call.lineno} calls {_HELPER_NAME} without "
                "use_gradient_checkpointing and compute_dtype"
            )


def test_the_fifteen_trainer_call_sites_use_the_helper():
    """The sites #2609 names: sft x3 (text, vision, audio) and twelve other trainers."""
    expected = {
        "trainer/sft.py": 3,
        "trainer/bco.py": 1,
        "trainer/dpo.py": 1,
        "trainer/embedding.py": 1,
        "trainer/grpo.py": 1,
        "trainer/ipo.py": 1,
        "trainer/kto.py": 1,
        "trainer/online_dpo.py": 1,
        "trainer/orpo.py": 1,
        "trainer/ppo.py": 1,
        "trainer/pretrain.py": 1,
        "trainer/reward_model.py": 1,
        "trainer/simpo.py": 1,
    }
    counts = {}
    for path in _source_files():
        found = len(_helper_calls(ast.parse(path.read_text(encoding="utf-8"))))
        if found:
            counts[path.relative_to(_SRC).as_posix()] = found
    assert counts == expected


def test_importing_the_helper_does_not_import_torch():
    """Runtime property (an AST guard cannot see a transitive import)."""
    import subprocess

    code = (
        "import sys; import soup_cli.utils.kbit_prepare; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
