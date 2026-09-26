"""#837: fused Fast-LoRA SwiGLU MLP correctness tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _deps():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    return torch


def _randomise_b(model):
    torch = _deps()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.2)


def _make_model(
    targets,
    *,
    dropout=0.0,
    act="silu",
    module_act=None,
    ranks=None,
    bias=True,
    modules_to_save=None,
):
    _deps()
    import torch.nn as nn
    from peft import LoraConfig, inject_adapter_in_model

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 12, bias=bias)
            self.up_proj = nn.Linear(8, 12, bias=bias)
            self.down_proj = nn.Linear(12, 8, bias=bias)
            activation = module_act or act
            self.act_fn = nn.SiLU() if activation in {"silu", "swish"} else nn.GELU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_act=act)
            self.mlp = TinyMLP()

        def forward(self, x):
            return self.mlp(x)

    model = Model()
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(targets),
        rank_pattern=ranks or {},
        modules_to_save=modules_to_save,
    )
    inject_adapter_in_model(config, model)
    _randomise_b(model)
    return model


def _adapter_grads(model):
    return {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if "lora_" in name and p.grad is not None
    }


class TestMath:
    def test_float64_gradcheck_all_six_adapter_matrices_unequal_ranks(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import _mlp_function

        torch.manual_seed(3)
        x = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
        wg = torch.randn(12, 8, dtype=torch.float64)
        wu = torch.randn(12, 8, dtype=torch.float64)
        wd = torch.randn(8, 12, dtype=torch.float64)
        bg = torch.randn(12, dtype=torch.float64)
        bu = torch.randn(12, dtype=torch.float64)
        bd = torch.randn(8, dtype=torch.float64)
        ag = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        bgl = torch.randn(12, 2, dtype=torch.float64, requires_grad=True)
        au = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        bul = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
        ad = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
        bdl = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
        fn = _mlp_function()

        def call(x_, ag_, bgl_, au_, bul_, ad_, bdl_):
            return fn.apply(
                x_, wg, bg, wu, bu, wd, bd,
                ag_, bgl_, au_, bul_, ad_, bdl_,
                1.7, 0.8, 2.1, None, None, None,
            )

        assert torch.autograd.gradcheck(
            call, (x, ag, bgl, au, bul, ad, bdl), eps=1e-6, atol=1e-5, rtol=1e-4
        )

    @pytest.mark.parametrize(
        "targets",
        [
            ("gate_proj", "up_proj", "down_proj"),
            ("gate_proj",),
            ("up_proj",),
            ("down_proj",),
            ("gate_proj", "down_proj"),
        ],
    )
    def test_fp32_forward_and_backward_match_peft_for_partial_adapters(self, targets):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        torch.manual_seed(11)
        model = _make_model(targets)
        x_ref = torch.randn(2, 5, 8, requires_grad=True)
        ref = model(x_ref)
        loss_ref = ref.square().mean()
        loss_ref.backward()
        ref_x = x_ref.grad.detach().clone()

        ref_grads = _adapter_grads(model)

        for p in model.parameters():
            p.grad = None
        x_fast = x_ref.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        out = model(x_fast)
        out.square().mean().backward()

        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(x_fast.grad, ref_x)
        got_grads = _adapter_grads(model)
        assert got_grads.keys() == ref_grads.keys()
        for name in ref_grads:
            torch.testing.assert_close(got_grads[name], ref_grads[name], msg=name)

    def test_unequal_rank_pattern_matches_peft(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        torch.manual_seed(17)
        model = _make_model(
            ("gate_proj", "up_proj", "down_proj"),
            ranks={"gate_proj": 2, "up_proj": 3, "down_proj": 4},
        )
        ranks = tuple(
            model.mlp.__getattr__(name).lora_A["default"].weight.shape[0]
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        assert ranks == (2, 3, 4)

        x0 = torch.randn(7, 8, requires_grad=True)
        ref = model(x0)
        ref.sum().backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for p in model.parameters():
            p.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        got = model(x1)
        got.sum().backward()
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)


class TestPatching:
    def test_patch_is_idempotent_and_reversible(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj"))
        assert patch_fast_lora_mlp(model) == 1
        assert patch_fast_lora_mlp(model) == 0
        assert unpatch_fast_lora_mlp(model) == 1
        assert unpatch_fast_lora_mlp(model) == 0
        assert "forward" not in vars(model.mlp)

    def test_non_silu_activation_falls_back_without_patch(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj",), act="gelu")
        assert patch_fast_lora_mlp(model) == 0

    def test_nonzero_dropout_delegates_to_original_mlp(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj"), dropout=0.25)
        model.train()
        calls = []
        original = model.mlp.forward

        def spy(x):
            calls.append(True)
            return original(x)

        model.mlp.forward = spy
        assert patch_fast_lora_mlp(model) == 1
        torch.manual_seed(9)
        model(torch.randn(2, 8))
        assert calls == [True]

    def test_modules_to_save_projection_keeps_peft_ownership(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(
            ("gate_proj", "up_proj"), modules_to_save=["down_proj"]
        )

        assert hasattr(model.mlp.down_proj, "modules_to_save")
        assert patch_fast_lora_mlp(model) == 0

    def test_real_peft_8bit_layers_are_not_patched(self):
        _deps()
        bnb = pytest.importorskip("bitsandbytes")
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class Tiny8BitMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = bnb.nn.Linear8bitLt(8, 12, bias=False)
                self.up_proj = bnb.nn.Linear8bitLt(8, 12, bias=False)
                self.down_proj = bnb.nn.Linear8bitLt(12, 8, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.is_loaded_in_8bit = True
                self.mlp = Tiny8BitMLP()

        model = Model()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )

        assert all(
            type(getattr(model.mlp, name)).__name__ == "Linear8bitLt"
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        assert patch_fast_lora_mlp(model) == 0

    def test_saturation_is_finite_on_mps_bfloat16(self):
        torch = _deps()
        if not torch.backends.mps.is_available():
            pytest.skip("Apple MPS is required")
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj", "up_proj", "down_proj")).to(
            device="mps", dtype=torch.bfloat16
        )
        with torch.no_grad():
            model.mlp.gate_proj.get_base_layer().weight.fill_(2.5)
        assert patch_fast_lora_mlp(model) == 1
        x = torch.ones(3, 8, device="mps", dtype=torch.bfloat16, requires_grad=True)
        y = model(x)
        y.float().sum().backward()
        assert torch.isfinite(y).all()
        assert torch.isfinite(x.grad).all()


class TestFastPathAndScope:
    @pytest.mark.parametrize(
        ("targets", "ranks"),
        [
            pytest.param(("gate_proj",), None, id="gate"),
            pytest.param(("up_proj",), None, id="up"),
            pytest.param(("down_proj",), None, id="down"),
            pytest.param(("gate_proj", "down_proj"), None, id="gate-down"),
            pytest.param(
                ("gate_proj", "up_proj", "down_proj"), None, id="all-equal-rank"
            ),
            pytest.param(
                ("gate_proj", "up_proj", "down_proj"),
                {"gate_proj": 2, "up_proj": 3, "down_proj": 4},
                id="all-unequal-rank",
            ),
        ],
    )
    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
    @pytest.mark.parametrize("shape", [(4, 8), (2, 5, 8)], ids=["2d", "3d"])
    @pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"], ids=["fp32", "bf16"])
    def test_kernel_is_taken_across_targets_bias_rank_and_dtype(
        self, targets, ranks, bias, shape, dtype_name
    ):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        dtype = getattr(torch, dtype_name)
        model = _make_model(targets, bias=bias, ranks=ranks).to(dtype=dtype)
        x = torch.randn(*shape, dtype=dtype, requires_grad=True)
        before = model(x)
        assert type(before.grad_fn).__name__ != "_FastLoraSwiGLUBackward"

        assert patch_fast_lora_mlp(model) == 1
        out = model(x)
        assert type(out.grad_fn).__name__ == "_FastLoraSwiGLUBackward"

    def test_cpu_autocast_preserves_pefts_bfloat16_output(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        torch.manual_seed(29)
        model = _make_model(("gate_proj", "up_proj", "down_proj"))
        x = torch.randn(2, 5, 8, requires_grad=True)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference = model(x)
        assert reference.dtype == torch.bfloat16

        assert patch_fast_lora_mlp(model) == 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x)

        assert output.dtype == reference.dtype
        assert type(output.grad_fn).__name__ == "_FastLoraSwiGLUBackward"

    def test_single_projection_keeps_pefts_bfloat16_output_under_cpu_autocast(self):
        torch = _deps()
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora import patch_fast_lora_single_projection

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = nn.Linear(8, 8)

        torch.manual_seed(0)
        model = Model()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["o_proj"],
            ),
            model,
        )
        x = torch.randn(2, 5, 8, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reference = model.o_proj(x)
        assert reference.dtype == torch.bfloat16

        assert patch_fast_lora_single_projection(model) == 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model.o_proj(x)

        assert output.dtype == reference.dtype
        assert type(output.grad_fn).__name__ == "_FastLoraSingleProjectionBackward"

    def test_module_gelu_is_refused_even_when_parent_config_says_silu(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj",), act="silu", module_act="gelu")
        assert patch_fast_lora_mlp(model) == 0

    def test_missing_module_activation_fails_closed(self):
        _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        model = _make_model(("gate_proj",), act="silu")
        del model.mlp.act_fn
        assert patch_fast_lora_mlp(model) == 0

    def test_unadapted_int8_sibling_refuses_the_whole_mlp(self):
        torch = _deps()
        bnb = pytest.importorskip("bitsandbytes")
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(8, 12)
                self.up_proj = bnb.nn.Linear8bitLt(8, 12, has_fp16_weights=False)
                self.down_proj = bnb.nn.Linear8bitLt(12, 8, has_fp16_weights=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.mlp = TinyMLP()

            def forward(self, x):
                return self.mlp(x)

        model = Model().to("cpu")
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["gate_proj"],
            ),
            model,
        )
        x = torch.randn(2, 3, 8, requires_grad=True)
        reference = model(x)
        assert not model.mlp.up_proj.weight.is_floating_point()
        assert patch_fast_lora_mlp(model) == 1
        output = model(x)
        assert type(output.grad_fn).__name__ != "_FastLoraSwiGLUBackward"
        torch.testing.assert_close(output, reference)

    def test_moe_expert_path_is_out_of_scope(self):
        _deps()
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class Expert(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(8, 12)
                self.up_proj = nn.Linear(8, 12)
                self.down_proj = nn.Linear(12, 8)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.experts = nn.ModuleList([Expert()])

        model = Model()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        assert patch_fast_lora_mlp(model) == 0

    @pytest.mark.gpu
    @pytest.mark.parametrize("seed", [3, 7, 11])
    def test_nf4_forward_backward_and_all_adapter_grads_match_peft(self, seed):
        torch = _deps()
        bnb = pytest.importorskip("bitsandbytes")
        import torch.nn as nn
        from peft import LoraConfig, inject_adapter_in_model

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class TinyNF4MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = bnb.nn.Linear4bit(
                    8, 12, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.up_proj = bnb.nn.Linear4bit(
                    8, 12, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.down_proj = bnb.nn.Linear4bit(
                    12, 8, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_act="silu")
                self.mlp = TinyNF4MLP()

            def forward(self, x):
                return self.mlp(x)

        torch.manual_seed(seed)
        model = Model()
        with torch.no_grad():
            model.mlp.up_proj.weight.mul_(4)
        model = model.to("cuda")
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        _randomise_b(model)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert getattr(model.mlp, name).get_base_layer().weight.quant_state is not None

        reference_model = copy.deepcopy(model)
        x_ref = torch.randn(
            2, 3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        reference = reference_model(x_ref)
        assert type(reference.grad_fn).__name__ != "_FastLoraSwiGLUBackward"
        reference.float().square().mean().backward()
        reference_x_grad = x_ref.grad.detach().clone()
        reference_grads = _adapter_grads(reference_model)

        oracle_x = x_ref.detach().double().requires_grad_(True)
        oracle_adapters = {}

        def oracle_projection(layer, value, projection_name):
            base = layer.get_base_layer()
            dense = bnb.functional.dequantize_4bit(
                base.weight, base.weight.quant_state
            ).double()
            adapter = layer.active_adapters[0]
            lora_a = layer.lora_A[adapter].weight.detach().double().requires_grad_(True)
            lora_b = layer.lora_B[adapter].weight.detach().double().requires_grad_(True)
            oracle_adapters[
                f"mlp.{projection_name}.lora_A.{adapter}.weight"
            ] = lora_a
            oracle_adapters[
                f"mlp.{projection_name}.lora_B.{adapter}.weight"
            ] = lora_b
            base_out = torch.nn.functional.linear(value, dense)
            lora_out = torch.nn.functional.linear(
                torch.nn.functional.linear(value, lora_a), lora_b
            )
            return base_out + lora_out * float(layer.scaling[adapter])

        oracle_gate = oracle_projection(
            reference_model.mlp.gate_proj, oracle_x, "gate_proj"
        )
        oracle_up = oracle_projection(reference_model.mlp.up_proj, oracle_x, "up_proj")
        oracle_hidden = torch.nn.functional.silu(oracle_gate) * oracle_up
        oracle = oracle_projection(
            reference_model.mlp.down_proj, oracle_hidden, "down_proj"
        )
        oracle.square().mean().backward()
        oracle_x_grad = oracle_x.grad.detach()
        oracle_grads = {
            name: parameter.grad.detach()
            for name, parameter in oracle_adapters.items()
        }

        x = x_ref.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        out = model(x)
        assert type(out.grad_fn).__name__ == "_FastLoraSwiGLUBackward"
        out.float().square().mean().backward()

        def assert_within_twice_peft_error(actual, peft_value, oracle_value, name):
            actual_error = (actual.double() - oracle_value).abs().max().item()
            peft_error = (peft_value.double() - oracle_value).abs().max().item()
            assert actual_error <= 2.0 * peft_error + 1e-8, (
                f"{name}: kernel error {actual_error:.6g} exceeds twice "
                f"peft error {peft_error:.6g}"
            )

        assert_within_twice_peft_error(out, reference, oracle.detach(), "output")
        assert_within_twice_peft_error(
            x.grad, reference_x_grad, oracle_x_grad, "input gradient"
        )
        got_grads = _adapter_grads(model)
        assert got_grads.keys() == reference_grads.keys()
        for name, grad in got_grads.items():
            assert_within_twice_peft_error(
                grad, reference_grads[name], oracle_grads[name], name
            )


class TestStreamedModel:
    def test_mlp_kernel_matches_resident_peft_across_reused_stream_buffers(
        self, tmp_path
    ):
        torch = _deps()
        pytest.importorskip("transformers")
        pytest.importorskip("safetensors")
        from peft import LoraConfig, TaskType, get_peft_model
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        def lora_config():
            return LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM,
            )

        torch.manual_seed(41)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            max_position_embeddings=64,
        )
        weights = tmp_path / "model"
        weights.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                key: value.contiguous()
                for key, value in LlamaForCausalLM(config).state_dict().items()
            },
            str(weights / "model.safetensors"),
        )
        config.save_pretrained(str(weights))

        shards = str(tmp_path / "shards")
        index = shard_checkpoint(str(weights), shards, dtype="float32", arch="llama")
        streamed, _runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=shards,
            index=index,
            lora_config=lora_config(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=5,
        )
        assert patch_fast_lora_mlp(streamed) == 4
        _randomise_b(streamed)

        resident = get_peft_model(
            AutoModelForCausalLM.from_pretrained(str(weights), dtype=torch.float32),
            lora_config(),
        )
        resident_params = dict(resident.named_parameters())
        copied = 0
        for name, parameter in streamed.named_parameters():
            if "lora_" not in name or parameter.is_meta:
                continue
            target = next(
                (
                    candidate
                    for candidate_name, candidate in resident_params.items()
                    if candidate_name.endswith(name)
                ),
                None,
            )
            assert target is not None, f"resident model has no parameter ending in {name!r}"
            with torch.no_grad():
                target.copy_(parameter)
            copied += 1
        assert copied == 24

        input_ids = torch.randint(0, 64, (1, 8))
        labels = torch.randint(0, 64, (1, 8))
        seen = []
        hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, output: seen.append(type(output.grad_fn).__name__)
            )
            for module in streamed.modules()
            if getattr(module, "_soup_fast_lora_mlp", False)
        ]
        streamed_loss = streamed(input_ids=input_ids, labels=labels).loss
        for hook in hooks:
            hook.remove()
        assert seen == ["_FastLoraSwiGLUBackward"] * 4, seen
        streamed_loss.backward()
        resident_loss = resident(input_ids=input_ids, labels=labels).loss
        resident_loss.backward()

        torch.testing.assert_close(streamed_loss, resident_loss, rtol=1e-5, atol=1e-5)
        streamed_grads = {
            name: parameter.grad
            for name, parameter in streamed.named_parameters()
            if "lora_" in name and parameter.grad is not None
        }
        resident_grads = {
            name: parameter.grad
            for name, parameter in resident.named_parameters()
            if "lora_" in name and parameter.grad is not None
        }
        assert len(streamed_grads) == len(resident_grads) == 24
        for name, grad in streamed_grads.items():
            target = next(
                (
                    candidate
                    for candidate_name, candidate in resident_grads.items()
                    if candidate_name.endswith(name)
                ),
                None,
            )
            assert target is not None, f"no resident gradient for {name!r}"
            torch.testing.assert_close(grad, target, rtol=1e-4, atol=1e-5, msg=name)


class TestRealLlamaAndSavedBytes:
    def test_real_llama_mlp_matches_peft(self):
        torch = _deps()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = LlamaConfig(
                    hidden_size=8,
                    intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    hidden_act="silu",
                )
                self.mlp = LlamaMLP(self.config)

            def forward(self, x):
                return self.mlp(x)

        torch.manual_seed(31)
        model = Wrapper()
        inject_adapter_in_model(
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
            model,
        )
        _randomise_b(model)
        x0 = torch.randn(2, 5, 8, requires_grad=True)
        ref = model(x0)
        ref.square().mean().backward()
        ref_x = x0.grad.detach().clone()
        ref_grads = _adapter_grads(model)
        for param in model.parameters():
            param.grad = None

        x1 = x0.detach().clone().requires_grad_(True)
        assert patch_fast_lora_mlp(model) == 1
        got = model(x1)
        got.square().mean().backward()
        torch.testing.assert_close(got, ref)
        torch.testing.assert_close(x1.grad, ref_x)
        for name, grad in _adapter_grads(model).items():
            torch.testing.assert_close(grad, ref_grads[name], msg=name)

    def test_saved_tensor_bytes_are_lower_than_peft(self):
        torch = _deps()
        from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp

        def saved_bytes(model):
            seen = set()
            total = 0

            def pack(tensor):
                nonlocal total
                key = (
                    tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                )
                if key not in seen:
                    seen.add(key)
                    total += tensor.numel() * tensor.element_size()
                return tensor

            x = torch.randn(2, 64, 8, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                model(x).sum().backward()
            return total

        torch.manual_seed(0)
        model = _make_model(("gate_proj", "up_proj", "down_proj"))
        plain = saved_bytes(model)
        for param in model.parameters():
            param.grad = None
        assert patch_fast_lora_mlp(model) == 1
        fast = saved_bytes(model)

        assert 0 < fast < plain
        print(f"#837 saved bytes: peft={plain} fast={fast}")
