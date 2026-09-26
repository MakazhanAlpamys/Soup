"""#350 — make FSDP + BNB 4-bit a runnable configuration.

The shipped 70B recipe reached FSDP with uint8 quantized storage and fp32 LoRA
parameters. FSDP cannot flatten either the integer storage or the mixed
bf16/fp32 unit, so both decisions are tested here without requiring eight GPUs.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


class _FakeData:
    def __init__(self, dtype: object) -> None:
        self.dtype = dtype
        self.casts: list[object] = []

    def to(self, dtype: object) -> "_FakeData":
        self.dtype = dtype
        self.casts.append(dtype)
        return self


class _FakeParam:
    def __init__(
        self,
        dtype: object,
        *,
        requires_grad: bool,
        floating: bool = True,
    ) -> None:
        self.requires_grad = requires_grad
        self._floating = floating
        self.data = _FakeData(dtype)

    @property
    def dtype(self) -> object:
        return self.data.dtype

    def is_floating_point(self) -> bool:
        return self._floating


class _FakeModel:
    def __init__(self, **params: _FakeParam) -> None:
        self._params = params

    def named_parameters(self):
        return self._params.items()


def test_fsdp_4bit_resolves_storage_to_compute_dtype() -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    original = TrainingConfig(quantization="4bit")
    resolved = resolve_fsdp_qlora_quant_storage(
        original,
        fsdp=True,
        compute_dtype="bfloat16",
    )

    assert original.bnb_4bit_quant_storage is None
    assert resolved.bnb_4bit_quant_storage == "bfloat16"


def test_runtime_resolution_asks_the_shared_compute_dtype_probe(monkeypatch) -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils import gpu
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    monkeypatch.setattr(gpu, "get_compute_dtype", lambda: "bfloat16")
    resolved = resolve_fsdp_qlora_quant_storage(
        TrainingConfig(quantization="4bit"),
        fsdp=True,
    )

    assert resolved.bnb_4bit_quant_storage == "bfloat16"
    assert resolve_fsdp_qlora_quant_storage(
        resolved,
        fsdp=True,
        compute_dtype="bfloat16",
    ) is resolved


def test_resolved_storage_reaches_bitsandbytes_config(monkeypatch) -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils import gpu
    from soup_cli.utils.quant_menu import (
        build_quantization_config_for_loader,
        resolve_fsdp_qlora_quant_storage,
    )

    bf16 = object()
    fake_torch = ModuleType("torch")
    fake_torch.uint8 = object()  # type: ignore[attr-defined]
    fake_torch.float16 = object()  # type: ignore[attr-defined]
    fake_torch.bfloat16 = bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    fake_transformers = ModuleType("transformers")
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(gpu, "get_compute_dtype", lambda: bf16)

    resolved = resolve_fsdp_qlora_quant_storage(
        TrainingConfig(quantization="4bit"),
        fsdp=True,
        compute_dtype="bfloat16",
    )
    quant_config = build_quantization_config_for_loader(tcfg=resolved, base="m")

    assert quant_config.bnb_4bit_quant_storage is bf16


def test_fsdp_4bit_overrides_integer_storage() -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    original = TrainingConfig(
        quantization="4bit",
        bnb_4bit_quant_storage="uint8",
    )
    resolved = resolve_fsdp_qlora_quant_storage(
        original,
        fsdp=True,
        compute_dtype="float16",
    )

    assert resolved.bnb_4bit_quant_storage == "float16"


def test_storage_resolution_is_independent_of_fsdp_compile() -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    for compile_enabled in (False, True):
        original = TrainingConfig(
            quantization="4bit",
            use_fsdp2_compile=compile_enabled,
        )
        resolved = resolve_fsdp_qlora_quant_storage(
            original,
            fsdp=True,
            compute_dtype="bfloat16",
        )
        assert resolved.use_fsdp2_compile is compile_enabled
        assert resolved.bnb_4bit_quant_storage == "bfloat16"


def test_non_fsdp_or_non_4bit_config_is_unchanged() -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    four_bit = TrainingConfig(quantization="4bit")
    eight_bit = TrainingConfig(quantization="8bit")

    assert resolve_fsdp_qlora_quant_storage(
        four_bit,
        fsdp=False,
        compute_dtype="bfloat16",
    ) is four_bit
    assert resolve_fsdp_qlora_quant_storage(
        eight_bit,
        fsdp=True,
        compute_dtype="bfloat16",
    ) is eight_bit


def test_fsdp_4bit_rejects_non_floating_compute_storage() -> None:
    from soup_cli.config.schema import TrainingConfig
    from soup_cli.utils.quant_menu import resolve_fsdp_qlora_quant_storage

    tcfg = TrainingConfig(quantization="4bit")

    try:
        resolve_fsdp_qlora_quant_storage(tcfg, fsdp=True, compute_dtype="uint8")
    except ValueError as exc:
        assert "floating compute dtype" in str(exc)
    else:  # pragma: no cover - the assertion above is the intended path
        raise AssertionError("integer FSDP quant storage was accepted")


def test_fsdp_qlora_casts_every_trainable_float_to_compute_dtype() -> None:
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fsdp_qlora

    model = _FakeModel(
        lora_A=_FakeParam("float32", requires_grad=True),
        lora_B=_FakeParam("float32", requires_grad=True),
        modules_to_save_head=_FakeParam("float16", requires_grad=True),
        frozen_base=_FakeParam("float32", requires_grad=False),
        packed_4bit=_FakeParam("uint8", requires_grad=False, floating=False),
    )

    casted = align_trainable_dtype_for_fsdp_qlora(
        model,
        fsdp=True,
        quantization="4bit",
        compute_dtype="bfloat16",
    )

    assert casted == 3
    assert all(
        param.dtype == "bfloat16"
        for param in model._params.values()
        if param.requires_grad and param.is_floating_point()
    )
    assert model._params["frozen_base"].dtype == "float32"
    assert model._params["packed_4bit"].dtype == "uint8"


def test_fsdp_qlora_alignment_is_gated_to_the_exact_combination() -> None:
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fsdp_qlora

    for fsdp, quantization in ((False, "4bit"), (True, "8bit"), (False, "8bit")):
        param = _FakeParam("float32", requires_grad=True)
        model = _FakeModel(lora_A=param)
        assert align_trainable_dtype_for_fsdp_qlora(
            model,
            fsdp=fsdp,
            quantization=quantization,
            compute_dtype="bfloat16",
        ) == 0
        assert param.dtype == "float32"


def test_train_command_wires_both_fsdp_qlora_guards() -> None:
    """A tested helper that the command never calls would leave #350 live."""
    import inspect

    from soup_cli.commands import train

    source = inspect.getsource(train.train)
    assert "resolve_fsdp_qlora_quant_storage" in source
    assert "align_trainable_dtype_for_fsdp_qlora" in source
    assert source.index("resolve_fsdp_qlora_quant_storage") < source.index(
        "trainer_wrapper.setup(dataset)"
    )
    assert source.index("trainer_wrapper.setup(dataset)") < source.index(
        "align_trainable_dtype_for_fsdp_qlora"
    )


def test_llama3_70b_fsdp2_recipe_pins_bf16_storage() -> None:
    import yaml

    from soup_cli.recipes.catalog import RECIPES

    recipe = yaml.safe_load(RECIPES["llama3-70b-fsdp2"].yaml_str)
    assert recipe["training"]["quantization"] == "4bit"
    assert recipe["training"]["bnb_4bit_quant_storage"] == "bfloat16"
    assert recipe["training"]["use_fsdp2_compile"] is True


def test_alignment_accepts_wrapper_model_shape() -> None:
    """Document the exact command-boundary object shape used after setup()."""
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fsdp_qlora

    wrapper = SimpleNamespace(
        model=_FakeModel(lora_A=_FakeParam("float32", requires_grad=True))
    )
    assert align_trainable_dtype_for_fsdp_qlora(
        wrapper.model,
        fsdp=True,
        quantization="4bit",
        compute_dtype="bfloat16",
    ) == 1
