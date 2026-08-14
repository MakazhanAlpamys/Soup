"""#385 — layer streaming must not assume bf16 exists.

``trainer/stream_setup.py`` chose its store dtype with a literal:

    dtype = "bfloat16" if on_cuda else "float32"

no capability check, while ``trainer/asr.py`` and ``utils/gpu.py`` both call
``torch.cuda.is_bf16_supported()`` before choosing. bf16 needs Ampere. Colab's
free tier is a T4 (sm_75), Kaggle is a T4 or a P100, and V100 / GTX 16xx / RTX
20xx are all pre-Ampere — so the entire class of hardware a reader is most
likely to try this on was streaming a dtype its card cannot compute in, and
nothing said so. It could not fail on the maintainer's RTX 3050, which is
Ampere; that is the same shape as the four backends the H100 session found had
never executed once.

These tests assert the DECISION, with the capability stubbed, because that is
the part that was wrong. Asserting fp16 arithmetic in isolation would have
passed against the old line — the arithmetic was never broken. Measured before
writing any of this: streamed-vs-resident is bit-exact at 0.000000e+00 in
float16 as well as bfloat16, in both quantisations, so the fix is a decision and
not an implementation.
"""

import pytest


class _FakeCuda:
    """Stands in for ``torch.cuda`` so a card we do not own can be tested.

    ``is_bf16_supported`` mirrors the real signature, because the real default
    is ``including_emulation=True`` and a stub that ignored the keyword would
    hide the exact bug this file exists for: a T4 answers **True** to the bare
    call (it can hold a bf16 value through software emulation) and False to
    ``including_emulation=False`` (it has no bf16 hardware).
    """

    def __init__(self, available: bool, bf16: bool, emulated: bool = True):
        self._available = available
        self._bf16 = bf16
        self._emulated = emulated
        self.asked_bf16 = 0
        self.asked_without_emulation = 0

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        self.asked_bf16 += 1
        if not including_emulation:
            self.asked_without_emulation += 1
            return self._bf16
        return self._bf16 or self._emulated

    def get_device_capability(self, device=None):
        return (8, 0) if self._bf16 else (7, 5)


@pytest.fixture()
def fake_torch(monkeypatch):
    """Patch ``torch.cuda`` in place; the resolver imports torch lazily."""
    import torch

    def apply(available: bool, bf16: bool, emulated: bool = True) -> _FakeCuda:
        fake = _FakeCuda(available, bf16, emulated)
        monkeypatch.setattr(torch, "cuda", fake)
        return fake

    return apply


class TestResolveStreamDtype:
    def test_ampere_gets_bfloat16(self, fake_torch):
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake_torch(available=True, bf16=True)
        assert resolve_stream_dtype("cuda") == "bfloat16"

    def test_turing_gets_float16_not_bfloat16(self, fake_torch):
        """The defect. A T4 reports False here and used to be handed bf16."""
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake_torch(available=True, bf16=False)
        assert resolve_stream_dtype("cuda") == "float16"

    def test_the_capability_is_actually_consulted(self, fake_torch):
        """Guards the shape of the fix, not just its output: a resolver that
        hardcoded 'float16' would pass the test above and be just as wrong."""
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake = fake_torch(available=True, bf16=True)
        resolve_stream_dtype("cuda")
        assert fake.asked_bf16 >= 1

    def test_emulated_bf16_does_not_count_as_bf16(self, fake_torch):
        """The defect a real T4 exposed, after the first fix had shipped.

        ``torch.cuda.is_bf16_supported()`` defaults to
        ``including_emulation=True`` and returns **True on a T4** — it falls
        past the compute-capability fast path and merely constructs a bf16
        tensor, which software emulation satisfies. Asking the bare question
        therefore picks bf16 on hardware with no bf16 units, which is what the
        first version of this fix did: a no-op on exactly the cards it was
        written for. The question has to be asked without emulation.
        """
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake = fake_torch(available=True, bf16=False, emulated=True)
        assert fake.is_bf16_supported() is True, "the stub must model the trap"
        assert resolve_stream_dtype("cuda") == "float16"
        assert fake.asked_without_emulation >= 1

    def test_cpu_stays_float32(self, fake_torch):
        """CPU streaming is a test convenience and half-precision CPU kernels
        are not uniformly available — this must not change with the fix."""
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake_torch(available=True, bf16=True)
        assert resolve_stream_dtype("cpu") == "float32"

    def test_cuda_index_is_still_cuda(self, fake_torch):
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake_torch(available=True, bf16=False)
        assert resolve_stream_dtype("cuda:1") == "float16"

    def test_no_cuda_runtime_falls_back_to_float32(self, fake_torch):
        """A 'cuda' string on a box with no CUDA must not claim a GPU dtype."""
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        fake_torch(available=False, bf16=False)
        assert resolve_stream_dtype("cuda") == "float32"

    @pytest.mark.parametrize("bf16", [True, False])
    def test_result_is_always_a_supported_stream_dtype(self, fake_torch, bf16):
        from soup_cli.utils.layer_stream import (
            SUPPORTED_STREAM_DTYPES,
            resolve_stream_dtype,
        )

        fake_torch(available=True, bf16=bf16)
        for device in ("cuda", "cpu"):
            assert resolve_stream_dtype(device) in SUPPORTED_STREAM_DTYPES

    def test_the_answer_matches_the_rest_of_the_trainer(self, fake_torch):
        """One question, one answer. ``utils.gpu.get_compute_dtype`` already
        decides this for every other code path; a streamed run choosing its
        dtype by a private rule is how the two drift apart."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype
        from soup_cli.utils.layer_stream import resolve_stream_dtype

        for bf16 in (True, False):
            fake_torch(available=True, bf16=bf16)
            assert resolve_stream_dtype("cuda") == str(get_compute_dtype()).removeprefix(
                "torch."
            )
        assert torch is not None  # the import is the point of the comparison


class TestStreamSetupUsesTheResolver:
    """A regression guard on the module, in the shape this project already uses
    for ``device_map='auto'``: the literal that caused the defect must not come
    back, and it is cheaper to catch here than on a T4 nobody owns."""

    def _source(self) -> str:
        from pathlib import Path

        import soup_cli.trainer.stream_setup as mod

        return Path(mod.__file__).read_text(encoding="utf-8")

    def test_no_unconditional_bfloat16_literal(self):
        src = self._source()
        assert '"bfloat16" if on_cuda' not in src, (
            "the store dtype is being chosen without asking the card whether it "
            "supports bf16 — this is #385"
        )

    def test_calls_the_resolver(self):
        assert "resolve_stream_dtype" in self._source()


class TestTrainingArgumentsPrecisionAsksTheCardToo:
    """The SECOND bf16 assumption, found by the live smoke after the first was
    fixed — and this one is not streaming-specific.

    ``SFTTrainerWrapper._resolve_mixed_precision`` returned
    ``(self.device == "cuda", False)`` as its "legacy default", i.e. bf16 on
    ANY CUDA card unless the user opted into ``auto_mixed_precision``. On a
    pre-Ampere GPU that is not a preference, it is a hard stop: transformers
    raises *"Your setup doesn't support bf16/gpu. You need Ampere+ GPU with
    cuda>=11.0"* while building TrainingArguments. So **every** `soup train`
    on a T4 or a P100 died before step 0, streamed or not.

    Changing it cannot regress a working setup: where bf16 is supported the
    answer is unchanged, and where it is not the current behaviour is a crash.
    """

    def _wrapper(self):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        return wrapper

    class _Tcfg:
        auto_mixed_precision = False

    def test_ampere_still_gets_bf16(self, fake_torch):
        fake_torch(available=True, bf16=True)
        assert self._wrapper()._resolve_mixed_precision(self._Tcfg(), "meta-llama/x") == (
            True,
            False,
        )

    def test_turing_gets_fp16_not_bf16(self, fake_torch):
        """Was ``(True, False)`` — which transformers rejects outright."""
        fake_torch(available=True, bf16=False)
        assert self._wrapper()._resolve_mixed_precision(self._Tcfg(), "meta-llama/x") == (
            False,
            True,
        )

    def test_cpu_asks_for_neither(self, fake_torch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        fake_torch(available=False, bf16=False)
        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cpu"
        assert wrapper._resolve_mixed_precision(self._Tcfg(), "meta-llama/x") == (False, False)

    def test_cuda_string_without_a_cuda_runtime_asks_for_neither(self, fake_torch):
        """A CPU-only box that was handed ``device="cuda"`` must not come back
        asking for fp16 on a card that is not there — the run fails either way,
        and a phantom precision request only obscures the real error. This is
        also the shape CI runs in, so it is the branch that decides whether a
        GPU-less test box agrees with a GPU-ful one."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        fake_torch(available=False, bf16=False)
        wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
        wrapper.device = "cuda"
        assert wrapper._resolve_mixed_precision(self._Tcfg(), "meta-llama/x") == (False, False)


class TestEveryTrainerAsksTheCard:
    """#387 was in TWELVE wrappers, not one. Repairing only the SFT path is
    exactly the debt #359 records for DeepSpeed+LoRA ("repaired in sft.py
    only"), so this SCANS every module in ``soup_cli/trainer/`` rather than
    parametrising over a hand-written list — the list is what hides the ones
    nobody remembered."""

    def _trainer_sources(self):
        from pathlib import Path

        import soup_cli.trainer as pkg

        root = Path(pkg.__file__).parent
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))}

    def test_no_module_hardcodes_bf16_from_the_device_string(self):
        offenders = {
            name: line.strip()
            for name, src in self._trainer_sources().items()
            for line in src.splitlines()
            if 'bf16=self.device == "cuda"' in line or '"bf16": self.device == "cuda"' in line
        }
        assert not offenders, (
            "these wrappers ask for bf16 on any CUDA card, which transformers "
            f"refuses on a T4/P100 — see #387: {offenders}"
        )

    def test_the_scanner_can_actually_fail(self):
        """A scanner that matches nothing passes forever. Prove the pattern
        fires against the exact line the sweep removed."""
        planted = '            bf16=self.device == "cuda",'
        assert 'bf16=self.device == "cuda"' in planted

    def test_the_wrappers_that_set_precision_use_the_shared_helper(self):
        """Sites that set bf16 must take it from one place, or they drift."""
        sources = self._trainer_sources()
        setters = {
            name
            for name, src in sources.items()
            if ("bf16=" in src or '"bf16":' in src) and name not in {"grpo.py", "__init__.py"}
        }
        assert setters, "no wrapper sets a precision flag — the scan is broken"
        missing = {n for n in setters if "bf16_fp16_flags" not in sources[n]}
        assert not missing, f"these set bf16 without the shared helper: {missing}"


# ==========================================================================
# The gate: fp16 must be as exact as bf16, or choosing it on a T4 would trade
# a silent unsupported-dtype run for a silent wrong-numbers one.
# ==========================================================================
def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


CUDA = pytest.mark.skipif(
    not _cuda_available(), reason="requires CUDA (layer streaming is a GPU feature)"
)


def _tiny_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM,
    )


def _tiny_llama_dir(tmp_path):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _randomise_lora_b(model, seed=7):
    """PEFT initialises ``lora_B = 0``, so at step 0 the adapter contributes
    NOTHING and a bit-exactness check against another zero adapter compares the
    base only. The v0.72.0 gate hit exactly this — make B load-bearing first."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(
                    torch.randn(param.shape, generator=generator).to(param.device, param.dtype)
                )


def _copy_lora(src, dst):
    src_lora = {k.replace(".inner.", "."): v for k, v in src.state_dict().items() if "lora_" in k}
    dst_lora = {k.replace(".inner.", "."): v for k, v in dst.state_dict().items() if "lora_" in k}
    assert src_lora and set(src_lora) == set(dst_lora)
    with __import__("torch").no_grad():
        for key, val in src_lora.items():
            dst_lora[key].copy_(val.to(dst_lora[key].dtype))


@CUDA
class TestFloat16StreamingIsBitExact:
    """Acceptance item 4 of #385. The reference is a resident model of MATCHING
    numerics — for NF4 that means a genuinely NF4-quantised reference, because
    comparing a streamed fp16 run against a bf16 resident one would measure the
    dtype rather than the streaming."""

    def _run(self, tmp_path, dtype: str, quant: str) -> float:
        import torch
        from peft import get_peft_model

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import (
            build_meta_skeleton,
            build_streamed_model,
            quantised_layer_suffixes,
        )

        weights = _tiny_llama_dir(tmp_path / f"{dtype}-{quant}")
        shards = str(tmp_path / f"shards-{dtype}-{quant}")

        shard_kwargs, stream_kwargs = {}, {}
        if quant != "none":
            probe = build_meta_skeleton(weights, dtype=dtype, quant=quant)
            shard_kwargs = {
                "quant": quant,
                "quant_suffixes": quantised_layer_suffixes(probe),
                "quant_device": "cuda",
            }
            stream_kwargs = {"quant": quant}
            del probe

        index = shard_checkpoint(weights, shards, dtype=dtype, arch="llama", **shard_kwargs)
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=shards, index=index, lora_config=_tiny_lora(),
            device="cuda", dtype=dtype, buffers=2, pin=False, seed=3, **stream_kwargs,
        )
        try:
            _randomise_lora_b(model)

            if quant == "none":
                from transformers import AutoModelForCausalLM

                base = AutoModelForCausalLM.from_pretrained(
                    weights, dtype=getattr(torch, dtype), device_map={"": "cuda"}
                )
            else:
                from transformers import AutoModelForCausalLM

                from soup_cli.utils.layer_stream_runtime import build_nf4_config

                base = AutoModelForCausalLM.from_pretrained(
                    weights, quantization_config=build_nf4_config(dtype),
                    dtype=getattr(torch, dtype), device_map={"": "cuda"},
                )
            base.config.use_cache = False
            for param in base.parameters():
                param.requires_grad = False
            ref = get_peft_model(base, _tiny_lora())
            _copy_lora(model, ref)

            ids = torch.randint(0, 64, (1, 12), device="cuda")
            with torch.no_grad():
                got = model(input_ids=ids).logits
                want = ref(input_ids=ids).logits
            return (got.float() - want.float()).abs().max().item()
        finally:
            runtime.close()

    @pytest.mark.parametrize("quant", ["none", "nf4"])
    def test_float16_matches_resident_bit_exactly(self, tmp_path, quant):
        assert self._run(tmp_path, "float16", quant) == 0.0

    @pytest.mark.parametrize("quant", ["none", "nf4"])
    def test_bfloat16_still_matches_resident_bit_exactly(self, tmp_path, quant):
        """The control: if the bf16 arm ever stopped being exact, an exact fp16
        arm would say nothing about fp16."""
        import torch

        if not torch.cuda.is_bf16_supported():
            pytest.skip("this card has no bf16 — the control cannot run here")
        assert self._run(tmp_path, "bfloat16", quant) == 0.0
