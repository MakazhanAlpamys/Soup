"""#1266: MoLE trained its router gate in bf16 on CUDA with no fp32 master copy.

``task: moe_lora_routing`` freezes the base and every task LoRA and trains one
tensor, the router gate ``nn.Linear(hidden, N, bias=False)``. When the device
string was exactly ``"cuda"``, ``setup()`` cast that gate to bfloat16
(``gate.to("cuda", dtype=torch.bfloat16)``). ``train()`` runs no autocast, and
nothing kept an fp32 copy for the optimizer, so the gradient and both AdamW
moments were bf16 as well:

* an AdamW step of about ``lr`` is below half a bf16 ulp for every
  ``|w| >= 2**-7`` at the default ``lr=2e-5``, and the gate's ``nn.Linear`` init
  draws from U(-1/sqrt(hidden), 1/sqrt(hidden)). After 8 steps 91.4% of the gate
  was bit-identical at hidden 64 and 82.6% at hidden 576;
* ``"cuda:0"`` kept the gate fp32, because the check was ``== "cuda"``;
* under the pre-Ampere flags open PR #868 adds (``fp16=True``) the fp16
  GradScaler would receive a bf16 gradient, which its CUDA unscale kernel
  rejects (#425), and ``align_trainable_dtype_for_fp16`` casts only ``lora_``
  parameters, so it never reached the gate.

The dtype decision reads the device string, so ``MoleRoutingTrainerWrapper(cfg,
device="cuda")`` on a CPU-only box builds what a GPU run builds. The one call of
that branch that needs a card on ``main`` is the gate's placement; the fixture
below rewrites the DEVICE of a CUDA ``nn.Module.to`` call to ``"cpu"`` and keeps
its dtype, so the unfixed code fails an assertion here rather than failing to
find a GPU. Training runs through ``use_cpu=True``. Nothing here touches a GPU.
"""

from __future__ import annotations

import contextlib
import json
from types import MethodType, SimpleNamespace

import pytest
import yaml

ROWS = [
    {"text": "we add two and two so the answer is four"},
    {"text": "we add two and three then check it is five"},
    {"text": "step one we add one and three so the answer is four"},
    {"text": "three plus two is five so the answer is five"},
]
_WORDS = (
    "two", "plus", "is", "four", "five", "so", "the", "answer", "step", "one", "three",
    "we", "add", "then", "check", "it", "and",
)
_MAX_LENGTH = 64
_DEFAULT_LR = 2.0e-5  # the schema default; the docs/training.md MoLE example sets no lr

# ``h64`` is the shape of the #1235 fixture. ``h576`` is the hidden size of
# SmolLM2-135M, the base in the docs/training.md MoLE example (9 heads, 3 KV
# heads); everything else stays tiny. The share of the gate that a bf16 step
# cannot move follows the init bound 1/sqrt(hidden), so both are pinned.
_GEOMETRIES = {
    "h64": {
        "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 4,
        "intermediate_size": 128,
    },
    "h576": {
        "hidden_size": 576, "num_attention_heads": 9, "num_key_value_heads": 3,
        "intermediate_size": 256,
    },
}


def _build_base(model_dir, geometry):
    """A tiny random-init Llama and a word-level tokenizer, saved in bf16 like most
    published checkpoints."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=64, num_hidden_layers=2, max_position_embeddings=128,
        **_GEOMETRIES[geometry],
    )).to(torch.bfloat16).save_pretrained(str(model_dir))
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in _WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
        "model_max_length": 128,
    }), encoding="utf-8")
    return str(model_dir)


def _build_adapters(base_dir, root, count=2):
    """Task LoRAs with a RANDOM ``lora_B``, so each adapter produces different logits.

    peft initialises ``lora_B`` to zero, and then every adapter equals the base and
    the gate gets no gradient in any precision.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    paths = []
    for index in range(count):
        out = root / f"task_adapter_{index}"
        torch.manual_seed(100 + index)
        base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
        peft_model = get_peft_model(base, LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        ))
        with torch.no_grad():
            for name, param in peft_model.named_parameters():
                if "lora_B" in name:
                    param.normal_(0.0, 0.5)
        peft_model.save_pretrained(str(out))
        paths.append(out.as_posix())
    return paths


@pytest.fixture(scope="module")
def mole_inputs(tmp_path_factory):
    """``mole_inputs(geometry)`` -> ``(base_dir, [adapter dirs])``, built once each."""
    built: dict[str, tuple[str, list[str]]] = {}

    def get(geometry="h64"):
        if geometry not in built:
            root = tmp_path_factory.mktemp(f"mole1266-{geometry}")
            base = _build_base(root / "tiny-llama", geometry)
            built[geometry] = (base, _build_adapters(base, root))
        return built[geometry]

    return get


def _cfg(base, adapters, out_dir, *, lr=None):
    """The docs/training.md MoLE example with the base and adapters swapped for the
    tiny ones: 4 rows, batch 1, 2 epochs = 8 optimizer steps."""
    from soup_cli.config.loader import load_config_from_string

    training = {
        "mole_task_adapters": adapters, "epochs": 2, "batch_size": 1,
        "gradient_accumulation_steps": 1, "logging_steps": 1, "save_steps": 10_000,
        "seed": 0,
    }
    if lr is not None:
        training["lr"] = lr
    return load_config_from_string(yaml.safe_dump({
        "base": base, "task": "moe_lora_routing",
        "data": {"train": "unused.jsonl", "max_length": _MAX_LENGTH},
        "training": training,
        "output": out_dir.as_posix(),
    }))


@contextlib.contextmanager
def _cuda_placement_on_cpu():
    """Send a CUDA ``nn.Module.to`` call to the CPU, keeping its dtype argument.

    ``main`` placed the gate with ``gate.to("cuda", dtype=torch.bfloat16)``, the one
    line of the CUDA branch that needs a card: ``setup()`` decides everything else
    from the device string alone.
    """
    import torch

    real_to = torch.nn.Module.to

    def to_cpu_instead_of_cuda(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            args = ("cpu",) + args[1:]
        if str(kwargs.get("device", "")).startswith("cuda"):
            kwargs = {**kwargs, "device": "cpu"}
        return real_to(self, *args, **kwargs)

    torch.nn.Module.to = to_cpu_instead_of_cuda
    try:
        yield
    finally:
        torch.nn.Module.to = real_to


def _setup(base, adapters, out_dir, *, device="cuda", lr=None):
    from soup_cli.trainer.mole_routing import MoleRoutingTrainerWrapper

    wrapper = MoleRoutingTrainerWrapper(_cfg(base, adapters, out_dir, lr=lr), device=device)
    with _cuda_placement_on_cpu():
        wrapper.setup({"train": ROWS})
    return wrapper


def _gate_weight(wrapper):
    return wrapper.model.mole_gate.gate.weight


def _frozen_dtypes(wrapper):
    """The dtypes of the frozen base and of the task LoRAs, separately."""
    base, lora = set(), set()
    for name, param in wrapper.model.named_parameters():
        if name.startswith("mole_gate."):
            continue
        (lora if "lora_" in name else base).add(param.dtype)
    return base, lora


def _unchanged_fraction(before, after):
    same = int((before == after.detach().to(before.dtype)).sum())
    return same / before.numel(), same, before.numel()


@pytest.fixture
def cpu_run(monkeypatch):
    """Keep every run on the CPU, and hand ``train()`` the precision flags a card gets.

    ``train()`` on ``main`` passes neither ``bf16`` nor ``fp16``; open PR #868 adds them
    from ``bf16_fp16_flags``. ``cpu_run(bf16=True)`` is what an Ampere+ card will get,
    ``cpu_run(fp16=True)`` a pre-Ampere one; by default both are off, as today.
    ``use_cpu=True`` also keeps a dev box that has a card from training on it, and
    the tests assert ``args.use_cpu`` so a patch that did not take cannot pass
    silently. A factory rather than a subclass, so the arguments object stays a
    real ``TrainingArguments``.
    """
    import sys

    import transformers

    # The first lazy attribute access can replace sys.modules["transformers"] with a
    # new module object, and train()'s ``from transformers import ...`` reads that
    # one. Resolve first, then patch the object the import will see.
    real = transformers.TrainingArguments
    module = sys.modules["transformers"]
    chosen = {"bf16": False, "fp16": False}

    def cpu_training_arguments(*args, **kwargs):
        return real(*args, **{**kwargs, "use_cpu": True, **chosen})

    monkeypatch.setattr(module, "TrainingArguments", cpu_training_arguments)

    def apply(*, bf16=False, fp16=False):
        chosen.update(bf16=bf16, fp16=fp16)

    return apply


@pytest.fixture
def optimizer_probe(monkeypatch):
    """Record the trainable dtypes when the optimizer is built, and per optimizer step
    the dtype of the gate's gradient and of its two AdamW moments."""
    import transformers

    seen: dict = {"trainable": {}, "grad": [], "moments": []}
    real_create = transformers.Trainer.create_optimizer

    def spy(self, *args, **kwargs):
        seen["trainable"].update(
            {n: p.dtype for n, p in self.model.named_parameters() if p.requires_grad}
        )
        optimizer = real_create(self, *args, **kwargs)
        weight = self.model.mole_gate.gate.weight

        def before_step(opt, _args, _kwargs):
            seen["grad"].append(None if weight.grad is None else weight.grad.dtype)

        def after_step(opt, _args, _kwargs):
            state = opt.state[weight]
            seen["moments"].append({key: state[key].dtype for key in ("exp_avg", "exp_avg_sq")})

        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)
        return optimizer

    monkeypatch.setattr(transformers.Trainer, "create_optimizer", spy)
    return seen


class TestTheGateIsAnFp32MasterWeight:
    """The gate is created as an fp32 parameter whatever the device string says."""

    def test_the_gate_is_fp32_after_setup_on_cuda(self, mole_inputs, tmp_path):
        """On ``main`` the CUDA branch left it bf16: 128 of 128 elements at hidden 64."""
        import torch

        wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")

        trainable = {
            name: p.dtype for name, p in wrapper.model.named_parameters() if p.requires_grad
        }
        assert list(trainable) == ["mole_gate.gate.weight"], trainable
        assert trainable["mole_gate.gate.weight"] == torch.float32, trainable

    def test_cuda_and_cuda0_take_the_same_decision(self, mole_inputs, tmp_path):
        """``soup train`` passes ``"cuda"``; a direct caller may pass ``"cuda:0"``. ``main``
        compared the string with ``==``: bf16 for one, fp32 for the other."""
        import torch

        dtypes = {
            device: _gate_weight(
                _setup(*mole_inputs("h64"), tmp_path / str(index), device=device)
            ).dtype
            for index, device in enumerate(("cuda", "cuda:0"))
        }
        assert dtypes == {"cuda": torch.float32, "cuda:0": torch.float32}, dtypes

    def test_the_gate_is_fp32_whatever_the_default_dtype(self, mole_inputs, tmp_path):
        """``nn.Linear`` takes torch's process-wide default dtype, which transformers
        switches while it builds a model and remote code can leave switched. The gate is
        fp32 because it is created that way, not because the default happens to be."""
        import torch

        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")
        finally:
            torch.set_default_dtype(previous)

        assert _gate_weight(wrapper).dtype == torch.float32

    @pytest.mark.parametrize(
        "device, base_dtype", [("cuda", "bfloat16"), ("cpu", "float32")], ids=["cuda", "cpu"]
    )
    def test_the_frozen_base_and_task_loras_stay_as_they_were(
        self, mole_inputs, tmp_path, device, base_dtype
    ):
        """Control: only the gate's dtype changes. The frozen base still loads bf16 on
        ``"cuda"`` (it never takes an optimizer step) and peft keeps the task LoRAs in
        fp32; none of them trains."""
        import torch

        wrapper = _setup(*mole_inputs("h64"), tmp_path, device=device)

        base, lora = _frozen_dtypes(wrapper)
        assert base == {getattr(torch, base_dtype)}, base
        assert lora == {torch.float32}, lora
        trainable = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
        assert trainable == ["mole_gate.gate.weight"], trainable


class TestEightStepsAtTheDefaultLr:
    """The issue's run on the CUDA branch: 4 rows, batch 1, 2 epochs, ``seed: 0``."""

    @pytest.mark.parametrize("lr", [None, 1.0e-4], ids=["default-lr", "lr-1e-4"])
    @pytest.mark.parametrize("geometry", ["h64", "h576"])
    def test_at_most_one_percent_of_the_gate_is_bit_identical(
        self, mole_inputs, tmp_path, cpu_run, geometry, lr
    ):
        """``main`` left 117/128 (91.4%) at h64 and 951/1152 (82.6%) at h576 bit-identical
        at the default lr, and 333/1152 (28.9%) at h576 with ``lr=1e-4``."""
        import torch

        wrapper = _setup(*mole_inputs(geometry), tmp_path, device="cuda", lr=lr)
        before = _gate_weight(wrapper).detach().clone()

        wrapper.train()

        trainer = wrapper.trainer
        assert trainer.args.use_cpu
        assert trainer.state.global_step == 8
        assert trainer.args.learning_rate == pytest.approx(_DEFAULT_LR if lr is None else lr)
        after = _gate_weight(wrapper)
        # Not 0.0: an element whose gradient is exactly zero on every step stays put
        # even on an fp32 master weight, and the CPU bf16 kernels decide which ones
        # (#1250's CI on Windows CPython 3.10/3.11 left 0.3% of a group unmoved where
        # 3.12 left none). The defect left 82-91% bit-identical, so 1% separates them.
        fraction, same, total = _unchanged_fraction(before, after)
        assert fraction <= 0.01, (
            f"{same}/{total} gate elements bit-identical after 8 steps at lr={lr}"
        )
        assert after.dtype == torch.float32, "the gate did not stay an fp32 master weight"

    def test_its_gradient_and_adamw_moments_are_fp32_on_every_step(
        self, mole_inputs, tmp_path, cpu_run, optimizer_probe
    ):
        """AdamW keeps its moments in the parameter's dtype, so a bf16 gate got bf16
        moments too: there was no fp32 copy anywhere."""
        import torch

        wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")

        wrapper.train()

        assert wrapper.trainer.args.use_cpu
        assert optimizer_probe["trainable"] == {"mole_gate.gate.weight": torch.float32}
        assert optimizer_probe["grad"] == [torch.float32] * 8, optimizer_probe["grad"]
        moments = optimizer_probe["moments"]
        assert len(moments) == 8, moments
        assert all(
            m == {"exp_avg": torch.float32, "exp_avg_sq": torch.float32} for m in moments
        ), moments

    @pytest.mark.parametrize("geometry", ["h64", "h576"])
    def test_the_cpu_branch_is_unchanged(self, mole_inputs, tmp_path, cpu_run, geometry):
        """Control: ``device="cpu"`` loaded everything fp32 and trained the whole gate
        before this fix too."""
        import torch

        wrapper = _setup(*mole_inputs(geometry), tmp_path, device="cpu")
        assert _frozen_dtypes(wrapper) == ({torch.float32}, {torch.float32})
        before = _gate_weight(wrapper).detach().clone()

        wrapper.train()

        assert wrapper.trainer.args.use_cpu
        assert wrapper.trainer.state.global_step == 8
        assert _gate_weight(wrapper).dtype == torch.float32
        fraction, same, total = _unchanged_fraction(before, _gate_weight(wrapper))
        assert fraction <= 0.01, f"{same}/{total} gate elements bit-identical"


class TestUnderTheFlagsACardGets:
    """``(bf16=True, fp16=False)`` on Ampere+ and ``(bf16=False, fp16=True)`` on a
    T4/V100/P100, the flags open PR #868 passes. Injected here, since ``train()`` on
    ``main`` passes neither."""

    @pytest.mark.parametrize(
        "bf16, fp16", [(True, False), (False, True)], ids=["ampere-bf16", "pre-ampere-fp16"]
    )
    def test_every_trainable_parameter_is_fp32_when_the_optimizer_is_built(
        self, mole_inputs, tmp_path, cpu_run, optimizer_probe, bf16, fp16
    ):
        """``align_trainable_dtype_for_fp16`` casts only ``lora_`` parameters and the gate
        has none, so the gate has to be fp32 by itself for the fp16 GradScaler."""
        import torch

        cpu_run(bf16=bf16, fp16=fp16)
        wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")

        wrapper.train()

        args = wrapper.trainer.args
        assert args.use_cpu
        assert (args.bf16, args.fp16) == (bf16, fp16)
        assert optimizer_probe["trainable"] == {"mole_gate.gate.weight": torch.float32}, (
            optimizer_probe["trainable"]
        )

    def test_bf16_autocast_trains_the_gate(self, mole_inputs, tmp_path, cpu_run):
        """Accelerate autocasts ``model.forward`` and converts its outputs to fp32; the
        gate then runs on those outputs outside the forward. At the default lr the fp32
        gate moves; a bf16 one (``main``) did not."""
        import torch

        cpu_run(bf16=True)
        wrapper = _setup(*mole_inputs("h576"), tmp_path, device="cuda")
        before = _gate_weight(wrapper).detach().clone()

        wrapper.train()

        trainer = wrapper.trainer
        assert trainer.args.use_cpu
        assert trainer.args.bf16 and trainer.accelerator.native_amp, "autocast was not on"
        assert trainer.state.global_step == 8
        fraction, same, total = _unchanged_fraction(before, _gate_weight(wrapper))
        assert fraction <= 0.01, f"{same}/{total} gate elements bit-identical"
        assert _gate_weight(wrapper).dtype == torch.float32

    def test_an_fp16_autocast_grad_scaler_step_trains_the_gate(self, mole_inputs, tmp_path):
        """An optimizer step the way Accelerate runs fp16 on a CUDA card: autocast only
        ``model.forward``, convert its outputs to fp32, and scale the loss. CPU autocast
        and the CPU GradScaler stand in for the CUDA ones; CUDA's unscale kernel has no
        bf16 implementation (#425) and the CPU one does, so the gradient dtype it would
        reject is asserted directly."""
        import torch
        from accelerate.utils import convert_outputs_to_fp32
        from transformers import Trainer

        from soup_cli.trainer.mole_routing import (
            _build_collator,
            _prepare_mole_dataset,
            make_mole_trainer_class,
        )

        wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")
        model, tokenizer = wrapper.model, wrapper.tokenizer
        autocast = torch.autocast(device_type="cpu", dtype=torch.float16)
        model.forward = MethodType(
            convert_outputs_to_fp32(autocast(model.forward.__func__)), model
        )
        batch = _build_collator(tokenizer)(_prepare_mole_dataset(ROWS, tokenizer, _MAX_LENGTH))
        weight = model.mole_gate.gate.weight
        before = weight.detach().clone()
        optimizer = torch.optim.AdamW([weight], lr=_DEFAULT_LR)
        scaler = torch.amp.GradScaler("cpu")
        compute_loss = make_mole_trainer_class(Trainer).compute_loss

        taken_at = None
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(SimpleNamespace(), model, batch)
            scaler.scale(loss).backward()
            assert weight.grad is not None, "the loss did not reach the gate"
            assert weight.grad.dtype == torch.float32, weight.grad.dtype
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale:  # no inf/nan found: the step was taken
                taken_at = attempt
                break
        assert taken_at is not None, "the GradScaler skipped every step"
        fraction, same, total = _unchanged_fraction(before, weight)
        assert fraction <= 0.01, f"{same}/{total} gate elements bit-identical after one step"


class TestTheRoutingCastsBothWays:
    """The frozen base can hand the gate a dtype other than its own. ``compute_loss``
    casts the router input to the gate's dtype, and the routing weights back to the
    logits' dtype for the blend; the gradient goes back through both casts."""

    @pytest.mark.parametrize(
        "base_dtype, gate_dtype",
        [
            ("float32", "float32"),
            ("bfloat16", "float32"),
            ("bfloat16", "bfloat16"),
            ("float32", "bfloat16"),
        ],
        ids=["fp32-base-fp32-gate", "bf16-base-fp32-gate", "bf16-base-bf16-gate",
             "fp32-base-bf16-gate"],
    )
    def test_compute_loss_casts_into_and_out_of_the_gate(self, base_dtype, gate_dtype):
        """``bf16 base, fp32 gate`` is the CUDA branch after this fix. The other pairs are
        the CPU branch, ``main``'s CUDA branch, and fp32 outputs of an autocast forward
        meeting a bf16 gate."""
        import torch
        from transformers import Trainer

        from soup_cli.trainer.mole_routing import make_mole_trainer_class
        from soup_cli.utils.mole_routing import MoleGatingConfig, build_gating_kernel

        base = getattr(torch, base_dtype)
        gate_type = getattr(torch, gate_dtype)

        class _PeftShaped(torch.nn.Module):
            """``disable_adapter()`` / ``set_adapter()`` and a forward that returns hidden
            states and logits in the base dtype, different for every adapter."""

            def __init__(self):
                super().__init__()
                self.active = "task_0"
                self.disabled = False
                self.mole_gate = build_gating_kernel(MoleGatingConfig(
                    num_task_adapters=2, hidden_dim=8, temperature=1.0, top_k=2,
                )).to(gate_type)
                self._soup_mole_adapter_names = ["task_0", "task_1"]

            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def set_adapter(self, name):
                self.active = name

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                seed = 0 if self.disabled else 1 + int(self.active.rsplit("_", 1)[-1])
                generator = torch.Generator().manual_seed(seed)
                rows, tokens = input_ids.shape
                hidden = torch.randn(rows, tokens, 8, generator=generator).to(base)
                logits = torch.randn(rows, tokens, 16, generator=generator).to(base)
                return SimpleNamespace(hidden_states=(hidden,), logits=logits)

        model = _PeftShaped()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5], [5, 6, 7, -100, -100]]),
        }
        compute_loss = make_mole_trainer_class(Trainer).compute_loss

        loss, outputs = compute_loss(SimpleNamespace(), model, batch, return_outputs=True)
        loss.backward()

        assert torch.isfinite(loss)
        assert outputs["logits"].dtype == base, "the blend left the logits' dtype"
        grad = model.mole_gate.gate.weight.grad
        assert grad is not None, "the loss did not reach the gate"
        assert grad.dtype == gate_type, grad.dtype
        assert float(grad.float().abs().sum()) > 0.0


class TestTheSavedGate:
    """``mole_gate.pt`` is fp32, and ``soup serve --mole`` loads both it and a bf16 file
    an older run wrote."""

    def test_mole_gate_pt_is_fp32_and_serves_bit_exactly(
        self, mole_inputs, tmp_path, monkeypatch, cpu_run
    ):
        """``main`` saved the CUDA branch's gate in bf16."""
        import torch

        from soup_cli.utils.mole_routing import load_mole_for_serve

        monkeypatch.chdir(tmp_path)
        wrapper = _setup(*mole_inputs("h64"), tmp_path / "run", device="cuda")
        result = wrapper.train()
        trained = _gate_weight(wrapper).detach()

        saved = torch.load(result["gate_path"], map_location="cpu", weights_only=True)
        assert list(saved) == ["gate.weight"], list(saved)
        assert saved["gate.weight"].dtype == torch.float32, saved["gate.weight"].dtype
        assert torch.equal(saved["gate.weight"], trained)
        served = load_mole_for_serve("run")
        assert served.gate.gate.weight.dtype == torch.float32
        assert torch.equal(served.gate.gate.weight, trained)

    def test_the_file_is_fp32_even_when_the_module_is_16_bit_at_save(
        self, mole_inputs, tmp_path, monkeypatch, cpu_run
    ):
        """A DeepSpeed bf16/fp16 engine casts the module to 16-bit in place and keeps its
        fp32 master copy inside its optimizer, so the module the save reads can be
        16-bit. DeepSpeed is not installed here; an in-place cast after training stands
        in for it. The file is fp32 whatever the module holds."""
        import torch
        import transformers

        real_train = transformers.Trainer.train

        def train_then_cast_to_bf16(self, *args, **kwargs):
            result = real_train(self, *args, **kwargs)
            self.model.mole_gate.to(torch.bfloat16)
            return result

        monkeypatch.setattr(transformers.Trainer, "train", train_then_cast_to_bf16)
        wrapper = _setup(*mole_inputs("h64"), tmp_path, device="cuda")

        result = wrapper.train()

        held = _gate_weight(wrapper).detach()
        assert held.dtype == torch.bfloat16, "the stand-in cast did not take"
        saved = torch.load(result["gate_path"], map_location="cpu", weights_only=True)
        assert saved["gate.weight"].dtype == torch.float32, saved["gate.weight"].dtype
        assert torch.equal(saved["gate.weight"], held.float())

    def test_an_old_bf16_gate_file_still_loads(self, mole_inputs, tmp_path, monkeypatch):
        """Control: runs before this fix wrote a bf16 ``mole_gate.pt`` on CUDA. The serve
        loader builds an fp32 gate and ``load_state_dict`` upcasts into it, exactly."""
        import torch

        from soup_cli.utils.mole_routing import (
            MoleServeManifest,
            load_mole_for_serve,
            write_mole_manifest,
        )

        base, adapters = mole_inputs("h64")
        monkeypatch.chdir(tmp_path)
        run = tmp_path / "old_run"
        write_mole_manifest(MoleServeManifest(
            base=base, adapters=tuple(adapters), num_task_adapters=2, hidden_dim=64,
            top_k=2, temperature=1.0,
        ), str(run))
        old = (torch.randn(2, 64, generator=torch.Generator().manual_seed(7)) * 0.1).to(
            torch.bfloat16
        )
        torch.save({"gate.weight": old}, str(run / "mole_gate.pt"))

        served = load_mole_for_serve("old_run")

        assert served.gate.gate.weight.dtype == torch.float32
        assert torch.equal(served.gate.gate.weight, old.float())
        ids = torch.tensor([[1, 5, 6]])
        out = served.generate(ids, torch.ones_like(ids), max_new_tokens=2)
        assert out.shape == (1, 5), out.shape
