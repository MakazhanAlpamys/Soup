"""#1235: PRM on CUDA trained bf16 weights with no fp32 master copy.

``task: prm`` fine-tunes every base parameter plus an ``nn.Linear(hidden, 1)``
reward head. ``prm.py`` loaded all of them in bfloat16 whenever the device string
was exactly ``"cuda"``, and nothing kept an fp32 copy for the optimizer:

* under the bf16 mixed precision an Ampere+ card gets, Accelerate autocasts only
  ``model.forward`` and converts its outputs to fp32, while ``compute_loss`` runs
  the bf16 reward head outside that forward, so the first step raised
  ``mat1 and mat2 must have the same dtype``;
* with the head patched to fp32, an AdamW step of about ``lr`` is below half a
  bf16 ulp for every ``|w| >= 2**-7`` at ``lr=2e-5``, so most of the base never
  moved (the RMSNorm weights, which start at 1.0, not at all);
* a pre-Ampere card gets fp16 autocast plus a GradScaler, whose CUDA unscale
  kernel has no bf16 path (#425), and PRM has no ``lora_`` parameters for
  ``align_trainable_dtype_for_fp16`` to recast.

The dtype decision reads the device string, so ``PRMTrainerWrapper(cfg,
device="cuda")`` on a CPU-only box builds the model a GPU run builds. Mixed
precision runs as CPU autocast through ``use_cpu=True``, which goes through the
same Accelerate wrapper (autocast the forward, convert its outputs to fp32) that
CUDA autocast does. Nothing here touches a GPU.

Those CPU bf16/fp16 steps run with oneDNN off (``onednn_off`` in conftest.py). On
part of GitHub's ``windows-latest`` fleet the first bf16 matmul oneDNN runs dies
with ``0xc000001d`` and takes the rest of the cell with it; the dtypes this file
pins do not depend on which library multiplies the matrices.
"""

from __future__ import annotations

import io
import json
import re
from types import MethodType, SimpleNamespace

import pytest
import yaml

ROWS = [
    {"prompt": "two plus two", "completions": ["we add two and two", "so the answer is four"],
     "labels": [1.0, 1.0]},
    {"prompt": "two plus three", "completions": ["we add two and three", "the answer is four"],
     "labels": [1.0, 0.0]},
    {"prompt": "one plus three", "completions": ["step one we add", "then check it is four"],
     "labels": [1.0, 1.0]},
    {"prompt": "three plus two", "completions": ["we add", "so the answer is four"],
     "labels": [1.0, 0.0]},
]
_WORDS = (
    "two", "plus", "is", "four", "five", "so", "the", "answer", "step", "one", "three",
    "we", "add", "then", "check", "it",
)
_MAX_LENGTH = 64

# The groups a PRM step trains. ``lm_head`` is left out on purpose: PRM reads the
# hidden states, so the logits never reach the loss and ``lm_head`` gets no
# gradient in any precision.
_TRAINED_GROUPS = ("embed_tokens", "attn/mlp", "norms", "reward_head")

# Rich colours per character on Linux CI; compare on plain, whitespace-collapsed text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


def _group(name: str) -> str:
    if "reward_head" in name:
        return "reward_head"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "norms"
    if "embed_tokens" in name:
        return "embed_tokens"
    return "attn/mlp"


@pytest.fixture(scope="module")
def prm_base(tmp_path_factory):
    """A tiny random-init Llama and a word-level tokenizer, saved to disk.

    Saved in bf16, like most published checkpoints, so a load that falls back to
    the checkpoint's own dtype (``"auto"``) cannot pass for an explicit fp32 load.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM

    model_dir = tmp_path_factory.mktemp("prm1235") / "tiny-llama"
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
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


def _cfg(base, out_dir, *, lr=1.0e-5, epochs=2, batch_size=1, training=None):
    """The ``docs/training.md`` PRM example, with the base swapped for the tiny model."""
    from soup_cli.config.loader import load_config_from_string

    block = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "logging_steps": 1,
        "save_steps": 10_000, "gradient_accumulation_steps": 1,
    }
    block.update(training or {})
    return load_config_from_string(yaml.safe_dump({
        "base": base, "task": "prm",
        "data": {"format": "prm", "train": "unused.jsonl", "max_length": _MAX_LENGTH},
        "training": block,
        "output": str(out_dir),
    }))


def _setup(base, out_dir, *, device="cuda", **kwargs):
    from soup_cli.trainer.prm import PRMTrainerWrapper

    wrapper = PRMTrainerWrapper(_cfg(base, out_dir, **kwargs), device=device)
    wrapper.setup({"train": ROWS})
    return wrapper


def _snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _unchanged_fraction_per_group(before, model):
    """Per group, the fraction of elements that are bit-identical after training."""
    totals: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        same = int((before[name] == param.detach().to(before[name].dtype)).sum())
        entry = totals.setdefault(_group(name), [0, 0])
        entry[0] += same
        entry[1] += param.numel()
    return {group: same / total for group, (same, total) in totals.items()}


def _max_delta_per_tensor(before, model, groups):
    return {
        name: float((param.detach().float() - before[name].float()).abs().max())
        for name, param in model.named_parameters()
        if _group(name) in groups
    }


@pytest.fixture
def precision(monkeypatch, onednn_off):
    """Pin the ``(bf16, fp16)`` flags ``train()`` asks for, and keep the run on CPU.

    ``use_cpu=True`` is what lets ``bf16=True`` run on a box with no GPU, as CPU
    autocast inside Accelerate's own forward wrapper. It also keeps a dev box that
    has a card from training on it; the tests assert ``args.use_cpu`` so a patch
    that did not take cannot pass silently. A factory rather than a subclass, so
    the arguments object stays a real, picklable ``TrainingArguments`` for
    ``save_model``. The step runs with oneDNN off (``onednn_off``), which the tests
    assert for the same reason.
    """
    import sys

    import transformers

    import soup_cli.trainer.prm as prm_mod

    # The first lazy attribute access can replace sys.modules["transformers"] with
    # a new module object, and ``train()``'s ``from transformers import ...``
    # reads that one. Resolve first, then patch the object the import will see.
    real = transformers.TrainingArguments
    module = sys.modules["transformers"]

    def cpu_training_arguments(*args, **kwargs):
        return real(*args, **{**kwargs, "use_cpu": True})

    monkeypatch.setattr(module, "TrainingArguments", cpu_training_arguments)

    def apply(bf16, fp16):
        monkeypatch.setattr(prm_mod, "bf16_fp16_flags", lambda device, **_kw: (bf16, fp16))

    apply(False, False)
    return apply


class TestTheLoadDtype:
    """Every trainable parameter, the reward head included, is an fp32 master weight."""

    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
    def test_every_trainable_parameter_loads_as_fp32_on_cuda(self, prm_base, tmp_path, device):
        """On ``main`` all 90,497 elements of this model were bf16 for ``"cuda"``."""
        import torch

        wrapper = _setup(prm_base, tmp_path, device=device)
        params = dict(wrapper.model.named_parameters())
        assert {"reward_head.weight", "reward_head.bias"} <= set(params)
        assert all(p.requires_grad for p in params.values()), "PRM trains every parameter"
        wrong = {name: str(p.dtype) for name, p in params.items() if p.dtype != torch.float32}
        assert not wrong, f"not fp32 master weights on device={device!r}: {wrong}"

    def test_cuda_and_cuda0_take_the_same_decision(self, prm_base, tmp_path):
        """``soup train`` passes ``"cuda"``; a direct caller may pass ``"cuda:0"``.
        ``main`` compared the string with ``==`` and loaded bf16 for one, fp32 for
        the other."""
        import torch

        dtypes = {
            device: {p.dtype for p in _setup(prm_base, tmp_path / str(i), device=device)
                     .model.parameters()}
            for i, device in enumerate(("cuda", "cuda:0"))
        }
        assert dtypes["cuda"] == dtypes["cuda:0"] == {torch.float32}, dtypes

    @pytest.mark.parametrize("device", ["cpu", "mps"])
    def test_cpu_and_mps_stay_fp32(self, prm_base, tmp_path, device):
        """Control: CPU was fp32 already, and #564 chose fp32 masters for MPS."""
        import torch

        wrapper = _setup(prm_base, tmp_path, device=device)
        assert {p.dtype for p in wrapper.model.parameters()} == {torch.float32}

    def test_a_deepspeed_config_does_not_change_the_decision(self, prm_base, tmp_path):
        """The load dtype is decided in ``setup()``, before any engine exists. A
        DeepSpeed bf16/fp16 engine later casts the parameters to 16-bit and builds its
        own fp32 master copy from that cast, so this pins only the load decision."""
        import torch

        from soup_cli.trainer.prm import PRMTrainerWrapper

        ds_config = tmp_path / "ds_zero2.json"
        ds_config.write_text(json.dumps({
            "train_micro_batch_size_per_gpu": "auto",
            "bf16": {"enabled": "auto"},
            "zero_optimization": {"stage": 2},
        }), encoding="utf-8")
        wrapper = PRMTrainerWrapper(
            _cfg(prm_base, tmp_path), device="cuda", deepspeed_config=str(ds_config)
        )
        wrapper.setup({"train": ROWS})
        assert {p.dtype for p in wrapper.model.parameters()} == {torch.float32}

    def test_lora_r_zero_takes_the_same_decision(self, prm_base, tmp_path):
        """``lora.r: 0`` is the one LoRA block PRM accepts (#795). PRM is a full
        fine-tune whether or not it is spelled out, so the decision cannot hang on
        ``is_full_finetune``, which is False for the default block."""
        import torch

        wrapper = _setup(prm_base, tmp_path, training={"lora": {"r": 0}})
        assert {p.dtype for p in wrapper.model.parameters()} == {torch.float32}


class TestBf16MixedPrecisionTrains:
    """The flags an Ampere+ card gets, ``(bf16=True, fp16=False)``.

    ``lr`` 1e-5 is the ``docs/training.md`` example and 2e-5 the schema default.
    """

    @pytest.mark.parametrize("lr", [1.0e-5, 2.0e-5], ids=["docs-lr", "default-lr"])
    def test_one_step_runs_and_moves_the_norms_and_projections(
        self, prm_base, tmp_path, precision, lr
    ):
        """On ``main`` this step raised ``mat1 and mat2 must have the same dtype``.

        One AdamW step moves each element by at most ``lr``. RMSNorm weights start
        at 1.0, whose nearest bf16 neighbours are 2**-8 below and 2**-7 above, so a
        bf16 base leaves every one of them where it was.
        """
        import torch

        precision(True, False)
        wrapper = _setup(prm_base, tmp_path, lr=lr, epochs=1, batch_size=len(ROWS))
        before = _snapshot(wrapper.model)

        wrapper.train()

        trainer = wrapper.trainer
        assert trainer.args.use_cpu
        assert not torch.backends.mkldnn.enabled, "the bf16 step ran with oneDNN on"
        assert trainer.args.bf16 and trainer.accelerator.native_amp, "autocast was not on"
        assert trainer.state.global_step == 1
        deltas = _max_delta_per_tensor(before, wrapper.model, ("norms", "attn/mlp"))
        assert len(deltas) == 5 + 2 * 7, sorted(deltas)
        frozen = sorted(name for name, delta in deltas.items() if delta == 0.0)
        assert not frozen, f"a step at lr={lr} left these bit-identical: {frozen}"
        norm_step = max(delta for name, delta in deltas.items() if _group(name) == "norms")
        assert lr * 0.5 <= norm_step <= lr * 1.01, (norm_step, lr)

    @pytest.mark.parametrize("lr", [1.0e-5, 2.0e-5], ids=["docs-lr", "default-lr"])
    def test_eight_steps_leave_no_trained_group_bit_identical(
        self, prm_base, tmp_path, precision, lr
    ):
        """The issue's run: 4 rows, batch 1, 2 epochs. With the head patched to fp32
        and the base left bf16, 100% of the norms and 84.7% of the attention/MLP
        elements were still bit-identical after these 8 steps."""
        import torch

        precision(True, False)
        wrapper = _setup(prm_base, tmp_path, lr=lr)
        before = _snapshot(wrapper.model)

        wrapper.train()

        assert wrapper.trainer.args.use_cpu
        assert not torch.backends.mkldnn.enabled, "the bf16 steps ran with oneDNN on"
        assert wrapper.trainer.args.bf16 and wrapper.trainer.accelerator.native_amp
        assert wrapper.trainer.state.global_step == 8
        unchanged = _unchanged_fraction_per_group(before, wrapper.model)
        assert set(_TRAINED_GROUPS) <= set(unchanged), unchanged
        stuck = {g: unchanged[g] for g in _TRAINED_GROUPS if unchanged[g] == 1.0}
        assert not stuck, f"bit-identical after 8 steps at lr={lr}: {stuck}"
        # Not 0.0: an element whose gradient is exactly zero on every step stays put
        # even on fp32 master weights, and the CPU bf16 autocast kernels decide which
        # elements those are. CI on Windows CPython 3.10/3.11 left 1 of 320 norm
        # elements (0.3%) unmoved where 3.12 left none. The defect this pins left 100%
        # of the norms and 84.7% of attn/mlp bit-identical, so 1% still separates them.
        assert unchanged["norms"] <= 0.01, unchanged
        assert unchanged["attn/mlp"] <= 0.01, unchanged
        assert unchanged["reward_head"] <= 0.01, unchanged


class TestPreAmpereFp16:
    """The flags a T4/V100/P100 gets, ``(bf16=False, fp16=True)``: fp16 autocast plus
    a GradScaler. On CPU Accelerate enables neither, so the scaler step is driven by
    hand below with the same wrapper Accelerate applies on CUDA."""

    def test_every_trainable_parameter_is_fp32_when_the_optimizer_is_built(
        self, prm_base, tmp_path, precision, monkeypatch
    ):
        """``align_trainable_dtype_for_fp16`` recasts only ``lora_`` parameters and
        PRM has none, so the load itself has to hand the GradScaler fp32 weights."""
        import torch
        import transformers

        precision(False, True)
        seen: dict[str, torch.dtype] = {}
        real_create = transformers.Trainer.create_optimizer

        def spy(self, *args, **kwargs):
            seen.update(
                {n: p.dtype for n, p in self.model.named_parameters() if p.requires_grad}
            )
            return real_create(self, *args, **kwargs)

        monkeypatch.setattr(transformers.Trainer, "create_optimizer", spy)
        wrapper = _setup(prm_base, tmp_path, epochs=1, batch_size=len(ROWS))

        wrapper.train()

        assert wrapper.trainer.args.use_cpu
        assert not torch.backends.mkldnn.enabled, "the fp16 step ran with oneDNN on"
        assert wrapper.trainer.args.fp16 is True
        assert {"reward_head.weight", "reward_head.bias"} <= set(seen), sorted(seen)
        wrong = {n: str(d) for n, d in seen.items() if d != torch.float32}
        assert not wrong, f"the fp16 GradScaler would unscale these: {wrong}"

    def test_an_fp16_autocast_grad_scaler_step_trains(self, prm_base, tmp_path, onednn_off):
        """An optimizer step the way Accelerate runs fp16 on a CUDA card: autocast
        only ``model.forward``, convert its outputs to fp32 (``prepare_model`` in
        ``accelerate/accelerator.py``), and scale the loss. CPU autocast and the CPU
        GradScaler stand in for the CUDA ones, with oneDNN off (``onednn_off``).

        The scaler starts at 2**16 and, as on a card, skips the steps whose scaled
        gradients overflow fp16 and halves the scale (this model takes its first
        step at 2**14). The loop runs until a step is taken, like a training run.
        """
        import torch
        from accelerate.utils import convert_outputs_to_fp32
        from transformers import Trainer

        from soup_cli.trainer.prm import (
            _build_collator,
            _prepare_prm_dataset,
            make_prm_trainer_class,
        )

        wrapper = _setup(prm_base, tmp_path)
        model, tokenizer = wrapper.model, wrapper.tokenizer
        autocast = torch.autocast(device_type="cpu", dtype=torch.float16)
        model.forward = MethodType(
            convert_outputs_to_fp32(autocast(model.forward.__func__)), model
        )
        batch = _build_collator(tokenizer)(_prepare_prm_dataset(ROWS, tokenizer, _MAX_LENGTH))
        trained = {n: p for n, p in model.named_parameters() if _group(n) in _TRAINED_GROUPS}
        before = _snapshot(model)
        optimizer = torch.optim.AdamW(trained.values(), lr=1.0e-5)
        scaler = torch.amp.GradScaler("cpu")
        compute_loss = make_prm_trainer_class(Trainer).compute_loss

        taken_at = None
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(SimpleNamespace(model=model), model, batch)
            scaler.scale(loss).backward()
            # CUDA's unscale kernel has no bf16 implementation (#425); the CPU one
            # does, so the dtype it would reject is asserted directly.
            grads = {n: p.grad.dtype for n, p in trained.items() if p.grad is not None}
            assert set(grads) == set(trained), sorted(set(trained) - set(grads))
            assert set(grads.values()) == {torch.float32}, grads
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale:  # no inf/nan found: the step was taken
                taken_at = attempt
                break
        assert taken_at is not None, "the GradScaler skipped every step"
        assert not torch.backends.mkldnn.enabled, "the fp16 step ran with oneDNN on"
        deltas = _max_delta_per_tensor(before, model, ("norms", "attn/mlp"))
        frozen = sorted(name for name, delta in deltas.items() if delta == 0.0)
        assert not frozen, f"an fp16-autocast step left these bit-identical: {frozen}"


class TestTheRewardHeadTakesTheForwardsDtype:
    """The head runs in ``compute_loss``, outside the forward Accelerate wraps, so it
    must take whatever dtype that forward hands back."""

    @pytest.mark.parametrize(
        "hidden_dtype, head_dtype",
        [
            ("float32", "float32"),
            ("bfloat16", "bfloat16"),
            ("float32", "bfloat16"),
            ("bfloat16", "float32"),
        ],
        ids=["fp32-fp32", "bf16-bf16", "fp32-hidden-bf16-head", "bf16-hidden-fp32-head"],
    )
    def test_compute_loss_accepts_the_hidden_state_dtype(self, hidden_dtype, head_dtype):
        """The matched pairs are controls. ``fp32 hidden, bf16 head`` is the #1235
        crash site: Accelerate's fp32 outputs meeting ``main``'s bf16 head."""
        import torch
        from transformers import Trainer

        from soup_cli.trainer.prm import make_prm_trainer_class

        class _Forward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(8, 4)
                self.reward_head = torch.nn.Linear(4, 1).to(getattr(torch, head_dtype))

            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                hidden = self.embed(input_ids).to(getattr(torch, hidden_dtype))
                return SimpleNamespace(hidden_states=(hidden,))

        model = _Forward()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "step_positions": torch.tensor([[1, 3], [2, -1]]),
            "labels": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        }
        compute_loss = make_prm_trainer_class(Trainer).compute_loss

        loss = compute_loss(SimpleNamespace(model=model), model, batch)
        loss.backward()

        assert torch.isfinite(loss)
        assert model.reward_head.weight.grad is not None
        assert model.embed.weight.grad is not None, "the base fell out of the graph"
        assert float(model.embed.weight.grad.abs().sum()) > 0.0


class TestThePreflight:
    """The VRAM pre-flight budgets PRM at fp32 master weights plus optimizer state,
    and a card that cannot hold that is refused rather than trained in pure bf16."""

    @staticmethod
    def _cfg():
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(
            "base: Qwen/Qwen2.5-1.5B-Instruct\ntask: prm\n"
            "data: {format: prm, train: x.jsonl, max_length: 2048}\n"
            "training: {batch_size: 1}\n"
        )

    def test_the_budget_matches_the_dtype_the_trainer_loads(self, prm_base, tmp_path):
        """Weights and gradients once each in the parameter dtype, and AdamW's two
        moments, which torch keeps in the parameter dtype too: 16 bytes per
        parameter at fp32. ``main`` budgeted that while loading bf16 (8 bytes).

        Default optimizer only: PRM ignores ``training.optimizer`` and always runs
        HF's AdamW (#805), while the pre-flight prices whatever that setting says."""
        import torch

        from soup_cli.commands.train import _build_hardware_fit_input
        from soup_cli.utils.hardware_fit import estimate_peak_vram_gb

        loaded = {p.dtype for p in _setup(prm_base, tmp_path).model.parameters()}
        assert len(loaded) == 1, loaded
        per_param = torch.empty((), dtype=next(iter(loaded))).element_size()
        fit = _build_hardware_fit_input(self._cfg())
        assert fit is not None and (fit.peft, fit.quant) == ("full", "none"), fit
        budget = estimate_peak_vram_gb(fit)

        assert budget.weights_gb == pytest.approx(fit.params_b * per_param), loaded
        assert budget.gradients_gb == pytest.approx(fit.params_b * per_param), loaded
        assert budget.optimizer_gb == pytest.approx(fit.params_b * 2 * per_param), loaded
        per_param_total = budget.weights_gb + budget.gradients_gb + budget.optimizer_gb
        assert per_param_total / fit.params_b == pytest.approx(16.0)

    @pytest.fixture
    def printed(self, monkeypatch):
        from rich.console import Console

        from soup_cli.commands import train as train_cmd

        buffer = io.StringIO()
        monkeypatch.setattr(
            train_cmd, "console",
            Console(file=buffer, width=200, force_terminal=False, color_system=None),
        )
        return buffer

    def test_a_card_that_cannot_hold_fp32_masters_is_refused(self, printed):
        """A 16 GB T4/V100 and a 1.5B PRM: ~28 GB predicted with the margin."""
        import typer

        from soup_cli.commands.train import _hardware_fit_preflight

        with pytest.raises(typer.Exit) as excinfo:
            _hardware_fit_preflight(
                self._cfg(), {"memory_total_bytes": 16 * 10**9}, allow_oom_attempt=False
            )

        assert excinfo.value.exit_code == 1
        out = _plain(printed.getvalue())
        assert "Hardware-fit gate" in out, out
        assert "exceeds 16.0 GB available" in out, out
        assert "weights 6.0 | optim 12.0 | grads 6.0" in out, out

    def test_a_card_that_can_hold_them_is_not(self, printed):
        """Control: the same run on an 80 GB card passes silently."""
        from soup_cli.commands.train import _hardware_fit_preflight

        _hardware_fit_preflight(
            self._cfg(), {"memory_total_bytes": 80 * 10**9}, allow_oom_attempt=False
        )
        assert printed.getvalue() == ""


class TestTheQuantizationMessages:
    """The #795 warning and refusal said the trainer loads the base "at checkpoint
    precision", which PRM no longer does. What holds for all eight tasks is that
    none of them quantises the base."""

    _PRM = "base: org/model\ntask: prm\ndata: {train: x.jsonl}\n"

    def test_the_warning_says_the_base_is_never_quantised(self, monkeypatch):
        from rich.console import Console

        from soup_cli.config import loader
        from soup_cli.config.loader import load_config_from_string

        buffer = io.StringIO()
        monkeypatch.setattr(
            loader, "console",
            Console(file=buffer, width=500, force_terminal=False, color_system=None),
        )
        cfg = load_config_from_string(self._PRM + "training: {quantization: 4bit}\n")

        assert cfg.training.quantization == "none"
        out = _plain(buffer.getvalue())
        assert "task='prm' and is ignored: its trainer never quantises the base" in out, out
        assert "checkpoint precision" not in out, out

    def test_the_refusal_says_the_base_is_never_quantised(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="its trainer never quantises the base") as exc:
            load_config_from_string(self._PRM + "training: {quantization: gptq}\n")
        assert "task='prm' does not apply training.quantization" in str(exc.value)
        assert "checkpoint precision" not in str(exc.value)


class TestTheOneDnnSwitchOffStaysScoped:
    """Last in the file on purpose: it runs after every test that used ``onednn_off``.

    The fixture must move only the steps that ask for it. One that leaked (made
    autouse or session-wide, or no longer restored) would quietly move every later
    test in the session off oneDNN, and a skipped oneDNN path is invisible in a
    passing run.
    """

    def test_the_fixture_turns_it_off(self, onednn_off):
        import torch

        assert not torch.backends.mkldnn.enabled

    def test_outside_the_fixture_it_is_back_on(self):
        import torch

        assert torch.backends.mkldnn.enabled, "onednn_off leaked out of the tests using it"
