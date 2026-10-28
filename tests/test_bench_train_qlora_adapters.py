"""``soup bench train`` on a 4-bit or 8-bit model, whose adapters TRL casts to bf16 (#2215).

TRL 0.29's ``SFTTrainer`` casts the trainable parameters of any 4-bit or 8-bit model
to bf16, on every card. Two things in ``bench train`` broke on that:

* the fingerprint hashed ``tensor.numpy()``, and numpy has no bfloat16, so the run
  crashed before its first step with ``TypeError: Got unsupported ScalarType
  BFloat16`` (every card);
* ``soup train`` casts those adapters back to fp32 on an fp16 card, in
  ``SFTTrainerWrapper.train()``, which ``bench train`` does not call, so on a T4 or
  V100 the GradScaler failed with ``"_amp_foreach_non_finite_check_and_unscale_cuda"
  not implemented for 'BFloat16'``.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

torch = pytest.importorskip("torch")

from soup_cli.bench.collector import fingerprint_trainable, summarize_trainable  # noqa: E402

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def _model(dtype, shape=(2, 3)):
    model = torch.nn.Module()
    model.register_parameter(
        "w", torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(shape).to(dtype))
    )
    model.register_parameter(
        "frozen", torch.nn.Parameter(torch.ones(2, dtype=dtype), requires_grad=False)
    )
    return model


class TestFingerprintEveryFloatDtype:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
    def test_it_fingerprints_without_raising(self, dtype):
        snapshot = summarize_trainable(_model(dtype))
        assert snapshot.count == 1
        assert len(snapshot.fingerprint_first) == 64

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
    def test_a_changed_weight_changes_it(self, dtype):
        model = _model(dtype)
        before = fingerprint_trainable(model)
        with torch.no_grad():
            model.w[0, 0] += 1.0
        assert fingerprint_trainable(model) != before

    def test_a_zero_dim_bf16_parameter(self):
        model = torch.nn.Module()
        model.register_parameter("s", torch.nn.Parameter(torch.tensor(1.5, dtype=torch.bfloat16)))
        before = fingerprint_trainable(model)
        with torch.no_grad():
            model.s += 1.0
        assert fingerprint_trainable(model) != before

    def test_the_same_values_in_another_dtype_are_another_fingerprint(self):
        assert fingerprint_trainable(_model(torch.bfloat16)) != fingerprint_trainable(
            _model(torch.float16)
        )


class TestFingerprintUnchangedForNumpyDtypes:
    """Only the bf16 case changed: for every dtype numpy could already hash, the
    digest is the same bytes as before."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64], ids=str)
    def test_it_matches_the_numpy_bytes_digest(self, dtype):
        model = _model(dtype)
        hasher = hashlib.sha256()
        hasher.update(b"w")
        hasher.update(model.w.detach().cpu().contiguous().numpy().tobytes())
        assert fingerprint_trainable(model) == hasher.hexdigest()


def _workdir(tmp_path, monkeypatch, quantization="none"):
    for mod in ("transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")
    from tests.test_issue341_seed_and_fullft import _ROWS, _tiny_llama_dir

    base = _tiny_llama_dir(tmp_path)
    rows = [
        {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
        for u, a in _ROWS
    ]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (tmp_path / "soup.yaml").write_text(
        yaml.safe_dump({
            "base": base,
            "task": "sft",
            "backend": "transformers",
            "modality": "text",
            "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
            "training": {
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "quantization": quantization,
                "lr": 1e-2,
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
            },
            "output": str(tmp_path / "out"),
        }),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _bench(device, before_train=None, steps=4, warmup=1):
    from soup_cli.bench.train_run import run_bench_train
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset

    return run_bench_train(
        load_config("soup.yaml"), steps=steps, warmup=warmup, device=device,
        load_dataset=lambda c: load_dataset(c.data), before_train=before_train,
    )


def _lora_dtypes(model):
    return {str(p.dtype) for n, p in model.named_parameters() if p.requires_grad and "lora_" in n}


class TestThroughARealTrainer:
    def test_a_run_with_bf16_adapters_is_measured_and_valid(self, tmp_path, monkeypatch):
        """The shape TRL leaves a QLoRA model in on a bf16 card: trainable adapters
        in bf16 when the first fingerprint is taken."""
        _workdir(tmp_path, monkeypatch)

        def to_bf16(wrapper):
            for param in wrapper.trainer.model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        report = _bench("cpu", before_train=to_bf16)
        assert report["valid"] is True, report["failures"]
        assert report["checks"]["parameters_changed"] is True

    def test_it_aligns_the_adapter_dtype_as_soup_train_does(self, tmp_path, monkeypatch):
        """``soup train`` calls ``align_trainable_dtype_for_fp16`` with the trainer's
        own precision flags just before training; bench must too, or it measures
        (and on fp16 cards crashes) a run ``soup train`` never makes."""
        import soup_cli.utils.mixed_precision as mixed_precision

        _workdir(tmp_path, monkeypatch)
        calls = []
        real = mixed_precision.align_trainable_dtype_for_fp16

        def spy(model, *, fp16, bf16):
            calls.append((model, fp16, bf16))
            return real(model, fp16=fp16, bf16=bf16)

        monkeypatch.setattr(mixed_precision, "align_trainable_dtype_for_fp16", spy)
        seen = {}

        def record(wrapper):
            seen["model"] = wrapper.trainer.model
            seen["flags"] = (wrapper.trainer.args.fp16, wrapper.trainer.args.bf16)

        _bench("cpu", before_train=record)
        assert len(calls) == 1
        model, fp16, bf16 = calls[0]
        assert model is seen["model"]
        assert (fp16, bf16) == seen["flags"]


class TestFlagsArePassedThrough:
    def test_it_passes_the_trainers_own_flags_not_a_fixed_pair(self, tmp_path, monkeypatch):
        """On a CPU runner both flags are False, so the test above cannot tell the
        trainer's flags from a hardcoded pair. Give the trainer a non-default pair and
        stop at the alignment, so nothing trains with the fake flags."""
        import soup_cli.utils.mixed_precision as mixed_precision
        from soup_cli.trainer.sft import SFTTrainerWrapper

        _workdir(tmp_path, monkeypatch)
        holder = {}
        real_setup = SFTTrainerWrapper.setup

        class AlignmentReachedError(Exception):
            pass

        def setup(self, *args, **kwargs):
            out = real_setup(self, *args, **kwargs)
            self.trainer.args.fp16, self.trainer.args.bf16 = True, False
            holder["wrapper"] = self
            return out

        def spy(model, *, fp16, bf16):
            holder["call"] = (model, fp16, bf16)
            raise AlignmentReachedError

        monkeypatch.setattr(SFTTrainerWrapper, "setup", setup)
        monkeypatch.setattr(mixed_precision, "align_trainable_dtype_for_fp16", spy)
        with pytest.raises(AlignmentReachedError):
            _bench("cpu")
        model, fp16, bf16 = holder["call"]
        assert model is holder["wrapper"].trainer.model
        assert (fp16, bf16) == (True, False)


class TestAlignmentOrder:
    def test_it_aligns_before_the_before_train_hook_and_the_first_fingerprint(
        self, tmp_path, monkeypatch
    ):
        """Only the GPU cases pinned the order, so a CPU runner would not notice the
        alignment moving behind ``before_train`` or the nothing-to-train check. The
        hook stops the run, so nothing trains."""
        import soup_cli.utils.mixed_precision as mixed_precision

        _workdir(tmp_path, monkeypatch)
        order = []

        class HookReachedError(Exception):
            pass

        def spy(model, *, fp16, bf16):
            order.append("align")
            return 0

        def hook(wrapper):
            order.append("before_train")
            raise HookReachedError

        monkeypatch.setattr(mixed_precision, "align_trainable_dtype_for_fp16", spy)
        with pytest.raises(HookReachedError):
            _bench("cpu", before_train=hook)
        assert order == ["align", "before_train"]


@pytest.mark.gpu
@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
class TestAQuantizedModelOnCuda:
    def test_on_an_fp16_card_the_run_trains_with_fp32_adapters(
        self, tmp_path, monkeypatch, quantization
    ):
        """A T4/V100 run: no bf16 hardware, so fp16 with a GradScaler. Forced here by
        reporting no bf16 support, which is the question Soup's precision choice asks."""
        pytest.importorskip("bitsandbytes")
        _workdir(tmp_path, monkeypatch, quantization=quantization)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
        seen = {}

        def record(wrapper):
            seen["args"] = (wrapper.trainer.args.fp16, wrapper.trainer.args.bf16)
            seen["dtypes"] = _lora_dtypes(wrapper.trainer.model)

        report = _bench("cuda", before_train=record)
        assert seen["args"] == (True, False)
        assert seen["dtypes"] == {"torch.float32"}
        assert report["valid"] is True, report["failures"]

    def test_with_the_cards_own_precision_the_run_is_valid(
        self, tmp_path, monkeypatch, quantization
    ):
        """On a bf16 card this is the fingerprint-over-bf16-adapters case; on an fp16
        card it is the case above without the forcing."""
        pytest.importorskip("bitsandbytes")
        _workdir(tmp_path, monkeypatch, quantization=quantization)
        report = _bench("cuda")
        assert report["valid"] is True, report["failures"]
