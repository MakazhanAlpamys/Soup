"""#634 — MLX backend ignores saved adapter checkpoints on --resume.

Two mechanical gaps, per the maintainer's own scoping on the issue:

1. ``_resolve_checkpoint("auto", ...)`` only looks for ``checkpoint-N``
   directories (the transformers/unsloth shape). mlx-lm's tuner writes
   step-numbered ``NNNNNNN_adapters.safetensors`` FILES instead, so "auto"
   resume never found them — this is the exact bug reproduction, confirmed
   against the pre-fix function shape below.
2. ``MLXSFTTrainerWrapper.train()`` accepted ``resume_from_checkpoint`` and
   printed a warning that it does nothing with it.

Deliberately NOT in scope, per the issue: resuming the dataset iterator to
the exact saved step. mlx-lm's LoRA trainer exposes no optimizer state and
no step count, so this is a weights-only warm start, not a full resume —
the docstring on ``_load_checkpoint_weights`` says so, on purpose.

Caveat that still matters, even though most of this now runs: MLX only
runs on Apple Silicon. ``TestTrainWiresCheckpointLoadingAfterLoraIsApplied``
executes the real ``train()`` method body against minimal fake ``mlx`` /
``mlx_lm`` modules registered in ``sys.modules`` — real control flow, real
call ordering, not source-text inspection — but the fakes are stand-ins
for real mlx-lm behaviour (LoRA conversion, the actual training loop,
what ``load_weights(strict=False)`` does on a real tensor-shape mismatch),
not a substitute for it. What genuinely cannot be verified here and is
flagged as such rather than implied: an end-to-end train -> interrupt ->
``--resume auto`` run against real mlx-lm.
"""

from __future__ import annotations

import json
import re
import sys
import types

import pytest


def _write_fake_safetensors(path, tensor_names):
    """A minimal but structurally real .safetensors file: the real 8-byte
    length-prefixed JSON header format, with dummy F32 tensor data behind
    it. Real enough for _count_safetensors_tensors to parse without mlx."""
    header = {}
    offset = 0
    for name in tensor_names:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(b"\x00" * offset)


def _mlx_wrapper(tmp_path, **training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

    cfg = SoupConfig(
        base="mlx-community/Llama-3.1-8B-Instruct-4bit",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="chatml"),
        training=TrainingConfig(**training),
        output=str(tmp_path),
    )
    return MLXSFTTrainerWrapper(cfg)


class TestHighestNumberedCheckpointIsOrderIndependent:
    """_highest_numbered_mlx_checkpoint picks by parsed step number via
    max(), not by sorting whatever order the filesystem enumerated files in,
    and not by comparing filenames as strings. The first test's list is
    explicit and adversarial on ORDER — the highest step is neither first
    nor last — so a key/sort mistake can't coincidentally still return the
    right file the way it could if this only iterated a real directory in
    on-disk order. That alone doesn't rule out a key/TYPE mistake though:
    equal-width zero-padded numbers sort identically whether compared as
    strings or as ints, so a regression that keyed by ``path.name`` instead
    of the parsed int would pass an order-independence check built only from
    padded names. The second test uses unpadded, different-width numbers
    (99 vs 100) specifically because string order and numeric order disagree
    there, which is what actually pins the key down to the parsed int."""

    def test_the_highest_step_wins_regardless_of_input_order(self, tmp_path):
        from soup_cli.commands.train import _highest_numbered_mlx_checkpoint

        names = ["0000200_adapters.safetensors", "0000300_adapters.safetensors",
                 "0000100_adapters.safetensors"]
        paths = []
        for name in names:
            path = tmp_path / name
            path.write_bytes(b"")
            paths.append(path)

        result = _highest_numbered_mlx_checkpoint(paths)
        assert result == tmp_path / "0000300_adapters.safetensors"

    def test_the_highest_step_wins_by_numeric_value_not_string_order(self, tmp_path):
        """"100_adapters.safetensors" < "99_adapters.safetensors" as strings
        (```"1" < "9"```) but 100 > 99 as integers. Only a key that parses
        the digits to an int before comparing picks the right file here."""
        from soup_cli.commands.train import _highest_numbered_mlx_checkpoint

        names = ["99_adapters.safetensors", "100_adapters.safetensors"]
        paths = []
        for name in names:
            path = tmp_path / name
            path.write_bytes(b"")
            paths.append(path)

        result = _highest_numbered_mlx_checkpoint(paths)
        assert result == tmp_path / "100_adapters.safetensors"

    def test_non_matching_files_are_ignored(self, tmp_path):
        from soup_cli.commands.train import _highest_numbered_mlx_checkpoint

        matching = tmp_path / "0000050_adapters.safetensors"
        matching.write_bytes(b"")
        other = tmp_path / "adapters.safetensors"
        other.write_bytes(b"")

        result = _highest_numbered_mlx_checkpoint([other, matching])
        assert result == matching

    def test_empty_input_returns_none(self):
        from soup_cli.commands.train import _highest_numbered_mlx_checkpoint

        assert _highest_numbered_mlx_checkpoint([]) is None


class TestMlxCheckpointResolutionIgnoresExperimentName:
    """The review's second blocking finding: mlx_sft.py's output_dir is
    always Path(cfg.output), flat — unlike trainer/sft.py, which nests
    under output_dir / cfg.experiment_name. An MLX resolver that copies
    that nesting looks in a directory MLX never writes to, and "auto"
    reports no checkpoint found on every config that sets experiment_name
    — the exact #634 symptom, for a config the original fix didn't cover."""

    def test_auto_finds_the_checkpoint_when_experiment_name_is_set(self, tmp_path):
        """MLX writes directly into output_dir regardless of
        experiment_name, so the checkpoint sits at the top level even
        though the config names an experiment."""
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "0000300_adapters.safetensors").write_bytes(b"")

        result = _resolve_checkpoint(
            "auto", str(tmp_path), experiment_name="my-experiment", backend="mlx"
        )
        assert result == str(tmp_path / "0000300_adapters.safetensors")

    def test_experiment_name_does_not_change_the_result(self, tmp_path):
        """Same output dir, with and without experiment_name set — MLX's
        resolution must not depend on it either way."""
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "0000300_adapters.safetensors").write_bytes(b"")

        with_name = _resolve_checkpoint(
            "auto", str(tmp_path), experiment_name="my-experiment", backend="mlx"
        )
        without_name = _resolve_checkpoint("auto", str(tmp_path), backend="mlx")
        assert with_name == without_name == str(tmp_path / "0000300_adapters.safetensors")


class TestMlxCheckpointResolutionFindsStepNumberedFiles:
    """_resolve_checkpoint / _resolve_mlx_checkpoint — no MLX import needed,
    this is pure filesystem logic."""

    def test_auto_picks_the_highest_numbered_snapshot(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "0000100_adapters.safetensors").write_bytes(b"")
        (tmp_path / "0000300_adapters.safetensors").write_bytes(b"")
        (tmp_path / "0000200_adapters.safetensors").write_bytes(b"")

        result = _resolve_checkpoint("auto", str(tmp_path), backend="mlx")
        assert result == str(tmp_path / "0000300_adapters.safetensors")

    def test_auto_falls_back_to_the_final_adapter_file(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "adapters.safetensors").write_bytes(b"")

        result = _resolve_checkpoint("auto", str(tmp_path), backend="mlx")
        assert result == str(tmp_path / "adapters.safetensors")

    def test_auto_prefers_numbered_snapshot_over_final_file(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "adapters.safetensors").write_bytes(b"")
        (tmp_path / "0000050_adapters.safetensors").write_bytes(b"")

        result = _resolve_checkpoint("auto", str(tmp_path), backend="mlx")
        assert result == str(tmp_path / "0000050_adapters.safetensors")

    def test_auto_returns_none_when_output_dir_is_empty(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        assert _resolve_checkpoint("auto", str(tmp_path), backend="mlx") is None

    def test_auto_returns_none_when_output_dir_missing(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        missing = tmp_path / "does-not-exist"
        assert _resolve_checkpoint("auto", str(missing), backend="mlx") is None

    def test_direct_path_to_an_adapter_file_is_accepted(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        checkpoint = tmp_path / "0000042_adapters.safetensors"
        checkpoint.write_bytes(b"")

        result = _resolve_checkpoint(str(checkpoint), str(tmp_path), backend="mlx")
        assert result == str(checkpoint)

    def test_direct_path_to_a_missing_file_is_rejected(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        result = _resolve_checkpoint(
            str(tmp_path / "nope.safetensors"), str(tmp_path), backend="mlx"
        )
        assert result is None


class TestNonMlxBackendsAreUnaffected:
    """Regression guard: the pre-existing transformers/unsloth directory
    logic is untouched, and demonstrates the bug this issue reports — the
    old (and still-default) code path genuinely cannot see MLX's files."""

    def test_default_backend_still_resolves_checkpoint_directories(self, tmp_path):
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-300").mkdir()
        (tmp_path / "checkpoint-200").mkdir()

        result = _resolve_checkpoint("auto", str(tmp_path))
        assert result == str(tmp_path / "checkpoint-300")

    def test_the_default_path_cannot_see_mlx_style_files(self, tmp_path):
        """The #634 bug, reproduced directly: an MLX run's own output
        resolves to nothing under the backend-unaware (default) path,
        because _adapters.safetensors is a file, not a checkpoint-N dir."""
        from soup_cli.commands.train import _resolve_checkpoint

        (tmp_path / "0000300_adapters.safetensors").write_bytes(b"")

        assert _resolve_checkpoint("auto", str(tmp_path)) is None
        assert _resolve_checkpoint("auto", str(tmp_path), backend="mlx") is not None


class TestCountSafetensorsTensors:
    """The MLX-independent half of the #392-shaped post-load check: the
    checkpoint FILE declares at least one tensor, parsed from the format's
    own header — no mlx or safetensors package needed to check it."""

    def test_counts_declared_tensors(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        path = tmp_path / "adapters.safetensors"
        _write_fake_safetensors(path, ["lora_a", "lora_b", "lora_c"])

        assert _count_safetensors_tensors(str(path)) == 3

    def test_empty_header_counts_zero(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        path = tmp_path / "empty.safetensors"
        _write_fake_safetensors(path, [])

        assert _count_safetensors_tensors(str(path)) == 0

    def test_metadata_key_is_not_counted_as_a_tensor(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        path = tmp_path / "metadata_only.safetensors"
        header_bytes = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
        path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes)

        assert _count_safetensors_tensors(str(path)) == 0

    def test_garbage_header_length_raises_value_error_not_memory_error(self, tmp_path):
        """A non-safetensors file's first 8 bytes decode to an arbitrary
        integer. Unbounded, that integer sizes a ``f.read()`` call and can
        demand gigabytes from an 8-byte file. Bounded against the file's own
        size, it must fail fast with a ``ValueError`` naming the path."""
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        path = tmp_path / "not_a_checkpoint.bin"
        path.write_bytes((2**40).to_bytes(8, "little") + b"short")

        with pytest.raises(ValueError, match=re.escape(str(path))):
            _count_safetensors_tensors(str(path))

    def test_truncated_json_raises_value_error(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        path = tmp_path / "truncated.safetensors"
        header_bytes = json.dumps({"lora_a": {"shape": [1]}}).encode("utf-8")
        # Declare the full header length but only write half of it.
        truncated = header_bytes[: len(header_bytes) // 2]
        path.write_bytes(len(header_bytes).to_bytes(8, "little") + truncated)

        with pytest.raises(ValueError, match=re.escape(str(path))):
            _count_safetensors_tensors(str(path))

    def test_directory_path_raises_value_error_not_is_a_directory_error(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        directory = tmp_path / "adapters.safetensors"
        directory.mkdir()

        with pytest.raises(ValueError, match=re.escape(str(directory))):
            _count_safetensors_tensors(str(directory))

    def test_missing_file_raises_value_error_not_file_not_found_error(self, tmp_path):
        from soup_cli.trainer.mlx_sft import _count_safetensors_tensors

        missing = tmp_path / "does_not_exist.safetensors"

        with pytest.raises(ValueError, match=re.escape(str(missing))):
            _count_safetensors_tensors(str(missing))


class TestLoadCheckpointWeights:
    """MLXSFTTrainerWrapper._load_checkpoint_weights, tested against a fake
    model object and a real (minimal) .safetensors file — no real mlx/mlx-lm
    import is reachable here."""

    def test_it_calls_load_weights_non_strict(self, tmp_path):
        calls = []

        class _FakeModel:
            def load_weights(self, path, strict=True):
                calls.append((path, strict))

        checkpoint = tmp_path / "0000100_adapters.safetensors"
        _write_fake_safetensors(checkpoint, ["lora_a"])

        wrapper = _mlx_wrapper(tmp_path)
        wrapper.model = _FakeModel()

        wrapper._load_checkpoint_weights(str(checkpoint))

        assert calls == [(str(checkpoint), False)]

    def test_empty_checkpoint_is_rejected_before_load_weights_runs(self, tmp_path):
        calls = []

        class _FakeModel:
            def load_weights(self, path, strict=True):
                calls.append((path, strict))

        checkpoint = tmp_path / "0000100_adapters.safetensors"
        _write_fake_safetensors(checkpoint, [])

        wrapper = _mlx_wrapper(tmp_path)
        wrapper.model = _FakeModel()

        with pytest.raises(ValueError, match="no tensors"):
            wrapper._load_checkpoint_weights(str(checkpoint))

        assert calls == [], "load_weights must not run against an empty checkpoint"


def _install_fake_mlx(monkeypatch):
    """Register minimal fake ``mlx`` / ``mlx_lm`` modules in ``sys.modules``
    so ``MLXSFTTrainerWrapper.train()`` runs for real, without Apple
    Silicon or the real packages installed.

    Covers exactly the surface ``train()`` touches: ``mlx.core`` (the
    ``_require_mlx`` import probe), ``mlx.optimizers.AdamW``, and
    ``mlx_lm.tuner``'s ``callbacks`` / ``datasets`` / ``trainer`` / ``utils``
    submodules. This exercises the real method body — real control flow,
    real call ordering — rather than asserting on its source text.
    """
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_optimizers = types.ModuleType("mlx.optimizers")
    mlx_optimizers.AdamW = lambda **kwargs: object()

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm_tuner = types.ModuleType("mlx_lm.tuner")

    mlx_lm_tuner_callbacks = types.ModuleType("mlx_lm.tuner.callbacks")

    class _FakeTrainingCallback:
        pass

    mlx_lm_tuner_callbacks.TrainingCallback = _FakeTrainingCallback

    mlx_lm_tuner_datasets = types.ModuleType("mlx_lm.tuner.datasets")
    mlx_lm_tuner_datasets.CacheDataset = lambda rows: rows
    mlx_lm_tuner_datasets.create_dataset = lambda rows, tokenizer, args: rows

    mlx_lm_tuner_trainer = types.ModuleType("mlx_lm.tuner.trainer")

    class _FakeTrainingArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    mlx_lm_tuner_trainer.TrainingArgs = _FakeTrainingArgs

    def _fake_train(**kwargs):
        callback = kwargs.get("training_callback")
        if callback is not None:
            callback.on_train_loss_report({"train_loss": 0.5})

    mlx_lm_tuner_trainer.train = _fake_train

    mlx_lm_tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    mlx_lm_tuner_utils.linear_to_lora_layers = lambda model, num_layers, config: None

    fake_modules = {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx.optimizers": mlx_optimizers,
        "mlx_lm": mlx_lm,
        "mlx_lm.tuner": mlx_lm_tuner,
        "mlx_lm.tuner.callbacks": mlx_lm_tuner_callbacks,
        "mlx_lm.tuner.datasets": mlx_lm_tuner_datasets,
        "mlx_lm.tuner.trainer": mlx_lm_tuner_trainer,
        "mlx_lm.tuner.utils": mlx_lm_tuner_utils,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class _FakeMlxModel:
    """Duck-types just what train() touches on the model: freeze(),
    layers, and load_weights(). Records whether freeze() (part of
    _apply_lora) ran before load_weights() — the real ordering
    constraint, checked behaviourally instead of by source position."""

    def __init__(self):
        self.frozen = False
        self.layers = []
        self.load_weights_calls = []

    def freeze(self):
        self.frozen = True

    def load_weights(self, path, strict=True):
        self.load_weights_calls.append((path, strict, self.frozen))


class TestTrainWiresCheckpointLoadingAfterLoraIsApplied:
    """train() itself, executed for real against fake mlx/mlx_lm modules —
    not inspected at the source level. Exercises the actual method body:
    real branching on resume_from_checkpoint, real call ordering."""

    def _ready_wrapper(self, tmp_path):
        wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1)
        wrapper.model = _FakeMlxModel()
        wrapper.tokenizer = object()
        wrapper._dataset = {"train": [{"text": "hi"}], "val": []}
        return wrapper

    def test_checkpoint_weights_load_after_lora_is_applied(self, tmp_path, monkeypatch):
        _install_fake_mlx(monkeypatch)
        wrapper = self._ready_wrapper(tmp_path)
        checkpoint = tmp_path / "0000100_adapters.safetensors"
        _write_fake_safetensors(checkpoint, ["lora_a"])

        wrapper.train(resume_from_checkpoint=str(checkpoint))

        assert wrapper.model.load_weights_calls == [(str(checkpoint), False, True)], (
            "load_weights must run after freeze() (part of _apply_lora) — "
            "the saved file holds only LoRA-shaped tensors, which don't "
            "exist on the model until the linear layers are converted"
        )

    def test_no_load_when_no_checkpoint_given(self, tmp_path, monkeypatch):
        """#634's original bug: resume_from_checkpoint was accepted and
        never read again. Proven behaviourally in both directions — this
        test pins that omitting it does nothing, the one above pins that
        supplying it does something."""
        _install_fake_mlx(monkeypatch)
        wrapper = self._ready_wrapper(tmp_path)

        wrapper.train(resume_from_checkpoint=None)

        assert wrapper.model.load_weights_calls == []


class TestWriterAndResolverAgreeOnOutputDirWithExperimentName:
    """The maintainer's own reproduction for the last blocking gap: nothing
    ties _resolve_mlx_checkpoint's flat-output-dir assumption to what
    mlx_sft.py's train() actually writes to disk — they were verified
    separately, never against each other. This runs train() for real
    against the fake mlx/mlx_lm harness, with the fake trainer.train()
    writing an adapter file the way the real one does (to args.adapter_file,
    which train() builds from output_dir alone), on a config that sets
    experiment_name the way a real run would. It then resolves "auto"
    against that same config through _resolve_checkpoint — the actual CLI
    entry point, backend and experiment_name included — and asserts it
    finds the file train() just wrote. If mlx_sft.py ever starts nesting
    output under experiment_name the way the transformers/unsloth trainer
    does, the write and the resolve would point at different directories
    and this fails, instead of both sides silently drifting apart again."""

    def test_resolver_finds_the_file_train_just_wrote(self, tmp_path, monkeypatch):
        _install_fake_mlx(monkeypatch)

        def _fake_train_that_saves_an_adapter(**kwargs):
            args = kwargs["args"]
            _write_fake_safetensors(args.adapter_file, ["lora_a"])

        sys.modules["mlx_lm.tuner.trainer"].train = _fake_train_that_saves_an_adapter

        wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1)
        wrapper.config.experiment_name = "exp1"
        wrapper.model = _FakeMlxModel()
        wrapper.tokenizer = object()
        wrapper._dataset = {"train": [{"text": "hi"}], "val": []}

        wrapper.train()

        from soup_cli.commands.train import _resolve_checkpoint

        resolved = _resolve_checkpoint(
            "auto", wrapper.config.output, wrapper.config.experiment_name, backend="mlx"
        )

        assert resolved == str(tmp_path / "adapters.safetensors"), (
            "train() writes to output_dir flat and _resolve_mlx_checkpoint "
            "reads output_dir flat - if either side starts nesting under "
            "experiment_name without the other, resolution silently breaks"
        )


class TestResumeWiringReachesTheBackendAwareResolver:
    """The blocking gap from review: every other test above calls
    _resolve_checkpoint directly with backend="mlx" already supplied, so
    none of them exercise the actual seam between the CLI and the
    resolver — the single keyword the whole fix hangs on. This does, via
    _resolve_resume_or_exit, the function train() itself calls, with
    _resolve_checkpoint replaced by a spy that records what it was called
    with."""

    def _cfg(self, tmp_path, backend):
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

        return SoupConfig(
            base="meta-llama/Llama-3.1-8B-Instruct",
            task="sft",
            backend=backend,
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(),
            output=str(tmp_path),
        )

    def test_mlx_backend_reaches_the_resolver_as_mlx(self, tmp_path, monkeypatch):
        import soup_cli.commands.train as train_module

        calls = []

        def _spy(resume, output_dir, experiment_name=None, *, backend="transformers"):
            calls.append(backend)
            return str(tmp_path / "0000100_adapters.safetensors")

        monkeypatch.setattr(train_module, "_resolve_checkpoint", _spy)

        result = train_module._resolve_resume_or_exit("auto", self._cfg(tmp_path, "mlx"))

        assert calls == ["mlx"], (
            "the CLI must pass backend=cfg.backend through to _resolve_checkpoint - "
            "dropping that keyword is the exact #634 regression this pins"
        )
        assert result == str(tmp_path / "0000100_adapters.safetensors")

    def test_transformers_backend_reaches_the_resolver_as_transformers(
        self, tmp_path, monkeypatch
    ):
        """Negative control: the default backend must still say so, not
        "mlx" leaking in from a hardcoded value."""
        import soup_cli.commands.train as train_module

        calls = []

        def _spy(resume, output_dir, experiment_name=None, *, backend="transformers"):
            calls.append(backend)
            return str(tmp_path / "checkpoint-100")

        monkeypatch.setattr(train_module, "_resolve_checkpoint", _spy)

        train_module._resolve_resume_or_exit("auto", self._cfg(tmp_path, "transformers"))

        assert calls == ["transformers"]

    def test_no_resume_requested_skips_the_resolver_entirely(self, tmp_path):
        import soup_cli.commands.train as train_module

        result = train_module._resolve_resume_or_exit(None, self._cfg(tmp_path, "mlx"))

        assert result is None

    def test_unresolvable_checkpoint_exits_rather_than_returning(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.train as train_module

        monkeypatch.setattr(train_module, "_resolve_checkpoint", lambda *a, **k: None)

        with pytest.raises(typer.Exit):
            train_module._resolve_resume_or_exit("auto", self._cfg(tmp_path, "mlx"))
