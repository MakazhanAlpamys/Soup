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

Caveat that matters more here than usual: MLX only runs on Apple Silicon,
so the parts of this fix that call into mlx-lm's real training loop
(``mlx_sft.MLXSFTTrainerWrapper.train()`` end to end) cannot be executed in
this environment and are NOT covered by a running test below — only
inspected at the source level. What IS executed and asserted: the
checkpoint-path resolution (pure filesystem logic, no MLX import needed),
and ``_load_checkpoint_weights`` in isolation against a fake model object
duck-typing ``load_weights``.
"""

from __future__ import annotations

import inspect


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
    max(), not by sorting whatever order the filesystem enumerated files in.
    The list passed in here is explicit and adversarial — the highest step
    is neither first nor last — so a key/sort mistake can't coincidentally
    still return the right file the way it could if this only iterated a
    real directory in on-disk order."""

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


class TestLoadCheckpointWeights:
    """MLXSFTTrainerWrapper._load_checkpoint_weights, tested against a fake
    model object — no real mlx/mlx-lm import is reachable here."""

    def test_it_calls_load_weights_non_strict(self, tmp_path):
        calls = []

        class _FakeModel:
            def load_weights(self, path, strict=True):
                calls.append((path, strict))

        wrapper = _mlx_wrapper(tmp_path)
        wrapper.model = _FakeModel()

        wrapper._load_checkpoint_weights("/some/0000100_adapters.safetensors")

        assert calls == [("/some/0000100_adapters.safetensors", False)]


class TestTrainWiresCheckpointLoadingAfterLoraIsApplied:
    """train() itself calls into real mlx-lm and cannot run without Apple
    Silicon; inspected at the source level instead of executed. This is a
    real limitation, not a substitute for running it — flagged as such in
    the PR."""

    def test_the_old_no_op_warning_is_gone(self):
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        source = inspect.getsource(MLXSFTTrainerWrapper.train)
        assert "does not support --resume yet" not in source

    def test_checkpoint_loading_is_called_after_lora_is_applied(self):
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        source = inspect.getsource(MLXSFTTrainerWrapper.train)
        lora_pos = source.index("self._apply_lora(")
        load_pos = source.index("self._load_checkpoint_weights(")
        assert lora_pos < load_pos, (
            "_load_checkpoint_weights must run after _apply_lora — the saved "
            "file holds only LoRA-shaped tensors, which don't exist on the "
            "model until the linear layers are converted"
        )

    def test_checkpoint_loading_is_skipped_when_none_is_given(self):
        """No resume requested -> no _load_checkpoint_weights call, no
        change in behaviour from before this fix."""
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        source = inspect.getsource(MLXSFTTrainerWrapper.train)
        assert "if resume_from_checkpoint is not None:" in source
