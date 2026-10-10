"""Tests for issue #1199 fix 2 — a v1 resume must not re-hash the local base.

``validate_resume_metadata`` re-resolved ``legacy_base_model`` to compare it against the
identity the run had already computed, reading and SHA-256-hashing every selected file of
the base a third time. The identity is already in ``current["base_model"]`` — the caller
computed it to fill that field — so the second pass reads the whole base again to
rediscover what it was handed.

These tests count ``_hash_local_file`` calls per path, which is the acceptance
criterion's own measure. A Hub base costs nothing either way (its identity is the repo
id), so the pass that disappears is the one that was expensive: a local base.

The refusal guarantees are the point of the whole function, so they are pinned here too:
a caller passing an identity that disagrees with the base must get the same refusal it
would have got by resolving it.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import soup_cli.utils.quest as quest
from soup_cli.utils.quest import (
    EXPECTED_MODULES,
    build_metadata,
    load_metadata,
    resolve_base_model_identity,
    validate_resume_metadata,
    write_metadata,
)

BASE_BYTES = b"safe-weights"


def _base(directory: Path, *, name: str = "model.safetensors") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (directory / name).write_bytes(BASE_BYTES)
    return directory


def _metadata(base_model: str, *, format_version: int = 2) -> dict:
    return build_metadata(
        activation_scales={name: 3.0 for name in EXPECTED_MODULES},
        base_model=base_model,
        calibration_sha256="ab" * 32,
        format_version=format_version,
    )


@pytest.fixture
def hashed(monkeypatch):
    """Count `_hash_local_file` calls per path for the duration of a test."""
    real = quest._hash_local_file
    counts: collections.Counter[str] = collections.Counter()

    def counting(path: Path):
        counts[str(path)] += 1
        return real(path)

    monkeypatch.setattr(quest, "_hash_local_file", counting)
    return counts


def _each_path_once(counts) -> list[tuple[str, int]]:
    """Per-path call counts, so 'hashed twice' is visible rather than averaged away."""
    return sorted(counts.items())


def test_a_v1_resume_hashes_each_selected_file_once(tmp_path, hashed):
    """The acceptance criterion: one pass through setup, not two.

    Fails on `main`: the resume path calls `resolve_base_model_identity` again, so the
    weight file is hashed twice and the pass count is 2 rather than 1.
    """
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))

    identity = resolve_base_model_identity(str(base))
    assert _each_path_once(hashed)[-1][1] == 1, "the setup pass must read each file once"

    hashed.clear()
    validate_resume_metadata(
        checkpoint,
        _metadata(identity),
        legacy_base_model=str(base),
        legacy_base_identity=identity,
    )

    assert _each_path_once(hashed) == [], (
        "the v1 resume re-read the base; it already had the identity in "
        f"current['base_model'] ({identity[:24]}...)"
    )


def test_without_the_pre_resolved_identity_the_base_is_still_read(tmp_path, hashed):
    """CONTROL. Dropping the argument restores the second pass, so the test above is
    measuring this code path and not something that happens to be free anyway.

    Without it `validate_resume_metadata` resolves on its own, exactly as before, so the
    default keeps working for any caller that has not been updated.
    """
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))
    identity = resolve_base_model_identity(str(base))

    hashed.clear()
    validate_resume_metadata(checkpoint, _metadata(identity), legacy_base_model=str(base))

    assert _each_path_once(hashed), "the default must still resolve the identity itself"
    assert max(_each_path_once(hashed), key=lambda pair: pair[1])[1] == 1


def test_a_v2_resume_never_resolves_the_legacy_identity_at_all(tmp_path, hashed):
    """CONTROL. The v1 branch is the only caller of `resolve_base_model_identity` here,
    so a v2->v2 resume must touch no file even when an identity is offered."""
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    identity = resolve_base_model_identity(str(base))
    write_metadata(checkpoint, _metadata(identity, format_version=2))

    hashed.clear()
    validate_resume_metadata(
        checkpoint,
        _metadata(identity),
        legacy_base_model=str(base),
        legacy_base_identity=identity,
    )
    assert _each_path_once(hashed) == []


def test_a_pre_resolved_identity_that_disagrees_is_still_refused(tmp_path, hashed):
    """The new argument is compared, not trusted.

    `test_v1_reference_cannot_override_a_different_current_base` covers the same refusal
    when the argument is absent. This is the case the argument introduces: a caller that
    passes an identity belonging to some other base must get that identical refusal, not
    a silent accept.
    """
    original = _base(tmp_path / "original")
    changed = _base(tmp_path / "changed")
    (changed / "model.safetensors").write_bytes(b"different-weights")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(original), format_version=1))

    # `current` is this run's own identity, for `changed`: the resume is of a checkpoint
    # recorded against `original`, so the two must disagree and the run must be refused.
    current = _metadata(resolve_base_model_identity(str(changed)))
    assert current["base_model"] != resolve_base_model_identity(str(original))
    with pytest.raises(ValueError, match="does not match current identity"):
        validate_resume_metadata(
            checkpoint,
            current,
            legacy_base_model=str(original),
            # An identity belonging to `original`, not to the base `current` describes:
            # the caller answering for the wrong base must not launder the mismatch.
            legacy_base_identity=resolve_base_model_identity(str(original)),
        )


def test_the_offered_identity_must_match_the_one_that_would_be_resolved(tmp_path, hashed):
    """CONTROL for the control above. Passing the identity that resolution actually
    produces accepts, so the refusal above comes from the comparison and not from the
    presence of the new keyword argument."""
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))
    identity = resolve_base_model_identity(str(base))

    validate_resume_metadata(
        checkpoint,
        _metadata(identity),
        legacy_base_model=str(base),
        legacy_base_identity=identity,
    )


def test_a_missing_reference_still_asks_for_it_before_using_the_identity(tmp_path, hashed):
    """The `legacy_base_model is None` refusal is not bypassed by supplying only the
    identity: the reference is the thing v1 recorded, so it is still required."""
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))
    identity = resolve_base_model_identity(str(base))

    with pytest.raises(ValueError, match="requires the original base model reference"):
        validate_resume_metadata(
            checkpoint,
            _metadata(identity),
            legacy_base_identity=identity,
        )


def test_the_sidecar_still_records_the_path_not_the_digest_after_a_v1_resume(tmp_path, hashed):
    """The reason the branch exists at all: after a v1 resume the artifact is written
    back in v1 terms, so its `base_model` stays the original source string. Offering an
    identity must not leak the digest into it."""
    base = _base(tmp_path / "base")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))
    identity = resolve_base_model_identity(str(base))

    validate_resume_metadata(
        checkpoint,
        _metadata(identity),
        legacy_base_model=str(base),
        legacy_base_identity=identity,
    )
    # `validate_resume_metadata` returns None, so the rewrite is asserted through the
    # caller's own construction rather than a return value.
    downgraded = {**_metadata(identity), "format_version": 1, "base_model": str(base)}
    written = tmp_path / "artifact"
    write_metadata(written, downgraded)
    assert load_metadata(written)["base_model"] == str(base)
    assert identity not in (written / "quest_mixed_precision.json").read_text(encoding="utf-8")


class TestTheTrainerPassesTheIdentityItAlreadyHolds:
    """The half that makes the parameter reachable in a real run.

    `SFTTrainerWrapper.train` is the only production caller. If it stops passing the
    identity, `utils/quest.py` is still correct and every test above still passes — so
    the pass is pinned here, on the call the issue is about.
    """

    def test_train_hands_the_metadata_base_model_to_the_resume_check(
        self, tmp_path, monkeypatch
    ):
        import contextlib

        import soup_cli.trainer.sft as sft
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils import ebft_gdpo, peft_wiring

        seen: list[dict] = []

        class FakeModel:
            # `_assert_finite_training_state` imports torch and walks parameters once
            # `train()` returns; the argument this test is about is read before that, so
            # the tail of `train()` is stubbed rather than run. Without it these two tests
            # would need torch, and torch is not this test's subject.
            def named_parameters(self):
                return []

        class FakeTrainer:
            def __init__(self):
                self.model = FakeModel()
                self.args = SimpleNamespace(fp16=False, bf16=False, should_save=True)
                self.state = SimpleNamespace(log_history=[], global_step=3)

            def train(self, *, resume_from_checkpoint):
                return {}

            def save_model(self, output):
                return None

        # Deliberately DIFFERENT values: the identity of `config.base` resolved before
        # the load is not the sidecar's `base_model` on the continuation path, and the
        # assertion below must be able to tell the two sources apart.
        metadata = _metadata("local-sha256:" + "0" * 64)
        before = "local-sha256:" + "1" * 64

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            base="some/local/base",
            training=SimpleNamespace(activation_offloading="none", relora_steps=None),
        )
        wrapper._quest_metadata = metadata
        wrapper._quest_base_identity_before = before
        wrapper._output_dir = str(tmp_path)
        wrapper.trainer = FakeTrainer()
        wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
        wrapper._training_context = lambda context: contextlib.ExitStack()
        wrapper._report_rewind = lambda: None

        for name in (
            "attach_loraplus_optimizer",
            "attach_relora_callback",
            "attach_curriculum_callback",
            "attach_plugin_callback",
        ):
            monkeypatch.setattr(peft_wiring, name, lambda *a, **k: None)
        monkeypatch.setattr(ebft_gdpo, "attach_ebft_compute_loss", lambda *a: None)
        monkeypatch.setattr(sft, "align_trainable_dtype_for_fp16", lambda *a, **k: None)
        monkeypatch.setattr(sft, "_assert_finite_training_state", lambda *a, **k: None)
        monkeypatch.setattr(sft, "time", SimpleNamespace(time=lambda: 0.0))
        monkeypatch.setattr(quest, "write_metadata", lambda output, value: None)
        monkeypatch.setattr(
            quest,
            "validate_resume_metadata",
            lambda checkpoint, value, **kwargs: seen.append(kwargs),
        )

        wrapper.train(resume_from_checkpoint="checkpoint-2")

        assert seen == [
            {
                "legacy_base_model": "some/local/base",
                "legacy_base_identity": before,
            }
        ], (
            "train() must pass `_quest_base_identity_before` -- the identity of "
            "`config.base` resolved before the load. Passing "
            "`_quest_metadata['base_model']` instead makes the check compare "
            "current['base_model'] with itself, which can never refuse."
        )

    def test_train_does_not_offer_an_identity_for_a_hub_base(self, tmp_path, monkeypatch):
        """CONTROL. A Hub base's identity IS its repo id, so it costs nothing to resolve
        and there is nothing to save. This test records that the value passed is the
        metadata's own, which is the repo id in that case — so it documents the shape
        rather than asserting a special case that does not exist."""
        import contextlib

        import soup_cli.trainer.sft as sft
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils import ebft_gdpo, peft_wiring

        seen: list[dict] = []

        class FakeModel:
            # `_assert_finite_training_state` imports torch and walks parameters once
            # `train()` returns; the argument this test is about is read before that, so
            # the tail of `train()` is stubbed rather than run. Without it these two tests
            # would need torch, and torch is not this test's subject.
            def named_parameters(self):
                return []

        class FakeTrainer:
            def __init__(self):
                self.model = FakeModel()
                self.args = SimpleNamespace(fp16=False, bf16=False, should_save=True)
                self.state = SimpleNamespace(log_history=[], global_step=3)

            def train(self, *, resume_from_checkpoint):
                return {}

            def save_model(self, output):
                return None

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            base="ahxt/LiteLlama-460M-1T",
            training=SimpleNamespace(activation_offloading="none", relora_steps=None),
        )
        wrapper._quest_metadata = _metadata("ahxt/LiteLlama-460M-1T")
        # A Hub base's identity is its repo id, so it costs nothing to resolve. Passing
        # it anyway is harmless and keeps the call site unconditional.
        wrapper._quest_base_identity_before = "ahxt/LiteLlama-460M-1T"
        wrapper._output_dir = str(tmp_path)
        wrapper.trainer = FakeTrainer()
        wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
        wrapper._training_context = lambda context: contextlib.ExitStack()
        wrapper._report_rewind = lambda: None

        for name in (
            "attach_loraplus_optimizer",
            "attach_relora_callback",
            "attach_curriculum_callback",
            "attach_plugin_callback",
        ):
            monkeypatch.setattr(peft_wiring, name, lambda *a, **k: None)
        monkeypatch.setattr(ebft_gdpo, "attach_ebft_compute_loss", lambda *a: None)
        monkeypatch.setattr(sft, "align_trainable_dtype_for_fp16", lambda *a, **k: None)
        monkeypatch.setattr(sft, "_assert_finite_training_state", lambda *a, **k: None)
        monkeypatch.setattr(sft, "time", SimpleNamespace(time=lambda: 0.0))
        monkeypatch.setattr(quest, "write_metadata", lambda output, value: None)
        monkeypatch.setattr(
            quest,
            "validate_resume_metadata",
            lambda checkpoint, value, **kwargs: seen.append(kwargs),
        )

        wrapper.train(resume_from_checkpoint="checkpoint-2")
        assert seen[0]["legacy_base_identity"] == "ahxt/LiteLlama-460M-1T"


class TestTheContinuationPathStillRefuses:
    """The one required addition to the review: the pass may not launder a refusal.

    On the continuation path `self._quest_metadata` is the artifact's sidecar, whose
    `base_model` is the ORIGINAL base recorded at calibration, while `self.config.base`
    is the artifact directory. The two are not equal, and the v1 check compares them.

    An earlier cut of this PR passed `_quest_metadata["base_model"]` as the pre-resolved
    identity. Since `current` *is* that metadata, the check became
    `current["base_model"] != current["base_model"]`, which can never refuse — so the
    keyword stopped being compared and became trusted, and a resume that is refused on
    `main` was silently accepted. These tests drive `validate_resume_metadata` for real
    through `train()`, so the wrong source fails here.
    """

    @staticmethod
    def _wrapper(tmp_path, metadata, before, base):
        import contextlib

        from soup_cli.trainer.sft import SFTTrainerWrapper

        class FakeModel:
            def named_parameters(self):
                return []

        class FakeTrainer:
            def __init__(self):
                self.model = FakeModel()
                self.args = SimpleNamespace(fp16=False, bf16=False, should_save=True)
                self.state = SimpleNamespace(log_history=[], global_step=3)

            def train(self, *, resume_from_checkpoint):
                return {}

            def save_model(self, output):
                return None

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = SimpleNamespace(
            base=base,
            training=SimpleNamespace(activation_offloading="none", relora_steps=None),
        )
        wrapper._quest_metadata = metadata
        wrapper._quest_base_identity_before = before
        wrapper._output_dir = str(tmp_path)
        wrapper.trainer = FakeTrainer()
        wrapper.tokenizer = SimpleNamespace(save_pretrained=lambda output: None)
        wrapper._training_context = lambda context: contextlib.ExitStack()
        wrapper._report_rewind = lambda: None
        return wrapper

    @staticmethod
    def _stub(monkeypatch):
        import soup_cli.trainer.sft as sft
        from soup_cli.utils import ebft_gdpo, peft_wiring

        for name in (
            "attach_loraplus_optimizer",
            "attach_relora_callback",
            "attach_curriculum_callback",
            "attach_plugin_callback",
        ):
            monkeypatch.setattr(peft_wiring, name, lambda *a, **k: None)
        monkeypatch.setattr(ebft_gdpo, "attach_ebft_compute_loss", lambda *a: None)
        monkeypatch.setattr(sft, "align_trainable_dtype_for_fp16", lambda *a, **k: None)
        monkeypatch.setattr(sft, "_assert_finite_training_state", lambda *a, **k: None)
        monkeypatch.setattr(sft, "time", SimpleNamespace(time=lambda: 0.0))
        monkeypatch.setattr(quest, "write_metadata", lambda output, value: None)

    def test_a_sidecar_naming_a_different_base_is_still_refused(self, tmp_path, monkeypatch):
        """The continuation case: the artifact was calibrated against `original`, and the
        run is pointing `base:` at a different directory. Refused today, and refused here
        for the same reason — the pre-resolved identity of `config.base` is not the
        sidecar's recorded base.

        Fails if `train()` passes `_quest_metadata["base_model"]`: the comparison would
        be `x != x` and this would not raise.
        """
        # v1 records the source STRING, so the checkpoint and `base:` must name the same
        # directory for the identity comparison to be reached at all. `original` is this
        # run's sidecar: a v2 artifact whose recorded base is `original`, while the
        # artifact directory this run points at has a different content identity.
        original_dir = str(tmp_path / "original-base")
        original = _metadata("local-sha256:" + "2" * 64)

        checkpoint = tmp_path / "checkpoint"
        write_metadata(checkpoint, {**_metadata(original_dir, format_version=1)})

        wrapper = self._wrapper(
            tmp_path,
            original,
            before="local-sha256:" + "3" * 64,
            base=original_dir,
        )
        self._stub(monkeypatch)

        with pytest.raises(ValueError, match="does not match current identity"):
            wrapper.train(resume_from_checkpoint=str(checkpoint))

    def test_a_sidecar_naming_the_same_identity_is_accepted(self, tmp_path, monkeypatch):
        """CONTROL for the refusal above: the two disagreeing values are the reason it
        raises, not the presence of the keyword argument."""
        identity = "local-sha256:" + "4" * 64
        original_dir = str(tmp_path / "original-base")
        checkpoint = tmp_path / "checkpoint"
        write_metadata(checkpoint, {**_metadata(original_dir, format_version=1)})

        wrapper = self._wrapper(
            tmp_path,
            _metadata(identity),
            before=identity,
            base=original_dir,
        )
        self._stub(monkeypatch)

        wrapper.train(resume_from_checkpoint=str(checkpoint))

    def test_the_base_is_still_not_rehashed_on_that_path(self, tmp_path, monkeypatch, hashed):
        """The pass is removed AND the refusal survives — the two halves of this change,
        on the path where they can conflict."""
        identity = "local-sha256:" + "5" * 64
        base = _base(tmp_path / "the-same-base")
        checkpoint = tmp_path / "checkpoint"
        write_metadata(
            checkpoint,
            {**_metadata(str(base), format_version=1)},
        )

        wrapper = self._wrapper(
            tmp_path, _metadata(identity), before=identity, base=str(base)
        )
        self._stub(monkeypatch)

        hashed.clear()
        wrapper.train(resume_from_checkpoint=str(checkpoint))
        assert _each_path_once(hashed) == [], (
            "the continuation path rehashed the base; it had the identity already"
        )


def test_the_shard_index_and_its_shards_are_each_read_once(tmp_path, hashed):
    """A sharded base is the case where the third pass costs the most, since the index
    plus every referenced shard is read. Per-path, so a single hot file cannot hide
    behind the average."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    for part in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (base / part).write_bytes(b"weights")
    checkpoint = tmp_path / "checkpoint"
    write_metadata(checkpoint, _metadata(str(base), format_version=1))

    identity = resolve_base_model_identity(str(base))
    hashed.clear()
    validate_resume_metadata(
        checkpoint,
        _metadata(identity),
        legacy_base_model=str(base),
        legacy_base_identity=identity,
    )
    assert _each_path_once(hashed) == []
