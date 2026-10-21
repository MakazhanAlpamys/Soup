"""Portable packing token/target/position and refusal contracts."""

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.mlx_packing import PackedDataset, validate_packing_config


def test_ordered_exact_fit_tail_and_distinct_labels():
    rows = [([2, 3, 4], [7, 3, 4]), ([2, 5, 6], [-100, 5, 6]),
            ([2, 7, 8, 9, 1], [-100, -100, 10, 9, 1])]
    packs = PackedDataset(rows, kind="labels", max_length=6, vocab_size=16)
    assert packs[0] == ([2, 3, 4, 2, 5, 6], [-100, 3, 4, -100, 5, 6],
                        [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
    assert packs[1] == ([2, 7, 8, 9, 1], [-100, -100, 10, 9, 1], [2]*5, [0, 1, 2, 3, 4])
    assert packs.metadata == {
        "strategy": "ordered-next-fit-isolated-llama", "records": 3, "packs": 2,
        "supervised_tokens": 7, "input_tokens": 11, "capacity_tokens": 12,
        "utilization": 11/12,
    }
    assert rows[0][1][0] == 7  # Ignored standalone causal position doesn't mutate input.
    retained = [value for _, labels in rows for value in labels[1:] if value != -100]
    assert [value for i in range(len(packs)) for value in packs[i][1] if value != -100] == retained


@pytest.mark.parametrize("kind,target", [("mask", [False, True, False]), ("span", 1)])
def test_masks_spans_padding_and_boundary_targets(kind, target):
    rows = [([2, 1, 1], target), ([2, 1, 1], target)]  # PAD=EOS=1
    packs = PackedDataset(rows, kind=kind, max_length=6, vocab_size=8)
    expected = [-100, 1, -100] if kind == "mask" else [-100, 1, 1]
    assert packs[0][0] == [2, 1, 1, 2, 1, 1]
    assert packs[0][1] == expected * 2
    assert packs[0][3] == [0, 1, 2] * 2


def test_individual_truncation_preserves_target_once():
    row = ([2, 3, 4, 5, 6, 7], [-100, -100, -100, -100, 6, 7])
    pack = PackedDataset([row], kind="labels", max_length=5, vocab_size=8)
    assert pack[0] == ([2, 3, 4, 5, 6], [-100, -100, -100, -100, 6], [0]*5, list(range(5)))
    with pytest.raises(ValueError, match="causal-loss target"):
        PackedDataset([row], kind="labels", max_length=4, vocab_size=8)


@pytest.mark.parametrize("rows,kind,reason", [
    ([], "labels", "nonempty"),
    ([([], [])], "labels", "nonempty"),
    ([([2, 3], [-100, -100])], "labels", "causal-loss target"),
    ([([2, 3], [-100])], "labels", "length"),
    ([([True, 3], [-100, 3])], "labels", "integer"),
    ([([2, 9], [-100, 3])], "labels", "vocabulary"),
    ([([2, 3], [0, 2])], "mask", "binary"),
    ([([2, 3], [0])], "mask", "length"),
    ([([2, 3], True)], "span", "nonnegative integer"),
    ([([2, 3], -1)], "span", "nonnegative integer"),
])
def test_malformed_or_unsupervised_records_refused(rows, kind, reason):
    with pytest.raises(ValueError, match=reason):
        PackedDataset(rows, kind=kind, max_length=8, vocab_size=8)


def config(**training):
    return SoupConfig(base="local-base", backend="mlx", data={"train": "train.jsonl"},
                      training={"quantization": "none", "packing": True, **training})


@pytest.mark.parametrize("training,reason", [
    ({"batch_size": 2}, "batch_size"),
    ({"gradient_checkpointing": True}, "gradient_checkpointing"),
    ({"lora": {"use_dora": True}}, "ordinary LoRA"),
    ({"lora": {"rank_pattern": {"q_proj": 4}}}, "ordinary LoRA"),
    ({"lora": {"alpha_pattern": {"q_proj": 4}}}, "ordinary LoRA"),
])
def test_exact_configuration_guard_with_valid_control(training, reason):
    validate_packing_config(config())
    with pytest.raises(ValueError, match=reason):
        validate_packing_config(config(**training))


def test_streaming_guard_and_valid_control():
    cfg = config()
    validate_packing_config(cfg)
    payload = cfg.model_dump()
    payload["data"]["streaming"] = True
    cfg = SoupConfig.model_validate(payload)
    with pytest.raises(ValueError, match="streaming"):
        validate_packing_config(cfg)


@pytest.mark.parametrize("training,reason", [
    ({"batch_size": 2}, "packing requires batch_size"),
    ({"gradient_checkpointing": True}, "packing requires gradient_checkpointing"),
    ({"lora": {"use_dora": True}}, "packing supports ordinary LoRA"),
    ({"lora": {"rank_pattern": {"q_proj": 4}}}, "packing supports ordinary LoRA"),
    ({"lora": {"alpha_pattern": {"q_proj": 4}}}, "packing supports ordinary LoRA"),
])
def test_public_setup_reaches_each_new_refusal_before_model_loading(monkeypatch, training, reason):
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from tests.test_issue634_mlx_resume import _install_fake_mlx

    _install_fake_mlx(monkeypatch)

    def must_not_load(*args, **kwargs):
        raise AssertionError("packing refusal must precede model loading")

    monkeypatch.setattr("soup_cli.utils.mlx.load_mlx_model", must_not_load)
    with pytest.raises(ValueError, match=reason):
        MLXSFTTrainerWrapper(config(**training)).setup({"train": [{"text": "x"}] * 2})
