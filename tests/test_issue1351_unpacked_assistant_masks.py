"""Tests for issue #1351: unpacked SFT runs keep the ``assistant_masks`` column.

#1304 added ``_ensure_assistant_masks`` and calls it on every SFT split, packed
or not (``trainer/sft.py:915-918``). Only the packed call is covered: every case
in ``tests/test_issue1236_packing_loss_mask.py`` compares packed with unpacked
counts, and the unpacked path still carries ``labels``, so gating both calls on
``tcfg.packing`` keeps that file green.

The unpacked call is what ``use_liger: true`` rides on. When Soup's Liger patch
lands it sets ``use_liger_kernel=True`` (``trainer/sft.py:1064-1065``), and trl
0.29.1's ``SFTTrainer._prepare_dataset`` then keeps only ``input_ids``,
``seq_lengths``, ``completion_mask`` and ``assistant_masks``
(``trl/trainer/sft_trainer.py:1179-1182``). ``labels`` is dropped, so without
``assistant_masks`` the collator supervises the system and user tokens too.

The one Liger setup test (``tests/test_v07300.py:397-400``) skips without
``liger_kernel``, which ``[dev]`` does not install. Neither case below needs it:
the column-selection branch reads ``args.use_liger_kernel``, not the package.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from tests.test_issue1236_packing_loss_mask import _build_sft_wrapper  # noqa: E402

_ROW = {
    "messages": [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "world reply"},
    ]
}


def _assert_mask_matches_labels(dataset) -> None:
    """The mask is 1 exactly where ``labels`` is supervised, and some tokens are masked."""
    for masks, labels in zip(dataset["assistant_masks"], dataset["labels"]):
        assert masks == [int(int(x) != -100) for x in labels]
        assert 0 < sum(masks) < len(masks), masks


class TestUnpackedAssistantMasks:
    def test_unpacked_train_and_eval_carry_assistant_masks(self, tmp_path, monkeypatch):
        """Both splits keep the column, and its content, when ``packing`` is false."""
        wrapper = _build_sft_wrapper(tmp_path, monkeypatch, packing=False)
        wrapper.setup({"train": [_ROW] * 8, "val": [_ROW] * 4})

        assert "assistant_masks" in wrapper.trainer.train_dataset.column_names
        assert wrapper.trainer.eval_dataset is not None
        assert "assistant_masks" in wrapper.trainer.eval_dataset.column_names
        _assert_mask_matches_labels(wrapper.trainer.train_dataset)
        _assert_mask_matches_labels(wrapper.trainer.eval_dataset)

    def test_trl_liger_column_selection_keeps_the_mask(self, tmp_path, monkeypatch):
        """trl's real Liger column selection, then the trainer's own collator.

        Driven through ``_prepare_dataset`` rather than a reproduction of it, in
        the manner of ``tests/test_issue791_preprocess_cache_eos.py:289``, so the
        pin follows trl's actual column list. The expected positions come from
        the ``labels`` column, which nothing on this path derives from
        ``assistant_masks``.
        """
        from types import SimpleNamespace

        from trl import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer

        wrapper = _build_sft_wrapper(tmp_path, monkeypatch, packing=False)
        wrapper.setup({"train": [_ROW] * 8})
        train = wrapper.trainer.train_dataset
        expected_labels = [[int(x) for x in row] for row in train["labels"]]
        expected = sum(x != -100 for row in expected_labels for x in row)
        total = sum(len(row) for row in expected_labels)
        assert 0 < expected < total, (expected, total)

        args = SFTConfig(
            output_dir=str(tmp_path / "liger"),
            max_length=128,
            packing=False,
            shuffle_dataset=False,  # keep row order
            dataset_num_proc=None,  # no multiprocessing (Windows spawn re-imports)
            use_cpu=True,
            report_to=[],
        )
        args.use_liger_kernel = True  # this branch reads the flag, not the package

        prepared = SFTTrainer._prepare_dataset(
            SimpleNamespace(_is_vlm=False),
            train,
            wrapper.trainer.processing_class,
            args,
            False,
            None,
            "train",
        )
        assert "labels" not in prepared.column_names, prepared.column_names

        batch = wrapper.trainer.data_collator([prepared[i] for i in range(len(prepared))])
        supervised = int((batch["labels"] != -100).sum())
        assert supervised == expected, (
            f"Liger column selection supervises {supervised} positions, "
            f"the labels column {expected}"
        )
        for index, row in enumerate(expected_labels):
            assert batch["labels"][index, : len(row)].tolist() == row, index
