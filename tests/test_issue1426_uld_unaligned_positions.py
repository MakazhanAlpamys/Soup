"""Aligned ULD must skip student positions with no teacher target (#1426).

``uld_aligned_loss`` mean-pools teacher logits onto student positions through
a character alignment. A student position that aligns to no teacher token
(a byte-fallback piece decoding to U+FFFD, an empty piece, a special token)
got an all-zero teacher row, which softmaxes to uniform. It still sat in the
loss mask, so each one added about ``V/2`` to the loss while aligned
positions scored near zero.

A position now counts only when its alignment names an existing teacher
position, in the sum and in the denominator, under every mask (labels,
attention mask, none). The trainer decodes student special tokens to ``""``
as it already does for the teacher, so they align to nothing and stay out.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

V = 32000

# (student tokens, teacher tokens); the middle student piece has no teacher
# counterpart in the two #1426 cases.
_CASES = {
    "identical tokens (control)": (["a", "b", "c"], ["a", "b", "c"]),
    "byte-fallback piece (U+FFFD)": (["caf", "�", " ok"], ["café", " ok"]),
    "empty decoded piece": (["a", "", "c"], ["a", "c"]),
}


def _cfg(vocab: int = V):
    from soup_cli.utils.uld import ULDConfig

    return ULDConfig(
        strategy="wasserstein_aligned",
        student_vocab_size=vocab,
        teacher_vocab_size=vocab,
    )


def _peaked(n_pos: int, hot: int = 0, vocab: int = V):
    logits = torch.full((1, n_pos, vocab), -10.0)
    logits[..., hot] = 10.0
    return logits


def _loss(s_tok, t_tok, student=None, teacher=None, **masks):
    from soup_cli.utils.uld import uld_aligned_loss

    student = _peaked(len(s_tok)) if student is None else student
    teacher = _peaked(len(t_tok)) if teacher is None else teacher
    return uld_aligned_loss(
        student, teacher, [s_tok], [t_tok], config=_cfg(), **masks
    )


class TestUnalignedPositionsLeaveTheLoss:
    @pytest.mark.parametrize("name", list(_CASES))
    def test_issue_cases_score_zero_with_labels(self, name):
        s_tok, t_tok = _CASES[name]
        labels = torch.zeros((1, len(s_tok)), dtype=torch.long)
        assert float(_loss(s_tok, t_tok, labels=labels)) == 0.0

    @pytest.mark.parametrize("name", list(_CASES))
    def test_issue_cases_score_zero_with_attention_mask_only(self, name):
        s_tok, t_tok = _CASES[name]
        mask = torch.ones((1, len(s_tok)), dtype=torch.long)
        assert float(_loss(s_tok, t_tok, attention_mask=mask)) == 0.0

    @pytest.mark.parametrize("name", list(_CASES))
    def test_issue_cases_score_zero_without_any_mask(self, name):
        s_tok, t_tok = _CASES[name]
        assert float(_loss(s_tok, t_tok)) == 0.0

    def test_disagreement_on_an_aligned_position_still_counts(self):
        s_tok, t_tok = _CASES["empty decoded piece"]
        labels = torch.zeros((1, 3), dtype=torch.long)
        # A flatter student at position 0 (aligned to teacher 0) must move it.
        student = _peaked(3)
        student[0, 0] = 0.0
        assert float(_loss(s_tok, t_tok, student=student, labels=labels)) > 0.1

    def test_mean_is_over_aligned_positions_only(self):
        """The unaligned position leaves the denominator too, not just the sum."""
        s_tok, t_tok = ["a", "", "c", "d"], ["a", "c", "d"]
        student = _peaked(4)
        student[0, 0] = 0.0  # flat student at aligned position 0
        labels = torch.zeros((1, 4), dtype=torch.long)
        # Causal shift keeps positions 0..2; position 1 is unaligned.
        got = _loss(s_tok, t_tok, student=student, labels=labels)

        aligned_only = _loss(
            ["a", "c"], ["a", "c"],
            student=student[:, [0, 2, 3]], teacher=_peaked(2),
            labels=torch.zeros((1, 3), dtype=torch.long),
        )
        # Same two supervised targets (positions 0 and 2 of the original).
        torch.testing.assert_close(got, aligned_only)
        assert float(got) > 0.1

    def test_all_supervised_positions_unaligned_is_finite_zero(self):
        s_tok, t_tok = ["�", "", "�"], ["x"]
        labels = torch.zeros((1, 3), dtype=torch.long)
        loss = _loss(s_tok, t_tok, labels=labels)
        assert torch.isfinite(loss)
        assert float(loss) == 0.0

    def test_alignment_to_a_missing_teacher_index_is_not_a_target(self):
        from soup_cli.utils.uld import aligned_position_mask

        assert aligned_position_mask([[0], [], [5], [-1], [1, 9]], 2) == [
            True, False, False, False, True,
        ]

    def test_gradient_reaches_only_aligned_positions(self):
        s_tok, t_tok = _CASES["byte-fallback piece (U+FFFD)"]
        student_flat = _peaked(3)
        student_flat[0, :, 0] = 0.0
        student_flat.requires_grad_(True)
        labels = torch.zeros((1, 3), dtype=torch.long)
        _loss(s_tok, t_tok, student=student_flat, labels=labels).backward()
        # Positions 0 and 1 are kept by the causal shift; 1 is unaligned.
        assert student_flat.grad[0, 0].abs().sum() > 0
        assert student_flat.grad[0, 1].abs().sum() == 0


# --- trainer level: student special tokens never reach the ULD loss --------


class _CharTokenizer:
    """One token per character; ``special`` ids decode to ``""`` when skipped."""

    def __init__(self, chars: str, special: dict[int, str] | None = None):
        self.special = dict(special or {})
        self.vocab = dict(self.special)
        for c in chars:
            self.vocab[len(self.vocab)] = c
        self.ids = {v: k for k, v in self.vocab.items() if k not in self.special}

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids, skip_special_tokens: bool = False):
        return "".join(
            "" if (skip_special_tokens and int(i) in self.special) else self.vocab[int(i)]
            for i in ids
        )

    def batch_decode(self, rows, skip_special_tokens: bool = False):
        return [self.decode(r, skip_special_tokens=skip_special_tokens) for r in rows]

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True,
                 max_length=None):
        rows = [[self.ids[c] for c in t][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _PeakedLM(torch.nn.Module):
    """Every position puts its mass on token 0, whatever the input."""

    def __init__(self, vocab: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab))
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab), -10.0)
        logits[..., 0] = 10.0
        return types.SimpleNamespace(logits=logits + self.bias)


def test_trainer_keeps_student_special_tokens_out_of_aligned_uld(monkeypatch, tmp_path):
    """End to end: a student BOS/EOS the teacher never sees adds nothing.

    Student and teacher agree on every real token, so the only thing that can
    make the distillation term non-zero is a student-only special token being
    scored against a uniform teacher row.
    """
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    import soup_cli.utils.uld as uld
    from tests.test_issue720_distill_grad_accum import _compile_distill_trainer

    vocab = 64
    student_tok = _CharTokenizer("abc", special={0: "<s>", 1: "</s>"})
    teacher_tok = _CharTokenizer("abc")

    seen = {}
    real_loss = uld.uld_aligned_loss

    def spy(*args, **kwargs):
        seen["student_tokens"] = args[2]
        seen["loss"] = real_loss(*args, **kwargs)
        return seen["loss"]

    monkeypatch.setattr(uld, "uld_aligned_loss", spy)

    trainer_cls = _compile_distill_trainer()
    trainer_cls.compute_loss.__globals__.update({
        "_sequence_mode": False,
        "_minillm_on_policy": False,
        "_uld_aligned": True,
        "_uld_teacher_tokenizer": teacher_tok,
        "_student_tokenizer": student_tok,
        "teacher_ref": _PeakedLM(vocab).eval(),
        "_uld_projection": types.SimpleNamespace(config=_cfg(vocab)),
        "_minillm_cb": None,
        "_CE_WEIGHT": 0.5,
        "_DISTILL_WEIGHT": 0.5,
    })
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "uld1426"), report_to=[], use_cpu=True
    )
    model = _PeakedLM(vocab)
    trainer = trainer_cls(model=model, args=args)

    # <s> a b c </s> a b: specials at the start and in the middle.
    ids = torch.tensor([[0, 2, 3, 4, 1, 2, 3]])
    batch = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
    trainer.compute_loss(model, batch)

    assert seen["student_tokens"] == [["", "a", "b", "c", "", "a", "b"]]
    assert float(seen["loss"]) == 0.0
