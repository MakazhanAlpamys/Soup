"""Complete tokenizer identity with real tokenization pipelines, no finite probes."""

import copy

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from soup_cli.trainer.mlx_distill_data import compatibility_identity  # noqa: E402


def tokenizer():
    backend = Tokenizer(models.WordLevel({
        "<unk>": 0, "<eos>": 1, "<pad>": 2, "alpha": 3, "beta": 4,
    }, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.normalizer = normalizers.NFKC()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
        chat_template="{% for m in messages %}{{ m['content'] }} <eos> {% endfor %}",
    )


def test_identical_tokenizers_ignore_model_name_and_allow_matching_padded_outputs():
    student, teacher = tokenizer(), tokenizer()
    student.name_or_path, teacher.name_or_path = "small-student", "larger-teacher"
    evidence = compatibility_identity(student, teacher, 8, 8)
    assert evidence["named_tokens"] == 5
    assert evidence["output_vocabulary_size"] == 8
    assert len(evidence["sha256"]) == 64


@pytest.mark.parametrize("change", [
    "permuted-ids", "normalizer", "pre-tokenizer", "added-token", "special-token",
    "template", "output-dimension", "out-of-range", "unverifiable",
])
def test_equal_lengths_or_probe_text_cannot_hide_incompatibility(change):
    student = tokenizer()
    teacher = copy.deepcopy(student)
    vocab = 8
    assert compatibility_identity(student, teacher, vocab, vocab)["named_tokens"] == 5
    if change == "permuted-ids":
        teacher.backend_tokenizer.model = models.WordLevel({
            "<unk>": 0, "<eos>": 1, "<pad>": 2, "alpha": 4, "beta": 3,
        }, unk_token="<unk>")
        assert len(teacher) == len(student)
    elif change == "normalizer":
        teacher.backend_tokenizer.normalizer = normalizers.Lowercase()
    elif change == "pre-tokenizer":
        teacher.backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    elif change == "added-token":
        student.add_tokens([AddedToken("special", lstrip=False)])
        teacher.add_tokens([AddedToken("special", lstrip=True)])
        assert len(teacher) == len(student)
    elif change == "special-token":
        teacher.pad_token = "<eos>"
    elif change == "template":
        teacher.chat_template += "extra"
    elif change == "output-dimension":
        vocab = 9
    elif change == "out-of-range":
        student.add_tokens([f"extra{i}" for i in range(5)])
        teacher = copy.deepcopy(student)
    else:
        teacher = object()
    with pytest.raises(ValueError, match="MLX distillation.*tokenizer|output vocabulary"):
        compatibility_identity(student, teacher, 8, vocab)
