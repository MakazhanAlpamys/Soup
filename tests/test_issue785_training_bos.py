"""#785 — train_on_responses_only=false trained on a doubled BOS.

The default SFT path (``train_on_responses_only``) builds ids with
``add_special_tokens=False`` in ``data/loss_mask.py``, so it never doubled the
BOS. The opt-out text path did not: ``data/sft_format.py`` handed TRL a
``{"text"}`` column, and TRL 0.29.1's language-modeling ``tokenize_fn``
(``sft_trainer.py`` ``processing_class(text=...)``) re-tokenised it with the
tokenizer's default ``add_special_tokens=True``. A template that renders
``{{ bos_token }}`` (Llama-3 / Gemma / Mistral) on a tokenizer whose
post-processor also prepends BOS therefore trained on ``[bos, bos, ...]``.
After #782 fixed inference to send exactly one BOS, those adapters are one
token off from how they are prompted.

The invariant pinned here mirrors #781/#782 on the training side: when a
template renders the sequence, the tokenizer adds nothing on top. That is what
``apply_chat_template(tokenize=True)`` does and what inference now does. It is
deliberately NOT "drop a duplicate BOS": Soup's ``data.chat_template`` presets
render no BOS at all, so their adapters must train on none.

Every tokenizer below is a real ``transformers`` fast tokenizer built offline
from an in-memory vocab, so ``apply_chat_template`` is the genuine Jinja
renderer and the BOS/EOS come from a genuine ``tokenizers`` post-processor.
"""

import hashlib

import pytest

_SPECIALS = [
    "<unk>", "<s>", "</s>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|im_start|>", "<|im_end|>",
]
_WORDS = [
    "You", "are", "terse", ".", "What", "is", "the", "capital", "of", "France",
    "?", "Paris", "system", "user", "assistant",
]
_BOS_ID = _SPECIALS.index("<s>")

_BOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
# Renders the tokenizer's EOS itself, so the template is the source of the stop token.
_BOS_EOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {{ eos_token }} {% endfor %}"
)

_USER = {"role": "user", "content": "What is the capital of France?"}
_ANSWER = {"role": "assistant", "content": "Paris ."}
_MESSAGES = [_USER, _ANSWER]


def _tokenizer(chat_template, *, post_processor="bos"):
    """A real fast tokenizer whose post-processor adds ``post_processor``."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    single = {"bos": "<s> $A", "bos_eos": "<s> $A </s>", None: None}[post_processor]
    if single is not None:
        backend.post_processor = processors.TemplateProcessing(
            single=single,
            special_tokens=[("<s>", _BOS_ID), ("</s>", _SPECIALS.index("</s>"))],
        )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = chat_template
    return tok


def _legacy_format_row(tok):
    """The ``train_on_responses_only=false`` row builder."""
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.sft_format import build_format_row

    dcfg = DataConfig(
        train="train.jsonl",
        train_on_responses_only=False,
        train_on_messages_with_train_field=False,
        max_length=2048,
    )
    return build_format_row(tokenizer=tok, data_cfg=dcfg)


class TestLegacyTextPathBOS:
    def test_bos_template_trains_on_exactly_one_bos(self):
        """A template that renders {{ bos_token }} must train on one BOS, and the
        ids must equal HF's own one-step encode (what inference sends post-#782).

        Fails on pre-fix main: that path returned a ``{"text"}`` row, so there is
        no ``input_ids`` to read, and TRL then tokenized the text to two BOS.
        """
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        row = _legacy_format_row(tok)({"messages": _MESSAGES})

        assert "input_ids" in row, (
            "legacy path must pre-tokenize so TRL never re-adds special tokens"
        )
        ids = row["input_ids"]
        assert ids.count(_BOS_ID) == 1, f"expected one BOS, got ids={ids}"

        reference = tok.apply_chat_template(
            _MESSAGES, tokenize=True, add_generation_prompt=False, return_dict=True
        )["input_ids"]
        assert ids == list(reference)
        # Full-sequence training: every token is a target, none masked.
        assert row["labels"] == ids
        assert set(row["attention_mask"]) == {1}

    def test_the_prefix_encoding_would_have_doubled_the_bos(self):
        """Document the defect: the pre-fix mechanism (render to text, then let
        the tokenizer add its defaults) yields two BOS on the same tokenizer."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        text = tok.apply_chat_template(
            _MESSAGES, tokenize=False, add_generation_prompt=False
        )
        pre_fix_ids = tok(text=text)["input_ids"]  # TRL's default add_special_tokens=True
        assert pre_fix_ids.count(_BOS_ID) == 2

        fixed_ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert fixed_ids.count(_BOS_ID) == 1
        assert fixed_ids != pre_fix_ids

    def test_preset_template_trains_on_zero_bos(self):
        """Soup's data.chat_template presets render no BOS; the path must add
        none, even on a tokenizer whose post-processor prepends one."""
        from soup_cli.data.chat_templates import apply_chat_template_override

        tok = _tokenizer(None, post_processor="bos")
        apply_chat_template_override(tok, "chatml")
        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids.count(_BOS_ID) == 0, f"chatml preset must add no BOS, got {ids}"


class TestTrainingEOSPreserved:
    """The BOS fix must not cost the stop token (maintainer req 1 on #785).

    On the training side, dropping a tokenizer-appended trailing EOS is the
    opposite of #782's inference goal: it teaches run-on generation. Behaviour
    stated per case below.
    """

    _EOS_ID = _SPECIALS.index("</s>")

    def test_eos_preserved_when_template_renders_none_and_tokenizer_appends(self):
        """Template renders no EOS, tokenizer's post-processor appends one.

        Pre-fix (render then tokenize with defaults) kept that appended EOS.
        add_special_tokens=False alone would drop it; the fix re-appends it, so
        the trained EOS count is unchanged and the row still ends on the stop
        token."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=False)
        pre_fix_eos = tok(text=text)["input_ids"].count(self._EOS_ID)

        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids.count(self._EOS_ID) == pre_fix_eos == 1
        assert ids[-1] == self._EOS_ID, "row must end on the stop token"
        assert ids.count(_BOS_ID) == 1, "and still carry exactly one BOS"

    def test_eos_unchanged_when_template_renders_it(self):
        """Template renders the EOS itself; the tokenizer appends none.

        The template's EOS carries through untouched and none is re-added, so the
        trained EOS count is unchanged from the pre-fix path and only the leading
        duplicate BOS is removed."""
        tok = _tokenizer(_BOS_EOS_TEMPLATE, post_processor="bos")
        text = tok.apply_chat_template(_MESSAGES, tokenize=False, add_generation_prompt=False)
        pre_fix = tok(text=text)["input_ids"]

        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids.count(self._EOS_ID) == pre_fix.count(self._EOS_ID)
        assert ids[-1] == self._EOS_ID, "row still ends on the stop token"
        assert ids == pre_fix[1:], "only diff is the removed leading BOS"
        assert ids.count(_BOS_ID) == 1

    def test_no_eos_invented_when_neither_side_provides_one(self):
        """Template renders no EOS and the tokenizer appends none: pre-fix had no
        stop token, and the fix invents none (unchanged)."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids.count(self._EOS_ID) == 0


class TestTemplatedOnly:
    """The rule applies only to text a chat template rendered (maintainer req 2)."""

    def test_no_template_is_rejected_not_silently_stripped(self):
        """A tokenizer with no chat_template never reaches add_special_tokens=False:
        the text path raises rather than silently dropping the tokenizer's own
        (only) special tokens. Same behaviour as main (it raised before too)."""
        from soup_cli.data.loss_mask import build_full_sequence_labels

        tok = _tokenizer(None, post_processor="bos")  # no chat_template
        tok.chat_template = None
        with pytest.raises(ValueError, match="chat_template"):
            build_full_sequence_labels(_MESSAGES, tok, max_length=2048)


class TestPreprocessCachePathEOS:
    """`soup data preprocess` must handle EOS exactly like live training.

    The BOS fix (#785) lives on two paths: ``loss_mask.build_full_sequence_labels``
    (live tokenize) and ``preprocess_dataset`` (the AOT cache read straight into
    the trainer by ``sft._maybe_load_pretokenized``). Both now tokenize the
    rendered template with ``add_special_tokens=False``, which also drops the
    tokenizer post-processor's EOS. #788 review (AmirF194): the cache path did not
    re-append it, so a BOS-but-not-EOS template silently trained on a missing stop
    token from the cache. These pin the two paths to identical ids.
    """

    _EOS_ID = _SPECIALS.index("</s>")

    def _run_preprocess(self, tmp_path, monkeypatch, tok, task="sft"):
        transformers = pytest.importorskip("transformers")
        datasets = pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            f"base: x/y\ntask: {task}\n"
            "data:\n  train: ./d.jsonl\n  format: chatml\n  max_length: 2048\n",
            encoding="utf-8",
        )
        (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
        )
        # Control the rows directly so the assertion is about tokenization, not
        # format conversion. preprocess_dataset local-imports this name.
        monkeypatch.setattr(
            "soup_cli.data.loader.load_dataset",
            lambda *a, **k: {"train": [{"messages": _MESSAGES}]},
        )
        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
        assert result.exit_code == 0, result.output
        cache_dirs = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
        assert len(cache_dirs) == 1, cache_dirs
        ds = datasets.load_from_disk(str(cache_dirs[0]))
        return list(ds[0]["input_ids"])

    def test_cache_preserves_eos_the_tokenizer_appends(self, tmp_path, monkeypatch):
        """Template renders BOS but not EOS; the tokenizer appends EOS. The cached
        row must still end on the stop token and match the live path exactly.

        Fails without the #788 re-append: the cache row would end one token short
        (EOS dropped), which reaches the trainer verbatim."""
        from soup_cli.data.loss_mask import build_full_sequence_labels

        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        ids = self._run_preprocess(tmp_path, monkeypatch, tok)
        live = build_full_sequence_labels(_MESSAGES, tok, max_length=2048)["input_ids"]

        assert ids == live, "cache path must equal the live-training path"
        assert ids[-1] == self._EOS_ID, "cached row must end on the stop token"
        assert ids.count(self._EOS_ID) == 1
        assert ids.count(_BOS_ID) == 1

    def test_cache_invents_no_eos_when_tokenizer_appends_none(self, tmp_path, monkeypatch):
        """Tokenizer appends no EOS: the cache path must not invent one, matching
        the live path (control against over-eager re-appending)."""
        from soup_cli.data.loss_mask import build_full_sequence_labels

        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        ids = self._run_preprocess(tmp_path, monkeypatch, tok)
        live = build_full_sequence_labels(_MESSAGES, tok, max_length=2048)["input_ids"]

        assert ids == live
        assert ids.count(self._EOS_ID) == 0
        assert ids.count(_BOS_ID) == 1


class TestPreprocessCacheKey:
    def test_cache_key_changed_from_pre_fix_blob(self):
        """The #785 tokenization change must invalidate old preprocess caches:
        the key includes a schema token, so it differs from the old blob's hash.
        Fails if someone drops the schema token and reverts to the old format.
        """
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        args = dict(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
        )
        old_blob = (
            f"{args['dataset_path']}\x1f{args['tokenizer_name']}"
            f"\x1f{args['max_length']}\x1f{args['format_name']}"
        )
        old_key = hashlib.sha256(old_blob.encode("utf-8")).hexdigest()[:16]
        assert make_preprocess_cache_key(**args) != old_key
