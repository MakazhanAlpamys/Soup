"""#876: ``soup data preprocess`` and the live path disagreed on the leading BOS.

The live SFT path (``train_on_responses_only=false``) renders with
``add_special_tokens=False`` (``data/loss_mask.py`` :func:`_tokenize_only`), so its
leading BOS comes from the chat template alone. The preprocess cache tokenises the
rendered text with ``add_special_tokens=True`` and, until #876, removed a leading BOS
only when it was *doubled*. A ``data.chat_template`` preset renders no BOS, so on a
tokenizer whose post-processor prepends one the cache trained on one leading BOS
and the live path on zero (#781 established zero is right for such a template).

The fix drops the leading BOS the *post-processor* contributes, whatever the
template renders, so the cache keeps exactly the template's BOS like the live path.

It also closes an ordering hole the same fix would otherwise widen: #791 appends
the training EOS only when ``len(input_ids) < max_length``, but that check ran
after a BOS was removed, so a row the tokenizer had truncated to the budget read
as one short of it and gained an EOS the live path (append, then truncate) does
not have.

Every tokenizer is a real ``transformers`` fast tokenizer built offline, so the
chat template is the genuine Jinja renderer and BOS/EOS come from a genuine
``tokenizers`` post-processor.
"""


import pytest

_SPECIALS = [
    "<unk>", "<s>", "</s>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]
_WORDS = [
    "You", "are", "terse", ".", "What", "is", "the", "capital", "of", "France",
    "?", "Paris", "Berlin", "Germany", "system", "user", "assistant",
]
_BOS_ID = _SPECIALS.index("<s>")
_EOS_ID = _SPECIALS.index("</s>")

_BODY = "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
# Renders the BOS itself (the #785 doubled-BOS shape).
_BOS_TEMPLATE = "{{ bos_token }}" + _BODY
# A preset: renders no BOS, so any leading BOS is the post-processor's.
_PRESET_TEMPLATE = _BODY

_TEMPLATES = {"bos_template": _BOS_TEMPLATE, "preset": _PRESET_TEMPLATE}
_POST_PROCESSORS = ["bos", "bos_eos", None]

_SHORT = [
    {"role": "user", "content": "What is the capital of France ?"},
    {"role": "assistant", "content": "Paris ."},
]
_LONG = [
    {"role": "user", "content": "What is the capital of France ? " * 40},
    {"role": "assistant", "content": "Paris . " * 40},
]


def _tokenizer(chat_template, *, post_processor):
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
            special_tokens=[("<s>", _BOS_ID), ("</s>", _EOS_ID)],
        )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = chat_template
    return tok


def _live_ids(tok, messages, *, max_length):
    """What the live ``train_on_responses_only=false`` path trains on."""
    from soup_cli.data.loss_mask import build_full_sequence_labels

    return build_full_sequence_labels(messages, tok, max_length=max_length)["input_ids"]


def _leading_bos(ids):
    count = 0
    while count < len(ids) and ids[count] == _BOS_ID:
        count += 1
    return count


def _run_preprocess(tmp_path, monkeypatch, tok, *, messages, max_length=2048):
    """Run the real ``soup data preprocess`` on one chat row; return its ids."""
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: x/y\ntask: sft\n"
        "data:\n  train: ./d.jsonl\n  format: chatml\n"
        f"  max_length: {max_length}\n",
        encoding="utf-8",
    )
    (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(
        "soup_cli.data.loader.load_dataset",
        lambda *a, **k: {"train": [{"messages": messages}]},
    )
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    assert result.exit_code == 0, result.output
    cache_dirs = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
    assert len(cache_dirs) == 1, cache_dirs
    ds = datasets.load_from_disk(str(cache_dirs[0]))
    assert len(ds) == 1
    return list(ds[0]["input_ids"]), list(ds[0]["attention_mask"])


class TestLeadingBOSAgreesWithTheLivePath:
    @pytest.mark.parametrize("post_processor", _POST_PROCESSORS)
    @pytest.mark.parametrize("template", sorted(_TEMPLATES))
    def test_every_combination_of_the_issue_table(
        self, tmp_path, monkeypatch, template, post_processor
    ):
        """All six rows of #876's table. On ``main`` the two ``preset`` rows with a
        BOS-adding post-processor cached one leading BOS against the live path's
        zero; the other four already agreed and are the controls.

        An untruncated row is compared whole, not only by BOS count, so a fix that
        agrees on the BOS by dropping or adding some other token cannot pass."""
        tok = _tokenizer(_TEMPLATES[template], post_processor=post_processor)
        cached, mask = _run_preprocess(tmp_path, monkeypatch, tok, messages=_SHORT)
        live = _live_ids(tok, _SHORT, max_length=2048)

        assert _leading_bos(cached) == _leading_bos(live), (
            f"leading BOS: cache {_leading_bos(cached)}, live {_leading_bos(live)}"
        )
        assert cached == live, f"cache {cached} != live {live}"
        assert mask == [1] * len(cached), "attention mask tracks the ids"

    def test_the_preset_bos_row_really_disagreed_on_main(self):
        """Sanity for the table: the preset renders no BOS, and the post-processor
        adds one under ``add_special_tokens=True``. If this stopped holding, the
        disagreeing rows above would pass without exercising anything."""
        tok = _tokenizer(_PRESET_TEMPLATE, post_processor="bos")
        text = tok.apply_chat_template(_SHORT, tokenize=False, add_generation_prompt=False)
        assert _leading_bos(tok(text, add_special_tokens=False)["input_ids"]) == 0
        assert _leading_bos(tok(text, add_special_tokens=True)["input_ids"]) == 1


class TestTruncation:
    @pytest.mark.parametrize("template", sorted(_TEMPLATES))
    def test_truncated_row_gains_no_eos_the_live_path_lacks(
        self, tmp_path, monkeypatch, template
    ):
        """BOS-only post-processor, row longer than ``max_length``. The live path
        appends the EOS and then truncates, so its row ends on content. On ``main``
        the ``bos_template`` row still gained an EOS: the doubled BOS was removed
        after truncation, the row read as one under budget, and #791's
        ``len < max_length`` check appended. Dropping the post-processor BOS for a
        preset too would have spread that to the ``preset`` row."""
        tok = _tokenizer(_TEMPLATES[template], post_processor="bos")
        cached, _ = _run_preprocess(
            tmp_path, monkeypatch, tok, messages=_LONG, max_length=64
        )
        live = _live_ids(tok, _LONG, max_length=64)

        assert len(live) == 64 and live[-1] != _EOS_ID, "sanity: live row truncated"
        assert cached[-1] != _EOS_ID, f"truncated cache row gained an EOS: {cached}"
        assert _leading_bos(cached) == _leading_bos(live)
        assert len(cached) <= 64

    @pytest.mark.parametrize("template", sorted(_TEMPLATES))
    def test_truncated_row_keeps_the_post_processor_eos(
        self, tmp_path, monkeypatch, template
    ):
        """#876 criterion 2, and the #788 round-2 regression: with a post-processor
        that appends EOS, HF truncation reserves room for it, so a truncated cache
        row still ends on its trained EOS. Removing the BOS must not cost it."""
        tok = _tokenizer(_TEMPLATES[template], post_processor="bos_eos")
        cached, _ = _run_preprocess(
            tmp_path, monkeypatch, tok, messages=_LONG, max_length=64
        )
        live = _live_ids(tok, _LONG, max_length=64)

        assert cached[-1] == _EOS_ID and cached.count(_EOS_ID) == 1, cached
        assert _leading_bos(cached) == _leading_bos(live)

    def test_a_row_that_exactly_fits_with_the_bos_still_gets_its_eos(
        self, tmp_path, monkeypatch
    ):
        """The other side of the truncation check. A preset row whose content plus
        the post-processor BOS is exactly ``max_length`` was NOT truncated: the live
        path's content plus EOS fits the same budget and ends on EOS. Deciding
        "truncated" from the pre-strip length alone would withhold that EOS."""
        tok = _tokenizer(_PRESET_TEMPLATE, post_processor="bos")
        text = tok.apply_chat_template(_LONG, tokenize=False, add_generation_prompt=False)
        content = tok(text, add_special_tokens=False)["input_ids"]
        max_length = len(content) + 1
        assert 64 <= max_length, "schema floor for data.max_length"

        cached, _ = _run_preprocess(
            tmp_path, monkeypatch, tok, messages=_LONG, max_length=max_length
        )
        live = _live_ids(tok, _LONG, max_length=max_length)

        assert live[-1] == _EOS_ID and len(live) == max_length, "sanity: live fits"
        assert cached == live


class TestTheHelpers:
    @pytest.mark.parametrize(
        "post_processor, expected", [("bos", 1), ("bos_eos", 1), (None, 0)]
    )
    def test_the_probe_measures_only_the_post_processor(self, post_processor, expected):
        """The count is a property of the tokenizer, not of any template: a
        ``{{ bos_token }}`` template must not change it."""
        from soup_cli.data.loss_mask import post_processor_leading_bos_count

        for template in (_BOS_TEMPLATE, _PRESET_TEMPLATE):
            tok = _tokenizer(template, post_processor=post_processor)
            assert post_processor_leading_bos_count(tok) == expected

    def test_an_unmeasurable_tokenizer_falls_back_to_the_doubled_rule(self):
        """If the probe cannot run, the helper must behave exactly as before #876:
        remove a doubled BOS, and never a lone one it cannot attribute."""
        from soup_cli.data.loss_mask import (
            post_processor_leading_bos_count,
            strip_post_processor_leading_bos,
        )

        class _Refuses:
            bos_token_id = _BOS_ID

            def __call__(self, *a, **k):
                raise RuntimeError("no probe")

        tok = _Refuses()
        assert post_processor_leading_bos_count(tok) is None
        doubled = strip_post_processor_leading_bos(tok, [_BOS_ID, _BOS_ID, 9], [1, 1, 1], None)
        assert doubled == ([_BOS_ID, 9], [1, 1])
        lone = strip_post_processor_leading_bos(tok, [_BOS_ID, 9], [1, 1], None)
        assert lone == ([_BOS_ID, 9], [1, 1])

    def test_only_bos_tokens_are_removed(self):
        """A count says how many BOS to expect, not how many tokens to cut: a row
        that does not start with BOS keeps its first token."""
        from soup_cli.data.loss_mask import strip_post_processor_leading_bos

        tok = _tokenizer(_PRESET_TEMPLATE, post_processor="bos")
        assert strip_post_processor_leading_bos(tok, [9, 10], [1, 1], 1) == ([9, 10], [1, 1])


class TestCacheKey:
    def test_schema_bumped_so_a_v3_cache_is_rejected(self):
        """The cached bytes change for preset rows, so a cache built under #791's
        v3 encoding must not be silently reused."""
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key
        from tests.preprocess_cache_blob import CURRENT_SCHEMA, blob_key

        args = dict(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
            mask_mode="responses_only",
            task="sft",
            chat_template=None,
        )

        current = make_preprocess_cache_key(**args)
        assert current != blob_key("v3", **args), "a v3 (#791) cache must be rejected"
        assert current != blob_key("v4", **args), "a v4 (#1067) cache must be rejected"
        assert current != blob_key("v5", **args), "a v5 (#876) cache must be rejected"
        assert current == blob_key(CURRENT_SCHEMA, **args)
