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
_EOS_ID = _SPECIALS.index("</s>")

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


def _trl_main_lm_ids(tok, messages):
    """The ids the SFT language-modeling text path trained on for ``messages`` on
    ``main`` (the baseline #788 must be measured against).

    Reproduces TRL 0.29.1's two real operations on a legacy ``{"text"}`` row using
    the real tokenizer: ``add_eos`` appends ``eos_token`` as a string when the
    rendered text does not already end with it (``sft_trainer.py`` ``add_eos``,
    lines 1026-1031), then TRL tokenizes with the tokenizer's default
    ``add_special_tokens=True`` (line 1131). ``tok(text=text)`` alone omits
    ``add_eos`` and under-counts the trained EOS — the mistake the first #788
    attempt made and the reason these tests are baselined against this instead.
    """
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not text.endswith(tok.eos_token):
        text = text + tok.eos_token
    return tok(text)["input_ids"]


class TestLegacyTextPathBOS:
    def test_bos_template_trains_on_one_bos_and_keeps_the_eos(self):
        """A template that renders {{ bos_token }} must train on one BOS (not the
        two ``main`` produced) while keeping the trailing EOS ``main`` trained on.

        The body equals inference's one-step encode (add_special_tokens=False,
        what #782 sends) plus the training stop token inference does not send.

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

        main_ids = _trl_main_lm_ids(tok, _MESSAGES)
        assert main_ids.count(_BOS_ID) == 2, "sanity: main doubled the BOS here"
        # The stop token main trained on (TRL add_eos) is preserved, not dropped.
        assert ids[-1] == _EOS_ID and main_ids[-1] == _EOS_ID
        assert ids.count(_EOS_ID) == main_ids.count(_EOS_ID) == 1

        inference = tok.apply_chat_template(
            _MESSAGES, tokenize=True, add_generation_prompt=False,
            add_special_tokens=False, return_dict=True,
        )["input_ids"]
        assert ids == list(inference) + [_EOS_ID]
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

    ``main``'s live text path trained on the EOS TRL's ``add_eos`` appends as a
    string, independent of the tokenizer's post-processor. Dropping it teaches
    run-on generation. Every case is baselined against ``_trl_main_lm_ids`` (what
    ``main`` trained on), not ``tok(text=...)`` which omits ``add_eos``. The fix
    reproduces TRL's rule: end on exactly one EOS, removing ``main``'s duplicates
    but never dropping the stop token.
    """

    @pytest.mark.parametrize(
        "post_processor",
        ["bos", None],
        ids=["bos-only (Gemma/Llama-2)", "no-post-processor (Qwen)"],
    )
    def test_eos_kept_when_post_processor_appends_none(self, post_processor):
        """The regression the #788 review caught. Template renders no EOS on a
        tokenizer whose post-processor appends none. ``main`` still trained on one
        EOS because TRL's ``add_eos`` appends it as a string; probing the
        post-processor (the first #788 attempt) added none and dropped the stop
        token 1 -> 0. The fix keeps it."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor=post_processor)
        main_ids = _trl_main_lm_ids(tok, _MESSAGES)
        assert main_ids.count(_EOS_ID) == 1, "main trained on one EOS via add_eos"

        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids[-1] == _EOS_ID and ids.count(_EOS_ID) == 1, (
            f"fix must keep the EOS main trained on, not drop it; got {ids}"
        )

    def test_eos_not_duplicated_when_post_processor_also_appends(self):
        """Template renders no EOS; the post-processor appends one AND TRL's
        add_eos appended another, so ``main`` trained on a duplicated EOS. The fix
        ends on exactly one stop token — main's duplicate removed, like the BOS."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        main_ids = _trl_main_lm_ids(tok, _MESSAGES)
        assert main_ids.count(_EOS_ID) == 2, "sanity: main duplicated the EOS here"

        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]
        assert ids[-1] == _EOS_ID, "row ends on the stop token"
        assert ids.count(_EOS_ID) == 1, "exactly one, the duplicate removed"
        assert ids.count(_BOS_ID) == 1

    def test_template_rendered_eos_is_kept_and_not_re_appended(self):
        """Template renders the EOS itself; the fix adds none because the sequence
        already ends on it (TRL's rule: append only if not already ending on EOS).
        The body equals inference's encode; main's add_eos duplicate is gone."""
        tok = _tokenizer(_BOS_EOS_TEMPLATE, post_processor="bos")
        ids = _legacy_format_row(tok)({"messages": _MESSAGES})["input_ids"]

        assert ids[-1] == _EOS_ID, "row ends on the template's stop token"
        inference = tok.apply_chat_template(
            _MESSAGES, tokenize=True, add_generation_prompt=False,
            add_special_tokens=False, return_dict=True,
        )["input_ids"]
        assert ids == list(inference), "nothing re-appended; already ends on EOS"
        assert ids.count(_BOS_ID) == 1

        main_ids = _trl_main_lm_ids(tok, _MESSAGES)
        assert main_ids.count(_EOS_ID) == ids.count(_EOS_ID) + 1, (
            "main's add_eos duplicated the trailing EOS; the fix does not"
        )


class TestTemplatedOnly:
    """The rule applies only to text a chat template rendered (maintainer req 2)."""

    def test_no_template_row_raises_rather_than_being_silently_pretokenized(self):
        """A tokenizer with no chat_template must raise, not be silently stripped
        into wrong training data. Routed through ``build_format_row`` — the real
        dispatch, present on ``main`` too — so it discriminates the raise
        behaviour rather than the mere absence of a #788 import."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.sft_format import build_format_row

        tok = _tokenizer(None, post_processor="bos")  # no chat_template
        tok.chat_template = None
        dcfg = DataConfig(
            train="t.jsonl",
            train_on_responses_only=False,
            train_on_messages_with_train_field=False,
            max_length=2048,
        )
        format_row = build_format_row(tokenizer=tok, data_cfg=dcfg)
        with pytest.raises(ValueError, match="chat_template"):
            format_row({"messages": _MESSAGES})


class TestPreprocessCachePathEOS:
    """`soup data preprocess` EOS behaviour is pinned to ``main`` (post-processor
    only), deliberately NOT to the live path.

    #788 de-duplicates the BOS on the preprocess cache path just as on the live
    path, but it must NOT change what EOS the cache trains on: ``main``'s
    preprocess never went through TRL's ``add_eos``, so its EOS is whatever the
    post-processor added. The live path now reproduces ``add_eos`` and so trains
    on an EOS the cache does not (Qwen shape). That cache-vs-live mismatch is
    pre-existing on ``main`` and tracked in #791 — not resolved here.
    """

    def _run_preprocess(
        self, tmp_path, monkeypatch, tok, *, task="sft", rows=None, max_length=2048
    ):
        transformers = pytest.importorskip("transformers")
        datasets = pytest.importorskip("datasets")
        from typer.testing import CliRunner

        from soup_cli.cli import app

        rows = rows if rows is not None else [{"messages": _MESSAGES}]
        monkeypatch.chdir(tmp_path)
        (tmp_path / "soup.yaml").write_text(
            f"base: x/y\ntask: {task}\n"
            "data:\n  train: ./d.jsonl\n  format: chatml\n"
            f"  max_length: {max_length}\n",
            encoding="utf-8",
        )
        (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
        )
        # Control the rows directly so the assertion is about tokenization, not
        # format conversion. preprocess_dataset local-imports this name.
        monkeypatch.setattr(
            "soup_cli.data.loader.load_dataset", lambda *a, **k: {"train": rows}
        )
        result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
        assert result.exit_code == 0, result.output
        cache_dirs = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
        assert len(cache_dirs) == 1, cache_dirs
        ds = datasets.load_from_disk(str(cache_dirs[0]))
        return list(ds[0]["input_ids"])

    def _main_preprocess_ids(self, tok, *, messages=None, max_length=2048):
        """What ``main``'s preprocess chat path produced: ``tokenizer(text)`` with
        the default ``add_special_tokens=True`` and the same truncation budget (no
        TRL ``add_eos`` on this path). HF truncation reserves room for the
        post-processor's specials, so a truncated ``main`` row still ends on EOS."""
        messages = messages if messages is not None else _MESSAGES
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return tok(
            text, max_length=max_length, truncation=True, add_special_tokens=True
        )["input_ids"]

    def test_cache_eos_matches_main_when_post_processor_appends(self, tmp_path, monkeypatch):
        """bos_eos post-processor: ``main``'s preprocess kept one EOS (from the
        post-processor). The cache keeps exactly that and drops the doubled BOS."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        ids = self._run_preprocess(tmp_path, monkeypatch, tok)
        main_pp = self._main_preprocess_ids(tok)

        assert ids.count(_EOS_ID) == main_pp.count(_EOS_ID) == 1, "EOS unchanged vs main"
        assert main_pp.count(_BOS_ID) == 2 and ids.count(_BOS_ID) == 1, "BOS de-duplicated"

    def test_cache_keeps_mains_zero_eos_for_no_eos_post_processor(self, tmp_path, monkeypatch):
        """Qwen/BOS-only shape: ``main``'s preprocess added no EOS, so the cache
        keeps zero rather than adopting the live path's one. Documents that the
        cache and live path diverge here — the #791 mismatch."""
        from soup_cli.data.loss_mask import build_full_sequence_labels

        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        ids = self._run_preprocess(tmp_path, monkeypatch, tok)
        main_pp = self._main_preprocess_ids(tok)

        assert ids.count(_EOS_ID) == main_pp.count(_EOS_ID) == 0, "EOS unchanged vs main"
        assert ids.count(_BOS_ID) == 1, "BOS de-duplicated"
        live = build_full_sequence_labels(_MESSAGES, tok, max_length=2048)["input_ids"]
        assert live.count(_EOS_ID) == 1, "live trains on an EOS the cache does not (#791)"

    def test_pretrain_cache_is_byte_identical_to_main(self, tmp_path, monkeypatch):
        """Pretrain rows never went through a template, so #788 must leave them
        exactly as ``main``: ``add_special_tokens=True`` and no EOS re-append.
        Kills the two surviving mutations (add_special_tokens=False on pretrain,
        and the EOS re-append applied to pretrain)."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        raw = "You are terse ."
        ids = self._run_preprocess(
            tmp_path, monkeypatch, tok, task="pretrain", rows=[{"text": raw}]
        )
        assert ids == tok(raw, add_special_tokens=True)["input_ids"]

    def test_cache_keeps_eos_on_truncated_row(self, tmp_path, monkeypatch):
        """The #788 round-2 blocker. A row long enough to truncate must still end
        on the post-processor's EOS, exactly as ``main``. The first round-2 attempt
        (``add_special_tokens=False``, re-append the EOS, slice ``[:max_length]``)
        filled the whole budget with content and sliced the re-appended EOS back
        off (1 -> 0). Tokenising as ``main`` reserves truncation room for the EOS,
        so it survives; we only drop the one duplicated leading BOS."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos_eos")
        long_msgs = [
            {"role": "user", "content": "What is the capital of France ? " * 40},
            {"role": "assistant", "content": "Paris . " * 40},
        ]
        ids = self._run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": long_msgs}], max_length=64
        )
        main_pp = self._main_preprocess_ids(tok, messages=long_msgs, max_length=64)
        assert len(main_pp) == 64, "sanity: main filled the truncation budget"
        assert main_pp[0] == main_pp[1] == _BOS_ID, "sanity: main doubled the BOS"
        assert ids == main_pp[1:], "cache == main minus the one duplicated leading BOS"
        assert ids[-1] == _EOS_ID and ids.count(_EOS_ID) == 1, (
            f"truncated row must still end on the stop token; got {ids}"
        )
        assert ids.count(_BOS_ID) == 1, "doubled BOS removed even under truncation"

    def test_cache_keeps_mains_eos_when_template_renders_it(self, tmp_path, monkeypatch):
        """Template renders the EOS itself AND the post-processor appends one, so
        ``main`` trained on several trailing EOS. The cache reproduces ``main``'s
        EOS count (it does NOT collapse to one like the live path — that divergence
        is #791), minus only the duplicated leading BOS."""
        tok = _tokenizer(_BOS_EOS_TEMPLATE, post_processor="bos_eos")
        ids = self._run_preprocess(tmp_path, monkeypatch, tok)
        main_pp = self._main_preprocess_ids(tok)

        assert main_pp[0] == main_pp[1] == _BOS_ID, "sanity: main doubled the BOS"
        assert ids == main_pp[1:], "cache == main minus the duplicated BOS"
        assert ids.count(_EOS_ID) == main_pp.count(_EOS_ID) > 1, "EOS count pinned to main"
        assert ids[-1] == _EOS_ID and ids.count(_BOS_ID) == 1


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


class TestDataDoctorLegacyMatchesTraining:
    """``soup data doctor --show-mask`` must X-ray what training actually
    consumes. Its legacy (``train_on_responses_only=false``) branch therefore
    calls the SAME builder as the trainer, :func:`build_full_sequence_labels`.

    #788 regression guard: ``main`` re-rendered here with
    ``apply_chat_template(tokenize=True)`` and the tokenizer's default specials,
    which doubled the BOS and skipped TRL's ``add_eos`` — so the X-ray diverged
    from training. Reverting ``_build_row_labels`` to that now fails here rather
    than silently. (Without this test the revert passes the whole suite.)
    """

    @pytest.mark.parametrize(
        "post_processor",
        ["bos_eos", "bos", None],
        ids=["bos+eos", "bos-only", "no-post-processor (Qwen)"],
    )
    def test_legacy_branch_matches_the_training_builder(self, post_processor):
        from soup_cli.data.loss_mask import build_full_sequence_labels
        from soup_cli.utils.data_doctor import _build_row_labels

        tok = _tokenizer(_BOS_TEMPLATE, post_processor=post_processor)
        doctor = _build_row_labels(
            tok,
            _MESSAGES,
            max_length=2048,
            train_on_responses_only=False,
            train_on_messages_with_train_field=False,
            include_eot=True,
        )
        training = build_full_sequence_labels(_MESSAGES, tok, max_length=2048)
        assert doctor["input_ids"] == training["input_ids"], (
            "the mask X-ray must tokenize identically to training"
        )
        assert doctor["labels"] == training["labels"]
        # and it is genuinely the post-#785 shape, not main's doubled BOS
        assert doctor["input_ids"].count(_BOS_ID) == 1
