"""#791: ``soup data preprocess`` cached chat rows without the EOS the live
path trains on, so a cached run could train with no stop token.

The live SFT path (``train_on_responses_only=false``) reproduces TRL 0.29.1's
``add_eos`` rule: a chat row that does not already end on ``eos_token`` gets one
(``data/loss_mask.py`` :func:`build_full_sequence_labels` ->
:func:`append_training_eos`). ``soup data preprocess`` tokenised the rendered
template directly (``commands/data.py`` ``preprocess_dataset``), so a row only
carried an EOS when the template rendered one or the tokenizer's post-processor
appended one. On a template that renders no EOS and a Qwen-shaped tokenizer whose
post-processor appends none, the live path trained on one stop token and the
cached path on zero. It was silent: the loss curve looks normal and
``soup data doctor`` renders the live path, not the cache.

#791 makes the preprocess cache apply the same ``add_eos`` rule so the two paths
train on the same EOS count, and bumps the preprocess cache-key schema so a cache
built under the old (no-EOS) encoding is rejected rather than silently reused.

The leading BOS still differs between the cache and the live path by design
(the cache is pinned to ``main``'s one post-processor BOS, the live path renders
template-only); that divergence is tracked separately in #876, so these tests
assert the EOS *count* agrees rather than byte-equality.

Every tokenizer is a real ``transformers`` fast tokenizer built offline, so
``apply_chat_template`` is the genuine Jinja renderer and the BOS/EOS come from a
genuine ``tokenizers`` post-processor.
"""

import hashlib
import sys

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

# Renders the tokenizer's BOS but no EOS: the shape that lost the stop token.
_BOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
)
# Renders the EOS itself, so the template is already the source of the stop token.
_BOS_EOS_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {{ eos_token }} {% endfor %}"
)

_ROWS = [
    [
        {"role": "user", "content": "What is the capital of France ?"},
        {"role": "assistant", "content": "Paris ."},
    ],
    [
        {"role": "user", "content": "What is the capital of Germany ?"},
        {"role": "assistant", "content": "Berlin ."},
    ],
]


def _tokenizer(chat_template, *, post_processor="bos"):
    """A real fast tokenizer whose post-processor adds ``post_processor``.

    ``bos`` prepends only BOS (the Llama/Qwen family shape that renders its own
    chat specials and, for many models, appends no EOS); ``None`` adds neither
    (the barest Qwen shape); ``bos_eos`` appends both.
    """
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


def _live_eos_count(tok, messages, *, max_length=2048):
    """The EOS count the live training path trains on for ``messages``.

    Post-#788 the ``train_on_responses_only=false`` path pre-tokenises via
    :func:`build_full_sequence_labels`, which applies TRL's ``add_eos`` rule in
    token space. That is the live path training actually consumes, so it is the
    baseline the cache must agree with.
    """
    from soup_cli.data.loss_mask import build_full_sequence_labels

    ids = build_full_sequence_labels(messages, tok, max_length=max_length)["input_ids"]
    return ids.count(_EOS_ID)


def _trl_main_text_eos_count(tok, messages):
    """The EOS count ``main``'s SFT language-modeling text path trained on.

    Reproduces TRL 0.29.1's two real operations on a legacy ``{"text"}`` row:
    ``add_eos`` appends ``eos_token`` as a string when the rendered text does not
    already end with it (``sft_trainer.py:1026-1038``), then TRL tokenises with
    the tokenizer's default ``add_special_tokens=True``. This is what the EOS
    count of a non-cached run came out to before #788 pre-tokenised the path, and
    it agrees with :func:`_live_eos_count`; both are the stop token the cache
    must match.
    """
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not text.endswith(tok.eos_token):
        text = text + tok.eos_token
    return tok(text)["input_ids"].count(_EOS_ID)


def _run_preprocess(tmp_path, monkeypatch, tok, *, task="sft", rows, max_length=2048):
    """Run ``soup data preprocess`` on ``rows`` and return the cached rows' ids."""
    transformers = pytest.importorskip("transformers")
    datasets = pytest.importorskip("datasets")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)
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
    return [list(ds[i]["input_ids"]) for i in range(len(ds))]


class TestCacheEOSMatchesLivePath:
    def test_no_eos_template_cache_matches_live_eos_per_row(self, tmp_path, monkeypatch):
        """The #791 defect. Template renders no EOS, post-processor appends none
        (Qwen shape): ``main`` cached zero stop tokens per row while the live path
        trained on one. The cache now matches the live path per row.

        Fails on pre-#791 ``main``: the cached count is 0 and the live count is 1.
        """
        rows = [{"messages": m} for m in _ROWS]
        tok = _tokenizer(_BOS_TEMPLATE, post_processor=None)

        cached = _run_preprocess(tmp_path, monkeypatch, tok, rows=rows)
        assert len(cached) == len(_ROWS)
        for ids, messages in zip(cached, _ROWS):
            live = _live_eos_count(tok, messages)
            assert live == _trl_main_text_eos_count(tok, messages) == 1, (
                "sanity: the live path trains on exactly one EOS here"
            )
            assert ids.count(_EOS_ID) == live, (
                f"cached row must train on the live EOS count; got {ids}"
            )
            assert ids[-1] == _EOS_ID, "the stop token is the trailing token"

    def test_bos_only_post_processor_also_gains_the_eos(self, tmp_path, monkeypatch):
        """BOS-only post-processor (Llama shape) with a no-EOS template: the cache
        had a BOS but no EOS on ``main``. #791 appends the EOS and leaves the one
        BOS (its de-duplication and the #876 BOS divergence are unaffected)."""
        rows = [{"messages": _ROWS[0]}]
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")

        (ids,) = _run_preprocess(tmp_path, monkeypatch, tok, rows=rows)
        assert ids.count(_EOS_ID) == _live_eos_count(tok, _ROWS[0]) == 1
        assert ids[-1] == _EOS_ID
        assert ids.count(_BOS_ID) == 1, "BOS unchanged by #791"


class TestControls:
    def test_template_that_renders_eos_gets_no_second_eos(self, tmp_path, monkeypatch):
        """Control: a template that already renders the EOS must not gain a second
        one. ``append_training_eos`` is a no-op when the row already ends on EOS,
        so the cache keeps exactly the count the live path has."""
        rows = [{"messages": _ROWS[0]}]
        tok = _tokenizer(_BOS_EOS_TEMPLATE, post_processor="bos")

        (ids,) = _run_preprocess(tmp_path, monkeypatch, tok, rows=rows)
        live = _live_eos_count(tok, _ROWS[0])
        assert ids.count(_EOS_ID) == live, "no second EOS beyond what the live path has"
        assert ids[-1] == _EOS_ID

    def test_pretrain_row_is_byte_identical_to_main(self, tmp_path, monkeypatch):
        """Control: pretrain rows never go through a chat template, so #791 must
        leave them exactly as ``main``: ``add_special_tokens=True`` and no EOS
        re-append. The post-processor appends no EOS, so a raw row does not end on
        one; a mutation that drops the ``not is_pretrain`` guard would append the
        stop token and change the row, which this catches."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor="bos")
        raw = "You are terse ."
        expected = tok(raw, add_special_tokens=True)["input_ids"]
        assert expected[-1] != _EOS_ID, "guard: the raw row must not already end on EOS"

        (ids,) = _run_preprocess(
            tmp_path, monkeypatch, tok, task="pretrain", rows=[{"text": raw}]
        )
        assert ids == expected

    def test_truncated_row_is_not_pushed_past_the_budget(self, tmp_path, monkeypatch):
        """Control: a chat row long enough to fill ``max_length`` must not gain an
        EOS that pushes it over budget. The live path appends-then-truncates and
        so keeps no trailing EOS for such a row; the cache must not exceed the
        budget either. Kills a mutation that appends unconditionally."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor=None)
        long_msgs = [
            {"role": "user", "content": "What is the capital of France ? " * 40},
            {"role": "assistant", "content": "Paris . " * 40},
        ]
        (ids,) = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": long_msgs}], max_length=64
        )
        assert len(ids) == 64, "a filled row stays within the truncation budget"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="TRL's SFTTrainer._prepare_dataset runs its dataset prep inside "
    "PartialState().main_process_first(); the full training path this stands in for "
    "hangs on Windows via PartialState (the same obstacle #788 hit), so this runs on "
    "the ubuntu and macOS cells, which is where the baseline needs pinning.",
)
class TestRealTRLPrepareDatasetBaseline:
    """#791 acceptance criterion 1 asks for the live baseline through TRL's real
    ``_prepare_dataset`` rather than a reproduction. The other tests use
    :func:`_live_eos_count` (``build_full_sequence_labels``) and
    :func:`_trl_main_text_eos_count` because a full ``SFTTrainer`` construction hangs
    on Windows ``PartialState``. This class removes the ambiguity on the platforms
    where it does not: it drives TRL 0.29.1's genuine ``SFTTrainer._prepare_dataset``
    over the rendered ``{"text"}`` rows the legacy SFT path feeds it and asserts the
    cache trains on the same EOS count TRL actually produces.

    ``_prepare_dataset`` reads only ``self._is_vlm`` on the language-modeling text
    path, so an unbound call runs TRL's real map/``add_eos``/tokenize pipeline without
    constructing the trainer (which is what pulls in the accelerator init).
    """

    def _real_prepare_eos_counts(self, out_dir, tok, messages_rows, *, max_length=2048):
        datasets = pytest.importorskip("datasets")
        pytest.importorskip("trl")
        from types import SimpleNamespace

        from trl import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer

        # The legacy SFT text path renders the chat template to a {"text"} row and
        # lets TRL tokenize it; that is the row shape whose EOS handling #785/#788
        # concerned. Feed exactly that to the real _prepare_dataset.
        texts = [
            {"text": tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)}
            for m in messages_rows
        ]
        ds = datasets.Dataset.from_list(texts)
        args = SFTConfig(
            output_dir=str(out_dir),
            max_length=max_length,
            packing=False,
            shuffle_dataset=False,  # keep row order for the per-row comparison
            dataset_num_proc=None,  # no multiprocessing (Windows spawn re-imports)
            use_cpu=True,
            report_to=[],
        )
        prepared = SFTTrainer._prepare_dataset(
            SimpleNamespace(_is_vlm=False), ds, tok, args, False, None, "train"
        )
        return [list(prepared[i]["input_ids"]).count(_EOS_ID) for i in range(len(prepared))]

    def test_cache_matches_trl_real_prepare_dataset_eos(self, tmp_path, monkeypatch):
        """The #791 defect shape (no-EOS template, no-EOS post-processor), pinned
        against TRL's real ``_prepare_dataset`` instead of a reproduction: TRL's
        genuine pipeline trains on one EOS per row, and the cache now matches it
        per row (and my :func:`_trl_main_text_eos_count` reproduction agrees, which
        is what the reproduction was standing in for)."""
        tok = _tokenizer(_BOS_TEMPLATE, post_processor=None)

        real = self._real_prepare_eos_counts(tmp_path / "trl", tok, _ROWS)
        cached = _run_preprocess(
            tmp_path / "pre", monkeypatch, tok, rows=[{"messages": m} for m in _ROWS]
        )
        assert len(real) == len(cached) == len(_ROWS)
        for real_eos, ids, messages in zip(real, cached, _ROWS):
            assert real_eos == 1, "TRL's real prepare trains on exactly one EOS here"
            assert real_eos == _trl_main_text_eos_count(tok, messages), (
                "the hand reproduction must equal TRL's real _prepare_dataset"
            )
            assert real_eos == _live_eos_count(tok, messages)
            assert ids.count(_EOS_ID) == real_eos, "cache matches the real TRL baseline"
            assert ids[-1] == _EOS_ID


class TestCacheKey:
    def test_cache_key_schema_bumped_for_791(self):
        """The #791 tokenization change must invalidate caches built under the #785
        (v2) encoding: the schema token advanced, so the key differs from both the
        v2 blob and the pre-schema blob. Fails if the schema token is reverted."""
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        args = dict(
            dataset_path="data/train.jsonl",
            tokenizer_name="meta-llama/Llama-3.1-8B",
            max_length=2048,
            format_name="chatml",
        )

        def _blob_key(schema):
            prefix = f"{schema}\x1f" if schema else ""
            blob = (
                f"{prefix}{args['dataset_path']}\x1f{args['tokenizer_name']}"
                f"\x1f{args['max_length']}\x1f{args['format_name']}"
            )
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

        current = make_preprocess_cache_key(**args)
        assert current != _blob_key("v2"), "a v2 (#785) cache must be rejected"
        assert current != _blob_key(""), "a pre-schema cache must be rejected"
        assert current == _blob_key("v3"), "current schema is v3 (#791)"
