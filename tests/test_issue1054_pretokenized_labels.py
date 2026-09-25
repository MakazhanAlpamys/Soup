"""#1054: ``soup data preprocess`` cached rows with no ``labels`` column, so a
``pre_tokenized`` run trained on the prompt while the equivalent live run masked it.

The live SFT path builds ``labels`` through ``data/loss_mask.py``'s builders and
masks everything that is not an assistant response (``train_on_responses_only``,
default true). ``preprocess_dataset`` wrote only ``{input_ids, attention_mask}``,
so the cache reached TRL with no labels and TRL's collator fell back to
``labels = input_ids`` -- full-sequence loss. Two runs over the same rows therefore
optimised different objectives depending only on whether the data was preprocessed
first, and nothing raised: the loss curve looks entirely normal.

Because the failure is silent, the assertions here are about which tokens are
**unmasked** (``!= IGNORE_INDEX``), never about the presence of a ``labels``
column -- a present-but-degenerate array (all ``-100``, or all unmasked) must
fail exactly like a missing one. The parity assertions compare the trained token
SEQUENCE rather than a count, because a mask shifted one position keeps the count
exact while training on a prompt token.

The groups:
- the cache carries a loss mask equal to the live path's (``TestCachedLabels``);
- every flag the live builder reads reaches the cache and the key --
  ``mask_history``, ``train_on_eot`` (``TestMaskFlagsReachTheCache``);
- every trainer forwards the training config to the gate, or its own cache is
  unloadable (``TestEveryTrainerPassesTheTrainingConfig``);
- the mask setting is a cache-key input, so a cache built under one masking config
  is refused by the other (``TestCacheKeyCoversMaskMode``);
- ``soup train`` reports the cached row count rather than 0 (``TestSampleCount``).

The tokenizer is a real ``transformers`` fast tokenizer built offline (same shape
as tests/test_issue791_preprocess_cache_eos.py), so ``apply_chat_template`` is the
genuine Jinja renderer.
"""

import json
from types import SimpleNamespace

import pytest

from soup_cli.data.loss_mask import IGNORE_INDEX

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

_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
)

# The assistant's EOS sits OUTSIDE ``{% generation %}``, so ``train_on_eot``
# decides whether it is trained. On ``_TEMPLATE`` the EOS falls inside the span
# either way, which makes the flag inert and any test built on it vacuous.
_GEN_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|> "
    "{% if m['role'] == 'assistant' %}"
    "{% generation %}{{ m['content'] }}{% endgeneration %} {{ eos_token }} "
    "{% else %}{{ m['content'] }} <|end|> {% endif %}{% endfor %}"
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
    # Multi-turn. A single-turn row has one assistant span and so exactly two
    # mask boundaries; an off-by-one in the mask transfer survives a per-row
    # COUNT check on either. These rows have 2N boundaries and are the ones
    # that pin ``align_labels_to_ids`` -- keep at least one of each here.
    [
        {"role": "system", "content": "You are terse ."},
        {"role": "user", "content": "What is the capital of France ?"},
        {"role": "assistant", "content": "Paris ."},
        {"role": "user", "content": "What is the capital of Germany ?"},
        {"role": "assistant", "content": "Berlin ."},
    ],
    [
        {"role": "user", "content": "What is the capital of France ?"},
        {"role": "assistant", "content": "Paris ."},
        {"role": "user", "content": "What is the capital of Germany ?"},
        {"role": "assistant", "content": "Berlin ."},
        {"role": "user", "content": "What is the capital of Germany ?"},
        {"role": "assistant", "content": "Berlin ."},
    ],
]


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(_SPECIALS + _WORDS)}
    backend = tokenizers.Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(_SPECIALS)
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", _BOS_ID), ("</s>", _EOS_ID)],
    )
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = _TEMPLATE
    return tok


def _unmasked(labels):
    """Count of positions that contribute to the loss."""
    return sum(1 for label in labels if label != IGNORE_INDEX)


def _trained(input_ids, labels):
    """The token sequence the loss is actually computed over.

    The cached row and the live one need not be the same length (#791 appends a
    training EOS to the cached one), so label POSITIONS cannot be compared
    -- but this sequence can: it is offset-invariant, and unlike a count of
    unmasked positions it does not survive a mask shifted by one, which would
    train on a prompt token and drop the span's last token while keeping the
    count identical.
    """
    return [
        token for token, label in zip(input_ids, labels) if label != IGNORE_INDEX
    ]


def _run_preprocess(tmp_path, monkeypatch, tok, *, rows, task="sft", data_extra=""):
    """Run the real ``soup data preprocess`` CLI and return the cache directory.

    Deliberately the CLI path, not a hand-built cache directory: the missing
    ``labels`` column was a defect of that code path specifically.
    """
    transformers = pytest.importorskip("transformers")
    from typer.testing import CliRunner

    from soup_cli.cli import app

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        f"base: x/y\ntask: {task}\n"
        "data:\n  train: ./d.jsonl\n  format: chatml\n"
        "  max_length: 128\n" + data_extra,
        encoding="utf-8",
    )
    (tmp_path / "d.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok
    )
    # Control the rows directly so the assertions are about masking, not format
    # conversion. preprocess_dataset local-imports this name.
    monkeypatch.setattr(
        "soup_cli.data.loader.load_dataset", lambda *a, **k: {"train": rows}
    )
    result = CliRunner().invoke(app, ["data", "preprocess", "soup.yaml", "--yes"])
    assert result.exit_code == 0, result.output
    cache_dirs = [p for p in (tmp_path / ".soup-tokenized").iterdir() if p.is_dir()]
    assert len(cache_dirs) == 1, cache_dirs
    return cache_dirs[0]


def _load_cache(cache_dir):
    """Load the cache the way training loads it."""
    from soup_cli.utils.data_pipeline import load_pretokenized_dataset

    return load_pretokenized_dataset(str(cache_dir))


class TestCachedLabels:
    def test_cache_masks_the_same_tokens_the_live_path_masks(
        self, tmp_path, monkeypatch
    ):
        """Acceptance criterion #1. The cached rows' unmasked-token count equals
        the live path's, and is strictly greater than zero.

        ``build_assistant_only_labels`` IS the live ``train_on_responses_only``
        path (``data/sft_format.py`` calls it per row), so it is the baseline.
        The > 0 half matters on its own: a naive "the counts match" assertion
        passes when both sides degenerate to zero trained tokens -- the exact
        shape of the v0.73.3 all-``-100`` regression.

        regression: must fail without the fix -- pre-#1054 the cache has no
        ``labels`` column at all.
        """
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _tokenizer()
        rows = [{"messages": m} for m in _ROWS]
        ds = _load_cache(_run_preprocess(tmp_path, monkeypatch, tok, rows=rows))

        assert "labels" in ds.column_names, "the cache must carry a loss mask"
        assert len(ds) == len(_ROWS)
        for index, messages in enumerate(_ROWS):
            cached = list(ds[index]["labels"])
            cached_ids = list(ds[index]["input_ids"])
            built = build_assistant_only_labels(messages, tok, max_length=128)
            live = built["labels"]
            assert _unmasked(live) > 0, "sanity: the live path trains on some tokens"
            cached_trained = _trained(cached_ids, cached)
            live_trained = _trained(built["input_ids"], live)
            # Sequence, not count: a mask shifted one position keeps the count
            # exact while training on a prompt token. Multi-turn rows above give
            # this 2N boundaries to catch that at.
            assert cached_trained == live_trained, (
                f"row {index}: cache trains on {tok.decode(cached_trained)!r}, "
                f"live path on {tok.decode(live_trained)!r}"
            )
            assert len(cached) == len(cached_ids)

    def test_prompt_tokens_stay_masked(self, tmp_path, monkeypatch):
        """The defect's actual symptom: the prompt was trained on. The unmasked
        region must be a strict subset of the row, and must not cover the user
        turn's content tokens."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
            )
        )
        labels = list(ds[0]["labels"])
        input_ids = list(ds[0]["input_ids"])
        assert 0 < _unmasked(labels) < len(labels), (
            "an all-masked or all-unmasked row is the silent failure this guards"
        )
        france_id = len(_SPECIALS) + _WORDS.index("France")
        assert france_id in input_ids, "guard: the prompt token is in the row"
        for token, label in zip(input_ids, labels):
            if token == france_id:
                assert label == IGNORE_INDEX, "prompt tokens must not be trained on"

    def test_input_ids_are_unchanged_by_the_labels_fix(self, tmp_path, monkeypatch):
        """Control: only ``labels`` is new. The cached ``input_ids`` keep the
        #785 BOS de-duplication and the #791 training EOS -- the byte-parity this
        path is deliberately pinned to."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
            )
        )
        input_ids = list(ds[0]["input_ids"])
        assert input_ids.count(_BOS_ID) == 1, "#785: exactly one leading BOS"
        assert input_ids[-1] == _EOS_ID, "#791: the training EOS is still appended"

    def test_full_sequence_mode_trains_on_every_token(self, tmp_path, monkeypatch):
        """``train_on_responses_only: false`` caches an unmasked row -- the mask
        follows the config rather than being hardcoded."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path,
                monkeypatch,
                tok,
                rows=[{"messages": _ROWS[0]}],
                data_extra="  train_on_responses_only: false\n",
            )
        )
        labels = list(ds[0]["labels"])
        assert labels == list(ds[0]["input_ids"])

    def test_pretrain_rows_are_unmasked(self, tmp_path, monkeypatch):
        """Control: pretraining trains on every token by design -- no masking is
        added to the raw-text branch."""
        tok = _tokenizer()
        ds = _load_cache(
            _run_preprocess(
                tmp_path,
                monkeypatch,
                tok,
                task="pretrain",
                rows=[{"text": "You are terse ."}],
            )
        )
        assert list(ds[0]["labels"]) == list(ds[0]["input_ids"])


class TestMaskFlagsReachTheCache:
    """Every flag the live builder reads has to reach the cache AND the key.

    ``mask_history`` (#761) and ``train_on_eot`` both change which tokens the
    live path trains on, so a cache that ignores either trains on a different
    set under a key that claims otherwise -- #1054 exactly, one flag along.

    The default fixture template cannot show this: ``_TEMPLATE`` renders no EOS
    after the assistant span, so ``include_eot`` is inert on it and a test built
    there passes whether or not the flag is forwarded. ``_GEN_TEMPLATE`` puts the
    EOS OUTSIDE ``{% generation %}``, where the flag decides whether it is
    trained -- measured on the live builder, the four combinations give 4 / 2 /
    6 / 3 trained tokens, all distinct.
    """

    def _cached_trained(self, tmp_path, monkeypatch, tok, *, messages, data_extra):
        ds = _load_cache(
            _run_preprocess(
                tmp_path,
                monkeypatch,
                tok,
                rows=[{"messages": messages}],
                data_extra=data_extra,
            )
        )
        return _trained(list(ds[0]["input_ids"]), list(ds[0]["labels"]))

    def test_mask_history_cache_matches_the_live_path(self, tmp_path, monkeypatch):
        """(a) ``mask_history: true`` on a multi-turn row: the cache trains the
        LAST assistant span only, and exactly the tokens the live path trains.

        regression: must fail without the fix -- the cache ignored
        ``mask_history`` and trained every span.
        """
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _tokenizer()
        messages = _ROWS[2]
        cached = self._cached_trained(
            tmp_path, monkeypatch, tok,
            messages=messages, data_extra="  mask_history: true\n",
        )
        built = build_assistant_only_labels(
            messages, tok, max_length=128, mask_history=True
        )
        live = _trained(built["input_ids"], built["labels"])
        assert live, "sanity: the live path trains on some tokens"
        assert cached == live, (
            f"cache trains on {tok.decode(cached)!r}, live path on "
            f"{tok.decode(live)!r}"
        )
        # And it is genuinely narrower than the unmasked-history default, so a
        # cache that dropped the flag could not pass by coincidence.
        without = build_assistant_only_labels(messages, tok, max_length=128)
        assert len(live) < len(_trained(without["input_ids"], without["labels"]))

    def test_mask_history_changes_the_cache_key(self, tmp_path, monkeypatch):
        """(b) The flag changes the cached row, so it must change the key --
        otherwise a cache built one way loads silently under the other."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.utils.data_pipeline import (
            make_preprocess_cache_key,
            preprocess_mask_mode,
        )

        common = dict(
            dataset_path="d.jsonl", tokenizer_name="x/y",
            max_length=128, format_name="chatml", task="sft",
        )
        keys = {
            make_preprocess_cache_key(
                **common,
                mask_mode=preprocess_mask_mode(
                    DataConfig(train="d.jsonl", mask_history=history),
                    SimpleNamespace(train_on_eot=eot),
                ),
            )
            for history in (False, True)
            for eot in (False, True)
        }
        assert len(keys) == 4, "each mask-flag combination needs its own key"

    def test_train_on_eot_changes_what_the_cache_trains(self, tmp_path, monkeypatch):
        """(c) ``train_on_eot`` on a template whose EOS sits outside the
        assistant span, alone and combined with ``mask_history``.

        The combined case is what kills ``endswith("+eot")``: the mode is then
        ``responses_only+eot+mask_history``, ``endswith`` is False, and the cache
        silently drops the EOT while the key says it has it.

        regression: must fail without the fix -- both against ``include_eot``
        hardcoded off and against ``endswith`` matching.
        """
        from soup_cli.data.loss_mask import build_assistant_only_labels

        tok = _tokenizer()
        tok.chat_template = _GEN_TEMPLATE
        messages = _ROWS[2]
        for history in (False, True):
            data_extra = "  mask_history: true\n" if history else ""
            cached = self._cached_trained(
                tmp_path / f"eot-{history}", monkeypatch, tok,
                messages=messages,
                data_extra=data_extra + "training:\n  train_on_eot: true\n",
            )
            built = build_assistant_only_labels(
                messages, tok, max_length=128,
                include_eot=True, mask_history=history,
            )
            live = _trained(built["input_ids"], built["labels"])
            assert cached == live, (
                f"mask_history={history}: cache trains on {tok.decode(cached)!r}, "
                f"live path on {tok.decode(live)!r}"
            )
            # The EOS must actually be in the trained set, or this asserts
            # nothing about the flag.
            assert live[-1] == _EOS_ID, "guard: train_on_eot trains the EOS"
            no_eot = build_assistant_only_labels(
                messages, tok, max_length=128, mask_history=history
            )
            assert len(live) > len(_trained(no_eot["input_ids"], no_eot["labels"]))


_TF_ROW = [
    {"role": "system", "content": "You are terse .", "train": False},
    {"role": "user", "content": "What is the capital of France ?", "train": True},
    {"role": "assistant", "content": "Paris .", "train": False},
    {"role": "user", "content": "What is the capital of Germany ?"},
    {"role": "assistant", "content": "Berlin ."},
]


class TestTrainFieldCache:
    def test_train_field_cache_matches_the_live_path(self, tmp_path, monkeypatch):
        """The one mode whose mask depends on per-message data: the cache must
        train exactly what ``build_per_message_train_labels`` trains live.

        regression: must fail if the ``train_field`` branch builds the
        assistant-only mask or returns ``input_ids`` like ``full``.
        """
        from soup_cli.data.loss_mask import (
            build_assistant_only_labels,
            build_per_message_train_labels,
        )

        tok = _tokenizer()
        ds = _load_cache(_run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _TF_ROW}],
            data_extra="  train_on_responses_only: false\n"
                       "  train_on_messages_with_train_field: true\n",
        ))
        cached = _trained(list(ds[0]["input_ids"]), list(ds[0]["labels"]))
        built = build_per_message_train_labels(_TF_ROW, tok, max_length=128)
        live = _trained(built["input_ids"], built["labels"])
        assert cached == live
        other = build_assistant_only_labels(_TF_ROW, tok, max_length=128)
        assert live != _trained(other["input_ids"], other["labels"]), "guard: modes differ"
        assert len(live) < len(ds[0]["input_ids"]), "guard: not every token"


class TestEveryTrainerPassesTheTrainingConfig:
    """Every caller of the gate must hand it ``training``, or its cache is
    unloadable.

    ``preprocess_mask_mode`` reads ``training.train_on_eot``; a caller that omits
    it derives a different mode than ``soup data preprocess`` wrote and the cache
    is refused by a hash that re-running preprocess reproduces exactly -- a loop
    with no way out. A behavioural test per caller does not scale: it pins the
    callers that exist when it is written and silently ignores the next trainer
    to grow a ``pre_tokenized`` path. Read the call sites out of the AST instead,
    so a new one is covered the day it is added.

    regression: must fail without the fix -- on main ``pretrain.py`` passes three
    arguments. Dropping ``tcfg`` from EITHER call site fails this.
    """

    def _gate_calls(self):
        import ast
        from pathlib import Path

        import soup_cli.trainer as trainer_pkg

        found = []
        for path in sorted(Path(trainer_pkg.__file__).parent.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_maybe_load_pretokenized"
                ):
                    found.append((path.name, node))
        return found

    def test_every_call_site_forwards_a_training_config(self):
        calls = self._gate_calls()
        # Guard: an empty sweep would pass vacuously if the helper were renamed.
        names = {name for name, _ in calls}
        assert {"sft.py", "pretrain.py"} <= names, (
            f"expected the known call sites, found {sorted(names)} -- if the "
            "helper moved, re-point this sweep rather than deleting it"
        )
        import ast

        for name, node in calls:
            passed = len(node.args) + len(node.keywords)
            assert passed >= 4, (
                f"{name}:{node.lineno} calls _maybe_load_pretokenized with "
                f"{passed} arguments; it must also forward the training config, "
                "or a train_on_eot cache built by `soup data preprocess` can "
                "never be loaded back (#1054)"
            )
            # A count alone accepts ``None`` or ``cfg`` in the fourth slot.
            keywords = {kw.arg: kw.value for kw in node.keywords}
            fourth = node.args[3] if len(node.args) > 3 else keywords.get("tcfg")
            assert fourth is not None, f"{name}:{node.lineno} passes no training config"
            source = ast.unparse(fourth)
            assert source == "tcfg" or "training" in source, (
                f"{name}:{node.lineno} forwards {source!r}, not the training config"
            )


class TestMissingLabelsIsRefused:
    def test_validate_raises_on_a_cache_without_labels(self):
        """A stale (pre-fix) or externally produced cache must be refused by name
        rather than silently skipping the causal-loss-target check -- skipping is
        what let the label-less cache reach TRL's ``labels = input_ids`` fallback.

        regression: must fail without the fix -- the old code returned here.
        """
        from soup_cli.trainer.sft import _validate_pretokenized_targets

        dataset = SimpleNamespace(column_names=["input_ids", "attention_mask"])
        with pytest.raises(ValueError, match="no 'labels' column") as exc:
            _validate_pretokenized_targets(dataset, split="train", max_length=128)
        # An external dataset has no preprocess run to repeat.
        assert "add a 'labels' column" in str(exc.value)


class TestCacheKeyCoversMaskMode:
    @pytest.mark.parametrize("bad", ["", "responses_only\x00", None])
    def test_bad_mask_mode_is_rejected(self, bad):
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        with pytest.raises(ValueError, match="mask_mode"):
            make_preprocess_cache_key(
                dataset_path="./d.jsonl",
                tokenizer_name="x/y",
                max_length=128,
                format_name="chatml",
                mask_mode=bad,
                task="sft",
            )

    @pytest.mark.parametrize(
        ("dropped", "expected", "absent"),
        [
            (("mask_mode",), "predates loss-mask keying (#1054)", "#1067"),
            (
                ("mask_mode", "chat_template"),
                "predates chat_template keying (#1067)",
                "#1054",
            ),
        ],
    )
    def test_an_older_cache_names_the_gap(
        self, tmp_path, monkeypatch, dropped, expected, absent
    ):
        """A v5-shaped cache (no ``mask_mode`` in metadata, an older key) names
        #1054; one also missing ``chat_template`` names only the older #1067."""
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
        )
        meta_path = cache_dir / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for field in dropped:
            meta.pop(field)
        meta["cache_key"] = "0" * 16  # the older schema hashed other inputs
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
        )
        with pytest.raises(ValueError, match="cache hash mismatch") as exc:
            _maybe_load_pretokenized(dcfg, "x/y", Console())
        assert expected in str(exc.value)
        assert absent not in str(exc.value)

    def test_mask_mode_changes_the_key(self):
        """Acceptance criterion #2, at the hash. Same dataset / tokenizer /
        max_length / format under a different masking config must not collide."""
        from soup_cli.utils.data_pipeline import make_preprocess_cache_key

        common = dict(
            dataset_path="./d.jsonl",
            tokenizer_name="x/y",
            max_length=128,
            format_name="chatml",
            task="sft",
        )
        keys = {
            make_preprocess_cache_key(**common, mask_mode=mode)
            for mode in ("responses_only", "train_field", "full")
        }
        assert len(keys) == 3, "each masking mode must hash to its own cache key"

    def test_train_on_eot_is_a_distinct_mask_mode(self):
        """``training.train_on_eot`` widens the live mask over the trailing EOT,
        so a cache built without it must not be reused by a run with it -- the
        same defect shape as #1054 itself, one config layer up."""
        from soup_cli.utils.data_pipeline import preprocess_mask_mode

        dcfg = SimpleNamespace(
            train_on_responses_only=True, train_on_messages_with_train_field=False
        )
        plain = preprocess_mask_mode(dcfg, SimpleNamespace(train_on_eot=False))
        eot = preprocess_mask_mode(dcfg, SimpleNamespace(train_on_eot=True))
        assert plain == "responses_only"
        assert eot == "responses_only+eot"
        assert plain != eot
        # train_on_eot only reaches the assistant-only builder, as in
        # build_format_row -- the other two modes ignore it.
        for other in (
            SimpleNamespace(
                train_on_responses_only=False,
                train_on_messages_with_train_field=True,
            ),
            SimpleNamespace(
                train_on_responses_only=False,
                train_on_messages_with_train_field=False,
            ),
        ):
            assert preprocess_mask_mode(
                other, SimpleNamespace(train_on_eot=True)
            ) == preprocess_mask_mode(other, SimpleNamespace(train_on_eot=False))

    def test_mask_mode_mirrors_build_format_row(self):
        """Both call sites derive the mode from one function, so they cannot
        drift the way the cache path drifted from the live path in #1054."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.utils.data_pipeline import preprocess_mask_mode

        assert preprocess_mask_mode(DataConfig(train="d.jsonl")) == "responses_only"
        assert (
            preprocess_mask_mode(
                DataConfig(train="d.jsonl", train_on_responses_only=False)
            )
            == "full"
        )
        assert (
            preprocess_mask_mode(
                DataConfig(
                    train="d.jsonl",
                    train_on_responses_only=False,
                    train_on_messages_with_train_field=True,
                )
            )
            == "train_field"
        )

    def test_cache_built_under_another_mask_mode_is_refused(
        self, tmp_path, monkeypatch
    ):
        """Acceptance criterion #2, end to end: a cache preprocessed with the
        default assistant-only mask is refused when loaded by a config that asks
        for the per-message ``train`` field mask.

        regression: must fail without the fix -- the key ignored the masking
        config, so the stale cache loaded silently under the wrong objective.
        """
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
        )
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
            train_on_responses_only=False,
            train_on_messages_with_train_field=True,
        )
        with pytest.raises(ValueError, match="cache hash mismatch"):
            _maybe_load_pretokenized(dcfg, "x/y", Console())

    def test_matching_mask_mode_still_loads(self, tmp_path, monkeypatch):
        """Control: the gate must not reject a cache built under the SAME config
        -- a key that rejects everything would pass the test above."""
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path, monkeypatch, tok, rows=[{"messages": _ROWS[0]}]
        )
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
        )
        loaded = _maybe_load_pretokenized(dcfg, "x/y", Console())
        assert loaded is not None
        train_ds, _ = loaded
        assert len(train_ds) == 1

    def test_train_on_eot_cache_loads_through_the_pretrain_caller(
        self, tmp_path, monkeypatch
    ):
        """``task: pretrain`` + ``train_on_eot: true`` is a config the schema
        allows (pretrain is in the sft-family set). ``trainer/pretrain.py`` must
        derive the same mask mode ``soup data preprocess`` wrote, or the cache is
        unloadable and re-running preprocess as the error advises regenerates the
        very same rejected key.

        regression: must fail without the fix -- ``pretrain.py`` called the gate
        without ``tcfg``, dropping the ``+eot`` suffix on its side only.
        """
        from rich.console import Console

        from soup_cli.config.schema import DataConfig
        from soup_cli.trainer.sft import _maybe_load_pretokenized

        tok = _tokenizer()
        cache_dir = _run_preprocess(
            tmp_path,
            monkeypatch,
            tok,
            task="pretrain",
            rows=[{"text": "You are terse ."}],
            data_extra="training:\n  train_on_eot: true\n",
        )
        dcfg = DataConfig(
            train="./d.jsonl",
            format="pre_tokenized",
            tokenized_path=str(cache_dir.relative_to(tmp_path)),
            max_length=128,
        )
        tcfg = SimpleNamespace(train_on_eot=True)
        # Exactly what trainer/pretrain.py now passes.
        loaded = _maybe_load_pretokenized(dcfg, "x/y", Console(), tcfg, task="pretrain")
        assert loaded is not None
        train_ds, _ = loaded
        assert len(train_ds) == 1


class TestSampleCount:
    def test_pretokenized_count_comes_from_the_cache(self, tmp_path):
        """Acceptance criterion #3. ``load_dataset`` sees the ORIGINAL chatml file
        under a ``pre_tokenized`` config and drops every row for want of an
        ``input_ids`` column, so ``soup train`` printed "0 train samples" for a run
        that then trained on the whole cache.

        regression: must fail without the fix -- the count was
        ``len(dataset['train'])``, i.e. 0.
        """
        from soup_cli.commands.train import _train_sample_count

        cache = tmp_path / "cache"
        cache.mkdir(parents=True)
        (cache / "metadata.json").write_text(
            json.dumps({"cache_key": "deadbeef", "row_count": 6}), encoding="utf-8"
        )
        dcfg = SimpleNamespace(format="pre_tokenized", tokenized_path=str(cache))
        assert _train_sample_count(dcfg, {"train": []}) == 6

    def test_non_pretokenized_count_is_the_loaders_view(self):
        """Control: every other format still reports what the loader returned."""
        from soup_cli.commands.train import _train_sample_count

        dcfg = SimpleNamespace(format="chatml", tokenized_path=None)
        assert _train_sample_count(dcfg, {"train": [1, 2, 3]}) == 3

    def test_negative_row_count_falls_back(self, tmp_path):
        """Control: a corrupt ``row_count`` must not print a negative count."""
        from soup_cli.commands.train import _train_sample_count

        (tmp_path / "metadata.json").write_text(
            json.dumps({"row_count": -3}), encoding="utf-8"
        )
        dcfg = SimpleNamespace(format="pre_tokenized", tokenized_path=str(tmp_path))
        assert _train_sample_count(dcfg, {"train": [1, 2]}) == 2

    def test_unreadable_metadata_falls_back(self, tmp_path):
        """Control: a cache without usable metadata must not crash the launch."""
        from soup_cli.commands.train import _train_sample_count

        dcfg = SimpleNamespace(
            format="pre_tokenized", tokenized_path=str(tmp_path / "missing")
        )
        assert _train_sample_count(dcfg, {"train": [1, 2]}) == 2
