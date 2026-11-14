"""Prefix-tree batches (``soup_cli.utils.prefix_tree``).

Examples that begin with the same tokens are laid out as one tree: the shared
beginning once, the continuations after it, a 4D ancestor mask and depth
positions. The claim under test is equality with the ordinary batch holding the
same examples:

- the layout (rows, parents, positions, labels, repeated parents) by hand and by
  a brute-force reconstruction of every example from its row;
- logits bit-identical on CPU, the float64 loss bit-identical, gradients to
  float rounding, on Llama, Mistral, Qwen2 and GPT-2, with the stock shifted
  loss and through a layer-streamed model;
- the sampler and collator keep every example exactly once per epoch and never
  exceed the step's rows or tokens.
"""

from __future__ import annotations

import random

import pytest

from soup_cli.utils.prefix_tree import (
    IGNORE_INDEX,
    PrefixTreeBatchSampler,
    PrefixTreeCollator,
    ancestor_mask,
    group_by_shared_prefix,
    pack_prefix_tree,
    prefix_tree_stats,
    shareable_prefix,
)

X = IGNORE_INDEX


def _example(prompt, answer):
    """Prompt tokens are not trained on; answer tokens are."""
    return list(prompt) + list(answer), [X] * len(prompt) + list(answer)


def _rebuild(row):
    """Every member's tokens, positions and attention set, read back from the row."""
    mask = ancestor_mask(row.parents)
    out = []
    for path in row.paths:
        tokens = [row.input_ids[index] for index in path]
        positions = [row.position_ids[index] for index in path]
        sees = [
            sorted(other for other in range(len(row)) if mask[index][other]) for index in path
        ]
        out.append((tokens, positions, sees))
    return out


def _assert_layout_is_faithful(examples, rows):
    """Each example is in exactly one row, token for token, at its own positions,
    seeing exactly its own earlier tokens; each trained token is scored once, by
    the stock shift, against the prediction made at its predecessor."""
    seen = sorted(member for row in rows for member in row.members)
    assert seen == list(range(len(examples)))
    for row in rows:
        scored = [index for index, label in enumerate(row.labels) if label != X]
        claimed = []
        for member, path, (tokens, positions, sees) in zip(row.members, row.paths, _rebuild(row)):
            ids, labels = examples[member]
            assert tokens == list(ids)
            assert positions == list(range(len(ids)))
            for at, index in enumerate(path):
                assert sees[at] == sorted(path[: at + 1]), "a token sees more than its ancestors"
            for at, label in enumerate(labels):
                if at == 0 or label == X:
                    continue
                index = path[at]
                assert row.labels[index] == label
                # The stock loss scores logits[index - 1] against labels[index].
                assert index >= 1 and path[at - 1] == index - 1
                claimed.append(index)
        assert sorted(claimed) == scored, "a trained token is scored twice or not at all"


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------
class TestPacking:
    def test_a_shared_prompt_is_held_once(self):
        examples = [_example([1, 2, 3], [10, 11]), _example([1, 2, 3], [20])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        #            prompt     answer A   3 again  answer B
        assert row.input_ids == (1, 2, 3, 10, 11, 3, 20)
        assert row.position_ids == (0, 1, 2, 3, 4, 2, 3)
        assert row.parents == (-1, 0, 1, 2, 3, 1, 5)
        assert row.labels == (X, X, X, 10, 11, X, 20)
        assert row.paths == ((0, 1, 2, 3, 4), (0, 1, 5, 6))
        assert row.members == (0, 1)
        assert row.source_tokens == 9 and len(row) == 7
        _assert_layout_is_faithful(examples, [row])

    def test_the_first_branch_needs_no_repeated_parent(self):
        examples = [_example([1, 2, 3], [10]), _example([1, 2, 3], [20]), _example([1, 2, 3], [30])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.input_ids == (1, 2, 3, 10, 3, 20, 3, 30)
        assert list(row.input_ids).count(3) == 3, "one original and one per later branch"

    def test_an_untrained_branch_start_needs_no_repeated_parent(self):
        """Only a TRAINED token has to follow its parent in the row."""
        examples = [_example([1, 2, 7], [10]), _example([1, 2, 8], [20])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.input_ids == (1, 2, 7, 10, 8, 20)
        assert row.parents == (-1, 0, 1, 2, 1, 4)
        _assert_layout_is_faithful(examples, [row])

    def test_trained_tokens_are_never_shared(self):
        """Two copies of one example: the prompt once, the answer twice, so the
        loss's token-mean weighs the answer as the ordinary batch does."""
        examples = [_example([1, 2], [10, 11]), _example([1, 2], [10, 11])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.input_ids == (1, 2, 10, 11, 2, 10, 11)
        assert [label for label in row.labels if label != X] == [10, 11, 10, 11]
        _assert_layout_is_faithful(examples, [row])

    def test_sharing_stops_at_the_first_trained_token(self):
        """A later untrained stretch (the next user turn) is past a trained token."""
        ids = [1, 2, 10, 5, 6, 30]
        labels = [X, X, 10, X, X, 30]
        other = ([1, 2, 10, 5, 6, 31], [X, X, 10, X, X, 31])
        (row,) = pack_prefix_tree([(ids, labels), other], max_tokens=64)
        assert row.input_ids == (1, 2, 10, 5, 6, 30, 2, 10, 5, 6, 31)
        _assert_layout_is_faithful([(ids, labels), other], [row])

    def test_the_first_token_is_shareable_whatever_its_label(self):
        """``labels = input_ids`` (train on everything): the stock shift never
        scores position 0, so that one token is shared and its label ignored."""
        examples = [([1, 5, 6], [1, 5, 6]), ([1, 7], [1, 7])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.input_ids == (1, 5, 6, 1, 7)
        assert row.labels == (X, 5, 6, X, 7)
        _assert_layout_is_faithful(examples, [row])

    def test_an_example_that_is_a_prefix_of_another_adds_nothing(self):
        examples = [_example([1, 2, 3, 4], [10]), ([1, 2, 3], [X, X, X])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert len(row) == 5
        assert row.paths[1] == (0, 1, 2)

    def test_several_levels_of_sharing(self):
        """System prompt for all, a document for some, a question for a pair."""
        system, doc_a, doc_b = [1, 2, 3], [4, 5], [6]
        examples = [
            _example(system + doc_a + [7], [10]),
            _example(system + doc_a + [7], [11]),
            _example(system + doc_a + [8], [12]),
            _example(system + doc_b, [13]),
        ]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.source_tokens == 7 + 7 + 7 + 5
        assert len(row) == 3 + 2 + (1 + 1) + (1 + 1) + (1 + 1) + (1 + 1)
        _assert_layout_is_faithful(examples, [row])

    def test_rows_respect_the_token_budget(self):
        examples = [_example([1, 2, 3, 4], [10 + n]) for n in range(6)]
        rows = pack_prefix_tree(examples, max_tokens=9)
        # 4 + 1, then 2 per further branch: 5, 7, 9 -> three examples per row.
        assert [len(row) for row in rows] == [9, 9]
        assert [row.members for row in rows] == [(0, 1, 2), (3, 4, 5)]
        _assert_layout_is_faithful(examples, rows)

    def test_a_row_always_takes_its_first_example(self):
        rows = pack_prefix_tree([_example(range(1, 20), [30]), _example([1], [31])], max_tokens=8)
        assert [len(row) for row in rows] == [20, 2]

    def test_unrelated_examples_are_a_forest_in_one_row(self):
        examples = [_example([1, 2], [10]), _example([3, 4], [11])]
        (row,) = pack_prefix_tree(examples, max_tokens=64)
        assert row.input_ids == (1, 2, 10, 3, 4, 11)
        assert row.parents == (-1, 0, 1, -1, 3, 4)
        assert row.position_ids == (0, 1, 2, 0, 1, 2)
        _assert_layout_is_faithful(examples, [row])

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, None])
    def test_a_bad_budget_is_refused(self, bad):
        with pytest.raises(ValueError, match="max_tokens"):
            pack_prefix_tree([_example([1], [2])], max_tokens=bad)

    def test_mismatched_or_empty_examples_are_refused(self):
        with pytest.raises(ValueError, match="one label per token"):
            pack_prefix_tree([([1, 2], [X])], max_tokens=8)
        with pytest.raises(ValueError, match="empty"):
            pack_prefix_tree([([], [])], max_tokens=8)

    def test_nothing_to_pack(self):
        assert pack_prefix_tree([], max_tokens=8) == []

    @pytest.mark.parametrize("seed", range(25))
    def test_random_examples_are_laid_out_faithfully(self, seed):
        rng = random.Random(seed)
        stems = [[rng.randrange(1, 9) for _ in range(rng.randrange(1, 12))] for _ in range(4)]
        examples = []
        for _ in range(rng.randrange(2, 14)):
            stem = list(rng.choice(stems))[: rng.randrange(1, 12)]
            tail = [rng.randrange(1, 9) for _ in range(rng.randrange(0, 6))]
            ids = stem + tail
            cut = rng.randrange(0, len(ids) + 1)
            labels = [
                X if at < cut or rng.random() < 0.2 else token for at, token in enumerate(ids)
            ]
            examples.append((ids, labels))
        budget = rng.choice([6, 12, 24, 1000])
        rows = pack_prefix_tree(examples, max_tokens=budget)
        _assert_layout_is_faithful(examples, rows)
        for row in rows:
            assert len(row) <= budget or len(row.members) == 1
        stats = prefix_tree_stats(rows)
        assert stats.examples == len(examples)
        assert stats.source_tokens == sum(len(ids) for ids, _ in examples)


class TestAncestorMask:
    def test_by_hand(self):
        mask = ancestor_mask((-1, 0, 1, 0, -1))
        assert mask == [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, False, False, True, False],
            [False, False, False, False, True],
        ]

    @pytest.mark.parametrize("parents", [(0,), (-1, 1), (-1, 2, 0), (-2,)])
    def test_a_parent_must_precede_its_child(self, parents):
        with pytest.raises(ValueError, match="precede"):
            ancestor_mask(parents)


# --------------------------------------------------------------------------
# which examples meet
# --------------------------------------------------------------------------
class TestGrouping:
    def test_shareable_prefix_ends_at_the_first_trained_token(self):
        assert shareable_prefix([1, 2, 3, 4], [X, X, 3, X], X) == (1, 2)
        assert shareable_prefix([1, 2], [1, 2], X) == (1,), "position 0 is never scored"
        assert shareable_prefix([1, 2], [X, X], X) == (1, 2)

    def test_one_system_prompt_is_one_flat_group(self):
        prefixes = [list(range(40)) + [100 + n] for n in range(5)]
        (group,) = group_by_shared_prefix(prefixes, min_group_tokens=16)
        assert sorted(group) == [0, 1, 2, 3, 4]

    def test_a_document_per_few_examples_nests(self):
        system = list(range(20))
        doc_a = system + list(range(100, 130))
        doc_b = system + list(range(200, 230))
        prefixes = [doc_a + [1], doc_b + [2], doc_a + [3], system + [900], doc_b + [4]]
        (top,) = group_by_shared_prefix(prefixes, min_group_tokens=16)
        nested = sorted(sorted(child) for child in top if isinstance(child, list))
        assert nested == [[0, 2], [1, 4]]
        assert [child for child in top if not isinstance(child, list)] == [3]

    def test_a_short_common_beginning_does_not_group(self):
        prefixes = [[1, 2, 3, 10], [1, 2, 3, 11], [9, 9]]
        assert sorted(group_by_shared_prefix(prefixes, min_group_tokens=16)) == [0, 1, 2]
        (pair, single) = sorted(
            group_by_shared_prefix(prefixes, min_group_tokens=3), key=lambda c: isinstance(c, int)
        )
        assert sorted(pair) == [0, 1] and single == 2

    @pytest.mark.parametrize("bad", [0, -3])
    def test_a_bad_threshold_is_refused(self, bad):
        with pytest.raises(ValueError, match="min_group_tokens"):
            group_by_shared_prefix([[1]], min_group_tokens=bad)


def _dataset(seed=0, documents=6, per_document=5, extra=7):
    rng = random.Random(seed)
    system = [rng.randrange(5, 50) for _ in range(24)]
    examples = []
    for _ in range(documents):
        document = system + [rng.randrange(50, 99) for _ in range(rng.randrange(20, 40))]
        for _ in range(per_document):
            question = [rng.randrange(100, 150) for _ in range(rng.randrange(3, 8))]
            answer = [rng.randrange(150, 199) for _ in range(rng.randrange(2, 9))]
            examples.append(_example(document + question, answer))
    for _ in range(extra):
        examples.append(_example(system + [rng.randrange(200, 250)], [rng.randrange(150, 199)]))
    rng.shuffle(examples)
    return examples


class TestSampler:
    def test_every_example_once_per_epoch(self):
        examples = _dataset()
        sampler = PrefixTreeBatchSampler(examples, max_tokens=96, batch_size=2, seed=3)
        for _ in range(3):
            batches = list(sampler)
            assert len(batches) == len(sampler)
            assert sorted(i for batch in batches for i in batch) == list(range(len(examples)))

    def test_documents_meet_in_a_row(self):
        examples = _dataset()
        sampler = PrefixTreeBatchSampler(examples, max_tokens=96, batch_size=1, seed=3)
        assert sampler.stats.examples == len(examples)
        assert sampler.stats.ratio > 2.0, sampler.stats
        shuffled = pack_prefix_tree(examples, max_tokens=96)
        assert sampler.stats.tree_tokens < prefix_tree_stats(shuffled).tree_tokens

    def test_a_batch_repacks_within_the_steps_budget(self):
        """The collator packs a batch's examples again; never more rows than the
        step's, never a row over the budget (see the sampler's docstring)."""
        for seed in range(12):
            examples = _dataset(seed=seed)
            for batch_size in (1, 2, 3):
                sampler = PrefixTreeBatchSampler(
                    examples, max_tokens=80, batch_size=batch_size, seed=seed
                )
                for batch in sampler:
                    rows = pack_prefix_tree([examples[i] for i in batch], max_tokens=80)
                    assert len(rows) <= batch_size
                    assert all(len(row) <= 80 or len(row.members) == 1 for row in rows)

    def test_seeded_and_reshuffled_per_epoch(self):
        examples = _dataset()
        first = PrefixTreeBatchSampler(examples, max_tokens=96, batch_size=2, seed=5)
        again = PrefixTreeBatchSampler(examples, max_tokens=96, batch_size=2, seed=5)
        other = PrefixTreeBatchSampler(examples, max_tokens=96, batch_size=2, seed=6)
        assert first.rows == again.rows
        epoch_0, epoch_1 = list(first), list(first)
        assert epoch_0 == list(again)
        assert epoch_0 != epoch_1, "the same batches in the same order every epoch"
        assert len(epoch_0) == len(epoch_1)
        assert first.rows != other.rows
        first.set_epoch(0)
        assert list(first) == epoch_0

    def test_drop_last(self):
        examples = [_example([n, n + 1, n + 2, n + 3], [9]) for n in range(1, 16, 5)]
        keep = PrefixTreeBatchSampler(examples, max_tokens=5, batch_size=2)
        drop = PrefixTreeBatchSampler(examples, max_tokens=5, batch_size=2, drop_last=True)
        assert len(keep.rows) == 3
        assert len(keep) == 2 and len(list(keep)) == 2
        assert len(drop) == 1 and len(list(drop)) == 1

    def test_refusals(self):
        with pytest.raises(ValueError, match="no examples"):
            PrefixTreeBatchSampler([], max_tokens=8, batch_size=1)
        with pytest.raises(ValueError, match="batch_size"):
            PrefixTreeBatchSampler([_example([1], [2])], max_tokens=8, batch_size=0)
        with pytest.raises(TypeError, match="seed"):
            PrefixTreeBatchSampler([_example([1], [2])], max_tokens=8, batch_size=1, seed="x")
        with pytest.raises(ValueError, match="one label per token"):
            PrefixTreeBatchSampler([([1, 2], [X])], max_tokens=8, batch_size=1)


# --------------------------------------------------------------------------
# tensors
# --------------------------------------------------------------------------
def _pad_collate(pad=0, side="right"):
    """What a stock causal-LM collator returns: padded ids, a 2D mask, labels."""

    def collate(features):
        import torch

        width = max(len(ids) for ids, _ in features)
        ids = torch.full((len(features), width), pad, dtype=torch.long)
        mask = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.full((len(features), width), X, dtype=torch.long)
        for row, (tokens, row_labels) in enumerate(features):
            span = slice(0, len(tokens)) if side == "right" else slice(width - len(tokens), width)
            ids[row, span] = torch.tensor(tokens)
            mask[row, span] = 1
            labels[row, span] = torch.tensor(row_labels)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


class TestCollation:
    def test_shapes_padding_and_the_mask(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree import collate_prefix_tree

        rows = pack_prefix_tree(
            [_example([1, 2, 3], [10, 11]), _example([1, 2, 3], [20]), _example([4], [5])],
            max_tokens=7,
        )
        assert [len(row) for row in rows] == [7, 2]
        batch = collate_prefix_tree(rows, pad_token_id=99)
        assert batch["input_ids"].shape == (2, 7)
        assert batch["attention_mask"].shape == (2, 1, 7, 7)
        assert batch["attention_mask"].dtype == torch.bool
        assert batch["input_ids"][1].tolist() == [4, 5, 99, 99, 99, 99, 99]
        assert batch["labels"][1].tolist() == [X, 5, X, X, X, X, X]
        assert batch["position_ids"][1].tolist() == [0, 1, 0, 0, 0, 0, 0]
        for r, row in enumerate(rows):
            expected = torch.eye(7, dtype=torch.bool)
            expected[: len(row), : len(row)] = torch.tensor(ancestor_mask(row.parents))
            assert torch.equal(batch["attention_mask"][r, 0], expected)
        assert bool(batch["attention_mask"].any(dim=-1).all()), "a row with no key is NaN"

    def test_the_additive_form(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree import collate_prefix_tree

        rows = pack_prefix_tree([_example([1, 2], [3]), _example([1, 2], [4])], max_tokens=16)
        boolean = collate_prefix_tree(rows, pad_token_id=0)["attention_mask"]
        additive = collate_prefix_tree(rows, pad_token_id=0, additive_dtype=torch.float32)[
            "attention_mask"
        ]
        assert additive.dtype == torch.float32
        assert torch.equal(additive == 0, boolean)
        assert float(additive.min()) == torch.finfo(torch.float32).min

    def test_no_rows_is_refused(self):
        pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree import collate_prefix_tree

        with pytest.raises(ValueError, match="no rows"):
            collate_prefix_tree([], pad_token_id=0)

    @pytest.mark.parametrize("side", ["right", "left"])
    def test_the_collator_strips_padding_from_either_side(self, side):
        pytest.importorskip("torch")
        examples = [_example([1, 2, 3], [10, 11]), _example([1, 2, 3], [20]), _example([7], [8])]
        collator = PrefixTreeCollator(
            _pad_collate(side=side), max_tokens=64, pad_token_id=0, max_rows=1
        )
        batch = collator(examples)
        assert batch["input_ids"].tolist() == [[1, 2, 3, 10, 11, 3, 20, 7, 8]]
        assert batch["labels"].tolist() == [[X, X, X, 10, 11, X, 20, X, 8]]
        assert collator.examples == 3
        assert collator.source_tokens == 11 and collator.tree_tokens == 9

    def test_more_rows_than_the_step_allows_is_an_error(self):
        pytest.importorskip("torch")
        collator = PrefixTreeCollator(_pad_collate(), max_tokens=4, pad_token_id=0, max_rows=1)
        with pytest.raises(RuntimeError, match="disagree"):
            collator([_example([1, 2], [3]), _example([4, 5], [6])])

    def test_a_batch_key_it_cannot_carry_is_refused(self):
        torch = pytest.importorskip("torch")
        inner = _pad_collate()

        def with_extra(features):
            batch = inner(features)
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

        collator = PrefixTreeCollator(with_extra, max_tokens=64, pad_token_id=0)
        with pytest.raises(ValueError, match="token_type_ids"):
            collator([_example([1, 2], [3])])

    def test_bad_arguments(self):
        with pytest.raises(TypeError, match="callable"):
            PrefixTreeCollator(None, max_tokens=8, pad_token_id=0)
        with pytest.raises(TypeError, match="pad_token_id"):
            PrefixTreeCollator(_pad_collate(), max_tokens=8, pad_token_id=None)


# --------------------------------------------------------------------------
# the arithmetic, on real models
# --------------------------------------------------------------------------
def _tiny_model(arch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    common = dict(vocab_size=97, hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    if arch == "llama":
        config = transformers.LlamaConfig(
            intermediate_size=128, num_key_value_heads=2, max_position_embeddings=256, **common
        )
        model = transformers.LlamaForCausalLM(config)
    elif arch == "mistral":
        config = transformers.MistralConfig(
            intermediate_size=128, num_key_value_heads=2, max_position_embeddings=256, **common
        )
        model = transformers.MistralForCausalLM(config)
    elif arch == "qwen2":
        config = transformers.Qwen2Config(
            intermediate_size=128, num_key_value_heads=2, max_position_embeddings=256, **common
        )
        model = transformers.Qwen2ForCausalLM(config)
    elif arch == "gpt2":
        config = transformers.GPT2Config(
            vocab_size=97, n_embd=64, n_layer=2, n_head=4, n_positions=256,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        model = transformers.GPT2LMHeadModel(config)
    else:  # pragma: no cover
        raise AssertionError(arch)
    return model.double().train()


def _model_examples(seed=1):
    rng = random.Random(seed)

    def tokens(count):
        return [rng.randrange(3, 97) for _ in range(count)]

    system = tokens(18)
    document = tokens(9)
    return [
        _example(system + document + tokens(3), tokens(5)),
        _example(system + document + tokens(2), tokens(7)),
        _example(system + document, tokens(4)),
        _example(system + tokens(4), tokens(6)),
        _example(tokens(6), tokens(3)),
    ]


def _plain_batch(examples):
    return _pad_collate()(examples)


def _assert_same_logits(arch, got, want, what):
    """Bit for bit on the Llama family. GPT-2's Conv1D projections round
    differently for a different row count (3e-16 in float64), so there the
    comparison is to rounding."""
    import torch

    if arch == "gpt2":
        assert float((got - want).abs().max()) < 1e-13, what
    else:
        assert torch.equal(got, want), what


def _float64_loss(logits, labels):
    import torch

    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=X
    )


@pytest.mark.parametrize("arch", ["llama", "mistral", "qwen2", "gpt2"])
class TestExactness:
    def test_logits_match_token_by_token(self, arch):
        import torch

        from soup_cli.utils.prefix_tree import collate_prefix_tree

        model = _tiny_model(arch).eval()
        examples = _model_examples()
        rows = pack_prefix_tree(examples, max_tokens=1000)
        assert len(rows) == 1 and len(rows[0]) < rows[0].source_tokens
        packed = collate_prefix_tree(rows, pad_token_id=0)
        plain = _plain_batch(examples)
        with torch.no_grad():
            want = model(
                input_ids=plain["input_ids"], attention_mask=plain["attention_mask"]
            ).logits
            got = model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            ).logits
        for member, path in zip(rows[0].members, rows[0].paths):
            _assert_same_logits(arch, got[0, list(path)], want[member, : len(path)], member)

    def test_loss_and_gradients_match_the_ordinary_batch(self, arch):
        import torch

        from soup_cli.utils.prefix_tree import collate_prefix_tree

        model = _tiny_model(arch)
        examples = _model_examples()
        plain = _plain_batch(examples)
        model.zero_grad()
        want = _float64_loss(
            model(input_ids=plain["input_ids"], attention_mask=plain["attention_mask"]).logits,
            plain["labels"],
        )
        want.backward()
        want_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        packed = collate_prefix_tree(pack_prefix_tree(examples, max_tokens=1000), pad_token_id=0)
        model.zero_grad()
        got = _float64_loss(
            model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            ).logits,
            packed["labels"],
        )
        got.backward()
        # The same trained tokens, the same logits: the loss differs only by the
        # order its terms are summed in.
        assert abs(float(got.detach()) - float(want.detach())) < 1e-12
        largest = max(float(grad.abs().max()) for grad in want_grads.values())
        assert largest > 1e-3, "vacuous: no gradient"
        for name, param in model.named_parameters():
            # Not bit-identical: the shared tokens' gradient is one sum instead
            # of one per example, and Llama-family norms round through float32.
            assert torch.allclose(param.grad, want_grads[name], rtol=0, atol=1e-6 * largest), name

    def test_the_stock_shifted_loss_needs_no_change(self, arch):
        """``model(..., labels=...)``, the loss a Trainer uses, on the packed batch."""
        from soup_cli.utils.prefix_tree import collate_prefix_tree

        model = _tiny_model(arch)
        examples = _model_examples()
        plain = _plain_batch(examples)
        packed = collate_prefix_tree(pack_prefix_tree(examples, max_tokens=1000), pad_token_id=0)
        want = model(**plain).loss
        got = model(**packed).loss
        assert abs(float(got) - float(want)) < 1e-5 * abs(float(want))

    def test_several_rows_and_a_budget(self, arch):
        import torch

        from soup_cli.utils.prefix_tree import collate_prefix_tree

        model = _tiny_model(arch).eval()
        examples = _model_examples(seed=4) + _model_examples(seed=5)
        rows = pack_prefix_tree(examples, max_tokens=60)
        assert len(rows) > 1
        packed = collate_prefix_tree(rows, pad_token_id=0)
        plain = _plain_batch(examples)
        with torch.no_grad():
            want = model(
                input_ids=plain["input_ids"], attention_mask=plain["attention_mask"]
            ).logits
            got = model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            ).logits
        for r, row in enumerate(rows):
            for member, path in zip(row.members, row.paths):
                _assert_same_logits(arch, got[r, list(path)], want[member, : len(path)], member)

    def test_the_model_check_passes(self, arch):
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        model = _tiny_model(arch)
        verdict = verify_prefix_tree_model(model, pad_token_id=0)
        assert verdict.exact and verdict.reason == "" and verdict.mask == "bool"
        assert verdict.error <= verdict.bound < 1e-6
        assert verdict.mask_control > 100 * verdict.bound, "the mask control says nothing"
        assert verdict.scale > 0
        assert model.training, "the check left the model in eval mode"

    def test_the_model_check_on_real_examples(self, arch):
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        model = _tiny_model(arch)
        verdict = verify_prefix_tree_model(model, pad_token_id=0, examples=_model_examples()[:3])
        assert verdict.exact, verdict
        unrelated = [_model_examples(seed=7)[4], _model_examples(seed=8)[4]]
        forest = verify_prefix_tree_model(model, pad_token_id=0, examples=unrelated)
        assert forest.exact, forest


class TestTheModelCheckRefuses:
    def test_a_model_that_ignores_position_ids(self):
        """Positions left at the row index: every branch after the first sits at
        the wrong place, which is what the check exists to catch."""
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        model = _tiny_model("gpt2")
        real = model.forward

        def forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
            return real(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.forward = forward
        verdict = verify_prefix_tree_model(model, pad_token_id=0)
        assert not verdict.exact
        assert verdict.error > 0.05 * verdict.scale
        assert "more than" in verdict.reason

    def test_a_model_that_ignores_the_4d_mask(self):
        """Every token sees every earlier token of the row: branches mix."""
        import torch

        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        model = _tiny_model("llama")
        real = model.forward

        def forward(input_ids=None, attention_mask=None, **kwargs):
            if attention_mask is not None and attention_mask.dim() == 4:
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            return real(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.forward = forward
        verdict = verify_prefix_tree_model(model, pad_token_id=0)
        assert not verdict.exact and verdict.reason

    def test_one_example_is_not_enough(self):
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        with pytest.raises(ValueError, match="at least two examples"):
            verify_prefix_tree_model(
                _tiny_model("llama"), pad_token_id=0, examples=[_example([1, 2], [3])]
            )

    def test_the_control_differs_on_a_position_sensitive_model(self):
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        verdict = verify_prefix_tree_model(_tiny_model("gpt2"), pad_token_id=0)
        assert verdict.exact
        assert verdict.control is not None and verdict.control > 100 * verdict.bound
        assert verdict.mask_control is not None and verdict.mask_control > 100 * verdict.bound

    def test_a_model_without_a_vocabulary_size(self):
        torch = pytest.importorskip("torch")
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model

        with pytest.raises(ValueError, match="vocab_size"):
            verify_prefix_tree_model(torch.nn.Linear(2, 2), pad_token_id=0)


# --------------------------------------------------------------------------
# through a layer-streamed model
# --------------------------------------------------------------------------
@pytest.mark.parametrize("quant", ["none", "nf4"])
class TestThroughLayerStreaming:
    """Streaming is about where the weights live; a prefix-tree row is only a
    different ``input_ids`` / mask / positions. The streamed model must give the
    ordinary batch's loss and LoRA gradients."""

    def _examples(self):
        rng = random.Random(9)

        def tokens(count):
            return [rng.randrange(3, 64) for _ in range(count)]

        system = tokens(10)
        return [
            _example(system + tokens(2), tokens(4)),
            _example(system + tokens(3), tokens(3)),
            _example(system, tokens(5)),
        ]

    def test_loss_and_lora_gradients_match(self, tmp_path, quant):
        import torch

        from soup_cli.utils.prefix_tree import collate_prefix_tree
        from tests.test_stream_recompute import _stream

        model, _runtime = _stream(tmp_path, quant)
        examples = self._examples()

        def step(batch):
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            return loss.detach().clone(), {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }

        want_loss, want = step(_plain_batch(examples))
        rows = pack_prefix_tree(examples, max_tokens=1000)
        assert len(rows) == 1 and len(rows[0]) < rows[0].source_tokens
        got_loss, got = step(collate_prefix_tree(rows, pad_token_id=0))
        assert abs(float(got_loss) - float(want_loss)) < 1e-5 * abs(float(want_loss))
        assert want and want.keys() == got.keys()
        largest = max(float(grad.abs().max()) for grad in want.values())
        assert largest > 0, "vacuous: every LoRA gradient is zero"
        for name in want:
            assert torch.allclose(got[name], want[name], rtol=0, atol=2e-4 * largest), name

    def test_the_model_check_passes(self, tmp_path, quant):
        from soup_cli.utils.prefix_tree import verify_prefix_tree_model
        from tests.test_stream_recompute import _stream

        model, _runtime = _stream(tmp_path, quant)
        verdict = verify_prefix_tree_model(model, pad_token_id=0, vocab_size=64)
        assert verdict.exact, verdict
        assert verdict.relative_error < 1e-4
        assert verdict.mask_control > 10 * verdict.bound
