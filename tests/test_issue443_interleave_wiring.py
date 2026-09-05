"""`data.interleave` is schema-validated but never consumed at training time (issue #443).

`parse_interleave`/`InterleaveSpec` have been fully implemented and unit-tested since
v0.42.0, and the schema has validated `data.interleave`'s shape since the same release
— but `load_dataset()` never called `parse_interleave`, so every multi-dataset mixture
request silently trained on nothing but `data.train`'s single path (the same gap #330
and #442 papered over in their respective renderers, deliberately deferring the real
fix to this issue).

The maintainer's decision on #443 (Option A, scope fixed by him): `DataConfig.train`
now accepts `str | list[str]`; a list requires `data.interleave`, applies to local file
paths only, and is combined into one row set BEFORE the existing `val_split` line in
`_finalize` runs (so a single path stays byte-identical). `packing` / `multipack` +
`interleave` are rejected at parse time with a message naming the reason.
`data.streaming` / an HF-hub dataset name + `interleave` were ALSO rejected here in
#443's v1 — that refusal was lifted in #459 for a data.streaming=true local/remote-URI
list and for an all-HF-hub-dataset-name list respectively (each a new, separately
implemented path — see tests/test_issue459_interleave_streaming_hub.py); a list mixing
hub names with local/remote entries, and a streaming hub-name list, still refuse.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import DataConfig, SoupConfig
from soup_cli.data.loader import load_dataset


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text(
        "\n".join(f'{{"text": {t!r}}}' for t in texts).replace("'", '"') + "\n",
        encoding="utf-8",
    )


def _cfg(tmp_path: Path, **data_overrides) -> SoupConfig:
    data = {
        "train": str(tmp_path / "a.jsonl"),
        "format": "plaintext",
        "val_split": 0.0,
    }
    data.update(data_overrides)
    return SoupConfig.model_validate(
        {
            "base": "test-base",
            "task": "sft",
            "data": data,
            "training": {"epochs": 1},
            "output": str(tmp_path / "out"),
        }
    )


# ---------------------------------------------------------------------------
# Loader-level: the actual acceptance criteria
# ---------------------------------------------------------------------------


def test_single_path_train_output_is_byte_identical_to_baseline(tmp_path):
    # Golden test pinning the single-path branch — must not change AT ALL
    # as a side effect of wiring interleave in. Any refactor that touches
    # this branch's output shape fails here.
    _write_jsonl(tmp_path / "a.jsonl", ["A-0", "A-1", "A-2"])
    cfg = _cfg(tmp_path)
    result = load_dataset(cfg.data)
    assert result == {
        "train": [
            {"text": "A-0"},
            {"text": "A-1"},
            {"text": "A-2"},
        ]
    }


def test_data_interleave_is_actually_consumed_not_just_parsed(tmp_path):
    # The acceptance criterion in the maintainer's own words: a test that
    # fails if data.interleave is ignored — asserting on the rows that
    # reach the trainer, not on the parsed config. Before #443,
    # load_dataset() never looked at data.interleave at all, so this would
    # have silently returned only dataset A's rows (or crashed on a list
    # train_path) — either way, this test fails on the pre-#443 code.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="under",
    )
    result = load_dataset(cfg.data)
    texts = {row["text"] for row in result["train"]}
    assert any(t.startswith("A-") for t in texts)
    assert any(t.startswith("B-") for t in texts)


def test_interleave_concat(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(4)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 7
    assert sum(t.startswith("A-") for t in texts) == 4
    assert sum(t.startswith("B-") for t in texts) == 3


def test_interleave_under_truncates_to_smallest(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="under",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 6
    assert sum(t.startswith("A-") for t in texts) == 3
    assert sum(t.startswith("B-") for t in texts) == 3


def test_interleave_over_upsamples_to_largest(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(10)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(3)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="over",
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 20
    assert sum(t.startswith("A-") for t in texts) == 10
    b_texts = [t for t in texts if t.startswith("B-")]
    assert len(b_texts) == 10
    # B only has 3 unique rows — over-sampling must repeat, not invent rows.
    counts = Counter(b_texts)
    assert set(counts) == {"B-0", "B-1", "B-2"}
    assert sum(counts.values()) == 10


def test_interleave_probs_matches_requested_ratio(tmp_path):
    # 30 + 10 = 40 total, probs chosen to divide evenly so the expected
    # counts are exact (no apportionment-rounding ambiguity).
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(30)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(10)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": [0.75, 0.25]},
    )
    result = load_dataset(cfg.data)
    texts = [row["text"] for row in result["train"]]
    assert len(texts) == 40
    assert sum(t.startswith("A-") for t in texts) == 30
    assert sum(t.startswith("B-") for t in texts) == 10


def test_val_split_applied_once_after_mixing(tmp_path):
    # Proves "one slice, the same line as today in _finalize": the split
    # fraction must be computed off the COMBINED length, not per-source.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(8)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(2)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
        val_split=0.2,
    )
    result = load_dataset(cfg.data)
    combined_len = len(result["train"]) + len(result["val"])
    assert combined_len == 10
    assert len(result["val"]) == int(10 * 0.2)


def test_interleave_over_val_split_has_no_duplicate_row_across_train_and_val(tmp_path):
    # #680: "over" cycles B's 2 rows up to A's 8, so a naive split-after-mix
    # can put one copy of a cycled row in train and another in val.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(8)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(2)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="over",
        val_split=0.1,
    )
    result = load_dataset(cfg.data)
    train_texts = {row["text"] for row in result["train"]}
    val_texts = {row["text"] for row in result["val"]}
    assert not (train_texts & val_texts)
    assert len(result["train"]) + len(result["val"]) == 16


def test_interleave_probs_val_split_has_no_duplicate_row_across_train_and_val(tmp_path):
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(8)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(2)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave={"strategy": "probs", "probs": [0.5, 0.5]},
        val_split=0.1,
    )
    result = load_dataset(cfg.data)
    train_texts = {row["text"] for row in result["train"]}
    val_texts = {row["text"] for row in result["val"]}
    assert not (train_texts & val_texts)


def test_interleave_probs_val_split_no_overlap_regardless_of_source_order(tmp_path):
    # Same two sources, B listed first and given the larger probs share:
    # the no-overlap guarantee must not depend on which source happens to
    # be the one that gets padded, or on where its copies land once combined.
    _write_jsonl(tmp_path / "a.jsonl", [f"A-{i}" for i in range(8)])
    _write_jsonl(tmp_path / "b.jsonl", [f"B-{i}" for i in range(2)])
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "b.jsonl"), str(tmp_path / "a.jsonl")],
        interleave={"strategy": "probs", "probs": [0.7, 0.3]},
        val_split=0.5,
    )
    result = load_dataset(cfg.data)
    train_texts = {row["text"] for row in result["train"]}
    val_texts = {row["text"] for row in result["val"]}
    assert not (train_texts & val_texts)


# ---------------------------------------------------------------------------
# Schema-level: back-compat pin + the four parse-time refusals
# ---------------------------------------------------------------------------


def test_interleave_train_single_string_still_accepted_by_schema():
    # Schema-level companion to the loader's byte-identical pin: a bare
    # string data.train must keep validating exactly as before #443.
    cfg = DataConfig(train="d.jsonl", format="auto")
    assert cfg.train == "d.jsonl"
    assert cfg.interleave is None


def test_train_list_of_one_entry_rejected():
    with pytest.raises(ValidationError, match=">= 2 entries"):
        DataConfig(train=["only-one.jsonl"], interleave="concat")


def test_train_list_without_interleave_rejected(tmp_path):
    with pytest.raises(ValidationError, match="requires data.interleave"):
        _cfg(tmp_path, train=["a.jsonl", "b.jsonl"])


def test_interleave_with_packing_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="fixed blocks"):
        SoupConfig.model_validate(
            {
                "base": "test-base",
                "task": "sft",
                "data": {
                    "train": [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
                    "interleave": "concat",
                    "format": "plaintext",
                },
                "training": {"epochs": 1, "packing": True},
                "output": str(tmp_path / "out"),
            }
        )


def test_interleave_with_multipack_rejected_with_reason(tmp_path):
    with pytest.raises(ValidationError, match="mixture ratio"):
        SoupConfig.model_validate(
            {
                "base": "test-base",
                "task": "sft",
                "data": {
                    "train": [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
                    "interleave": "concat",
                    "format": "plaintext",
                },
                "training": {"epochs": 1, "multipack": True},
                "output": str(tmp_path / "out"),
            }
        )


def test_interleave_with_streaming_local_files_now_accepted(tmp_path):
    # #459 lifted this refusal for local-file (and remote-URI) lists —
    # data.streaming=true + data.interleave now validates. The full
    # streaming-interleave loader behaviour is covered by
    # tests/test_issue459_interleave_streaming_hub.py; this just pins that
    # schema no longer refuses the combination it used to.
    cfg = _cfg(
        tmp_path,
        train=[str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")],
        interleave="concat",
        streaming=True,
    )
    assert cfg.data.streaming is True
    assert cfg.data.interleave == "concat"


def test_interleave_with_streaming_hub_list_still_rejected_with_reason(tmp_path):
    # #459 implements streaming for local/remote lists and eager loading
    # for all-hub-name lists, but NOT the combination of the two — see
    # test_issue459_interleave_streaming_hub.py for the full matrix.
    with pytest.raises(ValidationError, match="does not support data.streaming=true"):
        _cfg(
            tmp_path,
            train=["org/dataset-a", "org/dataset-b"],
            interleave="concat",
            streaming=True,
        )


def test_interleave_with_mixed_local_and_hub_entries_rejected_with_reason(tmp_path):
    # A list mixing a local file path with an HF-hub dataset name is still
    # rejected — #459 added dedicated paths for an all-local/remote list
    # and a separate all-hub list, not a mix of the two (see
    # test_issue459_interleave_streaming_hub.py for the full matrix).
    with pytest.raises(ValidationError, match="not a mix"):
        _cfg(
            tmp_path,
            train=[str(tmp_path / "a.jsonl"), "org/some-dataset"],
            interleave="concat",
        )


def test_interleave_rendered_overlay_yaml_round_trips_through_loader(tmp_path):
    # Belt-and-braces: a config assembled purely from YAML text (not
    # Python dict kwargs) round-trips through the schema and the loader.
    _write_jsonl(tmp_path / "a.jsonl", ["A-0", "A-1"])
    _write_jsonl(tmp_path / "b.jsonl", ["B-0", "B-1"])
    yaml_text = (
        "base: test-base\n"
        "task: sft\n"
        "data:\n"
        f"  train:\n    - {tmp_path / 'a.jsonl'}\n    - {tmp_path / 'b.jsonl'}\n"
        "  interleave: concat\n"
        "  format: plaintext\n"
        "  val_split: 0.0\n"
        "training:\n"
        "  epochs: 1\n"
        f"output: {tmp_path / 'out'}\n"
    )
    cfg = load_config_from_string(yaml_text)
    result = load_dataset(cfg.data)
    assert len(result["train"]) == 4


# ---------------------------------------------------------------------------
# Enumerating test — every known consumer of cfg.data.train, one row each.
#
# Maintainer's review of PR #460: widening data.train to a list made three
# OTHER consumers (mcp_server/registry.py, utils/terraform_plan.py,
# utils/annex_xi.py) silently go blind — each filtered on
# isinstance(path, str), so a list contributed nothing / coerced to the
# same sentinel as "no dataset" / stringified into a bogus path. He asked
# for one durable, enumerating test covering every known consumer (the 4
# already fixed earlier in this PR + these 3), not a bespoke test per
# site — so a future Nth consumer gets added to THIS table, not a new
# test file.
# ---------------------------------------------------------------------------


def _write_url_jsonl(path: Path, count: int, domain: str) -> None:
    lines = [f'{{"text": "row {i} https://{domain}/p{i}"}}' for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matched_fixture(tmp_path: Path):
    """Two data shapes carrying the SAME effective dataset.

    ``single``: one file, 8 example.com rows + 4 other.org rows (12 total).
    ``as_list``: the identical 12 rows split into two files (8 + 4),
    combined via interleave 'concat' — same total rows, same domain set.
    """
    single_path = tmp_path / "single.jsonl"
    lines = [f'{{"text": "row {i} https://example.com/p{i}"}}' for i in range(8)]
    lines += [f'{{"text": "row {i} https://other.org/p{i}"}}' for i in range(4)]
    single_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    _write_url_jsonl(a_path, 8, "example.com")
    _write_url_jsonl(b_path, 4, "other.org")

    single_cfg = _cfg(tmp_path, train=str(single_path))
    list_cfg = _cfg(
        tmp_path, train=[str(a_path), str(b_path)], interleave="concat"
    )
    return single_cfg, list_cfg, single_path, a_path, b_path


def test_every_data_train_consumer_handles_list_non_degenerately(tmp_path, monkeypatch):
    """Enumerates ALL known consumers of cfg.data.train / data.train-derived
    paths. Each row asserts the list-shaped result is non-degenerate (not
    silently empty/zero/blind) compared to the equivalent single-path
    result. A future consumer is added as a new row here, not a new test.
    """
    # registry._collect_external_protected_inputs and terraform_plan's
    # compute_dataset_sha both gate on is_under_cwd(path), checked against
    # the real process cwd — must actually chdir, not just pass tmp_path.
    monkeypatch.chdir(tmp_path)

    from soup_cli.commands.cost import _get_dataset_size
    from soup_cli.commands.data import _cache_key_dataset_path
    from soup_cli.commands.ship import _MAX_DATA_SHA_BYTES, _compute_provenance, _safe_hash_file
    from soup_cli.commands.train import _lr_finder_dataset_path
    from soup_cli.mcp_server.registry import _collect_external_protected_inputs
    from soup_cli.utils.annex_xi import load_top_domains_from_jsonl
    from soup_cli.utils.terraform_plan import build_plan, compute_dataset_sha

    single_cfg, list_cfg, single_path, a_path, b_path = _matched_fixture(tmp_path)

    # 1. cost._get_dataset_size(cfg) -> (row_count, is_estimated)
    single_size, single_est = _get_dataset_size(single_cfg)
    list_size, list_est = _get_dataset_size(list_cfg)
    assert single_size == list_size == 12, "row counts must match — same total data"
    assert single_est is False and list_est is False, (
        "must actually read the files, not fall back to the default estimate"
    )

    # 2. data._cache_key_dataset_path(cfg) -> str fed to make_preprocess_cache_key
    import json as _json

    list_key_input = _cache_key_dataset_path(list_cfg)
    assert list_key_input, "must not collapse to an empty/blind cache key input"
    # JSON-parse rather than substring-search: json.dumps escapes backslashes
    # in Windows paths, so a raw `str(a_path) in list_key_input` check would
    # false-negative there.
    parsed_key_input = _json.loads(list_key_input)
    assert parsed_key_input["train"] == [str(a_path), str(b_path)], (
        "list cache key input must name BOTH files, not just the first"
    )

    # 3. ship._compute_provenance(cfg) -> {"data_sha": <64-hex>, ...}
    single_prov = _compute_provenance(single_cfg)
    list_prov = _compute_provenance(list_cfg)
    assert len(single_prov.get("data_sha", "")) == 64
    assert len(list_prov.get("data_sha", "")) == 64
    # must not silently equal a hash of only the first file — the pre-fix
    # blind-to-a-single-file failure mode.
    assert list_prov["data_sha"] != _safe_hash_file(str(a_path), _MAX_DATA_SHA_BYTES)

    # 4. train._lr_finder_dataset_path(train) -> a real, existing file path
    single_lr_path = _lr_finder_dataset_path(single_cfg.data.train)
    list_lr_path = _lr_finder_dataset_path(list_cfg.data.train)
    assert Path(single_lr_path).is_file() and Path(single_lr_path).stat().st_size > 0
    assert Path(list_lr_path).is_file() and Path(list_lr_path).stat().st_size > 0

    # 5. registry._collect_external_protected_inputs(cfg) -> list[ProtectedFile]
    single_protected = _collect_external_protected_inputs(single_cfg)
    list_protected = _collect_external_protected_inputs(list_cfg)
    single_hits = [p for p in single_protected if p.path == str(single_path.resolve())]
    list_hits = [
        p for p in list_protected
        if p.path in (str(a_path.resolve()), str(b_path.resolve()))
    ]
    assert len(single_hits) == 1, "single path contributes exactly one protected entry"
    assert len(list_hits) == 2, "list must contribute one protected entry PER file, not zero"

    # 6. terraform_plan.build_plan({...}) -> TrainingPlan.dataset_sha
    zero = "0" * 64
    single_plan = build_plan({"base": "b", "task": "sft", "data": {"train": str(single_path)}})
    list_plan = build_plan(
        {"base": "b", "task": "sft", "data": {"train": [str(a_path), str(b_path)]}}
    )
    assert single_plan.dataset_sha != zero
    assert list_plan.dataset_sha != zero, (
        "list must not silently produce the all-zero 'missing dataset' sentinel"
    )
    assert list_plan.dataset_sha != compute_dataset_sha(str(a_path)), (
        "must not silently hash only the first file"
    )

    # 7. annex_xi.load_top_domains_from_jsonl(path) -> top domains, aggregated
    single_domains = {d for d, _ in load_top_domains_from_jsonl(str(single_path))}
    list_domains = {d for d, _ in load_top_domains_from_jsonl([str(a_path), str(b_path)])}
    assert single_domains == list_domains == {"example.com", "other.org"}, (
        "list must aggregate domains across every file, matching the "
        "equivalent single-path corpus — not go blank or report only one file"
    )
