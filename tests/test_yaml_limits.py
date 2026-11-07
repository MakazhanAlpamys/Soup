"""The limited YAML loader: one size, nesting and expansion rule for every reader."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from soup_cli.utils import yaml_limits
from soup_cli.utils.yaml_limits import (
    MAX_CONFIG_BYTES,
    MAX_YAML_NESTING,
    YamlLimitError,
    check_yaml_expanded_size,
    expanded_node_count,
    load_yaml_file_limited,
    load_yaml_limited,
)

# ---------------------------------------------------------------------------
# Documents named by the limit they sit under or cross
# ---------------------------------------------------------------------------


def _flow_sequences(depth: int) -> str:
    return "[" * depth + "]" * depth


def _flow_mappings(depth: int) -> str:
    return "{a: " * depth + "1" + "}" * depth


def _block_sequences(depth: int) -> str:
    return "- " * depth + "x\n"


def _block_mappings(depth: int) -> str:
    lines = [" " * level + "k:" for level in range(depth)]
    return "\n".join(lines) + " v\n"


def _alternating(depth: int) -> str:
    """Mappings and sequences in turn: every collection counts as a level."""
    opened = "".join("[" if level % 2 else "{a: " for level in range(depth))
    closed = "".join("]" if level % 2 else "}" for level in reversed(range(depth)))
    return opened + ("" if depth % 2 == 0 else "1") + closed


_NESTED_SPELLINGS = [
    _flow_sequences,
    _flow_mappings,
    _block_sequences,
    _block_mappings,
    _alternating,
]


def _alias_chain(lists: int) -> str:
    """``lists`` single-item lists, each holding the one before it by alias.

    The text never nests more than two levels, but the loaded object nests
    ``lists + 1``: the root mapping plus every list in the chain.
    """
    doc = "l0: &a0 [x]\n"
    for index in range(1, lists):
        doc += f"l{index}: &a{index} [*a{index - 1}]\n"
    return doc


def _alias_fan_out(levels: int) -> str:
    """Ten references per level: the node count grows tenfold with each one."""
    doc = "l0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for index in range(1, levels + 1):
        doc += f"l{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]\n"
    return doc


# ---------------------------------------------------------------------------
# Valid documents load exactly as yaml.safe_load loads them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "a: 1\n",
        "- a\n- b\n",
        "plain scalar",
        "a: 2e-5\nb: 1e3\nc: 0x10\nd: 1_000\ne: yes\nf: ~\ng: 2026-01-01\n",
        "base: &b {r: 8, alpha: 16}\none:\n  <<: *b\n  r: 4\ntwo: *b\n",
        "text: |\n  line one\n  line two\n\nfolded: >-\n  a\n  b\n",
        'q: "a\\tb"\ns: \'it\'\'s\'\n',
        "﻿a: 1\n",
        "a: [1, [2, [3, {b: [4]}]]]\n",
        "name: café ☃\n",
    ],
)
def test_valid_document_loads_as_safe_load_does(text):
    assert load_yaml_limited(text, "doc", max_bytes=4096) == yaml.safe_load(text)


def test_bytes_are_accepted_like_text():
    assert load_yaml_limited(b"a: [1, 2]\n", "doc", max_bytes=64) == {"a": [1, 2]}


def test_anything_but_text_or_bytes_is_a_type_error():
    with pytest.raises(TypeError, match="doc must be str or bytes, got list"):
        load_yaml_limited(["a: 1"], "doc", max_bytes=64)


def test_a_syntax_error_is_still_a_yaml_error():
    with pytest.raises(yaml.YAMLError):
        load_yaml_limited("a: [1, 2\nb: }", "doc", max_bytes=64)


def test_every_refusal_is_a_value_error():
    assert issubclass(YamlLimitError, ValueError)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_text_exactly_at_the_size_limit_loads():
    text = "a: " + "x" * 61
    assert len(text.encode()) == 64
    assert load_yaml_limited(text, "doc", max_bytes=64) == {"a": "x" * 61}


def test_text_one_byte_over_the_size_limit_is_refused():
    text = "a: " + "x" * 62
    assert len(text.encode()) == 65
    with pytest.raises(YamlLimitError, match=r"^doc is larger than 64 bytes; refusing$"):
        load_yaml_limited(text, "doc", max_bytes=64)


def test_size_is_counted_in_bytes_not_characters():
    text = "a: " + "é" * 31  # 34 characters, 65 bytes
    assert len(text) == 34
    assert len(text.encode()) == 65
    with pytest.raises(YamlLimitError, match="doc is larger than 64 bytes"):
        load_yaml_limited(text, "doc", max_bytes=64)
    assert load_yaml_limited(text, "doc", max_bytes=65) == {"a": "é" * 31}


def test_size_is_checked_before_anything_is_parsed(monkeypatch):
    def _no_parse(*args, **kwargs):
        raise AssertionError("an over-size document must not reach the parser")

    monkeypatch.setattr(yaml_limits, "_parse", _no_parse)
    with pytest.raises(YamlLimitError, match="is larger than 8 bytes"):
        load_yaml_limited("a: [1, 2, 3, 4]", "doc", max_bytes=8)


def test_the_config_size_limit_is_one_mebibyte():
    assert MAX_CONFIG_BYTES == 1024 * 1024


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_the_nesting_limit_is_one_hundred_levels():
    assert MAX_YAML_NESTING == 100


@pytest.mark.parametrize("spelling", _NESTED_SPELLINGS)
def test_document_nested_exactly_to_the_limit_loads(spelling):
    text = spelling(MAX_YAML_NESTING)
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


@pytest.mark.parametrize("spelling", _NESTED_SPELLINGS)
def test_document_nested_one_level_past_the_limit_is_refused(spelling):
    text = spelling(MAX_YAML_NESTING + 1)
    with pytest.raises(
        YamlLimitError,
        match=r"^doc nests collections more than 100 levels deep; refusing$",
    ):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_sibling_collections_do_not_add_up_to_a_depth():
    text = "[" + ", ".join(["[[]]"] * 500) + "]"
    assert load_yaml_limited(text, "doc", max_bytes=8192) == [[[]]] * 500


@pytest.mark.parametrize("spelling", [_flow_sequences, _block_sequences, _flow_mappings])
def test_document_nested_far_past_the_limit_gets_the_same_refusal(spelling):
    """Two hundred times the limit is refused exactly as one level past it."""
    text = spelling(20_000)
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        load_yaml_limited(text, "doc", max_bytes=1_000_000)


def test_parsing_stops_at_the_limit_rather_than_after_the_document(monkeypatch):
    """The refusal comes from the event stream, before any object is built."""

    def _no_check(*args, **kwargs):
        raise AssertionError("an over-nested document must not be built first")

    monkeypatch.setattr(yaml_limits, "check_yaml_expanded_size", _no_check)
    with pytest.raises(YamlLimitError, match="nests collections more than 100 levels"):
        load_yaml_limited(_flow_sequences(MAX_YAML_NESTING + 1), "doc", max_bytes=4096)


def test_nesting_reached_through_aliases_at_the_limit_loads():
    text = _alias_chain(MAX_YAML_NESTING - 1)  # root mapping + 99 lists
    data = load_yaml_limited(text, "doc", max_bytes=10**6)
    assert data == yaml.safe_load(text)


def test_nesting_reached_through_aliases_past_the_limit_is_refused():
    text = _alias_chain(MAX_YAML_NESTING)  # root mapping + 100 lists
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_document_past_the_interpreter_limit_is_a_refusal_too(monkeypatch):
    """With the nesting limit lifted, the interpreter's own limit still ends as a refusal."""
    monkeypatch.setattr(yaml_limits, "MAX_YAML_NESTING", 10**9)
    with pytest.raises(
        YamlLimitError,
        match=r"^doc is nested or chained too deeply to load; refusing$",
    ):
        load_yaml_limited(_flow_sequences(5000), "doc", max_bytes=10**6)


def test_document_rule_refuses_an_object_nested_past_the_limit():
    data: list = []
    for _ in range(MAX_YAML_NESTING):
        data = [data]
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        check_yaml_expanded_size(data, "doc", source_bytes=10**6)


def test_document_rule_accepts_an_object_nested_to_the_limit():
    data: list = []
    for _ in range(MAX_YAML_NESTING - 1):
        data = [data]
    check_yaml_expanded_size(data, "doc", source_bytes=10**6)


# ---------------------------------------------------------------------------
# The counter sees depth that only exists through shared references
# ---------------------------------------------------------------------------


def test_counter_treats_a_shared_chain_past_its_depth_as_over_the_limit():
    data = yaml.safe_load(_alias_chain(1500))
    assert expanded_node_count(data, limit=10**9) == 10**9 + 1


def test_counter_counts_a_shared_chain_within_its_depth_exactly():
    data = yaml.safe_load(_alias_chain(500))
    # Root: 1 + 500 keys. List k holds list k-1, so it expands to k + 2 nodes
    # (itself, the k lists inside it, and the one scalar at the bottom).
    assert expanded_node_count(data, limit=10**9) == 1 + 500 + sum(
        index + 2 for index in range(500)
    )


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def test_modest_alias_reuse_loads():
    text = _alias_fan_out(2)
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


def test_alias_fan_out_past_the_node_allowance_is_refused():
    text = _alias_fan_out(9)
    allowance = len(text.encode()) + yaml_limits.ALIAS_EXPANSION_SLACK
    with pytest.raises(
        YamlLimitError,
        match=(
            rf"^doc expands to more than {allowance} nodes from "
            rf"{len(text.encode())} bytes of YAML; refusing$"
        ),
    ):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_self_referencing_alias_is_refused():
    with pytest.raises(YamlLimitError, match="doc expands to more than"):
        load_yaml_limited("a: &x [1, *x]\n", "doc", max_bytes=64)


# ---------------------------------------------------------------------------
# Merge keys: the pairs they copy are charged to the node allowance as made
# ---------------------------------------------------------------------------


def _merge_fan(copies: int, keys: int = 10) -> str:
    """One mapping of ``keys`` pairs merged ``copies`` times into another."""
    base = ", ".join(f"k{index}: {index}" for index in range(keys))
    return f"b: &b {{{base}}}\nm:\n  <<: [" + ", ".join(["*b"] * copies) + "]\n"


def _merge_ladder(steps: int) -> str:
    """Each mapping merges the one before it and adds a key of its own."""
    text = "m0: &m0 {k0: 0}\n"
    for index in range(1, steps + 1):
        text += f"m{index}: &m{index} {{<<: *m{index - 1}, k{index}: {index}}}\n"
    return text


def _merge_chain(links: int) -> str:
    """The root merges the last of ``links`` mappings that each merge the one before."""
    text = "c0: &c0 {k: v}\n"
    for index in range(1, links):
        text += f"c{index}: &c{index} {{<<: *c{index - 1}}}\n"
    return text + f"<<: *c{links - 1}\n"


def _allowance(text: str) -> int:
    return len(text.encode()) + yaml_limits.ALIAS_EXPANSION_SLACK


@pytest.mark.parametrize(
    "text",
    [
        "a: &a {x: 1, y: 2}\nb:\n  <<: *a\n  y: 3\n",
        "a: &a {x: 1}\nb: &b {x: 2, y: 2}\nc:\n  <<: [*a, *b]\n  z: 3\n",
        "a: &a {x: 1}\nb: &b {<<: *a, y: 2}\nc: {<<: *b, z: 3}\n",
        "c:\n  <<: {x: 1}\n  <<: {x: 2, y: 2}\n",
        "c: {<<: {<<: {<<: {x: 1}, y: 2}, z: 3}}\n",
        "a: &a {}\nb: {<<: *a}\nc: {<<: []}\n",
        "c: {=: plain}\n",
        "a: &a {<<: *a, x: 1}\n",
        "a: &a {<<: [*a], x: 1}\n",
        "a: &a {x: 1, <<: *a, y: 2}\n",
        "a: &a {<<: *a, <<: *a, x: 1}\n",
        "a: &a {<<: *a}\n",
        "a: &a {x: 1}\nb: &b {<<: [*a, *b], y: 2}\n",
        "a: &a {x: 1}\nb: &b {y: 2, <<: [*b, *a], <<: *b}\n",
        "? ab\n: 1\n",
    ],
)
def test_merge_keys_load_as_safe_load_loads_them(text):
    assert load_yaml_limited(text, "doc", max_bytes=4096) == yaml.safe_load(text)


@pytest.mark.parametrize(
    "text",
    ["a: {<<: 1}\n", "a: &a {x: 1}\nb: {<<: [*a, 2, 3]}\n", "b: {<<: [[1]]}\n"],
)
def test_a_merge_of_something_else_is_the_same_yaml_error(text):
    with pytest.raises(yaml.YAMLError) as expected:
        yaml.safe_load(text)
    with pytest.raises(yaml.YAMLError) as raised:
        load_yaml_limited(text, "doc", max_bytes=4096)
    assert type(raised.value) is type(expected.value)
    assert str(raised.value) == str(expected.value)


def _copies_at_the_allowance(keys: int) -> int:
    """The most copies whose merged pairs (two nodes each) fit the allowance."""
    copies = 1
    while 2 * (copies + 1) * keys <= _allowance(_merge_fan(copies + 1, keys)):
        copies += 1
    return copies


def _merge_fan_of_size(copies: int, keys: int, size: int) -> str:
    """``_merge_fan`` padded with a trailing comment to exactly ``size`` bytes."""
    text = _merge_fan(copies, keys)
    padding = size - len(text.encode())
    assert padding >= 2, padding
    text += "#" + "x" * (padding - 2) + "\n"
    assert len(text.encode()) == size
    return text


#: 700 copies of a 10-pair mapping: 14,000 merged nodes (two per pair).
_COPIES, _KEYS = 700, 10
_MERGED = 2 * _COPIES * _KEYS


def test_merged_nodes_equal_to_the_allowance_load():
    size = _MERGED - yaml_limits.ALIAS_EXPANSION_SLACK
    text = _merge_fan_of_size(_COPIES, _KEYS, size)
    assert _allowance(text) == _MERGED
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


def test_merged_nodes_one_past_the_allowance_are_refused():
    size = _MERGED - yaml_limits.ALIAS_EXPANSION_SLACK - 1
    text = _merge_fan_of_size(_COPIES, _KEYS, size)
    assert _allowance(text) == _MERGED - 1
    with pytest.raises(YamlLimitError) as raised:
        load_yaml_limited(text, "doc", max_bytes=10**6)
    assert str(raised.value) == (
        f"doc expands to more than {_MERGED - 1} nodes from {size} bytes of YAML; refusing"
    )


def test_merged_pairs_exactly_at_the_node_allowance_load():
    copies = _copies_at_the_allowance(10)
    text = _merge_fan(copies)
    assert _allowance(text) - 2 * copies * 10 < 20  # within one merged mapping of it
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


def test_merged_pairs_one_mapping_past_the_node_allowance_are_refused():
    text = _merge_fan(_copies_at_the_allowance(10) + 1)
    with pytest.raises(YamlLimitError) as raised:
        load_yaml_limited(text, "doc", max_bytes=10**6)
    assert str(raised.value) == (
        f"doc expands to more than {_allowance(text)} nodes from "
        f"{len(text.encode())} bytes of YAML; refusing"
    )


@pytest.mark.parametrize(
    "text",
    [_merge_fan(5000, keys=50), _merge_ladder(1500)],
    ids=["fan", "ladder"],
)
def test_merging_stops_at_the_first_pair_past_the_allowance(text):
    """Counted on the loader itself: nothing is copied beyond one more mapping."""
    budget = _allowance(text)
    loader = yaml_limits._LimitedLoader(text, node_budget=budget)
    try:
        with pytest.raises(yaml_limits._MergeBudgetError):
            loader.get_single_data()
        assert budget < loader.merged_nodes <= budget + 2 * 1500
    finally:
        loader.dispose()


def test_merge_ladder_past_the_allowance_is_refused_by_name():
    with pytest.raises(YamlLimitError, match="^doc expands to more than"):
        load_yaml_limited(_merge_ladder(1500), "doc", max_bytes=10**6)


def test_merge_chain_at_the_limit_loads():
    text = _merge_chain(MAX_YAML_NESTING - 1)
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


@pytest.mark.parametrize("links", [MAX_YAML_NESTING, 20_000])
def test_merge_chain_past_the_limit_is_refused(links):
    with pytest.raises(
        YamlLimitError,
        match=r"^doc chains merge keys more than 100 levels deep; refusing$",
    ):
        load_yaml_limited(_merge_chain(links), "doc", max_bytes=10**6)


def test_a_mapping_that_merges_itself_is_charged_like_any_merge():
    """It loads as safe_load loads it, and its own pairs count towards the allowance."""
    text = "a: &a {<<: *a, x: 1, y: 2}\n"
    loader = yaml_limits._LimitedLoader(text, node_budget=10)
    try:
        assert loader.get_single_data() == yaml.safe_load(text) == {"a": {"x": 1, "y": 2}}
        assert loader.merged_nodes == 4
    finally:
        loader.dispose()


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("d: 2026-02-30\n", "day is out of range for month"),
        ("when:\n  - 2026-13-01\n", "month must be in 1..12"),
    ],
)
def test_a_scalar_that_cannot_be_built_is_a_yaml_error_and_a_value_error(text, reason):
    with pytest.raises(ValueError, match=reason):
        yaml.safe_load(text)
    with pytest.raises(yaml_limits.YamlScalarError) as raised:
        load_yaml_limited(text, "doc", max_bytes=64)
    assert isinstance(raised.value, yaml.YAMLError)
    assert isinstance(raised.value, ValueError)
    assert raised.value.problem == f"cannot build this value: {reason}"
    assert raised.value.problem_mark.line == text.count("\n") - 1


_MEASURE = """
import json, sys, time, tracemalloc
from soup_cli.utils.yaml_limits import YamlLimitError, load_yaml_limited
base = ", ".join(f"k{i}: {i}" for i in range(2000))
text = "b: &b {" + base + "}\\nm:\\n  <<: [" + ", ".join(["*b"] * 4000) + "]\\n"
tracemalloc.start()
started = time.perf_counter()
try:
    load_yaml_limited(text, "doc", max_bytes=10**6)
    outcome = "loaded"
except YamlLimitError as exc:
    outcome = str(exc)
print(json.dumps({"bytes": len(text), "outcome": outcome,
                  "seconds": time.perf_counter() - started,
                  "peak": tracemalloc.get_traced_memory()[1]}))
"""


def test_refusal_of_a_large_merge_stays_small_in_memory():
    """Eight million pairs asked for by 40 KB of text; the refusal keeps to a few MB."""
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _MEASURE],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["bytes"] < 64 * 1024
    assert report["outcome"].startswith("doc expands to more than"), report
    assert report["peak"] < 32 * 1024 * 1024, report


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_file_exactly_at_the_size_limit_loads(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: " + b"x" * 61)
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=64) == {"a": "x" * 61}


def test_file_one_byte_over_the_size_limit_is_refused(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: " + b"x" * 62)
    with pytest.raises(YamlLimitError, match=r"^f.yaml is larger than 64 bytes; refusing$"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)


def test_file_read_never_asks_for_more_than_one_byte_past_the_limit(tmp_path, monkeypatch):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"x" * 10_000)
    requested: list = []
    real_open = open

    class _Recording:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._handle.close()

        def read(self, size=-1):
            requested.append(size)
            return self._handle.read(size)

    def _open(file, mode="r", *args, **kwargs):
        assert mode == "rb"
        return _Recording(real_open(file, mode, *args, **kwargs))

    monkeypatch.setattr(yaml_limits, "open", _open, raising=False)
    with pytest.raises(YamlLimitError, match="is larger than 64 bytes"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)
    assert requested == [65]


def test_file_size_limit_counts_the_bytes_on_disk(tmp_path):
    """CRLF line ends count as two bytes even though they load as one."""
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: 1\r\n" * 11)  # 66 bytes, 55 once newlines are translated
    with pytest.raises(YamlLimitError, match="is larger than 64 bytes"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)


@pytest.mark.parametrize(
    "content",
    [
        b"a: 1\nb: [x, y]\n",
        b"a: 1\r\nb: [x, y]\r\n",
        b"\xef\xbb\xbfa: 1\nb: [x, y]\n",
        b"\xef\xbb\xbfa: 1\r\nb: [x, y]\r\n",
        b"a: 1\rb: [x, y]\r",
    ],
    ids=["lf", "crlf", "bom", "bom-crlf", "cr"],
)
def test_file_line_ends_and_bom_load_the_same_document(tmp_path, content):
    path = tmp_path / "f.yaml"
    path.write_bytes(content)
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=4096) == {"a": 1, "b": ["x", "y"]}


@pytest.mark.parametrize(
    "content",
    [
        b"t: |\r\n  one\r\n  two\r\n\r\nf: >\r\n  a\r\n  b\r\n",
        b'q: "one\r\n  two"\r\n',
        b"p: one\r\n  two\r\n",
        "name: café ☃\r\n".encode(),
    ],
    ids=["block-scalars", "double-quoted", "plain-multiline", "non-ascii"],
)
def test_file_is_decoded_exactly_as_read_text_decodes_it(tmp_path, content):
    path = tmp_path / "f.yaml"
    path.write_bytes(content)
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=4096) == expected


def test_file_that_is_not_utf8_is_still_a_decode_error(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: \xff\xfe\n")
    with pytest.raises(UnicodeDecodeError):
        load_yaml_file_limited(path, "f.yaml", max_bytes=4096)


def test_missing_file_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_file_limited(tmp_path / "absent.yaml", "absent.yaml", max_bytes=4096)


def test_file_over_the_nesting_limit_is_refused_by_name(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(_block_mappings(MAX_YAML_NESTING + 1), encoding="utf-8")
    with pytest.raises(YamlLimitError, match="f.yaml nests collections more than 100 levels"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=10**6)


def test_file_over_the_node_allowance_is_refused_by_name(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(_alias_fan_out(9), encoding="utf-8")
    with pytest.raises(YamlLimitError, match="f.yaml expands to more than"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=10**6)


@pytest.mark.parametrize(
    ("text", "name", "raised_by_pyyaml"),
    [
        ("a: !!bool foo\n", "bool", KeyError),
        ("a: !!bool ''\n", "bool", KeyError),
        ("a: !!timestamp foo\n", "timestamp", AttributeError),
        ('a: !!int ""\n', "int", IndexError),
        ('a: !!float ""\n', "float", IndexError),
    ],
)
def test_text_an_explicit_tag_cannot_convert_is_the_same_scalar_error(
    text, name, raised_by_pyyaml
):
    with pytest.raises(raised_by_pyyaml):
        yaml.safe_load(text)
    with pytest.raises(yaml_limits.YamlScalarError) as raised:
        load_yaml_limited(text, "doc", max_bytes=64)
    assert raised.value.problem == f"cannot build this value: not a valid {name}"
    assert (raised.value.problem_mark.line, raised.value.problem_mark.column) == (0, 3)


@pytest.mark.parametrize(
    "text",
    ["a: !!bool yes\n", "a: !!int '12'\n", "a: !!float '1.5'\n", "a: !!timestamp 2026-01-02\n"],
)
def test_text_an_explicit_tag_can_convert_loads_as_before(text):
    assert load_yaml_limited(text, "doc", max_bytes=64) == yaml.safe_load(text)
