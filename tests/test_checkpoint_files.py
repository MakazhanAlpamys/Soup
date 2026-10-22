"""#1806 — the checkpoint is the files the index names, not every ``*.safetensors``.

Mistral's Hub repositories ship the same weights twice: the Hugging Face shards
with ``model.safetensors.index.json`` and one ``consolidated.safetensors`` with
mistral-inference names, which transformers never opens. Four of Soup's file
filters, the Hub download pattern and the cache-snapshot copy took every
top-level ``*.safetensors``, so the parameter count, the download, the copy, the
spectrum scan and the layer-stream sharder all saw the model twice.

``utils/checkpoint_files.py`` is now the one answer, and every site reads it.
The fixture is the issue's reproduction: a tiny random Llama saved sharded, plus
the same weights under mistral-inference names in ``consolidated.safetensors``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from soup_cli.utils.checkpoint_files import (
    CheckpointFiles,
    checkpoint_files,
    hub_allow_patterns,
    hub_second_call,
    read_index_weight_map,
    select_checkpoint_files,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

_RENAME = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "norm.weight",
}
_LAYER_RENAME = (
    ("self_attn.q_proj", "attention.wq"), ("self_attn.k_proj", "attention.wk"),
    ("self_attn.v_proj", "attention.wv"), ("self_attn.o_proj", "attention.wo"),
    ("mlp.gate_proj", "feed_forward.w1"), ("mlp.down_proj", "feed_forward.w2"),
    ("mlp.up_proj", "feed_forward.w3"), ("input_layernorm", "attention_norm"),
    ("post_attention_layernorm", "ffn_norm"),
)


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    return cfg, LlamaForCausalLM(cfg)


def _write_consolidated(model, cfg, directory: Path) -> None:
    from safetensors.torch import save_file

    state = model.state_dict()
    rename = dict(_RENAME)
    for i in range(cfg.num_hidden_layers):
        for hf_name, mi_name in _LAYER_RENAME:
            rename[f"model.layers.{i}.{hf_name}.weight"] = f"layers.{i}.{mi_name}.weight"
    save_file(
        {rename[k]: v.clone().contiguous() for k, v in state.items()},
        str(directory / "consolidated.safetensors"),
    )


@pytest.fixture(scope="module")
def mistral_like(tmp_path_factory) -> dict:
    """Index + shards (the HF checkpoint) beside ``consolidated.safetensors``."""
    directory = tmp_path_factory.mktemp("mistral_like")
    cfg, model = _tiny_llama()
    model.save_pretrained(str(directory), max_shard_size="100KB")
    _write_consolidated(model, cfg, directory)
    names = sorted(p.name for p in directory.iterdir())
    assert "model.safetensors.index.json" in names, names
    shards = sorted(n for n in names if n.startswith("model-") and n.endswith(".safetensors"))
    assert len(shards) >= 2, names
    return {
        "dir": directory,
        "shards": shards,
        "true_params": sum(v.numel() for v in model.state_dict().values()),
        "hf_names": [n for n, _ in model.named_parameters()],
        "shard_bytes": sum(os.path.getsize(directory / s) for s in shards),
        "all_bytes": sum(os.path.getsize(directory / n) for n in names
                         if n.endswith(".safetensors")),
    }


# ---------------------------------------------------------------------------
# The helper's rules
# ---------------------------------------------------------------------------


class TestSelection:
    def test_an_index_decides_exactly(self):
        got = select_checkpoint_files(
            ["consolidated.safetensors", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors", "model.safetensors", "config.json"],
            {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
        )
        assert got.weights == ("model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors")
        assert got.ignored == ("consolidated.safetensors", "model.safetensors")
        assert got.missing == ()
        assert got.rule == "index"

    def test_an_index_naming_a_missing_shard_reports_it(self):
        got = select_checkpoint_files(
            ["model-00001-of-00002.safetensors"],
            {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"},
        )
        assert got.weights == ("model-00001-of-00002.safetensors",)
        assert got.missing == ("model-00002-of-00002.safetensors",)

    def test_without_an_index_the_single_file_wins(self):
        got = select_checkpoint_files(["consolidated.safetensors", "model.safetensors"], None)
        assert got.weights == ("model.safetensors",)
        assert got.ignored == ("consolidated.safetensors",)
        assert got.rule == "single"

    def test_without_an_index_the_shards_win(self):
        got = select_checkpoint_files(
            ["consolidated.safetensors", "model-00002-of-00002.safetensors",
             "model-00001-of-00002.safetensors"], None,
        )
        assert got.weights == ("model-00001-of-00002.safetensors",
                               "model-00002-of-00002.safetensors")
        assert got.rule == "shards"

    def test_an_unconventional_name_still_counts_but_consolidated_never(self):
        got = select_checkpoint_files(["weights.safetensors", "consolidated.00.safetensors"],
                                      None)
        assert got.weights == ("weights.safetensors",)
        assert got.ignored == ("consolidated.00.safetensors",)
        assert got.rule == "fallback"

    def test_consolidated_only_is_empty_and_the_message_names_it(self):
        got = select_checkpoint_files(["consolidated.safetensors"], None)
        assert got.weights == ()
        assert got.rule == "none"
        msg = got.why_empty("/m")
        assert "consolidated.safetensors" in msg
        assert "model.safetensors.index.json" in msg

    def test_nothing_at_all_keeps_the_old_message(self):
        got = select_checkpoint_files(["config.json"], None)
        assert got == CheckpointFiles((), (), (), "none")
        assert got.why_empty("/m") == "no .safetensors weight files found in /m"

    def test_hub_patterns_never_say_star_safetensors(self):
        assert hub_allow_patterns(None) == ["model.safetensors", "model-*-of-*.safetensors"]
        assert hub_allow_patterns({"a": "model-00001-of-00001.safetensors"}) == [
            "model-00001-of-00001.safetensors"
        ]
        assert "*.safetensors" not in hub_allow_patterns(None)

    def test_second_call_is_the_indexed_files_still_missing(self):
        got = select_checkpoint_files(["model-00002-of-00002.safetensors"],
                                      {"a": "weights-part1.safetensors",
                                       "b": "model-00002-of-00002.safetensors"})
        assert hub_second_call(got) == (["weights-part1.safetensors"], [])

    @pytest.mark.parametrize(
        "names,weight_map",
        [
            (["model-00001-of-00001.safetensors"], {"a": "model-00001-of-00001.safetensors"}),
            (["model.safetensors"], None),
            (["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"], None),
        ],
    )
    def test_no_second_call_when_the_first_holds_the_checkpoint(self, names, weight_map):
        assert hub_second_call(select_checkpoint_files(names, weight_map)) is None

    def test_second_call_without_index_is_the_fallback_rule_as_patterns(self):
        # Neither spelling came, so ask for what the local fallback would count:
        # every *.safetensors except consolidated*, top level or nested.
        import fnmatch

        allow, ignore = hub_second_call(select_checkpoint_files(["config.json"], None))
        assert allow == ["*.safetensors"]

        def fetched(name: str) -> bool:
            return (any(fnmatch.fnmatch(name, p) for p in allow)
                    and not any(fnmatch.fnmatch(name, p) for p in ignore))

        assert fetched("pytorch_model.safetensors")
        assert fetched("weights.safetensors")
        assert not fetched("consolidated.safetensors")
        assert not fetched("consolidated.00.safetensors")
        assert not fetched("sub/consolidated.safetensors")
        assert not fetched("consolidated.00.pth")


class TestIndexReading:
    def test_reads_the_weight_map_basenames(self, mistral_like):
        wm = read_index_weight_map(str(mistral_like["dir"]))
        assert wm is not None
        assert sorted(set(wm.values())) == mistral_like["shards"]

    def test_no_index_is_none(self, tmp_path: Path):
        assert read_index_weight_map(str(tmp_path)) is None

    def test_a_corrupt_index_is_a_named_failure(self, tmp_path: Path):
        (tmp_path / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            read_index_weight_map(str(tmp_path))

    def test_checkpoint_files_on_the_fixture(self, mistral_like):
        got = checkpoint_files(str(mistral_like["dir"]))
        assert list(got.weights) == mistral_like["shards"]
        assert got.ignored == ("consolidated.safetensors",)


# ---------------------------------------------------------------------------
# Every consumer, on the issue's layout
# ---------------------------------------------------------------------------


def test_model_size_from_name_counts_the_model_once(mistral_like):
    from soup_cli.utils.gpu import model_size_from_name

    got = round(model_size_from_name(str(mistral_like["dir"])) * 1e9)
    assert got == mistral_like["true_params"]


def test_a_leftover_single_file_beside_index_and_shards_is_not_counted(mistral_like, tmp_path):
    import shutil

    from soup_cli.utils.gpu import model_size_from_name

    target = tmp_path / "leftover"
    shutil.copytree(mistral_like["dir"], target)
    shutil.copy(target / mistral_like["shards"][0], target / "model.safetensors")
    got = round(model_size_from_name(str(target)) * 1e9)
    assert got == mistral_like["true_params"]


def test_plan_model_weights_lists_only_the_indexed_shards(mistral_like):
    from soup_cli.utils.spectrum_scan import plan_model_weights

    plan = plan_model_weights(str(mistral_like["dir"]))
    assert [name for name, _s, _m in plan.source_files] == mistral_like["shards"]
    assert plan.source_bytes == mistral_like["shard_bytes"]
    assert plan.source_bytes < mistral_like["all_bytes"]


def test_scan_reads_the_hf_names_only_and_selects_no_phantom(mistral_like):
    from soup_cli.utils.spectrum_scan import scan_weights_dir, select_unfrozen_parameters

    layers = scan_weights_dir(str(mistral_like["dir"]))
    names = {layer.name for layer in layers}
    assert not any(n.startswith(("layers.", "tok_embeddings", "output")) for n in names), names
    chosen = select_unfrozen_parameters(layers, top_percent=50)
    hf_names = mistral_like["hf_names"]
    phantom = [p for p in chosen if not any(n.startswith(p) for n in hf_names)]
    assert phantom == []


def test_source_weight_bytes_sums_the_shards_only(mistral_like):
    from soup_cli.utils.layer_shard import source_weight_bytes

    assert source_weight_bytes(str(mistral_like["dir"])) == mistral_like["shard_bytes"]


def test_shard_checkpoint_keeps_no_consolidated_tensor(mistral_like, tmp_path):
    from soup_cli.utils.layer_shard import shard_checkpoint

    index = shard_checkpoint(str(mistral_like["dir"]), str(tmp_path / "shards"), dtype="float32")
    assert index.total_params == mistral_like["true_params"]
    assert not any(k.startswith(("layers.", "tok_embeddings", "output", "norm."))
                   for k in index.extra_keys), index.extra_keys
    assert [name for name, _s, _m in index.source_files] == mistral_like["shards"]


# ---------------------------------------------------------------------------
# The Hub download and the snapshot copy
# ---------------------------------------------------------------------------


def test_hub_download_asks_for_the_indexed_files_not_star_safetensors(mistral_like, monkeypatch):
    from soup_cli.utils import hubs, spectrum_scan

    calls = []

    def fake(repo_id, *, cache_dir, allow_patterns=None, **kwargs):
        calls.append(list(allow_patterns or []))
        return str(mistral_like["dir"])

    monkeypatch.setattr(hubs, "snapshot_download", fake)
    plan = spectrum_scan.plan_model_weights("mistralai/Mistral-7B-Instruct-v0.3")
    # One call (a complete cache entry must keep resolving offline, #533), with
    # the Hugging Face spellings: the shards match, the consolidated copy does not.
    assert calls == [["model.safetensors", "model-*-of-*.safetensors", "*.json"]], calls
    import fnmatch

    for shard in mistral_like["shards"]:
        assert any(fnmatch.fnmatch(shard, p) for p in calls[0])
    assert not any(fnmatch.fnmatch("consolidated.safetensors", p) for p in calls[0])
    assert [name for name, _s, _m in plan.source_files] == mistral_like["shards"]


def test_an_index_naming_a_file_outside_the_hf_spellings_fetches_it_by_name(
    mistral_like, tmp_path, monkeypatch
):
    import fnmatch
    import shutil

    from soup_cli.utils import hubs, spectrum_scan

    # A repo whose index names `weights-part1.safetensors`: the first call's
    # patterns do not cover it, so a second call asks for exactly that file.
    repo = tmp_path / "repo"
    shutil.copytree(mistral_like["dir"], repo)
    first = mistral_like["shards"][0]
    os.rename(repo / first, repo / "weights-part1.safetensors")
    index_path = repo / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"] = {k: ("weights-part1.safetensors" if v == first else v)
                           for k, v in index["weight_map"].items()}
    index_path.write_text(json.dumps(index), encoding="utf-8")

    download = tmp_path / "download"  # what the fake Hub has delivered so far
    download.mkdir()
    calls = []

    def fake(repo_id, *, cache_dir, allow_patterns=None, **kwargs):
        calls.append(list(allow_patterns or []))
        for name in os.listdir(repo):
            if any(fnmatch.fnmatch(name, p) for p in allow_patterns or []):
                if not (download / name).exists():
                    shutil.copy(repo / name, download / name)
        return str(download)

    monkeypatch.setattr(hubs, "snapshot_download", fake)
    plan = spectrum_scan.plan_model_weights("org/odd-names")
    assert len(calls) == 2, calls
    assert calls[1] == ["weights-part1.safetensors", "*.json"]
    assert not (download / "consolidated.safetensors").exists()
    assert [name for name, _s, _m in plan.source_files] == sorted(
        ["weights-part1.safetensors", *mistral_like["shards"][1:]]
    )


def test_the_hub_wrapper_forwards_ignore_patterns_only_when_given(tmp_path, monkeypatch):
    import huggingface_hub

    from soup_cli.utils import hubs

    captured = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **kw: captured.append(kw) or str(tmp_path))
    hubs.snapshot_download("org/m", cache_dir=None, allow_patterns=["*.safetensors"],
                           ignore_patterns=["consolidated*.safetensors"], namespace_check=False)
    hubs.snapshot_download("org/m", cache_dir=None, allow_patterns=["*.json"],
                           namespace_check=False)
    assert captured[0]["ignore_patterns"] == ["consolidated*.safetensors"]
    assert "ignore_patterns" not in captured[1]  # the pre-#1806 call shape, unchanged
    with pytest.raises(TypeError, match="ignore_patterns"):
        hubs.snapshot_download("org/m", cache_dir=None, ignore_patterns="consolidated*",
                               namespace_check=False)


def _fake_hub(repo: Path, download: Path, calls: list):
    """A Hub that delivers the files of ``repo`` matching allow minus ignore patterns."""
    import fnmatch
    import shutil

    def fake(repo_id, *, cache_dir, allow_patterns=None, ignore_patterns=None, **kwargs):
        calls.append((list(allow_patterns or []), list(ignore_patterns or [])))
        for name in os.listdir(repo):
            if not any(fnmatch.fnmatch(name, p) for p in allow_patterns or []):
                continue
            if any(fnmatch.fnmatch(name, p) for p in ignore_patterns or []):
                continue
            if not (download / name).exists():
                shutil.copy(repo / name, download / name)
        return str(download)

    return fake


def _layout(mistral_like: dict, tmp_path: Path, name: str) -> Path:
    """Repository layouts without an index, built from the fixture's files."""
    import shutil

    repo = tmp_path / name
    repo.mkdir()
    src = mistral_like["dir"]
    first, second = mistral_like["shards"][:2]
    shutil.copy(src / "config.json", repo / "config.json")
    if name == "indexed":
        shutil.rmtree(repo)
        shutil.copytree(src, repo)
    elif name == "single":
        shutil.copy(src / first, repo / "model.safetensors")
        shutil.copy(src / "consolidated.safetensors", repo / "consolidated.safetensors")
    elif name == "shards-no-index":
        shutil.copy(src / first, repo / first)
        shutil.copy(src / second, repo / second)
        shutil.copy(src / second, repo / "extra.safetensors")
    elif name == "odd-single":
        shutil.copy(src / first, repo / "pytorch_model.safetensors")
        shutil.copy(src / "consolidated.safetensors", repo / "consolidated.safetensors")
    elif name == "consolidated-only":
        shutil.copy(src / "consolidated.safetensors", repo / "consolidated.safetensors")
    return repo


def test_a_repo_whose_only_weight_file_has_another_name_is_fetched_by_the_fallback_rule(
    mistral_like, tmp_path, monkeypatch
):
    # No index, no model.safetensors, no shards: the first call brings only the JSON,
    # and the second asks for every *.safetensors except consolidated* (#1806 review).
    from soup_cli.utils import hubs, spectrum_scan

    repo = _layout(mistral_like, tmp_path, "odd-single")
    download = tmp_path / "download"
    download.mkdir()
    calls = []
    monkeypatch.setattr(hubs, "snapshot_download", _fake_hub(repo, download, calls))
    plan = spectrum_scan.plan_model_weights("org/odd-single")
    assert len(calls) == 2, calls
    assert calls[1] == (["*.safetensors", "*.json"],
                        ["consolidated*.safetensors", "*/consolidated*.safetensors"])
    assert (download / "pytorch_model.safetensors").exists()
    assert not (download / "consolidated.safetensors").exists()
    assert [name for name, _s, _m in plan.source_files] == ["pytorch_model.safetensors"]


@pytest.mark.parametrize(
    "layout", ["indexed", "single", "shards-no-index", "odd-single", "consolidated-only"]
)
def test_what_the_download_fetches_is_what_the_local_rule_counts(
    mistral_like, tmp_path, monkeypatch, layout
):
    # The two rules agree: on every layout the safetensors the Hub delivered are
    # exactly the files the local helper selects on the full repository.
    from soup_cli.utils import hubs, spectrum_scan

    repo = _layout(mistral_like, tmp_path, layout)
    download = tmp_path / "download"
    download.mkdir()
    calls = []
    monkeypatch.setattr(hubs, "snapshot_download", _fake_hub(repo, download, calls))
    local = checkpoint_files(str(repo))
    if layout == "consolidated-only":
        with pytest.raises(FileNotFoundError, match="no .safetensors weight files"):
            spectrum_scan.plan_model_weights("org/x")
        fetched = sorted(n for n in os.listdir(download) if n.endswith(".safetensors"))
        assert fetched == [] == list(local.weights)
        return
    plan = spectrum_scan.plan_model_weights("org/x")
    fetched = sorted(n for n in os.listdir(download) if n.endswith(".safetensors"))
    assert fetched == list(local.weights)
    assert [name for name, _s, _m in plan.source_files] == list(local.weights)
    assert len(calls) == (2 if layout == "odd-single" else 1), calls


def _hf_cache_snapshot(root: Path, source: Path) -> Path:
    """Lay ``source`` out as ``models--org--name/snapshots/<sha>/`` with blob links."""
    repo = root / "models--org--name"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for i, name in enumerate(sorted(p.name for p in source.iterdir())):
        blob = blobs / f"blob{i:02d}"
        blob.write_bytes((source / name).read_bytes())
        os.symlink(str(blob), str(snapshot / name))
    return snapshot


@pytest.mark.requires_symlink
def test_the_snapshot_copy_covers_the_indexed_shards_only(mistral_like, tmp_path, monkeypatch):
    from soup_cli.utils import spectrum_scan

    snapshot = _hf_cache_snapshot(tmp_path / "hub", mistral_like["dir"])
    monkeypatch.setattr(spectrum_scan, "resolve_cache_dir", lambda: str(tmp_path / "soup-cache"))
    entries = spectrum_scan._snapshot_materialization_entries(
        str(snapshot), source_revision="a" * 40
    )
    copied = sorted(rel for rel, _src, _blob in entries if rel.endswith(".safetensors"))
    assert copied == mistral_like["shards"]
    plan = spectrum_scan.plan_model_weights(str(snapshot))
    assert plan.materialize_bytes == mistral_like["shard_bytes"]
    assert [name for name, _s, _m in plan.source_files] == mistral_like["shards"]


# ---------------------------------------------------------------------------
# The two documented edges
# ---------------------------------------------------------------------------


def test_no_index_and_a_single_model_safetensors_is_counted(tmp_path):
    from soup_cli.utils.gpu import model_size_from_name
    from soup_cli.utils.layer_shard import source_weight_bytes
    from soup_cli.utils.spectrum_scan import plan_model_weights

    cfg, model = _tiny_llama()
    model.save_pretrained(str(tmp_path))
    _write_consolidated(model, cfg, tmp_path)
    true_params = sum(v.numel() for v in model.state_dict().values())
    assert round(model_size_from_name(str(tmp_path)) * 1e9) == true_params
    assert [n for n, _s, _m in plan_model_weights(str(tmp_path)).source_files] == [
        "model.safetensors"
    ]
    assert source_weight_bytes(str(tmp_path)) == os.path.getsize(tmp_path / "model.safetensors")


def test_consolidated_only_is_refused_naming_the_file(tmp_path):
    from soup_cli.utils.gpu import _params_from_local_safetensors
    from soup_cli.utils.layer_shard import source_weight_bytes
    from soup_cli.utils.spectrum_scan import plan_model_weights, scan_weights_dir

    cfg, model = _tiny_llama()
    _write_consolidated(model, cfg, tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert _params_from_local_safetensors(str(tmp_path)) is None
    for call in (lambda: plan_model_weights(str(tmp_path)),
                 lambda: scan_weights_dir(str(tmp_path)),
                 lambda: source_weight_bytes(str(tmp_path))):
        with pytest.raises(FileNotFoundError, match="consolidated.safetensors"):
            call()


# ---------------------------------------------------------------------------
# Small guards, each pinned (#1906 review)
# ---------------------------------------------------------------------------


def test_an_index_with_an_empty_weight_map_is_a_named_failure(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="no weight_map"):
        read_index_weight_map(str(tmp_path))


def test_an_index_over_the_size_bound_is_a_named_failure(tmp_path: Path, monkeypatch):
    from soup_cli.utils import checkpoint_files as cf

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}), encoding="utf-8"
    )
    monkeypatch.setattr(cf, "_MAX_INDEX_BYTES", 16)  # the real bound is 64 MiB
    with pytest.raises(ValueError, match="exceeds"):
        read_index_weight_map(str(tmp_path))


def test_an_index_value_with_a_directory_part_resolves_to_its_basename(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "sub/model-1.safetensors"}}), encoding="utf-8"
    )
    assert read_index_weight_map(str(tmp_path)) == {"a": "model-1.safetensors"}


def _indexed_dir_with_a_directory_shard(tmp_path: Path) -> tuple[Path, int]:
    """An index naming two shards, the second of which is a directory, not a file."""
    import torch
    from safetensors.torch import save_file

    directory = tmp_path / "ckpt"
    directory.mkdir()
    save_file({"w": torch.zeros(3, 5)}, str(directory / "model-00001-of-00002.safetensors"))
    (directory / "model-00002-of-00002.safetensors").mkdir()
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001-of-00002.safetensors",
                                   "v": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    return directory, 15


def test_spectrum_discovery_skips_an_entry_that_is_not_a_file(tmp_path: Path):
    from soup_cli.utils.spectrum_scan import _discover_safetensors

    directory, _ = _indexed_dir_with_a_directory_shard(tmp_path)
    found = _discover_safetensors(str(directory))
    assert [os.path.basename(p) for p in found] == ["model-00001-of-00002.safetensors"]


def test_the_size_probe_skips_an_entry_that_is_not_a_file(tmp_path: Path):
    from soup_cli.utils.gpu import _params_from_local_safetensors

    directory, params = _indexed_dir_with_a_directory_shard(tmp_path)
    assert _params_from_local_safetensors(str(directory)) == pytest.approx(params / 1e9)
