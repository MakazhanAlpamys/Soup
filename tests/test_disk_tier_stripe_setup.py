"""R4 — setup: the roots reach the sharder, the depth reaches every consumer."""

import ast
import inspect
import io
import os
from types import SimpleNamespace

import pytest
from rich.console import Console

from soup_cli.trainer import stream_setup
from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD, MAX_STREAM_READ_AHEAD


def _tcfg(value):
    return SimpleNamespace(stream_read_ahead=value)


def _console():
    buffer = io.StringIO()
    return Console(file=buffer, width=200, color_system=None), buffer


def _depth(value, n_roots):
    console, out = _console()
    decision = stream_setup._effective_read_ahead(_tcfg(value), n_roots, console)
    return decision.depth, out.getvalue()


class TestTheDepth:
    def test_one_root_leaves_the_setting_alone(self):
        assert _depth(2, 1)[0] == 2
        assert _depth(1, 1)[0] == 1

    def test_the_default_becomes_one_more_than_the_drives(self):
        depth, said = _depth(DEFAULT_STREAM_READ_AHEAD, 2)
        assert depth == 3 and "stream_read_ahead" in said
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 3)[0] == 4

    def test_a_value_already_deep_enough_is_kept(self):
        """Review Focus 5: never bump what the operator already set high enough."""
        assert _depth(5, 2) == (5, "")

    def test_a_shallower_value_is_kept_with_a_warning_naming_the_idle_drives(self):
        depth, said = _depth(1, 2)
        assert depth == 1 and "idle" in said and "3" in said

    def test_the_default_never_exceeds_the_schema_ceiling(self):
        """Review Focus 5: seven roots want 8, the most the setting allows."""
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 7)[0] == 8 == MAX_STREAM_READ_AHEAD

    def test_the_clamp_holds_past_the_most_roots_the_setting_can_serve(self):
        """Seven roots reach the ceiling exactly, so only a bigger count shows the clamp."""
        assert _depth(DEFAULT_STREAM_READ_AHEAD, 9)[0] == MAX_STREAM_READ_AHEAD


def test_every_depth_consumer_in_the_setup_reads_the_effective_value():
    """The default N+1 is worth nothing if one consumer still reads the raw setting."""
    source = inspect.getsource(stream_setup)
    tree = ast.parse(source)
    raw_reads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "stream_read_ahead"
        and isinstance(node.value, ast.Name)
        and node.value.id == "tcfg"
    ]
    helper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_effective_read_ahead"
    )
    inside = range(helper.lineno, helper.end_lineno + 1)
    assert [line for line in raw_reads if line not in inside] == []


class TestTheDiskPreflight:
    def test_shares_split_the_estimate_across_roots_and_sum_to_it(self):
        assert stream_setup._stripe_write_shares(10, 1) == [10]
        shares = stream_setup._stripe_write_shares(10, 3)
        assert sum(shares) == 10 and max(shares) - min(shares) <= 1

    def test_a_full_stripe_volume_is_refused_by_name(self, tmp_path, monkeypatch):
        # `console` is the module-level Rich console: keep the panel off the real terminal.
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        full = str(tmp_path / "stripe")
        monkeypatch.setattr(
            stream_setup, "_disk_volume",
            lambda path: (2, 0) if path == full else (1, 10**15),
        )
        with pytest.raises(ValueError, match="layer-shard stripe"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=1, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=str(tmp_path), shard_bytes=10, shard_write_bytes=5,
                shard_path=str(tmp_path / "cache"), stripe_writes=((full, 5),),
            )

    def test_a_full_stripe_volume_names_the_stripe_variable(self, tmp_path, monkeypatch):
        """Final review Minor 6: the remedy named only the primary cache variables."""
        monkeypatch.setattr(stream_setup, "console", _console()[0])
        full = str(tmp_path / "stripe")
        monkeypatch.setattr(
            stream_setup, "_disk_volume",
            lambda path: (2, 0) if path == full else (1, 10**15),
        )
        with pytest.raises(ValueError, match="SOUP_LAYER_STREAM_STRIPE_DIRS"):
            stream_setup._render_stream_disk_preflight(
                source_bytes=1, materialized_copy_bytes=0, materialize_bytes=0,
                materialized_path=str(tmp_path), shard_bytes=10, shard_write_bytes=5,
                shard_path=str(tmp_path / "cache"), stripe_writes=((full, 5),),
            )

    def test_the_additional_writes_line_counts_every_stripe(self, tmp_path, monkeypatch):
        """A striped cache writes to several drives; the headline figure must add them up."""
        recorder = io.StringIO()
        monkeypatch.setattr(
            stream_setup, "console", Console(file=recorder, width=200, color_system=None)
        )
        monkeypatch.setattr(stream_setup, "_disk_volume", lambda path: (hash(path), 10**15))
        stream_setup._render_stream_disk_preflight(
            source_bytes=0, materialized_copy_bytes=0, materialize_bytes=0,
            materialized_path=str(tmp_path), shard_bytes=int(3e9), shard_write_bytes=int(1e9),
            shard_path=str(tmp_path / "cache"),
            stripe_writes=((str(tmp_path / "a"), int(1e9)), (str(tmp_path / "b"), int(1e9))),
        )
        assert "Additional writes before training: 3.00 GB" in recorder.getvalue()


def _drive_setup(tmp_path, monkeypatch, *, striped):
    """The real `_setup_streaming_transformers` on a tiny checkpoint, runtime boundary spied.

    Same harness as test_issue623 (simulated CUDA, ``build_streamed_model`` spied). Two temp
    folders share one volume, so ``volume_of`` is faked to tell them apart; everything else —
    the variable, the validation, the disk-kind callable, the sharder, the depth — is real.
    """
    import torch

    import soup_cli.utils.layer_stream as layer_stream
    import soup_cli.utils.layer_stream_runtime as runtime_module
    import soup_cli.utils.stripe_roots as stripe_module
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_issue623_stream_pin_allocation import _cfg_yaml, _tiny_checkpoint

    free = 10_000_000_000
    # A second call in the same folder reuses the checkpoint: rebuilding it would move its
    # mtimes, and the cache (keyed on them) would rightly be rewritten.
    existing = tmp_path / "model"
    weights = str(existing) if existing.is_dir() else _tiny_checkpoint(tmp_path)
    second_drive = tmp_path / "second-drive"
    second_drive.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    if striped:
        monkeypatch.setenv("SOUP_LAYER_STREAM_STRIPE_DIRS", str(second_drive))
    else:
        monkeypatch.delenv("SOUP_LAYER_STREAM_STRIPE_DIRS", raising=False)
    monkeypatch.setattr(
        stripe_module, "volume_of", lambda path: 2 if "second-drive" in str(path) else 1
    )
    monkeypatch.setattr(layer_stream, "free_ram_bytes", lambda: free)
    monkeypatch.setattr(
        layer_stream, "classify_disk_kind",
        lambda *_a, **_k: layer_stream.DiskClassification("nvme"),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a, **_k: (free, free))
    monkeypatch.setattr(runtime_module, "expandable_segments_status", lambda *_a, **_k: (True, ""))

    captured = {}

    def spy_build(**kwargs):
        from unittest.mock import MagicMock

        captured.update(kwargs)
        runtime = MagicMock()
        runtime.stats.return_value = {
            "tier": "ram", "store_bytes": 1_000_000, "disk_bytes": 1_000_000,
            "pinned": False, "read_ahead": kwargs.get("read_ahead"), "buffers": 2,
            "buffer_bytes": 4_000, "n_layers": 3,
        }
        return MagicMock(), runtime

    monkeypatch.setattr(runtime_module, "build_streamed_model", spy_build)
    buffer = io.StringIO()
    monkeypatch.setattr(
        stream_setup, "console", Console(file=buffer, width=400, color_system=None)
    )

    cfg = load_config_from_string(_cfg_yaml(weights, ""))
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.device = "cuda"
    wrapper._setup_streaming_transformers(cfg, cfg.training)
    return captured, " ".join(buffer.getvalue().split()), second_drive


class TestTheSetupPathUsesTheRoots:
    """R4 end to end: the variable reaches the sharder, the pre-flight and the reader."""

    def test_two_drives_stripe_the_cache_and_deepen_the_reader(self, tmp_path, monkeypatch):
        captured, said, second_drive = _drive_setup(tmp_path, monkeypatch, striped=True)
        index = captured["index"]
        assert index.stripe_roots == (os.path.realpath(second_drive),)
        assert index.layer_roots == (0, 1, 0)
        assert captured["read_ahead"] == DEFAULT_STREAM_READ_AHEAD + 1
        # Finding A: the page-lock advice downstream needs to know the depth was raised.
        assert captured["read_ahead_decision"].raised
        assert "Layer cache striped over" in said
        assert "layer-shard stripe 1" in said  # the second drive is charged on its own
        assert f"training.stream_read_ahead {DEFAULT_STREAM_READ_AHEAD} -> 3" in said
        stripe_folder = second_drive / os.listdir(second_drive)[0]
        assert any(name.startswith("layer_") for name in os.listdir(stripe_folder)), (
            "no layer file landed on the second drive"
        )

    def test_one_drive_is_left_exactly_as_it_was(self, tmp_path, monkeypatch):
        captured, said, _second_drive = _drive_setup(tmp_path, monkeypatch, striped=False)
        index = captured["index"]
        assert index.stripe_roots == () and index.layer_roots == ()
        assert captured["read_ahead"] == DEFAULT_STREAM_READ_AHEAD
        assert "striped" not in said and "stripe 1" not in said

    def test_a_striped_cache_is_reused_not_rewritten_on_the_next_run(self, tmp_path, monkeypatch):
        """The pre-flight asks the cache check about the SAME roots the sharder will use."""
        _drive_setup(tmp_path, monkeypatch, striped=True)
        _captured, said, _second_drive = _drive_setup(tmp_path, monkeypatch, striped=True)
        assert "(reusable)" in said
        assert "layer-shard stripe 1" not in said  # no write is planned for a cache we keep


def _spy_preflight(monkeypatch):
    """Record what the pre-flight is asked to print, then run the real thing."""
    seen = []
    real = stream_setup._render_stream_disk_preflight

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(stream_setup, "_render_stream_disk_preflight", spy)
    return seen


def _cache_files_bytes(tmp_path):
    """Every file of the cache(s) on disk, primary folder and stripe folders alike."""
    total = 0
    for top in (tmp_path / "cache", tmp_path / "second-drive"):
        for folder, _dirs, names in os.walk(top):
            total += sum(os.path.getsize(os.path.join(folder, name)) for name in names)
    return total


class TestAReusableCacheIsReportedAtItsRealSize:
    @pytest.mark.parametrize("striped", [False, True])
    def test_reusable_cache_reports_its_files_not_the_estimate(
        self, tmp_path, monkeypatch, striped
    ):
        _drive_setup(tmp_path, monkeypatch, striped=striped)  # builds the cache
        seen = _spy_preflight(monkeypatch)
        _captured, said, _second_drive = _drive_setup(tmp_path, monkeypatch, striped=striped)
        assert "(reusable)" in said
        assert seen[-1]["shard_write_bytes"] == 0
        assert seen[-1]["shard_bytes"] == _cache_files_bytes(tmp_path)
