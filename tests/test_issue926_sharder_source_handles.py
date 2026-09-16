"""#926 — the sharder must not hold every source shard's mapping open.

``shard_checkpoint`` entered every source ``*.safetensors`` into one
``ExitStack`` and kept all the memory maps alive for the whole of pass 2. On
Windows that dies with an access violation once ~96–106 GB of safetensors
mappings are alive in one process, whatever the file count (62 x 1.71 GB and
14 x 6.85 GB both crashed; per-file open/read/close passed — see
``benchmarks/probe-rtx5070-what-bounds-streaming.md`` §12–§14), so a real
Llama-3.1-70B (30 files, 141 GB) could not be sharded there at all.

Three properties are pinned, each discriminating against the shipped code:

- at most ``_MAX_LIVE_SOURCE_HANDLES`` source handles are alive at any moment
  of a sharding run (a counting ``safe_open`` stand-in; the all-handles pattern
  peaks at the FILE COUNT and fails it);
- every tensor the readers return OWNS its memory — ``get_tensor`` is a
  zero-copy view whose storage keeps the WHOLE file mapped for as long as the
  view lives (measured: the view still reads correctly after its handle's
  ``__exit__``), so a retained view would keep a mapping alive past the
  handle's release and defeat the bound;
- splitting the same tensors across many files, with one decoder layer
  straddling a file boundary, changes nothing in the written shards.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from soup_cli.utils.layer_shard import (  # noqa: E402
    _read_raw_tensor,
    _read_tensor,
    extras_shard_path,
    large_shard_path,
    layer_shard_path,
    shard_checkpoint,
)

N_LAYERS = 4
_HIDDEN = 64
_N_FILES = 6


# ==========================================================================
# fixtures: the same tensors as one file and as six
# ==========================================================================
def _llama_tensors(n_layers: int = N_LAYERS) -> dict:
    torch.manual_seed(926)
    blob = {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        blob[pre + "self_attn.q_proj.weight"] = torch.randn(_HIDDEN, _HIDDEN)
        blob[pre + "mlp.down_proj.weight"] = torch.randn(_HIDDEN, 2 * _HIDDEN)
        blob[pre + "input_layernorm.weight"] = torch.randn(_HIDDEN)
    blob["model.embed_tokens.weight"] = torch.randn(32, _HIDDEN)
    blob["model.norm.weight"] = torch.randn(_HIDDEN)
    blob["lm_head.weight"] = torch.randn(32, _HIDDEN)
    return {key: value.contiguous() for key, value in blob.items()}


def _write(path: Path, tensors: dict) -> None:
    save_file({key: value.clone() for key, value in tensors.items()}, str(path))


def _single_file_dir(tmp_path: Path) -> str:
    src = tmp_path / "single"
    src.mkdir()
    _write(src / "model.safetensors", _llama_tensors())
    return str(src)


def _multi_file_dir(tmp_path: Path) -> str:
    """The same tensors over six files, laid out the way HF checkpoints are.

    The embedding sits in the first file, the norm and head in the last, and
    decoder layer 1 STRADDLES files 2 and 3 (so a layer can need two handles
    at once), layer 2 straddles files 3 and 4.
    """
    tensors = _llama_tensors()

    def layer(idx: int, *names: str) -> list:
        return [f"model.layers.{idx}.{name}" for name in names]

    plan = [
        [
            "model.embed_tokens.weight",
            *layer(0, "self_attn.q_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight"),
        ],
        layer(1, "self_attn.q_proj.weight", "input_layernorm.weight"),
        [*layer(1, "mlp.down_proj.weight"), *layer(2, "self_attn.q_proj.weight")],
        layer(2, "mlp.down_proj.weight", "input_layernorm.weight"),
        layer(3, "self_attn.q_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight"),
        ["model.norm.weight", "lm_head.weight"],
    ]
    assert len(plan) == _N_FILES
    assert sorted(key for keys in plan for key in keys) == sorted(tensors)
    src = tmp_path / "multi"
    src.mkdir()
    for number, keys in enumerate(plan, start=1):
        name = f"model-{number:05d}-of-{_N_FILES:05d}.safetensors"
        _write(src / name, {key: tensors[key] for key in keys})
    return str(src)


class _CountingSafeOpen:
    """A ``safe_open`` stand-in that counts live handles.

    Counts from construction (the mapping exists from then) to ``__exit__``,
    so a handle that is opened and never released stays counted and shows up
    as a leak in ``live``.
    """

    live = 0
    peak = 0
    opened = 0
    real = safe_open

    def __init__(self, path, framework="pt", device="cpu"):
        cls = type(self)
        self._inner = cls.real(path, framework=framework, device=device)
        cls.live += 1
        cls.opened += 1
        cls.peak = max(cls.peak, cls.live)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        type(self).live -= 1
        return self._inner.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @classmethod
    def reset(cls) -> None:
        cls.live = cls.peak = cls.opened = 0


# ==========================================================================
# 1. the bound on live source handles
# ==========================================================================
class TestLiveHandleBound:
    def test_sharding_keeps_at_most_the_cap_alive(self, tmp_path, monkeypatch):
        import safetensors

        from soup_cli.utils.layer_shard import _MAX_LIVE_SOURCE_HANDLES

        src = _multi_file_dir(tmp_path)
        _CountingSafeOpen.reset()
        monkeypatch.setattr(safetensors, "safe_open", _CountingSafeOpen)

        index = shard_checkpoint(src, str(tmp_path / "shards"), dtype="float32")

        assert index.n_layers == N_LAYERS
        # The defect: the shipped pass 2 peaked at the file count (6 here,
        # 30 for a real 70B). The cap is the whole fix. `==`, not `<=`: this
        # fixture provably reaches the cap (layer 1 straddles two files), so a
        # regression that opened and evicted on every access would slip past a
        # one-sided bound.
        assert _CountingSafeOpen.peak == _MAX_LIVE_SOURCE_HANDLES
        # A POSITIVE CONTROL, not a cap check: any implementation, capped or
        # not, opens every file once per pass, so this only proves the
        # monkeypatch engaged and the run was not served from a warm cache.
        assert _CountingSafeOpen.opened >= 2 * _N_FILES
        # Everything is released when sharding returns.
        assert _CountingSafeOpen.live == 0

    def test_no_handle_survives_a_raise_inside_pass_two(self, tmp_path, monkeypatch):
        """The promise the LRU's docstring makes, at the level that matters.

        The isolated class test uses a fake opener; this drives the REAL
        ``shard_checkpoint`` into one of its own mid-loop refusals with real
        handles live, and requires the count to come back to zero.
        """
        import safetensors

        # A checkpoint whose layer 1 stores a shared key at a different shape:
        # shard_checkpoint refuses it partway through pass 2, after it has
        # opened real source handles.
        tensors = _llama_tensors(n_layers=2)
        tensors["model.layers.1.self_attn.q_proj.weight"] = torch.randn(_HIDDEN, 2 * _HIDDEN)
        src = tmp_path / "ragged"
        src.mkdir()
        for index, (key, value) in enumerate(tensors.items()):
            _write(src / f"model-{index:05d}.safetensors", {key: value})

        _CountingSafeOpen.reset()
        monkeypatch.setattr(safetensors, "safe_open", _CountingSafeOpen)
        with pytest.raises(ValueError, match="different stored shapes or dtypes"):
            shard_checkpoint(str(src), str(tmp_path / "shards"), dtype="float32")

        assert _CountingSafeOpen.opened > 0, "the stand-in never saw a real open"
        assert _CountingSafeOpen.live == 0, "pass 2 leaked a source handle on the raise path"

    def test_the_cap_is_a_small_constant(self):
        from soup_cli.utils.layer_shard import _MAX_LIVE_SOURCE_HANDLES

        # A layer needs one file, or two at a boundary, which is the design
        # rationale in the module comment. Anything above that is not a cap on
        # a 30-file checkpoint, it is a slightly smaller leak.
        assert 1 <= _MAX_LIVE_SOURCE_HANDLES <= 2


# ==========================================================================
# 2. the readers own their memory
# ==========================================================================
class TestReadersOwnTheirMemory:
    @pytest.fixture
    def one_file(self, tmp_path):
        path = tmp_path / "w.safetensors"
        _write(path, {"w": torch.arange(64, dtype=torch.float32).reshape(8, 8)})
        return str(path)

    def test_get_tensor_is_a_zero_copy_view_so_these_tests_discriminate(self, one_file):
        # The premise of the two tests below. If a future safetensors copies
        # in get_tensor, they stop discriminating, and this one says so.
        with safe_open(one_file, framework="pt") as handle:
            first = handle.get_tensor("w")
            second = handle.get_tensor("w")
            assert first.data_ptr() == second.data_ptr()

    def test_read_tensor_at_the_source_dtype_is_a_copy_not_the_view(self, one_file):
        with safe_open(one_file, framework="pt") as handle:
            view = handle.get_tensor("w")
            owned = _read_tensor(handle, "w", "float32")
            assert owned.data_ptr() != view.data_ptr()
            assert owned.dtype is torch.float32
            assert torch.equal(owned, view)

    def test_read_tensor_converting_dtype_is_owned_too(self, one_file):
        with safe_open(one_file, framework="pt") as handle:
            view = handle.get_tensor("w")
            owned = _read_tensor(handle, "w", "bfloat16")
            assert owned.data_ptr() != view.data_ptr()
            assert owned.dtype is torch.bfloat16

    def test_read_raw_tensor_is_a_copy_not_the_view(self, one_file):
        with safe_open(one_file, framework="pt") as handle:
            view = handle.get_tensor("w")
            owned = _read_raw_tensor(handle, "w")
            assert owned.data_ptr() != view.data_ptr()
            assert torch.equal(owned, view)

    def test_an_owned_tensor_survives_its_handle(self, one_file):
        handle = safe_open(one_file, framework="pt")
        owned = _read_tensor(handle, "w", "float32")
        handle.__exit__(None, None, None)
        assert float(owned.sum()) == float(sum(range(64)))


# ==========================================================================
# 3. the file layout does not change the shards
# ==========================================================================
class TestFileLayoutDoesNotChangeTheShards:
    def test_the_fixture_actually_forces_evictions(self):
        """Keeps the test below from decaying into a vacuous pass.

        Its power to prove "re-opening an evicted file preserves the bytes"
        rests entirely on the fixture having more files than the cap. Raise the
        cap without touching the fixture and the comparison keeps passing while
        testing nothing.
        """
        from soup_cli.utils.layer_shard import _MAX_LIVE_SOURCE_HANDLES

        assert _N_FILES > _MAX_LIVE_SOURCE_HANDLES

    def test_six_files_and_one_file_shard_byte_identically(self, tmp_path):
        multi_out = str(tmp_path / "multi_out")
        single_out = str(tmp_path / "single_out")
        multi = shard_checkpoint(_multi_file_dir(tmp_path), multi_out, dtype="float32")
        single = shard_checkpoint(_single_file_dir(tmp_path), single_out, dtype="float32")

        assert multi.layer_keys == single.layer_keys
        assert multi.extra_keys == single.extra_keys
        assert multi.large_keys == single.large_keys
        assert multi.total_params == single.total_params
        # embed_tokens + an untied lm_head: both go to their own large shard.
        assert set(multi.large_keys) == {"model.embed_tokens.weight", "lm_head.weight"}

        for idx in range(N_LAYERS):
            assert (
                Path(layer_shard_path(multi_out, idx)).read_bytes()
                == Path(layer_shard_path(single_out, idx)).read_bytes()
            ), idx
        assert (
            Path(extras_shard_path(multi_out)).read_bytes()
            == Path(extras_shard_path(single_out)).read_bytes()
        )
        for key in multi.large_keys:
            assert (
                Path(large_shard_path(multi_out, key)).read_bytes()
                == Path(large_shard_path(single_out, key)).read_bytes()
            ), key


# ==========================================================================
# 4. the LRU itself, over fake handles
# ==========================================================================
class _FakeOpener:
    """Records open/close order; no files involved."""

    def __init__(self, fail_on: str = ""):
        self.events: list = []
        self.fail_on = fail_on

    def __call__(self, path, framework="pt"):
        if path == self.fail_on:
            raise OSError(f"cannot open {path}")
        self.events.append(("open", path))
        return _FakeHandle(self, path)

    def opens(self) -> list:
        return [path for kind, path in self.events if kind == "open"]

    def closes(self) -> list:
        return [path for kind, path in self.events if kind == "close"]


class _FakeHandle:
    def __init__(self, opener: _FakeOpener, path: str, raise_on_exit: bool = False):
        self.opener = opener
        self.path = path
        self.closed = False
        self.raise_on_exit = raise_on_exit

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        self.opener.events.append(("close", self.path))
        if self.raise_on_exit:
            raise RuntimeError(f"exit failed for {self.path}")
        return None


class TestSourceHandlesLRU:
    def test_evicts_the_least_recently_used_and_releases_the_rest_on_exit(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        with _SourceHandles(["a", "b", "c", "d"], opener, capacity=2) as handles:
            first_a = handles["a"]
            handle_b = handles["b"]
            assert handles["a"] is first_a  # a hit re-uses, and refreshes, the handle
            handle_c = handles["c"]  # b is now the least recently used, not a
            assert handle_b.closed
            assert not first_a.closed and not handle_c.closed
            handles["d"]  # evicts a: c was touched after it
            assert first_a.closed
            assert not handle_c.closed
        assert opener.opens() == ["a", "b", "c", "d"]
        # b and a were evicted; c and d were released when the block ended.
        assert opener.closes() == ["b", "a", "c", "d"]

    def test_an_evicted_path_is_reopened_as_a_fresh_handle(self):
        """The contract every caller leans on, and the one the walk above stops short of.

        ``shard_checkpoint`` returns to an evicted file constantly: the extras
        loop re-reads ``model.embed_tokens.weight`` long after later layers
        evicted its file. Returning the closed object, or refusing, would be a
        silent wrong-read or a crash, and the eviction walk alone cannot see it.
        """
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        with _SourceHandles(["a", "b", "c"], opener, capacity=2) as handles:
            first_a = handles["a"]
            handles["b"]
            handles["c"]  # evicts a
            assert first_a.closed
            second_a = handles["a"]
            assert second_a is not first_a, "returned the handle it had already closed"
            assert not second_a.closed
        assert opener.opens() == ["a", "b", "c", "a"]

    def test_close_is_idempotent(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        handles = _SourceHandles(["a"], opener, capacity=2)
        handles["a"]
        handles.close()
        handles.close()  # a second drain must not raise, nor close twice
        assert opener.closes() == ["a"]

    def test_every_handle_is_released_when_more_than_one_exit_raises(self):
        """With capacity 2, "both live handles failed to close" is an ordinary case."""
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        handles = _SourceHandles(["a", "b", "c"], opener, capacity=3)
        handles["a"].raise_on_exit = True
        handles["b"].raise_on_exit = True
        handles["c"]
        with pytest.raises(RuntimeError, match="exit failed for a"):
            handles.close()
        # Every handle still got its release attempt; only the first propagates.
        assert opener.closes() == ["a", "b", "c"]

    def test_the_mapping_mixins_that_would_open_files_are_refused(self):
        """``Mapping`` builds get/items/values on ``__getitem__``, which opens files.

        ``handles.get(path)`` reads as a side-effect-free peek and is not one;
        ``items``/``values`` would open every shard in the checkpoint, which is
        the cost this class exists to remove.
        """
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        with _SourceHandles(["a", "b"], opener) as handles:
            for call in (
                lambda: handles.get("a"),
                lambda: handles.items(),
                lambda: handles.values(),
                lambda: handles == {"a": 1},
            ):
                with pytest.raises(TypeError, match="does not support"):
                    call()
        assert opener.events == [], "a refused mixin still opened a file"

    def test_membership_and_iteration_do_not_open_files(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        with _SourceHandles(["a", "b"], opener) as handles:
            assert "a" in handles
            assert "zzz" not in handles
            assert len(handles) == 2
            assert sorted(handles) == ["a", "b"]
        assert opener.events == []

    def test_unknown_path_is_refused_and_a_released_cache_is_refused(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        with _SourceHandles(["a"], opener) as handles:
            with pytest.raises(KeyError):
                handles["not-a-discovered-shard"]
            handles["a"]
        with pytest.raises(RuntimeError, match="released"):
            handles["a"]
        assert opener.closes() == ["a"]

    def test_capacity_must_be_positive(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        with pytest.raises(ValueError, match="capacity"):
            _SourceHandles(["a"], _FakeOpener(), capacity=0)

    def test_a_failed_open_still_releases_the_live_handles(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener(fail_on="c")
        with pytest.raises(OSError, match="cannot open c"):
            with _SourceHandles(["a", "b", "c"], opener, capacity=3) as handles:
                handles["a"]
                handles["b"]
                handles["c"]
        assert opener.closes() == ["a", "b"]

    def test_a_raising_exit_does_not_stop_the_other_releases(self):
        from soup_cli.utils.layer_shard import _SourceHandles

        opener = _FakeOpener()
        handles = _SourceHandles(["a", "b", "c"], opener, capacity=3)
        handles["a"]
        handles["b"].raise_on_exit = True
        handles["c"]
        with pytest.raises(RuntimeError, match="exit failed for b"):
            handles.close()
        assert opener.closes() == ["a", "b", "c"]
