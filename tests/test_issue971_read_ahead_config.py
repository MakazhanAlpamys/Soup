"""#971 — training.stream_read_ahead, mirroring stream_buffers."""

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD

_BASE = """
base: meta-llama/Llama-3.1-8B
task: sft
data:
  train: data.jsonl
training:
  batch_size: 1
  quantization: 4bit
  stream_layers: true
  lora:
    r: 8
"""


def test_default_is_two():
    config = load_config_from_string(_BASE)
    assert config.training.stream_read_ahead == 2


@pytest.mark.parametrize("value", [1, 4, 8])
def test_accepts_the_documented_range(value):
    config = load_config_from_string(_BASE + f"  stream_read_ahead: {value}\n")
    assert config.training.stream_read_ahead == value


@pytest.mark.parametrize("value", [0, 9, -1])
def test_refuses_out_of_range_by_name(value):
    with pytest.raises(ValueError, match="stream_read_ahead"):
        load_config_from_string(_BASE + f"  stream_read_ahead: {value}\n")


def test_refuses_bool_as_int():
    with pytest.raises(ValueError, match="must be an int, not bool"):
        load_config_from_string(_BASE + "  stream_read_ahead: true\n")


def test_is_footgun_rejected_while_streaming_is_off():
    off = _BASE.replace("stream_layers: true", "stream_layers: false")
    with pytest.raises(ValueError, match="stream_read_ahead"):
        load_config_from_string(off + "  stream_read_ahead: 4\n")


def test_the_schema_bound_and_the_runtime_bound_are_the_same_object():
    """A second copy of the bound would let the message and the check disagree."""
    from soup_cli.utils.async_disk_source import (
        MAX_STREAM_READ_AHEAD,
        MIN_STREAM_READ_AHEAD,
    )
    from soup_cli.utils.layer_stream import (
        MAX_STREAM_READ_AHEAD as LS_MAX,
    )
    from soup_cli.utils.layer_stream import (
        MIN_STREAM_READ_AHEAD as LS_MIN,
    )

    assert (LS_MIN, LS_MAX) == (MIN_STREAM_READ_AHEAD, MAX_STREAM_READ_AHEAD)


class TestItIsRefusedWhereItCannotTakeEffect:
    """`stream_source: ram` INSISTS on the RAM tier — it never falls back to
    disk (stream_setup: 'ram' insists, 'disk' forces, 'auto' falls back). The
    read-ahead reader belongs to the disk tier, so a non-default depth beside
    `ram` is a setting that validates, documents and does nothing: #748's class,
    which this very field was caught by.
    """

    def _yaml(self, *, source: str, depth: str = "") -> str:
        return _BASE + f"  stream_source: {source}\n" + depth

    def test_a_non_default_depth_beside_ram_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_string(
                self._yaml(source="ram", depth="  stream_read_ahead: 4\n")
            )
        message = str(excinfo.value)
        # BOTH names: the user set one field and is being refused because of
        # the other, so a message naming only one leaves them guessing.
        assert "stream_read_ahead" in message, message
        assert "stream_source" in message, message
        assert "4" in message, message

    def test_the_default_is_accepted_beside_ram(self):
        """A default is not a decision. Refusing it would make `stream_source:
        ram` unusable with any config that never mentions the depth at all."""
        config = load_config_from_string(self._yaml(source="ram"))
        assert config.training.stream_source == "ram"
        assert config.training.stream_read_ahead == DEFAULT_STREAM_READ_AHEAD

    @pytest.mark.parametrize("source", ["auto", "disk"])
    @pytest.mark.parametrize("depth", [1, 4, 8])
    def test_every_depth_is_accepted_on_the_tiers_that_can_use_it(self, source, depth):
        """The controls. Without them a validator that refused a non-default
        depth outright — or on every tier — would pass the refusal test."""
        config = load_config_from_string(
            self._yaml(source=source, depth=f"  stream_read_ahead: {depth}\n")
        )
        assert config.training.stream_read_ahead == depth
        assert config.training.stream_source == source


#: A 70B-ish shape: 80 decoder layers, and vocabulary weights far larger than a
#: decoder layer, which is what makes the "one slot each at any depth" clause
#: load-bearing rather than a rounding error.
_N_LAYERS = 80
_LAYER_BYTES = 441 * 10**6
_LARGE_STORE_BYTES = 2 * 2_100 * 10**6
_EMBED_BYTES = 600 * 10**6


def _plan(*, read_ahead, available_ram_bytes):
    from soup_cli.utils.layer_stream import build_stream_plan

    return build_stream_plan(
        arch="llama",
        n_layers=_N_LAYERS,
        layer_bytes=_LAYER_BYTES,
        embed_bytes=_EMBED_BYTES,
        large_store_bytes=_LARGE_STORE_BYTES,
        large_buffer_bytes=2_100 * 10**6,
        available_ram_bytes=available_ram_bytes,
        pinned_limit_bytes=None,
        buffers=2,
        disk_kind="nvme",
        read_ahead=read_ahead,
    )


class TestTheStagingIsCharged:
    """The disk tier predicted ZERO host residency while the async reader
    page-locks whole layers for the life of the run.

    True of the synchronous source it replaced — that one allocated per call
    and held nothing — and false of this one. The pre-flight's job is to say
    what a run will cost BEFORE it runs, and this was the one resource it
    described as free.
    """

    _RAM = 10**12  # comfortably fits the store
    _NO_RAM = 10**9  # store is ~39 GB, so this forces the disk tier

    @pytest.mark.parametrize("read_ahead", [1, 2, 4, 8])
    def test_the_arithmetic_is_depth_times_a_layer_plus_the_vocabulary_once(
        self, read_ahead
    ):
        plan = _plan(read_ahead=read_ahead, available_ram_bytes=self._NO_RAM)
        assert plan.tier == "disk"
        assert plan.read_ahead == read_ahead
        assert plan.staging_bytes == read_ahead * _LAYER_BYTES + _LARGE_STORE_BYTES

    def test_the_vocabulary_slots_are_not_multiplied_by_the_depth(self):
        """The finding behind the clause: embed and an untied lm_head are ONE
        layer each, so a deeper read-ahead cannot give them a second slot. A
        formula that multiplied the whole store by the depth would agree with
        the test above at depth 1 and disagree here."""
        shallow = _plan(read_ahead=1, available_ram_bytes=self._NO_RAM)
        deep = _plan(read_ahead=8, available_ram_bytes=self._NO_RAM)
        grew = deep.staging_bytes - shallow.staging_bytes
        assert grew == 7 * _LAYER_BYTES, (shallow.staging_bytes, deep.staging_bytes)

    def test_depth_beyond_the_layer_count_buys_nothing(self):
        from soup_cli.utils.layer_stream import staging_bytes_for

        assert staging_bytes_for(
            read_ahead=8, n_layers=3, layer_bytes=_LAYER_BYTES
        ) == 3 * _LAYER_BYTES

    def test_the_ram_tier_charges_none_of_it(self):
        """The control. There is no reader on the RAM tier — the whole base is
        resident and already charged as `store_bytes`, so charging staging too
        would double-count it."""
        plan = _plan(read_ahead=8, available_ram_bytes=self._RAM)
        assert plan.tier == "ram"
        assert plan.staging_bytes == 0
        assert plan.store_bytes > 0


class TestThePanelSaysWhatTheStagingCosts:
    def _render(self, plan):
        import io

        from rich.console import Console

        from soup_cli.utils.layer_stream import render_stream_panel

        buffer = io.StringIO()
        Console(file=buffer, width=200, no_color=True).print(render_stream_panel(plan))
        return buffer.getvalue()

    def test_the_disk_tier_prints_the_depth_and_the_figure(self):
        plan = _plan(read_ahead=4, available_ram_bytes=TestTheStagingIsCharged._NO_RAM)
        out = self._render(plan)
        assert "host staging" in out, out
        assert "read_ahead 4" in out, out
        # 4 x 441 MB + 4200 MB = 5964 MB, computed not hardcoded.
        assert f"{plan.staging_bytes / 1e6:.0f} MB" in out, out
        assert "page-locked when the box allows" in out, out

    def test_the_depth_printed_is_the_depth_configured(self):
        """Two depths, each asserting the other's figure is absent: a panel that
        hardcoded one would satisfy a single case."""
        shallow = _plan(
            read_ahead=1, available_ram_bytes=TestTheStagingIsCharged._NO_RAM
        )
        deep = _plan(read_ahead=8, available_ram_bytes=TestTheStagingIsCharged._NO_RAM)
        shallow_out, deep_out = self._render(shallow), self._render(deep)
        # Present-AND-absent. Absence alone passes vacuously against a panel
        # that prints no staging line at all, which is the mutation this pair
        # exists to catch.
        assert "read_ahead 1" in shallow_out, shallow_out
        assert "read_ahead 8" in deep_out, deep_out
        assert f"{shallow.staging_bytes / 1e6:.0f} MB" in shallow_out, shallow_out
        assert f"{deep.staging_bytes / 1e6:.0f} MB" in deep_out, deep_out
        assert f"{shallow.staging_bytes / 1e6:.0f} MB" not in deep_out, deep_out
        assert f"{deep.staging_bytes / 1e6:.0f} MB" not in shallow_out, shallow_out

    def test_the_ram_tier_does_not_print_it(self):
        out = self._render(
            _plan(read_ahead=8, available_ram_bytes=TestTheStagingIsCharged._RAM)
        )
        assert "host staging" not in out, out


class TestTheStagingMustFitFreeRam:
    """The RAM tier has had this check since v0.72.0 (`free_ram_bytes` against
    `choose_tier`'s 0.7 headroom, strict `<`). The disk tier had none at all —
    on a box chosen for this tier BECAUSE its RAM cannot hold the model."""

    def _validate(self, *, staging, read_ahead, free_ram, resident=0):
        from soup_cli.trainer.stream_setup import _validate_stream_staging_ram_fit

        return _validate_stream_staging_ram_fit(
            staging_bytes=staging,
            read_ahead=read_ahead,
            free_ram=free_ram,
            resident_ram=resident,
        )

    def test_it_refuses_by_name_and_offers_the_knob(self):
        with pytest.raises(ValueError) as excinfo:
            self._validate(staging=5 * 10**9, read_ahead=4, free_ram=4 * 10**9)
        message = str(excinfo.value)
        assert "training.stream_read_ahead" in message, message
        assert "5.00 GB" in message, message
        assert "currently 4" in message, message

    def test_it_charges_the_resident_extras_too(self):
        """1.0 GB of staging fits 2.0 GB of free RAM on its own (budget 1.4 GB);
        with 0.5 GB of resident extras it does not. Without the second term the
        refusal would come one allocation too late."""
        self._validate(staging=10**9, read_ahead=2, free_ram=2 * 10**9)
        with pytest.raises(ValueError, match="stream_read_ahead"):
            self._validate(
                staging=10**9, read_ahead=2, free_ram=2 * 10**9, resident=5 * 10**8
            )

    def test_a_run_that_fits_is_not_refused(self):
        """The control: a check that refused everything would pass the case
        above and brick the tier."""
        self._validate(staging=10**9, read_ahead=2, free_ram=10 * 10**9)

    def test_at_the_floor_it_does_not_suggest_lowering_the_depth(self):
        """1 is MIN_STREAM_READ_AHEAD. An impossible remedy is worse than none —
        it reads as "you did not try hard enough" — and the real remedies (free
        RAM, smaller base) still stand."""
        with pytest.raises(ValueError) as excinfo:
            self._validate(staging=5 * 10**9, read_ahead=1, free_ram=4 * 10**9)
        message = str(excinfo.value)
        assert "Lower training.stream_read_ahead" not in message, message
        assert "free RAM" in message, message


def test_a_pathological_number_of_layer_shapes_is_refused(tmp_path):
    """Staging is allocated per DISTINCT layer spec, and a spec with one member
    takes a full slot at ANY depth — so an index whose every layer differs would
    page-lock the whole model, which is what this tier exists to avoid. A real
    model has 1-3 shapes; the gate is 8."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from soup_cli.utils.async_disk_source import _MAX_SPEC_GROUPS, AsyncDiskSource
    from soup_cli.utils.layer_shard import layer_shard_path

    n_layers = _MAX_SPEC_GROUPS + 1
    out = tmp_path / "shards"
    out.mkdir()
    spec = []
    for idx in range(n_layers):
        # A different SHAPE per layer, so every one is its own spec group.
        save_file(
            {"w": torch.zeros(idx + 1, dtype=torch.float32)},
            layer_shard_path(str(out), idx),
        )
        spec.append({"w": ((idx + 1,), "float32")})

    with pytest.raises(ValueError) as excinfo:
        AsyncDiskSource(str(out), n_layers, spec, pin=False)
    message = str(excinfo.value)
    assert f"{n_layers} distinct" in message, message
    assert str(_MAX_SPEC_GROUPS) in message, message

    # The control: one shape fewer is accepted, so the gate is a bound and not
    # a blanket refusal of heterogeneous indexes.
    source = AsyncDiskSource(str(out), _MAX_SPEC_GROUPS, spec[:_MAX_SPEC_GROUPS], pin=False)
    try:
        assert source.get(0, "w") is not None
    finally:
        source.close()
