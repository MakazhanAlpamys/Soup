"""The torch-only half of ``training.packed_lora`` on CPU: the activation stash, the pinned-memory
pool, the host-parked tables and the start-up gradient check's decision.

These modules import on a CPU build of torch, so they are measured by the coverage gate. What needs
CUDA is faked at the edge: events and streams do nothing (a CPU copy is already in order), and the
CUDA runtime's host-register calls record the addresses they were given. The same code runs for
real in ``tests/test_packed_lora_layers_gpu.py``.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from soup_cli.kernels.packed_lora import hostembed, int8check, offload  # noqa: E402
from soup_cli.kernels.packed_lora.policy import Policy  # noqa: E402


class _Cudart:
    def __init__(self):
        self.locked = {}
        self.refuse = False

    def cudaHostRegister(self, ptr, n_bytes, flags):  # noqa: N802 - the CUDA runtime's name
        if self.refuse:
            return 1
        self.locked[ptr] = n_bytes
        return 0

    def cudaHostUnregister(self, ptr):  # noqa: N802
        return 0 if self.locked.pop(ptr, None) is not None else 1


@pytest.fixture
def fake_cuda(monkeypatch):
    """Events, streams and the host-register calls of CUDA, faked so the stash runs on CPU."""

    class Event:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, stream=None):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0  # ms

    class Stream:
        device = torch.device("cpu")

        def __init__(self, device=None):
            pass

        def wait_event(self, event):
            pass

        def wait_stream(self, stream):
            pass

        def synchronize(self):
            pass

    main = Stream()
    cudart = _Cudart()
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    real_empty = torch.empty

    def empty(*args, pin_memory=False, **kwargs):  # a CPU-only torch has no pinned allocator
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    return cudart


CPU = torch.device("cpu")


# --- the pool -----------------------------------------------------------------------------


def test_the_pool_locks_what_it_allocates_and_unlocks_it_once(fake_cuda):
    pool = offload.PinnedPool()
    buffer = pool.alloc(4096)
    assert buffer.numel() == 4096 and pool.locked_bytes() == 4096
    assert len(fake_cuda.locked) == 1
    assert pool.unregister(buffer.data_ptr()) is True
    assert pool.unregister(buffer.data_ptr()) is False  # a second call does nothing
    assert pool.unregister(12345) is False  # an address the pool never saw
    assert pool.locked_bytes() == 0


def test_closing_the_pool_unlocks_everything(fake_cuda):
    pool = offload.PinnedPool()
    pool.alloc(1000), pool.alloc(2000)
    assert pool.locked_bytes() == 3000
    pool.close()
    assert pool.locked_bytes() == 0 and not fake_cuda.locked


def test_a_refused_lock_falls_back_to_the_torch_pinned_allocator(fake_cuda):
    fake_cuda.refuse = True
    pool = offload.PinnedPool()
    assert pool.alloc(512).numel() == 512 and pool.locked_bytes() == 0


# --- the link probe -----------------------------------------------------------------------


def test_the_link_is_unknown_without_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert offload.measure_link() is None


def test_a_fast_link_is_measured_and_not_slow(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    speed = offload.measure_link()  # 32 MiB in a faked millisecond
    assert speed == pytest.approx((32 << 20) / 1e-3)
    assert offload.is_slow(speed) is False


def test_a_failing_probe_answers_unknown(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: 1 / 0)
    assert offload.measure_link() is None


@pytest.mark.parametrize("speed,slow", [(None, False), (1e9, True), (3.9e9, True), (4e9, False),
                                        (12e9, False)])
def test_is_slow(speed, slow):
    assert offload.is_slow(speed) is slow


def test_stash_capacity_is_the_layer_size_of_the_config():
    config = types.SimpleNamespace(hidden_size=1536, intermediate_size=8960,
                                   num_attention_heads=12, num_key_value_heads=2)
    assert offload.stash_capacity(config, save_x=True, tokens=4096) == offload.layer_bytes(
        4096, 1536, 8960, 1536, 256, True, True, False)
    qwen3 = types.SimpleNamespace(**{**vars(config), "head_dim": 128})
    assert offload.stash_capacity(qwen3, tokens=64, qk_norm=True) == offload.layer_bytes(
        64, 1536, 8960, 1536, 256, False, True, True)


# --- the stash ----------------------------------------------------------------------------


def _tensors(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(5, 7, generator=gen), torch.randn(3, 11, generator=gen).bfloat16(),
            torch.randint(-100, 100, (257,), generator=gen, dtype=torch.int8))


def test_an_offloaded_layer_comes_back_bit_identical(fake_cuda):
    pool = offload.PinnedPool()
    stash = offload.Stash(3, 4096, pool, eager=True)
    assert len(stash.host) == 3 and pool.locked_bytes() == 3 * 4096
    stash.resident_from = 3  # all offloaded
    stash.begin_step(CPU)
    saved = _tensors()
    stash.save(1, *saved[:2])
    stash.save(1, saved[2])
    stash.release()
    stash.prefetch(1, CPU, 0, 3)
    got = stash.load(1, CPU, 0, 3)
    assert all(torch.equal(a, b) and a.dtype == b.dtype and a.shape == b.shape
               for a, b in zip(saved, got))
    # every tensor starts on an aligned offset of the layer's buffer
    assert [at % offload.ALIGN for at, *_ in stash.layout[1]] == [0, 0, 0]
    stash.close()
    assert pool.locked_bytes() == 0 and stash.host == []


def test_a_resident_layer_keeps_its_tensors_and_gives_them_up_on_load(fake_cuda):
    stash = offload.Stash(2, 0, offload.PinnedPool(), eager=False)
    stash.resident_from = 0
    stash.begin_step(CPU)
    saved = _tensors()
    stash.save(0, *saved)
    got = stash.load(0, CPU, 0, 3)
    assert all(a is b for a, b in zip(saved, got))  # contiguous: the tensors themselves
    assert stash.kept[0][:3] == [None, None, None]  # and no reference is kept afterwards


def test_a_resident_layer_keeps_a_compact_copy_of_a_strided_view(fake_cuda):
    stash = offload.Stash(1, 0, offload.PinnedPool(), eager=False)
    stash.resident_from = 0
    stash.begin_step(CPU)
    full = torch.randn(64, 8, 16, dtype=torch.bfloat16)
    view = full[:, ::4]  # GQA: 2 kv heads of 8 repeated heads
    stash.save(0, view)
    (kept,) = stash.load(0, CPU, 0, 1)
    assert torch.equal(kept, view) and kept.is_contiguous()
    assert kept.untyped_storage().nbytes() == view.numel() * 2


def test_the_buffer_of_a_layer_not_made_ahead_is_made_at_first_use(fake_cuda):
    pool = offload.PinnedPool()
    stash = offload.Stash(3, 2048, pool, n_host=1, eager=True)
    assert [h is None for h in stash.host] == [False, True, True]
    stash.resident_from = 3
    stash.begin_step(CPU)
    stash.save(2, torch.ones(10))
    assert stash.host[2] is not None and pool.locked_bytes() == 2 * 2048


def test_the_in_flight_bound_waits_for_old_copies(fake_cuda):
    stash = offload.Stash(1, 1 << 16, offload.PinnedPool())
    stash.resident_from = 1
    stash.begin_step(CPU)
    for i in range(offload.IN_FLIGHT + 3):
        stash.save(0, torch.full((8,), float(i)))
    assert len(stash.in_flight) == offload.IN_FLIGHT
    stash.release(keep=2)
    assert len(stash.held) == 2


def test_release_with_nothing_to_drop_is_a_no_op(fake_cuda):
    stash = offload.Stash(1, 64, offload.PinnedPool())
    stash.begin_step(CPU)
    stash.release(keep=3)
    assert len(stash.held) == 0


def test_a_new_step_forgets_the_last_one(fake_cuda):
    stash = offload.Stash(2, 4096, offload.PinnedPool())
    stash.resident_from = 2
    stash.begin_step(CPU)
    stash.save(0, torch.ones(4))
    stash.begin_step(CPU)
    assert stash.layout == [[], []] and stash.used == [0, 0] and not stash.held


def test_the_stash_fits_a_step_of_that_size_or_smaller():
    stash = offload.Stash(4, 1000, offload.PinnedPool(), eager=False)
    assert stash.fits(4, 1000) and stash.fits(4, 10)
    assert not stash.fits(4, 1001) and not stash.fits(5, 10)


def test_resident_only_has_no_buffers_and_everything_resident():
    stash = offload.Stash.resident_only(5, offload.PinnedPool())
    assert stash.resident_from == 0 and stash.capacity == 0 and stash.host == [None] * 5


def test_the_resident_layers_are_the_last_ones():
    stash = offload.Stash(8, 0, offload.PinnedPool(), eager=False)
    assert stash.plan_count(3) == 5 and stash.plan_count(0) == 8 and stash.plan_count(99) == 0
    assert stash.plan_count(-4) == 8


@pytest.mark.parametrize("mode,resident_from", [("resident", 0), ("offload", 8)])
def test_the_modes_of_the_per_step_plan(mode, resident_from):
    stash = offload.Stash(8, 0, offload.PinnedPool(), eager=False)
    assert stash.plan(mode, 100, 0, CPU) == resident_from


def test_a_step_the_buffers_cannot_hold_keeps_every_layer_resident():
    stash = offload.Stash(8, 0, offload.PinnedPool(), eager=False)
    assert stash.plan("offload", 100, 0, CPU, tokens_ok=False) == 0


def test_auto_keeps_as_many_last_layers_as_the_free_memory_holds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (5 << 30, 8 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 1 << 30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    stash = offload.Stash(28, 0, offload.PinnedPool(), eager=False)
    per_layer, reserve = 300 << 20, 1 << 30
    free = offload.free_gpu_bytes(CPU)
    assert free == 6 << 30
    count = (free - reserve - offload.SAFETY_BYTES) // per_layer
    assert stash.plan("auto", per_layer, reserve, CPU) == 28 - count
    assert stash.plan("auto", 10 << 30, reserve, CPU) == 28  # nothing fits: all offloaded


# --- the host-parked tables ---------------------------------------------------------------


@pytest.fixture
def pool(fake_cuda):
    return offload.PinnedPool()


def _table(rows=2000, hidden=16, dtype=torch.bfloat16):
    return torch.nn.Parameter(torch.randn(rows, hidden).to(dtype), requires_grad=False)


def test_a_weight_that_is_not_on_the_gpu_or_not_half_precision_is_not_copied(pool):
    weight = hostembed.HostWeight(pool)
    assert weight.copy_to_host(_table()) is False  # a CPU tensor: nothing to move off a GPU
    assert weight.host is None


def test_copy_to_host_is_idempotent(pool):
    weight = hostembed.HostWeight(pool)
    weight.host = torch.zeros(1)
    assert weight.copy_to_host(_table()) is True and weight.host.numel() == 1


def test_activate_checks_the_copy_before_freeing_the_gpu_weight(pool):
    param = _table()
    weight = hostembed.HostWeight(pool)
    assert weight.activate(param) is False  # no host copy yet
    weight.host = param.detach().clone()
    weight.host[997] += 1  # a row the sample compares
    assert weight.activate(param) is False and not weight.active
    weight.host = param.detach().clone()
    assert weight.activate(param) is True and weight.active
    assert param.data.data_ptr() == weight.host.data_ptr()
    assert weight.activate(param) is True  # already there


def test_activate_refuses_another_shape_or_dtype(pool):
    param = _table()
    weight = hostembed.HostWeight(pool)
    weight.host = param.detach().clone().float()
    assert weight.activate(param) is False
    weight.host = param.detach().clone()[:100]
    assert weight.activate(param) is False


def test_restore_puts_the_values_back_and_keeps_the_host_copy(pool):
    param = _table()
    values = param.detach().clone()
    weight = hostembed.HostWeight(pool)
    weight.host = values.clone()
    weight.activate(param)
    param.data = weight.host  # the parameter now lives on the "host"
    weight.restore(param, CPU)
    assert not weight.active and torch.equal(param.detach(), values) and weight.host is not None
    weight.restore(param, CPU)  # twice is fine


def test_drop_host_unlocks_only_a_weight_that_is_back_on_the_gpu(pool):
    param = _table()
    weight = hostembed.HostWeight(pool)
    host = torch.empty(4)
    assert pool.register(host)
    weight.host = host
    weight.active = True
    weight.drop_host()
    assert weight.host is host and pool.locked_bytes() > 0  # still the live storage
    weight.active = False
    weight.drop_host()
    assert weight.host is None and pool.locked_bytes() == 0
    weight.close()
    del param


def test_gather_returns_the_rows_of_the_ids_and_reuses_its_staging_buffer(pool):
    weight = hostembed.HostWeight(pool)
    weight.host = _table(rows=100).detach().clone()
    ids = torch.tensor([5, 5, 99, 0, 17])
    rows = weight.gather(ids, CPU)
    assert torch.equal(rows, weight.host[ids])
    staging = weight.staging
    weight.gather(torch.tensor([1, 2]), CPU)
    assert weight.staging is staging  # big enough: no new buffer
    weight.close()
    assert weight.staging is None and weight.host is None


def test_tied_tables_are_one_weight(pool):
    tied = hostembed.HostTables(pool, tied=True)
    assert tied.head is tied.embed and tied.weights() == [tied.embed]
    untied = hostembed.HostTables(pool, tied=False)
    assert untied.head is not untied.embed and len(untied.weights()) == 2


def _stub_copy(monkeypatch):
    def copy_to_host(self, weight):
        self.host = weight.detach().clone()
        return True

    monkeypatch.setattr(hostembed.HostWeight, "copy_to_host", copy_to_host)


@pytest.mark.parametrize("tied", [True, False])
def test_park_and_restore_round_trip(pool, monkeypatch, tied):
    _stub_copy(monkeypatch)
    embed_p = _table()
    head_p = embed_p if tied else _table()
    values = (embed_p.detach().clone(), head_p.detach().clone())
    tables = hostembed.HostTables(pool, tied)
    assert tables.park(embed_p, head_p) is True
    assert tables.active_embed() and tables.active_head() and tables.any_active()
    tables.restore(embed_p, head_p, CPU)
    assert not tables.any_active()
    assert torch.equal(embed_p.detach(), values[0]) and torch.equal(head_p.detach(), values[1])
    tables.close()


def test_park_can_leave_one_table_where_it_is(pool, monkeypatch):
    _stub_copy(monkeypatch)
    embed_p, head_p = _table(), _table()
    tables = hostembed.HostTables(pool, tied=False)
    assert tables.park(embed_p, head_p, embed=False, head=True) is True
    assert tables.active_head() and not tables.active_embed()


def test_park_reports_a_table_that_could_not_move(pool, monkeypatch):
    monkeypatch.setattr(hostembed.HostWeight, "copy_to_host", lambda self, weight: False)
    tables = hostembed.HostTables(pool, tied=False)
    assert tables.park(_table(), _table()) is False and not tables.any_active()


def test_gather_pair_is_one_copy_for_a_tied_table_and_two_otherwise(pool):
    ids, targets = torch.tensor([3, 4, 5]), torch.tensor([7, 8])
    tied = hostembed.HostTables(pool, tied=True)
    tied.embed.host = _table(rows=50).detach().clone()
    rows, target_rows = tied.gather_pair(ids, targets, CPU)
    assert torch.equal(rows, tied.embed.host[ids]) and torch.equal(target_rows,
                                                                   tied.embed.host[targets])
    untied = hostembed.HostTables(pool, tied=False)
    untied.embed.host, untied.head.host = _table(rows=50).detach().clone(), _table(
        rows=50).detach().clone()
    rows, target_rows = untied.gather_pair(ids, targets, CPU)
    assert torch.equal(rows, untied.embed.host[ids])
    assert torch.equal(target_rows, untied.head.host[targets])
    assert torch.equal(untied.gather(ids, CPU), untied.embed.host[ids])


# --- the start-up gradient check's decision -----------------------------------------------


def _fake_packed(inter):
    pack = types.SimpleNamespace(shapes=[0] * 4 + [(inter,)])
    return types.SimpleNamespace(layers=[types.SimpleNamespace(pack=pack)], dtype=torch.bfloat16)


@pytest.fixture
def fake_layers(monkeypatch):
    """A stand-in for ``layers`` (which imports Triton): the check only asks it whether the width
    can be packed."""
    module = types.ModuleType("soup_cli.kernels.packed_lora.layers")

    def uses_int8(layer, tokens, switch):
        raise AssertionError("not used by the check")

    uses_int8.supported = lambda layer, tokens: tokens > 16 and layer.pack.shapes[4][0] % 64 == 0
    module.uses_int8 = uses_int8
    monkeypatch.setitem(sys.modules, "soup_cli.kernels.packed_lora.layers", module)


@pytest.mark.parametrize("rounding,cosine,expected", [
    ("grad", 0.9985, True),  # stochastic rounding: the bar is 0.998 ...
    ("nearest", 0.9985, False),  # ... round-to-nearest keeps the 0.999 bar
    ("nearest", 0.9995, True),
    ("grad", 0.9975, False),
    ("grad", 0.5, False),
])
def test_the_int8_check_compares_the_cosine_with_the_bar_of_the_rounding(
        fake_layers, monkeypatch, rounding, cosine, expected):
    pol = Policy(int8_rounding=rounding)
    switch, lines = pol.int8_switch(), []
    monkeypatch.setattr(int8check, "measure", lambda layer, dtype, rounding_: (cosine, 0.05))
    assert int8check.decide(_fake_packed(512), switch, pol.rounding(), lines.append) is expected
    assert switch.decided is expected and switch.check == {"cosine": cosine, "rel_err": 0.05}
    assert f"int8_backward=auto -> {'on' if expected else 'off'}" in lines[0]
    assert ("stochastic rounding" in lines[0]) is (rounding != "nearest")


def test_a_failing_int8_check_is_off_and_says_so(fake_layers, monkeypatch):
    def boom(layer, dtype, rounding):
        raise RuntimeError("no int8 gemm here")

    pol = Policy()
    switch, lines = pol.int8_switch(), []
    monkeypatch.setattr(int8check, "measure", boom)
    assert int8check.decide(_fake_packed(512), switch, pol.rounding(), lines.append) is False
    assert switch.decided is False
    assert "the gradient check failed: RuntimeError('no int8 gemm here')" in lines[0]
    switch = pol.int8_switch()  # and silently without a log function
    assert int8check.decide(_fake_packed(512), switch, pol.rounding()) is False


def test_a_width_the_packs_cannot_hold_skips_the_check_and_turns_int8_off(fake_layers):
    pol = Policy()
    switch = pol.int8_switch()
    assert int8check.decide(_fake_packed(500), switch, pol.rounding()) is False
    assert switch.decided is False


def test_a_decided_or_fixed_switch_is_not_checked_again(fake_layers):
    pol = Policy(int8_backward="on")
    switch = pol.int8_switch()
    assert int8check.decide(_fake_packed(512), switch, pol.rounding()) is True
    assert switch.decided is None  # nothing was measured
    off = Policy(int8_backward="off").int8_switch()
    assert int8check.decide(_fake_packed(512), off, pol.rounding()) is False


def test_the_check_inputs_are_heavy_tailed_and_reproducible():
    gen = torch.Generator().manual_seed(1234)
    a = int8check._heavy(64, 512, 1.0, gen, "cpu", spikes=0.01)
    gen = torch.Generator().manual_seed(1234)
    assert torch.equal(a, int8check._heavy(64, 512, 1.0, gen, "cpu", spikes=0.01))
    assert float(a.abs().max()) > 5 * float(a.abs().median())  # the 6-sigma outliers
    assert int8check._cosine(a, a) == pytest.approx(1.0)
    assert int8check._cosine(a, -a) == pytest.approx(-1.0)


def test_a_half_precision_gpu_weight_is_copied(pool, monkeypatch):
    def copy_from_gpu(self, weight):
        self.host = torch.zeros(3)

    monkeypatch.setattr(hostembed.HostWeight, "_copy_from_gpu", copy_from_gpu)
    weight = hostembed.HostWeight(pool)
    on_gpu = types.SimpleNamespace(dtype=torch.bfloat16, is_cuda=True)
    assert weight.copy_to_host(on_gpu) is True and weight.host.numel() == 3
    other = hostembed.HostWeight(pool)
    assert other.copy_to_host(types.SimpleNamespace(dtype=torch.float32, is_cuda=True)) is False


def test_the_self_test_compares_with_a_tolerance():
    from soup_cli.kernels.packed_lora import selftest

    a = torch.tensor([1.0, 2.0, 3.0])
    assert selftest._close(a, a * 1.0001, 1e-3, 1e-6) is True
    assert selftest._close(a, a * 1.1, 1e-3, 1e-6) is False
