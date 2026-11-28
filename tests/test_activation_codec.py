"""activation_codec: per-row absmax int8 / fp8_e4m3 / int4 codecs, their byte arithmetic and
the refusal of non-finite rows. Torch is imported by the tests, never by the module."""

from __future__ import annotations

import pytest

from soup_cli.utils import activation_codec

torch = pytest.importorskip("torch")

BF16_TINY = 2.0**-133  # the smallest bf16 denormal
BF16_MAX = torch.finfo(torch.bfloat16).max


def _adversarial(hidden: int = 96):
    """One bf16 row per hazard, (rows, hidden)."""
    gen = torch.Generator().manual_seed(5)
    normal = torch.randn(hidden, generator=gen)
    rows = {
        "gaussian": normal,
        "massive_outlier": torch.cat([torch.tensor([3.0e4]), normal[1:] * 0.5]),
        "two_signs_outlier": torch.cat([torch.tensor([-2.0e3, 1.9e3]), normal[2:]]),
        "zeros": torch.zeros(hidden),
        "one_nonzero": torch.cat([torch.tensor([1e-3]), torch.zeros(hidden - 1)]),
        "denormals": torch.arange(hidden, dtype=torch.float32) % 5 * BF16_TINY,
        "mixed_denormal": torch.cat([torch.tensor([1.0]), torch.full((hidden - 1,), BF16_TINY)]),
        "bf16_max": torch.cat([torch.tensor([BF16_MAX, -BF16_MAX]), normal[2:] * 1e30]),
        "bf16_min_normal": normal * torch.finfo(torch.bfloat16).tiny,
        "constant": torch.full((hidden,), -7.25),
    }
    names = list(rows)
    x = torch.stack([rows[name] for name in names]).to(torch.bfloat16)
    return names, x


def _element_bound(name, x, absmax):
    if name == "int8":
        codec_error = absmax / 254
    elif name == "int4":
        codec_error = absmax / 14
    else:
        codec_error = x.abs() * 2.0**-4 + absmax * 2.0**-10 / 448
    # The restored value is rounded to bf16: half an ulp, or half the denormal grid.
    return codec_error * (1 + 2.0**-8) + x.abs() * 2.0**-8 + BF16_TINY


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_round_trip_stays_inside_the_element_bound_on_adversarial_rows(name):
    names, x = _adversarial()
    restored = activation_codec.round_trip(x, name)
    assert restored.dtype == torch.bfloat16
    assert restored.shape == x.shape
    xf, rf = x.float(), restored.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True)
    assert torch.isfinite(rf).all(), "a finite bf16 row must restore finite, extremes included"
    excess = (rf - xf).abs() - _element_bound(name, xf, absmax)
    worst = excess.amax(dim=-1)
    bad = [names[i] for i in range(len(names)) if worst[i] > 0]
    assert not bad, f"{name} breaks its element bound on rows {bad}"


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_the_row_absmax_element_and_zero_rows_restore_exactly(name):
    names, x = _adversarial(hidden=97)  # odd: int4 packs a pad nibble
    restored = activation_codec.round_trip(x, name)
    idx = x.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, idx), x.gather(-1, idx))
    zero = names.index("zeros")
    assert torch.equal(restored[zero], torch.zeros_like(restored[zero]))
    assert not torch.signbit(restored[zero]).any()


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_refused_not_encoded(name, bad):
    """A NaN or Inf has no absmax to scale by: encoding it would store garbage."""
    x = torch.randn(3, 16).to(torch.bfloat16)
    x[1, 4] = bad
    with pytest.raises(activation_codec.NonFiniteActivationError, match="1 of 3 rows"):
        activation_codec.encode(x, name)
    with pytest.raises(activation_codec.NonFiniteActivationError):
        activation_codec.round_trip(x, name)
    clean = x[[0, 2]]
    assert torch.isfinite(activation_codec.round_trip(clean, name)).all()


def test_scales_are_the_float32_row_absmax():
    x = torch.tensor([[1.5, -3.0, 0.25], [0.0, 0.0, 0.0], [BF16_TINY, 0.0, -BF16_TINY]])
    x = x.to(torch.bfloat16)
    for name in ("int8", "fp8_e4m3", "int4"):
        _payload, scale = activation_codec.encode(x, name)
        assert scale.dtype == torch.float32
        assert torch.equal(scale, x.float().abs().amax(dim=-1))


def test_payload_dtypes_and_identity_is_untouched():
    x = torch.randn(1, 4, 10).to(torch.bfloat16)
    assert activation_codec.encode(x, "int8")[0].dtype == torch.int8
    assert activation_codec.encode(x, "fp8_e4m3")[0].dtype == torch.float8_e4m3fn
    packed = activation_codec.encode(x, "int4")[0]
    assert packed.dtype == torch.uint8 and packed.shape == (1, 4, 5)
    payload, scale = activation_codec.encode(x, "bf16")
    assert payload is x and scale is None
    assert activation_codec.round_trip(x, "bf16") is x


def test_unknown_codec_is_refused():
    with pytest.raises(ValueError, match="unknown codec"):
        activation_codec.codec_of("int2")


def test_chunk_bytes_on_the_7b_shape():
    shape = (1, 512, 4096)
    assert activation_codec.compressed_chunk_bytes(shape, "bf16") == 512 * 4096 * 2
    assert activation_codec.compressed_chunk_bytes(shape, "int8") == 512 * 4096 + 512 * 4
    assert activation_codec.compressed_chunk_bytes(shape, "fp8_e4m3") == 512 * 4096 + 512 * 4
    assert activation_codec.compressed_chunk_bytes(shape, "int4") == 512 * 2048 + 512 * 4
    assert activation_codec.payload_chunk_bytes(shape, "int8") == 512 * 4096
    assert activation_codec.scale_chunk_bytes(shape, "int8") == 2048
    assert activation_codec.scale_chunk_bytes(shape, "bf16") == 0


def test_tokens_at_budget_is_floor_arithmetic():
    budget = 32 * 4096 * 2 * 1000 + 5
    assert activation_codec.tokens_at_budget(budget, n_layers=32, hidden=4096, codec="bf16") == 1000
    assert activation_codec.tokens_at_budget(budget, n_layers=32, hidden=4096, codec="int8") == 1998


@pytest.mark.parametrize("hidden", [1, 7, 97, 4097])
def test_odd_hidden_int4_packs_one_pad_nibble_and_round_trips(hidden):
    """An odd row packs to ceil(hidden / 2) bytes: the pad nibble is real storage."""
    x = torch.randn(2, 5, hidden).to(torch.bfloat16)
    payload, scale = activation_codec.encode(x, "int4")
    assert payload.shape == (2, 5, (hidden + 1) // 2)
    assert payload.numel() * payload.element_size() == activation_codec.payload_chunk_bytes(
        (2, 5, hidden), "int4")
    assert scale.numel() * scale.element_size() == activation_codec.scale_chunk_bytes(
        (2, 5, hidden), "int4")
    restored = activation_codec.round_trip(x, "int4")
    assert restored.shape == x.shape
    idx = x.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, idx), x.gather(-1, idx))


def _torch():
    return pytest.importorskip("torch")


def _reference_fp8(tensor):
    """The original operation order, independent of the shared codec entry points."""
    torch = _torch()
    values = tensor.float()
    absmax = values.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    payload = (values / safe * 448).clamp_(-448, 448).to(torch.float8_e4m3fn)
    return (payload.float() / 448 * absmax).to(tensor.dtype)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_shared_fp8_matches_the_original_operation_order(dtype):
    torch = _torch()
    from soup_cli.utils import activation_codec as codec

    tensor = torch.randn(3, 7, 19, generator=torch.Generator().manual_seed(51))
    tensor = tensor.to(getattr(torch, dtype))
    expected = _reference_fp8(tensor)
    assert torch.equal(codec.round_trip(tensor, codec.FP8), expected)
    assert codec.payload_chunk_bytes((1, 512, 4096), codec.FP8) == 2_097_152
    assert codec.scale_chunk_bytes((1, 512, 4096), codec.FP8) == 2048


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_shared_fp8_codecs_keep_zero_denormal_and_extreme_absmax_exact(name):
    torch = _torch()
    from soup_cli.utils import activation_codec as codec

    maximum = torch.finfo(torch.bfloat16).max
    tiny = 2.0 ** -133
    tensor = torch.tensor(
        [[0, 0, 0, 0, 0], [maximum, -maximum, 1e30, -1e30, 0],
         [tiny, -tiny, 0, tiny, 0], [-7.25, -7.25, -7.25, -7.25, -7.25]],
        dtype=torch.bfloat16,
    )
    payload, scale = codec.encode(tensor, name)
    restored = codec.decode(payload, scale, name, tensor.dtype, 5)
    assert torch.isfinite(restored).all()
    index = tensor.float().abs().argmax(dim=-1, keepdim=True)
    assert torch.equal(restored.gather(-1, index), tensor.gather(-1, index))
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert torch.equal(scale, tensor.float().abs().amax(dim=-1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_shared_fp8_refuses_nonfinite_values(bad):
    torch = _torch()
    from soup_cli.utils.activation_codec import FP8, NonFiniteActivationError, encode

    tensor = torch.zeros((2, 3, 5))
    tensor[1, 2, 4] = bad
    with pytest.raises(NonFiniteActivationError, match="1 of 6 rows"):
        encode(tensor, FP8)


# --------------------------------------------------------------------------
# decode refuses what does not fit, before it allocates
# --------------------------------------------------------------------------
def _encoded(name, hidden=10):
    x = torch.randn(2, 3, hidden).to(torch.bfloat16)
    payload, scale = activation_codec.encode(x, name)
    return x, payload, scale


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_decode_without_a_scale_names_the_scale(name):
    _x, payload, _scale = _encoded(name)
    with pytest.raises(ValueError, match="scale is required"):
        activation_codec.decode(payload, None, name, torch.bfloat16, 10)


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3", "int4"])
def test_decode_with_a_hidden_wider_than_the_payload_is_refused(name):
    _x, payload, scale = _encoded(name)
    with pytest.raises(ValueError, match="hidden=40 needs a payload width"):
        activation_codec.decode(payload, scale, name, torch.bfloat16, 40)


@pytest.mark.parametrize("name", ["int8", "fp8_e4m3"])
def test_decode_with_a_hidden_narrower_than_an_unpacked_payload_is_refused(name):
    _x, payload, scale = _encoded(name)
    with pytest.raises(ValueError, match="hidden=4 needs a payload width of 4"):
        activation_codec.decode(payload, scale, name, torch.bfloat16, 4)


def test_decode_int4_accepts_an_odd_hidden_and_refuses_a_mismatched_one():
    x, payload, scale = _encoded("int4", hidden=9)
    assert activation_codec.decode(payload, scale, "int4", torch.bfloat16, 9).shape == x.shape
    with pytest.raises(ValueError, match="hidden=7 needs a payload width of 4"):
        activation_codec.decode(payload, scale, "int4", torch.bfloat16, 7)


@pytest.mark.parametrize("bad", [0, -3, True, 2.5])
def test_decode_refuses_a_hidden_that_is_not_a_positive_integer(bad):
    _x, payload, scale = _encoded("int8")
    with pytest.raises(ValueError, match="hidden must be a positive integer"):
        activation_codec.decode(payload, scale, "int8", torch.bfloat16, bad)


def test_decode_with_a_scale_of_the_wrong_shape_is_refused():
    _x, payload, scale = _encoded("int8")
    with pytest.raises(ValueError, match="scale shape"):
        activation_codec.decode(payload, scale[:, :2], "int8", torch.bfloat16, 10)


def test_store_compress_of_the_identity_has_no_scale_and_is_always_finite():
    x = torch.tensor([[1.0, float("nan")]]).to(torch.bfloat16)
    payload, scale, finite = activation_codec.store_compress(x, "bf16")
    assert payload is x and scale is None
    assert finite.dtype == torch.bool and finite.shape == () and bool(finite)


def test_store_compress_of_a_lossy_codec_flags_a_non_finite_row():
    x = torch.ones(2, 4).to(torch.bfloat16)
    x[1, 0] = float("inf")
    _payload, scale, finite = activation_codec.store_compress(x, "int8")
    assert scale is not None and not bool(finite)
    assert bool(activation_codec.store_compress(torch.ones(2, 4), "int8")[2])
