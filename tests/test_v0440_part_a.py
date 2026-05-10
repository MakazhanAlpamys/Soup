"""v0.44.0 Part A — Live monitoring utility tests.

Covers: tail_latency, gpu_monitor, sse_train_stream, qr_url,
llama_server_timings, tool_outputs.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from soup_cli.utils.gpu_monitor import (
    GpuSample,
    parse_nvidia_smi_csv,
)
from soup_cli.utils.llama_server_timings import (
    LlamaServerTimings,
    format_kv_bar,
    parse_timings,
)
from soup_cli.utils.qr_url import (
    build_phone_url,
    render_qr_ascii,
    validate_token,
)
from soup_cli.utils.sse_train_stream import (
    TrainEvent,
    format_sse_frame,
    to_payload,
)
from soup_cli.utils.tail_latency import (
    TailLatencySummary,
    percentile,
    summarise_latency,
    update_ema,
)
from soup_cli.utils.tool_outputs import (
    ToolCallTimer,
    ToolOutputsBuffer,
)

# --- tail_latency -----------------------------------------------------------

def test_update_ema_initialises_with_first_sample():
    assert update_ema(None, 5.0, 0.1) == 5.0


def test_update_ema_rejects_bool():
    with pytest.raises(ValueError):
        update_ema(None, True, 0.1)  # type: ignore[arg-type]


def test_update_ema_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        update_ema(None, 1.0, 0.0)
    with pytest.raises(ValueError):
        update_ema(None, 1.0, 1.5)
    with pytest.raises(ValueError):
        update_ema(None, 1.0, True)  # type: ignore[arg-type]


def test_update_ema_blends():
    assert update_ema(10.0, 20.0, 0.5) == pytest.approx(15.0)


def test_update_ema_rejects_non_finite_sample():
    with pytest.raises(ValueError):
        update_ema(None, float("inf"), 0.1)


def test_percentile_empty_returns_none():
    assert percentile([], 50) is None


def test_percentile_pct_bounds():
    with pytest.raises(ValueError):
        percentile([1.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0], 101)
    with pytest.raises(ValueError):
        percentile([1.0], True)  # type: ignore[arg-type]


def test_percentile_basic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(samples, 0) == 1.0
    assert percentile(samples, 50) == 3.0
    assert percentile(samples, 100) == 5.0


def test_percentile_interpolated():
    # 4 samples, p25 -> rank=0.75 -> interpolation between idx 0 and 1.
    assert percentile([0.0, 4.0, 8.0, 12.0], 25) == pytest.approx(3.0)


def test_summarise_latency_empty():
    summary = summarise_latency([])
    assert summary == TailLatencySummary(0, None, None, None, None, None)


def test_summarise_latency_basic():
    summary = summarise_latency([10.0, 20.0, 30.0, 40.0, 50.0])
    assert summary.count == 5
    assert summary.mean == pytest.approx(30.0)
    assert summary.p50 == pytest.approx(30.0)
    assert summary.ema is not None and summary.ema > 0


# --- gpu_monitor ------------------------------------------------------------

def test_parse_nvidia_smi_csv_happy_path():
    text = (
        "0, NVIDIA RTX 4090, 87, 12, 8192, 24576, 65, 320.5\n"
        "1, NVIDIA H100, 50, 5, 16384, 81920, 55, 410.0\n"
    )
    samples = parse_nvidia_smi_csv(text)
    assert len(samples) == 2
    assert samples[0] == GpuSample(
        index=0,
        name="NVIDIA RTX 4090",
        util_gpu_pct=87.0,
        util_mem_pct=12.0,
        mem_used_mb=8192.0,
        mem_total_mb=24576.0,
        temp_c=65.0,
        power_w=320.5,
    )


def test_parse_nvidia_smi_csv_handles_na_fields():
    text = "0, GPU0, [N/A], 0, 1024, 2048, 50, [Not Supported]\n"
    samples = parse_nvidia_smi_csv(text)
    assert samples[0].util_gpu_pct is None
    assert samples[0].power_w is None


def test_parse_nvidia_smi_csv_skips_malformed_rows():
    text = (
        "garbage,not enough cols\n"
        "0, GPU0, 50, 5, 1024, 2048, 50, 100\n"
        "abc, GPU?, 50, 5, 1024, 2048, 50, 100\n"
    )
    samples = parse_nvidia_smi_csv(text)
    assert len(samples) == 1


def test_parse_nvidia_smi_csv_rejects_null_byte_name():
    text = "0, evil\x00name, 50, 5, 1024, 2048, 50, 100\n"
    samples = parse_nvidia_smi_csv(text)
    assert samples == []


def test_parse_nvidia_smi_csv_type_check():
    with pytest.raises(TypeError):
        parse_nvidia_smi_csv(b"bytes")  # type: ignore[arg-type]


# --- sse_train_stream -------------------------------------------------------

def test_train_event_default_ts_is_now():
    event = TrainEvent(type="metric")
    assert abs(event.ts - time.time()) < 5


def test_train_event_invalid_type():
    with pytest.raises(ValueError):
        TrainEvent(type="bogus")


def test_train_event_invalid_message():
    with pytest.raises(ValueError):
        TrainEvent(type="log", message="\x00bad")


def test_train_event_oversize_message():
    with pytest.raises(ValueError):
        TrainEvent(type="log", message="x" * 5000)


def test_to_payload_omits_none():
    event = TrainEvent(type="metric", ts=1.0, step=10, loss=0.5)
    payload = to_payload(event)
    assert payload == {"type": "metric", "ts": 1.0, "step": 10, "loss": 0.5}


def test_format_sse_frame_shape():
    event = TrainEvent(type="status", ts=1.0, message="ok")
    frame = format_sse_frame(event)
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    body = frame[len("data: "):].strip()
    assert json.loads(body) == {"type": "status", "ts": 1.0, "message": "ok"}


# --- qr_url ------------------------------------------------------------------

def test_validate_token_happy():
    validate_token("aBcDeFgHiJkLmNoP")  # 16 chars


def test_validate_token_too_short():
    with pytest.raises(ValueError):
        validate_token("short")


def test_validate_token_invalid_chars():
    with pytest.raises(ValueError):
        validate_token("a" * 16 + "!")


def test_build_phone_url_loopback_http_ok():
    url = build_phone_url(
        scheme="http",
        host="127.0.0.1",
        port=8000,
        token="x" * 32,
    )
    assert "127.0.0.1:8000" in url
    assert "?token=" in url


def test_build_phone_url_lan_http_rejected():
    with pytest.raises(ValueError, match="loopback"):
        build_phone_url(
            scheme="http",
            host="192.168.1.10",
            port=8000,
            token="x" * 32,
        )


def test_build_phone_url_https_lan_ok():
    url = build_phone_url(
        scheme="https",
        host="my.lan.host",
        port=443,
        token="x" * 32,
    )
    assert url.startswith("https://my.lan.host:443/")


def test_build_phone_url_invalid_port():
    with pytest.raises(ValueError):
        build_phone_url(
            scheme="https", host="x", port=0, token="x" * 32
        )
    with pytest.raises(ValueError):
        build_phone_url(
            scheme="https", host="x", port=True, token="x" * 32
        )  # type: ignore[arg-type]


def test_build_phone_url_invalid_scheme():
    with pytest.raises(ValueError):
        build_phone_url(
            scheme="ftp", host="x", port=80, token="x" * 32
        )


def test_render_qr_ascii_returns_none_or_string():
    result = render_qr_ascii("https://example.com")
    # qrcode might not be installed — both outcomes are valid.
    assert result is None or isinstance(result, str)


def test_render_qr_ascii_rejects_empty():
    with pytest.raises(ValueError):
        render_qr_ascii("")


# --- llama_server_timings ---------------------------------------------------

def test_parse_timings_happy_path():
    payload = {
        "timings": {
            "prompt_n": 100,
            "prompt_ms": 1000.0,
            "prompt_per_token_ms": 10.0,
            "predicted_n": 50,
            "predicted_ms": 2000.0,
            "predicted_per_token_ms": 40.0,
        },
        "kv_cache_used": 1024,
        "kv_cache_size": 4096,
    }
    timings = parse_timings(payload)
    assert timings.prompt_tokens == 100
    assert timings.kv_cache_pct == pytest.approx(25.0)


def test_parse_timings_missing_fields():
    timings = parse_timings({})
    assert timings == LlamaServerTimings(
        None, None, None, None, None, None, None, None, None
    )


def test_parse_timings_rejects_non_dict():
    with pytest.raises(TypeError):
        parse_timings("not a dict")  # type: ignore[arg-type]


def test_parse_timings_clamps_pct():
    payload = {
        "kv_cache_used": 999_999_999,
        "kv_cache_size": 1024,
    }
    timings = parse_timings(payload)
    assert timings.kv_cache_pct == 100.0


def test_format_kv_bar_renders():
    bar = format_kv_bar(50.0, width=10)
    assert bar.endswith("50.0%")
    assert "█" in bar


def test_format_kv_bar_none():
    bar = format_kv_bar(None, width=4)
    assert "--%" in bar


def test_format_kv_bar_invalid_width():
    with pytest.raises(ValueError):
        format_kv_bar(50.0, width=0)
    with pytest.raises(TypeError):
        format_kv_bar(50.0, width=True)  # type: ignore[arg-type]


# --- tool_outputs -----------------------------------------------------------

def test_tool_outputs_buffer_records():
    buffer = ToolOutputsBuffer()
    buffer.record_call(
        name="fetch_url",
        started_ts=1000.0,
        duration_ms=42.5,
        success=True,
        output_preview="ok",
    )
    snap = buffer.snapshot()
    assert len(snap) == 1
    assert snap[0].name == "fetch_url"
    assert snap[0].duration_ms == pytest.approx(42.5)


def test_tool_outputs_buffer_truncates_long_output():
    buffer = ToolOutputsBuffer()
    buffer.record_call(
        name="x",
        started_ts=1.0,
        duration_ms=1.0,
        success=True,
        output_preview="x" * 99999,
    )
    assert len(buffer.snapshot()[0].output_preview) <= 4096


def test_tool_outputs_buffer_rejects_invalid_name():
    buffer = ToolOutputsBuffer()
    with pytest.raises(ValueError):
        buffer.record_call(
            name="bad\x00name",
            started_ts=1.0,
            duration_ms=1.0,
            success=True,
            output_preview="",
        )


def test_tool_outputs_buffer_rejects_bool_started_ts():
    buffer = ToolOutputsBuffer()
    with pytest.raises(TypeError):
        buffer.record_call(
            name="x",
            started_ts=True,  # type: ignore[arg-type]
            duration_ms=1.0,
            success=True,
            output_preview="",
        )


def test_tool_outputs_buffer_rejects_negative_duration():
    buffer = ToolOutputsBuffer()
    with pytest.raises(ValueError):
        buffer.record_call(
            name="x",
            started_ts=1.0,
            duration_ms=-1.0,
            success=True,
            output_preview="",
        )


def test_tool_outputs_buffer_rejects_nonbool_success():
    buffer = ToolOutputsBuffer()
    with pytest.raises(TypeError):
        buffer.record_call(
            name="x",
            started_ts=1.0,
            duration_ms=1.0,
            success=1,  # type: ignore[arg-type]
            output_preview="",
        )


def test_tool_outputs_snapshot_limit():
    buffer = ToolOutputsBuffer()
    for idx in range(5):
        buffer.record_call(
            name=f"t{idx}",
            started_ts=float(idx),
            duration_ms=1.0,
            success=True,
            output_preview="",
        )
    assert len(buffer.snapshot(limit=2)) == 2
    assert buffer.snapshot(limit=2)[-1].name == "t4"
    with pytest.raises(ValueError):
        buffer.snapshot(limit=-1)


def test_tool_call_timer_records_success():
    buffer = ToolOutputsBuffer()
    with ToolCallTimer(buffer, name="my_tool") as timer:
        timer.set_output("result")
    snap = buffer.snapshot()
    assert len(snap) == 1
    assert snap[0].success is True
    assert snap[0].output_preview == "result"


def test_tool_call_timer_records_exception():
    buffer = ToolOutputsBuffer()
    with pytest.raises(RuntimeError):
        with ToolCallTimer(buffer, name="bad"):
            raise RuntimeError("boom")
    snap = buffer.snapshot()
    assert snap[0].success is False
    assert snap[0].error is not None
    assert "boom" in snap[0].error


def test_tool_outputs_buffer_clear():
    buffer = ToolOutputsBuffer()
    buffer.record_call(
        name="x",
        started_ts=1.0,
        duration_ms=1.0,
        success=True,
        output_preview="",
    )
    buffer.clear()
    assert buffer.snapshot() == []


def test_tail_latency_summary_frozen():
    summary = TailLatencySummary(0, None, None, None, None, None)
    with pytest.raises(Exception):
        summary.count = 99  # type: ignore[misc]


def test_train_event_rejects_non_finite_ts():
    with pytest.raises(ValueError):
        TrainEvent(type="metric", ts=float("nan"))


def test_train_event_rejects_bool_ts():
    with pytest.raises(ValueError):
        TrainEvent(type="metric", ts=True)  # type: ignore[arg-type]


def test_summarise_latency_rejects_non_finite():
    with pytest.raises(ValueError):
        summarise_latency([float("nan")])


def test_percentile_rejects_non_finite():
    with pytest.raises(ValueError):
        percentile([1.0, math.inf], 50)
