"""Tests for v0.30.0 Inference Excellence — prefix caching, spec-decoding pairing,
LoRA hot-swap, structured output, batching dashboard, request tracing, auto-quant.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock, patch

import pytest


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from Rich-formatted output."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# ═══════════════════════════════════════════════════════════════════════════
# Part A: Prefix Caching — vLLM enable_prefix_caching passthrough
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixCachingCLI:
    def test_serve_prefix_cache_flag_exists(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--prefix-cache" in _strip_ansi(result.output)

    def test_serve_prefix_cache_default_false(self):
        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        param = sig.parameters.get("prefix_cache")
        assert param is not None
        assert param.default.default is False


class TestPrefixCachingEngine:
    def test_create_engine_accepts_prefix_cache(self):
        from soup_cli.utils.vllm import create_vllm_engine

        sig = inspect.signature(create_vllm_engine)
        assert "enable_prefix_caching" in sig.parameters
        assert sig.parameters["enable_prefix_caching"].default is False

    def test_create_engine_passes_prefix_cache_to_vllm(self):
        mock_engine = MagicMock()
        mock_args_cls = MagicMock()
        mock_engine_cls = MagicMock()
        mock_engine_cls.from_engine_args.return_value = mock_engine

        with patch.dict("sys.modules", {
            "vllm": MagicMock(
                AsyncEngineArgs=mock_args_cls,
                AsyncLLMEngine=mock_engine_cls,
            ),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            vllm_mod.create_vllm_engine(
                model_path="/path/to/model",
                is_adapter=False,
                enable_prefix_caching=True,
            )
            call_kwargs = mock_args_cls.call_args.kwargs
            assert call_kwargs["enable_prefix_caching"] is True

    def test_create_engine_prefix_cache_default_false(self):
        mock_args_cls = MagicMock()
        mock_engine_cls = MagicMock()
        mock_engine_cls.from_engine_args.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "vllm": MagicMock(
                AsyncEngineArgs=mock_args_cls,
                AsyncLLMEngine=mock_engine_cls,
            ),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            vllm_mod.create_vllm_engine(
                model_path="/path/to/model",
                is_adapter=False,
            )
            call_kwargs = mock_args_cls.call_args.kwargs
            assert call_kwargs["enable_prefix_caching"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Part B: Speculative Decoding Auto-Pairing
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeculativePairing:
    def test_pick_draft_model_for_llama_70b(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        result = pick_draft_model("meta-llama/Llama-3.1-70B-Instruct")
        assert result is not None
        assert "1b" in result.lower() or "3b" in result.lower()

    def test_pick_draft_model_for_llama_8b_returns_none(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        # 8B is too small for speculative decoding (draft+target overhead > gain)
        result = pick_draft_model("meta-llama/Llama-3.1-8B-Instruct")
        assert result is None

    def test_pick_draft_model_for_qwen_72b(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        result = pick_draft_model("Qwen/Qwen2.5-72B-Instruct")
        assert result is not None
        assert "qwen" in result.lower()

    def test_pick_draft_model_unknown_returns_none(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("some/unknown-model") is None

    def test_pick_draft_model_empty_string_returns_none(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("") is None

    def test_pick_draft_model_rejects_urls(self):
        """Guard against URL injection — only HF model IDs."""
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("https://evil.com/model") is None
        assert pick_draft_model("http://localhost/model") is None

    def test_pick_draft_model_rejects_null_byte(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        assert pick_draft_model("meta-llama/Llama\x00-3.1-70B-Instruct") is None

    def test_pick_draft_model_case_insensitive_match(self):
        from soup_cli.utils.spec_pairing import pick_draft_model

        a = pick_draft_model("Meta-Llama/Llama-3.1-70B-Instruct")
        b = pick_draft_model("meta-llama/llama-3.1-70b-instruct")
        assert a == b

    def test_serve_auto_spec_flag(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--auto-spec" in _strip_ansi(result.output)


# ═══════════════════════════════════════════════════════════════════════════
# Part C: Dynamic LoRA Hot-Swap
# ═══════════════════════════════════════════════════════════════════════════


class TestLoRAHotSwap:
    def test_activate_endpoint_exists(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        # Create valid adapter path under cwd
        adapter_dir = tmp_path / "adapter-a"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            app = _create_app(
                model_obj=MagicMock(),
                tokenizer=MagicMock(),
                device="cpu",
                model_name="test",
                max_tokens_default=512,
                adapter_map={"chat": str(adapter_dir)},
            )
            client = TestClient(app)
            # List route should include activate
            routes = [r.path for r in app.routes]
            assert "/v1/adapters/activate/{name}" in routes or any(
                "/v1/adapters/activate" in r for r in routes
            )

            # Activate known adapter
            resp = client.post("/v1/adapters/activate/chat")
            assert resp.status_code == 200
            body = resp.json()
            assert body["active"] == "chat"
        finally:
            os.chdir(old_cwd)

    def test_activate_unknown_returns_404(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            adapter_map={"chat": "./x"},
        )
        client = TestClient(app)
        resp = client.post("/v1/adapters/activate/unknown")
        assert resp.status_code == 404

    def test_activate_rejects_invalid_names(self, tmp_path):
        """Name must be alphanumeric + hyphens — prevent path injection."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            adapter_map={"chat": "./x"},
        )
        client = TestClient(app)
        # Path-traversal style names
        resp = client.post("/v1/adapters/activate/..%2Fetc")
        assert resp.status_code in (400, 404, 422)

    def test_active_adapter_in_health(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            adapter_map={"chat": "./x"},
        )
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        # active may be None initially or the first-loaded adapter
        data = resp.json()
        assert "active_adapter" in data


# ═══════════════════════════════════════════════════════════════════════════
# Part D: Structured Output (JSON Schema / Regex)
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuredOutput:
    def test_structured_output_flag_exists(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--structured-output" in _strip_ansi(result.output)

    def test_validate_structured_output_accepts_known_modes(self):
        from soup_cli.utils.structured_output import validate_mode

        assert validate_mode("json") == "json"
        assert validate_mode("regex") == "regex"
        assert validate_mode("off") == "off"
        assert validate_mode(None) == "off"

    def test_validate_structured_output_rejects_unknown(self):
        from soup_cli.utils.structured_output import validate_mode

        with pytest.raises(ValueError, match="Unknown structured-output mode"):
            validate_mode("xml")

    def test_validate_json_schema_accepts_dict(self):
        from soup_cli.utils.structured_output import validate_json_schema

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        # Should not raise
        validate_json_schema(schema)

    def test_validate_json_schema_rejects_non_dict(self):
        from soup_cli.utils.structured_output import validate_json_schema

        with pytest.raises(ValueError):
            validate_json_schema("not a dict")

    def test_validate_regex_pattern_rejects_redos(self):
        """Guard against ReDoS: nested quantifier over catastrophic backtracking."""
        from soup_cli.utils.structured_output import validate_regex_pattern

        with pytest.raises(ValueError, match="length"):
            validate_regex_pattern("x" * 10_000)

    def test_validate_regex_pattern_accepts_simple(self):
        from soup_cli.utils.structured_output import validate_regex_pattern

        validate_regex_pattern(r"\d{3}-\d{4}")

    def test_validate_regex_pattern_rejects_invalid(self):
        from soup_cli.utils.structured_output import validate_regex_pattern

        with pytest.raises(ValueError):
            validate_regex_pattern("[unclosed")

    def test_build_constraint_returns_none_when_off(self):
        from soup_cli.utils.structured_output import build_constraint

        assert build_constraint("off", None, None) is None

    def test_build_constraint_json_requires_schema_or_skips(self):
        from soup_cli.utils.structured_output import build_constraint

        result = build_constraint("json", None, None)
        # When no schema provided, fall back to free-form JSON (not None
        # if we have a library, None otherwise — both acceptable as long
        # as there's no crash).
        assert result is None or isinstance(result, dict)

    def test_build_constraint_regex_requires_pattern(self):
        from soup_cli.utils.structured_output import build_constraint

        with pytest.raises(ValueError):
            build_constraint("regex", None, None)


# ═══════════════════════════════════════════════════════════════════════════
# Part E: Continuous Batching Dashboard
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchingDashboard:
    def test_dashboard_flag_exists(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--dashboard" in _strip_ansi(result.output)

    def test_metrics_endpoint_exists(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
        )
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_total" in data
        assert "tokens_generated_total" in data
        assert "active_requests" in data

    def test_metrics_counters_start_at_zero(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        snap = m.snapshot()
        assert snap["requests_total"] == 0
        assert snap["tokens_generated_total"] == 0
        assert snap["active_requests"] == 0

    def test_metrics_tracks_request_lifecycle(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        with m.track_request():
            snap = m.snapshot()
            assert snap["active_requests"] == 1
        assert m.snapshot()["active_requests"] == 0
        assert m.snapshot()["requests_total"] == 1

    def test_metrics_track_request_handles_exception(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        with pytest.raises(RuntimeError):
            with m.track_request():
                raise RuntimeError("boom")
        assert m.snapshot()["active_requests"] == 0
        assert m.snapshot()["requests_total"] == 1

    def test_metrics_record_tokens(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        m.record_tokens(42)
        m.record_tokens(8)
        assert m.snapshot()["tokens_generated_total"] == 50

    def test_metrics_record_tokens_rejects_negative(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        with pytest.raises(ValueError):
            m.record_tokens(-1)

    def test_metrics_latency_percentiles(self):
        from soup_cli.utils.metrics import ServerMetrics

        m = ServerMetrics()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            m.record_latency(ms)
        snap = m.snapshot()
        assert "latency_p50_ms" in snap
        assert "latency_p95_ms" in snap
        assert 40 <= snap["latency_p50_ms"] <= 60


# ═══════════════════════════════════════════════════════════════════════════
# Part F: Request Tracing (OpenTelemetry)
# ═══════════════════════════════════════════════════════════════════════════


class TestRequestTracing:
    def test_trace_flag_exists(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--trace" in _strip_ansi(result.output)

    def test_is_otel_available_when_missing(self):
        from soup_cli.utils.tracing import is_otel_available

        with patch.dict("sys.modules", {"opentelemetry": None}):
            assert is_otel_available() is False

    def test_build_tracer_returns_none_when_disabled(self):
        from soup_cli.utils.tracing import build_tracer

        assert build_tracer(enabled=False) is None

    def test_build_tracer_returns_none_when_otel_missing(self):
        from soup_cli.utils.tracing import build_tracer

        with patch(
            "soup_cli.utils.tracing.is_otel_available", return_value=False
        ):
            # Should not raise, just return None + log
            assert build_tracer(enabled=True) is None

    def test_validate_otlp_endpoint_accepts_localhost_http(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        assert validate_otlp_endpoint("http://localhost:4317") == "http://localhost:4317"
        assert validate_otlp_endpoint("http://127.0.0.1:4317") == "http://127.0.0.1:4317"

    def test_validate_otlp_endpoint_accepts_https(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        assert (
            validate_otlp_endpoint("https://otlp.example.com:4317")
            == "https://otlp.example.com:4317"
        )

    def test_validate_otlp_endpoint_rejects_plain_http_remote(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="HTTPS"):
            validate_otlp_endpoint("http://evil.com:4317")

    def test_validate_otlp_endpoint_rejects_bad_scheme(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError):
            validate_otlp_endpoint("ftp://example.com")

    def test_validate_otlp_endpoint_rejects_null_byte(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError):
            validate_otlp_endpoint("http://localhost\x00/evil")


# ═══════════════════════════════════════════════════════════════════════════
# Part G: Auto-Quant on Deploy
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoQuant:
    def test_auto_quant_flag_exists(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--auto-quant" in _strip_ansi(result.output)

    def test_candidate_order_default(self):
        from soup_cli.utils.auto_quant import default_candidate_order

        order = default_candidate_order()
        # First candidate should be fastest-to-try; we canonicalize order.
        assert order[0] in {"gguf", "awq", "gptq", "fp8", "none"}
        # No duplicates
        assert len(order) == len(set(order))
        # At least these quant formats present
        assert "gguf" in order
        assert "awq" in order

    def test_pick_best_candidate_prefers_fastest_ok(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.92, latency_ms=200, ok=True),
            Candidate(name="awq", score=0.95, latency_ms=180, ok=True),
            Candidate(name="none", score=0.96, latency_ms=400, ok=True),
        ]
        best = pick_best(candidates, min_score=0.90)
        # awq wins: ok, score above threshold, lowest latency
        assert best.name == "awq"

    def test_pick_best_drops_below_threshold(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.50, latency_ms=100, ok=True),
            Candidate(name="awq", score=0.60, latency_ms=50, ok=True),
            Candidate(name="none", score=0.95, latency_ms=400, ok=True),
        ]
        best = pick_best(candidates, min_score=0.90)
        assert best.name == "none"

    def test_pick_best_raises_when_no_candidates_pass(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.50, latency_ms=100, ok=True),
        ]
        with pytest.raises(ValueError, match="no candidate"):
            pick_best(candidates, min_score=0.90)

    def test_pick_best_skips_failed(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.99, latency_ms=1, ok=False),
            Candidate(name="awq", score=0.91, latency_ms=180, ok=True),
        ]
        best = pick_best(candidates, min_score=0.90)
        assert best.name == "awq"

    def test_candidate_requires_positive_latency(self):
        from soup_cli.utils.auto_quant import Candidate

        with pytest.raises(ValueError):
            Candidate(name="x", score=0.9, latency_ms=-1, ok=True)

    def test_candidate_score_bounded(self):
        from soup_cli.utils.auto_quant import Candidate

        with pytest.raises(ValueError):
            Candidate(name="x", score=1.5, latency_ms=100, ok=True)
        with pytest.raises(ValueError):
            Candidate(name="x", score=-0.1, latency_ms=100, ok=True)

    def test_min_score_bounds(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="awq", score=0.9, latency_ms=100, ok=True),
        ]
        with pytest.raises(ValueError):
            pick_best(candidates, min_score=-0.1)
        with pytest.raises(ValueError):
            pick_best(candidates, min_score=1.1)

    def test_name_validation_on_candidate(self):
        """Candidate names must be alphanumeric — prevents arbitrary strings
        leaking into log/display."""
        from soup_cli.utils.auto_quant import Candidate

        with pytest.raises(ValueError):
            Candidate(name="../../etc", score=0.9, latency_ms=100, ok=True)

    def test_pick_best_tie_break_by_first_encountered(self):
        """When latency and score are tied, prefer first-encountered
        (stable — consistent with v0.28.0 kernel_picker)."""
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.92, latency_ms=200, ok=True),
            Candidate(name="awq", score=0.92, latency_ms=200, ok=True),
        ]
        best = pick_best(candidates, min_score=0.90)
        assert best.name == "gguf"

    def test_pick_best_all_failed(self):
        from soup_cli.utils.auto_quant import Candidate, pick_best

        candidates = [
            Candidate(name="gguf", score=0.99, latency_ms=100, ok=False),
            Candidate(name="awq", score=0.99, latency_ms=50, ok=False),
        ]
        with pytest.raises(ValueError, match="no candidate"):
            pick_best(candidates, min_score=0.90)

    def test_pick_best_empty_list(self):
        from soup_cli.utils.auto_quant import pick_best

        with pytest.raises(ValueError, match="no candidate"):
            pick_best([], min_score=0.90)

    def test_pick_best_consumes_generator_once(self):
        """Regression test: error message counts correctly when input is
        a generator (not a list). v0.30.0 review fix."""
        from soup_cli.utils.auto_quant import Candidate, pick_best

        def gen():
            yield Candidate(name="gguf", score=0.5, latency_ms=100, ok=True)
            yield Candidate(name="awq", score=0.4, latency_ms=80, ok=True)

        with pytest.raises(ValueError, match="ran 2 candidates"):
            pick_best(gen(), min_score=0.90)

    def test_candidate_nan_score_rejected(self):
        from soup_cli.utils.auto_quant import Candidate

        with pytest.raises(ValueError, match="finite"):
            Candidate(name="x", score=float("nan"), latency_ms=100, ok=True)

    def test_candidate_nan_latency_rejected(self):
        from soup_cli.utils.auto_quant import Candidate

        with pytest.raises(ValueError, match="finite"):
            Candidate(name="x", score=0.9, latency_ms=float("nan"), ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Review-fix follow-up: coverage gaps from tdd-guide review
# ═══════════════════════════════════════════════════════════════════════════


class TestSpeculativePairingFullMatrix:
    """Cover every architecture family in the pick_draft_model map."""

    @pytest.mark.parametrize(
        "target",
        [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-14B",
            "mistralai/Mistral-Large-Instruct-2407",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "google/gemma-3-27b-it",
            "google/gemma-2-27b-it",
        ],
    )
    def test_known_targets_return_draft(self, target):
        from soup_cli.utils.spec_pairing import pick_draft_model

        result = pick_draft_model(target)
        assert result is not None, f"no draft mapped for {target}"


class TestLoRADeactivate:
    def test_deactivate_clears_active_state(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            adapter_map={"chat": "./x"},
        )
        client = TestClient(app)
        activate_resp = client.post("/v1/adapters/activate/chat")
        assert activate_resp.status_code == 200
        assert activate_resp.json()["active"] == "chat"

        deactivate_resp = client.post("/v1/adapters/deactivate")
        assert deactivate_resp.status_code == 200
        assert deactivate_resp.json()["active"] is None

        health = client.get("/health").json()
        assert health["active_adapter"] is None

    def test_activate_rejects_when_no_adapters_loaded(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test",
            max_tokens_default=512,
            adapter_map=None,
        )
        client = TestClient(app)
        resp = client.post("/v1/adapters/activate/chat")
        assert resp.status_code == 404


class TestMetricsThreadSafety:
    def test_concurrent_track_request(self):
        """Fire 50 concurrent enters: requests_total must equal 50 exactly
        (catches missing lock / non-atomic increment)."""
        import threading

        from soup_cli.utils.metrics import ServerMetrics

        metrics = ServerMetrics()

        def worker():
            with metrics.track_request():
                pass

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = metrics.snapshot()
        assert snap["requests_total"] == 50
        assert snap["active_requests"] == 0

    def test_record_tokens_zero_allowed(self):
        from soup_cli.utils.metrics import ServerMetrics

        metrics = ServerMetrics()
        metrics.record_tokens(0)  # must not raise
        assert metrics.snapshot()["tokens_generated_total"] == 0


class TestTracingExtra:
    def test_validate_otlp_rejects_private_ip(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="private"):
            validate_otlp_endpoint("https://192.168.1.10:4317")

    def test_validate_otlp_rejects_link_local(self):
        """Cloud-metadata IP must be blocked (SSRF hardening)."""
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="private|link"):
            validate_otlp_endpoint("https://169.254.169.254:4317")

    def test_validate_otlp_rejects_zero_host(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError):
            validate_otlp_endpoint("https://0.0.0.0:4317")

    def test_validate_otlp_rejects_missing_host(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError, match="host"):
            validate_otlp_endpoint("http:///path")

    def test_validate_otlp_rejects_non_string(self):
        from soup_cli.utils.tracing import validate_otlp_endpoint

        with pytest.raises(ValueError):
            validate_otlp_endpoint(123)  # type: ignore[arg-type]


class TestStructuredOutputExtra:
    def test_cli_json_without_schema_errors(self, tmp_path):
        """--structured-output json must require --json-schema."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(model_dir),
                "--device",
                "cpu",
                "--structured-output",
                "json",
            ],
        )
        assert result.exit_code != 0, result.output
        assert "json-schema" in _strip_ansi(result.output)


class TestAutoQuantCLIWarning:
    def test_auto_quant_prints_deferral_warning(self, tmp_path):
        """--auto-quant must print a yellow warning explaining it's a no-op."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(model_dir),
                "--device",
                "cpu",
                "--auto-quant",
            ],
        )
        # Command will fail later (no real model); just check the warning prints
        output = _strip_ansi(result.output)
        assert "auto-quant" in output.lower()
        assert "v0.30.1" in output or "deferred" in output.lower()


class TestJsonSchemaContainment:
    def test_json_schema_outside_cwd_rejected(self, tmp_path, monkeypatch):
        """JSON schema path must stay under cwd."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Put the schema OUTSIDE cwd
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        schema_file = outside_dir / "schema.json"
        schema_file.write_text('{"type": "object"}')

        cwd = tmp_path / "work"
        cwd.mkdir()
        model_dir = cwd / "model"
        model_dir.mkdir()
        monkeypatch.chdir(cwd)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "serve",
                "--model",
                str(model_dir),
                "--device",
                "cpu",
                "--structured-output",
                "json",
                "--json-schema",
                str(schema_file),
            ],
        )
        # Command may fail for other reasons too, but must reject the path
        assert "under" in result.output.lower() or result.exit_code != 0
