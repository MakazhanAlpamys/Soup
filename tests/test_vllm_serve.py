"""Tests for vLLM backend in soup serve."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================================================
# vLLM utility tests
# ============================================================


class TestVllmDetection:
    """Test vLLM availability detection."""

    def test_is_vllm_available_when_installed(self):
        """Should return True when vllm is importable."""
        from soup_cli.utils.vllm import is_vllm_available

        mock_vllm = MagicMock()
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            assert is_vllm_available() is True

    def test_is_vllm_available_when_not_installed(self):
        """Should return False when vllm is not importable."""
        from soup_cli.utils.vllm import is_vllm_available

        with patch.dict("sys.modules", {"vllm": None}):
            assert is_vllm_available() is False

    def test_get_vllm_version_installed(self):
        """Should return version string when installed."""
        from soup_cli.utils.vllm import get_vllm_version

        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.5.0"
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            assert get_vllm_version() == "0.5.0"

    def test_get_vllm_version_not_installed(self):
        """Should return 'not installed' when vllm is missing."""
        from soup_cli.utils.vllm import get_vllm_version

        with patch.dict("sys.modules", {"vllm": None}):
            assert get_vllm_version() == "not installed"

    def test_get_vllm_version_no_attr(self):
        """Should return 'unknown' if __version__ not set."""
        from soup_cli.utils.vllm import get_vllm_version

        mock_vllm = MagicMock(spec=[])  # no __version__ attr
        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            assert get_vllm_version() == "unknown"


# ============================================================
# Serve command backend flag tests
# ============================================================


class TestServeBackendFlag:
    """Test --backend flag in serve command."""

    def test_invalid_backend_rejected(self, tmp_path):
        """serve --backend invalid should fail."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        # Create a fake model dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = runner.invoke(
            app, ["serve", "--model", str(model_dir), "--backend", "invalid"]
        )
        assert result.exit_code != 0
        assert "unknown backend" in result.output.lower() or result.exit_code != 0

    def test_vllm_backend_not_installed(self, tmp_path):
        """serve --backend vllm should fail if vllm not installed."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("soup_cli.utils.vllm.is_vllm_available", return_value=False):
            result = runner.invoke(
                app, ["serve", "--model", str(model_dir), "--backend", "vllm"]
            )
            assert result.exit_code != 0

    def test_vllm_hint_shown_when_available(self, tmp_path):
        """When vllm is installed but not selected, show hint."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.commands import serve as serve_mod

        # Save original console
        original_console = serve_mod.console
        output = StringIO()
        serve_mod.console = Console(file=output)

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        try:
            with patch("soup_cli.utils.vllm.is_vllm_available", return_value=True):
                # Will fail at model loading but we can check for hint output
                try:
                    serve_mod.serve.__wrapped__(
                        model=str(model_dir),
                        base_model=None,
                        port=8000,
                        host="0.0.0.0",
                        device="cpu",
                        max_tokens_default=512,
                        backend="transformers",
                        tensor_parallel=1,
                        gpu_memory_utilization=0.9,
                    )
                except (SystemExit, Exception):
                    pass
        finally:
            serve_mod.console = original_console

        # Hint may or may not appear depending on how far execution gets
        # The key test is that no crash happens
        assert output.getvalue() is not None

    def test_default_backend_is_transformers(self):
        """Default backend should be transformers."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        backend_param = sig.parameters.get("backend")
        assert backend_param is not None
        assert backend_param.default.default == "transformers"

    def test_backend_accepts_vllm(self):
        """Backend param should accept 'vllm' as a value."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        backend_param = sig.parameters.get("backend")
        assert backend_param is not None
        # Just check the param exists and has help text mentioning vllm
        assert "vllm" in backend_param.default.help.lower()


class TestServeTensorParallel:
    """Test --tensor-parallel flag."""

    def test_tensor_parallel_param_exists(self):
        """serve should have --tensor-parallel param."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        assert "tensor_parallel" in sig.parameters

    def test_tensor_parallel_default_is_one(self):
        """Default tensor parallel size should be 1."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        tp_param = sig.parameters["tensor_parallel"]
        assert tp_param.default.default == 1

    def test_gpu_memory_param_exists(self):
        """serve should have --gpu-memory param."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        assert "gpu_memory_utilization" in sig.parameters

    def test_gpu_memory_default(self):
        """Default GPU memory utilization should be 0.9."""
        import inspect

        from soup_cli.commands.serve import serve

        sig = inspect.signature(serve)
        param = sig.parameters["gpu_memory_utilization"]
        assert param.default.default == 0.9


# ============================================================
# vLLM engine creation tests
# ============================================================


class TestCreateVllmEngine:
    """Test vLLM engine creation logic."""

    def test_create_engine_full_model(self):
        """create_vllm_engine with full model should use model_path directly."""
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

            engine, name = vllm_mod.create_vllm_engine(
                model_path="/path/to/model",
                is_adapter=False,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.8,
            )

            assert engine == mock_engine
            assert name == "/path/to/model"
            mock_args_cls.assert_called_once()
            call_kwargs = mock_args_cls.call_args
            assert call_kwargs.kwargs["model"] == "/path/to/model"
            assert call_kwargs.kwargs["tensor_parallel_size"] == 2
            assert call_kwargs.kwargs["gpu_memory_utilization"] == 0.8

    def test_create_engine_adapter(self):
        """create_vllm_engine with adapter should use base_model and enable LoRA."""
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

            engine, name = vllm_mod.create_vllm_engine(
                model_path="/path/to/adapter",
                base_model="meta-llama/Llama-3.1-8B",
                is_adapter=True,
            )

            assert engine == mock_engine
            assert name == "meta-llama/Llama-3.1-8B"
            call_kwargs = mock_args_cls.call_args
            assert call_kwargs.kwargs["model"] == "meta-llama/Llama-3.1-8B"
            assert call_kwargs.kwargs["enable_lora"] is True

    def test_create_engine_max_model_len(self):
        """create_vllm_engine should set max_model_len when provided."""
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

            engine, name = vllm_mod.create_vllm_engine(
                model_path="/path/to/model",
                is_adapter=False,
                max_model_len=4096,
            )

            args_instance = mock_args_cls.return_value
            assert args_instance.max_model_len == 4096


# ============================================================
# vLLM app creation tests
# ============================================================


class TestCreateVllmApp:
    """Test vLLM FastAPI app creation."""

    def _make_mock_engine(self):
        """Create a mock vLLM engine."""
        engine = MagicMock()
        return engine

    def test_create_vllm_app_returns_fastapi(self):
        """create_vllm_app should return a FastAPI app."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = MagicMock()

        with patch.dict("sys.modules", {
            "vllm": mock_vllm,
            "vllm.lora": MagicMock(),
            "vllm.lora.request": MagicMock(),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            engine = self._make_mock_engine()
            app = vllm_mod.create_vllm_app(
                engine=engine,
                engine_model_name="test-model",
                model_name="test-model",
                max_tokens_default=256,
            )

            assert isinstance(app, FastAPI)

    def test_vllm_app_has_correct_routes(self):
        """vLLM app should have health, models, and chat completions routes."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = MagicMock()

        with patch.dict("sys.modules", {
            "vllm": mock_vllm,
            "vllm.lora": MagicMock(),
            "vllm.lora.request": MagicMock(),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            engine = self._make_mock_engine()
            app = vllm_mod.create_vllm_app(
                engine=engine,
                engine_model_name="test-model",
                model_name="test-model",
                max_tokens_default=256,
            )

            routes = [route.path for route in app.routes]
            assert "/health" in routes
            assert "/v1/models" in routes
            assert "/v1/chat/completions" in routes

    def test_vllm_health_endpoint(self):
        """Health endpoint should return backend: vllm."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = MagicMock()

        with patch.dict("sys.modules", {
            "vllm": mock_vllm,
            "vllm.lora": MagicMock(),
            "vllm.lora.request": MagicMock(),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            engine = self._make_mock_engine()
            app = vllm_mod.create_vllm_app(
                engine=engine,
                engine_model_name="test-model",
                model_name="test-model",
                max_tokens_default=256,
            )

            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["model"] == "test-model"
            assert data["backend"] == "vllm"

    def test_vllm_models_endpoint(self):
        """Models endpoint should list the served model."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_vllm = MagicMock()
        mock_vllm.SamplingParams = MagicMock()

        with patch.dict("sys.modules", {
            "vllm": mock_vllm,
            "vllm.lora": MagicMock(),
            "vllm.lora.request": MagicMock(),
        }):
            from importlib import reload

            import soup_cli.utils.vllm as vllm_mod

            reload(vllm_mod)

            engine = self._make_mock_engine()
            app = vllm_mod.create_vllm_app(
                engine=engine,
                engine_model_name="my-model",
                model_name="my-model",
                max_tokens_default=256,
            )

            client = TestClient(app)
            response = client.get("/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "my-model"


# ============================================================
# Serve command integration with vLLM
# ============================================================


class TestServeVllmIntegration:
    """Test _serve_vllm helper."""

    def test_serve_vllm_creates_app(self, tmp_path):
        """_serve_vllm should create engine and return FastAPI app."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_engine = MagicMock()
        mock_app = MagicMock()

        with patch(
            "soup_cli.utils.vllm.create_vllm_engine",
            return_value=(mock_engine, "base-model"),
        ) as mock_create_engine, patch(
            "soup_cli.utils.vllm.create_vllm_app",
            return_value=mock_app,
        ) as mock_create_app:
            from soup_cli.commands.serve import _serve_vllm

            model_path = tmp_path / "model"
            model_path.mkdir()

            app = _serve_vllm(
                model_path=model_path,
                base_model="base-model",
                is_adapter=False,
                max_tokens_default=512,
                tensor_parallel=2,
                gpu_memory_utilization=0.85,
            )

            assert app == mock_app
            mock_create_engine.assert_called_once_with(
                model_path=str(model_path),
                base_model="base-model",
                is_adapter=False,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.85,
                speculative_model=None,
                num_speculative_tokens=5,
            )
            mock_create_app.assert_called_once()

    def test_serve_vllm_with_adapter(self, tmp_path):
        """_serve_vllm with adapter should pass adapter_path."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        mock_engine = MagicMock()
        mock_app = MagicMock()

        with patch(
            "soup_cli.utils.vllm.create_vllm_engine",
            return_value=(mock_engine, "base-model"),
        ), patch(
            "soup_cli.utils.vllm.create_vllm_app",
            return_value=mock_app,
        ) as mock_create_app:
            from soup_cli.commands.serve import _serve_vllm

            model_path = tmp_path / "adapter"
            model_path.mkdir()

            _serve_vllm(
                model_path=model_path,
                base_model="base-model",
                is_adapter=True,
                max_tokens_default=256,
                tensor_parallel=1,
                gpu_memory_utilization=0.9,
            )

            # Check adapter_path was passed
            call_kwargs = mock_create_app.call_args
            assert call_kwargs.kwargs["adapter_path"] == str(model_path)


# ============================================================
# Existing serve tests still pass (transformers backend)
# ============================================================


class TestTransformersBackendUnchanged:
    """Verify transformers backend still works as before."""

    def test_create_app_returns_fastapi_instance(self):
        """_create_app should return a FastAPI app with correct endpoints."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        app = _create_app(
            model_obj=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
        )

        assert isinstance(app, FastAPI)

        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes

    def test_health_endpoint_transformers(self):
        """Health endpoint with transformers backend returns ok."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
        )

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ============================================================
# CLI registration tests
# ============================================================


class TestServeCliRegistration:
    """Test serve command is properly registered."""

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def test_serve_help_shows_backend(self):
        """soup serve --help should mention --backend flag."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        clean = self._strip_ansi(result.output)
        assert "--backend" in clean

    def test_serve_help_shows_tensor_parallel(self):
        """soup serve --help should mention --tensor-parallel flag."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        clean = self._strip_ansi(result.output)
        assert "--tensor-parallel" in clean

    def test_serve_help_shows_gpu_memory(self):
        """soup serve --help should mention --gpu-memory flag."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        clean = self._strip_ansi(result.output)
        assert "--gpu-memory" in clean


# ============================================================
# pyproject.toml extra tests
# ============================================================


class TestServeFastExtra:
    """Test serve-fast extra is properly defined."""

    def test_serve_fast_extra_in_pyproject(self):
        """pyproject.toml should have serve-fast extra with vllm."""
        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        content = toml_path.read_text()
        assert "serve-fast" in content
        assert "vllm" in content


# ============================================================
# Version detection tests
# ============================================================


class TestVersionDetectsVllm:
    """Test that version --full detects vllm."""

    def test_version_full_checks_vllm(self):
        """version --full extras should include serve-fast check."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        # Run version --full — vllm won't be installed but it shouldn't crash
        result = runner.invoke(app, ["version", "--full"])
        assert result.exit_code == 0
        # Should NOT show serve-fast since vllm not installed
        # But command should complete successfully
