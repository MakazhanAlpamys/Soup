"""Tests for SGLang backend — detection, runtime creation, serve --backend flag, FastAPI app."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest


def _has_fastapi():
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


# ─── SGLang Detection Tests ──────────────────────────────────────────────


class TestSGLangDetection:
    """Test SGLang availability detection."""

    def test_check_sglang_available_when_installed(self):
        """check_sglang_available should return True when sglang is importable."""
        mock_sglang = MagicMock()
        with mock_patch.dict("sys.modules", {"sglang": mock_sglang}):
            from soup_cli.utils.sglang import check_sglang_available

            assert check_sglang_available() is True

    def test_check_sglang_available_when_not_installed(self):
        """check_sglang_available should return False when sglang import fails."""
        from soup_cli.utils.sglang import check_sglang_available

        # Just verify the function exists and is callable
        assert callable(check_sglang_available)

    def test_get_sglang_version_when_installed(self):
        """get_sglang_version should return version string."""
        mock_sglang = MagicMock()
        mock_sglang.__version__ = "0.3.0"
        with mock_patch.dict("sys.modules", {"sglang": mock_sglang}):
            from soup_cli.utils.sglang import get_sglang_version

            version = get_sglang_version()
            assert version == "0.3.0"

    def test_get_sglang_version_when_not_installed(self):
        """get_sglang_version should return 'not installed' when unavailable."""
        from soup_cli.utils.sglang import get_sglang_version

        # If sglang is not actually installed, should return "not installed"
        try:
            import sglang  # noqa: F401
        except ImportError:
            assert get_sglang_version() == "not installed"


# ─── SGLang Runtime Creation Tests ───────────────────────────────────────


class TestSGLangRuntimeCreation:
    """Test SGLang runtime creation."""

    def test_create_sglang_runtime_full_model(self):
        """create_sglang_runtime should create runtime for a full model."""
        mock_runtime = MagicMock()
        mock_sgl = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            runtime, model_name = create_sglang_runtime(
                model_path="/path/to/model",
            )

        mock_sgl.Runtime.assert_called_once()
        call_kwargs = mock_sgl.Runtime.call_args[1]
        assert call_kwargs["model_path"] == "/path/to/model"
        # Default-deny, matching the vLLM path. This asserted True until the
        # SGLang trust-contract fix: the runtime hardcoded trust_remote_code=True
        # at both call sites, so `soup serve --backend sglang` executed a
        # model's custom code whether or not the user passed
        # --trust-remote-code. See tests/test_sglang_trust_contract.py for the
        # opt-in path and the serve.py wiring.
        assert call_kwargs["trust_remote_code"] is False
        assert model_name == "/path/to/model"

    def test_create_sglang_runtime_with_adapter(self):
        """create_sglang_runtime should pass lora_paths for adapter."""
        mock_runtime = MagicMock()
        mock_sgl = MagicMock()
        mock_sgl.Runtime.return_value = mock_runtime

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            runtime, model_name = create_sglang_runtime(
                model_path="/path/to/adapter",
                base_model="meta-llama/Llama-3.1-8B",
                is_adapter=True,
            )

        call_kwargs = mock_sgl.Runtime.call_args[1]
        assert call_kwargs["model_path"] == "meta-llama/Llama-3.1-8B"
        assert call_kwargs["lora_paths"] == ["/path/to/adapter"]
        assert model_name == "meta-llama/Llama-3.1-8B"

    def test_create_sglang_runtime_tensor_parallel(self):
        """create_sglang_runtime should pass tp_size."""
        mock_sgl = MagicMock()
        mock_sgl.Runtime.return_value = MagicMock()

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            create_sglang_runtime(
                model_path="/model",
                tensor_parallel_size=4,
            )

        call_kwargs = mock_sgl.Runtime.call_args[1]
        assert call_kwargs["tp_size"] == 4


# ─── SGLang FastAPI App Tests ────────────────────────────────────────────


@pytest.mark.skipif(
    not _has_fastapi(),
    reason="fastapi not installed",
)
class TestSGLangApp:
    """Test SGLang FastAPI app creation (requires fastapi)."""

    def test_create_sglang_app_returns_fastapi(self):
        """create_sglang_app should return a FastAPI application."""
        mock_runtime = MagicMock()

        from soup_cli.utils.sglang import create_sglang_app

        app = create_sglang_app(
            runtime=mock_runtime,
            runtime_model_name="test-model",
            model_name="test",
        )

        assert app is not None

    def test_sglang_app_has_health_endpoint(self):
        """SGLang app should have /health endpoint."""
        mock_runtime = MagicMock()

        from soup_cli.utils.sglang import create_sglang_app

        app = create_sglang_app(
            runtime=mock_runtime,
            runtime_model_name="test-model",
            model_name="test",
        )

        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_sglang_app_has_models_endpoint(self):
        """SGLang app should have /v1/models endpoint."""
        mock_runtime = MagicMock()

        from soup_cli.utils.sglang import create_sglang_app

        app = create_sglang_app(
            runtime=mock_runtime,
            runtime_model_name="test-model",
            model_name="test",
        )

        routes = [route.path for route in app.routes]
        assert "/v1/models" in routes

    def test_sglang_app_has_chat_completions_endpoint(self):
        """SGLang app should have /v1/chat/completions endpoint."""
        mock_runtime = MagicMock()

        from soup_cli.utils.sglang import create_sglang_app

        app = create_sglang_app(
            runtime=mock_runtime,
            runtime_model_name="test-model",
            model_name="test",
        )

        routes = [route.path for route in app.routes]
        assert "/v1/chat/completions" in routes


# ─── Serve Command SGLang Tests ──────────────────────────────────────────


class TestServeSGLangCommand:
    """Test serve command with --backend sglang."""

    def test_serve_sglang_backend_in_help(self):
        """soup serve --help should mention sglang backend."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert "sglang" in result.output

    def test_serve_sglang_not_installed_shows_error(self, tmp_path):
        """serve --backend sglang should show install hint when sglang missing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Create a dummy model path
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        with mock_patch(
            "soup_cli.utils.sglang.check_sglang_available", return_value=False
        ):
            result = runner.invoke(app, [
                "serve",
                "--model", str(model_dir),
                "--backend", "sglang",
            ])
        assert result.exit_code != 0

    @pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
    def test_serve_unknown_backend_shows_error(self, tmp_path):
        """serve --backend invalid should show error listing valid backends."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(app, [
            "serve",
            "--model", str(model_dir),
            "--backend", "invalid_backend",
        ])
        assert result.exit_code != 0
        assert "sglang" in result.output  # Should list valid backends


# ─── Doctor SGLang Detection Tests ───────────────────────────────────────


class TestDoctorSGLang:
    """Test that soup doctor checks for SGLang."""

    def test_doctor_includes_sglang(self):
        """DEPS list should include sglang."""
        from soup_cli.commands.doctor import DEPS

        pkg_names = [dep[1] for dep in DEPS]
        assert "sglang" in pkg_names

    def test_doctor_includes_librosa(self):
        """DEPS list should include librosa."""
        from soup_cli.commands.doctor import DEPS

        pkg_names = [dep[1] for dep in DEPS]
        assert "librosa" in pkg_names


# ─── SGLang SSRF Validation Tests ────────────────────────────────────────


class TestSGLangSSRF:
    """Test SSRF protection in SGLang runtime creation."""

    def test_create_runtime_blocks_http_model_path(self):
        """HTTP URLs for model_path should be rejected."""
        mock_sgl = MagicMock()

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            with pytest.raises(ValueError, match="not a URL"):
                create_sglang_runtime(
                    model_path="http://evil.com/model",
                )

    def test_create_runtime_blocks_http_base_model(self):
        """HTTP URLs for base_model should be rejected."""
        mock_sgl = MagicMock()

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            with pytest.raises(ValueError, match="not a URL"):
                create_sglang_runtime(
                    model_path="/path/to/adapter",
                    base_model="https://evil.com/model",
                    is_adapter=True,
                )

    def test_create_runtime_allows_hf_model_id(self):
        """HuggingFace model IDs should be allowed."""
        mock_sgl = MagicMock()
        mock_sgl.Runtime.return_value = MagicMock()

        with mock_patch.dict("sys.modules", {"sglang": mock_sgl}):
            from soup_cli.utils.sglang import create_sglang_runtime

            runtime, name = create_sglang_runtime(
                model_path="meta-llama/Llama-3.1-8B",
            )
        assert name == "meta-llama/Llama-3.1-8B"
