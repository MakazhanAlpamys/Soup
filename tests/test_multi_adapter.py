"""Tests for multi-adapter serving — Part B of v0.22.0."""

from unittest.mock import MagicMock

import pytest


class TestAdapterValidation:
    """Test adapter name and path validation."""

    def test_valid_adapter_name(self):
        from soup_cli.commands.serve import _validate_adapter_name

        assert _validate_adapter_name("my-adapter") is True
        assert _validate_adapter_name("chat") is True
        assert _validate_adapter_name("code-v2") is True
        assert _validate_adapter_name("model123") is True

    def test_invalid_adapter_name_path_separator(self):
        from soup_cli.commands.serve import _validate_adapter_name

        assert _validate_adapter_name("../evil") is False
        assert _validate_adapter_name("path/to") is False
        assert _validate_adapter_name("path\\to") is False

    def test_invalid_adapter_name_special_chars(self):
        from soup_cli.commands.serve import _validate_adapter_name

        assert _validate_adapter_name("") is False
        assert _validate_adapter_name("name with spaces") is False
        assert _validate_adapter_name("name\x00null") is False

    def test_adapter_path_traversal_protection(self, tmp_path):
        from soup_cli.commands.serve import _validate_adapter_path

        # Valid path under cwd
        adapter_dir = tmp_path / "adapters" / "chat"
        adapter_dir.mkdir(parents=True)
        assert _validate_adapter_path(str(adapter_dir), cwd=str(tmp_path)) is True

        # Path traversal attempt
        assert _validate_adapter_path("../../etc/passwd", cwd=str(tmp_path)) is False

    def test_adapter_path_must_exist(self, tmp_path):
        from soup_cli.commands.serve import _validate_adapter_path

        assert _validate_adapter_path(
            str(tmp_path / "nonexistent"), cwd=str(tmp_path)
        ) is False


class TestParseAdapters:
    """Test --adapters flag parsing."""

    def test_parse_single_adapter(self):
        from soup_cli.commands.serve import _parse_adapters

        result = _parse_adapters(["chat=./adapters/chat"])
        assert result == {"chat": "./adapters/chat"}

    def test_parse_multiple_adapters(self):
        from soup_cli.commands.serve import _parse_adapters

        result = _parse_adapters([
            "chat=./adapters/chat",
            "code=./adapters/code",
            "medical=./adapters/med",
        ])
        assert len(result) == 3
        assert result["chat"] == "./adapters/chat"
        assert result["code"] == "./adapters/code"
        assert result["medical"] == "./adapters/med"

    def test_parse_invalid_format(self):
        from soup_cli.commands.serve import _parse_adapters

        with pytest.raises(ValueError, match="key=path"):
            _parse_adapters(["invalid_no_equals"])

    def test_parse_empty_list(self):
        from soup_cli.commands.serve import _parse_adapters

        result = _parse_adapters([])
        assert result == {}

    def test_parse_none(self):
        from soup_cli.commands.serve import _parse_adapters

        result = _parse_adapters(None)
        assert result == {}


class TestMultiAdapterApp:
    """Test multi-adapter FastAPI app."""

    @pytest.fixture
    def multi_adapter_app(self):
        """Create a FastAPI app with multiple mock adapters."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        adapter_map = {
            "chat": "/fake/path/chat",
            "code": "/fake/path/code",
        }

        app = _create_app(
            model_obj=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
            adapter_map=adapter_map,
        )
        return app

    def test_adapters_endpoint_exists(self, multi_adapter_app):
        """GET /v1/adapters should exist."""
        routes = [route.path for route in multi_adapter_app.routes]
        assert "/v1/adapters" in routes

    def test_adapters_endpoint_lists_adapters(self, multi_adapter_app):
        """GET /v1/adapters returns loaded adapters."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        client = TestClient(multi_adapter_app)
        response = client.get("/v1/adapters")
        assert response.status_code == 200
        data = response.json()
        assert "adapters" in data
        names = [adapter["name"] for adapter in data["adapters"]]
        assert "chat" in names
        assert "code" in names
        # Security: paths should NOT be exposed
        for adapter in data["adapters"]:
            assert "path" not in adapter

    def test_no_adapters_returns_empty(self):
        """GET /v1/adapters with no adapters returns empty list."""
        try:
            from fastapi.testclient import TestClient
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
        client = TestClient(app)
        response = client.get("/v1/adapters")
        assert response.status_code == 200
        data = response.json()
        assert data["adapters"] == []

    def test_unknown_adapter_returns_404(self, multi_adapter_app):
        """Request with unknown adapter name returns 404."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        client = TestClient(multi_adapter_app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "adapter": "nonexistent",
            },
        )
        assert response.status_code == 404

    def test_app_without_adapters_still_works(self):
        """App with no adapters should work normally."""
        try:
            import fastapi  # noqa: F401
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
        routes = [route.path for route in app.routes]
        assert "/v1/chat/completions" in routes
        assert "/v1/adapters" in routes


class TestAdapterPathAbsolute:
    """Test adapter path validation with absolute paths."""

    def test_adapter_path_absolute_outside_cwd(self, tmp_path):
        """Absolute path to real directory outside cwd should be rejected."""
        import tempfile

        from soup_cli.commands.serve import _validate_adapter_path

        with tempfile.TemporaryDirectory() as external_dir:
            assert _validate_adapter_path(external_dir, cwd=str(tmp_path)) is False

    def test_adapters_endpoint_schema_strict(self):
        """Adapter objects should only contain 'name' key — no path leakage."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            import pytest
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
            adapter_map={"chat": "/fake/path"},
        )
        client = TestClient(app)
        response = client.get("/v1/adapters")
        for adapter in response.json()["adapters"]:
            assert set(adapter.keys()) == {"name"}


class TestAdaptersBackendRejection:
    """Test --adapters rejected for non-transformers backends."""

    def test_adapters_with_vllm_backend_rejected(self, tmp_path):
        """--adapters with --backend vllm should fail."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        result = runner.invoke(app, [
            "serve", "--model", str(model_dir),
            "--backend", "vllm",
            "--adapters", f"chat={adapter_dir}",
        ])
        assert result.exit_code != 0
        assert "transformers" in result.output.lower() or "not" in result.output.lower()


class TestMultiAdapterCLI:
    """Test multi-adapter CLI flags."""

    def test_adapters_flag_invalid_format(self, tmp_path):
        """--adapters with invalid format shows error."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        result = runner.invoke(app, [
            "serve", "--model", str(model_dir),
            "--adapters", "invalid_no_equals",
        ])
        assert result.exit_code != 0

    def test_adapters_flag_bad_name(self, tmp_path):
        """--adapters with invalid adapter name shows error."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        result = runner.invoke(app, [
            "serve", "--model", str(model_dir),
            "--adapters", f"../evil={adapter_dir}",
        ])
        assert result.exit_code != 0
