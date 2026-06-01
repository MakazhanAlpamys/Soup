"""Tests for soup serve — inference server command."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestServeValidation:
    """Test serve command argument validation."""

    def test_model_path_not_found(self):
        """serve should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--model", "/nonexistent/path"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "not installed" in result.output.lower()

    def test_fastapi_import_error(self):
        """serve should fail gracefully if FastAPI not installed."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        with patch.dict("sys.modules", {"fastapi": None}):
            result = runner.invoke(app, ["serve", "--model", "."])
            # Either import error or model not found
            assert result.exit_code != 0


class TestCreateApp:
    """Test the FastAPI app creation."""

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

        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/v1/models" in routes
        assert "/v1/chat/completions" in routes

    def test_health_endpoint(self):
        """Health endpoint should return ok status."""
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
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "test-model"
        assert data["device"] == "cpu"

    def test_models_endpoint(self):
        """Models endpoint should list the loaded model."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
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

    def test_chat_completions_endpoint(self):
        """Chat completions should return OpenAI-compatible response."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        with patch("soup_cli.commands.serve._generate_response") as mock_gen:
            mock_gen.return_value = ("Hello there!", 10, 5)

            app = _create_app(
                model_obj=MagicMock(),
                tokenizer=MagicMock(),
                device="cpu",
                model_name="test-model",
                max_tokens_default=256,
            )

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello there!"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["usage"]["prompt_tokens"] == 10
            assert data["usage"]["completion_tokens"] == 5

    def test_chat_completions_streaming(self):
        """Streaming should return SSE events."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.commands.serve import _create_app

        with patch("soup_cli.commands.serve._generate_response") as mock_gen:
            mock_gen.return_value = ("Hello world", 10, 5)

            app = _create_app(
                model_obj=MagicMock(),
                tokenizer=MagicMock(),
                device="cpu",
                model_name="test-model",
                max_tokens_default=256,
            )

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

            assert response.status_code == 200
            text = response.text
            assert "data:" in text
            assert "[DONE]" in text


class TestStreamResponse:
    """Test SSE streaming."""

    def test_stream_response_yields_chunks(self):
        """_stream_response should yield SSE events."""
        from soup_cli.commands.serve import _stream_response

        with patch("soup_cli.commands.serve._generate_response") as mock_gen:
            mock_gen.return_value = ("Hello world", 5, 2)

            chunks = list(_stream_response(
                model=MagicMock(),
                tokenizer=MagicMock(),
                messages=[{"role": "user", "content": "test"}],
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                model_name="test",
            ))

            # Should have word chunks + final chunk + [DONE]
            assert len(chunks) >= 3
            assert chunks[-1] == "data: [DONE]\n\n"

            # First chunk should be valid SSE JSON
            first_data = json.loads(chunks[0].replace("data: ", "").strip())
            assert first_data["object"] == "chat.completion.chunk"
            assert first_data["choices"][0]["delta"]["content"] == "Hello"


class TestDetectBaseModel:
    """Test base model detection from adapter config."""

    def test_detect_base_model_from_adapter(self, tmp_path):
        """Should read base model from adapter_config.json."""
        from soup_cli.commands.serve import _detect_base_model

        config_file = tmp_path / "adapter_config.json"
        config_file.write_text(json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B"
        }))

        result = _detect_base_model(config_file)
        assert result == "meta-llama/Llama-3.1-8B"

    def test_detect_base_model_missing_key(self, tmp_path):
        """Should return None if key is missing."""
        from soup_cli.commands.serve import _detect_base_model

        config_file = tmp_path / "adapter_config.json"
        config_file.write_text(json.dumps({"other_key": "value"}))

        result = _detect_base_model(config_file)
        assert result is None

    def test_detect_base_model_invalid_json(self, tmp_path):
        """Should return None for invalid JSON."""
        from soup_cli.commands.serve import _detect_base_model

        config_file = tmp_path / "adapter_config.json"
        config_file.write_text("not json")

        result = _detect_base_model(config_file)
        assert result is None


# v0.71.1 #230 — soup serve --record-thumbs
class TestRecordThumbs:
    def _client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from soup_cli.commands.serve import _create_app
        from soup_cli.utils.local_rl import init_local_rl_db

        monkeypatch.chdir(tmp_path)
        db = "rl.db"
        init_local_rl_db(db)
        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
            record_thumbs_db=db,
        )
        return TestClient(app), db

    def test_flag_in_help(self):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["serve", "--help"])
        assert result.exit_code == 0, result.output
        # Under color (CI sets FORCE_COLOR), Rich inserts ANSI codes between the
        # two dashes of an option name, so strip them before the substring check.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--record-thumbs" in clean

    def test_thumbs_endpoint_registered_when_db_set(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        client, _ = self._client(tmp_path, monkeypatch)
        routes = [r.path for r in client.app.routes]
        assert "/v1/thumbs" in routes

    def test_thumbs_endpoint_records_feedback(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        import sqlite3

        client, db = self._client(tmp_path, monkeypatch)
        resp = client.post(
            "/v1/thumbs",
            json={"prompt": "Q?", "response": "A.", "thumb": "up"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["thumb"] == "up"
        with sqlite3.connect(tmp_path / db) as conn:
            rows = conn.execute("SELECT prompt, response, thumb FROM thumbs").fetchall()
        assert rows == [("Q?", "A.", "up")]

    def test_thumbs_endpoint_rejects_bad_thumb(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        client, _ = self._client(tmp_path, monkeypatch)
        resp = client.post(
            "/v1/thumbs",
            json={"prompt": "Q?", "response": "A.", "thumb": "sideways"},
        )
        assert resp.status_code == 400

    def test_thumbs_endpoint_rejects_missing_fields(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        client, _ = self._client(tmp_path, monkeypatch)
        resp = client.post("/v1/thumbs", json={"prompt": "Q?"})
        assert resp.status_code == 400

    def test_thumbs_endpoint_404_when_not_enabled(self, tmp_path, monkeypatch):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")
        from soup_cli.commands.serve import _create_app

        monkeypatch.chdir(tmp_path)
        app = _create_app(
            model_obj=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            model_name="test-model",
            max_tokens_default=256,
        )  # no record_thumbs_db
        client = TestClient(app)
        resp = client.post(
            "/v1/thumbs",
            json={"prompt": "Q?", "response": "A.", "thumb": "up"},
        )
        assert resp.status_code == 404

    def test_thumbs_endpoint_rejects_non_string_thumb(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        client, _ = self._client(tmp_path, monkeypatch)
        resp = client.post(
            "/v1/thumbs",
            json={"prompt": "Q?", "response": "A.", "thumb": 1},
        )
        assert resp.status_code == 400

    def test_thumbs_endpoint_rejects_non_dict_body(self, tmp_path, monkeypatch):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")
        client, _ = self._client(tmp_path, monkeypatch)
        # A JSON array body does not match the dict-typed payload; FastAPI
        # rejects it (422) or the handler's isinstance guard does (400).
        resp = client.post("/v1/thumbs", json=[])
        assert resp.status_code in (400, 422)

    def test_serve_validates_record_thumbs_db_path_source(self):
        # Source-grep regression: the startup path must validate the
        # operator-supplied db path before init/use (v0.71.1 #230). Guards
        # against a future refactor dropping the containment check.
        import inspect

        from soup_cli.commands import serve

        src = inspect.getsource(serve)
        assert "validate_db_path(record_thumbs)" in src
