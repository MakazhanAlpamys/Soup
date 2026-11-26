"""Tests for the Web UI Export & Deploy to Ollama endpoints."""

from unittest.mock import MagicMock, patch

from tests.ui_helpers import ui_client


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token

    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestUIExportOllamaStatus:
    """Test GET /api/export/ollama/status."""

    def test_status_requires_auth(self):
        client = ui_client()
        res = client.get("/api/export/ollama/status")
        assert res.status_code in (401, 403)

    def test_status_detected(self):
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"):
            res = client.get("/api/export/ollama/status", headers=_auth_headers())
            assert res.status_code == 200
            data = res.json()
            assert data["available"] is True
            assert data["version"] == "0.5.4"

    def test_status_not_detected(self):
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value=None):
            res = client.get("/api/export/ollama/status", headers=_auth_headers())
            assert res.status_code == 200
            data = res.json()
            assert data["available"] is False
            assert data["version"] is None


class TestUIExportOllamaDeploy:
    """Test POST /api/export/ollama."""

    def test_deploy_requires_auth(self):
        client = ui_client()
        res = client.post("/api/export/ollama", json={"model_path": "./output"})
        assert res.status_code in (401, 403)

    def test_deploy_fails_when_ollama_missing(self, tmp_path):
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value=None):
            res = client.post(
                "/api/export/ollama",
                json={"model_path": str(tmp_path)},
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "not installed" in data["error"].lower()

    def test_deploy_fails_when_model_path_missing(self):
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"):
            res = client.post(
                "/api/export/ollama",
                json={"model_path": "/path/to/nonexistent/model"},
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    def test_deploy_fails_when_name_invalid(self, tmp_path):
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "invalid/name/with/slashes",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "invalid model name" in data["error"].lower()

    def test_deploy_fails_when_export_fails(self, tmp_path):
        """When GGUF export fails, endpoint must report error, never success."""
        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"), patch(
            "soup_cli.commands.export.export", side_effect=RuntimeError("llama.cpp quantize error")
        ):
            res = client.post(
                "/api/export/ollama",
                json={"model_path": str(tmp_path)},
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "llama.cpp quantize error" in data["error"]

    def test_deploy_fails_when_ollama_deploy_fails(self, tmp_path):
        """When ollama create fails, endpoint must report error, never success."""
        fake_gguf = tmp_path / "model.q4_k_m.gguf"
        fake_gguf.write_text("fake gguf content")

        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"), patch(
            "soup_cli.utils.ollama.deploy_to_ollama",
            return_value=(False, "GPU memory exhausted (out of VRAM)"),
        ):
            res = client.post(
                "/api/export/ollama",
                json={"model_path": str(tmp_path)},
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "out of VRAM" in data["error"]

    def test_deploy_success_with_first_reply(self, tmp_path):
        """Test full successful adapter export, deployment and first reply query."""
        fake_gguf = tmp_path / "adapter.q4_k_m.gguf"
        fake_gguf.write_text("fake gguf data")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Hello! I am your fine-tuned model running inside Ollama."
        }

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_resp

        client = ui_client()
        with patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"), patch(
            "soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")
        ), patch("httpx.Client", return_value=mock_client_instance):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "soup-my-adapter",
                    "test_prompt": "Hello!",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["model"] == "soup-my-adapter"
            assert (
                data["first_reply"]
                == "Hello! I am your fine-tuned model running inside Ollama."
            )
