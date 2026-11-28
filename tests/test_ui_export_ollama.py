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
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch(
                "soup_cli.commands.export.export",
                side_effect=RuntimeError("llama.cpp quantize error"),
            ),
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
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch(
                "soup_cli.utils.ollama.deploy_to_ollama",
                return_value=(False, "GPU memory exhausted (out of VRAM)"),
            ),
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
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")),
            patch("httpx.Client", return_value=mock_client_instance),
        ):
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
            assert data["registered"] is True
            assert data["verified"] is True
            assert data["model"] == "soup-my-adapter"
            assert data["first_reply"] == "Hello! I am your fine-tuned model running inside Ollama."

    def test_deploy_exports_gguf_when_none_exists(self, tmp_path):
        """Exercise real export entrypoint with minimal model dir, stubbed conversion/deploy."""
        import json
        from pathlib import Path

        model_dir = tmp_path / "minimal_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

        llama_dir = tmp_path / "mock_llama"
        llama_dir.mkdir()
        (llama_dir / "convert_hf_to_gguf.py").touch()

        def fake_convert(script, model_path, output_path, outtype, **kwargs):
            Path(output_path).write_text("fake f16 gguf")

        def fake_quantize(llama_dir, in_path, output_path, quant_type, **kwargs):
            Path(output_path).write_text("fake quantized gguf")

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.commands.export._transformers_missing", return_value=False),
            patch("soup_cli.commands.export._check_gguf_capabilities", return_value=None),
            patch("soup_cli.commands.export._find_llama_cpp", return_value=llama_dir),
            patch("soup_cli.commands.export._install_convert_deps", return_value=None),
            patch("soup_cli.commands.export._run_convert", side_effect=fake_convert),
            patch("soup_cli.commands.export._run_quantize", side_effect=fake_quantize),
            patch(
                "soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")
            ) as mock_deploy,
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(model_dir),
                    "deploy_name": "soup-exported-model",
                    "quant": "q4_k_m",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["registered"] is True
            assert data["verified"] is None

            # Verify deployment used the exact resulting GGUF file generated by export
            mock_deploy.assert_called_once()
            target_name, modelfile_content = mock_deploy.call_args[0]
            assert target_name == "soup-exported-model"
            expected_gguf = model_dir / f"{model_dir.name}.q4_k_m.gguf"
            assert expected_gguf.exists()
            assert str(expected_gguf) in modelfile_content

    def test_deploy_exports_requested_quantization_when_different_exists(self, tmp_path):
        """If directory has another quantization (e.g. q8_0), export requested quant."""
        from pathlib import Path

        existing_gguf = tmp_path / "model.q8_0.gguf"
        existing_gguf.write_text("q8_0 gguf content")

        def fake_export(*args, **kwargs):
            out_file = Path(kwargs["output"])
            out_file.write_text("q4_k_m gguf content")
            return out_file

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.commands.export.export", side_effect=fake_export) as mock_export,
            patch(
                "soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")
            ) as mock_deploy,
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "soup-re-quant-model",
                    "quant": "q4_k_m",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            mock_export.assert_called_once()
            assert mock_export.call_args.kwargs["quant"] == "q4_k_m"

            mock_deploy.assert_called_once()
            assert ".q4_k_m.gguf" in str(mock_deploy.call_args[0][1])

    def test_deploy_prompt_omitted_returns_verified_none(self, tmp_path):
        """When test_prompt is omitted, registration succeeds with verified=None."""
        fake_gguf = tmp_path / "model.q4_k_m.gguf"
        fake_gguf.write_text("fake gguf data")

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")),
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "soup-no-prompt-model",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["registered"] is True
            assert data["verified"] is None
            assert data["first_reply"] is None
            assert data["verification_error"] is None

    def test_deploy_verification_failed_registration_succeeded(self, tmp_path):
        """When verification fails or returns empty, registration still succeeds."""
        fake_gguf = tmp_path / "model.q4_k_m.gguf"
        fake_gguf.write_text("fake gguf data")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_resp

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")),
            patch("httpx.Client", return_value=mock_client_instance),
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "soup-failed-verify-model",
                    "test_prompt": "Hello test!",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["registered"] is True
            assert data["verified"] is False
            assert data["first_reply"] is None
            assert "HTTP 500" in data["verification_error"]

    def test_deploy_verification_empty_reply_shows_failure(self, tmp_path):
        """When verification returns an empty reply string, mark verified=False."""
        fake_gguf = tmp_path / "model.q4_k_m.gguf"
        fake_gguf.write_text("fake gguf data")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "   "}

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_resp

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch("soup_cli.utils.ollama.deploy_to_ollama", return_value=(True, "success")),
            patch("httpx.Client", return_value=mock_client_instance),
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "deploy_name": "soup-empty-reply-model",
                    "test_prompt": "Hello test!",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["registered"] is True
            assert data["verified"] is False
            assert data["first_reply"] is None
            assert "empty response" in data["verification_error"]

    def test_deploy_different_quant_fails_reports_available(self, tmp_path):
        """When requested quant export fails, error must list existing available GGUFs."""
        existing_gguf = tmp_path / "model.q8_0.gguf"
        existing_gguf.write_text("q8_0 content")

        client = ui_client()
        with (
            patch("soup_cli.utils.ollama.detect_ollama", return_value="0.5.4"),
            patch(
                "soup_cli.commands.export.export", side_effect=RuntimeError("quantization missing")
            ),
        ):
            res = client.post(
                "/api/export/ollama",
                json={
                    "model_path": str(tmp_path),
                    "quant": "q4_k_m",
                },
                headers=_auth_headers(),
            )
            assert res.status_code == 400
            data = res.json()
            assert data["success"] is False
            assert "quantization missing" in data["error"]
            assert "model.q8_0.gguf" in data["error"]
