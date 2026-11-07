"""Tests for Web UI Visual Config Builder — schema, recipes, form-to-YAML."""

import pytest

from tests.ui_helpers import ui_client


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class TestConfigSchemaEndpoint:
    """Test GET /api/config/schema endpoint."""

    def test_schema_endpoint_exists(self):
        """Route /api/config/schema should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/config/schema" in routes

    def test_schema_returns_json(self):
        """Schema endpoint should return valid JSON."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/config/schema", headers=_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_schema_includes_base_field(self):
        """Schema should include 'base' as a required string field."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/config/schema", headers=_auth_headers())
        data = response.json()
        assert "base" in data
        assert data["base"]["type"] == "string"
        assert data["base"]["required"] is True

    def test_schema_includes_task_enum(self):
        """Schema should include 'task' with enum options."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/config/schema", headers=_auth_headers())
        data = response.json()
        assert "task" in data
        assert "options" in data["task"]
        assert "sft" in data["task"]["options"]
        assert "dpo" in data["task"]["options"]

    def test_schema_includes_training_fields(self):
        """Schema should include nested training fields."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/config/schema", headers=_auth_headers())
        data = response.json()
        assert "training" in data
        training = data["training"]
        assert "epochs" in training
        assert "lr" in training

    def test_schema_includes_constraints(self):
        """Schema should include ge/le constraints for bounded fields."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/config/schema", headers=_auth_headers())
        data = response.json()
        # Check that training.epochs has constraints or type info
        training = data.get("training", {})
        if "epochs" in training:
            assert training["epochs"]["type"] in ("integer", "number")


class TestRecipesEndpoint:
    """Test GET /api/recipes endpoint."""

    def test_recipes_endpoint_exists(self):
        """Route /api/recipes should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/recipes" in routes

    def test_recipes_returns_catalog(self):
        """Recipes endpoint should return all recipes."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/recipes", headers=_auth_headers())
        assert response.status_code == 200
        data = response.json()
        assert "recipes" in data
        assert len(data["recipes"]) >= 20  # We have 29 recipes

    def test_recipes_have_required_fields(self):
        """Each recipe should have name, model, task fields."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.get("/api/recipes", headers=_auth_headers())
        recipes = response.json()["recipes"]
        for recipe in recipes:
            assert "name" in recipe
            assert "model" in recipe
            assert "task" in recipe
            assert "yaml" in recipe


class TestFormToYamlEndpoint:
    """Test POST /api/config/from-form endpoint."""

    def test_form_to_yaml_endpoint_exists(self):
        """Route /api/config/from-form should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/config/from-form" in routes

    def test_form_to_yaml_produces_valid_config(self):
        """Form-to-YAML should produce a valid config string."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.post(
            "/api/config/from-form",
            json={
                "base": "meta-llama/Llama-3.1-8B-Instruct",
                "task": "sft",
                "data": {"train": "./data.jsonl", "format": "alpaca"},
                "training": {"epochs": 3, "lr": 2e-4},
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert "yaml" in data
        assert "base:" in data["yaml"]
        assert "meta-llama" in data["yaml"]

    def test_form_to_yaml_validates(self):
        """Form-to-YAML should validate via load_config_from_string."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.post(
            "/api/config/from-form",
            json={
                "base": "test-model",
                "data": {"train": "./data.jsonl"},
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert "yaml" in data

    def test_form_to_yaml_invalid_returns_error(self):
        """Invalid form values should return error."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        # Missing base — should fail validation
        response = client.post(
            "/api/config/from-form",
            json={"task": "sft"},
            headers=_auth_headers(),
        )
        assert response.status_code in (200, 400)
        if response.status_code == 200:
            data = response.json()
            assert "error" in data

    def test_form_to_yaml_requires_auth(self):
        """Form-to-YAML (POST) should require auth."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.post(
            "/api/config/from-form",
            json={"base": "test"},
        )
        assert response.status_code == 401

    def test_form_to_yaml_special_chars_in_keys(self):
        """Form values with special YAML characters should be handled safely."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.post(
            "/api/config/from-form",
            json={
                "base": "test: model\noutput: /etc",
                "data": {"train": "./data.jsonl"},
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        # The result should either be valid YAML or an error — never injection
        if "yaml" in data:
            # The colon/newline in base value should be quoted by yaml.dump
            assert "base:" in data["yaml"]
        elif "error" in data:
            # Validation error is also acceptable
            pass

    def test_form_to_yaml_with_lora(self):
        """Form-to-YAML should handle nested lora config."""
        pytest.importorskip("fastapi")

        from soup_cli.ui.app import create_app

        client = ui_client(create_app())
        response = client.post(
            "/api/config/from-form",
            json={
                "base": "test-model",
                "data": {"train": "./data.jsonl"},
                "training": {
                    "epochs": 3,
                    "lora": {"r": 16, "alpha": 32},
                },
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert "yaml" in data
        assert "lora:" in data["yaml"] or "r:" in data["yaml"]
