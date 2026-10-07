"""v0.44.0 Part C — UI env knob tests.

The UI plugin registry (``soup_cli.ui.plugins``) that this file also covered was
removed in #1465: nothing rendered a registered tab. Its tests went with it; the
``resolve_ui_env`` tests stay (``utils/ui_env.py`` is a separate question).
"""

from __future__ import annotations

import pytest

from soup_cli.utils.ui_env import UiEnv, resolve_ui_env

# --- UI env knobs -----------------------------------------------------------

def test_resolve_ui_env_empty():
    env = resolve_ui_env({})
    assert env == UiEnv(None, None, None, None, None)


def test_resolve_ui_env_full():
    env = resolve_ui_env(
        {
            "API_HOST": "127.0.0.1",
            "API_PORT": "8080",
            "API_KEY": "secret-key-1234",
            "GRADIO_HOST": "0.0.0.0",
            "GRADIO_PORT": "7860",
        }
    )
    assert env.api_host == "127.0.0.1"
    assert env.api_port == 8080
    assert env.api_key == "secret-key-1234"
    assert env.gradio_host == "0.0.0.0"
    assert env.gradio_port == 7860


def test_resolve_ui_env_invalid_port():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "0"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "99999"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_PORT": "not-int"})


def test_resolve_ui_env_invalid_host():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "bad host with spaces"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "x\x00bad"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_HOST": "x" * 300})


def test_resolve_ui_env_blank_treated_as_missing():
    env = resolve_ui_env({"API_HOST": "  ", "API_KEY": "  "})
    assert env.api_host is None
    assert env.api_key is None


def test_resolve_ui_env_invalid_key():
    with pytest.raises(ValueError):
        resolve_ui_env({"API_KEY": "k\x00ey"})
    with pytest.raises(ValueError):
        resolve_ui_env({"API_KEY": "k" * 1000})


def test_ui_env_frozen():
    env = UiEnv(None, None, None, None, None)
    with pytest.raises(Exception):
        env.api_host = "x"  # type: ignore[misc]
