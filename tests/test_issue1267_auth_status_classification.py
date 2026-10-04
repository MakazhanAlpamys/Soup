"""Tests for #1267: 401/403 must be classified from the exception, not bare digits."""

import urllib.error
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from soup_cli.utils.errors import format_friendly_error


def _render(exc: Exception) -> str:
    buf = StringIO()
    console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", console):
        format_friendly_error(exc, verbose=False)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _HTTPStatusError(Exception):
    """Minimal stand-in for requests/httpx/huggingface_hub status errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.response = _FakeResponse(status_code)


# --- false positives: digits inside a message are not a status ---


def test_row_number_401_is_not_an_auth_failure():
    out = _render(ValueError(
        "train row 401: no causal-loss target remains after tokenization/truncation "
        "at data.max_length=64; the assistant response is absent or fully truncated."
    ))
    assert "Authentication failed." not in out
    assert "train row 401" in out


def test_row_number_403_is_not_an_access_denial():
    out = _render(ValueError("train row 403: no causal-loss target remains"))
    assert "Access denied." not in out
    assert "train row 403" in out


def test_config_digits_in_message_are_not_an_access_denial():
    out = _render(ValueError(
        "train row 7: no causal-loss target remains at data.max_length=4030"
    ))
    assert "Access denied." not in out
    assert "max_length=4030" in out


def test_tensor_size_with_401_digits_keeps_the_real_message():
    out = _render(RuntimeError(
        "The size of tensor a (4013) must match the size of tensor b (4096) "
        "at non-singleton dimension 0"
    ))
    assert "Authentication failed." not in out
    assert "tensor a (4013)" in out


def test_matmul_shapes_with_403_digits_keep_the_real_message():
    out = _render(RuntimeError(
        "mat1 and mat2 shapes cannot be multiplied (2x4030 and 4096x8)"
    ))
    assert "Access denied." not in out
    assert "2x4030" in out


def test_index_digits_keep_the_real_message():
    out = _render(IndexError(
        "index 40132 is out of bounds for dimension 0 with size 32000"
    ))
    assert "Authentication failed." not in out
    assert "40132" in out


def test_adapter_path_digits_keep_the_real_message():
    out = _render(ValueError(
        "Can't find 'adapter_config.json' at 'output/checkpoint-4010'"
    ))
    assert "Authentication failed." not in out
    assert "checkpoint-4010" in out


def test_permission_error_with_date_path_keeps_the_real_message():
    out = _render(PermissionError("[Errno 13] Permission denied: 'runs/20260401'"))
    assert "Authentication failed." not in out
    assert "Permission denied" in out


def test_revision_sha_containing_401_keeps_the_real_message():
    exc = _HTTPStatusError(
        "404 Client Error: Not Found for url: "
        "https://huggingface.co/model/resolve/9c1f4012abc/config.json",
        404,
    )
    out = _render(exc)
    assert "Authentication failed." not in out
    assert "9c1f4012" in out


def test_request_id_containing_403_keeps_the_real_message():
    exc = _HTTPStatusError(
        "404 Client Error: Not Found for url (Request ID: Root=1-68d4b2c1-4a4039d2)",
        404,
    )
    out = _render(exc)
    assert "Access denied." not in out
    assert "4a4039d2" in out


# --- genuine 401/403 keep (or newly get) the specific hint ---


def test_httpx_style_401_gets_the_authentication_hint():
    exc = _HTTPStatusError(
        "Client error '401 Unauthorized' for url 'https://api.openai.com/v1/chat/completions'",
        401,
    )
    out = _render(exc)
    assert "Authentication failed." in out


def test_status_403_with_401_digits_in_url_maps_to_access_denied():
    exc = _HTTPStatusError(
        "Client error '403 Forbidden' for url 'https://api.example.com/v1/projects/40172/x'",
        403,
    )
    out = _render(exc)
    assert "Access denied." in out
    assert "Authentication failed." not in out


def test_wrapped_exception_status_is_used():
    # peft/transformers wrap hub errors: the status lives on the __cause__.
    cause = _HTTPStatusError("401 Client Error: Unauthorized for url", 401)
    wrapper = RuntimeError("Failed to load adapter")
    wrapper.__cause__ = cause
    out = _render(wrapper)
    assert "Authentication failed." in out


def test_urllib_http_error_401_gets_the_authentication_hint():
    exc = urllib.error.HTTPError(
        "https://huggingface.co/api/whoami-v2", 401, "Unauthorized", None, None
    )
    out = _render(exc)
    assert "Authentication failed." in out


def test_http_phrasing_without_a_response_object_is_accepted():
    out = _render(Exception("401 Client Error: Unauthorized for url: https://x"))
    assert "Authentication failed." in out


def test_gated_repo_error_keeps_its_own_hint():
    gated_cls = type("GatedRepoError", (Exception,), {})
    exc = gated_cls("403 Client Error: Forbidden for url (gated repo)")
    exc.response = _FakeResponse(403)
    out = _render(exc)
    assert "license acceptance" in out
    assert "Access denied." not in out


# --- wrapped hub errors and object-status probes (#1268 review) ---


def _named_error(name: str, message: str, status: int) -> Exception:
    """An exception whose class name matches a huggingface_hub error, carrying a status."""
    exc = type(name, (Exception,), {})(message)
    exc.response = _FakeResponse(status)
    return exc


def test_wrapped_repository_not_found_keeps_the_real_message():
    # peft: PeftConfig.from_pretrained("output/checkpoint-4000") on a missing local
    # directory asks the Hub, which answers an unauthenticated user with a 401
    # RepositoryNotFoundError; peft re-raises a ValueError `from` it.
    cause = _named_error("RepositoryNotFoundError", "401 Client Error. Repository Not Found", 401)
    try:
        raise ValueError("Can't find 'adapter_config.json' at 'output/checkpoint-4000'") from cause
    except ValueError as exc:
        out = _render(exc)
    assert "Authentication failed." not in out
    assert "checkpoint-4000" in out


def test_wrapped_gated_repo_error_keeps_the_gated_hint():
    # transformers: cached_files raises OSError("You are trying to access a gated
    # repo...") from the GatedRepoError.
    cause = _named_error("GatedRepoError", "401 Client Error. Cannot access gated repo", 401)
    try:
        raise OSError("You are trying to access a gated repo.") from cause
    except OSError as exc:
        out = _render(exc)
    assert "license acceptance" in out
    assert "Authentication failed." not in out


def test_implicitly_chained_status_is_used():
    # Raised inside `except` without `from`: the status lives on __context__ only.
    try:
        try:
            raise _HTTPStatusError("boom", 403)
        except _HTTPStatusError:
            raise RuntimeError("upload failed")
    except RuntimeError as exc:
        out = _render(exc)
    assert "Access denied." in out


def test_wrapped_urllib_error_uses_its_code():
    # urllib's HTTPError also says "HTTP Error 403" in its text, which the text
    # fallback now reads anywhere in the chain (#1361); the phrase-free variant in
    # test_issue1361_auth_status_text_chain.py pins the `.code` probe on its own.
    cause = urllib.error.HTTPError("https://x/y", 403, "Forbidden", None, None)
    try:
        raise RuntimeError("download failed") from cause
    except RuntimeError as exc:
        out = _render(exc)
    assert "Access denied." in out


def test_direct_status_code_attribute_is_used():
    # SDK errors (for example openai.APIStatusError) carry `.status_code` on the exception.
    exc = Exception("request failed")
    exc.status_code = 401
    assert "Authentication failed." in _render(exc)


@pytest.mark.parametrize(
    ("message", "hint"),
    [
        ("Client error '401 Unauthorized' for url 'https://x'", "Authentication failed."),
        ("401 Client Error: Unauthorized for url: https://x", "Authentication failed."),
        ("HTTP Error 401: Unauthorized", "Authentication failed."),
        ("request failed: HTTP 401", "Authentication failed."),
        ("Error code: 401 - {'error': 'invalid_api_key'}", "Authentication failed."),
        ("Client error '403 Forbidden' for url 'https://x'", "Access denied."),
        ("403 Client Error: Forbidden for url: https://x", "Access denied."),
        ("HTTP Error 403: Forbidden", "Access denied."),
        ("request failed: HTTP 403", "Access denied."),
        ("Error code: 403 - {'error': 'forbidden'}", "Access denied."),
    ],
)
def test_each_http_phrasing_reaches_its_hint(message, hint):
    assert hint in _render(Exception(message))
