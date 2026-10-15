"""Bound decoded HTTP response bodies before JSON parsing or buffering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

JUDGE_RESPONSE_LIMIT_BYTES = 1024 * 1024
GENERATION_RESPONSE_LIMIT_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class ResponseTooLargeError(ValueError):
    """A selected endpoint exceeded its decoded response budget."""


def post_capped(
    url: str, *, max_bytes: int, endpoint_label: str, **kwargs: Any,
) -> httpx.Response:
    """POST and return a bounded, decoded response, always closing the stream.

    Callers retain their endpoint validation, request headers and status/retry
    policies. Error text contains only the caller's setting label and limit,
    never the URL, credentials, request headers or response body.
    """
    import httpx

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    def too_large() -> ResponseTooLargeError:
        return ResponseTooLargeError(
            f"Response for {endpoint_label} exceeds the {max_bytes} bytes limit"
        )

    with httpx.stream("POST", url, **kwargs) as response:
        try:
            declared_length = int(response.headers.get("Content-Length", ""))
        except ValueError:
            declared_length = 0
        if declared_length > max_bytes:
            raise too_large()

        chunks: list[bytes] = []
        size = 0
        # iter_bytes counts decompressed content, including gzip replies.
        for chunk in response.iter_bytes(chunk_size=_READ_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise too_large()
            chunks.append(chunk)

        headers = response.headers.copy()
        for name in ("Content-Encoding", "Content-Length", "Transfer-Encoding"):
            headers.pop(name, None)
        headers["Content-Length"] = str(size)
        return httpx.Response(
            response.status_code, headers=headers, content=b"".join(chunks),
            request=response.request, extensions=response.extensions,
        )
