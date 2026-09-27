"""DeepSpeed-MII serve backend (v0.27.0).

MII (Model Implementations for Inference) provides high-throughput serving
with tensor parallelism. See https://github.com/deepspeedai/DeepSpeed-MII.

All imports are lazy so that `soup --help` stays fast.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def is_mii_available() -> bool:
    """Return True when the ``mii`` package is importable.

    Honours test-injected stubs: if ``sys.modules["mii"]`` is explicitly set
    to ``None`` (pytest's idiom for "pretend this module is absent"), the
    ``import mii`` statement below will raise ``ImportError`` without
    consulting disk.
    """
    if "mii" in sys.modules and sys.modules["mii"] is None:
        return False
    try:
        import mii  # noqa: F401
    except ImportError:
        return False
    return True


def create_mii_pipeline(
    model_path: str,
    tensor_parallel: int = 1,
    max_length: int = 4096,
    replica_num: int = 1,
    tokenizer: Any = None,
) -> Any:
    """Create a DeepSpeed-MII pipeline.

    Args:
        model_path: HF model id or local path.
        tensor_parallel: TP size (must evenly divide GPU count).
        max_length: Max sequence length the pipeline will handle.
        replica_num: Number of replicas (for multi-node).
        tokenizer: the served model's HF tokenizer, the same object
            :func:`build_mii_app` renders chat templates with. When given, MII
            tokenizes prompts through :func:`create_mii_tokenizer` around it
            instead of loading its own copy, which is what lets a template
            that already rendered BOS reach the engine with one BOS (#785).
            None keeps MII loading the tokenizer from ``model_path`` itself.

    Raises:
        ImportError: If the ``deepspeed-mii`` package is not installed.
    """
    if not is_mii_available():
        raise ImportError(
            "deepspeed-mii is not installed. "
            "Install with: pip install \"soup-cli[mii]\" "
            "or pip install deepspeed-mii"
        )

    import mii

    kwargs: dict[str, Any] = {}
    if tokenizer is not None:
        kwargs["tokenizer"] = create_mii_tokenizer(tokenizer)

    return mii.pipeline(
        model_path,
        tensor_parallel=tensor_parallel,
        max_length=max_length,
        replica_num=replica_num,
        **kwargs,
    )


class TokenizedPrompt(str):
    """A rendered prompt carrying the ids the engine must run on instead (#785).

    ``MIIPipeline.__call__`` accepts strings only, and MII encodes each one
    itself with its tokenizer's default ``add_special_tokens=True``, so a
    template that renders ``{{ bos_token }}`` reached the model with two of
    them. The pipeline hands each prompt to ``tokenizer.encode`` untouched
    (``RaggedBatchBase._put_request`` on MII 0.3.3), so the prompt itself
    carries the ids :func:`~soup_cli.utils.vllm.build_engine_prompt` built
    with no tokenizer special tokens, and the tokenizer
    :func:`create_mii_tokenizer` gave the pipeline returns them verbatim. A
    plain ``str`` is the legacy role-prefixed prompt, which carries no special
    tokens of its own and keeps getting the engine's.
    """

    __slots__ = ("token_ids",)

    def __new__(cls, text: str, token_ids: Any) -> "TokenizedPrompt":
        prompt = super().__new__(cls, text)
        prompt.token_ids = list(token_ids)
        return prompt


def encode_mii_prompt(tokenizer: Any, input: Any, **kwargs: Any) -> Any:
    """The encode MII's ``HFTokenizer`` delegates to, minus the doubled BOS.

    A :class:`TokenizedPrompt` encodes to the ids it carries. Anything else is
    the exact call MII 0.3.3 makes, ``tokenizer.encode(input,
    return_tensors="pt")``, so the legacy prompt is tokenized as it always was.
    """
    token_ids = getattr(input, "token_ids", None)
    if token_ids is None:
        return tokenizer.encode(input, **kwargs)
    if kwargs.get("return_tensors") == "pt":
        import torch

        return torch.tensor([token_ids], dtype=torch.long)
    return list(token_ids)


def create_mii_tokenizer(tokenizer: Any) -> Any:
    """Wrap an HF tokenizer for ``mii.pipeline(tokenizer=...)`` (#785).

    The seam on MII 0.3.3, checked against ``mii/modeling/tokenizers.py``:
    ``ModelConfig.tokenizer`` accepts an ``MIITokenizerWrapper``, but
    ``load_tokenizer`` still wraps whatever it is given in a fresh
    ``HFTokenizer``, whose ``encode(input)`` calls
    ``self.tokenizer.encode(input, return_tensors="pt").flatten()`` on the
    object. So the object handed in is used as the HF tokenizer, not as the
    wrapper, and has to answer that call: ``encode`` accepts
    ``return_tensors``, ``__len__`` backs ``vocab_size``, and every other
    attribute (``eos_token_id``, ``convert_tokens_to_ids``, ``decode``,
    ``pad_token``, ...) is the real tokenizer's. Subclassing ``HFTokenizer``
    is what satisfies the config's type check, and keeps the object a working
    wrapper should a later MII stop re-wrapping it.

    Deliberate side effect: when the handed-in tokenizer has no ``pad_token``,
    the wrapper sets one on it in place (``eos_token``). This mutates the
    caller's object — the same tokenizer ``build_mii_app`` renders chat
    templates with — and so runs against the repo's "never mutate" style rule
    on purpose: it is exactly what MII's own ``HFTokenizer`` does to a tokenizer
    it loads from a path, so matching it keeps the wrapped and unwrapped paths
    encoding identically.
    """
    from mii.modeling.tokenizers import HFTokenizer

    class SoupMiiTokenizer(HFTokenizer):
        def __init__(self, hf_tokenizer: Any) -> None:
            if getattr(hf_tokenizer, "pad_token", None) is None:
                # What HFTokenizer does to a tokenizer it loads from a path.
                hf_tokenizer.pad_token = hf_tokenizer.eos_token
            super().__init__(hf_tokenizer)

        def __len__(self) -> int:
            # Also what makes the wrapper truthy: ModelConfig replaces a falsy
            # ``tokenizer`` with ``model_name_or_path``.
            return len(self.tokenizer)

        def __getattr__(self, name: str) -> Any:
            if name == "tokenizer":
                raise AttributeError(name)
            return getattr(self.tokenizer, name)

        def encode(self, input: Any, **kwargs: Any) -> Any:
            if not kwargs:
                # As the wrapper: the flat tensor HFTokenizer.encode returns.
                return encode_mii_prompt(self.tokenizer, input, return_tensors="pt").flatten()
            # As the HF tokenizer under MII 0.3.3's HFTokenizer, which passes
            # return_tensors="pt" and flattens the result itself.
            return encode_mii_prompt(self.tokenizer, input, **kwargs)

    return SoupMiiTokenizer(tokenizer)


try:
    from pydantic import BaseModel as _BaseModel

    class _MiiMessage(_BaseModel):
        role: str
        content: str

    class _MiiChatRequest(_BaseModel):
        model: str = ""
        messages: list[_MiiMessage]
        max_tokens: Optional[int] = None
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

except ImportError:
    _MiiMessage = None  # type: ignore[assignment]
    _MiiChatRequest = None  # type: ignore[assignment]


def _ensure_mii_request_models():
    if _MiiChatRequest is None:
        raise ImportError("pydantic is required for the MII server")
    return _MiiMessage, _MiiChatRequest


def _check_prompt_length(response: Any, prompt_token_ids: Optional[list[int]]) -> None:
    """Say so when MII encoded a templated prompt to a different length than Soup did.

    ``Response.prompt_length`` is what the engine actually ran on. For a
    templated prompt it must equal the ids the shared encoder produced; one
    more means the engine's tokenizer added its own BOS again, i.e. the
    pipeline was not built with the wrapped tokenizer (#785).
    """
    if prompt_token_ids is None:
        return
    prompt_length = getattr(response, "prompt_length", None)
    if isinstance(prompt_length, int) and prompt_length != len(prompt_token_ids):
        logger.warning(
            "MII tokenized the templated prompt to %d ids but Soup encoded it "
            "to %d; the pipeline is not using the tokenizer soup serve wrapped, "
            "so the template's special tokens are being re-added (#785)",
            prompt_length,
            len(prompt_token_ids),
        )


def build_mii_app(
    pipeline: Any,
    model_name: str,
    max_tokens_default: int = 512,
    tokenizer: Any = None,
) -> Any:
    """Wrap a DeepSpeed-MII pipeline as a minimal OpenAI-compatible FastAPI
    app exposing ``/v1/chat/completions`` and ``/v1/models`` (#38, v0.33.0).

    The pipeline is held by closure so a single MII instance handles all
    requests (MII pipelines are thread-safe for concurrent generation).

    Args:
        pipeline: result of :func:`create_mii_pipeline` or any callable
            ``pipeline(prompts, max_new_tokens=...) -> [GeneratedResponse]``
            with ``.generated_text`` attributes.
        model_name: stable id surfaced in /v1/models and the response.
        max_tokens_default: default for requests that don't specify max_tokens.
        tokenizer: the served model's HF tokenizer, whose chat template is
            applied to incoming messages. ``None`` degrades to the legacy
            role-prefixed prompt — the same fallback the vLLM and transformers
            backends use when a model ships no template.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    # THE shared prompt builder and finish_reason mapper (#332, #333). Imported
    # here rather than at module scope to keep ``soup --help`` fast, per the
    # module's lazy-import policy. Reusing them is the point: a second copy is
    # how the MII backend drifted from the other two in the first place.
    from soup_cli.utils.vllm import build_engine_prompt, resolve_finish_reason

    # Models are module-level (not closure) so FastAPI's introspection can
    # resolve forward refs.
    _ensure_mii_request_models()

    app = FastAPI(title=f"soup-cli MII serve [{model_name}]")
    # Loopback-only CORS — mirrors v0.30.0 transformers backend policy.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    def _list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": model_name, "object": "model",
                "created": 0, "owned_by": "soup-cli-mii",
            }],
        }

    @app.post("/v1/chat/completions")
    def _chat(request: _MiiChatRequest) -> dict:
        import time
        import uuid

        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="streaming not supported on the MII backend yet",
            )
        if request.max_tokens is not None and (
            request.max_tokens < 1 or request.max_tokens > 16384
        ):
            raise HTTPException(
                status_code=400, detail="max_tokens must be in [1, 16384]",
            )
        # #785: MII tokenizes the string itself with add_special_tokens=True,
        # which put a second BOS in front of the one the template rendered.
        # Its pipeline takes strings only, so a templated prompt goes over as
        # a str that carries its ids, and the tokenizer create_mii_pipeline
        # gave MII returns those. The legacy fallback stays a plain str and
        # is tokenized exactly as before.
        prompt, prompt_token_ids = build_engine_prompt(request.messages, tokenizer)
        engine_prompt = (
            prompt if prompt_token_ids is None else TokenizedPrompt(prompt, prompt_token_ids)
        )

        max_tokens = request.max_tokens or max_tokens_default
        try:
            responses = pipeline(
                [engine_prompt],
                max_new_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        except Exception:
            raise HTTPException(status_code=500, detail="Internal server error")

        if not responses:
            raise HTTPException(status_code=500, detail="No response generated")
        first = responses[0]
        text = getattr(first, "generated_text", None) or str(first)
        _check_prompt_length(first, prompt_token_ids)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": resolve_finish_reason(first, max_tokens),
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    return app
