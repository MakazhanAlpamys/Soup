"""SFT row-formatter factory (v0.36.0 Part A).

Builds the ``format_row`` function used by ``SFTTrainerWrapper`` based on
``DataConfig`` flags. Three modes:

- ``train_on_responses_only=True`` (default): pre-tokenise to
  ``{input_ids, labels, attention_mask}`` with non-assistant tokens masked
  to ``IGNORE_INDEX``. SFTTrainer detects pre-tokenised columns and skips
  its own tokenization.
- ``train_on_messages_with_train_field=True``: like above but uses the
  per-message ``train: bool`` field.
- both False: legacy ``{text}`` path. SFTTrainer tokenizes on its own (TRL
  heuristic — known to be wrong for multi-turn chat data; left as opt-out
  for backwards compat).

Tokenizer-without-chat-template degrades to the legacy text path and emits
a single warning. v0.36.0 Part C will harden this into a hard error once
``chat_template`` is a first-class config field.
"""

from __future__ import annotations

from typing import Any, Callable

from soup_cli.config.schema import DataConfig
from soup_cli.data.chat_templates import apply_chat_template_override
from soup_cli.data.loss_mask import (
    build_assistant_only_labels,
    build_per_message_train_labels,
)


def build_format_row(
    tokenizer: Any,
    data_cfg: DataConfig,
    console: Any | None = None,
    training_cfg: Any | None = None,
) -> Callable[[dict], dict]:
    """Factory: return the ``format_row`` function appropriate for ``data_cfg``.

    v0.53.2 #137: when ``training_cfg.reasoning_effort`` is set the gpt-oss
    ``<|reasoning_effort|>...<|/reasoning_effort|>`` control tag is injected
    into the system message before formatting. When ``training_cfg.train_on_eot``
    is true the loss mask is extended to include the trailing EOT/EOS token
    after each assistant span (axolotl ``train_on_eot``).
    """
    # v0.36.0 Part C: apply chat-template override BEFORE deciding on path.
    # Override may turn a templateless tokenizer into a usable one. The
    # override warning surfaces here (not at sft.py call site) so it fires
    # exactly once per setup, regardless of whether the legacy text path or
    # the loss-mask path is selected.
    apply_chat_template_override(tokenizer, data_cfg.chat_template, console=console)

    has_template = bool(getattr(tokenizer, "chat_template", None))
    use_responses_only = bool(data_cfg.train_on_responses_only)
    use_train_field = bool(data_cfg.train_on_messages_with_train_field)
    max_length = int(data_cfg.max_length)

    reasoning_effort = (
        getattr(training_cfg, "reasoning_effort", None) if training_cfg else None
    )
    include_eot = bool(
        getattr(training_cfg, "train_on_eot", False) if training_cfg else False
    )

    # v0.53.7 #87: custom prompt_strategy live runtime. Resolves the
    # ``module.path:fn_name`` spec once at setup time (fail-fast on bad import)
    # and applies the transform to each row BEFORE template rendering.
    prompt_strategy_spec = getattr(data_cfg, "prompt_strategy", None)

    if (use_responses_only or use_train_field) and not has_template:
        if console is not None:
            console.print(
                "[yellow]train_on_responses_only requested but tokenizer "
                "has no chat_template — falling back to text path. Pass "
                "data.chat_template explicitly to enable masking.[/]"
            )
        return _wrap_with_prompt_strategy(
            _wrap_with_reasoning_effort(
                _legacy_text_format_row(tokenizer), reasoning_effort
            ),
            prompt_strategy_spec,
        )

    if use_train_field:
        inner = _build_per_message_format_row(tokenizer, max_length)
    elif use_responses_only:
        inner = _build_assistant_only_format_row(
            tokenizer, max_length, include_eot=include_eot
        )
    else:
        inner = _legacy_text_format_row(tokenizer)
    return _wrap_with_prompt_strategy(
        _wrap_with_reasoning_effort(inner, reasoning_effort),
        prompt_strategy_spec,
    )


def _wrap_with_prompt_strategy(
    inner: Callable[[dict], dict], spec: Any | None
) -> Callable[[dict], dict]:
    """v0.53.7 #87 — apply ``data.prompt_strategy`` transform per-row.

    Resolves the spec eagerly at wrap-time so a bad import fails at setup
    rather than mid-training. Per-row callable exceptions are logged at
    DEBUG and the original row falls through (matches v0.33.0 #47
    CrossDocCollator silent-degrade policy).
    """
    if spec is None or not isinstance(spec, str):
        return inner
    # Resolve eagerly so trainer setup surfaces a bad spec loudly.
    from soup_cli.utils.data_pipeline import (
        apply_prompt_strategy,
        resolve_prompt_strategy,
    )

    resolve_prompt_strategy(spec)  # fail fast on import / signature errors

    def wrapped(example: dict) -> dict:
        transformed = apply_prompt_strategy(spec, example)
        if isinstance(transformed, dict):
            return inner(transformed)
        # apply_prompt_strategy may return a non-dict Mapping (e.g.
        # MappingProxyType); coerce so the existing format_row helpers see
        # an ordinary mutable dict.
        return inner(dict(transformed))

    return wrapped


def _wrap_with_reasoning_effort(
    inner: Callable[[dict], dict], level: Any | None
) -> Callable[[dict], dict]:
    """Decorate ``inner`` so each example's messages get a reasoning-effort prefix."""
    if level is None:
        return inner
    from soup_cli.utils.reasoning_effort import apply_reasoning_effort_prefix

    def wrapped(example: dict) -> dict:
        msgs = example.get("messages")
        if isinstance(msgs, list) and msgs:
            new_example = {
                **example,
                "messages": apply_reasoning_effort_prefix(msgs, level),
            }
            return inner(new_example)
        return inner(example)

    return wrapped


def _build_assistant_only_format_row(
    tokenizer: Any, max_length: int, include_eot: bool = False
) -> Callable[[dict], dict]:
    def format_row(example: dict) -> dict:
        return build_assistant_only_labels(
            example["messages"],
            tokenizer,
            max_length=max_length,
            include_eot=include_eot,
        )

    return format_row


def _build_per_message_format_row(
    tokenizer: Any, max_length: int
) -> Callable[[dict], dict]:
    def format_row(example: dict) -> dict:
        return build_per_message_train_labels(
            example["messages"], tokenizer, max_length=max_length
        )

    return format_row


def _legacy_text_format_row(tokenizer: Any) -> Callable[[dict], dict]:
    def format_row(example: dict) -> dict:
        if not getattr(tokenizer, "chat_template", None):
            # v0.36.0 Part C: hard error replaces the silent
            # ``f"{role}: {content}"`` fallback that produced garbage
            # training data on tokenizers without a chat template.
            raise ValueError(
                "Tokenizer has no chat_template. Pass "
                "data.chat_template: chatml (or llama3/qwen2.5/mistral/"
                "gemma3/phi4/deepseek-r1) in soup.yaml, or supply a raw "
                "Jinja string. The previous silent f-string fallback "
                "produced wrong loss labels and was removed in v0.36.0."
            )
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return format_row
