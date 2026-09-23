"""Laya-backed reward function for GRPO and PPO.

The optional ``laya`` dependency is imported only when a Laya reward is built.
The adapter deliberately evaluates one completion at a time so malformed model
responses can be reported with their completion index.
"""
from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


class LayaReward:
    """Stateful callable that scores completions with one Laya question."""

    def __init__(
        self, agent: Any, question: Mapping[str, Any], question_id: str, kind: str
    ) -> None:
        self.agent = agent
        self.question = copy.deepcopy(dict(question))
        self.question_id = question_id
        self.kind = kind
        self.__name__ = "laya_reward"

    def __call__(self, completions: Any, **kwargs: Any) -> list[float]:
        if completions is None:
            return []
        if isinstance(completions, (str, bytes)):
            raise TypeError("Laya reward completions must be a sequence, not a string")
        try:
            completion_list = list(completions)
        except TypeError as exc:
            raise TypeError("Laya reward completions must be a sequence") from exc
        rewards: list[float] = []
        for index, completion in enumerate(completion_list):
            try:
                completion_text = _text_from_value(completion, "completion", index)
                prompt_value = _prompt_for_index(kwargs, index, len(completion_list))
                prompt_text = (
                    _text_from_value(prompt_value, "prompt", index)
                    if prompt_value is not None
                    else None
                )
                state: Any = completion_text
                if prompt_text is not None:
                    state = {"prompt": prompt_text, "completion": completion_text}
                result = self.agent.predict(state, {self.question_id: copy.deepcopy(self.question)})
                rewards.append(_extract_reward(result, self.question_id, self.kind, index))
            except Exception as exc:
                if isinstance(exc, (TypeError, ValueError, RuntimeError)) and str(exc).startswith(
                    "Laya reward"
                ):
                    raise
                raise RuntimeError(
                    f"Laya reward prediction failed for completion index {index} "
                    f"and question {self.question_id!r}: {exc}"
                ) from exc
        return rewards


def build_laya_reward_fn(tcfg: Any, device: str) -> LayaReward:
    """Load Laya once and return a validated reward callable."""
    config = getattr(tcfg, "laya_reward", None)
    if config is None:
        raise ValueError("build_laya_reward_fn called without laya_reward configuration")
    checkpoint = config.checkpoint
    if os.path.exists(checkpoint):
        from soup_cli.utils.paths import is_under_cwd

        if not is_under_cwd(checkpoint):
            raise ValueError(
                "laya_reward checkpoint path must stay under the current working "
                f"directory; got {checkpoint!r}"
            )
        checkpoint = os.path.realpath(checkpoint)
    try:
        import laya
    except ImportError as exc:
        raise ImportError(
            "Laya reward requires the optional dependency; install it with "
            '`pip install "soup-cli[laya]"`.'
        ) from exc

    load_kwargs: dict[str, Any] = {"device": str(device)}
    if config.subfolder is not None:
        load_kwargs["subfolder"] = config.subfolder
    try:
        agent = laya.load(checkpoint, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Laya reward failed to load checkpoint {checkpoint!r}: {exc}"
        ) from exc
    question = config.question.model_dump(exclude_none=True)
    return LayaReward(agent, question, config.question.id, config.question.type)


def _prompt_for_index(kwargs: Mapping[str, Any], index: int, count: int) -> Any:
    """Select a prompt from TRL's singular/plural keyword variants."""
    if "prompts" in kwargs:
        prompts = kwargs["prompts"]
        if isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
            if index >= len(prompts):
                return None
            return prompts[index]
        raise TypeError("Laya reward prompt metadata 'prompts' must be a sequence")
    if "prompt" not in kwargs:
        return None
    prompt = kwargs["prompt"]
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        if prompt and all(isinstance(item, Mapping) for item in prompt):
            return prompt
        if len(prompt) == count:
            return prompt[index]
    return prompt


def _text_from_value(value: Any, label: str, index: int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"Laya reward {label} at completion index {index} is empty")
        parts: list[str] = []
        for message_index, message in enumerate(value):
            if not isinstance(message, Mapping) or "content" not in message:
                raise ValueError(
                    f"Laya reward {label} at completion index {index} has malformed "
                    f"message {message_index}; expected a mapping with content"
                )
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError(
                    f"Laya reward {label} at completion index {index} message "
                    f"{message_index} content must be a string"
                )
            parts.append(content)
        return "\n".join(parts)
    raise TypeError(
        f"Laya reward {label} at completion index {index} has unsupported shape "
        f"{type(value).__name__}"
    )


def _extract_reward(result: Any, question_id: str, kind: str, index: int) -> float:
    if not isinstance(result, Mapping) or not isinstance(result.get("answers"), Mapping):
        raise ValueError(
            f"Laya reward response at completion index {index} has no mapping 'answers'"
        )
    answer = result["answers"].get(question_id)
    if not isinstance(answer, Mapping):
        raise ValueError(
            f"Laya reward response at completion index {index} has no mapping answer "
            f"for question {question_id!r}"
        )
    primitive = answer.get(kind)
    if isinstance(primitive, bool) or not isinstance(primitive, Real):
        raise ValueError(
            f"Laya reward answer {question_id!r} at completion index {index} "
            f"must contain numeric {kind!r}, got {primitive!r}"
        )
    value = float(primitive)
    if not math.isfinite(value):
        raise ValueError(
            f"Laya reward answer {question_id!r} at completion index {index} "
            f"contains non-finite {kind!r} value {primitive!r}"
        )
    return value


__all__ = ["LayaReward", "build_laya_reward_fn"]
