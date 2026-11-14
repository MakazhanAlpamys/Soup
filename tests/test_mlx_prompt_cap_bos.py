"""Tests for BOS preservation on MLX ORPO and SimPO prompt cuts (#2567).

Under ``keep_end``, an overlong prompt starting with the tokenizer's BOS must keep
it in front, followed by the last ``max_prompt_length - 1`` tokens, matching
the transformers backend behavior from #2545. A prompt without BOS or under
``keep_start`` remains unchanged.
"""

from __future__ import annotations

import pytest

from soup_cli.trainer.mlx_orpo import build_orpo_example

BOS, EOS = 1, 2


class MockTokenizer:
    def __init__(self, has_bos: bool = True) -> None:
        self.bos_token_id = BOS if has_bos else None
        self.eos_token_id = EOS
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [3 + (ord(c) % 190) for c in text]
        if add_special_tokens and self.bos_token_id is not None:
            return [self.bos_token_id] + ids
        return ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **_: object,
    ) -> list[int] | str:
        parts = [f"<{m['role']}>{m['content']}|" for m in messages]
        if continue_final_message:
            parts[-1] = parts[-1][:-1]
        text = "".join(parts) + ("<assistant>" if add_generation_prompt else "")
        return self.encode(text, add_special_tokens=False) if tokenize else text


def _build(
    row: dict,
    tokenizer: MockTokenizer | None = None,
    max_prompt_length: int = 6,
    mode: str = "keep_end",
    max_length: int = 64,
):
    tok = tokenizer or MockTokenizer()
    return build_orpo_example(
        row,
        tok,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        truncation_mode=mode,
    )


def test_overlong_prompt_keeps_leading_bos_under_keep_end():
    # Prompt is 10 chars + BOS = 11 tokens; budget is 6 tokens
    row = {"prompt": "0123456789", "chosen": " yes", "rejected": " no"}
    ex = _build(row, max_prompt_length=6, mode="keep_end", max_length=12)
    prompt_ids = ex.chosen_ids[: ex.chosen_prompt_len]
    assert len(prompt_ids) == 6
    assert prompt_ids[0] == BOS
    # The last 5 tokens of "0123456789" (i.e. '5', '6', '7', '8', '9')
    expected_tail = [3 + (ord(c) % 190) for c in "56789"]
    assert prompt_ids[1:] == expected_tail
    assert ex.rejected_ids[: ex.rejected_prompt_len] == prompt_ids


def test_overlong_prompt_without_bos_drops_start_under_keep_end():
    tok = MockTokenizer(has_bos=False)
    row = {"prompt": "0123456789", "chosen": " yes", "rejected": " no"}
    ex = _build(row, tokenizer=tok, max_prompt_length=6, mode="keep_end", max_length=12)
    prompt_ids = ex.chosen_ids[: ex.chosen_prompt_len]
    assert len(prompt_ids) == 6
    assert BOS not in prompt_ids
    expected = [3 + (ord(c) % 190) for c in "456789"]
    assert prompt_ids == expected


def test_overlong_prompt_with_keep_start_keeps_start():
    row = {"prompt": "0123456789", "chosen": " yes", "rejected": " no"}
    ex = _build(row, max_prompt_length=6, mode="keep_start", max_length=12)
    prompt_ids = ex.chosen_ids[: ex.chosen_prompt_len]
    assert len(prompt_ids) == 6
    assert prompt_ids[0] == BOS
    expected_start = [3 + (ord(c) % 190) for c in "01234"]
    assert prompt_ids[1:] == expected_start


def test_prompt_that_fits_is_untouched():
    row = {"prompt": "hi", "chosen": " yes", "rejected": " no"}
    ex = _build(row, max_prompt_length=10, mode="keep_end")
    prompt_ids = ex.chosen_ids[: ex.chosen_prompt_len]
    assert prompt_ids[0] == BOS
    assert len(prompt_ids) == 1 + len("hi")

def test_a_simpo_batch_on_mlx_starts_every_row_with_the_bos():
    pytest.importorskip("mlx.core")
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.mlx_simpo import MLXSimPOTrainerWrapper
    from tests.test_mlx_simpo import _SIMPO_YAML

    wrapper = MLXSimPOTrainerWrapper(load_config_from_string(_SIMPO_YAML))
    wrapper.config.data.max_length = 12
    wrapper.tokenizer = MockTokenizer()
    (example,), _ = wrapper._examples(
        [{"prompt": "0123456789", "chosen": " yes", "rejected": " no"}]
    )
    tokens, _completion = wrapper._batch_arrays([example])
    assert tokens[:, 0].tolist() == [BOS, BOS]
    assert tokens[0, 1:6].tolist() == [3 + (ord(c) % 190) for c in "56789"]

