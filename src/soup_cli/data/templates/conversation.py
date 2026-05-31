"""Multi-turn conversation template for synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a conversational training data generator. Generate diverse, natural "
    "multi-turn dialogues between a user and an AI assistant."
)


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    turns: int = 4,
    topic: str = "general knowledge",
) -> str:
    """Build the generation prompt for multi-turn conversations.

    Args:
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        format_spec: Format specification string.
        turns: Number of conversation turns (2-10).
        topic: Conversation topic.

    Returns:
        Complete generation prompt string.
    """
    turns = max(2, min(10, turns))

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality multi-turn conversations.\n\n"
        f"Topic: {topic}\n"
        f"Turns per conversation: {turns} (user and assistant messages)\n\n"
        f"Each conversation should feel natural with follow-up questions "
        f"and contextual responses.\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
