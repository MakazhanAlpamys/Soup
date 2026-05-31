"""Tool-calling / function-calling instruction template for synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a tool-calling instruction data generator. Generate diverse, "
    "high-quality function-calling training examples. Each example must include "
    "the available tools, a user query, and a correct function call with "
    "valid JSON-encoded arguments."
)

TEMPLATE_SPEC = {
    "domains": {
        "weather": [
            "get_weather(city, units)",
            "get_forecast(city, days)",
            "get_air_quality(city)",
        ],
        "search": [
            "web_search(query, limit)",
            "search_images(query, limit)",
            "search_news(query, timeframe)",
        ],
        "database": [
            "query_users(filter, limit)",
            "get_user_by_id(user_id)",
            "count_records(table, filter)",
        ],
        "filesystem": [
            "read_file(path)",
            "list_dir(path)",
            "file_exists(path)",
        ],
    },
}


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    domain: str = "weather",
    num_turns: int = 1,
) -> str:
    """Build the generation prompt for tool-calling training examples.

    Args:
        count: Number of examples to generate.
        fmt: Output format (typically ``"tool-calling"``).
        format_spec: Format specification string embedded in prompt.
        domain: Tool domain (weather, search, database, filesystem).
        num_turns: Conversation turns per example (default 1).

    Returns:
        Complete generation prompt string.
    """
    tools = TEMPLATE_SPEC["domains"].get(domain, TEMPLATE_SPEC["domains"]["weather"])
    tool_list = "\n".join(f"  - {t}" for t in tools)

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality tool-calling examples.\n\n"
        f"Domain: {domain}\n"
        f"Available tools:\n{tool_list}\n\n"
        f"Each example must contain: a user question requiring a tool, "
        f"the tool schema, and a correct function call with JSON-encoded arguments.\n"
        f"Use {num_turns} turn(s) per example.\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
