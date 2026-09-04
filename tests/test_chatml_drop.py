"""#676 — chatml and video converters honour the drop contract."""

from soup_cli.data.formats import format_to_messages


def test_chatml_non_dict_message_returns_none():
    row = {"messages": ["hello"]}
    result = format_to_messages(row, "chatml")
    assert result is None


def test_chatml_valid_messages_still_convert():
    row = {"messages": [{"role": "user", "content": "hi"}]}
    result = format_to_messages(row, "chatml")
    assert result == {"messages": [{"role": "user", "content": "hi"}]}


def test_chatml_messages_not_a_list_returns_none():
    result = format_to_messages({"messages": "hello"}, "chatml")
    assert result is None


def test_video_non_dict_message_returns_none():
    row = {"video": "clip.mp4", "messages": ["hello"]}
    result = format_to_messages(row, "video")
    assert result is None


def test_video_valid_messages_still_convert():
    row = {"video": "clip.mp4", "messages": [{"role": "user", "content": "hi"}]}
    result = format_to_messages(row, "video")
    assert result == {
        "video": "clip.mp4",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_video_messages_not_a_list_returns_none():
    result = format_to_messages({"video": "clip.mp4", "messages": 5}, "video")
    assert result is None
