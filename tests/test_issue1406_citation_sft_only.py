"""Citation span weighting is implemented by the SFT RAFT route only."""

import pytest

from soup_cli.config.schema import SoupConfig


def test_pretrain_citation_weighting_is_refused() -> None:
    with pytest.raises(ValueError) as exc_info:
        SoupConfig(
            base="test/model", task="pretrain", data={"train": "raft.jsonl", "format": "raft"},
            training={"citation_faithful": True},
        )
    message = str(exc_info.value)
    assert "training.citation_faithful" in message
    assert "pretrain" in message
    assert "task: sft" in message
    assert "data.format: raft" in message


def test_sft_raft_still_accepts_citation_weighting() -> None:
    cfg = SoupConfig(
        base="test/model", task="sft", data={"train": "raft.jsonl", "format": "raft"},
        training={"citation_faithful": True, "citation_style": "bracket"},
    )
    assert cfg.training.citation_faithful is True


def test_pretrain_without_citation_weighting_is_unchanged() -> None:
    cfg = SoupConfig(base="test/model", task="pretrain", data={"train": "d.jsonl"})
    assert cfg.training.citation_faithful is False
