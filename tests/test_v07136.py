"""v0.71.36 — Data Moat II: semantic layer + canaries + replay."""

from __future__ import annotations

import re

import pytest

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(text: str) -> str:
    """Strip ANSI + collapse whitespace.

    Rich splits flag names with color codes and wraps at terminal width;
    asserting on raw output has broken CI three times (v0.71.26/32/35).
    """
    return " ".join(_ANSI_RE.sub("", text).split())


class TestResolvePooling:
    def test_allowlisted_model_short_circuits(self, monkeypatch):
        from soup_cli.utils import embed

        def _no_network(mid):
            raise AssertionError("should not reach network")

        monkeypatch.setattr(embed, "_fetch_pooling_config", _no_network)

        assert embed.resolve_pooling("sentence-transformers/all-MiniLM-L6-v2") == "mean"
        assert embed.resolve_pooling("sentence-transformers/all-mpnet-base-v2") == "mean"

    def test_allowlist_is_case_insensitive(self, monkeypatch):
        from soup_cli.utils import embed

        def _no_network(mid):
            raise AssertionError("should not reach network")

        monkeypatch.setattr(embed, "_fetch_pooling_config", _no_network)

        assert embed.resolve_pooling("Sentence-Transformers/All-MiniLM-L6-v2") == "mean"

    def test_cls_pooling_config_is_refused(self, monkeypatch):
        from soup_cli.utils import embed

        monkeypatch.setattr(
            embed,
            "_fetch_pooling_config",
            lambda mid: {
                "pooling_mode_mean_tokens": False,
                "pooling_mode_cls_token": True,
            },
        )
        with pytest.raises(ValueError, match="cls"):
            embed.resolve_pooling("org/some-encoder")

    def test_cls_precedence_over_contradictory_mean_flag(self, monkeypatch):
        """_NON_MEAN_POOLING_KEYS is checked BEFORE pooling_mode_mean_tokens.

        A self-contradictory config that sets both the cls flag AND the mean
        flag true must still be refused — and refused for being cls, not
        silently accepted because the mean flag also happens to be true.
        Pins the check ordering so a future refactor that collapses this
        into one flat check can't ship silently.
        """
        from soup_cli.utils import embed

        monkeypatch.setattr(
            embed,
            "_fetch_pooling_config",
            lambda mid: {
                "pooling_mode_mean_tokens": True,
                "pooling_mode_cls_token": True,
            },
        )
        with pytest.raises(ValueError, match="cls"):
            embed.resolve_pooling("org/contradictory-encoder")

    def test_mean_pooling_config_is_accepted(self, monkeypatch):
        from soup_cli.utils import embed

        monkeypatch.setattr(
            embed,
            "_fetch_pooling_config",
            lambda mid: {
                "pooling_mode_mean_tokens": True,
                "pooling_mode_cls_token": False,
            },
        )
        assert embed.resolve_pooling("some/mean-model") == "mean"

    def test_unfetchable_config_is_refused_not_assumed(self, monkeypatch):
        from soup_cli.utils import embed

        monkeypatch.setattr(embed, "_fetch_pooling_config", lambda mid: None)
        with pytest.raises(ValueError, match="cannot verify pooling"):
            embed.resolve_pooling("some/unknown-model")

    def test_refusal_names_what_it_found(self, monkeypatch):
        from soup_cli.utils import embed

        monkeypatch.setattr(
            embed,
            "_fetch_pooling_config",
            lambda mid: {
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": True,
            },
        )
        with pytest.raises(ValueError) as exc:
            embed.resolve_pooling("org/other-encoder")
        assert "max" in str(exc.value)

    @pytest.mark.parametrize("bad", ["", "   ", None, 123, True])
    def test_bad_model_id_rejected(self, bad):
        from soup_cli.utils.embed import resolve_pooling

        with pytest.raises((ValueError, TypeError)):
            resolve_pooling(bad)
