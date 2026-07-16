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


class TestEmbedTexts:
    def test_rejects_over_cap(self):
        from soup_cli.utils.embed import _MAX_ROWS, embed_texts

        with pytest.raises(ValueError, match="too many texts"):
            embed_texts(["x"] * (_MAX_ROWS + 1))

    def test_rejects_empty_list(self):
        from soup_cli.utils.embed import embed_texts

        with pytest.raises(ValueError, match="at least one text"):
            embed_texts([])

    def test_rejects_non_string_row(self):
        from soup_cli.utils.embed import embed_texts

        with pytest.raises(TypeError, match=r"texts\[1\] must be str"):
            embed_texts(["ok", 42])

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_rejects_bad_batch_size(self, bad):
        from soup_cli.utils.embed import embed_texts

        with pytest.raises((ValueError, TypeError)):
            embed_texts(["x"], batch_size=bad)

    def test_validate_texts_truncates_long_rows(self):
        from soup_cli.utils.embed import _MAX_CHARS_PER_ROW, _validate_texts

        out = _validate_texts(["a" * (_MAX_CHARS_PER_ROW + 500)])
        assert len(out[0]) == _MAX_CHARS_PER_ROW

    def test_validate_texts_rejects_bare_string(self):
        """A bare str is a sequence of chars — almost always a caller bug."""
        from soup_cli.utils.embed import _validate_texts

        with pytest.raises(TypeError, match="sequence of str"):
            _validate_texts("not a list")

    def test_mean_pool_masks_padding(self):
        """Padded positions must NOT contribute to the mean."""
        import numpy as np

        from soup_cli.utils.embed import _mean_pool

        torch = pytest.importorskip("torch")
        # 3 tokens; the last is padding carrying a huge value that would
        # wreck an unmasked mean (999 -> ~334 instead of 2.0).
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [999.0, 999.0]]])
        mask = torch.tensor([[1, 1, 0]])
        out = _mean_pool(hidden, mask).numpy()
        np.testing.assert_allclose(out, np.array([[2.0, 2.0]]), rtol=1e-6)

    def test_mean_pool_all_padding_does_not_divide_by_zero(self):
        import numpy as np

        from soup_cli.utils.embed import _mean_pool

        torch = pytest.importorskip("torch")
        hidden = torch.tensor([[[5.0, 5.0]]])
        mask = torch.tensor([[0]])
        out = _mean_pool(hidden, mask).numpy()
        assert np.isfinite(out).all()

    def test_l2_normalize_rows(self):
        import numpy as np

        from soup_cli.utils.embed import _l2_normalize

        vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        out = _l2_normalize(vecs)
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, rtol=1e-6)
        # a zero vector must not become NaN
        assert np.isfinite(out[1]).all()

    def test_l2_normalize_makes_cosine_a_dot_product(self):
        import numpy as np

        from soup_cli.utils.embed import _l2_normalize

        vecs = _l2_normalize(np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float32))
        assert float(vecs[0] @ vecs[1]) == pytest.approx(0.0, abs=1e-6)
        assert float(vecs[0] @ vecs[0]) == pytest.approx(1.0, abs=1e-6)

    def test_refuses_unverified_model_before_any_download(self, monkeypatch):
        """resolve_pooling must gate embed_texts BEFORE a model is fetched."""
        from soup_cli.utils import embed

        monkeypatch.setattr(embed, "_fetch_pooling_config", lambda mid: None)
        with pytest.raises(ValueError, match="cannot verify pooling"):
            embed.embed_texts(["hello"], model_id="org/unverified-encoder")


class TestGreedySemdedup:
    def _vecs(self, rows):
        import numpy as np

        arr = np.array(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms < 1e-12, 1.0, norms)

    def test_identical_vectors_collapse_to_one(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        vecs = self._vecs([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert rep.kept == (0,)
        assert rep.dropped == (1, 2)

    def test_orthogonal_vectors_all_kept(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        vecs = self._vecs([[1.0, 0.0], [0.0, 1.0]])
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert rep.kept == (0, 1)
        assert rep.dropped == ()

    def _exact_half_cosine(self):
        """Two UNIT vectors whose cosine is EXACTLY 0.5 in float32.

        The obvious construction — arccos(0.8) then cos/sin — yields
        0.800000011920929, i.e. strictly ABOVE the threshold, so `>=` and
        `>` behave identically and the boundary is never actually tested.
        Here x=0.5 is dyadic (exact in binary) and the first vector's y is
        exactly 0.0, so the dot product is exactly 0.5 with no rounding.
        """
        import numpy as np

        return np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=np.float32)

    def test_boundary_fixture_is_exact(self):
        """Guard the guard: if this drifts, the boundary tests go vacuous."""
        import numpy as np

        vecs = self._exact_half_cosine()
        assert float(vecs[0] @ vecs[1]) == 0.5
        assert float(np.linalg.norm(vecs[1])) == pytest.approx(1.0, abs=1e-6)

    def test_threshold_boundary_is_inclusive(self):
        """cosine == threshold must DROP (>=), not keep."""
        from soup_cli.utils.semdedup import greedy_semdedup

        rep = greedy_semdedup(self._exact_half_cosine(), threshold=0.5)
        assert rep.dropped == (1,), "cosine == threshold must drop"

    def test_just_below_threshold_is_kept(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        rep = greedy_semdedup(self._exact_half_cosine(), threshold=0.51)
        assert rep.dropped == ()

    def test_pairs_record_which_kept_row_and_cosine(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        vecs = self._vecs([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert len(rep.pairs) == 1
        dropped_idx, kept_idx, cosine = rep.pairs[0]
        assert (dropped_idx, kept_idx) == (2, 0)
        assert cosine == pytest.approx(1.0, abs=1e-5)

    def test_pairs_name_the_nearest_kept_row_not_just_any(self):
        """The provenance must identify WHICH kept row it collided with.

        Row 2 is near row 1 and far from row 0; a report that blindly cited
        kept[0] would pass a weaker test.
        """
        import numpy as np

        from soup_cli.utils.semdedup import greedy_semdedup

        theta = float(np.arccos(0.95))
        vecs = self._vecs([
            [1.0, 0.0],
            [0.0, 1.0],
            [float(np.sin(theta)), float(np.cos(theta))],
        ])
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert rep.pairs[0][0] == 2
        assert rep.pairs[0][1] == 1, "must cite the NEAREST kept row"

    def test_transitive_chain_keeps_first_only(self):
        """A~B, B~C but A!~C: greedy keeps A, drops B; C compared to A only."""
        import numpy as np

        from soup_cli.utils.semdedup import greedy_semdedup

        t1 = float(np.arccos(0.95))
        vecs = self._vecs([
            [1.0, 0.0],
            [float(np.cos(t1)), float(np.sin(t1))],
            [float(np.cos(2 * t1)), float(np.sin(2 * t1))],
        ])
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert rep.kept == (0, 2)
        assert rep.dropped == (1,)

    def test_single_row(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        rep = greedy_semdedup(self._vecs([[1.0, 0.0]]), threshold=0.9)
        assert rep.kept == (0,)

    def test_empty_input(self):
        import numpy as np

        from soup_cli.utils.semdedup import greedy_semdedup

        rep = greedy_semdedup(
            np.zeros((0, 2), dtype=np.float32), threshold=0.9
        )
        assert rep.kept == ()
        assert rep.dropped == ()

    def test_rejects_non_2d(self):
        import numpy as np

        from soup_cli.utils.semdedup import greedy_semdedup

        with pytest.raises(ValueError, match="2-D"):
            greedy_semdedup(np.zeros((5,), dtype=np.float32), threshold=0.9)

    def test_rejects_over_cap(self):
        import numpy as np

        from soup_cli.utils.semdedup import _MAX_SEMDEDUP_ROWS, greedy_semdedup

        vecs = np.zeros((_MAX_SEMDEDUP_ROWS + 1, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="too many rows"):
            greedy_semdedup(vecs, threshold=0.9)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "x"])
    def test_rejects_bad_threshold(self, bad):
        from soup_cli.utils.semdedup import greedy_semdedup

        with pytest.raises((ValueError, TypeError)):
            greedy_semdedup(self._vecs([[1.0, 0.0]]), threshold=bad)

    def test_kept_and_dropped_partition_the_input(self):
        from soup_cli.utils.semdedup import greedy_semdedup

        vecs = self._vecs(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]
        )
        rep = greedy_semdedup(vecs, threshold=0.9)
        assert sorted(rep.kept + rep.dropped) == list(range(5))
        assert not set(rep.kept) & set(rep.dropped)

    def test_report_is_frozen(self):
        import dataclasses

        from soup_cli.utils.semdedup import DedupReport

        rep = DedupReport(kept=(0,), dropped=(), pairs=(), threshold=0.9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rep.threshold = 0.5
