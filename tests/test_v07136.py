"""v0.71.36 — Data Moat II: semantic layer + canaries + replay."""

from __future__ import annotations

import json
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


class TestExtrasHintsAreEscaped:
    """Rich eats `[extra]` as a markup tag unless it is escaped.

    Unescaped, `pip install 'soup-cli[eval]'` renders as
    `pip install 'soup-cli'` -- which installs the base package WITHOUT the
    extra the user is missing, so following the hint appears to succeed and
    the feature still fails. Found during v0.71.36; same class as the
    v0.71.28 \\[mcp] fix, which only fixed its own site.

    Sites that do NOT go through Rich (plain exception text, docstrings)
    must NOT be escaped -- a backslash would show up literally.
    """

    # Rich style tags. A hint string carrying one of these is console.print
    # output; a bare `raise ImportError("... soup-cli[mlx]")` is NOT and must
    # be left alone (escaping it would print a literal backslash).
    _RICH_TAGS = ("[/]", "[bold]", "[red]", "[yellow]", "[green]", "[dim]")

    def test_rich_renders_unescaped_bracket_away(self):
        """Pin the underlying Rich behaviour this whole class exists for."""
        from io import StringIO

        from rich.console import Console

        def render(markup):
            buf = StringIO()
            Console(file=buf, force_terminal=False, width=100).print(markup)
            return buf.getvalue().strip()

        assert render("[bold]pip install 'soup-cli[train]'[/]") == (
            "pip install 'soup-cli'"
        ), "unescaped bracket must be eaten (this is the bug)"
        assert render("[bold]pip install 'soup-cli\\[train]'[/]") == (
            "pip install 'soup-cli[train]'"
        ), "escaped bracket must survive (this is the fix)"

    def test_no_unescaped_extras_hint_in_rich_markup(self):
        """Every hint carrying Rich markup must escape its bracket."""
        import pathlib
        import re

        import soup_cli

        root = pathlib.Path(soup_cli.__file__).parent
        bad_re = re.compile(r"(?<!\\)soup-cli\[[a-z][a-z0-9-]*\]")
        offenders = []
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.strip().startswith("#"):
                    continue  # a code comment never reaches Rich
                if not any(tag in line for tag in self._RICH_TAGS):
                    continue  # not Rich output — must NOT be escaped
                if bad_re.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "unescaped soup-cli[extra] inside Rich markup — the bracket is "
            "eaten and the printed command installs WITHOUT the extra:\n"
            + "\n".join(offenders)
        )

    @pytest.mark.parametrize(
        "argv,extra",
        [
            (["data", "dedup", "--help"], "soup-cli[train]"),
            (["train", "--help"], "soup-cli[carbon]"),
        ],
    )
    def test_typer_help_keeps_the_extra_bracket(self, argv, extra):
        """Typer help is Rich-rendered too — the bracket must survive.

        `--semantic` help shipped as "Requires soup-cli." before this fix.
        """
        from typer.testing import CliRunner

        from soup_cli.cli import app

        res = CliRunner(env={"COLUMNS": "200"}).invoke(app, argv)
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert extra in _clean(res.output), (
            f"{extra!r} missing from `{' '.join(argv)}` — Rich ate the bracket"
        )

    def test_plain_exception_hints_are_not_escaped(self):
        """The counter-rule: non-Rich text must NOT gain a backslash.

        `raise ImportError("... pip install 'soup-cli[mlx]'")` never reaches
        Rich, so escaping it would surface a literal backslash to the user.
        """
        import pathlib

        import soup_cli

        root = pathlib.Path(soup_cli.__file__).parent
        text = (root / "trainer" / "mlx_sft.py").read_text(encoding="utf-8")
        assert "'soup-cli[mlx]'" in text
        assert "soup-cli\\[mlx]" not in text


class TestDedupSemanticCli:
    def _write(self, tmp_path, rows):
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        return path

    def test_help_mentions_semantic(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        res = CliRunner().invoke(app, ["data", "dedup", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "semantic" in _clean(res.output)

    def test_semantic_dedups_via_injected_embedder(self, tmp_path, monkeypatch):
        """CLI wiring tested without a model download."""
        import numpy as np
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        rows = [
            {"text": "the cat sat on the mat"},
            {"text": "a feline rested upon the rug"},
            {"text": "quantum chromodynamics is hard"},
        ]
        path = self._write(tmp_path, rows)
        # rows 0 and 1 are "paraphrases" -> near-identical vectors
        fake = np.array(
            [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]], dtype=np.float32
        )
        monkeypatch.setattr(data_cmd, "embed_texts", lambda *a, **k: fake)
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app,
            ["data", "dedup", str(path), "--semantic",
             "--threshold", "0.9", "-o", "out.jsonl"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        kept = [
            json.loads(ln)
            for ln in (tmp_path / "out.jsonl").read_text().splitlines() if ln
        ]
        assert len(kept) == 2
        assert kept[0]["text"] == "the cat sat on the mat"
        assert kept[1]["text"] == "quantum chromodynamics is hard"

    def test_semantic_does_not_require_datasketch(self, tmp_path, monkeypatch):
        """--semantic must work for a user with [train] but NOT [data].

        The datasketch import used to sit at the TOP of dedup(), before the
        file check, so the semantic path would have died with 'datasketch
        not installed' despite never needing MinHash.
        """
        import builtins

        import numpy as np
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        real_import = builtins.__import__

        def _no_datasketch(name, *args, **kwargs):
            if name == "datasketch":
                raise ImportError("No module named 'datasketch'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_datasketch)
        monkeypatch.setattr(
            data_cmd, "embed_texts",
            lambda *a, **k: np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
            ),
        )
        path = self._write(tmp_path, [{"text": "a"}, {"text": "b"}])
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "dedup", str(path), "--semantic", "-o", "out.jsonl"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "datasketch" not in _clean(res.output)

    def test_semantic_import_error_is_friendly(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        path = self._write(tmp_path, [{"text": "a"}])

        def _boom(*a, **k):
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr(data_cmd, "embed_texts", _boom)
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "dedup", str(path), "--semantic"]
        )
        assert res.exit_code == 1
        assert "soup-cli[train]" in _clean(res.output)

    def test_semantic_pooling_refusal_surfaces(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        path = self._write(tmp_path, [{"text": "a"}])

        def _refuse(*a, **k):
            raise ValueError("cannot verify pooling for 'x/y'")

        monkeypatch.setattr(data_cmd, "embed_texts", _refuse)
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "dedup", str(path), "--semantic"]
        )
        assert res.exit_code == 1
        assert "pooling" in _clean(res.output)

    def test_minhash_path_untouched_without_flag(self, tmp_path, monkeypatch):
        """--semantic absent -> embed_texts must never be called."""
        pytest.importorskip("datasketch")
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        called = []
        monkeypatch.setattr(
            data_cmd, "embed_texts",
            lambda *a, **k: called.append(1) or None,
        )
        path = self._write(
            tmp_path, [{"text": "a b c"}, {"text": "x y z"}]
        )
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "dedup", str(path)])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert called == [], "MinHash path must not touch the embedder"

    def test_field_selects_what_is_embedded(self, tmp_path, monkeypatch):
        """--field composes with --semantic (same meaning as for MinHash)."""
        import numpy as np
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data as data_cmd

        seen = {}

        def _capture(texts, **kwargs):
            seen["texts"] = list(texts)
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        monkeypatch.setattr(data_cmd, "embed_texts", _capture)
        rows = [
            {"text": "keep me", "noise": "IGNORED"},
            {"text": "other", "noise": "IGNORED"},
        ]
        path = self._write(tmp_path, rows)
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app,
            ["data", "dedup", str(path), "--semantic",
             "--field", "text", "-o", "out.jsonl"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert seen["texts"] == ["keep me", "other"]
        assert not any("IGNORED" in t for t in seen["texts"])

    def test_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        path = self._write(tmp_path, [{"text": "a"}])
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        res = CliRunner().invoke(
            app,
            ["data", "dedup", str(path), "--semantic",
             "-o", str(tmp_path / "escape.jsonl")],
        )
        assert res.exit_code == 1
        assert "outside" in _clean(res.output).lower()


class TestResolveK:
    @pytest.mark.parametrize(
        "n_rows,expected",
        [(1, 1), (3, 1), (4, 2), (8, 2), (200, 10), (1000, 22), (100_000, 25)],
    )
    def test_auto_heuristic(self, n_rows, expected):
        from soup_cli.utils.topics import resolve_k

        assert resolve_k(n_rows, "auto") == expected

    def test_auto_is_capped_so_the_table_stays_one_screen(self):
        from soup_cli.utils.topics import _MAX_K, resolve_k

        assert resolve_k(10_000_000, "auto") == _MAX_K

    def test_explicit_k_used_as_is(self):
        from soup_cli.utils.topics import resolve_k

        assert resolve_k(1000, 7) == 7

    def test_explicit_k_is_not_capped_by_max_k(self):
        """--clusters is the operator's call; only 'auto' is capped."""
        from soup_cli.utils.topics import _MAX_K, resolve_k

        assert resolve_k(1000, _MAX_K + 5) == _MAX_K + 5

    def test_explicit_k_clamped_to_n_rows(self):
        from soup_cli.utils.topics import resolve_k

        assert resolve_k(3, 10) == 3

    def test_auto_is_case_and_space_insensitive(self):
        from soup_cli.utils.topics import resolve_k

        assert resolve_k(200, " AUTO ") == 10

    @pytest.mark.parametrize("bad", [0, -1, True, "seven", 1.5])
    def test_rejects_bad_k(self, bad):
        from soup_cli.utils.topics import resolve_k

        with pytest.raises((ValueError, TypeError)):
            resolve_k(100, bad)


class TestKmeans:
    def _blobs(self):
        import numpy as np

        rng = np.random.default_rng(0)
        left = rng.normal([-5.0, 0.0], 0.1, size=(20, 2))
        right = rng.normal([5.0, 0.0], 0.1, size=(20, 2))
        return np.vstack([left, right]).astype(np.float32)

    def test_separates_two_obvious_blobs(self):
        from soup_cli.utils.topics import kmeans

        labels = kmeans(self._blobs(), k=2, seed=0)
        assert len(set(labels[:20].tolist())) == 1
        assert len(set(labels[20:].tolist())) == 1
        assert labels[0] != labels[20]

    def _unstructured(self):
        """Points with NO cluster structure, so k-means++ init decides.

        Two well-separated blobs converge to the same partition from any
        init, which makes the seed untestable on them (verified: 6 distinct
        seeds -> 2 partitions, and an unseeded RNG passes a same-seed
        equality check by luck). Unstructured points give 6 distinct
        partitions for 6 seeds, so ignoring the seed is detectable.
        """
        import numpy as np

        return np.random.default_rng(0).random((60, 4)).astype(np.float32)

    def test_deterministic_by_seed(self):
        import numpy as np

        from soup_cli.utils.topics import kmeans

        vecs = self._unstructured()
        first = kmeans(vecs, k=5, seed=42)
        for _ in range(3):
            np.testing.assert_array_equal(
                first, kmeans(vecs, k=5, seed=42),
                err_msg="same seed must give the same partition",
            )

    def test_seed_actually_influences_the_result(self):
        """Guard the guard: if seeds stopped mattering, the test above
        would pass even with the seed ignored."""
        from soup_cli.utils.topics import kmeans

        vecs = self._unstructured()
        partitions = {
            tuple(kmeans(vecs, k=5, seed=seed).tolist()) for seed in range(6)
        }
        assert len(partitions) > 1, (
            "seeds produce identical partitions on this fixture — it cannot "
            "detect an unseeded RNG"
        )

    def test_k_equals_one(self):
        from soup_cli.utils.topics import kmeans

        labels = kmeans(self._blobs(), k=1, seed=0)
        assert set(labels.tolist()) == {0}

    def test_k_greater_than_n_is_clamped(self):
        import numpy as np

        from soup_cli.utils.topics import kmeans

        vecs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        labels = kmeans(vecs, k=5, seed=0)
        assert len(labels) == 2
        assert set(labels.tolist()) <= {0, 1}

    def test_labels_are_in_range(self):
        from soup_cli.utils.topics import kmeans

        labels = kmeans(self._blobs(), k=3, seed=1)
        assert all(0 <= int(x) < 3 for x in labels)

    def test_empty_input(self):
        import numpy as np

        from soup_cli.utils.topics import kmeans

        labels = kmeans(np.zeros((0, 2), dtype=np.float32), k=2, seed=0)
        assert len(labels) == 0

    def test_rejects_non_2d(self):
        import numpy as np

        from soup_cli.utils.topics import kmeans

        with pytest.raises(ValueError, match="2-D"):
            kmeans(np.zeros((5,), dtype=np.float32), k=2, seed=0)

    def test_rejects_over_cap(self):
        import numpy as np

        from soup_cli.utils.topics import _MAX_TOPIC_ROWS, kmeans

        vecs = np.zeros((_MAX_TOPIC_ROWS + 1, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="too many rows"):
            kmeans(vecs, k=2, seed=0)

    def test_identical_points_do_not_crash(self):
        """All-duplicate input makes every k-means++ distance zero."""
        import numpy as np

        from soup_cli.utils.topics import kmeans

        vecs = np.ones((10, 3), dtype=np.float32)
        labels = kmeans(vecs, k=3, seed=0)
        assert len(labels) == 10


class TestCtfidfLabels:
    def _global_vs_specific(self):
        """5 clusters, each ``["the", "the", "term_i"]``.

        The filler word is deliberately MORE frequent in-cluster than the
        specific term, so plain per-cluster TF picks "the" every time and
        ONLY the inverse-cluster-frequency term can demote it. The obvious
        fixture (specific term more frequent than the filler) is vacuous:
        plain TF already picks the specific term, so deleting the idf
        changes nothing.

        The margin is arithmetic, not luck: the idf ratio at 5 clusters is
        log(1+5/1)/log(1+5/5) = 2.58, and tf("the")/tf(term) = 2.0 < 2.58.
        """
        docs = [["the", "the", f"term{i}"] for i in range(5)]
        return docs, list(range(5))

    def test_cluster_specific_term_beats_global_term(self):
        """The property that makes c-TF-IDF != plain TF-IDF."""
        from soup_cli.utils.topics import ctfidf_labels

        docs, labels = self._global_vs_specific()
        out = ctfidf_labels(docs, labels, k=5, top_n=1)
        assert out == [(f"term{i}",) for i in range(5)], (
            "a term unique to one cluster must outrank a filler word that is "
            "more frequent but present in every cluster"
        )

    def test_the_filler_is_the_plain_tf_winner(self):
        """Guard the guard: prove plain TF really would pick the filler.

        If this ever fails, the fixture stopped discriminating and the test
        above would pass even with the idf term deleted.
        """
        from collections import Counter

        docs, _ = self._global_vs_specific()
        counts = Counter(docs[0])
        assert counts["the"] > counts["term0"]

    def test_top_n_respected(self):
        from soup_cli.utils.topics import ctfidf_labels

        out = ctfidf_labels(
            [["alpha", "beta", "gamma"], ["delta"]], [0, 1], k=2, top_n=2
        )
        assert len(out[0]) <= 2

    def test_empty_cluster_yields_empty_terms(self):
        from soup_cli.utils.topics import ctfidf_labels

        out = ctfidf_labels([["a"]], [0], k=3, top_n=2)
        assert out[1] == ()
        assert out[2] == ()

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.topics import ctfidf_labels

        with pytest.raises(ValueError, match="same length"):
            ctfidf_labels([["a"], ["b"]], [0], k=1, top_n=1)

    def test_out_of_range_label_ignored(self):
        from soup_cli.utils.topics import ctfidf_labels

        out = ctfidf_labels([["a"], ["b"]], [0, 99], k=1, top_n=1)
        assert out[0] == ("a",)

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_rejects_bad_top_n(self, bad):
        from soup_cli.utils.topics import ctfidf_labels

        with pytest.raises((ValueError, TypeError)):
            ctfidf_labels([["a"]], [0], k=1, top_n=bad)


class TestBuildTopicReport:
    def _rows(self, tag, count):
        return [
            {"messages": [{"role": "assistant", "content": f"{tag} {i}"}]}
            for i in range(count)
        ]

    def test_fractions_sum_to_one(self):
        from soup_cli.utils.topics import build_topic_report

        rows = self._rows("python code", 6) + self._rows("protein fold", 4)
        rep = build_topic_report(rows, [0] * 6 + [1] * 4, k=2)
        assert rep.n_rows == 10
        assert sum(t.fraction for t in rep.topics) == pytest.approx(1.0)
        assert {t.size for t in rep.topics} == {6, 4}

    def test_topics_sorted_by_coverage_desc(self):
        from soup_cli.utils.topics import build_topic_report

        rows = self._rows("small", 2) + self._rows("big", 8)
        rep = build_topic_report(rows, [0] * 2 + [1] * 8, k=2)
        assert [t.size for t in rep.topics] == [8, 2]

    def test_small_cluster_raises_gap_warning(self):
        from soup_cli.utils.topics import build_topic_report

        rows = self._rows("code", 99) + self._rows("safety", 1)
        rep = build_topic_report(
            rows, [0] * 99 + [1], k=2, min_fraction=0.02
        )
        assert any("thin" in w.lower() for w in rep.warnings)

    def test_no_warning_when_all_clusters_healthy(self):
        from soup_cli.utils.topics import build_topic_report

        rows = self._rows("code", 5) + self._rows("math", 5)
        rep = build_topic_report(rows, [0] * 5 + [1] * 5, k=2, min_fraction=0.02)
        assert rep.warnings == ()

    def test_member_indices_partition_the_input(self):
        from soup_cli.utils.topics import build_topic_report

        rows = self._rows("a", 3) + self._rows("b", 2)
        rep = build_topic_report(rows, [1, 0, 1, 0, 1], k=2)
        allidx = sorted(sum((list(t.member_indices) for t in rep.topics), []))
        assert allidx == [0, 1, 2, 3, 4]

    def test_zero_rows(self):
        from soup_cli.utils.topics import build_topic_report

        rep = build_topic_report([], [], k=1)
        assert rep.n_rows == 0
        assert rep.topics == ()

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.topics import build_topic_report

        with pytest.raises(ValueError, match="same length"):
            build_topic_report(self._rows("a", 2), [0], k=1)

    def test_report_is_frozen(self):
        import dataclasses

        from soup_cli.utils.topics import build_topic_report

        rep = build_topic_report(self._rows("a", 2), [0, 0], k=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rep.n_rows = 99


class TestDataTopicsCli:
    def _write(self, tmp_path, rows):
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        return path

    def _rows(self):
        rows = [
            {"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": f"python function def {i}"},
            ]}
            for i in range(5)
        ]
        rows += [
            {"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": f"protein enzyme fold {i}"},
            ]}
            for i in range(5)
        ]
        return rows

    def _fake_vecs(self):
        import numpy as np

        return np.vstack([
            np.tile([1.0, 0.0], (5, 1)), np.tile([0.0, 1.0], (5, 1))
        ]).astype(np.float32)

    def test_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        res = CliRunner().invoke(app, ["data", "topics", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "cluster" in _clean(res.output).lower()

    def test_happy_path_writes_report(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        path = self._write(tmp_path, self._rows())
        monkeypatch.setattr(cmd, "embed_texts", lambda *a, **k: self._fake_vecs())
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app,
            ["data", "topics", str(path), "--clusters", "2", "-o", "topics.json"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        data = json.loads((tmp_path / "topics.json").read_text(encoding="utf-8"))
        assert data["n_rows"] == 10
        assert len(data["topics"]) == 2
        assert sum(t["fraction"] for t in data["topics"]) == pytest.approx(1.0)
        # the two synthetic domains must separate
        labels = " ".join(t["label"] for t in data["topics"])
        assert "python" in labels and "protein" in labels

    def test_auto_clusters_runs(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        path = self._write(tmp_path, self._rows())
        monkeypatch.setattr(cmd, "embed_texts", lambda *a, **k: self._fake_vecs())
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "topics", str(path)])
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_bad_clusters_value_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        path = self._write(tmp_path, self._rows())
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "topics", str(path), "--clusters", "banana"]
        )
        assert res.exit_code == 1
        assert "auto" in _clean(res.output).lower()

    def test_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        path = self._write(tmp_path, self._rows())
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        res = CliRunner().invoke(
            app, ["data", "topics", str(path), "-o", str(tmp_path / "esc.json")]
        )
        assert res.exit_code == 1
        assert "outside" in _clean(res.output).lower()

    def test_missing_file(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "topics", "nope.jsonl"])
        assert res.exit_code == 1
        assert "not found" in _clean(res.output).lower()

    def test_empty_dataset(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "topics", str(path)])
        assert res.exit_code == 1
        assert "empty" in _clean(res.output).lower()

    def test_import_error_friendly(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        def _boom(*a, **k):
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr(cmd, "embed_texts", _boom)
        path = self._write(tmp_path, self._rows())
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "topics", str(path)])
        assert res.exit_code == 1
        assert "soup-cli[train]" in _clean(res.output)

    def test_pooling_refusal_surfaces(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        def _refuse(*a, **k):
            raise ValueError("cannot verify pooling for 'x/y'")

        monkeypatch.setattr(cmd, "embed_texts", _refuse)
        path = self._write(tmp_path, self._rows())
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(app, ["data", "topics", str(path)])
        assert res.exit_code == 1
        assert "pooling" in _clean(res.output)

    def test_markup_in_data_is_escaped(self, tmp_path, monkeypatch):
        """A crafted row must not inject Rich markup into the table."""
        import numpy as np
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        rows = [
            {"messages": [{"role": "assistant", "content": "[red]boom[/] alpha"}]}
        ] * 4
        path = self._write(tmp_path, rows)
        monkeypatch.setattr(
            cmd, "embed_texts",
            lambda *a, **k: np.tile([1.0, 0.0], (4, 1)).astype(np.float32),
        )
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "topics", str(path), "--clusters", "1"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_gap_warning_surfaces_in_output(self, tmp_path, monkeypatch):
        import numpy as np
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_topics as cmd

        rows = [
            {"messages": [{"role": "assistant", "content": f"code {i}"}]}
            for i in range(99)
        ]
        rows += [{"messages": [{"role": "assistant", "content": "safety"}]}]
        path = self._write(tmp_path, rows)
        vecs = np.vstack([
            np.tile([1.0, 0.0], (99, 1)), np.tile([0.0, 1.0], (1, 1))
        ]).astype(np.float32)
        monkeypatch.setattr(cmd, "embed_texts", lambda *a, **k: vecs)
        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "topics", str(path), "--clusters", "2"]
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "thin" in _clean(res.output).lower()


class TestComputePairLosses:
    """compute_eval_loss returns only the MEAN and silently compacts the
    list, so a skipped pair shifts every later index. Canary exposure needs
    per-item losses aligned to the input.
    """

    def _fake_torch_env(self, monkeypatch, loss_value=1.5):
        from soup_cli.utils import live_eval

        torch = pytest.importorskip("torch")

        def _fake_tokenize(tokenizer, prompt, target, *, max_length):
            if target == "":
                return torch.tensor([[1]]), torch.tensor([[-100]])
            return torch.tensor([[1, 2]]), torch.tensor([[-100, 2]])

        monkeypatch.setattr(live_eval, "_tokenize_pair", _fake_tokenize)

        class _Model:
            def eval(self):
                return self

            def __call__(self, **kwargs):
                class _Out:
                    loss = torch.tensor(loss_value)

                return _Out()

        return live_eval, _Model()

    def test_index_aligned_with_nan_for_skipped(self, monkeypatch):
        import math

        live_eval, model = self._fake_torch_env(monkeypatch)
        out = live_eval.compute_pair_losses(
            model, object(), [("p", "a"), ("p", ""), ("p", "b")], device="cpu"
        )
        assert len(out) == 3, "must be index-aligned with pairs"
        assert out[0] == pytest.approx(1.5)
        assert math.isnan(out[1]), "unusable pair must be nan, not dropped"
        assert out[2] == pytest.approx(1.5)

    def test_empty_pairs_returns_empty_list(self, monkeypatch):
        live_eval, model = self._fake_torch_env(monkeypatch)
        assert live_eval.compute_pair_losses(
            model, object(), [], device="cpu"
        ) == []

    def test_compute_eval_loss_is_mean_of_non_nan(self, monkeypatch):
        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval, "compute_pair_losses",
            lambda *a, **k: [1.0, float("nan"), 3.0],
        )
        assert live_eval.compute_eval_loss(
            object(), object(), [("p", "a")] * 3, device="cpu"
        ) == pytest.approx(2.0)

    def test_compute_eval_loss_all_nan_returns_nan(self, monkeypatch):
        import math

        from soup_cli.utils import live_eval

        monkeypatch.setattr(
            live_eval, "compute_pair_losses", lambda *a, **k: [float("nan")]
        )
        assert math.isnan(
            live_eval.compute_eval_loss(
                object(), object(), [("p", "a")], device="cpu"
            )
        )

    def test_compute_eval_loss_empty_returns_nan(self, monkeypatch):
        import math

        from soup_cli.utils import live_eval

        monkeypatch.setattr(live_eval, "compute_pair_losses", lambda *a, **k: [])
        assert math.isnan(
            live_eval.compute_eval_loss(object(), object(), [], device="cpu")
        )

    def test_refactor_preserves_mean_behaviour(self, monkeypatch):
        """compute_eval_loss must equal mean(non-nan compute_pair_losses)."""
        live_eval, model = self._fake_torch_env(monkeypatch, loss_value=2.0)
        pairs = [("p", "a"), ("p", ""), ("p", "b")]
        losses = live_eval.compute_pair_losses(
            model, object(), pairs, device="cpu"
        )
        finite = [x for x in losses if x == x]
        assert live_eval.compute_eval_loss(
            model, object(), pairs, device="cpu"
        ) == pytest.approx(sum(finite) / len(finite))

    def test_max_length_still_validated(self, monkeypatch):
        live_eval, model = self._fake_torch_env(monkeypatch)
        with pytest.raises((ValueError, TypeError)):
            live_eval.compute_pair_losses(
                model, object(), [("p", "a")], device="cpu", max_length=0
            )


class TestCanaryGeneration:
    def test_deterministic_by_seed(self):
        from soup_cli.utils.canary import generate_canaries

        assert generate_canaries(count=5, seed=7) == generate_canaries(
            count=5, seed=7
        )

    def test_different_seeds_differ(self):
        from soup_cli.utils.canary import generate_canaries

        assert generate_canaries(count=5, seed=1) != generate_canaries(
            count=5, seed=2
        )

    def test_secrets_unique(self):
        from soup_cli.utils.canary import generate_canaries

        canaries = generate_canaries(count=50, seed=0)
        assert len({c.secret for c in canaries}) == 50

    def test_controls_exclude_inserted(self):
        from soup_cli.utils.canary import generate_canaries, generate_controls

        inserted = generate_canaries(count=10, seed=0)
        controls = generate_controls(
            count=20, seed=0, exclude={c.secret for c in inserted}
        )
        assert not ({c.secret for c in controls} & {c.secret for c in inserted})

    def test_controls_share_carrier_with_canaries(self):
        """Controls must vary ONLY the secret — else the comparison is
        measuring the carrier, not memorization."""
        from soup_cli.utils.canary import generate_canaries, generate_controls

        canary = generate_canaries(count=1, seed=0)[0]
        control = generate_controls(count=1, seed=99, exclude=set())[0]
        assert control.carrier == canary.carrier

    def test_secret_shape_is_from_the_declared_space(self):
        """Controls are only a valid null if they share the secret space."""
        import re

        from soup_cli.utils.canary import generate_canaries, generate_controls

        shape = re.compile(r"^ [0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}$")
        for canary in generate_canaries(count=8, seed=3):
            assert shape.match(canary.secret), canary.secret
        for control in generate_controls(count=8, seed=4, exclude=set()):
            assert shape.match(control.secret), control.secret

    @pytest.mark.parametrize("bad", [0, -1, True, "5", 10_001])
    def test_rejects_bad_count(self, bad):
        from soup_cli.utils.canary import generate_canaries

        with pytest.raises((ValueError, TypeError)):
            generate_canaries(count=bad, seed=0)

    def test_canary_rows_are_messages_format(self):
        from soup_cli.utils.canary import canary_rows, generate_canaries

        rows = canary_rows(generate_canaries(count=2, seed=0))
        assert len(rows) == 2
        for row in rows:
            assert set(row) == {"messages"}
            assert [m["role"] for m in row["messages"]] == ["user", "assistant"]

    def test_canary_is_frozen(self):
        import dataclasses

        from soup_cli.utils.canary import Canary

        canary = Canary(carrier="c", secret="s")
        with pytest.raises(dataclasses.FrozenInstanceError):
            canary.secret = "other"


class TestCanaryManifest:
    def test_roundtrip(self, tmp_path, monkeypatch):
        from soup_cli.utils.canary import (
            generate_canaries,
            load_manifest,
            write_manifest,
        )

        monkeypatch.chdir(tmp_path)
        canaries = generate_canaries(count=3, seed=0)
        write_manifest(canaries, "m.json")
        assert load_manifest("m.json") == canaries

    def test_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.canary import generate_canaries, write_manifest

        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError):
            write_manifest(
                generate_canaries(count=1, seed=0), str(tmp_path / "esc.json")
            )

    def test_malformed_manifest_rejected(self, tmp_path, monkeypatch):
        from pathlib import Path

        from soup_cli.utils.canary import load_manifest

        monkeypatch.chdir(tmp_path)
        Path("bad.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="manifest"):
            load_manifest("bad.json")

    def test_non_list_canaries_rejected(self, tmp_path, monkeypatch):
        from pathlib import Path

        from soup_cli.utils.canary import load_manifest

        monkeypatch.chdir(tmp_path)
        Path("bad.json").write_text('{"canaries": "nope"}', encoding="utf-8")
        with pytest.raises(ValueError, match="manifest"):
            load_manifest("bad.json")

    def test_entry_missing_fields_rejected(self, tmp_path, monkeypatch):
        from pathlib import Path

        from soup_cli.utils.canary import load_manifest

        monkeypatch.chdir(tmp_path)
        Path("bad.json").write_text(
            '{"canaries": [{"carrier": "c"}]}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="carrier"):
            load_manifest("bad.json")

    def test_oversize_manifest_rejected(self, tmp_path, monkeypatch):
        from pathlib import Path

        from soup_cli.utils.canary import _MAX_MANIFEST_BYTES, load_manifest

        monkeypatch.chdir(tmp_path)
        Path("big.json").write_text(
            " " * (_MAX_MANIFEST_BYTES + 10), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="too large"):
            load_manifest("big.json")

    @pytest.mark.skipif(
        not hasattr(__import__("os"), "symlink"), reason="POSIX only"
    )
    def test_symlinked_manifest_rejected(self, tmp_path, monkeypatch):
        import os

        from soup_cli.utils.canary import load_manifest

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text('{"canaries": []}', encoding="utf-8")
        try:
            os.symlink(real, tmp_path / "link.json")
        except (OSError, NotImplementedError):
            pytest.skip("symlink unavailable")
        with pytest.raises(ValueError):
            load_manifest("link.json")


class TestComputeExposure:
    """The release's core claim. Fixtures use INTEGER control losses so
    `cheaper / n_controls` is exact — a percentile boundary compared with
    `<=` must not hinge on float noise (the semdedup boundary test was
    vacuous for exactly that reason).
    """

    def _controls_0_to_99(self):
        return [float(i) for i in range(100)]

    def test_below_all_controls_is_memorized(self):
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([0.1], [1.0, 2.0, 3.0, 4.0], ["s1"])
        assert out[0].percentile == 0.0
        assert out[0].memorized is True

    def test_middle_of_controls_not_memorized(self):
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([2.5], [1.0, 2.0, 3.0, 4.0], ["s1"])
        assert out[0].percentile == pytest.approx(0.5)
        assert out[0].memorized is False

    def test_percentile_boundary_exactly_0_01_is_memorized(self):
        """1/100 is exact in binary — `<=` must include it."""
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([0.5], self._controls_0_to_99(), ["s1"])
        assert out[0].percentile == 0.01, "fixture must land EXACTLY on 0.01"
        assert out[0].memorized is True, "<= 0.01 must count as memorized"

    def test_just_above_0_01_not_memorized(self):
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([1.5], self._controls_0_to_99(), ["s1"])
        assert out[0].percentile == 0.02
        assert out[0].memorized is False

    def test_ties_count_as_not_strictly_less(self):
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([1.0], [1.0, 1.0, 1.0, 1.0], ["s1"])
        assert out[0].percentile == 0.0

    def test_nan_canary_loss_is_unknown_not_memorized(self):
        """A NaN must never read as a leak."""
        import math

        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([float("nan")], [1.0, 2.0], ["s1"])
        assert out[0].memorized is False
        assert out[0].percentile == 1.0
        assert math.isnan(out[0].loss)

    def test_nan_controls_are_dropped_from_the_denominator(self):
        from soup_cli.utils.canary import compute_exposure

        out = compute_exposure([0.5], [1.0, float("nan"), 2.0], ["s1"])
        assert out[0].percentile == 0.0

    def test_zero_controls_refuses(self):
        from soup_cli.utils.canary import compute_exposure

        with pytest.raises(ValueError, match="at least one control"):
            compute_exposure([1.0], [], ["s1"])

    def test_all_nan_controls_refuses(self):
        """Refusing beats reporting OK against nothing."""
        from soup_cli.utils.canary import compute_exposure

        with pytest.raises(ValueError, match="at least one control"):
            compute_exposure([1.0], [float("nan")], ["s1"])

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.canary import compute_exposure

        with pytest.raises(ValueError, match="same length"):
            compute_exposure([1.0, 2.0], [1.0], ["s1"])


class TestClassifyCanary:
    def _exposures(self, percentiles):
        """Build exposures landing on EXACT percentiles via 100 controls."""
        from soup_cli.utils.canary import compute_exposure

        controls = [float(i) for i in range(100)]
        losses = [pct * 100 - 0.5 for pct in percentiles]
        return compute_exposure(
            losses, controls, [f"s{i}" for i in range(len(losses))]
        )

    def test_any_memorized_is_major(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(self._exposures([0.01, 0.5])) == "MAJOR"

    def test_bottom_ten_percent_is_minor(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(self._exposures([0.05, 0.5])) == "MINOR"

    def test_boundary_0_10_is_minor(self):
        from soup_cli.utils.canary import classify_canary

        exposures = self._exposures([0.10, 0.5])
        assert exposures[0].percentile == 0.10
        assert classify_canary(exposures) == "MINOR"

    def test_just_above_0_10_is_ok(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(self._exposures([0.11, 0.5])) == "OK"

    def test_all_typical_is_ok(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(self._exposures([0.5, 0.6])) == "OK"

    def test_major_beats_minor_when_both_present(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(self._exposures([0.01, 0.05])) == "MAJOR"

    def test_empty_exposures_is_ok(self):
        from soup_cli.utils.canary import classify_canary

        assert classify_canary(()) == "OK"


class TestBuildCanaryReport:
    def test_assembles_report(self):
        from soup_cli.utils.canary import build_canary_report

        controls = [float(i) for i in range(100)]
        rep = build_canary_report([0.5], controls, ["s1"])
        assert rep.verdict == "MAJOR"
        assert rep.n_controls == 100
        assert len(rep.exposures) == 1

    def test_n_controls_excludes_nan(self):
        from soup_cli.utils.canary import build_canary_report

        rep = build_canary_report([5.0], [1.0, float("nan"), 2.0], ["s1"])
        assert rep.n_controls == 2

    def test_report_is_frozen(self):
        import dataclasses

        from soup_cli.utils.canary import build_canary_report

        rep = build_canary_report([5.0], [1.0, 2.0], ["s1"])
        with pytest.raises(dataclasses.FrozenInstanceError):
            rep.verdict = "OK"

    def test_to_dict_roundtrips_through_json(self):
        from soup_cli.utils.canary import (
            build_canary_report,
            canary_report_to_dict,
        )

        rep = build_canary_report([5.0], [1.0, 2.0], ["s1"])
        payload = canary_report_to_dict(rep)
        assert json.loads(json.dumps(payload)) == payload
        assert payload["verdict"] == "OK"
        assert payload["n_controls"] == 2

    def test_to_dict_nan_loss_becomes_null_not_zero(self):
        """0.0 would read as the strongest possible leak signal."""
        from soup_cli.utils.canary import (
            build_canary_report,
            canary_report_to_dict,
        )

        rep = build_canary_report([float("nan")], [1.0, 2.0], ["s1"])
        payload = canary_report_to_dict(rep)
        assert payload["exposures"][0]["loss"] is None


class TestDataCanaryCli:
    def _dataset(self, tmp_path, count=10):
        path = tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]})
                for _ in range(count)
            ),
            encoding="utf-8",
        )
        return path

    def test_insert_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        res = CliRunner().invoke(app, ["data", "canary", "insert", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_check_help(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        res = CliRunner().invoke(app, ["data", "canary", "check", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_insert_adds_k_rows_and_manifest(self, tmp_path, monkeypatch):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        src = self._dataset(tmp_path, count=10)
        res = CliRunner().invoke(
            app,
            ["data", "canary", "insert", str(src), "-o", "out.jsonl",
             "--count", "4", "--manifest", "m.json"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        rows = [
            json.loads(x)
            for x in Path("out.jsonl").read_text().splitlines() if x
        ]
        assert len(rows) == 14
        manifest = json.loads(Path("m.json").read_text())
        assert len(manifest["canaries"]) == 4

    def test_inserted_rows_carry_the_manifest_secrets(self, tmp_path, monkeypatch):
        """The manifest must describe what actually landed in the data."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        src = self._dataset(tmp_path, count=5)
        res = CliRunner().invoke(
            app,
            ["data", "canary", "insert", str(src), "-o", "out.jsonl",
             "--count", "3", "--manifest", "m.json"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        blob = Path("out.jsonl").read_text()
        for entry in json.loads(Path("m.json").read_text())["canaries"]:
            assert entry["secret"].strip() in blob

    def test_insert_warns_manifest_is_sensitive(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        src = self._dataset(tmp_path, count=2)
        res = CliRunner().invoke(
            app, ["data", "canary", "insert", str(src), "-o", "o.jsonl",
                  "--manifest", "m.json"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        out = _clean(res.output).lower()
        assert "commit" in out or "secret" in out

    def test_insert_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        src = self._dataset(tmp_path, count=2)
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        res = CliRunner().invoke(
            app,
            ["data", "canary", "insert", str(src),
             "-o", str(tmp_path / "esc.jsonl"), "--manifest", "m.json"],
        )
        assert res.exit_code == 1
        assert "outside" in _clean(res.output).lower()

    def test_insert_manifest_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        src = self._dataset(tmp_path, count=2)
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        res = CliRunner().invoke(
            app,
            ["data", "canary", "insert", str(src), "-o", "o.jsonl",
             "--manifest", str(tmp_path / "esc.json")],
        )
        assert res.exit_code == 1
        assert "outside" in _clean(res.output).lower()

    def _seed_manifest(self, tmp_path, count=2):
        from soup_cli.utils.canary import generate_canaries, write_manifest

        write_manifest(generate_canaries(count=count, seed=0), "m.json")

    def test_check_major_exits_2(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=2)
        monkeypatch.setattr(
            cmd, "_load_pair", lambda *a, **k: (object(), object(), "cpu")
        )

        def _losses(model, tok, pairs, **kwargs):
            # first 2 are the canaries -> far cheaper than every control
            return [0.01 if i < 2 else float(i) for i in range(len(pairs))]

        monkeypatch.setattr(cmd, "compute_pair_losses", _losses)
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model", "--controls", "16"],
        )
        assert res.exit_code == 2, (res.output, repr(res.exception))
        assert "MAJOR" in _clean(res.output)

    def test_check_ok_exits_0(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=2)
        monkeypatch.setattr(
            cmd, "_load_pair", lambda *a, **k: (object(), object(), "cpu")
        )
        # Canaries sit mid-distribution: 5 of 16 controls are cheaper ->
        # percentile 0.31 -> OK. (An all-equal fixture would make NO control
        # strictly cheaper -> percentile 0.0 -> MAJOR, which is the rule
        # working correctly on an unrealistic input.)
        monkeypatch.setattr(
            cmd, "compute_pair_losses",
            lambda model, tok, pairs, **kw: (
                [5.0] * 2 + [float(i) for i in range(len(pairs) - 2)]
            ),
        )
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model", "--controls", "16"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "OK" in _clean(res.output)

    def test_check_splits_canary_and_control_losses_correctly(
        self, tmp_path, monkeypatch
    ):
        """The canary/control split must follow the manifest length."""
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=3)
        monkeypatch.setattr(
            cmd, "_load_pair", lambda *a, **k: (object(), object(), "cpu")
        )
        seen = {}

        def _losses(model, tok, pairs, **kwargs):
            seen["n_pairs"] = len(pairs)
            return [5.0] * 3 + [float(i) for i in range(len(pairs) - 3)]

        monkeypatch.setattr(cmd, "compute_pair_losses", _losses)
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model", "--controls", "8", "-o", "r.json"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert seen["n_pairs"] == 3 + 8, "3 canaries + 8 controls"
        from pathlib import Path

        report = json.loads(Path("r.json").read_text())
        assert len(report["exposures"]) == 3
        assert report["n_controls"] == 8

    def test_check_writes_report(self, tmp_path, monkeypatch):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=1)
        monkeypatch.setattr(
            cmd, "_load_pair", lambda *a, **k: (object(), object(), "cpu")
        )
        monkeypatch.setattr(
            cmd, "compute_pair_losses",
            lambda model, tok, pairs, **kw: (
                [5.0] + [float(i) for i in range(len(pairs) - 1)]
            ),
        )
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model", "-o", "r.json", "--controls", "8"],
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        report = json.loads(Path("r.json").read_text())
        assert report["verdict"] == "OK"
        assert report["n_controls"] == 8

    def test_check_model_load_import_error_is_friendly(
        self, tmp_path, monkeypatch
    ):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=1)

        def _boom(*a, **k):
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr(cmd, "_load_pair", _boom)
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "fake/model"],
        )
        assert res.exit_code == 1
        assert "soup-cli[train]" in _clean(res.output)

    def test_check_model_load_failure_is_friendly(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.commands import data_canary as cmd

        monkeypatch.chdir(tmp_path)
        self._seed_manifest(tmp_path, count=1)

        def _boom(*a, **k):
            raise OSError("no such model")

        monkeypatch.setattr(cmd, "_load_pair", _boom)
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "nope/model"],
        )
        assert res.exit_code == 1
        assert "could not load" in _clean(res.output).lower()

    def test_check_missing_manifest(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "nope.json",
                  "--base", "m"],
        )
        assert res.exit_code == 1

    def test_check_empty_manifest_refuses(self, tmp_path, monkeypatch):
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        Path("m.json").write_text('{"canaries": []}', encoding="utf-8")
        res = CliRunner().invoke(
            app, ["data", "canary", "check", "--manifest", "m.json",
                  "--base", "m"],
        )
        assert res.exit_code == 1
        assert "no canaries" in _clean(res.output).lower()


_REPLAY_YAML = """
base: HuggingFaceTB/SmolLM2-135M-Instruct
task: sft
data:
  train: train.jsonl
  replay: old.jsonl
  replay_ratio: 0.2
training:
  epochs: 1
"""

_PLAIN_YAML = (
    "base: m\ntask: sft\ndata:\n  train: t.jsonl\ntraining:\n  epochs: 1\n"
)


class TestReplaySchema:
    def _load(self, yaml_str):
        from soup_cli.config.loader import load_config_from_string

        return load_config_from_string(yaml_str)

    def test_happy_path(self):
        cfg = self._load(_REPLAY_YAML)
        assert cfg.data.replay == "old.jsonl"
        assert cfg.data.replay_ratio == 0.2

    def test_default_is_off(self):
        cfg = self._load(_PLAIN_YAML)
        assert cfg.data.replay is None
        assert cfg.data.replay_ratio == 0.1
        assert cfg.data.replay_seed is None

    def test_pretrain_allowed(self):
        cfg = self._load(
            _REPLAY_YAML.replace("task: sft", "task: pretrain").replace(
                "  train: train.jsonl",
                "  train: train.jsonl\n  format: plaintext",
            )
        )
        assert cfg.data.replay == "old.jsonl"

    def test_rejected_on_dpo(self):
        with pytest.raises(Exception, match="replay"):
            self._load(
                _REPLAY_YAML.replace("task: sft", "task: dpo").replace(
                    "  train: train.jsonl",
                    "  train: train.jsonl\n  format: dpo",
                )
            )

    def test_footgun_ratio_without_replay(self):
        yaml_str = (
            "base: m\ntask: sft\ndata:\n  train: t.jsonl\n"
            "  replay_ratio: 0.3\ntraining:\n  epochs: 1\n"
        )
        with pytest.raises(Exception, match="data.replay"):
            self._load(yaml_str)

    def test_footgun_seed_without_replay(self):
        yaml_str = (
            "base: m\ntask: sft\ndata:\n  train: t.jsonl\n"
            "  replay_seed: 7\ntraining:\n  epochs: 1\n"
        )
        with pytest.raises(Exception, match="data.replay"):
            self._load(yaml_str)

    def test_mutually_exclusive_with_packing(self):
        yaml_str = _REPLAY_YAML.replace(
            "training:\n  epochs: 1", "training:\n  epochs: 1\n  packing: true"
        )
        with pytest.raises(Exception, match="packing"):
            self._load(yaml_str)

    def test_mutually_exclusive_with_multipack(self):
        yaml_str = _REPLAY_YAML.replace(
            "training:\n  epochs: 1",
            "training:\n  epochs: 1\n  multipack: true",
        )
        with pytest.raises(Exception, match="multipack"):
            self._load(yaml_str)

    @pytest.mark.parametrize("bad", ["0.0", "0.6", "1.0", "-0.1"])
    def test_ratio_bounds(self, bad):
        with pytest.raises(Exception):
            self._load(
                _REPLAY_YAML.replace("replay_ratio: 0.2", f"replay_ratio: {bad}")
            )

    def test_ratio_boundary_0_5_allowed(self):
        cfg = self._load(
            _REPLAY_YAML.replace("replay_ratio: 0.2", "replay_ratio: 0.5")
        )
        assert cfg.data.replay_ratio == 0.5

    @pytest.mark.parametrize("bad", ['""', '"   "'])
    def test_replay_field_validator_rejects_blank(self, bad):
        with pytest.raises(Exception):
            self._load(_REPLAY_YAML.replace("replay: old.jsonl", f"replay: {bad}"))

    def test_replay_rejects_null_byte(self):
        from soup_cli.config.schema import DataConfig

        with pytest.raises(Exception, match="null"):
            DataConfig(train="t.jsonl", replay="a\x00b")

    def test_replay_rejects_overlong_path(self):
        from soup_cli.config.schema import DataConfig

        with pytest.raises(Exception, match="too long"):
            DataConfig(train="t.jsonl", replay="x" * 5000)

    def test_ratio_rejects_bool(self):
        with pytest.raises(Exception):
            self._load(
                _REPLAY_YAML.replace("replay_ratio: 0.2", "replay_ratio: true")
            )

    def test_seed_rejects_bool(self):
        with pytest.raises(Exception):
            self._load(
                _REPLAY_YAML.replace(
                    "replay_ratio: 0.2", "replay_seed: true"
                )
            )

    def test_replay_survives_model_dump(self):
        """Provenance rides the schema — the tracker/registry capture it."""
        cfg = self._load(_REPLAY_YAML)
        dumped = cfg.model_dump()
        assert dumped["data"]["replay"] == "old.jsonl"
        assert dumped["data"]["replay_ratio"] == 0.2
