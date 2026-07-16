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
