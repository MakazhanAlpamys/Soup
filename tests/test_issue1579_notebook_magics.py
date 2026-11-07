"""#1579: ``soup migrate --from unsloth`` reads a notebook that carries IPython magics.

Every code cell was concatenated and parsed as one Python module, so the first ``!pip``
or ``%%capture`` line made ``ast.parse`` raise and the whole notebook was refused with
"Could not parse notebook code (SyntaxError)", no cell, no line. Every official Unsloth
notebook starts with such a cell.

A leading ``!`` / ``%`` line is a magic only at statement level, which is decided by
whether the lines before it in the cell already parse, so a ``%`` operator or a ``!=`` at
the start of a continuation line is left alone (the maintainer's fourth criterion). A
cell whose first line starts with ``%%`` is skipped whole. Replaced lines become blank,
so line numbers stay stable for the source-order logic. A cell that parses today is
returned byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.migrate.unsloth import _strip_ipython_magics, migrate_unsloth

_LOAD = (
    "from unsloth import FastLanguageModel\n"
    "model, tokenizer = FastLanguageModel.from_pretrained(model_name='m', max_seq_length=2048,"
    " load_in_4bit=True)\n"
)
_GRPO = (
    "from trl import GRPOTrainer, GRPOConfig\n"
    "cfg = GRPOConfig(per_device_train_batch_size=4, learning_rate=5e-6)\n"
    "trainer = GRPOTrainer(model=model, reward_funcs=[f], args=cfg)\n"
)


def _migrate(tmp_path: Path, *cells: str) -> dict:
    nb = {"cells": [{"cell_type": "code", "source": [c]} for c in cells]}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return migrate_unsloth(path)


def _summary(result: dict) -> tuple:
    return result["task"], result["training"].get("lr"), result["training"].get("batch_size")


class TestMagicsAreSkipped:
    def test_the_issue_notebook_migrates_as_grpo(self, tmp_path: Path) -> None:
        result = _migrate(tmp_path, "%%capture\n!pip install unsloth trl\n", _LOAD, _GRPO)
        assert _summary(result) == ("grpo", 5e-6, 4)

    def test_line_magics_between_statements_change_nothing(self, tmp_path: Path) -> None:
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        with_magics = _migrate(
            tmp_path,
            "%env HF_HUB_OFFLINE=1\n" + _LOAD + "!nvidia-smi\n",
            "%load_ext autoreload\n" + _GRPO + "    %time print(1)\n",
        )
        assert with_magics == clean

    def test_a_non_python_cell_magic_cell_is_skipped_whole(self, tmp_path: Path) -> None:
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        with_cell_magic = _migrate(tmp_path, _LOAD, "%%bash\necho hi\nls -la | wc -l\n", _GRPO)
        assert with_cell_magic == clean

    @pytest.mark.parametrize("magic", ["%%capture", "%%time", "%%timeit", "%%prun"])
    def test_a_python_body_cell_magic_keeps_its_body(self, tmp_path: Path, magic: str) -> None:
        # Unsloth wraps real cells in %%capture; the GRPO config lives inside one here.
        result = _migrate(tmp_path, _LOAD, f"{magic}\n!pip install trl\n" + _GRPO)
        assert _summary(result) == ("grpo", 5e-6, 4)

    def test_a_magic_as_the_first_line_of_a_suite_is_still_a_magic(self, tmp_path: Path) -> None:
        # The header `if ...:` does not parse alone, but it is a complete statement
        # boundary, so the magic under it is blanked; the operator case below is not.
        cell = (
            "import os\n"
            "if os.environ.get('COLAB'):\n"
            "    !pip install unsloth\n"
            "    %env HF_HUB_OFFLINE=1\n"
            "for i in range(2):\n"
            "    %time print(i)\n"
        )
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        assert _migrate(tmp_path, _LOAD, cell, _GRPO) == clean


    def test_a_backslash_continued_magic_is_one_magic(self, tmp_path: Path) -> None:
        # IPython joins a `!` line ending in a backslash with the lines after it;
        # 18 of the 82 official Unsloth GRPO notebooks (gpt-oss among them) install
        # that way, and only the first line was blanked (maintainer's review).
        cell = (
            "%%capture\n"
            "import os\n"
            "if 'COLAB_' in ''.join(os.environ):\n"
            "    !uv pip install -qqq \\\n"
            "        \"torch>=2.8.0\" {_numpy} \\\n"
            "        unsloth\n"
            "!uv pip install --no-deps -qqq \\\n"
            "    trl==0.22.2\n"
        )
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        assert _migrate(tmp_path, cell, _LOAD, _GRPO) == clean
        assert _strip_ipython_magics(cell).count("\n") == cell.count("\n")

    def test_a_magic_under_nested_headers_is_still_a_magic(self, tmp_path: Path) -> None:
        # CodeRabbit on the first head: a one-space probe satisfies `if a:` but not the
        # nested `    if b:`; probing at the magic's own indentation does.
        cell = (
            "import os\n"
            "if os.environ.get('COLAB'):\n"
            "    if not os.path.exists('/content/x'):\n"
            "        !pip install unsloth\n"
            "    with open('log') as fh:\n"
            "        for line in fh:\n"
            "            %time print(line)\n"
        )
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        assert _migrate(tmp_path, _LOAD, cell, _GRPO) == clean


class TestContinuationLinesAreNotMagics:
    # The maintainer's fourth criterion: a `%` operator or a `!=` at the start of a
    # continuation line is Python, not a magic, and must survive untouched.
    _CONT = (
        "count, size, a, b = 7, 3, 1, 2\n"
        "total = (count\n"
        "         % size)\n"
        "flag = (a\n"
        "        != b)\n"
        "rest = count \\\n"
        "    % 2\n"
    )

    def test_a_parsing_cell_is_returned_byte_for_byte(self, monkeypatch) -> None:
        # The parse-first shortcut is the whole guarantee here: a cell that parses is
        # never probed line by line (pinned per the review; removing the shortcut
        # gives the same output, so only this catches it).
        from soup_cli.migrate import unsloth as mod

        def never(*args, **kwargs):
            raise AssertionError("a parsing cell was probed line by line")

        monkeypatch.setattr(mod, "_neutralise_magic", never)
        assert _strip_ipython_magics(self._CONT) == self._CONT

    def test_continuations_survive_even_in_a_cell_that_needs_transforming(self) -> None:
        cell = "!pip install x\n" + self._CONT
        out = _strip_ipython_magics(cell)
        assert out == "\n" + self._CONT  # the magic blanked, the operators kept
        assert out.count("\n") == cell.count("\n")  # line numbers stable

    def test_a_backslash_after_a_continuation_operator_is_not_a_magic(self) -> None:
        # `% 7 \` is Python (a continuation line), not a magic, so the line after it
        # stays Python even though it follows a backslash (review of #1583: without the
        # `replacement is not None` guard, `+ warmup` was blanked and silently lost).
        body = "steps, warmup = 9, 1\ntotal = steps \\\n    % 7 \\\n    + warmup\n"
        cell = "!pip install x\n" + body
        assert _strip_ipython_magics(cell) == "\n" + body

    def test_the_probes_do_not_repeat_an_invalid_escape_warning(self) -> None:
        # The real parse reports an invalid escape once; the per-line probes must not
        # repeat it with cell-local line numbers (SyntaxWarning on 3.12; earlier
        # Pythons emit a DeprecationWarning, where this passes either way).
        import warnings

        cell = "s = '\\{'\n!pip install x\n"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _strip_ipython_magics(cell) == "s = '\\{'\n\n"
        assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]

    def test_the_notebook_migrates_exactly_as_today(self, tmp_path: Path) -> None:
        clean = _migrate(tmp_path, _LOAD, _GRPO)
        with_cont = _migrate(tmp_path, _LOAD, self._CONT, _GRPO)
        assert with_cont == clean


class TestRealSyntaxErrors:
    def test_a_real_error_is_refused_naming_the_cell_and_line(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"cell 2.*line 2") as info:
            _migrate(tmp_path, _LOAD, "ok = 1\ndef broken(:\n", _GRPO)
        assert "Could not parse" in str(info.value)

    def test_a_magic_line_inside_an_open_bracket_is_still_a_syntax_error(
        self, tmp_path: Path
    ) -> None:
        # Not statement level, so not a magic: refused, as today.
        with pytest.raises(ValueError, match="Could not parse"):
            _migrate(tmp_path, _LOAD, "x = [\n!pip install y\n]\n", _GRPO)
