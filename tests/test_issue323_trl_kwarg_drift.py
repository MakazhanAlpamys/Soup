"""Issue #323 — every keyword a trainer passes to a trl config must exist.

The drift this guards is documented in #323 and its comment: the ``[train]``
extra pinned ``trl>=0.7.0`` with no upper bound, CI resolved 0.29.1 while the
dev box ran 0.19.1, and on 0.29.1 six trainers could not build their config at
all. **CI stayed green through all of it**, because the ``trl`` imports live
inside ``setup()`` and no test had ever called it -- the wrappers were only ever
instantiated. As the issue's comment puts it:

    an unbounded floor-only pin plus a test that never reaches the import is
    indistinguishable from working software.

Calling ``setup()`` needs a model and a dataset. This asks the same question
without either: read the keywords each wrapper passes to its trl config, and ask
the class *as installed* whether it still accepts them. The probe is
``_trl_compat.config_accepts``, which the wrappers already use at runtime -- a
capability question, never a version comparison, because a version table is
what was wrong twice.

**Known blind spot, stated rather than implied:** ``**splat`` arguments are
invisible to a static read. ``grpo.py`` builds ``GRPOConfig(**grpo_kwargs)`` and
``sft.py`` builds ``SFTConfig(**dict_args)``, so their keywords are not checked
here. What is checked is every keyword written literally at the call site --
which is where ``max_prompt_length`` lived before #326 moved it behind a helper.

The trl TRAINER classes are out of scope for the same reason inverted: they take
much of their signature through ``**kwargs``, so a static signature check
reports ``model`` and ``train_dataset`` as rejected. Measured, not assumed --
that false alarm is why the alias branch below filters on ``Config``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TRAINER_DIR = _ROOT / "src" / "soup_cli" / "trainer"
_TRAINER_SOURCES = sorted(_TRAINER_DIR.glob("*.py"))

#: ``*Config`` names that belong to Soup or to peft/transformers, not to trl.
#: Verified Soup-owned by ``test_the_exemptions_are_real``, so this list cannot
#: be used to silence a trl class that has started rejecting a keyword.
_SOUP_OWNED = {"ULDConfig", "MiniLLMConfig", "MoleGatingConfig"}
_THIRD_PARTY = {"LoraConfig", "BitsAndBytesConfig", "GenerationConfig", "AutoConfig"}


def _alias_map(tree: ast.AST) -> dict:
    """``{local_name: (trl_symbol, experimental_module)}``.

    #326 moved three configs out of the public namespace, so the wrappers hold
    them as ``orpo_config_cls = resolve_trl_symbol("ORPOConfig", ...)``. Without
    following that indirection the scan would silently cover only the wrappers
    that never had the harder break.
    """
    aliases: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "resolve_trl_symbol"):
            continue
        args = node.value.args
        if not args or not isinstance(args[0], ast.Constant):
            continue
        symbol = args[0].value
        fallback = None
        if len(args) > 1 and isinstance(args[1], ast.Constant):
            fallback = args[1].value
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = (symbol, fallback)
    return aliases


def _config_calls(path: pathlib.Path) -> list:
    """``(symbol, fallback_module, {keywords}, lineno)`` per trl config call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _alias_map(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in aliases and aliases[name][0].endswith("Config"):
            # Configs only. The same resolve_trl_symbol indirection also holds
            # the TRAINER classes, and those take much of their signature
            # through **kwargs -- asking inspect.signature about them reports
            # `model` and `train_dataset` as rejected, which is a false alarm,
            # measured. Trainer-argument drift is a real question and a
            # different one; it is not what this scan answers.
            symbol, fallback = aliases[name]
        elif name.endswith("Config") and name not in _SOUP_OWNED | _THIRD_PARTY:
            symbol, fallback = name, None
        else:
            continue
        # ``kw.arg is None`` is ``**splat`` -- invisible here, see module note.
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        if keywords:
            found.append((symbol, fallback, keywords, node.lineno))
    return found


_ALL_CALLS = [(path, call) for path in _TRAINER_SOURCES for call in _config_calls(path)]


class TestTrlConfigKeywordsStillExist:
    @pytest.fixture(autouse=True)
    def _need_trl(self):
        pytest.importorskip("trl")

    def test_the_scan_actually_finds_trl_config_calls(self):
        """A scan that finds nothing passes vacuously; this is its floor."""
        assert _TRAINER_DIR.is_dir(), _TRAINER_DIR
        symbols = {call[0] for _, call in _ALL_CALLS}
        assert len(symbols) >= 5, symbols
        # The two shapes that broke: a public class, and one moved to
        # trl.experimental and therefore reachable only through an alias.
        assert "DPOConfig" in symbols
        assert "ORPOConfig" in symbols

    @pytest.mark.parametrize(
        "path,call",
        _ALL_CALLS,
        ids=[f"{p.stem}-{c[0]}-{c[3]}" for p, c in _ALL_CALLS],
    )
    def test_every_literal_keyword_is_accepted_by_the_installed_class(self, path, call):
        from soup_cli.trainer._trl_compat import config_accepts, resolve_trl_symbol

        symbol, fallback, keywords, lineno = call
        config_cls = resolve_trl_symbol(symbol, fallback)
        rejected = sorted(k for k in keywords if not config_accepts(config_cls, k))
        assert not rejected, (
            f"{path.name}:{lineno} passes {rejected} to {symbol}, which the "
            f"installed trl does not accept. This is the #323 class of failure: "
            f"it surfaces at setup() on a user's box, not in CI, because no "
            f"test reaches the import."
        )


class TestTheScannerCanActuallyFail:
    """A scanner nobody has watched fail is indistinguishable from a broken one."""

    def test_a_bogus_keyword_is_reported(self, tmp_path):
        path = tmp_path / "fake.py"
        path.write_text(
            "x = DPOConfig(beta=0.1, definitely_not_a_field=3)\n", encoding="utf-8"
        )
        calls = _config_calls(path)
        assert len(calls) == 1
        symbol, fallback, keywords, _ = calls[0]
        assert symbol == "DPOConfig"
        assert "definitely_not_a_field" in keywords

        pytest.importorskip("trl")
        from soup_cli.trainer._trl_compat import config_accepts, resolve_trl_symbol

        cls = resolve_trl_symbol(symbol, fallback)
        assert not config_accepts(cls, "definitely_not_a_field")
        assert config_accepts(cls, "beta")

    def test_the_resolve_trl_symbol_indirection_is_followed(self, tmp_path):
        """Without this the scan would cover only the easy wrappers -- the three
        configs #326 moved are exactly the ones held behind an alias."""
        path = tmp_path / "fake.py"
        path.write_text(
            'cls = resolve_trl_symbol("ORPOConfig", "trl.experimental.orpo")\n'
            "cfg = cls(beta=0.1)\n",
            encoding="utf-8",
        )
        calls = _config_calls(path)
        assert len(calls) == 1
        symbol, fallback, keywords, _ = calls[0]
        assert (symbol, fallback) == ("ORPOConfig", "trl.experimental.orpo")
        assert keywords == {"beta"}

    def test_splat_only_calls_are_skipped_not_silently_passed(self, tmp_path):
        """The blind spot, pinned so it stays a known gap rather than a surprise."""
        path = tmp_path / "fake.py"
        path.write_text("x = GRPOConfig(**kwargs)\n", encoding="utf-8")
        assert _config_calls(path) == []

    def test_soup_owned_configs_are_not_mistaken_for_trl(self, tmp_path):
        path = tmp_path / "fake.py"
        path.write_text("x = ULDConfig(alpha=1)\n", encoding="utf-8")
        assert _config_calls(path) == []

    def test_the_exemptions_are_real(self):
        """The skip list is verified, not trusted: every Soup-owned name in it
        must be a class Soup actually defines, so it cannot be repurposed to
        silence a trl config."""
        import re

        source = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (_ROOT / "src" / "soup_cli").rglob("*.py")
        )
        for name in _SOUP_OWNED:
            assert re.search(rf"^class {name}\b", source, re.M), name

class TestTheDriftWorkflowCanActuallyFail:
    """The workflow shipped with no test of its own, and its most actionable
    output could not fail.

    A package declared with incompatible ranges in two extras cannot be
    installed by anyone who asks for both -- that is unambiguous and
    actionable. It was being printed into the step summary of a job that then
    exited 0, i.e. into somewhere nobody looks. A drift alarm that cannot go
    red is write-only (review of #486).

    The report is inspected as a parsed tree rather than pattern-matched,
    following what #496 did for the floor job: assertions on code SPELLING
    pass with the behaviour killed and the strings kept (#321).
    """

    @staticmethod
    def _report_source() -> str:
        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        raw = (root / ".github" / "workflows" / "dependency-drift.yml").read_text(
            encoding="utf-8"
        )
        workflow = yaml.safe_load(raw)
        job = next(iter(workflow["jobs"].values()))
        step = next(
            s for s in job["steps"] if "Report resolved versions" in s.get("name", "")
        )
        # `<<'PY'` is followed by a shell redirection on the SAME line
        # (`>> "$GITHUB_STEP_SUMMARY"`); the script starts on the next one.
        # The heredoc marker is followed by a shell redirection on the SAME
        # line, so the script begins on the next one; and it ends at the
        # closing marker.
        after = step["run"].split("<<" + chr(39) + "PY" + chr(39), 1)[1]
        raw_lines = after.splitlines()[1:]
        body_lines = []
        for line in raw_lines:
            if line.strip() == "PY":
                break
            body_lines.append(line)
        body = "".join(line + chr(10) for line in body_lines)
        lines = body.splitlines()
        indent = min(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()),
            default=0,
        )
        return "\n".join(line[indent:] if line.strip() else "" for line in lines)

    def test_the_workflow_still_parses_and_has_the_report_step(self):
        """Without this, every assertion below silently stops covering."""
        source = self._report_source()
        assert "contradictory" in source, (
            "the report step no longer computes contradictions; the assertions "
            "below would pass vacuously"
        )
        compile(source, "<drift-report>", "exec")

    def test_contradictory_declarations_fail_the_job(self):
        """The finding the author surfaced by hand must go red, not into the
        summary of a green run."""
        tree = ast.parse(self._report_source())
        guarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.Name) and test.id == "contradictory"):
                continue
            guarded.extend(
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "exit"
            )
        assert guarded, (
            "the report never calls sys.exit under `if contradictory:` -- an "
            "incompatible declaration would print into a green job's summary, "
            "which is write-only"
        )

    def test_a_clean_resolve_does_not_fail_the_job(self):
        """A guard that fires on correct code is one people delete, and a
        weekly alarm that is always red gets muted within a month."""
        tree = ast.parse(self._report_source())
        unconditional = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "exit"
        ]
        assert not unconditional, (
            "the report exits at module level, so a clean weekly run would go "
            "red too"
        )
