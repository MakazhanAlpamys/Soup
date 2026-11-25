"""``soup loop canary`` refuses an adapter whose last verdict was DON'T SHIP (review 5, F10).

The documented manual path, ``soup loop canary <adapter>``, promoted an adapter whose ``soup ship``
verdict was DON'T SHIP: the canary rule only asks that the canary is not worse than stable, and
stable had been rated down as often. Now the command refuses by name, using what the loop's own
logs say about that adapter (the latest line of the iteration that trained it, or a verdict file
``.soup-loops/verdicts/<adapter>.json``), unless ``--allow-dont-ship`` is given; the override is
recorded in ``loop.yaml`` (``canary_override``), the ``canary`` start line of ``progress.jsonl``
and, when the canary is promoted, the promotion's note and its line in ``versions.jsonl``.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.loop_daemon import WatchConfig, _promote
from soup_cli.utils.loop_state import LoopState, read_state, write_state
from tests.conftest import strip_ansi

ADAPTER = "loop-train-ab12cd34"
ITERATION = "iter-20261116T100000-aaaaaaaa"


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).replace("│", " ").split())


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".soup-loops").mkdir()
    return tmp_path


def _line(iteration_id=ITERATION, *, outcome="skipped", decision="DON'T SHIP", adapter=ADAPTER,
          served_as=None, run_id=None, reason="General benchmark(s) regressed past 5.00%"):
    return {
        "schema_version": 2, "iteration_id": iteration_id, "outcome": outcome,
        "started_at": "2026-11-16T10:00:00+00:00",
        "train": {"method": "sft", "run_id": run_id or adapter, "base": "b", "root_base": "b"},
        "artifacts": {
            "adapter": f".soup-loops/adapters/{adapter}", "merged": None, "gguf": None,
            "served_as": served_as,
        },
        "verdict": {"decision": decision, "method": "ship", "reason": reason},
    }


def _write_log(cwd, lines, raw=""):
    text = "".join(json.dumps(line) + "\n" for line in lines) + raw
    (cwd / ".soup-loops" / "iterations.jsonl").write_text(text, encoding="utf-8")


def _state(**kwargs) -> LoopState:
    return LoopState(
        served_model="org/base", eval_suite="e", baseline="b", base_model="org/base",
        status="running", **kwargs,
    )


class TestKnownDontShip:
    def test_a_skipped_iteration_with_a_dont_ship_verdict_is_a_no(self, cwd):
        _write_log(cwd, [_line()])

        text = ls.known_dont_ship(ADAPTER)

        assert text == (
            f"its last verdict is DON'T SHIP (iteration {ITERATION}: "
            "General benchmark(s) regressed past 5.00%)"
        )

    def test_the_adapter_is_found_by_served_name_and_by_run_id_too(self, cwd):
        _write_log(cwd, [_line(adapter="elsewhere", served_as="named-one", run_id="run-7")])

        assert ls.known_dont_ship("named-one") is not None
        assert ls.known_dont_ship("run-7") is not None
        assert ls.known_dont_ship("elsewhere") is not None

    def test_an_iteration_that_went_on_to_a_canary_is_not_a_no(self, cwd):
        # the check found nothing to verify (canary_decides): the loop itself deployed it
        _write_log(cwd, [_line(outcome="canary")])

        assert ls.known_dont_ship(ADAPTER) is None

    @pytest.mark.parametrize("outcome", ["promoted", "rolled_back", "failed"])
    def test_other_outcomes_are_not_a_no(self, cwd, outcome):
        _write_log(cwd, [_line(outcome=outcome)])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_a_ship_verdict_is_not_a_no(self, cwd):
        _write_log(cwd, [_line(decision="SHIP")])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_the_latest_line_of_an_iteration_wins(self, cwd):
        _write_log(cwd, [_line(), _line(outcome="promoted")])

        assert ls.known_dont_ship(ADAPTER) is None

    def test_the_newest_iteration_that_trained_it_decides(self, cwd):
        newer = "iter-20261116T110000-bbbbbbbb"
        _write_log(cwd, [_line(decision="SHIP", outcome="canary"), _line(newer)])

        assert newer in ls.known_dont_ship(ADAPTER)

    def test_an_adapter_the_loop_never_trained_is_unknown(self, cwd):
        _write_log(cwd, [_line()])

        assert ls.known_dont_ship("manual-sft-1") is None

    def test_no_log_is_unknown(self, cwd):
        assert ls.known_dont_ship(ADAPTER) is None

    def test_a_torn_and_a_foreign_line_do_not_stop_the_lookup(self, cwd):
        _write_log(cwd, [_line()], raw='{"half"\n[1, 2]\n{"iteration_id": 5}\n')

        assert ls.known_dont_ship(ADAPTER) is not None

    def test_a_verdict_file_named_after_the_adapter_counts(self, cwd):
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / "manual-sft-1.json").write_text(
            json.dumps({"decision": "DON'T SHIP", "failed_rule": "regression"}), encoding="utf-8"
        )

        assert ls.known_dont_ship("manual-sft-1") == (
            "its verdict file .soup-loops/verdicts/manual-sft-1.json says DON'T SHIP"
        )

    def test_a_ship_verdict_file_does_not(self, cwd):
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / "manual-sft-1.json").write_text(
            json.dumps({"decision": "SHIP"}), encoding="utf-8"
        )

        assert ls.known_dont_ship("manual-sft-1") is None

    def test_the_log_decides_before_a_verdict_file(self, cwd):
        _write_log(cwd, [_line(outcome="canary")])
        verdicts = cwd / ".soup-loops" / "verdicts"
        verdicts.mkdir()
        (verdicts / f"{ADAPTER}.json").write_text(
            json.dumps({"decision": "DON'T SHIP"}), encoding="utf-8"
        )

        assert ls.known_dont_ship(ADAPTER) is None

    @pytest.mark.parametrize("name", ["../x", "a/b", "a\\b", "", " ", "x" * 5 + "\n", "-lead"])
    def test_a_name_that_is_not_an_adapter_name_reads_nothing(self, cwd, name):
        assert ls.known_dont_ship(name) is None


SHIPPED = "loop-train-ef56ab78"
VERDICT_FILE_NAME = "manual-sft-1"
SHIPPED_VERDICT_FILE_NAME = "shipped-sft"

#: A Windows-only spelling: backslashes, the case Windows keeps, a junction. A junction needs no
#: privilege (unlike a symlink), so there is nothing to probe for: it is made or the test fails.
_ON_WINDOWS = pytest.mark.skipif(os.name != "nt", reason="a Windows-only spelling of the path")

#: Every way found to name the directory ``.soup-loops/adapters/<name>``: what the plan and the
#: iteration log write, what tab completion gives, what Windows and a ``..`` detour make of it,
#: and a link (symlink / junction) that leads there from inside or from outside the loop's tree.
#: A symlink spelling is skipped only where this account cannot create one (``tests/conftest.py``).
SPELLINGS = [
    "path", "dot-slash", "trailing-slash", "dot-leaf", "dotdot-leaf", "dotdot-via-sibling",
    "dotdot-via-verdicts", "absolute", "absolute-trailing-slash",
    pytest.param("backslashes", marks=_ON_WINDOWS),
    pytest.param("backslashes-trailing", marks=_ON_WINDOWS),
    pytest.param("mixed-separators", marks=_ON_WINDOWS),
    pytest.param("path-other-case", marks=_ON_WINDOWS),
    pytest.param("absolute-other-case", marks=_ON_WINDOWS),
    pytest.param("symlink-inside", marks=pytest.mark.requires_symlink),
    pytest.param("symlink-outside", marks=pytest.mark.requires_symlink),
    pytest.param("junction-inside", marks=_ON_WINDOWS),
    pytest.param("junction-outside", marks=_ON_WINDOWS),
]

# #3068: a direct checkpoint directory by relative or absolute path, with an
# optional trailing separator. On Windows backslashes are separators; on POSIX
# they are literal filename characters and cannot spell the same checkpoint.
CHECKPOINT_SPELLINGS = [
    "checkpoint-relative", "checkpoint-absolute", "checkpoint-relative-trailing",
    "checkpoint-absolute-trailing",
    pytest.param("checkpoint-backslashes", marks=_ON_WINDOWS),
    pytest.param("checkpoint-backslashes-trailing", marks=_ON_WINDOWS),
    pytest.param("checkpoint-through-adapter-link", marks=pytest.mark.requires_symlink),
    pytest.param("checkpoint-through-direct-link", marks=pytest.mark.requires_symlink),
]
# A filesystem that folds case must recognise all three spellings. A sensitive
# filesystem must keep the differently cased spelling distinct.
CASE_SPELLINGS = ["name-upper", "name-lower", "name-mixed"]


def _link(link: Path, target: Path, *, junction: bool) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if junction:
        import _winapi  # Windows only: the junction spellings are skipped elsewhere

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _spell(kind: str, cwd: Path, name: str) -> str:
    """The argument ``soup loop canary`` is given for the adapter directory ``name``."""
    relative = f".soup-loops/adapters/{name}"
    absolute = str(cwd / ".soup-loops" / "adapters" / name)
    adapter_dir = cwd / ".soup-loops" / "adapters" / name
    if kind.startswith("checkpoint-"):
        checkpoint_dir = adapter_dir / "checkpoint-12"
        checkpoint_dir.mkdir(exist_ok=True)
        if kind == "checkpoint-through-adapter-link":
            _link(cwd / "links" / "adapter", adapter_dir, junction=False)
            return "links/adapter/checkpoint-12"
        if kind == "checkpoint-through-direct-link":
            _link(cwd / "links" / "checkpoint", checkpoint_dir, junction=False)
            return "links/checkpoint"
        checkpoint = f"{relative}/checkpoint-12"
        if kind.startswith("checkpoint-absolute"):
            checkpoint = str(adapter_dir / "checkpoint-12")
        if kind.startswith("checkpoint-backslashes"):
            checkpoint = checkpoint.replace("/", "\\")
        if kind.endswith("-trailing"):
            checkpoint += os.sep
        return checkpoint
    if kind in CASE_SPELLINGS:
        return {
            "name-upper": name.upper(),
            "name-lower": name.lower(),
            "name-mixed": name.swapcase(),
        }[kind]
    spelled = {
        "bare": name,
        "path": relative,
        "dot-slash": f"./{relative}",
        "trailing-slash": f"{relative}/",
        "dot-leaf": f"{relative}/.",
        "dotdot-leaf": f"{relative}/../{name}",
        "dotdot-via-sibling": f".soup-loops/adapters/other/../{name}",
        "dotdot-via-verdicts": f".soup-loops/verdicts/../adapters/{name}",
        "absolute": absolute,
        "absolute-trailing-slash": absolute + os.sep,
        "backslashes": relative.replace("/", "\\"),
        "backslashes-trailing": relative.replace("/", "\\") + "\\",
        "mixed-separators": f".soup-loops/adapters\\{name}",
        "path-other-case": relative.upper(),
        "absolute-other-case": absolute.upper(),
    }
    if kind in spelled:
        return spelled[kind]
    junction = kind.startswith("junction")
    if kind.endswith("-inside"):
        link = cwd / ".soup-loops" / "adapters" / "alias"
        _link(link, adapter_dir, junction=junction)
        return ".soup-loops/adapters/alias"
    link = cwd / "links" / "alias"
    _link(link, adapter_dir, junction=junction)
    return "links/alias"


def _ship_verdict_file(cwd, name, decision):
    verdicts = cwd / ".soup-loops" / "verdicts"
    verdicts.mkdir(parents=True, exist_ok=True)
    (verdicts / f"{name}.json").write_text(json.dumps({"decision": decision}), encoding="utf-8")


def _tree(cwd):
    """The adapter directories, a sibling directory and the verdicts directory of a loop."""
    for name in (ADAPTER, VERDICT_FILE_NAME, SHIPPED, SHIPPED_VERDICT_FILE_NAME, "other"):
        (cwd / ".soup-loops" / "adapters" / name).mkdir(parents=True)
    (cwd / ".soup-loops" / "verdicts").mkdir(exist_ok=True)


@pytest.fixture(params=["iteration log", "verdict file"])
def dont_ship(request, cwd):
    """An adapter the loop knows is DON'T SHIP, found in the log or in a verdict file.

    Returns ``(name, why)``: the adapter's name and what ``known_dont_ship`` says about it.
    """
    _tree(cwd)
    if request.param == "iteration log":
        _write_log(cwd, [_line()])
        why = (
            f"its last verdict is DON'T SHIP (iteration {ITERATION}: "
            "General benchmark(s) regressed past 5.00%)"
        )
        return ADAPTER, why
    _ship_verdict_file(cwd, VERDICT_FILE_NAME, "DON'T SHIP")
    return VERDICT_FILE_NAME, (
        f"its verdict file .soup-loops/verdicts/{VERDICT_FILE_NAME}.json says DON'T SHIP"
    )


@pytest.fixture(params=["iteration log", "verdict file"])
def mixed_case_dont_ship(request, cwd):
    name = "Loop-Train-Ab12"
    (cwd / ".soup-loops" / "adapters" / name).mkdir(parents=True)
    if request.param == "iteration log":
        _write_log(cwd, [_line(adapter=name, reason="regressed")])
        return name, f"its last verdict is DON'T SHIP (iteration {ITERATION}: regressed)"
    _ship_verdict_file(cwd, name, "DON'T SHIP")
    return name, f"its verdict file .soup-loops/verdicts/{name}.json says DON'T SHIP"


@pytest.fixture
def loop_with_verdicts(cwd):
    """A loop where ``ADAPTER`` and ``VERDICT_FILE_NAME`` are DON'T SHIP; two others are SHIP."""
    _tree(cwd)
    _write_log(cwd, [
        _line(),
        _line("iter-20261116T120000-cccccccc", decision="SHIP", outcome="canary", adapter=SHIPPED),
    ])
    _ship_verdict_file(cwd, VERDICT_FILE_NAME, "DON'T SHIP")
    _ship_verdict_file(cwd, SHIPPED_VERDICT_FILE_NAME, "SHIP")
    return cwd


def _canary(*args):
    result = CliRunner().invoke(loop_app, ["canary", *args])
    return result, _plain(result.output)


def _seed_state(cwd, **kwargs):
    write_state(_state(**kwargs), str(cwd / ".soup" / "loop.yaml"))


def _refusal_tail(why):
    return (
        f": {why}. A promoted canary answers everyone's requests; pass --allow-dont-ship to "
        "canary it anyway (the override is recorded)"
    )


class TestEverySpellingOfTheAdapterDirectory:
    """The refusal must not depend on how the adapter is spelled (smoke P2-S57).

    ``soup loop canary <run>`` was refused, ``soup loop canary .soup-loops/adapters/<run>`` (the
    spelling tab completion gives, and the one the plan and the iteration log use) was not.
    """

    @pytest.mark.parametrize("kind", ["bare", *SPELLINGS, *CHECKPOINT_SPELLINGS])
    def test_the_lookup_sees_through_it(self, cwd, dont_ship, kind):
        name, why = dont_ship

        assert ls.known_dont_ship(_spell(kind, cwd, name)) == why

    @pytest.mark.parametrize("kind", ["bare", *SPELLINGS, *CHECKPOINT_SPELLINGS])
    def test_the_command_refuses_it_and_changes_nothing(self, cwd, dont_ship, kind):
        name, why = dont_ship
        spelling = _spell(kind, cwd, name)
        _seed_state(cwd)

        result, text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "refusing to canary" in text
        assert _refusal_tail(why) in text  # starts at the colon after the path: Rich-fold safe
        # A longer path is folded mid-word by the console, and a trailing backslash is doubled by
        # ``for_terminal`` (``rich.markup.escape`` does that so it cannot escape a closing tag).
        if len(spelling) <= 60 and not spelling.endswith("\\"):
            assert f"refusing to canary {spelling}{_refusal_tail(why)}" in text
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active is None and state.canary_override is None
        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()

    @pytest.mark.parametrize("kind", ["bare", *SPELLINGS, *CHECKPOINT_SPELLINGS])
    def test_the_override_accepts_it_and_records_what_was_typed(self, cwd, dont_ship, kind):
        name, why = dont_ship
        spelling = _spell(kind, cwd, name)
        _seed_state(cwd)

        result, _text = _canary(spelling, "--traffic", "5%", "--allow-dont-ship")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active == spelling
        assert state.canary_override == (
            f"{spelling} was canaried over a DON'T SHIP verdict with --allow-dont-ship ({why})"
        )

    @pytest.mark.parametrize("kind", ["path", "dot-slash", "absolute", "dotdot-leaf"])
    def test_a_removed_adapter_directory_is_still_that_adapter(self, cwd, dont_ship, kind):
        name, why = dont_ship
        spelling = _spell(kind, cwd, name)
        (cwd / ".soup-loops" / "adapters" / name).rmdir()

        assert ls.known_dont_ship(spelling) == why

    @pytest.mark.skipif(os.name != "nt", reason="Windows folds the case of a path")
    def test_the_case_of_the_directories_is_folded_when_the_adapter_is_gone(self, cwd, dont_ship):
        # nothing of the adapters directory exists to give the case back (a hand-run
        # ``soup ship --output .soup-loops/verdicts/<name>.json`` never made one)
        name, why = dont_ship
        shutil.rmtree(cwd / ".soup-loops" / "adapters")

        assert ls.known_dont_ship(f".soup-loops/ADAPTERS/{name}") == why


def _assert_case_variant_uses_verdict(cwd, spelling, why):
    _seed_state(cwd)
    result, text = _canary(spelling, "--traffic", "5%")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert _refusal_tail(why) in text
    assert read_state().canary_active is None

    result, _ = _canary(spelling, "--traffic", "5%", "--allow-dont-ship")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    state = read_state()
    assert state.canary_active == spelling
    assert state.canary_override is not None and why in state.canary_override


@pytest.mark.parametrize("kind", CASE_SPELLINGS)
def test_a_case_insensitive_filesystem_uses_the_same_verdict(cwd, mixed_case_dont_ship, kind):
    name, why = mixed_case_dont_ship
    spelling = _spell(kind, cwd, name)
    if not (cwd / ".soup-loops" / "adapters" / spelling).is_dir():
        pytest.skip("the adapter filesystem is case-sensitive")

    _assert_case_variant_uses_verdict(cwd, spelling, why)


@pytest.mark.parametrize("kind", CASE_SPELLINGS)
def test_a_case_sensitive_filesystem_keeps_names_distinct(cwd, mixed_case_dont_ship, kind):
    name, _ = mixed_case_dont_ship
    spelling = _spell(kind, cwd, name)
    other = cwd / ".soup-loops" / "adapters" / spelling
    if other.is_dir():
        pytest.skip("the adapter filesystem is case-insensitive")
    other.mkdir()
    _seed_state(cwd)

    result, text = _canary(spelling, "--traffic", "5%")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "canary set:" in text
    state = read_state()
    assert state.canary_active == spelling and state.canary_override is None


@pytest.mark.parametrize("kind", CASE_SPELLINGS)
def test_case_folding_lookup_on_case_sensitive_ci(cwd, mixed_case_dont_ship, kind, monkeypatch):
    name, why = mixed_case_dont_ship
    spelling = _spell(kind, cwd, name)
    if (cwd / ".soup-loops" / "adapters" / spelling).is_dir():
        pytest.skip("the native case-insensitive test covers this filesystem")
    # Simulate lookup of just this name so Linux also exercises the positive path.
    isdir, samefile = os.path.isdir, os.path.samefile
    typed = os.path.join(ls._ADAPTERS_DIR, spelling)
    actual = os.path.join(ls._ADAPTERS_DIR, name)

    def mapped(path):
        return actual if path == typed else path

    monkeypatch.setattr(os.path, "isdir", lambda path: isdir(mapped(path)))
    monkeypatch.setattr(os.path, "samefile", lambda a, b: samefile(mapped(a), mapped(b)))

    _assert_case_variant_uses_verdict(cwd, spelling, why)


@pytest.mark.parametrize("kind", CASE_SPELLINGS)
def test_case_folding_requires_the_same_directory_entry(cwd, mixed_case_dont_ship, kind,
                                                       monkeypatch):
    name, _ = mixed_case_dont_ship
    spelling = _spell(kind, cwd, name)
    adapters = cwd / ".soup-loops" / "adapters"
    if (adapters / spelling).is_dir():
        pytest.skip("requires a distinct spelling that does not exist on disk")

    # Fold only the directory probe. The real samefile cannot confirm the typed path.
    isdir = os.path.isdir
    typed = os.path.join(ls._ADAPTERS_DIR, spelling)
    actual = os.path.join(ls._ADAPTERS_DIR, name)
    monkeypatch.setattr(os.path, "isdir", lambda path: isdir(actual if path == typed else path))
    _seed_state(cwd)

    assert ls.known_dont_ship(spelling) is None
    result, text = _canary(spelling, "--traffic", "5%")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "canary set:" in text
    state = read_state()
    assert state.canary_active == spelling and state.canary_override is None


@pytest.mark.parametrize("kind", ["adapter-path", "checkpoint-parent", "checkpoint-name"])
@pytest.mark.parametrize("simulate_folding", [False, True], ids=["native", "simulated"])
def test_path_case_spelling_follows_directory_identity(
    cwd, dont_ship, kind, simulate_folding, monkeypatch,
):
    # Probe the volume, not the platform. A macOS or Windows volume can also be case-sensitive.
    root = cwd / ".soup-loops" / "adapters"
    probe = root / "case-probe"
    probe.touch()
    folds_case = probe.with_name("CASE-PROBE").is_file()
    probe.unlink()
    if simulate_folding and folds_case:
        pytest.skip("the native test covers this volume")

    name, why = dont_ship
    checkpoint = root / name / "checkpoint-12"
    checkpoint.mkdir()
    other_root = root.with_name("Adapters")
    other_checkpoint = checkpoint.with_name("Checkpoint-12")
    if kind == "checkpoint-name":
        typed, actual = other_checkpoint, checkpoint
        argument = typed
    else:
        typed, actual = other_root, root
        argument = typed / name
        if kind == "checkpoint-parent":
            argument /= checkpoint.name
    spelling = os.path.relpath(argument, cwd)

    if simulate_folding:
        # Linux also exercises POSIX realpath preserving case on a folding volume.
        isdir, samefile = os.path.isdir, os.path.samefile
        typed_path = os.path.realpath(typed)
        actual_path = os.path.realpath(actual)

        def mapped(path):
            if isinstance(path, str) and (
                path == typed_path or path.startswith(typed_path + os.sep)
            ):
                return actual_path + path[len(typed_path):]
            return path

        monkeypatch.setattr(os.path, "isdir", lambda path: isdir(mapped(path)))
        monkeypatch.setattr(os.path, "samefile", lambda a, b: samefile(mapped(a), mapped(b)))
    elif not folds_case:
        argument.mkdir(parents=True)
        _seed_state(cwd)
        assert ls.known_dont_ship(spelling) is None
        result, text = _canary(spelling, "--traffic", "5%")
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        state = read_state()
        assert state.canary_active == spelling and state.canary_override is None
        return

    _assert_case_variant_uses_verdict(cwd, spelling, why)


class TestSpellingsThatAreNotTheAdapter:
    """What the path mapping must leave alone: another adapter, another directory, no verdict."""

    @pytest.mark.skipif(os.name != "posix", reason="backslashes are literal on POSIX")
    def test_a_literal_backslash_name_is_not_the_checkpoint_path(self, cwd, dont_ship):
        name, _ = dont_ship
        checkpoint = _spell("checkpoint-relative", cwd, name)
        spelling = checkpoint.replace("/", "\\")
        (cwd / spelling).mkdir()
        _seed_state(cwd)

        result, text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        state = read_state()
        assert state.canary_active == spelling and state.canary_override is None

    @pytest.mark.parametrize("kind", [
        "bare", "path", "dot-slash", "trailing-slash", "absolute", "dotdot-leaf",
        "checkpoint-relative",
        pytest.param("symlink-inside", marks=pytest.mark.requires_symlink),
    ])
    @pytest.mark.parametrize("name", [SHIPPED, SHIPPED_VERDICT_FILE_NAME])
    def test_an_adapter_with_a_ship_verdict_is_accepted(self, loop_with_verdicts, kind, name):
        cwd = loop_with_verdicts
        spelling = _spell(kind, cwd, name)
        _seed_state(cwd)

        assert ls.known_dont_ship(spelling) is None
        result, text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active == spelling and state.canary_override is None

    @pytest.mark.parametrize("target", [ADAPTER, VERDICT_FILE_NAME])
    @pytest.mark.parametrize("where", [
        "elsewhere", "./elsewhere", ".soup-loops", ".soup-loops/verdicts",
        ".soup-loops/adapters-old", ".soup-loops/adapters/other",
        "other-root/.soup-loops/adapters",
    ])
    def test_the_same_last_component_elsewhere_is_a_different_adapter(
        self, loop_with_verdicts, target, where
    ):
        cwd = loop_with_verdicts
        (cwd / where).mkdir(parents=True, exist_ok=True)
        spelling = f"{where}/{target}"
        _seed_state(cwd)

        assert ls.known_dont_ship(spelling) is None
        result, text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active == spelling and state.canary_override is None

    @pytest.mark.parametrize("target", [ADAPTER, VERDICT_FILE_NAME])
    def test_the_same_last_component_in_an_absolute_path_elsewhere_is_a_different_adapter(
        self, loop_with_verdicts, target
    ):
        cwd = loop_with_verdicts
        (cwd / "elsewhere").mkdir()
        spelling = str(cwd / "elsewhere" / target)

        assert ls.known_dont_ship(spelling) is None

    @pytest.mark.parametrize(
        "kind", ["bare", "path", "dot-slash", "absolute", "checkpoint-relative"],
    )
    def test_an_adapter_with_no_verdict_at_all_is_accepted(self, loop_with_verdicts, kind):
        cwd = loop_with_verdicts
        (cwd / ".soup-loops" / "adapters" / "never-trained").mkdir()
        spelling = _spell(kind, cwd, "never-trained")
        _seed_state(cwd)

        assert ls.known_dont_ship(spelling) is None
        result, _text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state(str(cwd / ".soup" / "loop.yaml")).canary_override is None

    @pytest.mark.parametrize("suffix", ["subdir/checkpoint-12", "checkpoint-best", "notes"])
    def test_an_unrelated_subdirectory_is_not_a_checkpoint(self, loop_with_verdicts, suffix):
        path = f".soup-loops/adapters/{ADAPTER}/{suffix}"
        (loop_with_verdicts / path).mkdir(parents=True)
        assert ls.known_dont_ship(path) is None

    @pytest.mark.parametrize("file_exists", [True, False], ids=["file", "missing"])
    def test_a_checkpoint_spelling_requires_a_directory(self, cwd, dont_ship, file_exists):
        name, _ = dont_ship
        path = f".soup-loops/adapters/{name}/checkpoint-12"
        if file_exists:
            (cwd / path).touch()
        _seed_state(cwd)

        result, text = _canary(path, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        state = read_state()
        assert state.canary_active == path and state.canary_override is None

    @pytest.mark.parametrize("parent", ["elsewhere", ".soup-loops/adapters/other"])
    def test_a_checkpoint_without_a_direct_adapter_parent_is_not_matched(
        self, cwd, dont_ship, parent,
    ):
        name, _ = dont_ship
        path = f"{parent}/{name}/checkpoint-12"
        (cwd / path).mkdir(parents=True, exist_ok=True)
        _seed_state(cwd)

        result, text = _canary(path, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        assert read_state().canary_override is None

    @pytest.mark.requires_symlink
    def test_a_checkpoint_link_pointing_outside_adapters_is_not_matched(self, cwd, dont_ship):
        name, _ = dont_ship
        target = cwd / "elsewhere" / name / "checkpoint-12"
        target.mkdir(parents=True)
        path = f".soup-loops/adapters/{name}/checkpoint-12"
        _link(cwd / path, target, junction=False)
        _seed_state(cwd)

        result, text = _canary(path, "--traffic", "5%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set:" in text
        assert read_state().canary_override is None

    @pytest.mark.parametrize("spelling", [
        ".soup-loops/adapters", ".soup-loops/adapters/", ".soup-loops/adapters/.",
        ".soup-loops/adapters/other/..", ".soup-loops", ".", "..", "./",
    ])
    def test_the_adapter_directory_itself_and_its_parents_are_no_adapter(
        self, loop_with_verdicts, spelling
    ):
        assert ls.known_dont_ship(spelling) is None

    def test_a_directory_whose_name_is_no_adapter_name_reads_nothing(self, loop_with_verdicts):
        cwd = loop_with_verdicts
        (cwd / ".soup-loops" / "adapters" / "-odd").mkdir()
        _ship_verdict_file(cwd, "-odd", "DON'T SHIP")

        assert ls.known_dont_ship(".soup-loops/adapters/-odd") is None

    def test_a_path_with_a_nul_byte_is_left_to_the_policy_check(self, loop_with_verdicts):
        cwd = loop_with_verdicts
        spelling = f".soup-loops/adapters/{ADAPTER}\x00"
        _seed_state(cwd)

        assert ls.known_dont_ship(spelling) is None
        result, text = _canary(spelling, "--traffic", "5%")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "invalid canary policy: canary must be a non-empty NUL-free string or None" in text

    @pytest.mark.skipif(os.name != "nt", reason="a drive letter is a Windows thing")
    def test_a_path_on_another_drive_is_no_adapter_and_raises_nothing(self, loop_with_verdicts):
        drive = os.path.splitdrive(os.path.realpath("."))[0].upper()
        other = "Y:" if drive == "Z:" else "Z:"

        assert ls.known_dont_ship(f"{other}\\.soup-loops\\adapters\\{ADAPTER}") is None


class TestTheCommand:
    def _invoke(self, *args):
        result = CliRunner().invoke(loop_app, ["canary", *args])
        return result, _plain(result.output)

    def _seed(self, cwd, **kwargs):
        write_state(_state(**kwargs), str(cwd / ".soup" / "loop.yaml"))

    def test_it_refuses_by_name_and_changes_nothing(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke(ADAPTER, "--traffic", "50%")

        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert f"refusing to canary {ADAPTER}: its last verdict is DON'T SHIP" in text
        assert f"(iteration {ITERATION}" in text
        assert "pass --allow-dont-ship to canary it anyway (the override is recorded)" in text
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active is None and state.canary_override is None
        assert not (cwd / ".soup-loops" / "progress.jsonl").exists()

    def test_an_adapter_with_no_bad_verdict_is_canaried_as_before(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke("manual-sft-1", "--traffic", "50%")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "canary set: manual-sft-1 @ 50.0%" in text
        assert read_state(str(cwd / ".soup" / "loop.yaml")).canary_override is None

    def test_the_override_goes_through_and_is_recorded_three_ways(self, cwd):
        _write_log(cwd, [_line()])
        self._seed(cwd)

        result, text = self._invoke(ADAPTER, "--traffic", "50%", "--allow-dont-ship")

        assert result.exit_code == 0, (result.output, repr(result.exception))
        state = read_state(str(cwd / ".soup" / "loop.yaml"))
        assert state.canary_active == ADAPTER
        assert state.canary_override.startswith(
            f"{ADAPTER} was canaried over a DON'T SHIP verdict with --allow-dont-ship ("
        )
        assert ITERATION in state.canary_override
        progress = [
            json.loads(line)
            for line in (cwd / ".soup-loops" / "progress.jsonl").read_text("utf-8").splitlines()
        ]
        start = [row for row in progress if row["stage"] == "canary" and row["event"] == "start"]
        assert len(start) == 1 and "--allow-dont-ship" in start[0]["message"]

    def test_the_override_without_a_bad_verdict_records_nothing(self, cwd):
        self._seed(cwd)

        result, _text = self._invoke("manual-sft-1", "--allow-dont-ship")

        assert result.exit_code == 0
        assert read_state(str(cwd / ".soup" / "loop.yaml")).canary_override is None

    def test_a_hostile_log_reason_prints_no_control_bytes(self, cwd):
        _write_log(cwd, [_line(reason="boom \x1b]0;PWNED\x07 done")])
        self._seed(cwd)

        result, _text = self._invoke(ADAPTER)

        assert result.exit_code == 2 and "\x1b]" not in result.output


class TestThePromotionCarriesTheOverride:
    OVERRIDE = f"{ADAPTER} was canaried over a DON'T SHIP verdict with --allow-dont-ship (x)"

    def test_the_promotion_note_and_the_state_say_so(self, cwd):
        state = _state(canary_active=ADAPTER, canary_traffic_pct=50.0,
                       canary_override=self.OVERRIDE)

        promoted, note = _promote(state, WatchConfig(poll_interval_sec=1.0))

        assert promoted.production_name == ADAPTER
        assert promoted.canary_override is None
        assert note == f"canary {ADAPTER} promoted to production; {self.OVERRIDE}; " + (
            "its adapter path is unknown, so the next verdict compares against the base"
        )

    def test_a_promotion_without_an_override_is_as_before(self, cwd):
        state = _state(canary_active=ADAPTER, canary_traffic_pct=50.0)

        _promoted, note = _promote(state, WatchConfig(poll_interval_sec=1.0))

        assert "--allow-dont-ship" not in note

    def test_rollback_clears_it(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(
            _state(canary_active=ADAPTER, canary_traffic_pct=50.0, canary_rollout_id="r1",
                   canary_override=self.OVERRIDE),
            path,
        )

        result = CliRunner().invoke(loop_app, ["rollback"])

        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert read_state(path).canary_active is None
        assert read_state(path).canary_override is None


class TestStateField:
    def test_it_round_trips(self, cwd):
        path = str(cwd / ".soup" / "loop.yaml")
        write_state(_state(canary_override="an override"), path)

        assert read_state(path).canary_override == "an override"

    @pytest.mark.parametrize("bad", ["", 5, ["x"], "a\x1bb", "a\nb"])
    def test_a_bad_value_is_refused(self, bad):
        with pytest.raises((ValueError, TypeError)):
            replace(_state(), canary_override=bad)
