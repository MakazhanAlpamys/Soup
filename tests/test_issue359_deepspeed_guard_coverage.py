"""Issue #359 — the DeepSpeed empty-param-group guard, in every wrapper.

#336 fixed the failure in ``trainer/sft.py``: with LoRA, HF's no-decay optimizer
group comes out empty, DeepSpeed drops it, and torch's scheduler then hits a
1-vs-2 strict-``zip`` mismatch on ``base_lrs`` at the first
``lr_scheduler.step()``. Every other wrapper builds its own optimizer and had
the same exposure.

Coverage is derived by SCANNING ``soup_cli/trainer/`` rather than from a list of
names, following ``tests/test_device_map_distributed.py``. That file's own
history is the argument: its first version parametrized over the six trainers
the fix had touched and passed while nine more sites still carried the defect,
because a hand-written list cannot report what it does not name.
"""

from __future__ import annotations

import pathlib
import re
import types

import pytest

_TRAINER_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "trainer"
_TRAINER_SOURCES = sorted(_TRAINER_DIR.glob("*.py"))

#: A module builds its own trainer when it assigns a CALL to ``self.trainer``.
#: ``self.trainer = None`` in ``__init__`` is not construction, and a wrapper
#: that only forwards ``deepspeed_config`` to another wrapper (preference.py)
#: builds nothing and needs no guard of its own.
_BUILDS_TRAINER = re.compile(r"self\.trainer\s*=\s*(?!None\b)[A-Za-z_][\w.]*\s*\(")
_GUARD_CALL = re.compile(r"attach_empty_param_group_guard\s*\(")
#: DeepSpeed reachability: the wrapper takes a config it can be launched with.
_DEEPSPEED_CAPABLE = re.compile(r"\bdeepspeed_config\b")


def _code_without_comments(text: str) -> str:
    """Strip trailing comments so prose ABOUT the guard is not read as a call."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _needs_guard(path: pathlib.Path) -> bool:
    code = _code_without_comments(path.read_text(encoding="utf-8"))
    return bool(_DEEPSPEED_CAPABLE.search(code)) and bool(_BUILDS_TRAINER.search(code))


class TestGuardCoverage:
    def test_the_scan_actually_sees_the_trainer_package(self):
        """Without this, a moved source tree turns the check below into a
        vacuous pass over an empty file list."""
        assert _TRAINER_DIR.is_dir(), _TRAINER_DIR
        names = {path.name for path in _TRAINER_SOURCES}
        assert len(names) > 20
        assert {"sft.py", "dpo.py", "orpo.py", "simpo.py", "ppo.py"} <= names

    def test_the_scan_finds_more_than_the_one_wrapper_336_fixed(self):
        """#336 touched sft.py alone. If this ever reports one module, the
        detector has broken rather than the codebase having shrunk."""
        needing = [path.name for path in _TRAINER_SOURCES if _needs_guard(path)]
        assert len(needing) >= 15, needing
        assert "sft.py" in needing
        # Delegating wrapper: forwards deepspeed_config to DPO/SimPO/ORPO/IPO/BCO
        # and builds no trainer itself, so the guard belongs in those, not here.
        assert "preference.py" not in needing

    @pytest.mark.parametrize(
        "path", _TRAINER_SOURCES, ids=[p.stem for p in _TRAINER_SOURCES]
    )
    def test_every_deepspeed_capable_wrapper_attaches_the_guard(self, path):
        code = _code_without_comments(path.read_text(encoding="utf-8"))
        if not _needs_guard(path):
            return
        assert _GUARD_CALL.search(code), (
            f"{path.name} builds a trainer and accepts a deepspeed_config but "
            "never calls attach_empty_param_group_guard(); under --deepspeed "
            "with LoRA it dies at the first lr_scheduler.step() (#359)."
        )

    #: preference.py forwards ``deepspeed_config`` into five wrappers that each
    #: carry the guard, and builds no trainer of its own. It is the ONLY module
    #: allowed to hold a deepspeed_config without calling the guard, and it is
    #: named here so that the next such module has to be argued for rather than
    #: slipping through a regex that stopped matching.
    _GUARD_EXEMPT = {"preference.py"}

    def test_no_deepspeed_capable_module_quietly_loses_the_guard(self):
        """Closes the rebind evasion: ``_BUILDS_TRAINER`` requires the
        constructor CALL on the ``self.trainer =`` line, so an ordinary
        refactor to ``built = KTOTrainer(...); self.trainer = built`` makes a
        module invisible to the per-wrapper check above and the guard can be
        deleted with nothing failing. Keying on ``deepspeed_config`` instead of
        on construction cannot be evaded that way."""
        offenders = []
        for path in _TRAINER_SOURCES:
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            if not _DEEPSPEED_CAPABLE.search(code):
                continue
            if path.name in self._GUARD_EXEMPT:
                continue
            if not _GUARD_CALL.search(code):
                offenders.append(path.name)
        assert not offenders, (
            f"{', '.join(offenders)} accept a deepspeed_config but never call "
            "attach_empty_param_group_guard(). If a module genuinely forwards "
            "rather than builds, add it to _GUARD_EXEMPT with the reason."
        )

    def test_the_exemption_list_stays_earned(self):
        """An exemption that stops being true is worse than none: it silences
        the check above for a module that has since grown its own trainer."""
        for name in self._GUARD_EXEMPT:
            path = _TRAINER_DIR / name
            assert path.is_file(), f"_GUARD_EXEMPT names {name}, which no longer exists"
            code = _code_without_comments(path.read_text(encoding="utf-8"))
            assert not _BUILDS_TRAINER.search(code), (
                f"{name} is exempt from the guard check but now builds its own "
                "trainer; remove it from _GUARD_EXEMPT and attach the guard."
            )

    @pytest.mark.parametrize("path", _TRAINER_SOURCES, ids=[p.stem for p in _TRAINER_SOURCES])
    def test_the_guard_is_attached_only_under_deepspeed(self, path):
        """Every one of these blocks carries the comment "only under DeepSpeed
        so the ordinary path keeps its own optimizer" -- and until now only
        sft.py had a test enforcing it (test_issue336...:test_the_call_is_
        conditional_on_deepspeed reads that one module). An unconditional
        attach rewrites the optimizer of EVERY LoRA run, not just DeepSpeed
        ones, and no test noticed."""
        import ast as _ast

        source = path.read_text(encoding="utf-8")
        if not _GUARD_CALL.search(_code_without_comments(source)):
            return

        tree = _ast.parse(source)
        guarded = []

        def _is_deepspeed_test(node: _ast.AST) -> bool:
            return any(
                isinstance(sub, _ast.Attribute) and sub.attr == "deepspeed_config"
                for sub in _ast.walk(node)
            )

        def _walk(node: _ast.AST, under_deepspeed: bool) -> None:
            for child in _ast.iter_child_nodes(node):
                if isinstance(child, _ast.If):
                    inner = under_deepspeed or _is_deepspeed_test(child.test)
                    for stmt in child.body:
                        _walk(stmt, inner)
                    for stmt in child.orelse:
                        _walk(stmt, under_deepspeed)
                    continue
                if (
                    isinstance(child, _ast.Call)
                    and isinstance(child.func, _ast.Name)
                    and child.func.id == "attach_empty_param_group_guard"
                ):
                    guarded.append(under_deepspeed)
                _walk(child, under_deepspeed)

        _walk(tree, False)
        assert guarded, f"{path.name} matches the guard regex but has no parsed call"
        assert all(guarded), (
            f"{path.name} attaches the empty-param-group guard unconditionally. "
            "It must sit under `if self.deepspeed_config:` -- outside DeepSpeed "
            "the ordinary path keeps its own optimizer."
        )

    def test_the_patterns_would_catch_the_unguarded_shape(self):
        """A scanner nobody has watched fail is indistinguishable from a broken
        one. Both halves of the detector are exercised here."""
        assert _BUILDS_TRAINER.search("        self.trainer = DPOTrainer(**kwargs)")
        assert _BUILDS_TRAINER.search("self.trainer = orpo_trainer_cls(model=m)")
        assert not _BUILDS_TRAINER.search("        self.trainer = None")
        assert _GUARD_CALL.search("attach_empty_param_group_guard(self.trainer)")
        assert not _GUARD_CALL.search("# attach_empty_param_group_guard is needed")
        assert _DEEPSPEED_CAPABLE.search("deepspeed_config: Optional[str] = None,")

    def test_a_comment_mentioning_the_guard_does_not_satisfy_it(self):
        """The positive check reads code, not prose -- otherwise the note that
        explains the guard would be enough to pass without calling it."""
        source = 'self.trainer = DPOTrainer()  # attach_empty_param_group_guard(x)\n'
        assert not _GUARD_CALL.search(_code_without_comments(source))


class TestGuardToleratesTrainersWithoutCreateOptimizer:
    """Widening the guard's blast radius from one wrapper to eighteen makes this
    load-bearing: not every TRL trainer exposes ``create_optimizer``, and a
    guard that raises AttributeError while being attached would turn a
    DeepSpeed-only defect into a crash on the ordinary path.
    """

    def test_a_trainer_without_create_optimizer_is_declined_not_crashed(self):
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        trainer = types.SimpleNamespace()  # no create_optimizer at all
        assert attach_empty_param_group_guard(trainer) is False
        assert not hasattr(trainer, "create_optimizer")

    def test_a_non_callable_create_optimizer_is_declined(self):
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        trainer = types.SimpleNamespace(create_optimizer=None)
        assert attach_empty_param_group_guard(trainer) is False
        assert trainer.create_optimizer is None

    def test_a_real_create_optimizer_is_still_wrapped(self):
        from soup_cli.utils.deepspeed import attach_empty_param_group_guard

        class _Trainer:
            def __init__(self):
                self.calls = 0

            def create_optimizer(self):
                self.calls += 1
                return types.SimpleNamespace(
                    param_groups=[{"params": [1, 2]}, {"params": []}]
                )

        trainer = _Trainer()
        assert attach_empty_param_group_guard(trainer) is True
        optimizer = trainer.create_optimizer()
        assert trainer.calls == 1
        assert len(optimizer.param_groups) == 1  # the empty group is gone


class TestUserSuppliedDeepspeedFileIsResolved:
    """#359 criterion 4 — decided: resolve, but only what is provably invalid.

    A user file is the user's authority, so a config that uses no run-dependent
    keys comes back byte-identical and by the same path. But the two keys the
    resolver rewrites are not preferences, they are errors: DeepSpeed refuses a
    ``zero_hpz_partition_size`` the world size is not divisible by, and its fp16
    quantiser against a bf16 run raises ``expected mat1 and mat2 to have the
    same dtype`` inside ``deepspeed/runtime/zero/linear.py``. The documented way
    to customise ZeRO++ is to copy the preset JSON, which copies both defects --
    so leaving a user file unresolved hands the escape hatch a crash the presets
    are already protected from. The repair is announced, never silent.
    """

    _PLAIN = {"bf16": {"enabled": True}, "zero_optimization": {"stage": 2}}
    _ZERO_PP = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "zero_hpz_partition_size": 8,
            "zero_quantized_weights": True,
            "zero_quantized_gradients": True,
        },
    }

    @staticmethod
    def _write(tmp_path, payload, name="my.json"):
        import json

        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_file_without_run_dependent_keys_is_returned_untouched(self, tmp_path):
        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        path = self._write(tmp_path, self._PLAIN)
        assert resolve_user_deepspeed_file(str(path)) == str(path)

    def test_a_file_copying_the_zero_pp_placeholders_is_repaired_into_a_copy(
        self, tmp_path
    ):
        import json

        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        path = self._write(tmp_path, self._ZERO_PP)
        resolved = resolve_user_deepspeed_file(str(path), gpu_count=4)
        assert resolved != str(path)
        zero = json.loads(open(resolved, encoding="utf-8").read())["zero_optimization"]
        assert zero["zero_hpz_partition_size"] == 4
        assert zero["zero_quantized_weights"] is False
        assert zero["zero_quantized_gradients"] is False

    def test_the_users_file_on_disk_is_never_mutated(self, tmp_path):
        import json

        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        path = self._write(tmp_path, self._ZERO_PP)
        before = path.read_text(encoding="utf-8")
        resolve_user_deepspeed_file(str(path), gpu_count=4)
        assert path.read_text(encoding="utf-8") == before
        assert json.loads(before)["zero_optimization"]["zero_hpz_partition_size"] == 8

    def test_the_repair_is_announced(self, tmp_path, capsys):
        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        path = self._write(tmp_path, self._ZERO_PP)
        resolve_user_deepspeed_file(str(path), gpu_count=4)
        out = capsys.readouterr().out
        assert "DeepSpeed" in out
        assert str(path.name) in out or "zero_hpz_partition_size" in out

    def test_unreadable_json_is_passed_through_rather_than_crashing(self, tmp_path):
        """DeepSpeed reports a malformed config better than a Soup traceback,
        and refusing here would break a file DeepSpeed might well accept."""
        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        assert resolve_user_deepspeed_file(str(path)) == str(path)

    def test_a_missing_file_is_passed_through(self, tmp_path):
        from soup_cli.utils.deepspeed import resolve_user_deepspeed_file

        missing = str(tmp_path / "nope.json")
        assert resolve_user_deepspeed_file(missing) == missing

    def test_the_cli_routes_a_user_file_through_the_resolver(self, tmp_path, monkeypatch):
        """Wiring, pinned separately from behaviour: without this the resolver
        can be correct and still never run on the path that matters."""
        import soup_cli.utils.deepspeed as ds
        from soup_cli.commands.train import _resolve_deepspeed

        path = self._write(tmp_path, self._PLAIN)
        seen = {}

        def _fake(config_path, *, gpu_count=None):
            seen["path"] = config_path
            return "/tmp/resolved.json"

        monkeypatch.setattr(ds, "resolve_user_deepspeed_file", _fake)
        assert _resolve_deepspeed(str(path)) == "/tmp/resolved.json"
        assert seen["path"] == str(path)

    def test_a_named_preset_still_takes_the_preset_path(self, monkeypatch):
        import soup_cli.utils.deepspeed as ds
        from soup_cli.commands.train import _resolve_deepspeed

        monkeypatch.setattr(ds, "write_deepspeed_config", lambda stage: f"preset:{stage}")
        monkeypatch.setattr(
            ds, "resolve_user_deepspeed_file", lambda *a, **k: pytest.fail("not a file")
        )
        assert _resolve_deepspeed("zero2") == "preset:zero2"
