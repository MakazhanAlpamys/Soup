"""#755 — which of your settings does your backend actually read?

Backend support cannot be *inferred*. Three implementations were measured and
rejected before this one (all recorded in the issue):

1. static reachability over the import graph — every trainer's closure is ~150
   modules, so it detected **zero of five** independently-known MLX gaps;
2. static reads of the trainer module alone — finds those five, but reports
   ``train_on_responses_only`` and ``loraplus_lr_ratio`` as missing from the
   transformers path, where they live in helper modules;
3. runtime tracing under ``--dry-run`` — ``commands/train.py:1255`` exits before
   the trainer wrapper is constructed, so nothing is observed.

So the registry is **declared and reviewed**, and the guard's job is to stop it
drifting — which is the failure that produced #749, where ``max_grad_norm`` was
missing from ``mlx_sft.py``'s hand-maintained warning list while MLX dropped it.

No test here needs a GPU, MLX, the network, or a model download.
"""

from __future__ import annotations

import ast
import pathlib
import re
import textwrap

import pytest
import typer

from soup_cli.config.backend_support import REGISTRY as _REGISTRY
from soup_cli.config.schema import DataConfig, TrainingConfig

MLX_SFT = pathlib.Path(__file__).resolve().parents[1] / (
    "src/soup_cli/trainer/mlx_sft.py"
)

#: Every entry the registry declares as read-only-to-warn. Tests parametrise
#: over this so a new entry is covered without editing an assertion.
_WARNED_ENTRIES = [e for e in _REGISTRY[("sft", "mlx")] if e.trainer_reads]


@pytest.fixture
def config_at(tmp_path):
    """Write a soup.yaml with the given task/backend and training block."""

    def _make(task: str, backend: str, training: str = "", data: str = "") -> str:
        train_file = tmp_path / "train.jsonl"
        train_file.write_text(
            '{"instruction": "a", "output": "b"}\n', encoding="utf-8"
        )
        body = textwrap.dedent(
            f"""\
            base: some-model
            task: {task}
            backend: {backend}
            data:
              train: {train_file}
              format: alpaca
            {textwrap.indent(data, "  ") if data else ""}
            training:
            {textwrap.indent(training or "  epochs: 1", "  ")}
            output: {tmp_path / "out"}
            """
        )
        path = tmp_path / "soup.yaml"
        path.write_text(body, encoding="utf-8")
        return str(path)

    return _make


# --------------------------------------------------------------------------
# the registry itself
# --------------------------------------------------------------------------

def _unknown_registry_fields(registry) -> list[str]:
    """Registry entries that do not resolve to a real field on the models."""
    models = {"training": TrainingConfig, "data": DataConfig}
    unknown = []
    for entries in registry.values():
        for entry in entries:
            namespace, _, name = entry.field.partition(".")
            if namespace not in models or name not in models[namespace].model_fields:
                unknown.append(entry.field)
    return unknown


def test_registry_names_only_fields_that_exist():
    """A renamed field must not be able to hide behind a stale entry."""
    from soup_cli.config.backend_support import REGISTRY

    unknown = _unknown_registry_fields(REGISTRY)
    assert unknown == [], f"registry names fields that do not exist: {unknown}"


def test_an_entry_naming_a_field_that_no_longer_exists_is_caught():
    """Acceptance criterion 4 — the check must be able to fail.

    Every entry is valid today, so observing the check green proves nothing:
    deleting it entirely leaves the suite passing. This pins it against a
    renamed field instead.
    """
    from soup_cli.config.backend_support import IGNORED, SupportEntry

    stale = {
        ("sft", "mlx"): (
            SupportEntry("training.max_grad_norm", IGNORED, "real"),
            SupportEntry("training.gone_in_v075", IGNORED, "renamed away"),
            SupportEntry("nosuchsection.field", IGNORED, "bad namespace"),
        )
    }
    assert _unknown_registry_fields(stale) == [
        "training.gone_in_v075",
        "nosuchsection.field",
    ]


def test_every_registry_entry_carries_a_status_and_a_reason():
    from soup_cli.config.backend_support import REGISTRY, STATUSES

    bad = [
        entry.field
        for entries in REGISTRY.values()
        for entry in entries
        if entry.status not in STATUSES or not entry.reason.strip()
    ]
    assert bad == [], f"entries missing a valid status or a reason: {bad}"


def test_registry_covers_every_field_the_mlx_trainer_warns_about():
    """The drift that produced #749, pinned.

    ``mlx_sft.py`` names some unsupported settings in prose ("GaLore") and some
    by dotted path. Only the dotted ones can be matched mechanically, and those
    are the ones this asserts.
    """
    from soup_cli.config.backend_support import unsupported_for

    tree = ast.parse(MLX_SFT.read_text(encoding="utf-8"))
    warned: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
        ):
            for arg in ast.walk(node):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    warned.update(re.findall(r"\b(?:training|data)\.[a-z_]+", arg.value))

    assert warned, "found no dotted field names in mlx_sft.py's warning list"
    registered = {e.field for e in unsupported_for("sft", "mlx")}
    missing = sorted(warned - registered)
    assert missing == [], (
        f"mlx_sft.py warns about {missing} but the registry does not list them"
    )


def test_an_unregistered_task_backend_pair_reports_nothing():
    from soup_cli.config.backend_support import unsupported_for

    assert unsupported_for("sft", "transformers") == ()
    assert unsupported_for("no_such_task", "no_such_backend") == ()


# --------------------------------------------------------------------------
# check_config — only what the user set
# --------------------------------------------------------------------------

def test_a_setting_the_backend_ignores_is_reported(config_at):
    """Acceptance criterion 1.

    This used to assert on ``training.max_grad_norm``. #750 wired it, so it is
    honoured and no longer in the registry -- the guard below is what caught
    that. ``training.seed`` is the live example now: MLX seeds through
    ``mx.random`` and reads the field only to say so.
    """
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "mlx", "  seed: 42"))
    reported = {e.field for e in check_config(cfg)}
    assert "training.seed" in reported


def test_only_fields_the_user_actually_set_are_reported(config_at):
    """Not all 275 — and not the other MLX gaps the user never touched."""
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "mlx", "  seed: 42"))
    reported = {e.field for e in check_config(cfg)}
    assert reported == {"training.seed"}, (
        f"reported fields the user never set: {sorted(reported)}"
    )


def test_the_transformers_path_reports_nothing_for_the_same_config(config_at):
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "transformers", "  seed: 42"))
    assert check_config(cfg) == []


def test_transformers_is_not_flagged_for_helper_owned_fields(config_at):
    """Acceptance criterion 5 — the false positives implementation 2 produced.

    ``sft.py`` reads neither ``train_on_responses_only`` (it lives in
    ``data/sft_format.py``) nor ``loraplus_lr_ratio`` (in the #738 helper), so a
    per-module walker called both unsupported. They are supported.
    """
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(
        config_at(
            "sft",
            "transformers",
            "  loraplus_lr_ratio: 16.0",
            data="train_on_responses_only: true",
        )
    )
    reported = {e.field for e in check_config(cfg)}
    assert reported == set(), f"false positives on the transformers path: {reported}"


# --------------------------------------------------------------------------
# the doctor leg
# --------------------------------------------------------------------------

def test_doctor_config_names_the_ignored_field_and_its_reason(config_at, capsys):
    from soup_cli.commands.doctor import doctor

    path = config_at("sft", "mlx", "  seed: 42")
    # Every typer parameter passed explicitly — see #752.
    doctor(nccl=False, disk=False, config=path)

    out = capsys.readouterr().out
    assert "seed" in out
    assert "mx.random" in out, "the row must carry the reason, not just the name"


@pytest.mark.parametrize(
    "path, why",
    [("does_not_exist.yaml", "missing"), ("broken.yaml", "unparseable")],
)
def test_doctor_exits_non_zero_when_the_config_cannot_be_read(
    path, why, tmp_path, monkeypatch
):
    """A leg CI can gate on must not report success on a config it never read."""
    from soup_cli.commands.doctor import doctor

    monkeypatch.chdir(tmp_path)
    if why == "unparseable":
        (tmp_path / path).write_text("base: [unclosed\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(tmp_path / path))
    assert excinfo.value.exit_code != 0


def test_doctor_without_config_does_not_read_one(config_at, capsys):
    """The existing environment-only behaviour must be unchanged."""
    from soup_cli.commands.doctor import doctor

    doctor(nccl=False, disk=False, config=None)
    out = capsys.readouterr().out
    assert "Config check" not in out


# --------------------------------------------------------------------------
# the bidirectional guard: the registry cannot silently go stale
# --------------------------------------------------------------------------

def _fields_read_by(path: pathlib.Path) -> set[str]:
    """Attribute names and exact string constants in a module, docstrings out."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def _fields_read_outside(repo_root: pathlib.Path, modules) -> set[str]:
    """Names read by any module in the tree except the ones declared for a pair."""
    src = repo_root / "src" / "soup_cli"
    declared = {(repo_root / "src" / m).resolve() for m in modules}
    names: set[str] = set()
    for path in src.rglob("*.py"):
        if path.resolve() in declared or path.name == "schema.py":
            continue
        names |= _fields_read_by(path)
    return names


def _registry_drift(repo_root: pathlib.Path) -> list[str]:
    """Entries whose ``trainer_reads`` no longer matches the real source.

    Scans the **union** of every module declared for the pair. Grounding on the
    trainer alone was a hole: #734 wired four schedule fields that live 7-12
    times in ``mlx_optim.py`` and twice in ``mlx_sft.py``, and the guard caught
    it only because the call site happened to name them.
    """
    from soup_cli.config.backend_support import REGISTRY, TRAINER_MODULES

    problems: list[str] = []
    for pair, entries in REGISTRY.items():
        modules = TRAINER_MODULES.get(pair)
        if not modules:
            problems.append(f"{pair} has entries but no trainer modules declared")
            continue
        read: dict[str, str] = {}
        for module in modules:
            path = repo_root / "src" / module
            if not path.exists():
                problems.append(f"{pair}: declared module {module} does not exist")
                continue
            for name in _fields_read_by(path):
                read.setdefault(name, module)
        # A claimed gap must be a gap *relative to something*. If no module in
        # the tree reads the field at all, it is not "this backend ignores it"
        # -- it is a field nothing consumes anywhere, which is #748's territory
        # and #751's allowlist, not this table. Without this, an entry can
        # assert an unfounded gap for any field and never be contradicted.
        elsewhere = _fields_read_outside(repo_root, modules)
        for entry in entries:
            name = entry.field.split(".", 1)[1]
            if not entry.trainer_reads and name not in elsewhere:
                problems.append(
                    f"{entry.field}: declared a gap on {pair}, but no module "
                    f"outside {', '.join(modules)} reads it either — that is a "
                    f"globally unconsumed field (#748), not a per-backend gap"
                )
            if entry.trainer_reads and name not in read:
                problems.append(
                    f"{entry.field}: declared read-to-warn, but none of "
                    f"{', '.join(modules)} mentions it"
                )
            if not entry.trainer_reads and name in read:
                problems.append(
                    f"{entry.field}: declared unread, but {read[name]} now reads "
                    f"it — the field was wired; remove or reclassify the entry"
                )
    return problems


def test_the_registry_matches_what_the_backend_trainer_actually_reads():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    problems = _registry_drift(repo_root)
    assert problems == [], "\n".join(problems)


def test_the_declared_modules_cover_every_helper_the_trainer_imports():
    """The map must not silently shrink back to trainer-only.

    Derived from the trainer's own imports rather than hardcoded, so adding a
    helper to ``mlx_sft.py`` without declaring it here fails, and dropping one
    from the map fails too. Asserting a literal list would pass either way.
    """
    from soup_cli.config.backend_support import TRAINER_MODULES

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for pair, modules in TRAINER_MODULES.items():
        primary = repo_root / "src" / modules[0]
        tree = ast.parse(primary.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("soup_cli.trainer"):
                    imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("soup_cli.trainer"):
                        imported.add(alias.name)
        declared = {m.removesuffix(".py").replace("/", ".") for m in modules}
        for mod in sorted(imported):
            if mod not in declared:
                missing.append(f"{pair}: {modules[0]} imports {mod}, not declared")
    assert missing == [], "\n".join(missing)


def test_an_unfounded_gap_claim_is_caught():
    """@MakazhanAlpamys's mutation on #756, which the first version survived.

    He fabricated an entry declaring ``data.mask_history`` ignored on
    ``(sft, mlx)`` and the suite stayed green: the drift check only compared
    the entry against the declared modules, so an ``ignored`` claim about a
    field those modules never mention could not be contradicted.

    A gap is relative to something. If nothing anywhere reads the field, it is
    not "this backend ignores it" -- it is a field no code consumes, which is
    #748's subject and #751's allowlist.
    """
    import soup_cli.config.backend_support as bs

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fabricated = bs.SupportEntry(
        "data.mask_history", bs.IGNORED, "fabricated, unfounded claim"
    )
    real = bs.REGISTRY[("sft", "mlx")]
    try:
        bs.REGISTRY[("sft", "mlx")] = (fabricated,) + real
        problems = _registry_drift(repo_root)
    finally:
        bs.REGISTRY[("sft", "mlx")] = real

    assert any(
        "mask_history" in p and "globally unconsumed" in p for p in problems
    ), problems


def test_a_gap_for_a_field_another_backend_reads_is_accepted():
    """Control for the above — the check must not fire on a real gap.

    ``training.seed`` is read by the transformers path and ignored by MLX, so
    it is exactly the shape the registry exists to record.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    assert _registry_drift(repo_root) == []


def test_a_declared_module_that_does_not_exist_is_caught(tmp_path, monkeypatch):
    """A renamed helper must not silently shrink the guard's scope."""
    import soup_cli.config.backend_support as bs

    monkeypatch.setitem(
        bs.TRAINER_MODULES, ("sft", "mlx"), ("soup_cli/trainer/gone.py",)
    )
    problems = _registry_drift(tmp_path)
    assert any("does not exist" in p for p in problems), problems


def test_a_field_wired_only_inside_a_helper_is_caught(tmp_path, monkeypatch):
    """The hole this fix closes.

    #734 wired the schedule fields mostly inside ``mlx_optim.py``. Had it left
    nothing visible in ``mlx_sft.py``, a trainer-only scan would have passed.
    """
    import soup_cli.config.backend_support as bs

    trainer = tmp_path / "src" / "soup_cli" / "trainer"
    trainer.mkdir(parents=True)
    # the trainer never names the field ...
    (trainer / "mlx_sft.py").write_text(
        "def setup(cfg):\n    return build(cfg)\n", encoding="utf-8"
    )
    # ... the helper it delegates to does the reading
    (trainer / "mlx_optim.py").write_text(
        "def build(cfg):\n    return plan(warmup_ratio=cfg.training.warmup_ratio)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        bs.REGISTRY,
        ("sft", "mlx"),
        (bs.SupportEntry("training.warmup_ratio", bs.IGNORED, "stale on purpose"),),
    )
    monkeypatch.setitem(
        bs.TRAINER_MODULES,
        ("sft", "mlx"),
        ("soup_cli/trainer/mlx_sft.py", "soup_cli/trainer/mlx_optim.py"),
    )

    problems = _registry_drift(tmp_path)
    assert any("warmup_ratio" in p and "mlx_optim.py" in p for p in problems), problems


def test_wiring_a_field_makes_its_stale_entry_fail(tmp_path, monkeypatch):
    """Acceptance criterion 1, the other direction — and this one really happened.

    The registry used to carry ``training.max_grad_norm`` as unread on
    ``(sft, mlx)``. #750 wired it, this check went red naming it, and the entry
    was removed. Every surviving entry is now ``trainer_reads=True``, so there
    is no live entry left to prove the direction with — it is pinned against a
    synthetic registry rather than quietly dropped, because a direction nothing
    exercises is a direction that rots.
    """
    import soup_cli.config.backend_support as bs

    fake_src = tmp_path / "src" / "soup_cli" / "trainer"
    fake_src.mkdir(parents=True)
    (fake_src / "mlx_sft.py").write_text(
        textwrap.dedent(
            """\
            def build(tcfg):
                # the shape #750 introduced
                return clip(tcfg.max_grad_norm)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        bs.REGISTRY,
        ("sft", "mlx"),
        (
            bs.SupportEntry(
                "training.max_grad_norm",
                bs.IGNORED,
                "stale on purpose: the field is wired now",
            ),
        ),
    )
    monkeypatch.setitem(
        bs.TRAINER_MODULES, ("sft", "mlx"), ("soup_cli/trainer/mlx_sft.py",)
    )

    problems = _registry_drift(tmp_path)
    assert any("max_grad_norm" in p and "now reads it" in p for p in problems), problems


@pytest.mark.parametrize(
    "entry",
    [e for e in _WARNED_ENTRIES],
    ids=[e.field for e in _WARNED_ENTRIES],
)
def test_deleting_a_warning_makes_its_entry_fail(entry, tmp_path, monkeypatch):
    """The mirror case: an entry declared read-to-warn whose warning is gone.

    Parametrised over every ``trainer_reads=True`` entry rather than pinning
    ``use_galore``, so this cannot pass by fixture coincidence and a new entry
    is covered the moment it is added.
    """
    import soup_cli.config.backend_support as bs

    trainer = tmp_path / "src" / "soup_cli" / "trainer"
    trainer.mkdir(parents=True)
    (trainer / "mlx_sft.py").write_text(
        "def build(tcfg):\n    return None\n", encoding="utf-8"
    )
    monkeypatch.setitem(bs.REGISTRY, ("sft", "mlx"), (entry,))
    monkeypatch.setitem(
        bs.TRAINER_MODULES, ("sft", "mlx"), ("soup_cli/trainer/mlx_sft.py",)
    )

    problems = _registry_drift(tmp_path)
    name = entry.field.split(".", 1)[1]
    assert any(name in p and "declared read-to-warn" in p for p in problems), problems
