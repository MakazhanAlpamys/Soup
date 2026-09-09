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

from soup_cli.config.schema import DataConfig, TrainingConfig

MLX_SFT = pathlib.Path(__file__).resolve().parents[1] / (
    "src/soup_cli/trainer/mlx_sft.py"
)


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

def test_max_grad_norm_is_reported_ignored_on_mlx_sft(config_at):
    """Acceptance criterion 1, in the direction that is true on this branch."""
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "mlx", "  max_grad_norm: 0.5"))
    reported = {e.field for e in check_config(cfg)}
    assert "training.max_grad_norm" in reported


def test_only_fields_the_user_actually_set_are_reported(config_at):
    """Not all 275 — and not the other MLX gaps the user never touched."""
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "mlx", "  max_grad_norm: 0.5"))
    reported = {e.field for e in check_config(cfg)}
    assert reported == {"training.max_grad_norm"}, (
        f"reported fields the user never set: {sorted(reported)}"
    )


def test_the_transformers_path_reports_nothing_for_the_same_config(config_at):
    from soup_cli.config.backend_support import check_config
    from soup_cli.config.loader import load_config

    cfg = load_config(config_at("sft", "transformers", "  max_grad_norm: 0.5"))
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

def test_doctor_config_names_the_ignored_field_and_its_issue(config_at, capsys):
    from soup_cli.commands.doctor import doctor

    path = config_at("sft", "mlx", "  max_grad_norm: 0.5")
    # Every typer parameter passed explicitly — see #752.
    doctor(nccl=False, disk=False, config=path)

    out = capsys.readouterr().out
    assert "max_grad_norm" in out
    assert "749" in out, "the reason should point at the issue that recorded it"


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


def _registry_drift(repo_root: pathlib.Path) -> list[str]:
    """Entries whose ``trainer_reads`` no longer matches the trainer's source."""
    from soup_cli.config.backend_support import REGISTRY, TRAINER_MODULES

    problems: list[str] = []
    for pair, entries in REGISTRY.items():
        module = TRAINER_MODULES.get(pair)
        if module is None:
            problems.append(f"{pair} has entries but no trainer module declared")
            continue
        read = _fields_read_by(repo_root / "src" / module)
        for entry in entries:
            name = entry.field.split(".", 1)[1]
            if entry.trainer_reads and name not in read:
                problems.append(
                    f"{entry.field}: declared read-to-warn by {module}, "
                    f"but that module no longer mentions it"
                )
            if not entry.trainer_reads and name in read:
                problems.append(
                    f"{entry.field}: declared unread, but {module} now reads it "
                    f"— the field was wired; remove or reclassify the entry"
                )
    return problems


def test_the_registry_matches_what_the_backend_trainer_actually_reads():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    problems = _registry_drift(repo_root)
    assert problems == [], "\n".join(problems)


def test_wiring_a_field_makes_its_stale_entry_fail(tmp_path, monkeypatch):
    """Acceptance criterion 1, the other direction — what #750 will do.

    ``max_grad_norm`` is declared unread on ``(sft, mlx)``. When #750 lands,
    ``mlx_sft.py`` reads it, and this guard must go red so the entry cannot
    quietly keep telling users their setting is ignored when it no longer is.
    """
    import soup_cli.config.backend_support as bs

    fake_src = tmp_path / "src" / "soup_cli" / "trainer"
    fake_src.mkdir(parents=True)
    (fake_src / "mlx_sft.py").write_text(
        textwrap.dedent(
            """\
            def build(tcfg):
                # the shape of the #750 fix
                return clip(tcfg.max_grad_norm)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        bs.TRAINER_MODULES, ("sft", "mlx"), "soup_cli/trainer/mlx_sft.py"
    )

    problems = _registry_drift(tmp_path)
    assert any("max_grad_norm" in p and "now reads it" in p for p in problems), problems


def test_deleting_a_warning_makes_its_entry_fail(tmp_path, monkeypatch):
    """The mirror case: an entry declared read-to-warn whose warning is gone."""
    import soup_cli.config.backend_support as bs

    fake_src = tmp_path / "src" / "soup_cli" / "trainer"
    fake_src.mkdir(parents=True)
    (fake_src / "mlx_sft.py").write_text("def build(tcfg):\n    return None\n", "utf-8")
    monkeypatch.setitem(
        bs.TRAINER_MODULES, ("sft", "mlx"), "soup_cli/trainer/mlx_sft.py"
    )

    problems = _registry_drift(tmp_path)
    assert any("use_galore" in p and "no longer mentions it" in p for p in problems)
