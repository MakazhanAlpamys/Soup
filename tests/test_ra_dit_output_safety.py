from pathlib import Path

import pytest
import yaml

from soup_cli.utils.paths import ensure_distinct_paths
from soup_cli.utils.ra_dit_run import run_ra_dit
from tests.conftest import strip_ansi


def write_config(
    path: Path,
    output: str,
    stage: str,
    experiment_name: str | None = None,
) -> None:
    config = {
        "output": output,
        "training": {"ra_dit_stage": stage},
    }
    if experiment_name is not None:
        config["experiment_name"] = experiment_name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


@pytest.mark.parametrize(
    ("retriever_output", "generator_output"),
    [
        ("./output", "./output"),
        ("output", "./output"),
        ("output", "unused/../output"),
        ("output", "output/generator"),
        ("output/retriever", "output"),
    ],
    ids=["equal", "dot-alias", "dotdot-alias", "generator-child", "retriever-child"],
)
def test_ra_dit_rejects_overlapping_outputs_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retriever_output: str,
    generator_output: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("unused").mkdir()
    retriever = Path("retriever.yaml")
    generator = Path("generator.yaml")
    write_config(retriever, retriever_output, "retriever")
    write_config(generator, generator_output, "generator")

    original = {
        path.name: path.read_bytes()
        for path in (retriever, generator)
    }
    calls = []

    def fake_runner(config_path: str) -> None:
        calls.append(config_path)

    try:
        with pytest.raises(ValueError) as error:
            run_ra_dit(
                str(retriever),
                str(generator),
                _runner=fake_runner,
            )
        message = str(error.value)
        assert "--retriever-config" in message
        assert "--generator-config" in message
    finally:
        assert calls == [], "Training started before rejecting overlapping outputs"
        assert {
            path.name: path.read_bytes()
            for path in (retriever, generator)
        } == original
        assert not Path("output").exists()


def test_ra_dit_compares_effective_experiment_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_config(Path("retriever.yaml"), "./output", "retriever", "r1")
    write_config(Path("generator.yaml"), "./output/r1", "generator")
    calls = []

    try:
        with pytest.raises(ValueError):
            run_ra_dit(
                "retriever.yaml",
                "generator.yaml",
                _runner=calls.append,
            )
    finally:
        assert calls == [], "Training started despite effective output collision"


def test_ra_dit_links_effective_retriever_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_config(Path("retriever.yaml"), "./output", "retriever", "r1")
    write_config(Path("generator.yaml"), "./output", "generator", "g1")
    seen = []

    def fake_runner(config_path: str) -> None:
        config = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        )
        seen.append(config)

    result = run_ra_dit(
        "retriever.yaml",
        "generator.yaml",
        _runner=fake_runner,
    )

    expected_retriever = (tmp_path / "output" / "r1").resolve()
    expected_generator = (tmp_path / "output" / "g1").resolve()

    assert len(seen) == 2
    assert Path(result.retriever_output).resolve() == expected_retriever
    assert Path(result.generator_output).resolve() == expected_generator
    assert Path(result.retriever_model_used).resolve() == expected_retriever
    assert (
        Path(seen[1]["training"]["ra_dit_retriever_model"]).resolve()
        == expected_retriever
    )
    assert result.autolinked is True

@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("models", "models"),
        ("models", "./models"),
        ("models", "models/generator"),
        ("models/retriever", "models"),
    ],
    ids=["equal", "alias", "child", "parent"],
)
def test_directory_guard_rejects_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: str,
    second: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as error:
        ensure_distinct_paths(
            [("--retriever-config output", first)],
            [("--generator-config output", second)],
            reject_nested=True,
        )

    assert "--retriever-config" in str(error.value)
    assert "--generator-config" in str(error.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("models/r1", "models/g1"),
        ("models", "models-other"),
        ("retriever", "generator"),
    ],
    ids=["siblings", "shared-string-prefix", "separate"],
)
def test_directory_guard_allows_separate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first: str,
    second: str,
) -> None:
    monkeypatch.chdir(tmp_path)

    ensure_distinct_paths(
        [("--retriever-config output", first)],
        [("--generator-config output", second)],
        reject_nested=True,
    )

    assert list(tmp_path.iterdir()) == []


def test_directory_guard_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    ensure_distinct_paths(
        [("--input", "models")],
        [("--output", "models/result.jsonl")],
    )

    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize("plan_only", [False, True], ids=["live", "plan"])
@pytest.mark.parametrize(
    ("retriever_output", "generator_output"),
    [
        ("output", "./output"),
        ("output", "output/generator"),
        ("output/retriever", "output"),
    ],
    ids=["equal", "generator-child", "retriever-child"],
)
def test_cli_rejects_overlapping_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_only: bool,
    retriever_output: str,
    generator_output: str,
) -> None:
    from typer.testing import CliRunner

    from soup_cli.commands.ra_dit import app
    from soup_cli.utils import ra_dit_run

    monkeypatch.chdir(tmp_path)
    write_config(Path("retriever.yaml"), retriever_output, "retriever")
    write_config(Path("generator.yaml"), generator_output, "generator")
    calls = []

    def forbidden_train(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Training must not start")

    monkeypatch.setattr(
        ra_dit_run, "_run_train_subprocess", forbidden_train
    )

    args = [
        "--retriever-config", "retriever.yaml",
        "--generator-config", "generator.yaml",
    ]
    if plan_only:
        args.append("--plan-only")

    result = CliRunner().invoke(app, args)
    message = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == 2, (result.output, result.exception)
    assert "--retriever-config" in message
    assert "--generator-config" in message
    assert calls == []
    assert not Path("output").exists()


def test_plan_previews_effective_output_without_registry_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from soup_cli.commands.ra_dit import app
    from soup_cli.utils import ra_dit_run

    monkeypatch.chdir(tmp_path)
    write_config(Path("retriever.yaml"), "output", "retriever", "r1")
    write_config(Path("generator.yaml"), "output", "generator", "g1")
    calls = []

    def forbidden_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Plan must not access Registry or start training")

    monkeypatch.setattr(
        ra_dit_run, "_run_train_subprocess", forbidden_call
    )
    monkeypatch.setattr(
        ra_dit_run, "resolve_retriever_for_generator", forbidden_call
    )

    result = CliRunner().invoke(
        app,
        [
            "--retriever-config", "retriever.yaml",
            "--generator-config", "generator.yaml",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert str(Path("output") / "r1") in strip_ansi(result.output)
    assert str(Path("output") / "g1") in strip_ansi(result.output)
    assert "auto-link to stage-1 output" in strip_ansi(result.output)
    assert calls == []
    assert not Path("output").exists()

@pytest.mark.parametrize("plan_only", [False, True], ids=["live", "plan"])
@pytest.mark.parametrize(
    "retriever_output,retriever_experiment,generator_output,generator_experiment",
    [
        ("output", "r1", "output/r1", None),
        ("output", "r1", "output/r1", "g1"),
        ("output/r1", "child", "output", "r1"),
        ("output", "r1", "OUTPUT", "R1"),
    ],
    ids=[
        "effective-equal",
        "effective-generator-child",
        "effective-retriever-child",
        "windows-case",
    ],
)
def test_cli_rejects_effective_output_overlap_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan_only: bool,
    retriever_output: str,
    retriever_experiment: str,
    generator_output: str,
    generator_experiment: str | None,
) -> None:
    import os

    from typer.testing import CliRunner

    from soup_cli.commands.ra_dit import app
    from soup_cli.utils import ra_dit_run

    monkeypatch.chdir(tmp_path)
    if generator_output == "OUTPUT":
        if os.name != "nt":
            pytest.skip("Windows case-alias test")
        Path("output").mkdir()
        if not os.path.exists("OUTPUT") or not os.path.samefile("output", "OUTPUT"):
            pytest.skip("Working directory is case-sensitive")

    for filename, output, stage, experiment in [
        ("retriever.yaml", retriever_output, "retriever", retriever_experiment),
        ("generator.yaml", generator_output, "generator", generator_experiment),
    ]:
        config = {
            "output": output,
            "training": {"ra_dit_stage": stage},
        }
        if experiment is not None:
            config["experiment_name"] = experiment
        Path(filename).write_text(yaml.safe_dump(config), encoding="utf-8")

    calls = []

    def forbidden_training(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Training started before rejecting effective output overlap")

    monkeypatch.setattr(ra_dit_run, "_run_train_subprocess", forbidden_training)

    def snapshot() -> dict[str, bytes | None]:
        return {
            path.relative_to(tmp_path).as_posix(): (
                path.read_bytes() if path.is_file() else None
            )
            for path in tmp_path.rglob("*")
        }

    before = snapshot()
    arguments = [
        "--retriever-config",
        "retriever.yaml",
        "--generator-config",
        "generator.yaml",
    ]
    if plan_only:
        arguments.append("--plan-only")

    result = CliRunner().invoke(app, arguments)
    message = " ".join(strip_ansi(result.output).split())

    assert snapshot() == before
    assert calls == []
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "--retriever-config" in message
    assert "--generator-config" in message
    assert "same file" in message or "overlaps" in message
