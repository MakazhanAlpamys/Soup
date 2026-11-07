from pathlib import Path

import pytest
import yaml

from soup_cli.utils.ra_dit_run import run_ra_dit


@pytest.mark.parametrize(
    "fail_generator",
    [False, True],
    ids=["success", "generator-failure"],
)
def test_ra_dit_preserves_input_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_generator: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    retriever = Path("retriever.yaml")
    generator = Path("generator.yaml")

    retriever.write_bytes(
        b"# Original retriever config\n"
        b"output: ./out/retriever\n"
        b"training:\n"
        b"  ra_dit_stage: retriever\n"
    )
    generator.write_bytes(
        b"# Original generator config\n"
        b"output: ./out/generator\n"
        b"training:\n"
        b"  ra_dit_stage: generator\n"
    )
    original_retriever = retriever.read_bytes()
    original_generator = generator.read_bytes()
    calls: list[Path] = []

    def fake_runner(config_path: str) -> None:
        path = Path(config_path)
        calls.append(path)
        if len(calls) == 1:
            assert path.resolve() == retriever.resolve()
            return

        assert path.resolve() != generator.resolve()
        assert path.resolve() != retriever.resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (
            config["training"]["ra_dit_retriever_model"]
            == "./out/retriever"
        )
        if fail_generator:
            raise RuntimeError("Simulated generator failure")

    try:
        if fail_generator:
            with pytest.raises(RuntimeError, match="Simulated generator failure"):
                run_ra_dit(
                    str(retriever),
                    str(generator),
                    _runner=fake_runner,
                )
        else:
            result = run_ra_dit(
                str(retriever),
                str(generator),
                _runner=fake_runner,
            )
            assert result.autolinked is True
    finally:
        assert retriever.read_bytes() == original_retriever
        assert generator.read_bytes() == original_generator

    assert len(calls) == 2
    assert not calls[1].exists()
    assert not calls[1].parent.exists()
