"""The output help describes the archive written by `soup airgap-bundle`."""

import tarfile

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi


def test_output_help_names_the_uncompressed_tar_format(monkeypatch):
    monkeypatch.setenv("COLUMNS", "160")
    result = make_cli_runner().invoke(app, ["airgap-bundle", "--help"])
    text = " ".join(strip_ansi(result.output).replace("│", " ").split())

    assert result.exit_code == 0, (text, result.exception)
    assert "Output path for the uncompressed tar archive (cwd-contained)" in text


def test_output_with_gzip_suffix_is_an_uncompressed_tar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"tiny model fixture")
    output = tmp_path / "bundle.tar.gz"

    result = make_cli_runner().invoke(
        app, ["airgap-bundle", "--model", "model", "--output", output.name]
    )

    assert result.exit_code == 0, (strip_ansi(result.output), result.exception)
    with output.open("rb") as handle:
        assert handle.read(2) != b"\x1f\x8b", "bundle has a gzip header"
    with tarfile.open(output, "r:") as archive:
        assert archive.getmember("manifest.json").isfile()
