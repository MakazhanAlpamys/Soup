"""Tests for soup deploy ollama command and Ollama utilities."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.ollama import (
    ALLOWED_OLLAMA_PARAMS,
    FORMAT_TO_TEMPLATE,
    OLLAMA_TEMPLATES,
    SOUP_MODEL_PREFIX,
    create_modelfile,
    deploy_to_ollama,
    detect_ollama,
    infer_chat_template,
    list_soup_models,
    remove_model,
    validate_gguf_path,
    validate_model_name,
)

runner = CliRunner()

# Patch targets — lazy imports in deploy.py resolve to soup_cli.utils.ollama
_OLLAMA = "soup_cli.utils.ollama"


# ─── validate_model_name ───


def test_validate_model_name_valid():
    valid, err = validate_model_name("my-model")
    assert valid is True
    assert err == ""


def test_validate_model_name_with_colon():
    valid, err = validate_model_name("soup-model:latest")
    assert valid is True


def test_validate_model_name_with_dots():
    valid, err = validate_model_name("my.model.v2")
    assert valid is True


def test_validate_model_name_empty():
    valid, err = validate_model_name("")
    assert valid is False
    assert "empty" in err.lower()


def test_validate_model_name_too_long():
    valid, err = validate_model_name("a" * 129)
    assert valid is False
    assert "long" in err.lower()


def test_validate_model_name_with_slash():
    valid, err = validate_model_name("bad/name")
    assert valid is False
    assert "path separator" in err.lower()


def test_validate_model_name_with_backslash():
    valid, err = validate_model_name("bad\\name")
    assert valid is False
    assert "path separator" in err.lower()


def test_validate_model_name_with_null():
    valid, err = validate_model_name("bad\0name")
    assert valid is False
    assert "null" in err.lower()


def test_validate_model_name_starts_with_hyphen():
    valid, err = validate_model_name("-bad")
    assert valid is False
    assert "alphanumeric" in err.lower()


def test_validate_model_name_exactly_128_chars():
    valid, err = validate_model_name("a" * 128)
    assert valid is True
    assert err == ""


# ─── validate_gguf_path ───


def test_validate_gguf_path_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake gguf")
    valid, err = validate_gguf_path(gguf)
    assert valid is True
    assert err == ""


def test_validate_gguf_path_not_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    valid, err = validate_gguf_path(tmp_path / "missing.gguf")
    assert valid is False
    assert "not found" in err.lower()


def test_validate_gguf_path_is_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    valid, err = validate_gguf_path(tmp_path)
    assert valid is False
    assert "not a file" in err.lower()


def test_validate_gguf_path_bad_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "model.bin"
    bad.write_bytes(b"fake")
    valid, err = validate_gguf_path(bad)
    assert valid is False
    assert ".gguf" in err.lower()


def test_validate_gguf_path_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.gguf"
    outside.write_bytes(b"fake")
    try:
        valid, err = validate_gguf_path(outside)
        assert valid is False
        assert "current working directory" in err.lower()
    finally:
        outside.unlink(missing_ok=True)


# ─── detect_ollama ───


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_installed(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="ollama version is 0.6.2", stderr=""
    )
    version = detect_ollama()
    assert version == "0.6.2"


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_not_installed(mock_run):
    mock_run.side_effect = FileNotFoundError
    assert detect_ollama() is None


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)
    assert detect_ollama() is None


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_nonzero(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    assert detect_ollama() is None


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_no_version_match(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="ollama unknown", stderr=""
    )
    version = detect_ollama()
    assert version == "ollama unknown"


@patch(f"{_OLLAMA}.subprocess.run")
def test_detect_ollama_version_in_stderr(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0, stdout="", stderr="ollama version is 0.7.0"
    )
    version = detect_ollama()
    assert version == "0.7.0"


# ─── infer_chat_template ───


def test_infer_chatml():
    assert infer_chat_template("chatml") == "chatml"


def test_infer_alpaca():
    assert infer_chat_template("alpaca") == "llama"


def test_infer_sharegpt():
    assert infer_chat_template("sharegpt") == "chatml"


def test_infer_mistral():
    assert infer_chat_template("mistral") == "mistral"


def test_infer_unknown():
    assert infer_chat_template("xyz") is None


def test_infer_none():
    assert infer_chat_template(None) is None


def test_infer_case_insensitive():
    assert infer_chat_template("ChatML") == "chatml"


# ─── create_modelfile ───


def test_create_modelfile_basic():
    result = create_modelfile(Path("model.gguf"))
    assert "FROM" in result
    assert "model.gguf" in result
    assert "TEMPLATE" not in result


def test_create_modelfile_with_template():
    result = create_modelfile(Path("model.gguf"), template="chatml")
    assert "model.gguf" in result
    assert "TEMPLATE" in result
    assert "im_start" in result


def test_create_modelfile_with_system():
    result = create_modelfile(Path("m.gguf"), system_prompt="You are helpful.")
    assert 'SYSTEM "You are helpful."' in result


def test_create_modelfile_with_system_quotes():
    result = create_modelfile(Path("m.gguf"), system_prompt='Say "hello"')
    assert 'SYSTEM "Say \\"hello\\""' in result


def test_create_modelfile_with_params():
    result = create_modelfile(
        Path("m.gguf"), parameters={"temperature": "0.7", "top_p": "0.9"}
    )
    assert "PARAMETER temperature 0.7" in result
    assert "PARAMETER top_p 0.9" in result


def test_create_modelfile_full():
    result = create_modelfile(
        Path("out") / "model.gguf",
        template="llama",
        system_prompt="Be concise.",
        parameters={"temperature": "0.5"},
    )
    assert "FROM" in result
    assert "model.gguf" in result
    assert "TEMPLATE" in result
    assert "begin_of_text" in result
    assert 'SYSTEM "Be concise."' in result
    assert "PARAMETER temperature 0.5" in result


def test_create_modelfile_custom_template():
    custom = "{{ .Prompt }}"
    result = create_modelfile(Path("m.gguf"), template=custom)
    assert custom in result


def test_create_modelfile_rejects_unknown_param():
    with pytest.raises(ValueError, match="Unknown Ollama parameter"):
        create_modelfile(Path("m.gguf"), parameters={"evil_key": "value"})


def test_create_modelfile_rejects_newline_in_param_value():
    with pytest.raises(ValueError, match="illegal characters"):
        create_modelfile(
            Path("m.gguf"), parameters={"temperature": "0.7\nSYSTEM injected"}
        )


def test_create_modelfile_allows_valid_params():
    result = create_modelfile(
        Path("m.gguf"),
        parameters={"temperature": "0.7", "top_k": "40", "seed": "42"},
    )
    assert "PARAMETER temperature 0.7" in result
    assert "PARAMETER top_k 40" in result
    assert "PARAMETER seed 42" in result


# ─── deploy_to_ollama ───


@patch(f"{_OLLAMA}.subprocess.run")
def test_deploy_to_ollama_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="success")
    success, msg = deploy_to_ollama("test-model", "FROM m.gguf\n")
    assert success is True
    assert "success" in msg


@patch(f"{_OLLAMA}.subprocess.run")
def test_deploy_to_ollama_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="error creating model")
    success, msg = deploy_to_ollama("test-model", "FROM m.gguf\n")
    assert success is False
    assert "error" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_deploy_to_ollama_not_found(mock_run):
    mock_run.side_effect = FileNotFoundError
    success, msg = deploy_to_ollama("test-model", "FROM m.gguf\n")
    assert success is False
    assert "not found" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_deploy_to_ollama_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=300)
    success, msg = deploy_to_ollama("test-model", "FROM m.gguf\n")
    assert success is False
    assert "timed out" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_deploy_to_ollama_oserror(mock_run):
    mock_run.side_effect = OSError("Permission denied")
    success, msg = deploy_to_ollama("test-model", "FROM m.gguf\n")
    assert success is False
    assert "failed to run ollama" in msg.lower()


# ─── list_soup_models ───


@patch(f"{_OLLAMA}.subprocess.run")
def test_list_soup_models_found(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=(
            "NAME                    ID              SIZE      MODIFIED\n"
            "soup-my-model:latest    abc123def456    4.1 GB    2 hours ago\n"
            "llama3.1:latest         xyz789abc123    8.0 GB    3 days ago\n"
            "soup-code:latest        def456ghi789    3.2 GB    1 day ago\n"
        ),
    )
    models = list_soup_models()
    assert len(models) == 2
    assert models[0]["name"] == "soup-my-model:latest"
    assert models[1]["name"] == "soup-code:latest"


@patch(f"{_OLLAMA}.subprocess.run")
def test_list_soup_models_empty(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="NAME    ID    SIZE    MODIFIED\n",
    )
    assert list_soup_models() == []


@patch(f"{_OLLAMA}.subprocess.run")
def test_list_soup_models_ollama_not_found(mock_run):
    mock_run.side_effect = FileNotFoundError
    assert list_soup_models() == []


@patch(f"{_OLLAMA}.subprocess.run")
def test_list_soup_models_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)
    assert list_soup_models() == []


@patch(f"{_OLLAMA}.subprocess.run")
def test_list_soup_models_nonzero(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="")
    assert list_soup_models() == []


# ─── remove_model ───


@patch(f"{_OLLAMA}.subprocess.run")
def test_remove_model_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="deleted 'soup-test'")
    success, msg = remove_model("soup-test")
    assert success is True


@patch(f"{_OLLAMA}.subprocess.run")
def test_remove_model_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="model not found")
    success, msg = remove_model("soup-test")
    assert success is False
    assert "not found" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_remove_model_not_installed(mock_run):
    mock_run.side_effect = FileNotFoundError
    success, msg = remove_model("soup-test")
    assert success is False
    assert "not found" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_remove_model_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=30)
    success, msg = remove_model("soup-test")
    assert success is False
    assert "timed out" in msg.lower()


@patch(f"{_OLLAMA}.subprocess.run")
def test_remove_model_oserror(mock_run):
    mock_run.side_effect = OSError("Permission denied")
    success, msg = remove_model("soup-test")
    assert success is False
    assert "failed to run ollama" in msg.lower()


# ─── Constants ───


def test_ollama_templates_keys():
    assert "chatml" in OLLAMA_TEMPLATES
    assert "llama" in OLLAMA_TEMPLATES
    assert "mistral" in OLLAMA_TEMPLATES
    assert "vicuna" in OLLAMA_TEMPLATES
    assert "zephyr" in OLLAMA_TEMPLATES


def test_format_to_template_mapping():
    assert FORMAT_TO_TEMPLATE["chatml"] == "chatml"
    assert FORMAT_TO_TEMPLATE["alpaca"] == "llama"
    assert FORMAT_TO_TEMPLATE["sharegpt"] == "chatml"


def test_soup_model_prefix():
    assert SOUP_MODEL_PREFIX == "soup-"


def test_allowed_ollama_params():
    assert "temperature" in ALLOWED_OLLAMA_PARAMS
    assert "top_p" in ALLOWED_OLLAMA_PARAMS
    assert "seed" in ALLOWED_OLLAMA_PARAMS
    assert len(ALLOWED_OLLAMA_PARAMS) >= 10


# ─── CLI: soup deploy ollama --help ───


def test_deploy_ollama_help():
    result = runner.invoke(app, ["deploy", "ollama", "--help"])
    assert result.exit_code == 0
    assert "ollama" in result.output.lower()
    assert "model" in result.output.lower()
    assert "name" in result.output.lower()


def test_deploy_help():
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert "ollama" in result.output.lower()


# ─── CLI: soup deploy ollama --list ───


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
@patch(f"{_OLLAMA}.list_soup_models", return_value=[])
def test_deploy_list_empty(mock_list, mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--list"])
    assert result.exit_code == 0
    assert "no soup-deployed" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
@patch(
    f"{_OLLAMA}.list_soup_models",
    return_value=[{"name": "soup-test:latest", "size": "4.1 GB"}],
)
def test_deploy_list_with_models(mock_list, mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--list"])
    assert result.exit_code == 0
    assert "soup-test" in result.output


@patch(f"{_OLLAMA}.detect_ollama", return_value=None)
def test_deploy_list_no_ollama(mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--list"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


# ─── CLI: soup deploy ollama --remove ───


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
@patch(f"{_OLLAMA}.remove_model", return_value=(True, "deleted"))
def test_deploy_remove_success(mock_rm, mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--remove", "soup-test", "--yes"])
    assert result.exit_code == 0
    assert "deleted" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
@patch(f"{_OLLAMA}.remove_model", return_value=(False, "not found"))
def test_deploy_remove_failure(mock_rm, mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--remove", "soup-bad", "--yes"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_deploy_remove_invalid_name():
    result = runner.invoke(
        app, ["deploy", "ollama", "--remove", "bad/name", "--yes"]
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value=None)
def test_deploy_remove_no_ollama(mock_detect):
    result = runner.invoke(app, ["deploy", "ollama", "--remove", "soup-test", "--yes"])
    assert result.exit_code == 1


# ─── CLI: soup deploy ollama (deploy mode) ───


def test_deploy_missing_model():
    result = runner.invoke(app, ["deploy", "ollama"])
    assert result.exit_code == 1
    assert "--model" in result.output


def test_deploy_missing_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(app, ["deploy", "ollama", "--model", str(gguf)])
    assert result.exit_code == 1
    assert "--name" in result.output


def test_deploy_invalid_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(
        app, ["deploy", "ollama", "--model", str(gguf), "--name", "bad/name"]
    )
    assert result.exit_code == 1
    assert "invalid" in result.output.lower()


def test_deploy_gguf_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["deploy", "ollama", "--model", "missing.gguf", "--name", "soup-test"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value=None)
def test_deploy_ollama_not_installed(mock_detect, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(
        app,
        ["deploy", "ollama", "--model", str(gguf), "--name", "soup-test"],
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_deploy_invalid_template(mock_detect, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "deploy", "ollama",
            "--model", str(gguf),
            "--name", "soup-test",
            "--template", "nonexistent",
        ],
    )
    assert result.exit_code == 1
    assert "unknown template" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_deploy_bad_parameter_format(mock_detect, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "deploy", "ollama",
            "--model", str(gguf),
            "--name", "soup-test",
            "--template", "chatml",
            "--parameter", "bad_no_equals",
            "--yes",
        ],
    )
    assert result.exit_code == 1
    assert "invalid parameter" in result.output.lower()


@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_deploy_unknown_parameter_key(mock_detect, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake")
    result = runner.invoke(
        app,
        [
            "deploy", "ollama",
            "--model", str(gguf),
            "--name", "soup-test",
            "--template", "chatml",
            "--parameter", "evil_key=value",
            "--yes",
        ],
    )
    assert result.exit_code == 1
    assert "invalid parameter" in result.output.lower()


@patch(f"{_OLLAMA}.deploy_to_ollama", return_value=(True, "created"))
@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_deploy_full_success(mock_detect_fn, mock_deploy_fn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake gguf data")
    result = runner.invoke(
        app,
        [
            "deploy", "ollama",
            "--model", str(gguf),
            "--name", "soup-test",
            "--template", "chatml",
            "--system", "You are helpful.",
            "--parameter", "temperature=0.7",
            "--yes",
        ],
    )
    assert result.exit_code == 0
    assert "soup-test" in result.output
    assert "deploy complete" in result.output.lower()


@patch(f"{_OLLAMA}.deploy_to_ollama", return_value=(False, "disk full"))
@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_deploy_create_fails(mock_detect_fn, mock_deploy_fn, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fake gguf data")
    result = runner.invoke(
        app,
        [
            "deploy", "ollama",
            "--model", str(gguf),
            "--name", "soup-test",
            "--template", "chatml",
            "--yes",
        ],
    )
    assert result.exit_code == 1
    assert "disk full" in result.output.lower()


# ─── CLI: soup export --deploy ───


def test_export_deploy_flag_in_help():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output.lower()


# ─── _auto_detect_template ───


def test_auto_detect_from_soup_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: test\ndata:\n  train: data.jsonl\n  format: chatml\n",
        encoding="utf-8",
    )
    from soup_cli.commands.deploy import _auto_detect_template

    result = _auto_detect_template()
    assert result == "chatml"


def test_auto_detect_no_soup_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from soup_cli.commands.deploy import _auto_detect_template

    result = _auto_detect_template()
    assert result is None


def test_auto_detect_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(":::invalid", encoding="utf-8")
    from soup_cli.commands.deploy import _auto_detect_template

    result = _auto_detect_template()
    assert result is None


def test_auto_detect_no_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "soup.yaml").write_text(
        "base: test\ndata:\n  train: data.jsonl\n",
        encoding="utf-8",
    )
    from soup_cli.commands.deploy import _auto_detect_template

    result = _auto_detect_template()
    assert result is None


# ─── Export _auto_deploy_ollama ───


def test_export_deploy_unsupported_target():
    """_auto_deploy_ollama rejects non-ollama targets."""
    from soup_cli.commands.export import _auto_deploy_ollama

    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _auto_deploy_ollama(Path("m.gguf"), "model", "kubernetes", None)


@patch(f"{_OLLAMA}.detect_ollama", return_value=None)
def test_export_deploy_ollama_not_found(mock_detect):
    """_auto_deploy_ollama exits if Ollama not installed."""
    from soup_cli.commands.export import _auto_deploy_ollama

    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _auto_deploy_ollama(Path("m.gguf"), "model", "ollama", None)


@patch(f"{_OLLAMA}.deploy_to_ollama", return_value=(True, "ok"))
@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_export_deploy_ollama_success(mock_detect_fn, mock_deploy_fn):
    """_auto_deploy_ollama succeeds with valid inputs."""
    from soup_cli.commands.export import _auto_deploy_ollama

    _auto_deploy_ollama(Path("model.gguf"), "mymodel", "ollama", "soup-mymodel")
    mock_deploy_fn.assert_called_once()


@patch(f"{_OLLAMA}.deploy_to_ollama", return_value=(False, "fail"))
@patch(f"{_OLLAMA}.detect_ollama", return_value="0.6.2")
def test_export_deploy_ollama_create_fails(mock_detect_fn, mock_deploy_fn):
    """_auto_deploy_ollama exits on deploy failure."""
    from soup_cli.commands.export import _auto_deploy_ollama

    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _auto_deploy_ollama(Path("m.gguf"), "model", "ollama", "soup-model")


def test_export_deploy_invalid_name():
    """_auto_deploy_ollama rejects invalid model name."""
    from soup_cli.commands.export import _auto_deploy_ollama

    with pytest.raises((SystemExit, click.exceptions.Exit)):
        _auto_deploy_ollama(Path("m.gguf"), "model", "ollama", "bad/name")
