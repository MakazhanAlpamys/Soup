"""Interactive native consumer contracts for the shared MLX generation component."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_native_chat_keeps_history_clear_system_and_quit(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation

    histories = []
    monkeypatch.setattr(mlx_generation, 'load_native_model', lambda *a, **k: (None, None))

    def generate(model, tokenizer, messages, **kwargs):
        histories.append([dict(message) for message in messages])
        return f'answer-{len(histories)}', 3

    monkeypatch.setattr(mlx_generation, 'generate_native', generate)
    result = runner.invoke(app, ['chat', '--backend', 'mlx', '--model', str(tmp_path)],
                           input='one\ntwo\n/clear\n/system guide\nthree\n/quit\n')
    assert result.exit_code == 0, result.output
    assert histories == [
        [{'role': 'user', 'content': 'one'}],
        [{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': 'answer-1'},
         {'role': 'user', 'content': 'two'}],
        [{'role': 'system', 'content': 'guide'}, {'role': 'user', 'content': 'three'}],
    ]
    assert 'Assistant: answer-3' in result.output and 'Goodbye' in result.output


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, EOFError])
def test_native_chat_cancels_without_an_assistant_reply(tmp_path, monkeypatch, interruption):
    from soup_cli.utils import mlx_generation

    monkeypatch.setattr(mlx_generation, 'load_native_model', lambda *a, **k: (None, None))

    def generate(*args, **kwargs):
        raise interruption()

    monkeypatch.setattr(mlx_generation, 'generate_native', generate)
    result = runner.invoke(app, ['chat', '--backend', 'mlx', '--model', str(tmp_path)],
                           input='hello\n')
    assert result.exit_code == 0, result.output
    assert 'Goodbye' in result.output and 'Assistant:' not in result.output


@pytest.mark.parametrize('options,diagnostic', [
    (['--temperature', 'nan'], 'temperature'), (['--device', 'cuda'], 'device'),
    (['--max-tokens', '0'], 'max_tokens'), (['--hub', 'modelscope'], 'hub'),
])
def test_native_chat_option_refusals_precede_loading(tmp_path, options, diagnostic):
    result = runner.invoke(app, ['chat', '--backend', 'mlx', '--model', str(tmp_path)] + options)
    assert result.exit_code == 1
    assert diagnostic in result.output and 'Model loaded' not in result.output


@pytest.mark.parametrize('native', [False, True])
def test_chat_refuses_backend_artifact_mismatch(tmp_path, native):
    config = {'model': 'owner/base', 'lora_parameters': {}} if native else {
        'base_model_name_or_path': 'owner/base', 'r': 4,
    }
    (tmp_path / 'adapter_config.json').write_text(json.dumps(config))
    options = [] if native else ['--backend', 'mlx']
    result = runner.invoke(app, ['chat', '--model', str(tmp_path)] + options)
    assert result.exit_code == 1
    assert ('select --backend mlx' if native else 'native MLX metadata') in result.output
    assert 'Model loaded' not in result.output
