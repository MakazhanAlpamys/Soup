"""Interactive native consumer contracts for the shared MLX generation component."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()


def test_native_chat_keeps_history_clear_system_and_quit(tmp_path, monkeypatch):
    from soup_cli.utils import mlx_generation

    histories, loads, generations = [], [], []
    model_obj, tokenizer_obj = object(), object()

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return model_obj, tokenizer_obj

    monkeypatch.setattr(mlx_generation, 'load_native_model', load)

    def generate(model, tokenizer, messages, **kwargs):
        assert model is model_obj and tokenizer is tokenizer_obj
        generations.append(kwargs)
        histories.append([dict(message) for message in messages])
        return f'answer-{len(histories)}', 3

    monkeypatch.setattr(mlx_generation, 'generate_native', generate)
    result = runner.invoke(app, ['chat', '--backend', 'mlx', '--model', str(tmp_path),
                                '--base', str(tmp_path / 'base'), '--trust-remote-code',
                                '--max-tokens', '7', '--temperature', '0.25'],
                           input='one\ntwo\n/clear\n/system guide\nthree\n/quit\n')
    assert result.exit_code == 0, result.output
    assert loads == [((str(tmp_path), str(tmp_path / 'base')), {'trust_remote_code': True})]
    assert generations == [{'max_tokens': 7, 'temperature': 0.25}] * 3
    assert histories == [
        [{'role': 'user', 'content': 'one'}],
        [{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': 'answer-1'},
         {'role': 'user', 'content': 'two'}],
        [{'role': 'system', 'content': 'guide'}, {'role': 'user', 'content': 'three'}],
    ]
    plain = ' '.join(strip_ansi(result.output).split())
    assert 'Assistant: answer-3' in plain and 'Goodbye' in plain


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
    plain = ' '.join(strip_ansi(result.output).split())
    assert 'Goodbye' in plain and 'Assistant:' not in plain


@pytest.mark.parametrize('options,code,diagnostic', [
    (['--temperature', 'nan'], 1, 'temperature'), (['--device', 'cuda'], 1, 'device'),
    (['--max-tokens', '0'], 1, 'max_tokens'), (['--hub', 'modelscope'], 1, 'hub'),
    (['--backend', 'unknown'], 2, 'must be transformers or mlx'),
])
def test_native_chat_option_refusals_precede_loading(tmp_path, options, code, diagnostic):
    result = runner.invoke(app, ['chat', '--backend', 'mlx', '--model', str(tmp_path)] + options)
    assert result.exit_code == code
    plain = ' '.join(strip_ansi(result.output).split())
    assert diagnostic in plain and 'Model loaded' not in plain


@pytest.mark.parametrize('native', [False, True])
def test_chat_refuses_backend_artifact_mismatch(tmp_path, native):
    config = {'model': 'owner/base', 'lora_parameters': {}} if native else {
        'base_model_name_or_path': 'owner/base', 'r': 4,
    }
    (tmp_path / 'adapter_config.json').write_text(json.dumps(config))
    options = [] if native else ['--backend', 'mlx']
    result = runner.invoke(app, ['chat', '--model', str(tmp_path)] + options)
    assert result.exit_code == 1
    plain = ' '.join(strip_ansi(result.output).split())
    assert ('select --backend mlx' if native else 'native MLX metadata') in plain
    assert 'Model loaded' not in plain
