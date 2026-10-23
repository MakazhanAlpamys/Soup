"""Native merge dispatch and safe artifact publication contracts."""

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def _adapter(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text(json.dumps({
        'model': str(base), 'fine_tune_type': 'lora', 'num_layers': 1,
        'lora_parameters': {'rank': 2, 'scale': 3, 'dropout': 0, 'keys': ['lm_head']},
    }))
    (adapter / 'adapters.safetensors').write_bytes(b'not loaded in portable refusals')
    return adapter, base


def test_native_merge_rejects_peft_metadata_without_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter, _ = _adapter(tmp_path)
    (adapter / 'adapter_config.json').write_text('{"base_model_name_or_path":"owner/base"}')
    result = runner.invoke(app, ['merge', '--backend', 'mlx', '--adapter', str(adapter)])
    assert result.exit_code == 1
    assert 'native MLX metadata' in result.output
    assert not (tmp_path / 'merged').exists()


@pytest.mark.parametrize('options, diagnostic', [
    (['--save-format', '4bit'], 'save-format'),
    (['--hub', 'modelscope'], 'hub'),
    (['--double-quant'], 'double-quant'),
    (['--no-double-quant'], 'double-quant'),
    (['--dtype', 'int8'], 'dtype'),
])
def test_native_merge_refuses_unsupported_controls_before_loading(
    tmp_path, monkeypatch, options, diagnostic,
):
    monkeypatch.chdir(tmp_path)
    adapter, _ = _adapter(tmp_path)
    result = runner.invoke(app, ['merge', '--backend', 'mlx', '--adapter', str(adapter)] + options)
    assert result.exit_code != 0
    assert diagnostic in result.output
    assert not (tmp_path / 'merged').exists()


def test_default_merge_refuses_native_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter, _ = _adapter(tmp_path)
    result = runner.invoke(app, ['merge', '--adapter', str(adapter)])
    assert result.exit_code == 1
    assert 'select --backend mlx' in result.output
    assert not (tmp_path / 'merged').exists()


def test_native_merge_wrong_base_is_refused_before_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter, _ = _adapter(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()
    result = runner.invoke(app, ['merge', '--backend', 'mlx', '--adapter', str(adapter),
                                '--base', str(other)])
    assert result.exit_code == 1
    assert 'does not match' in result.output
    assert not (tmp_path / 'merged').exists()


@pytest.mark.parametrize('kind', ['adapter-child', 'base-child', 'ancestor', 'existing', 'escape'])
def test_native_destination_refusals_have_valid_control(tmp_path, monkeypatch, kind):
    from soup_cli.utils.mlx_merge import _validate_destination

    monkeypatch.chdir(tmp_path)
    adapter, base = _adapter(tmp_path)
    _validate_destination(tmp_path / 'valid-output', adapter, str(base))
    outputs = {
        'adapter-child': adapter / 'fused', 'base-child': base / 'fused',
        'ancestor': tmp_path, 'existing': tmp_path / 'existing',
        'escape': tmp_path.parent / 'outside',
    }
    output = outputs[kind]
    if kind == 'existing':
        output.mkdir()
        (output / 'sentinel').write_text('keep')
    expected = 'overlap' if kind in ('adapter-child', 'base-child', 'ancestor') else (
        'already exists' if kind == 'existing' else 'under cwd'
    )
    with pytest.raises(ValueError, match=expected):
        _validate_destination(output, adapter, str(base))
    result = runner.invoke(app, ['merge', '--backend', 'mlx', '--adapter', str(adapter),
                                '--output', str(output)])
    assert result.exit_code == 1
    assert expected in result.output
    assert not list(tmp_path.glob('.soup_mlx_merge_*'))
    if kind == 'existing':
        assert (output / 'sentinel').read_text() == 'keep'


@pytest.mark.requires_symlink
def test_native_merge_symlink_escape_and_dangling_output(tmp_path, monkeypatch):
    from soup_cli.utils.mlx_merge import _validate_destination

    monkeypatch.chdir(tmp_path)
    adapter, base = _adapter(tmp_path)
    escape = tmp_path / 'escape'
    escape.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='under cwd'):
        _validate_destination(escape / 'fused', adapter, str(base))
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'missing', target_is_directory=True)
    with pytest.raises(ValueError, match='already exists'):
        _validate_destination(dangling, adapter, str(base))
