"""Native custom evaluation dispatch and observable score/accounting contracts."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.eval import _save_custom_results as save_custom_results


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'model').mkdir()
    import soup_cli.commands.eval as command
    monkeypatch.setattr(command, '_save_custom_results', lambda *a, **k: None)
    return tmp_path


def invoke(rows, *, backend='mlx', extra=()):
    from pathlib import Path
    Path('tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return CliRunner().invoke(app, [
        'eval', 'custom', '--backend', backend, '--model', 'model', '--tasks', 'tasks.jsonl',
        '--output', 'report.json', *extra,
    ])


def stub_native(monkeypatch, outputs):
    import soup_cli.utils.mlx_generation as native
    monkeypatch.setattr(native, 'load_native_model', lambda *a, **k: (object(), object()))
    iterator = iter(outputs)

    def generate(model, tokenizer, messages, *, max_tokens, temperature):
        assert messages[0]['role'] == 'user'
        assert max_tokens == 512 and temperature == 0
        response = next(iterator)
        if isinstance(response, Exception):
            raise response
        return response, 3

    monkeypatch.setattr(native, 'generate_native', generate)


def test_native_real_strings_scored_with_categories_and_identity(workspace, monkeypatch):
    tool = json.dumps({'function': {'name': 'lookup', 'arguments': {'city': 'Paris'}}})
    stub_native(monkeypatch, ['#### 4', '14', tool, 'wrong'])
    rows = [
        {'prompt': 'sum', 'expected': '4', 'scoring': 'answer', 'category': 'math'},
        {'prompt': 'contains', 'expected': '4', 'scoring': 'contains', 'category': 'math'},
        {'prompt': 'tool', 'expected': tool, 'scoring': 'tool_call_match', 'category': 'tools'},
        {'prompt': 'last', 'expected': 'right', 'category': 'default'},
    ]
    result = invoke(rows)
    assert result.exit_code == 0, (result.output, result.exception)
    payload = json.loads((workspace / 'report.json').read_text())
    assert (payload['total'], payload['correct'], payload['accuracy']) == (4, 2, 0.5)
    assert payload['category_scores']['math'] == {'total': 2, 'correct': 1, 'accuracy': 0.5}
    assert payload['complete'] is True and payload['evaluated'] == payload['requested'] == 4
    assert [r['output'] for r in payload['results']] == ['#### 4', '14', tool, 'wrong']
    assert [r['matched'] for r in payload['results']] == [True, False, True, False]
    assert all(r['tokens_generated'] == 3 for r in payload['results'])
    assert payload['dataset_sha256'] == hashlib.sha256(
        (workspace / 'tasks.jsonl').read_bytes()).hexdigest()
    assert payload['base'] == 'model' and payload['adapter'] is None
    assert payload['backend'] == 'mlx' and 'python' in payload['runtime']
    assert payload['generation']['max_tokens'] == 512
    assert payload['generation']['temperature'] == 0


@pytest.mark.parametrize('stage', ['loading', 'generation', 'scoring', 'nonfinite'])
def test_errors_never_save_a_partial_score_as_complete(workspace, monkeypatch, stage):
    import soup_cli.commands.eval as command
    import soup_cli.eval.custom as custom
    import soup_cli.utils.mlx_generation as native
    stub_native(monkeypatch, ['yes', RuntimeError('native device failed') if stage == 'generation'
                              else 'no'])
    saved = []
    monkeypatch.setattr(command, '_save_custom_results', lambda *a, **k: saved.append(a))
    if stage == 'loading':
        def load(*args, **kwargs):
            raise ValueError('incompatible adapter tensor')
        monkeypatch.setattr(native, 'load_native_model', load)
    elif stage in ('scoring', 'nonfinite'):
        real = custom.score_task
        def score(task, output):
            if task.prompt == 'second':
                if stage == 'scoring':
                    raise ValueError('scorer unavailable')
                return custom.EvalResult(task, output, float('nan'), False)
            return real(task, output)
        monkeypatch.setattr(custom, 'score_task', score)
    result = invoke([{'prompt': 'first', 'expected': 'yes'},
                     {'prompt': 'second', 'expected': 'no'}])
    assert result.exit_code != 0
    assert ('loading' if stage == 'loading' else '2/2') in result.output.lower()
    assert 'incomplete' in result.output.lower()
    assert not (workspace / 'report.json').exists()
    assert saved == []


def test_empty_completion_is_an_actual_incorrect_answer(workspace, monkeypatch):
    stub_native(monkeypatch, [''])
    result = invoke([{'prompt': 'x', 'expected': 'yes'}])
    assert result.exit_code == 0, result.output
    p = json.loads((workspace / 'report.json').read_text())
    assert p['complete'] is True and p['total'] == 1 and p['correct'] == 0
    assert p['results'][0]['output'] == ''


@pytest.mark.parametrize('extra, diagnostic', [
    (('--temperature', 'nan'), 'finite'), (('--max-tokens', '0'), 'max'),
    (('--temperature', '-1'), 'temperature'),
])
def test_invalid_native_controls_refused_before_loading(workspace, extra, diagnostic):
    result = invoke([{'prompt': 'x'}], extra=extra)
    assert result.exit_code != 0 and diagnostic in result.output.lower()
    assert not (workspace / 'report.json').exists()


def test_default_transformers_custom_eval_still_scores(workspace, monkeypatch):
    import soup_cli.eval.custom as custom
    monkeypatch.setattr(custom, '_create_default_generator', lambda *a, **k: lambda p: 'yes')
    result = invoke([{'prompt': 'x', 'expected': 'yes'}], backend='transformers')
    assert result.exit_code == 0, (result.output, result.exception)
    p = json.loads((workspace / 'report.json').read_text())
    assert p['total'] == p['correct'] == 1 and p['accuracy'] == 1


def test_malformed_corpus_refused_before_native_loading(workspace):
    result = invoke([{'expected': 'yes'}])
    assert result.exit_code != 0 and 'prompt' in result.output
    assert not (workspace / 'report.json').exists()


def test_unknown_backend_refused(workspace):
    result = invoke([{'prompt': 'x'}], backend='cuda')
    assert result.exit_code != 0 and 'backend' in result.output.lower()


@pytest.mark.parametrize('response', [None, 123])
def test_nontext_generator_result_is_an_error(workspace, monkeypatch, response):
    stub_native(monkeypatch, [response])
    result = invoke([{'prompt': 'x', 'expected': 'yes'}])
    assert result.exit_code == 1 and 'completion text' in result.output
    assert not (workspace / 'report.json').exists()


def test_native_sampling_controls_are_not_ignored_on_transformers(workspace):
    result = invoke([{'prompt': 'x'}], backend='transformers', extra=('--max-tokens', '8'))
    assert result.exit_code == 1 and 'MLX backend only' in result.output
    assert not (workspace / 'report.json').exists()


def test_native_evidence_is_retained_in_tracker_without_output(workspace, monkeypatch):
    import soup_cli.commands.eval as command
    from soup_cli.experiment.tracker import ExperimentTracker

    # Restore the actual persistence boundary in this otherwise isolated fixture.
    monkeypatch.setattr(command, '_save_custom_results', save_custom_results)
    monkeypatch.setenv('SOUP_DB_PATH', str(workspace / 'tracker.db'))
    stub_native(monkeypatch, ['yes'])
    (workspace / 'tasks.jsonl').write_text('{"prompt":"x","expected":"yes"}\n')
    result = CliRunner().invoke(app, ['eval', 'custom', '--backend', 'mlx', '--model', 'model',
                                     '--tasks', 'tasks.jsonl'])
    assert result.exit_code == 0, (result.output, result.exception)
    tracker = ExperimentTracker()
    rows = tracker.get_eval_results()
    assert len(rows) == 1
    details = json.loads(rows[0]['details_json'])
    assert details['complete'] is True and details['requested'] == details['evaluated'] == 1
    assert details['results'][0]['output'] == 'yes'
    assert details['backend'] == 'mlx' and details['dataset_sha256']
