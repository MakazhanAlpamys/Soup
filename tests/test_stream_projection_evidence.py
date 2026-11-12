"""The archived result must not become a pass through rounding or missing evidence."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RECORD = ROOT / 'benchmarks/results/issue2162-rtx3090-v1'


@pytest.fixture(autouse=True)
def harness_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/harness'))


def helper():
    path = ROOT / 'benchmarks/harness/issue2162_audit.py'
    spec = importlib.util.spec_from_file_location('projection_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numbers_only_recomputes_all_three_experiments():
    result = helper().audit(RECORD)
    assert result['mode'] == 'numbers-only' and not result['verified']
    assert result['numbers_verified']
    assert len(result['missing_bulk_files']) == 11
    assert result['unavailable_checks']
    assert result['timing_processes'] == 56
    assert result['exact_update_replays'] == 168
    assert result['measured_updates'] == 2800
    assert result['confirmation_processes'] == 0
    assert all(not row['primary_passes'] for row in result['studies'])
    final = result['studies'][-1]
    assert final['statistics'][0]['median_reduction_percent'] == 1.9979418106243414
    assert final['memory_pass']


def test_near_threshold_and_control_do_not_rescue_primary():
    module = helper()
    primary = module.bootstrap([1.99999] * 6, 3)
    control = module.bootstrap([2.1] * 6, 3)
    assert control['passes_rule'] and not primary['passes_rule']
    assert not module.scoped_eligible(primary, control)


def test_dispatch_diagnostic_must_also_improve():
    module = helper()
    primary = module.bootstrap([2.1] * 6, 3)
    diagnostic = module.bootstrap([0.0] * 6, 3)
    assert not module.scoped_eligible(primary, diagnostic)


@pytest.mark.parametrize('values', [[2.1] * 4, [2.1] * 7, [2.1] * 5 + [float('nan')]])
def test_bootstrap_rejects_changed_roster_or_nonfinite_value(values):
    with pytest.raises(ValueError):
        helper().bootstrap(values, 3)


def test_memory_gate_uses_allocated_delta_and_exact_saved_bytes():
    module = helper()
    assert module.memory_gate(100, 200, 100)['passes']
    assert not module.memory_gate(100, 200 + 64 * 2**20 + 1, 100)['passes']


def test_tensor_change_breaks_original_receipt_hash():
    module = helper()
    refs = json.loads((RECORD/'reference-updates.json').read_text())
    entry = json.loads((RECORD/'scoped-v2-resume1/processes.jsonl').read_text().splitlines()[0])
    refs['2048'][0]['logits']['sha256'] = '0' * 64
    with pytest.raises(AssertionError, match='receipt hash'):
        module.expand_receipt(entry, refs)


def test_retained_data_hash_accepts_crlf_but_rejects_changed_byte(tmp_path):
    module = helper()
    path = tmp_path/'sample.json'
    path.write_bytes(b'{\r\n}\r\n')
    expected = module.digest(b'{\n}\n')
    module.verify_file(path, expected)
    path.write_bytes(b'{\n }\n')
    with pytest.raises(AssertionError):
        module.verify_file(path, expected)


def test_partial_bulk_cannot_claim_full_verification(tmp_path):
    result = helper().audit(RECORD, ROOT, tmp_path)
    assert result['mode'] == 'numbers-only' and not result['verified']
    assert result['sources_verified'] is None
    assert len(result['missing_bulk_files']) == 11


def test_numbers_only_report_lists_missing_operational_checks():
    from issue2162_report import render

    text = render(RECORD, helper().audit(RECORD))
    assert 'Full gate verdict: unavailable' in text
    assert 'issue2162-rtx3090-v1/timing-v1/resources.jsonl' in text
    assert '1.997942%' in text


def test_bulk_hashes_are_checked_and_corruption_cannot_fall_back(tmp_path):
    from issue2162_bulk import inspect

    root, bulk = tmp_path / 'record', tmp_path / 'bulk'
    root.mkdir()
    expected = helper().digest(b'original\n')
    manifest = dict(published_files_sha256={'resources.jsonl': expected},
                    bulk=dict(files_sha256={'resources.jsonl': expected}))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    _, _, missing = inspect(root, None, 'published_files_sha256')
    assert missing == ['record/resources.jsonl']
    target = bulk / 'record/resources.jsonl'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'original\r\n')
    assert not inspect(root, bulk, 'published_files_sha256')[2]
    target.write_bytes(b'changed\n')
    with pytest.raises(AssertionError, match='hash mismatch'):
        inspect(root, bulk, 'published_files_sha256')


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/absolute', 'a\\..\\escape'])
def test_bulk_manifest_rejects_escaping_paths(tmp_path, name):
    from issue2162_bulk import inspect

    value = '0' * 64
    manifest = dict(published_files_sha256={name: value},
                    bulk=dict(files_sha256={name: value}))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='path'):
        inspect(tmp_path, None, 'published_files_sha256')


def test_result_tree_stays_below_two_megabytes_without_bulk():
    files = [p for folder in RECORD.parent.glob('issue2162-*')
             for p in folder.rglob('*') if p.is_file()]
    assert sum(p.stat().st_size for p in files) < 2_000_000
    assert not any(p.name in {'resources.jsonl', 'objects.json', 'source-blobs.json',
                              'source-hashes.json'} for p in files)


def test_full_bulk_recomputes_all_historical_checks():
    if not os.environ.get('SOUP2162_BULK'):
        pytest.skip('set SOUP2162_BULK to the extracted archive for full evidence integration')
    result = helper().audit(RECORD, ROOT, Path(os.environ['SOUP2162_BULK']))
    assert result['mode'] == 'full' and result['verified']
    assert result['sources_verified'] == 2927 and not result['missing_bulk_files']
    expected = json.loads((RECORD / 'independent-audit.json').read_text())
    for key, value in expected.items():
        assert result[key] == value
    from issue2162_report import render

    assert render(RECORD, result) == (
        ROOT / 'benchmarks/gate-2162-auto-qkvo.md').read_text(encoding='utf-8')
