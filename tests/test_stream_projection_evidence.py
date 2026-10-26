"""The archived result must not become a pass through rounding or missing evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RECORD = ROOT / 'benchmarks/results/issue2162-rtx3090-v1'


def helper():
    path = ROOT / 'benchmarks/harness/issue2162_audit.py'
    spec = importlib.util.spec_from_file_location('projection_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_record_recomputes_all_three_experiments():
    result = helper().audit(RECORD)
    assert result['verified']
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
