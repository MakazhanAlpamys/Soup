"""Recompute the archived #2162 measurements without CUDA or model execution.

Shared tensor records reconstruct every original exactness receipt byte-for-byte.
Optional Git verification checks all frozen Python blobs in the original commits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import random
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

CAP = 8 * 2**30
GPU = 'GPU-05919122-a78e-9af3-3b3c-f9a14fb4d2f4'
STUDIES = ('timing-v1', 'reallocation-v1', 'scoped-v2-resume1')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def verify_file(path: Path, expected: str) -> None:
    assert digest(path.read_bytes().replace(b'\r\n', b'\n')) == expected, str(path)


def bootstrap(reductions: list[float], family: int) -> dict:
    import numpy as np

    count = 5 if family == 2 else 6
    if family not in (2, 3) or len(reductions) != count:
        raise ValueError('fixed five-pair or six-block roster required')
    if not all(math.isfinite(value) for value in reductions):
        raise ValueError('finite reductions required')
    indices = np.random.default_rng(42).integers(0, count, (10000, count))
    sampled = np.median(np.asarray(reductions, dtype=np.float64)[indices], axis=1)
    ordinary = np.quantile(sampled, [.025, .975], method='linear').tolist()
    adjusted = np.quantile(sampled, [.025/family, 1-.025/family], method='linear').tolist()
    median = statistics.median(reductions)
    return dict(median_reduction_percent=median, ordinary_95=ordinary,
                family_adjusted_ci=adjusted, reductions_percent=reductions,
                passes_rule=median >= 2 and ordinary[0] > 0 and adjusted[0] > 0)


def scoped_eligible(primary: dict, diagnostic: dict) -> bool:
    return (primary['passes_rule'] and diagnostic['ordinary_95'][0] > 0
            and diagnostic['family_adjusted_ci'][0] > 0)


def memory_gate(baseline: int, candidate: int, saved: int) -> dict:
    delta = candidate - baseline
    return dict(allocated_delta_bytes=delta, saved_bytes=saved,
                allowance_bytes=64 * 2**20, passes=delta <= saved + 64 * 2**20)


def expand_receipt(entry: dict, references: dict) -> dict:
    receipt = dict(entry['receipt'])
    receipt['steps'] = references[receipt['steps']['reference_sequence']]
    original = (json.dumps(receipt, indent=2) + '\n').replace('\n', '\r\n').encode()
    assert digest(original) == entry['original_sha256'], 'original receipt hash mismatch'
    return receipt


def verify_sources(root: Path, git_root: Path) -> int:
    bindings = [read(root/study/'source-blobs.json') for study in STUDIES]
    all_blobs = sorted({item['git_blob'] for binding in bindings
                        for item in binding['files'].values()})
    process = subprocess.run(['git', 'cat-file', '--batch'], cwd=git_root, check=True,
                             input=('\n'.join(all_blobs)+'\n').encode(), capture_output=True)
    stream, contents = io.BytesIO(process.stdout), {}
    for blob in all_blobs:
        header = stream.readline().decode().strip().split()
        assert len(header) == 3 and header[:2] == [blob, 'blob'], 'fetch frozen Git history'
        contents[blob] = stream.read(int(header[2]))
        assert stream.read(1) == b'\n'
    total = 0
    for binding in bindings:
        output = subprocess.check_output(['git', 'ls-tree', '-r', binding['commit']],
                                         cwd=git_root, text=True)
        tree = {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2]
                for line in output.splitlines()}
        for name, item in binding['files'].items():
            assert tree[name] == item['git_blob'], name
            raw = contents[item['git_blob']]
            assert item['git_to_frozen'] in ('identity', 'lf_to_crlf')
            if item['git_to_frozen'] == 'lf_to_crlf':
                raw = raw.replace(b'\n', b'\r\n')
            assert digest(raw) == item['sha256'], name
            total += 1
    return total


def validate_gate(gate: dict, references: dict, census: dict) -> None:
    seq = str(gate['shape'][1])
    assert gate['status'] == 'ok' and gate['shape'][0] == 1
    assert gate['steps'] == references[seq] and gate['optimizer_updates'] == 2
    for step in gate['steps']:
        assert step['parameters'].keys() == step['gradients'].keys()
        assert len(step['parameters']) == 256
        for tensor in [step['loss'], step['logits'], *step['parameters'].values(),
                       *step['gradients'].values()]:
            assert len(tensor['sha256']) == 64 and tensor['dtype']
            assert isinstance(tensor['shape'], list)
    assert gate['cap_binding'] == dict(verified=True, field='training.stream_vram_override',
                                     bytes=CAP, native_message_names_override=True)
    assert 0 < gate['peak_allocated'] <= gate['peak_reserved'] <= CAP
    assert gate['native'] == census[seq]['native']
    assert gate['native']['cap_bound'] == 'override'
    assert any('training.stream_vram_override' in message for message in gate['messages'])
    assert gate['attention_implementation'] == 'sdpa'
    if gate['profile']:
        assert gate['attention_ops']['aten::_scaled_dot_product_efficient_attention'] > 0
    assert gate['admission']['uuid'] == GPU and gate['admission']['power_limit'] == 420
    if gate['arm'] == 'qkvo':
        assert gate['cache_bytes'] == gate['plan']['raw_bytes']
        assert gate['cache_tensors'] == 4 * len(gate['plan']['selected_indices'])
    else:
        assert gate['cache_bytes'] == gate['cache_tensors'] == 0


def telemetry(samples: list[dict], timing: dict) -> dict:
    assert len(samples) >= 2
    times = [s['unix_s'] for s in samples]
    assert all(b > a for a, b in zip(times, times[1:]))
    start, end = timing['window_start_unix_s'], timing['window_end_unix_s']
    measured = [s for s in samples if start <= s['unix_s'] <= end]
    assert len(measured) >= 2 and times[0] <= start < end
    for sample in samples:
        assert len(sample['gpus']) == 4 and sample['free_ram'] >= 32 * 2**30
        assert sample['rss'] <= 16 * 2**30
        assert not sample['windows_ci_workers'] and not sample['linux_ci_workers']
        assert len({g[1].strip() for g in sample['gpus']}) == 4
        for gpu in sample['gpus']:
            assert float(gpu[4]) <= 75
            if gpu[1].strip() == GPU:
                assert float(gpu[5]) == 420
            else:
                assert float(gpu[3]) <= 5
    selected = [g for sample in measured for g in sample['gpus'] if g[1].strip() == GPU]
    others = [g for sample in measured for g in sample['gpus'] if g[1].strip() != GPU]
    return dict(samples=len(samples), measured_samples=len(measured),
                first_sample_unix_s=times[0],
                max_sample_gap_s=max(b-a for a, b in zip(times, times[1:])),
                measured_temperature_c=[min(float(g[4]) for g in selected),
                                        max(float(g[4]) for g in selected)],
                measured_sm_clock_mhz=[min(float(g[6]) for g in selected),
                                       max(float(g[6]) for g in selected)],
                max_other_gpu_util_percent=max(float(g[3]) for g in others))


def audit(root: Path, git_root: Path | None = None) -> dict:
    manifest = read(root/'manifest.json')
    for name, expected in manifest['published_files_sha256'].items():
        target = Path(os.path.realpath(root/name))
        assert os.path.commonpath([os.path.realpath(root), target]) == os.path.realpath(root)
        verify_file(target, expected)
    refs = read(root/'reference-updates.json')
    census = {str(seq): read(root/f'census-v1/seq{seq}.json') for seq in (2048, 4096)}
    all_gates, results, exact_count = {}, [], 0
    for study in ('exactness-v1', *STUDIES):
        folder = root/study
        metadata = manifest['studies'][study]
        for name, entry in metadata['retained_files'].items():
            raw = (folder/name).read_bytes().replace(b'\r\n', b'\n')
            if entry['newlines'] == 'crlf':
                raw = raw.replace(b'\n', b'\r\n')
            assert digest(raw) == entry['original_sha256']
        gates = {}
        for entry in read_rows(folder/'processes.jsonl'):
            assert entry['file'] not in gates
            assert entry['original_sha256'] == metadata['original_files_sha256'][entry['file']]
            gate = expand_receipt(entry, refs)
            validate_gate(gate, refs, census)
            gates[entry['file']] = gate
        assert len(gates) == metadata['successful_exactness_receipts']
        exact_count += len(gates)*2
        all_gates[study] = gates
        freeze, ledger = read(folder/'freeze.json'), read(folder/'ledger.json')
        assert all(e['returncode'] == 0 and e['guard_failure'] is None for e in ledger)
        census_hashes = manifest['studies']['census-v1']['original_files_sha256']
        reference_hashes = manifest['studies']['exactness-v1']['original_files_sha256']
        if study == 'exactness-v1':
            assert len(gates) == len(ledger) == len(freeze['roster']) == 10
            for seq in (2048, 4096):
                assert freeze['census'][str(seq)] == census_hashes[f'seq{seq}.json']
            continue
        original = read(folder/'original-audit.json')
        assert original['verified'] and not original['primary_passes']
        assert freeze['cap_bytes'] == CAP and freeze['gpu_uuid'] == GPU
        assert freeze['power_limit_w'] == 420
        hashes = {k.replace('\\', '/'): v for k, v in read(folder/'source-hashes.json').items()}
        bindings = read(folder/'source-blobs.json')
        assert {k: v['sha256'] for k, v in bindings['files'].items()} == hashes
        assert bindings['commit'].startswith(freeze['source_commit'])
        for seq in ((2048, 4096) if study == 'timing-v1' else (2048,)):
            expected_census = census_hashes[f'seq{seq}.json']
            expected_reference = reference_hashes[f'seq{seq}-auto.json']
            if study == 'timing-v1':
                assert freeze['fixed_census_hashes'][str(seq)] == expected_census
                assert freeze['reference_hashes'][str(seq)] == expected_reference
            else:
                assert freeze['census_sha256'] == expected_census
                assert freeze['reference_sha256'] == expected_reference
        for gate in gates.values():
            seq = gate['shape'][1]
            assert gate['census_sha256'] == census_hashes[f'seq{seq}.json']
            assert gate['reference_sha256'] == reference_hashes[f'seq{seq}-auto.json']
            if study == 'timing-v1':
                assert gate['plan'] == census[str(seq)]['projections']
            else:
                choices = [a for a in freeze['arms'].values()
                           if a['arm'] == gate['arm'] and a['allocation'] == gate['allocation']
                           and a.get('cache_scope', 'layer') == gate.get('cache_scope', 'layer')]
                assert len(choices) == 1
                assert gate['plan'] == choices[0]['plan']
                assert gate['effective_k'] == choices[0]['effective_k']
        if study != 'timing-v1':
            protocol_hash = metadata['retained_files']['RULE.md']['original_sha256']
            assert protocol_hash == freeze['protocol_sha256']
            assert freeze['recipe']['targets'] == ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            assert freeze['recipe']['lora_r'] == 8 and freeze['recipe']['alpha'] == 16
            assert [dict((k, v) for k, v in e.items() if k not in ('returncode', 'guard_failure'))
                    for e in ledger] == freeze['gates'] + freeze['timings']
            orders = list(itertools.permutations(freeze['arms']))
            random.Random(42).shuffle(orders)
            assert freeze['orders'] == [list(o) for o in orders]
            assert len(gates) == 27 and len(freeze['gates']) == 9
        else:
            assert [{k: e[k] for k in ('seq', 'pair', 'arm')} for e in ledger] == freeze['roster']
            assert len(gates) == 20
        by_tag = defaultdict(list)
        for row in read_rows(folder/'resources.jsonl'):
            by_tag[row['tag']].append(row['sample'])
        timings = {row['file'].removesuffix('-timing.json'): row['receipt']
                   for row in read_rows(folder/'timings.jsonl')}
        assert len(timings) == (20 if study == 'timing-v1' else 18)
        freeze_time = datetime.fromisoformat(freeze['at']).timestamp()
        rows = []
        for tag, timing in timings.items():
            suffix = '-initial-exactness.json' if study == 'timing-v1' else '-exactness.json'
            gate_name = tag + suffix
            gate = gates[gate_name]
            assert timing['status'] == 'ok' and timing['warmup_updates'] == 10
            assert timing['measured_updates'] == 50 and timing['verified_initial_updates'] == 2
            assert gate['continuation_optimizer_updates'] == 58
            assert len(timing['step_times_s']) == len(timing['losses']) == 50
            assert all(math.isfinite(t) and t > 0 for t in timing['step_times_s'])
            assert all(math.isfinite(loss) for loss in timing['losses'])
            assert timing['median_step_s'] == statistics.median(timing['step_times_s'])
            assert 0 < timing['peak_allocated'] <= timing['peak_reserved'] <= CAP
            assert timing['initial_reference_sha256'] == gate['reference_sha256']
            for key in ('shape', 'arm', 'native', 'plan', 'cap_bytes'):
                assert timing[key] == gate[key]
            resource = telemetry(by_tag[tag], timing)
            assert freeze_time < resource['first_sample_unix_s'], 'freeze after process start'
            rows.append(dict(tag=tag, median_step_s=timing['median_step_s'],
                             peak_allocated=timing['peak_allocated'],
                             peak_reserved=timing['peak_reserved'], **resource))
        if study == 'timing-v1':
            expected_order = [f"seq{e['seq']}-p{e['pair']}-{e['arm']}" for e in freeze['roster']]
        else:
            expected_order = [e['tag'] for e in freeze['timings']]
        actual_order = [r['tag'] for r in sorted(rows, key=lambda r: r['first_sample_unix_s'])]
        assert actual_order == expected_order, 'timing order differs from frozen roster'
        stats, memories = [], []
        if study == 'timing-v1':
            for seq in (2048, 4096):
                reductions = []
                for pair in range(1, 6):
                    base = timings[f'seq{seq}-p{pair}-auto']
                    candidate = timings[f'seq{seq}-p{pair}-qkvo']
                    reductions.append(100*(base['median_step_s']-candidate['median_step_s'])
                                      / base['median_step_s'])
                    memories.append(dict(pair=pair, sequence=seq, **memory_gate(
                        base['peak_allocated'], candidate['peak_allocated'],
                        candidate['plan']['raw_bytes'])))
                stats.append(dict(sequence=seq, **bootstrap(reductions, 2)))
        else:
            for contrast in freeze['contrasts']:
                candidate_name, baseline_name = contrast.split('_vs_')
                reductions = []
                for block in range(1, 7):
                    base = timings[f'block{block}-{baseline_name}']
                    candidate = timings[f'block{block}-{candidate_name}']
                    reductions.append(100*(base['median_step_s']-candidate['median_step_s'])
                                      / base['median_step_s'])
                    name = f'block{block}-{candidate_name}-exactness.json'
                    gate = gates[name]
                    memories.append(dict(block=block, contrast=contrast, **memory_gate(
                        base['peak_allocated'], candidate['peak_allocated'], gate['cache_bytes'])))
                stats.append(dict(contrast=contrast, **bootstrap(reductions, 3)))
        for computed, old in zip(stats, original['statistics'], strict=True):
            for key in ('median_reduction_percent', 'ordinary_95', 'passes_rule'):
                assert computed[key] == old[key]
            assert computed['family_adjusted_ci'] == old.get('family_adjusted_ci',
                                                            old.get('family_adjusted_97_5'))
        primary = stats[0]['passes_rule']
        if study == 'scoped-v2-resume1':
            primary = scoped_eligible(stats[0], stats[1])
            assert not primary and not read(folder/'stage-a-decision.json')['confirmation_eligible']
            assert original['confirmation_processes'] == 0
            recovery = read(folder/'recovery-amendment.json')
            prior = read(folder/'prior-attempt.json')
            assert recovery['completed_gates_reused'] == 6 and recovery['models_repeated'] == 0
            assert prior['archive_sha256'] == recovery['prior_archive_sha256']
            assert prior['files_sha256'] == recovery['previous_files_sha256']
            assert prior['ledger'] == recovery['previous_ledger']
            assert prior['ledger'][-1]['returncode'] != 0
            assert 'selected GPU lacks idle memory/temperature headroom' in prior['failed_gate_log']
            for entry in prior['ledger'][:6]:
                name = entry['tag']+'-exactness.json'
                assert prior['files_sha256'][name] == metadata['original_files_sha256'][name]
        assert primary == original['primary_passes']
        results.append(dict(study=study, primary_passes=primary, statistics=stats,
                            freeze_at=freeze['at'], timing_processes=len(timings),
                            memory_pass=all(m['passes'] for m in memories),
                            memory_checks=memories, processes=rows))
    source_count = verify_sources(root, git_root) if git_root is not None else None
    return dict(verified=True, exact_update_replays=exact_count, timing_processes=56,
                measured_updates=2800, warmup_updates=560, confirmation_processes=0,
                sources_verified=source_count, studies=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--verify-git-sources', type=Path, metavar='CHECKOUT')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.evidence, args.verify_git_sources)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    Console().print({k: v for k, v in result.items() if k != 'studies'})


if __name__ == '__main__':
    main()
