"""Render the #2162 record from independently verified raw measurements."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def interval(values: list[float]) -> str:
    return f'[{values[0]:.6f}%, {values[1]:.6f}%]'


def render(root: Path, audit: dict) -> str:
    assert audit['verified'] and audit['exact_update_replays'] == 168
    lines = [
        '# Incremental QKVO retention over native auto', '',
        'Refs #2162. **INCONCLUSIVE:** the final primary measured 1.997942% less time per '
        'step, below its predeclared 2% threshold. Projection-scoped dispatch also failed '
        'to improve reliably over layer-scoped dispatch. The conditional confirmation '
        'did not run. This record adds no production field or default.', '',
        '## Results', '',
        'Each percentage is the median of paired percentage reductions computed from '
        'each process\'s median step time. Positive means less time. The control result '
        '2.034504% does not replace the primary.', '',
        '| Experiment / comparison | Median reduction | Ordinary 95% | Adjusted interval |',
        '|---|---:|---|---|',
    ]
    for study in audit['studies']:
        for stat in study['statistics']:
            name = stat.get('contrast', f"seq{stat.get('sequence')}")
            lines.append(f"| {study['study']} / {name} | "
                         f"{stat['median_reduction_percent']:.6f}% | "
                         f"{interval(stat['ordinary_95'])} | "
                         f"{interval(stat['family_adjusted_ci'])} |")
    lines += [
        '', 'The first experiment uses a two-cell family (97.5% adjusted intervals). '
        'Each later experiment uses three contrasts (98.3333%). All use 10,000 whole-pair '
        'or whole-block bootstrap resamples, NumPy default_rng(42), and linear quantiles. '
        'Five pairs or six blocks produce discrete bootstrap distributions; these '
        'intervals are descriptive of small experiments, not a guarantee of population '
        'coverage. They are the original per-experiment corrections, not a correction '
        'across this adaptive sequence of experiments.', '',
        'The final family interval spans 2%. The result establishes neither that the '
        'population gain is above 2% nor that it is below 2%. Its numerical acceptance '
        'rule still failed. The earlier 1.421695% and 1.860896% failed rules remain failed.',
        '', '## Recipe and exactness', '',
        '- Model: `unsloth/mistral-7b-instruct-v0.3`, revision '
        '`60b99dcbc7f853b7cd18472a607e8d045b96f03e`; existing NF4 shards, bf16 compute, '
        'pinned RAM and two stream buffers.',
        '- LoRA: rank 8, alpha 16, dropout 0, bias none, targets '
        '`q_proj`, `k_proj`, `v_proj`, `o_proj`. There are 32 layers x 4 target modules '
        'x 2 trainable matrices (A/B) = **256** gradient tensors and 256 updated '
        'parameter tensors. #2161 used two targets (q/v), hence 128.',
        '- Model seed 42, nonzero B seed 7, synthetic token seed 1234, batch 1, '
        'sequence 2048 (4096 in the first secondary cell). PagedAdamW8bit, lr 2e-4; '
        'OMP/MKL threads 1, deterministic algorithms, TF32 disabled.',
        '- Actual selected backend: SDPA efficient attention, verified by profiler '
        'operators. Torch 2.14.1+cu130, transformers 5.19.0, peft 0.21.2, '
        'bitsandbytes 0.50.2. #2161 used forced SDPA math, so its 3.14% comparison '
        'against full recompute is not this comparison against auto.',
        '- All 84 successful full-model processes checked two complete optimizer '
        'updates, including logits, loss, every gradient and every updated parameter '
        'bit-for-bit. The 56 timing processes replayed the reference in their first '
        'two warmups, then completed 10 warmups total and 50 measured updates each '
        '(2,800 measured updates; 560 warmups). The 28 untimed gates add 56 updates.',
        '', '## Changes between the frozen experiments', '',
        'The initial public plan kept native K fixed. Later local protocols changed '
        'the implementation and design after seeing earlier results. They were frozen '
        'locally before their own runs, not publicly preregistered. The JSON times and '
        'resource timestamps below are local provenance, not an external timestamp '
        'attestation. Earlier experiments are retained rather than pooled or relabelled.', '',
        '| Experiment | Locally frozen UTC | First timing process UTC | Design |',
        '|---|---|---|---|',
    ]
    designs = [
        'Native K retained: seq2048 K11 + QKVO9; seq4096 K3 + QKVO8; five AB/BA pairs per cell.',
        'Seq2048: native K11, reduced K10 without QKVO, K10 + QKVO19; six three-arm blocks.',
        'Seq2048: native K11, layer-scoped K10 + QKVO19, projection-scoped K10 + QKVO19; '
        'six three-arm blocks. Conditional six-pair confirmation was ineligible.',
    ]
    for study, design in zip(audit['studies'], designs, strict=True):
        first = min(p['first_sample_unix_s'] for p in study['processes'])
        stamp = datetime.fromtimestamp(first, timezone.utc).isoformat()
        lines.append(f"| {study['study']} | {study['freeze_at']} | {stamp} | {design} |")
    lines += [
        '', 'Both six-block orders include all permutations, shuffled once with Python '
        'random.Random(42). Their frozen rule requires primary median >=2% and positive '
        'ordinary and adjusted lower bounds. The last experiment additionally requires '
        'positive lower bounds for scoped_vs_legacy. Only then may the frozen stage B '
        'run six new native/scoped pairs (seed 73, ordinary 95% lower bound >0 and median '
        '>=2%). Stage B cannot rescue a failed stage A.', '',
        'The source commits and all Python file hashes are in each `source-blobs.json` '
        'and `source-hashes.json`. The original freeze and protocol hashes are retained. '
        'The measurements refer to those commits, not later changes on main.', '',
        '## Memory and the original auto decision', '',
        'Both the allocator limit and `training.stream_vram_override` were '
        '**8,589,934,592 bytes (8 GiB)**. The native seq2048 message, preserved in '
        '`census-v1/logs.jsonl` with its original wrapping, is:', '',
        '```text',
        "training.stream_recompute='auto': the last 11 of 32 streamed layers keep their",
        'activations, the rest recompute (measured at batch 1 x seq 2048: one layer',
        '420.0 MB retained, charged 482.0 MB at reserved/allocated 1.15; budget 5.76 GB',
        '= training.stream_vram_override 8.59 GB minus the reserved peak 2.83 GB)',
        '```', '',
        'The underlying integers are 419,972,096 retained bytes per whole layer, '
        '482,015,738 charged bytes, a 5,756,682,240-byte budget and '
        '2,833,252,352-byte probe reserved peak. Native K11 leaves 454,509,122 bytes; '
        'K10 leaves 936,524,860. The latter holds 19 QKVO caches (layers 3..21), '
        '796,917,760 raw bytes / 914,648,638 charged bytes, alongside full retention '
        'of layers 22..31.', '',
        'The owner-requested #2161 memory check is evaluated here as an additional '
        'publication check: candidate allocated peak minus comparator allocated peak '
        '<= saved bytes + 64 MiB in every pair/block. The original #2162 protocols '
        'required exact cache accounting and both peaks <=8 GiB. These retrospective '
        'memory checks do not change their timing verdicts. Original peaks are exact '
        'bytes, so no decimal-GB rounding allowance is needed.', '',
        '| Experiment / contrast | Max allocated increase (bytes) | Saved bytes | Memory check |',
        '|---|---:|---:|---|',
    ]
    for study in audit['studies']:
        grouped = {}
        for row in study['memory_checks']:
            key = row.get('contrast', str(row.get('sequence')))
            grouped.setdefault(key, []).append(row)
        for name, values in grouped.items():
            lines.append(f"| {study['study']} / {name} | "
                         f"{max(v['allocated_delta_bytes'] for v in values):,} | "
                         f"{values[0]['saved_bytes']:,} | "
                         f"{'PASS' if all(v['passes'] for v in values) else 'FAIL'} |")
    lines += [
        '', 'For the K10 QKVO arms, the increase versus native K11 is net of the '
        'whole layer no longer retained. The reduced-K10 control in the reallocation '
        'experiment isolates the cache cost: exactly 796,917,760 additional allocated '
        'bytes. Final candidate allocated/reserved peaks are 6,941,702,656 / '
        '7,291,797,504 bytes (6.464964 / 6.791016 GiB). Native K11 peaks are '
        '6,547,979,776 / 7,182,745,600 bytes.', '',
        '## Hardware and competing activity', '',
        'Selected card: RTX 3090 24 GiB, UUID `GPU-05919122-a78e-9af3-3b3c-f9a14fb4d2f4`, '
        'unchanged 420 W power limit. This is a capped 24 GiB card, not a physical '
        '8 GiB card. Other cards kept 350/420/420 W and existing services.', '',
        'The read-only inventory at 2026-10-26 19:02 UTC reports NVIDIA driver 610.88 '
        'and Windows 11 Pro for Workstations 10.0.26200, last boot 12:07:23.5 UTC. '
        '**OS and driver were captured after the experiments**; the timing receipts '
        'do not independently attest those versions throughout the campaign. The '
        'following temperatures, SM clocks and other-GPU utilization are contemporaneous '
        'samples inside measured windows:', '',
        '| Experiment | Temperature C | SM clock MHz | Other-GPU max utilization '
        '| Max sampling gap s |',
        '|---|---|---|---:|---:|',
    ]
    for study in audit['studies']:
        rows = study['processes']
        temperature = [min(r['measured_temperature_c'][0] for r in rows),
                       max(r['measured_temperature_c'][1] for r in rows)]
        clocks = [min(r['measured_sm_clock_mhz'][0] for r in rows),
                  max(r['measured_sm_clock_mhz'][1] for r in rows)]
        lines.append(f"| {study['study']} | {temperature[0]:.0f}..{temperature[1]:.0f} | "
                     f"{clocks[0]:.0f}..{clocks[1]:.0f} | "
                     f"{max(r['max_other_gpu_util_percent'] for r in rows):.0f}% | "
                     f"{max(r['max_sample_gap_s'] for r in rows):.6f} |")
    lines += [
        '', 'The other GPUs were not assumed idle: their sampled utilization reached '
        '2%, below the fixed 5% admission/timing limit. Windows CI workers were checked '
        'each sample, Linux CI workers every fifth sample; none was reported active. '
        'Sampling cannot exclude activity between probes or host contention. All raw '
        'samples, including RSS, free RAM, host CPU and all four GPUs, remain in the record.',
        '', '## Every timing block', '',
        'These tables preserve the executed order. Each value is median seconds per '
        'step, followed by peak allocated / reserved **bytes**, including warmups. '
        'All 50 individual durations and losses per process are in each `timings.jsonl`.',
    ]
    for study in audit['studies']:
        folder = root/study['study']
        freeze = read(folder/'freeze.json')
        rows = {row['tag']: row for row in study['processes']}
        if study['study'] == 'timing-v1':
            groups = [[f'seq{seq}-p{pair}-{e["arm"]}' for e in freeze['roster']
                       if e['seq'] == seq and e['pair'] == pair]
                      for seq in (2048, 4096) for pair in range(1, 6)]
        else:
            groups = [[f'block{index}-{arm}' for arm in order]
                      for index, order in enumerate(freeze['orders'], 1)]
        lines += ['', f"### {study['study']}", '', '| Pair/block | Order and measurements |',
                  '|---|---|']
        for tags in groups:
            text = '; '.join(f"{tag}: **{rows[tag]['median_step_s']:.9f} s**, "
                             f"{rows[tag]['peak_allocated']} / {rows[tag]['peak_reserved']}"
                             for tag in tags)
            lines.append(f"| {tags[0].rsplit('-', 1)[0]} | {text} |")
    lines += [
        '', '## Preserved unsuccessful preparation and recovery', '',
        'The first scoped implementation failed CPU preflight before any model gate '
        'or timing. PyTorch 2.14 resets the replay cursor when a SAC mode is re-entered; '
        'the corrected implementation gives each projection call its own unmodified '
        'mode pair and tests single entry. Its failed CPU log is retained.', '',
        'The next attempt completed six gates, then refused candidate admission '
        'before torch import/model loading because selected-GPU residual utilization '
        'was 8%. Recovery waited for three consecutive readings under the same limits, '
        'reused those six gates byte-for-byte, and ran the remaining frozen roster. '
        'The refusal, previous ledger and file hashes are retained; no completed '
        'full-model gate or timing was repeated.', '',
        '## Reproduce the audit', '',
        'From a checkout with Python, NumPy and Rich:', '',
        '```sh',
        'python benchmarks/harness/issue2162_audit.py \\',
        '  --evidence benchmarks/results/issue2162-rtx3090-v1 \\',
        '  --verify-git-sources . --output audit-2162.json',
        'python benchmarks/harness/issue2162_report.py \\',
        '  --evidence benchmarks/results/issue2162-rtx3090-v1 \\',
        '  --audit audit-2162.json --output report-2162.md',
        '```', '',
        'Source verification requires the original commits in Git history (fetch '
        'the PR branch without a shallow limit). Without `--verify-git-sources`, '
        'the data audit still runs but reports source verification as unavailable. '
        'It never runs a model. Published Git blob bindings replace redundant '
        'source ZIPs and preserve their original hashes and newline encodings.', '',
        'Evidence: [manifest](results/issue2162-rtx3090-v1/manifest.json), '
        '[protocol guide](protocol-2162-auto-qkvo.md), '
        '[independent audit](results/issue2162-rtx3090-v1/independent-audit.json). '
        'Shared reference updates reconstruct all 84 original receipts byte-for-byte; '
        'timing rows and telemetry are retained without aggregation. The original '
        'independent audits and all earlier failed thresholds remain accessible.', '',
        'Any further experiment needs a different, justified cell with a clearly '
        'larger predicted benefit and a public protocol before execution. Its correction '
        'family must include the previous 1.421695%, 1.860896%, 1.997942% and 2.034504% '
        'results, as requested by the maintainer. No further round is started by this record.',
    ]
    return '\n'.join(lines)+'\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.evidence, read(args.audit)), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
