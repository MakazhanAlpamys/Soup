"""Negative controls for the prospective RAM and paired-evaluation protocol."""

import copy
import math

import pytest


def test_evaluation_matches_direct_causal_loss_without_gradients():
    from types import SimpleNamespace

    import torch

    from benchmarks.harness.giant_moe_evaluation import evaluate_examples

    class TinyCausal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16)

        def forward(self, input_ids, labels):
            assert not self.training and not torch.is_grad_enabled()
            logits = self.head(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 16),
                labels[:, 1:].reshape(-1),
            )
            return SimpleNamespace(loss=loss)

    model = TinyCausal()

    def make_batch(text):
        ids = torch.tensor([[int(value) for value in text.split()]])
        return {"input_ids": ids, "labels": ids}

    examples = [{"id": "short", "text": "1 2 3 4"}, {"id": "long", "text": "1 2 3 4 5 6"}]
    events = []
    actual = evaluate_examples(
        model, examples, make_batch, lambda kind, **row: events.append((kind, row)), "before"
    )
    assert [row["predicted_tokens"] for row in actual] == [3, 5]
    with torch.no_grad():
        expected = [float(model(**make_batch(row["text"])).loss) for row in examples]
    assert [row["loss"] for row in actual] == expected
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [event[0] for event in events] == [
        "evaluation_before_start",
        "evaluation_before",
        "evaluation_before",
    ]


def test_quality_uses_token_weighting_and_accepts_degradation():
    from benchmarks.harness.giant_moe_evaluation import paired_summary

    before = [
        dict(id="a", loss=1.0, predicted_tokens=2, token_sha256="a" * 64),
        dict(id="b", loss=3.0, predicted_tokens=6, token_sha256="b" * 64),
    ]
    after = copy.deepcopy(before)
    after[0]["loss"] = 2.0
    result = paired_summary(before, after, expected_ids=["a", "b"])
    assert result["before_loss"] == 2.5
    assert result["after_loss"] == 2.75
    assert result["loss_delta"] == 0.25
    assert result["before_perplexity"] == pytest.approx(math.exp(2.5))
    assert result["predicted_tokens"] == 8
    assert result["quality_gain_claimed"] is False


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "missing, duplicate or reordered paired evaluation examples"),
        ("reordered", "missing, duplicate or reordered paired evaluation examples"),
        ("tokens", "paired evaluation used different token IDs/counts"),
        ("digest", "paired evaluation used different token IDs/counts"),
        ("nan", "invalid loss, token count or token digest"),
    ],
)
def test_paired_comparison_fails_closed(defect, message):
    from benchmarks.harness.giant_moe_evaluation import paired_summary

    before = [
        dict(id=str(i), loss=2.0, predicted_tokens=63, token_sha256=str(i) * 64) for i in range(8)
    ]
    after = copy.deepcopy(before)
    if defect == "missing":
        after.pop()
    elif defect == "reordered":
        after.reverse()
    elif defect == "tokens":
        after[0]["predicted_tokens"] = 62
    elif defect == "digest":
        after[0]["token_sha256"] = "f" * 64
    else:
        after[0]["loss"] = float("nan")
    with pytest.raises(ValueError) as exc:
        paired_summary(before, after, expected_ids=[str(i) for i in range(8)])
    assert str(exc.value) == message


def test_real_frozen_protocol_is_integrity_checked(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from benchmarks.harness.giant_moe_evaluation import load_evaluation

    repo = Path(__file__).parents[1]
    root = repo / "benchmarks/protocols/giant-moe-acceptance-2026-10-29"
    protocol, examples = load_evaluation(root / "protocol.json", root / "evaluation-data.json")
    assert len(examples) == 8
    assert protocol["max_length"] == 64
    assert protocol["evaluation_data_sha256"] == (
        "5bf15fd1b4296f0c89eb33ae8fb9c363816ab3d9832b0c5f17c84f9d5e800d19"
    )
    training = json.loads((repo / "examples/giant-moe-pilot-data.json").read_bytes())
    canonical = json.dumps(training, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        == (protocol["training_data_canonical_sha256"])
    )
    damaged = tmp_path / "damaged.json"
    damaged.write_bytes((root / "evaluation-data.json").read_bytes() + b" ")
    with pytest.raises(
        ValueError, match="^evaluation data digest differs from the frozen protocol$"
    ):
        load_evaluation(root / "protocol.json", damaged)


def test_ram_rejects_large_host_and_unbounded_or_large_cgroup():
    from benchmarks.harness.giant_moe_ram import ram_constraint

    for maximum in (None, 64 * 2**30):
        with pytest.raises(ValueError) as exc:
            ram_constraint({"memory_max": maximum, "host_total": 128 * 2**30})
        assert (
            str(exc.value) == "pilot requires an actual RAM limit <=32 GiB, not process RSS alone"
        )


def test_ram_accepts_recorded_hard_cgroup_on_larger_host():
    from benchmarks.harness.giant_moe_ram import ram_constraint

    assert ram_constraint({"memory_max": 32 * 2**30, "host_total": 128 * 2**30}) == {
        "kind": "cgroup_v2",
        "hard_limit_bytes": 32 * 2**30,
    }


def test_physical_small_host_requires_zero_host_swap():
    from benchmarks.harness.giant_moe_ram import ram_constraint

    assert (
        ram_constraint({"memory_max": None, "host_total": 32 * 2**30, "host_swap_total": 0})["kind"]
        == "physical_host"
    )
    with pytest.raises(ValueError) as exc:
        ram_constraint({"memory_max": None, "host_total": 32 * 2**30, "host_swap_total": 1})
    assert str(exc.value) == "pilot requires an actual RAM limit <=32 GiB, not process RSS alone"


@pytest.mark.parametrize("mount_root", ["/", "/workload/owned"])
def test_legacy_cgroup_reads_current_container_limit_and_accounting(tmp_path, mount_root):
    from benchmarks.harness.giant_moe_ram import ram_constraint, snapshot

    proc, mount = tmp_path / "proc", tmp_path / "memory"
    (proc / "self").mkdir(parents=True)
    mount.mkdir()
    owned = mount / "workload/owned" if mount_root == "/" else mount
    owned.mkdir(parents=True, exist_ok=True)
    (proc / "self/cgroup").write_text("13:memory:/workload/owned\n", encoding="utf-8")
    (proc / "self/mountinfo").write_text(
        f"1 0 0:1 {mount_root} {mount} ro - cgroup cgroup rw,memory\n",
        encoding="utf-8",
    )
    (proc / "meminfo").write_text(
        "MemTotal: 134217728 kB\nSwapTotal: 8388608 kB\nSwapFree: 8388608 kB\n",
        encoding="utf-8",
    )
    values = {
        "memory.limit_in_bytes": str(32 * 2**30),
        "memory.usage_in_bytes": str(20 * 2**30),
        "memory.max_usage_in_bytes": str(31 * 2**30),
        "memory.memsw.limit_in_bytes": str(32 * 2**30),
        "memory.failcnt": "4",
        "memory.stat": "total_swap 0\ntotal_rss 1024\n",
        "memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 0\n",
    }
    for name, value in values.items():
        (owned / name).write_text(value, encoding="utf-8")
    phase = tmp_path / "phase.txt"
    phase.write_text("preparation", encoding="utf-8")
    actual = snapshot(phase, proc=proc)
    assert actual["cgroup_version"] == 1
    assert actual["memory_max"] == 32 * 2**30
    assert actual["memory_peak"] == 31 * 2**30
    assert actual["swap_current"] == 0 and actual["swap_max"] == 0
    assert actual["events"]["oom_kill"] == 0
    assert actual["events"]["limit_hits"] == 4  # Limit hits are not OOM kills.
    assert ram_constraint(actual)["kind"] == "cgroup_v1"


def test_v2_host_namespace_resolves_membership_instead_of_host_root(tmp_path):
    from benchmarks.harness.giant_moe_ram import locate_memory_cgroup

    proc, mount = tmp_path / "proc", tmp_path / "unified"
    (proc / "self").mkdir(parents=True)
    owned = mount / "workload/owned"
    owned.mkdir(parents=True)
    (mount / "memory.max").write_text("max", encoding="utf-8")
    (owned / "memory.max").write_text(str(32 * 2**30), encoding="utf-8")
    (proc / "self/cgroup").write_text("0::/workload/owned\n", encoding="utf-8")
    (proc / "self/mountinfo").write_text(
        f"1 0 0:1 / {mount} ro - cgroup2 cgroup rw\n",
        encoding="utf-8",
    )
    path, version = locate_memory_cgroup(proc)
    assert path == owned and version == 2


@pytest.mark.parametrize(
    "defect",
    [
        "oom",
        "host_oom",
        "swap",
        "missing_phase",
        "failure",
        "limit_changed",
        "none",
    ],
)
def test_ram_fit_is_not_claimed_after_failed_or_incomplete_run(defect):
    from benchmarks.harness.giant_moe_ram import summarize_ram

    phases = [
        "download",
        "preparation",
        "training",
        "evaluation_before",
        "evaluation_after",
        "reload",
    ]
    samples = [
        dict(
            phase=phase,
            memory_current=10,
            memory_peak=20,
            swap_current=0,
            swap_peak=0,
            events={"oom_kill": 0},
            process_rss_bytes=8,
            memory_max=32 * 2**30,
        )
        for phase in phases
    ]
    code = 0
    if defect == "oom":
        samples[-1]["events"]["oom_kill"] = 1
    elif defect == "host_oom":
        samples[0]["host_oom_kills"] = 0
        samples[-1]["host_oom_kills"] = 1
    elif defect == "swap":
        samples[-1]["swap_peak"] = 1
    elif defect == "missing_phase":
        samples.pop()
    elif defect == "failure":
        code = 1
    elif defect == "limit_changed":
        samples[-1]["memory_max"] = 64 * 2**30
    result = summarize_ram(samples, {"kind": "cgroup_v2", "hard_limit_bytes": 32 * 2**30}, code)
    assert result["ram_32gib_constrained_pilot_passed"] is (defect == "none")


def test_missing_kernel_peak_is_reported_as_unavailable_instead_of_zero():
    from benchmarks.harness.giant_moe_ram import REQUIRED_PHASES, summarize_ram

    samples = [
        dict(
            phase=phase,
            memory_current=10,
            memory_peak=None,
            swap_current=0,
            swap_peak=None,
            events={"oom_kill": 0},
            process_rss_bytes=8,
            memory_max=32 * 2**30,
        )
        for phase in sorted(REQUIRED_PHASES)
    ]
    result = summarize_ram(samples, {"kind": "cgroup_v2", "hard_limit_bytes": 32 * 2**30}, 0)
    assert result["ram_32gib_constrained_pilot_passed"] is True
    assert result["max_sampled_cgroup_bytes"] == 10
    assert result["kernel_cgroup_peak_bytes"] is None


@pytest.mark.parametrize(
    ("defect", "failures"),
    [
        ("none", []),
        (
            "wide_ram",
            [
                "invalid RAM accounting: pilot requires an actual RAM limit <=32 GiB, "
                "not process RSS alone"
            ],
        ),
        (
            "tokens",
            ["invalid paired evaluation: paired evaluation used different token IDs/counts"],
        ),
        ("late_before", ["evaluation arms do not surround training"]),
        ("baseline", ["unadapted zero-B baseline is not established"]),
        ("provenance", ["evaluation protocol/data provenance mismatch"]),
    ],
)
def test_combined_verdict_requires_all_prospective_evidence(defect, failures):
    import hashlib
    import json
    from pathlib import Path

    from benchmarks.harness.giant_moe_acceptance_verdict import acceptance_verdict
    from tests.test_giant_moe_harness_verdict import _fixture

    root = Path(__file__).parents[1] / "benchmarks/protocols/giant-moe-acceptance-2026-10-29"
    protocol_path, data_path = root / "protocol.json", root / "evaluation-data.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    events, index = _fixture()
    events[0]["unix"] = 1
    events[3]["unix"], events[4]["unix"] = 20, 30
    events[-1]["unix"] = 99
    extra = [
        dict(
            event="evaluation_protocol",
            unix=2,
            protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            data_sha256=protocol["evaluation_data_sha256"],
        ),
        dict(event="unadapted_baseline", unix=3, zero_lora_b_tensors=305),
    ]
    for i, identity in enumerate(protocol["example_ids"]):
        for stage, timestamp in (("before", 4 + i), ("after", 40 + i)):
            extra.append(
                dict(
                    event="evaluation_" + stage,
                    unix=timestamp,
                    id=identity,
                    loss=2.0,
                    predicted_tokens=63,
                    token_sha256=str(i) * 64,
                )
            )
    events = events[:-1] + extra + events[-1:]
    phases = [
        "download",
        "preparation",
        "training",
        "evaluation_before",
        "evaluation_after",
        "reload",
    ]
    samples = [
        dict(
            phase=phase,
            unix=i * 20,
            memory_current=10,
            memory_peak=20,
            swap_current=0,
            swap_peak=0,
            events={"oom_kill": 0},
            process_rss_bytes=8,
            memory_max=32 * 2**30,
            host_total=128 * 2**30,
        )
        for i, phase in enumerate(phases)
    ]
    if defect == "wide_ram":
        for sample in samples:
            sample["memory_max"] = 64 * 2**30
    elif defect == "tokens":
        next(row for row in events if row["event"] == "evaluation_after")["predicted_tokens"] = 62
    elif defect == "late_before":
        next(row for row in events if row["event"] == "evaluation_before")["unix"] = 31
    elif defect == "baseline":
        next(row for row in events if row["event"] == "unadapted_baseline")[
            "zero_lora_b_tensors"
        ] = 0
    elif defect == "provenance":
        extra[0]["protocol_sha256"] = "0" * 64
    answer = acceptance_verdict(events, index, 8151, samples, 0, protocol_path, data_path)
    assert answer["acceptance_measurements_complete"] is (not failures)
    assert answer["failures"] == failures
