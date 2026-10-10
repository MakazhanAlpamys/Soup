"""#1702 — the speed gate's schedule and decision rule, pinned on CPU.

``benchmarks/harness/issue1702_arena_pack_gate.py`` runs the plan written in issue #1702
before any measurement: in-order vs packed arenas, A B B A per seed, the RAM tier as the
subject and the disk tier as the negative control. The rule was fixed in advance, so the
code that applies it must not be free to drift: these tests hold every branch of it.
The runs themselves need a CUDA box and are recorded in ``benchmarks/gate-1702-*.md``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_HARNESS = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "issue1702_arena_pack_gate.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("issue1702_arena_pack_gate", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_harness()

GiB = 2**30
SEEDS = (3, 17, 29, 43, 59)


def _rows(subject, control, *, base: float = 400.0):
    """A full series whose per-seed differences are the ones given (None = a failed run)."""
    wanted = {gate.SUBJECT: dict(zip(SEEDS, subject)), gate.CONTROL: dict(zip(SEEDS, control))}
    rows = []
    for run in gate.schedule():
        difference = wanted[run["mode"]][run["seed"]]
        row = dict(run, status="ok", tok_per_s=base, pinned_bytes=4 * GiB)
        if run["mode"] == gate.SUBJECT:
            row["pinned_bytes"] = gate.EXPECTED_SUBJECT_PINNED[run["arm"]]
        if difference is None:
            # The block's first packed run, whichever order the block runs in.
            if run["arm"] == "packed" and run["repeat"] in (0, 1):
                row.update(status="failed", error="cuMemHostAlloc", tok_per_s=None)
        elif run["arm"] == "packed":
            row["tok_per_s"] = base * (1.0 + difference)
        rows.append(row)
    return rows


QUIET = (0.01, -0.01, 0.005, -0.005, 0.0)


class TestTheSchedule:
    def test_it_is_forty_runs_with_unique_names(self):
        runs = gate.schedule()
        assert len(runs) == 40
        assert len({run["name"] for run in runs}) == 40

    def test_every_block_is_a_b_b_a_and_the_first_arm_alternates_by_seed(self):
        runs = gate.schedule()
        for position, seed in enumerate(SEEDS):
            for mode in (gate.SUBJECT, gate.CONTROL):
                arms = [r["arm"] for r in runs if r["seed"] == seed and r["mode"] == mode]
                first = gate.ARMS[position % 2]
                other = gate.ARMS[1 - position % 2]
                assert arms == [first, other, other, first], (seed, mode)

    def test_a_block_is_never_interleaved_with_another(self):
        runs = gate.schedule()
        blocks = [(run["seed"], run["mode"]) for run in runs]
        assert blocks == [block for block in dict.fromkeys(blocks) for _ in range(4)]

    def test_the_mode_that_goes_first_alternates_by_seed(self):
        runs = gate.schedule()
        firsts = [next(r["mode"] for r in runs if r["seed"] == seed) for seed in SEEDS]
        assert firsts == ["ram", "disk", "ram", "disk", "ram"]


class TestTheDifference:
    def test_it_is_the_ratio_of_the_two_arm_means(self):
        rows = [
            {"arm": "in-order", "status": "ok", "tok_per_s": 390.0},
            {"arm": "packed", "status": "ok", "tok_per_s": 420.0},
            {"arm": "packed", "status": "ok", "tok_per_s": 400.0},
            {"arm": "in-order", "status": "ok", "tok_per_s": 410.0},
        ]
        assert gate.block_difference(rows) == pytest.approx(0.025)

    def test_a_block_with_a_failed_run_has_no_difference(self):
        """Even one that timed its steps before it died: failed is failed."""
        rows = [
            {"arm": "in-order", "status": "ok", "tok_per_s": 390.0},
            {"arm": "packed", "status": "failed", "tok_per_s": 405.0},
            {"arm": "packed", "status": "ok", "tok_per_s": 400.0},
            {"arm": "in-order", "status": "ok", "tok_per_s": 410.0},
        ]
        assert gate.block_difference(rows) is None

    def test_a_block_with_a_missing_run_has_no_difference(self):
        rows = [
            {"arm": "in-order", "status": "ok", "tok_per_s": 390.0},
            {"arm": "packed", "status": "ok", "tok_per_s": 400.0},
            {"arm": "in-order", "status": "ok", "tok_per_s": 410.0},
        ]
        assert gate.block_difference(rows) is None


class TestTheRule:
    def test_inside_the_band_is_no_difference(self):
        verdict = gate.decide(_rows((0.02, -0.02, 0.01, 0.0, -0.01), QUIET))
        assert verdict["verdict"] == "NO DIFFERENCE"
        assert verdict["band"] == pytest.approx(0.03)
        assert verdict["subject_mean"] == pytest.approx(0.0)

    def test_below_the_band_is_slower(self):
        verdict = gate.decide(_rows((-0.05, -0.04, -0.06, -0.05, -0.05), QUIET))
        assert verdict["verdict"] == "SLOWER"
        assert verdict["subject_mean"] == pytest.approx(-0.05)

    def test_above_the_band_is_faster(self):
        verdict = gate.decide(_rows((0.05, 0.04, 0.06, 0.05, 0.05), QUIET))
        assert verdict["verdict"] == "FASTER"

    def test_the_band_widens_to_the_largest_control_difference(self):
        """A control that swings 5% either way says 4% is noise on this box today."""
        control = (0.05, -0.05, 0.0, 0.0, 0.0)
        verdict = gate.decide(_rows((-0.04,) * 5, control))
        assert verdict["band"] == pytest.approx(0.05)
        assert verdict["verdict"] == "NO DIFFERENCE"
        assert gate.decide(_rows((-0.04,) * 5, QUIET))["verdict"] == "SLOWER"

    def test_exactly_on_the_band_is_not_outside_it(self):
        """1/16 is exact in binary, so the mean sits ON the band, not a rounding error off."""
        control = (0.0625, -0.0625, 0.0, 0.0, 0.0)
        verdict = gate.decide(_rows((-0.0625,) * 5, control))
        assert verdict["band"] == 0.0625
        assert verdict["subject_mean"] == -0.0625
        assert verdict["verdict"] == "NO DIFFERENCE"
        faster = gate.decide(_rows((0.0625,) * 5, control))
        assert faster["subject_mean"] == 0.0625
        assert faster["verdict"] == "NO DIFFERENCE"

    def test_a_control_that_is_not_centred_makes_the_series_inconclusive(self):
        verdict = gate.decide(_rows((-0.2,) * 5, (-0.05, -0.04, -0.03, -0.04, -0.04)))
        assert verdict["verdict"] == "INCONCLUSIVE"
        assert "the box was not quiet" in verdict["reason"]

    def test_a_failed_run_is_listed_against_its_arm_and_costs_its_block(self):
        verdict = gate.decide(_rows((0.0, None, 0.0, 0.0, 0.0), QUIET))
        assert verdict["subject_differences"][17] is None
        assert verdict["failed_runs"]["packed"] == ["seed17-ram-0-packed"]
        assert verdict["failed_runs"]["in-order"] == []
        assert verdict["verdict"] == "NO DIFFERENCE"

    @pytest.mark.parametrize(
        ("subject", "control", "needle"),
        [
            ((0.0, None, None, 0.0, 0.0), QUIET, "the RAM tier has 3 differences"),
            ((0.0,) * 5, (0.0, None, None, 0.0, 0.0), "the control has 3 differences"),
        ],
    )
    def test_fewer_than_four_blocks_is_inconclusive(self, subject, control, needle):
        verdict = gate.decide(_rows(subject, control))
        assert verdict["verdict"] == "INCONCLUSIVE"
        assert needle in verdict["reason"]

    def test_a_ram_run_with_the_wrong_arenas_voids_the_series(self):
        rows = _rows((0.0,) * 5, QUIET)
        target = next(r for r in rows if r["mode"] == gate.SUBJECT and r["arm"] == "in-order")
        target["pinned_bytes"] = 6 * GiB  # the arm switch did nothing
        verdict = gate.decide(rows)
        assert verdict["verdict"] == "VOID"
        assert "the in-order arm must show 8589934592" in verdict["reason"]

    def test_control_arms_that_page_lock_differently_void_the_series(self):
        rows = _rows((0.0,) * 5, QUIET)
        target = next(r for r in rows if r["mode"] == gate.CONTROL and r["arm"] == "packed")
        target["pinned_bytes"] = 2 * GiB
        verdict = gate.decide(rows)
        assert verdict["verdict"] == "VOID"
        assert "seed 3 control" in verdict["reason"]

    def test_the_verdict_is_json(self):
        json.dumps(gate.decide(_rows((0.0,) * 5, QUIET)))

    @pytest.mark.parametrize("stamp", ["box_before", "box_after"])
    def test_one_run_on_battery_makes_the_series_inconclusive(self, stamp):
        """Measured on this laptop: a step is 2.6x slower on battery (0.83 s against 0.31 s)."""
        rows = _rows((-0.2,) * 5, QUIET)
        for row in rows:
            row["box_before"] = {"on_ac_power": True}
            row["box_after"] = {"on_ac_power": True}
        assert gate.decide(rows)["verdict"] == "SLOWER"
        rows[7][stamp] = {"on_ac_power": False}
        verdict = gate.decide(rows)
        assert verdict["verdict"] == "INCONCLUSIVE"
        assert "battery power, first seed03-disk-3-in-order" in verdict["reason"]

    def test_a_stamp_that_could_not_be_read_accuses_nobody(self):
        rows = _rows((-0.2,) * 5, QUIET)
        rows[0]["box_before"] = {"on_ac_power": None}
        rows[1]["box_before"] = {}
        assert gate.runs_on_battery(rows) == []
        assert gate.decide(rows)["verdict"] == "SLOWER"


class TestTheStrictComparison:
    @staticmethod
    def _result(mode, arm, digests, status="ok"):
        return {"mode": mode, "arm": arm, "status": status, "digests": digests}

    def test_equal_digests_are_exact(self):
        steps = [["a", "b"], ["c", "d"], ["e", "f"]]
        results = [self._result(m, a, steps) for m in ("ram", "disk") for a in gate.ARMS]
        outcome = gate.compare_strict(results)
        assert outcome["ram"] == {"exact": True, "steps": 3, "digests_per_step": 2}
        assert outcome["disk"]["exact"] is True

    def test_one_different_digest_is_not_exact(self):
        steps = [["a", "b"], ["c", "d"]]
        results = [self._result(m, a, steps) for m in ("ram", "disk") for a in gate.ARMS]
        results[1] = self._result("ram", "packed", [["a", "b"], ["c", "X"]])
        outcome = gate.compare_strict(results)
        assert outcome["ram"]["exact"] is False
        assert outcome["disk"]["exact"] is True

    def test_a_failed_or_missing_arm_is_not_exact(self):
        steps = [["a"]]
        results = [
            self._result("ram", "in-order", steps),
            self._result("ram", "packed", steps, status="failed"),
            self._result("disk", "in-order", steps),
        ]
        outcome = gate.compare_strict(results)
        assert outcome["ram"]["exact"] is False
        assert outcome["disk"]["exact"] is False

    def test_no_digests_at_all_is_not_exact(self):
        results = [self._result(m, a, []) for m in ("ram", "disk") for a in gate.ARMS]
        assert gate.compare_strict(results)["ram"]["exact"] is False


class TestTheInOrderArm:
    def test_it_brings_back_the_plan_from_before_1702(self, monkeypatch):
        import soup_cli.utils.layer_stream_runtime as runtime
        from tests.test_issue1702_pinned_arena_packing import _QWEN3_8B_STORE_BYTES

        # Registered first, so the harness's process-wide switch is undone after the test.
        monkeypatch.setattr(runtime, "_plan_largest_first", runtime._plan_largest_first)
        packed = runtime.plan_pinned_arenas(_QWEN3_8B_STORE_BYTES).pinned_bytes
        gate.use_in_order_plan()
        in_order = runtime.plan_pinned_arenas(_QWEN3_8B_STORE_BYTES).pinned_bytes
        assert {"in-order": in_order, "packed": packed} == gate.EXPECTED_SUBJECT_PINNED


class TestTheDriver:
    def test_the_default_mode_prints_the_schedule_and_imports_no_torch(self):
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('gate', r'{_HARNESS}')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "code = module.main([])\n"
            "print('TORCH' if 'torch' in sys.modules else 'LIGHT')\n"
            "sys.exit(code)\n"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        # "<position> <name>" per run, then the marker: every second token is a run name.
        *listed, marker = done.stdout.split()
        assert marker == "LIGHT"
        assert listed[1::2] == [run["name"] for run in gate.schedule()]

    @pytest.mark.parametrize("mode", ["series", "strict", "trial"])
    def test_a_run_mode_without_weights_is_refused_by_name(self, mode, capsys):
        with pytest.raises(SystemExit) as raised:
            gate.parse_args(["--mode", mode])
        assert raised.value.code == 2
        assert f"--mode {mode} needs --weights" in capsys.readouterr().err

    def test_an_existing_series_is_never_appended_to(self, tmp_path, capsys, monkeypatch):
        def no_spawn(*_args, **_kwargs):
            raise AssertionError("a run was started over an existing series")

        monkeypatch.setattr(gate, "_spawn", no_spawn)
        (tmp_path / "series.jsonl").write_text("{}\n", encoding="utf-8")
        args = gate.parse_args(["--mode", "series", "--weights", "x", "--out", str(tmp_path)])
        assert gate.run_series(args) == 2
        assert "never appended to" in capsys.readouterr().out
        assert (tmp_path / "series.jsonl").read_text(encoding="utf-8") == "{}\n"

    def test_a_series_is_not_started_on_battery_power(self, tmp_path, capsys, monkeypatch):
        def no_spawn(*_args, **_kwargs):
            raise AssertionError("a run was timed on battery power")

        monkeypatch.setattr(gate, "_spawn", no_spawn)
        monkeypatch.setattr(gate, "box_stamp", lambda: {"on_ac_power": False})
        args = gate.parse_args(["--mode", "series", "--weights", "x", "--out", str(tmp_path)])
        assert gate.run_series(args) == 2
        assert "the box is on battery power before run" in capsys.readouterr().out
        assert not (tmp_path / "series.jsonl").exists()

    def test_a_series_stops_when_the_box_goes_on_battery(self, tmp_path, capsys, monkeypatch):
        power = iter([True, True, False])
        monkeypatch.setattr(gate, "box_stamp", lambda: {"on_ac_power": next(power)})
        def canned(_args, run, *, strict):
            pinned = gate.EXPECTED_SUBJECT_PINNED[run["arm"]]
            return dict(run, status="ok", tok_per_s=400.0, pinned_bytes=pinned)

        monkeypatch.setattr(gate, "_spawn", canned)
        args = gate.parse_args(
            ["--mode", "series", "--weights", "x", "--out", str(tmp_path), "--settle-s", "0"]
        )
        assert gate.run_series(args) == 0
        out = capsys.readouterr().out
        assert "the series stops here" in out
        assert "VERDICT   INCONCLUSIVE" in out
        journal = (tmp_path / "series.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["name"] for line in journal] == [
            run["name"] for run in gate.schedule()[:2]
        ]

    def test_the_verdict_mode_re_applies_the_rule_to_a_saved_series(self, tmp_path, capsys):
        rows = _rows((-0.05,) * 5, QUIET)
        (tmp_path / "series.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        assert gate.main(["--mode", "verdict", "--out", str(tmp_path)]) == 0
        assert "VERDICT   SLOWER" in capsys.readouterr().out
        saved = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
        assert saved["verdict"] == "SLOWER"

    def test_a_trial_on_a_machine_without_cuda_is_a_skip(self, tmp_path, monkeypatch, capsys):
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        out = tmp_path / "trial.json"
        assert gate.main(["--mode", "trial", "--weights", "x", "--out", str(out)]) == 0
        assert "SKIP: CUDA is required" in capsys.readouterr().out
        assert json.loads(out.read_text(encoding="utf-8"))["status"] == "skipped"


class TestTheStrictDriver:
    """``--mode strict`` with the four runs replaced by canned results."""

    @staticmethod
    def _run(monkeypatch, tmp_path, *, ram_pinned, packed_digest="b"):
        def canned(_args, run, *, strict):
            assert strict is True
            pinned = ram_pinned[run["arm"]] if run["mode"] == gate.SUBJECT else 4 * GiB
            last = packed_digest if (run["mode"], run["arm"]) == ("ram", "packed") else "b"
            return dict(run, status="ok", pinned_bytes=pinned, digests=[["a", last]] * 3)

        monkeypatch.setattr(gate, "_spawn", canned)
        tmp_path.mkdir(exist_ok=True)
        code = gate.main(["--mode", "strict", "--weights", "x", "--out", str(tmp_path)])
        return code, json.loads((tmp_path / "strict.json").read_text(encoding="utf-8"))

    def test_two_different_plans_with_equal_digests_pass(self, monkeypatch, tmp_path, capsys):
        code, summary = self._run(
            monkeypatch, tmp_path, ram_pinned=gate.EXPECTED_SUBJECT_PINNED
        )
        assert code == 0
        assert summary["faults"] == []
        assert summary["ram"]["exact"] is True
        assert summary["ram"]["pinned_bytes"] == {"in-order": 8 * GiB, "packed": 6 * GiB}
        assert "STRICT    ram: byte-identical" in capsys.readouterr().out

    def test_equal_digests_from_one_and_the_same_plan_prove_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        same = {"in-order": 6 * GiB, "packed": 6 * GiB}
        code, summary = self._run(monkeypatch, tmp_path, ram_pinned=same)
        assert code == 3
        assert summary["ram"]["exact"] is True
        assert "the in-order arm must show 8589934592" in summary["faults"][0]
        assert "FAULT     seed03-ram-in-order" in capsys.readouterr().out

    def test_a_different_digest_fails_the_control(self, monkeypatch, tmp_path, capsys):
        code, summary = self._run(
            monkeypatch, tmp_path, ram_pinned=gate.EXPECTED_SUBJECT_PINNED, packed_digest="X"
        )
        assert code == 3
        assert summary["ram"]["exact"] is False
        assert summary["disk"]["exact"] is True
        assert "STRICT    ram: DIFFERENT" in capsys.readouterr().out


class TestOneSpawnedRun:
    """``_spawn`` with the child process replaced: what the driver makes of each ending."""

    RUN = {"name": "seed03-ram-0-in-order", "seed": 3, "mode": "ram", "arm": "in-order"}

    @staticmethod
    def _args(tmp_path, *extra):
        return gate.parse_args(
            ["--mode", "series", "--weights", "w", "--out", str(tmp_path), *extra]
        )

    @staticmethod
    def _child(monkeypatch, *, write=None, returncode=0, raises=None):
        seen = {}

        def fake_run(command, *, stdout, stderr, env, timeout):
            seen.update(command=command, env=env, timeout=timeout)
            if raises is not None:
                raise raises
            if write is not None:
                target = Path(command[command.index("--out") + 1])
                target.write_text(json.dumps(write), encoding="utf-8")
            return subprocess.CompletedProcess(command, returncode)

        monkeypatch.setattr(gate.subprocess, "run", fake_run)
        return seen

    def test_a_finished_run_keeps_its_result_and_gains_the_schedule_fields(
        self, tmp_path, monkeypatch
    ):
        seen = self._child(monkeypatch, write={"status": "ok", "tok_per_s": 400.0})
        row = gate._spawn(self._args(tmp_path), self.RUN, strict=False)
        assert row["status"] == "ok" and row["tok_per_s"] == 400.0
        assert (row["seed"], row["mode"], row["arm"]) == (3, "ram", "in-order")
        assert row["returncode"] == 0
        assert "unix" in row["box_before"] and "unix" in row["box_after"]
        command = seen["command"]
        assert command[command.index("--arm") + 1] == "in-order"
        assert command[command.index("--tier") + 1] == "ram"
        assert "--strict" not in command
        assert seen["timeout"] == 900.0

    def test_a_strict_run_is_three_steps_no_warm_up_and_deterministic_cublas(
        self, tmp_path, monkeypatch
    ):
        seen = self._child(monkeypatch, write={"status": "ok"})
        gate._spawn(self._args(tmp_path), self.RUN, strict=True)
        command = seen["command"]
        assert "--strict" in command
        assert command[command.index("--warmup") + 1] == "0"
        assert command[command.index("--steps") + 1] == "3"
        assert seen["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_a_run_that_hangs_is_stopped_and_counted_as_failed(self, tmp_path, monkeypatch):
        hang = subprocess.TimeoutExpired(cmd="trial", timeout=30)
        self._child(monkeypatch, raises=hang)
        row = gate._spawn(self._args(tmp_path, "--run-timeout-s", "30"), self.RUN, strict=False)
        assert row["status"] == "failed"
        assert row["error"] == "stopped after 30 s without a result"
        assert row["returncode"] is None

    def test_a_child_that_dies_mid_run_is_failed_not_started(self, tmp_path, monkeypatch):
        self._child(monkeypatch, write={"status": "started"}, returncode=1)
        row = gate._spawn(self._args(tmp_path), self.RUN, strict=False)
        assert row["status"] == "failed"
        assert row["error"] == "the process ended without a result"
        assert row["returncode"] == 1

    def test_an_earlier_result_of_the_same_name_is_never_read_as_this_run(
        self, tmp_path, monkeypatch
    ):
        stale = tmp_path / "seed03-ram-0-in-order.json"
        stale.write_text(json.dumps({"status": "ok", "tok_per_s": 999.0}), encoding="utf-8")
        self._child(monkeypatch, returncode=1)  # the child writes nothing this time
        row = gate._spawn(self._args(tmp_path), self.RUN, strict=False)
        assert row["status"] == "failed"
        assert "no result file" in row["error"]
        assert "tok_per_s" not in row
