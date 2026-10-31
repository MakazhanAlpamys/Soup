"""The series-B comparator: between-run rule over A1 B1 B2 A2 run records."""

import hashlib
import json
import math
import shutil

import pytest

RUNS = ("A1", "B1", "B2", "A2")
GIB32 = 32 * 1024**3


def _harness(name):
    import importlib.util
    import sys
    from pathlib import Path

    harness = Path(__file__).resolve().parent.parent / "benchmarks" / "harness"
    if str(harness) not in sys.path:
        sys.path.insert(0, str(harness))
    spec = importlib.util.spec_from_file_location(name, harness / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cmp():
    return _harness("two_read_compare")


def _step(index, two_read, seconds):
    reads = 2 if two_read else 3
    return {
        "step": index, "seconds": seconds, "loss": 1.25 + index / 3,
        "grads_complete": True, "lora_grads_sha256": f"{index:x}" * 64,
        "expert_fetches": 40 * reads, "expert_bytes_read": 4000 * reads,
        "expert_read_seconds": 0.1, "ssd_read_bytes": 4000 * reads,
        "h2d_bytes": {"experts": 400 * reads, "dense": 77, "large": 9,
                      "total": 400 * reads + 86},
        "nvml_peak_process_bytes": 1000 + index, "cgroup_memory_peak_bytes": 2000 + index,
        "peak_host_bytes_so_far": 3000, "layer_loads": 4,
    }


def _quad(seconds=None):
    """A consistent PASS quadruple: name -> (record, step JSON). B is faster everywhere.

    ``_write`` fills in each record's ``cwd`` and ``expect_json`` path and sha256."""
    seconds = seconds or {"A1": [3.0, 2.0, 2.0], "B1": [2.5, 1.5, 1.5],
                          "B2": [2.5, 1.5, 1.5], "A2": [3.0, 2.0, 2.0]}
    quad = {}
    for index, name in enumerate(RUNS):
        two_read = name.startswith("B")
        run_id = f"{index + 1:032x}"
        step = {
            "label": name, "evidence": "cuda", "two_read_experts": two_read,
            "phase": "done", "supervisor_run_id": run_id,
            "args": {"store": "s", "out": f"{name}.json", "label": name,
                     "two_read_experts": two_read, "seed": 7, "steps": 3, "tier": "ram",
                     "deterministic": "strict"},
            "environment": {"hostname": "h", "python": "3.11", "soup_commit": "abc"},
            "numerics": {"deterministic_algorithms": True, "deterministic_warn_only": False},
            "attn_implementation": "sdpa", "experts_implementation": "grouped_mm",
            "initial_trainable_sha256": "a" * 64, "input_ids_sha256": "b" * 64,
            "cgroup": {"path": "/x", "memory_max": GIB32, "memory_swap_max": 0},
            "nvml_device": {"index": 0, "uuid": "GPU-1"},
            "memory": {"mode": "envelope", "nvml_total_bytes": 8 * 1024**3},
            "envelope": {"mib": 7000}, "store_bytes": 123,
            "store_expert_layout": "stacked", "drop_caches": "ok",
            "run_verdict": {"status": "ok"},
            "memory_verdict": {"mode": "envelope", "status": "ok", "step": None,
                               "reason": None},
            "steps": [_step(i, two_read, s) for i, s in enumerate(seconds[name])],
        }
        record = {
            "schema": "two-read-cgroup-run/2", "run_id": run_id, "status": "ok",
            "reason": None, "name": name,
            "host": {"hostname": "box", "kernel": "6.8", "mem_total_bytes": 1, "swaps": []},
            "cgroup": {"path": f"/sys/fs/cgroup/soup-bench/{name}",
                       "readback": {"memory.max": str(GIB32), "memory.swap.max": "0",
                                    "memory.oom.group": "1"}},
            "series": {"path": "/sys/fs/cgroup/soup-bench",
                       "required": {"memory_max": GIB32, "swap_max": 0},
                       "readback": {"memory.max": str(GIB32), "memory.swap.max": "0"},
                       "memory_current_before": 0, "memory_peak_after": 6000,
                       "events_before": {"oom_kill": 0}, "events_after": {"oom_kill": 0}},
            "memory_peak_bytes": 5000,
            "expect_json": {"exists": True, "run_verdict": {"status": "ok"},
                            "memory_verdict": step["memory_verdict"], "error": None},
        }
        quad[name] = (record, step)
    return quad


def _write(root, quad, relative=False, cwd=None):
    """Records and step JSON go to ``root/out``; each record's cwd is ``root`` unless
    ``cwd`` is given. Returns name -> record path."""
    out = root / "out"
    out.mkdir(exist_ok=True)
    paths = {}
    for name, (record, step) in quad.items():
        data = json.dumps(step).encode("utf-8")
        (out / f"{name}.step.json").write_bytes(data)
        record["cwd"] = str(root) if cwd is None else cwd
        record["expect_json"].update(
            path=f"out/{name}.step.json" if relative else str(out / f"{name}.step.json"),
            sha256=hashlib.sha256(data).hexdigest())
        paths[name] = out / f"{name}.run.json"
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    return paths


def _compare(cmp, root, record_paths, limits=False):
    argv = ["--runs", *map(str, record_paths), "--out", str(root / "compare.json")]
    if limits:
        argv += ["--memory-max", str(GIB32), "--swap-max", "0"]
    code = cmp.main(argv)
    report = json.loads((root / "compare.json").read_text(encoding="utf-8"))
    assert report["exit_code"] == code
    return code, report


def _run(cmp, tmp_path, quad, limits=False):
    paths = _write(tmp_path, quad)
    return _compare(cmp, tmp_path, [paths[n] for n in RUNS], limits)


def _refused_with(code, report, needle):
    assert (code, report["verdict"], report["timing"]) == (2, "REFUSED", None)
    assert any(needle in reason for reason in report["refusals"]), report["refusals"]


def test_consistent_quadruple_passes_and_classifies_only_agreeing_pairs(cmp, tmp_path):
    quad = _quad({"A1": [3.0, 2.0, 2.0, 2.0], "B1": [2.5, 1.5, 1.5, 2.0],
                  "B2": [2.5, 1.5, 2.5, 2.5], "A2": [3.0, 2.0, 2.0, 2.0]})
    code, report = _run(cmp, tmp_path, quad, limits=True)
    assert (code, report["verdict"], report["failures"]) == (0, "PASS", [])
    classes = [row["class"] for row in report["timing"]["later"]]
    # index 1 both faster; index 2 B2/A2 > 1; index 3 B1/A1 exactly 1
    assert classes == ["two-read faster", "inconclusive", "inconclusive"]
    assert report["timing"]["later_median_seconds"]["B2"] == 2.5
    assert report["per_run"]["B1"]["expert_bytes_read"] * 3 == \
        report["per_run"]["A1"]["expert_bytes_read"] * 2
    assert report["claims"] == cmp.CLAIMS


def _mutate_step(quad, name, index, key, value):
    entry = quad[name][1]["steps"][index]
    *path, last = key.split(".")
    for part in path:
        entry = entry[part]
    entry[last] = value


@pytest.mark.parametrize("key, value", [
    ("loss", math.nextafter(1.25 + 1 / 3, 10.0)),
    ("lora_grads_sha256", "f" * 64),
    ("expert_bytes_read", 8001),
    ("h2d_bytes.dense", 78),
])
def test_one_quantity_mismatch_fails_at_that_index(cmp, tmp_path, key, value):
    quad = _quad()
    _mutate_step(quad, "B2", 1, key, value)
    code, report = _run(cmp, tmp_path, quad)
    assert (code, report["verdict"]) == (1, "FAIL")
    assert [(f["index"], f["quantity"]) for f in report["failures"]] == [(1, key)]
    assert report["timing"] is not None


def test_zero_fetch_count_fails_even_when_3_to_2_holds(cmp, tmp_path):
    quad = _quad()
    for name in RUNS:
        _mutate_step(quad, name, 2, "expert_fetches", 0)
    code, report = _run(cmp, tmp_path, quad)
    assert code == 1
    assert [(f["index"], f["quantity"]) for f in report["failures"]] == \
        [(2, "expert_fetches")]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "+inf", "-inf"])
@pytest.mark.parametrize("runs", [RUNS, ("B1",)], ids=["all-four", "one-run"])
def test_a_non_finite_loss_refuses(cmp, tmp_path, runs, value):
    """``float.hex`` turns NaN and each infinity into one string, so four equal non-finite
    losses would compare equal and could pass. A non-finite loss is a broken input, not a
    result: the comparison is refused, naming every run and step, with no timing."""
    quad = _quad()
    for name in runs:
        _mutate_step(quad, name, 1, "loss", value)
    code, report = _run(cmp, tmp_path, quad, limits=True)
    _refused_with(code, report, "loss")
    named = [r for r in report["refusals"] if "loss" in r]
    assert named == [f"{name} step 1: loss {value!r} is not a finite number" for name in runs]
    assert report["failures"] == []


@pytest.mark.parametrize("dense_b2, verdict", [(77, "PASS"), (78, "FAIL")])
def test_unusable_seconds_never_change_the_verdict(cmp, tmp_path, dense_b2, verdict):
    quad = _quad()
    _mutate_step(quad, "B2", 1, "h2d_bytes.dense", dense_b2)
    _mutate_step(quad, "A1", 1, "seconds", None)
    del quad["B1"][1]["steps"][2]["seconds"]
    _, report = _run(cmp, tmp_path, quad)
    assert report["verdict"] == verdict
    later = report["timing"]["later"]
    assert later[0]["ratios"] == {"B1/A1": None, "B2/A2": 0.75}
    assert [row["class"] for row in later] == ["inconclusive", "inconclusive"]
    assert report["timing"]["later_median_seconds"]["A1"] is None
    assert report["timing"]["later_median_seconds"]["A2"] == 2.0


def _swap_arms(quad):
    quad["A1"][1]["two_read_experts"] = True


def _seed(quad):
    quad["B1"][1]["args"]["seed"] = 8


def _oom(quad):
    quad["A2"][0]["status"] = "oom-killed"


def _no_memory_verdict(quad):
    quad["B1"][1]["memory_verdict"] = None


def _null_digests(quad):
    for name in RUNS:
        for entry in quad[name][1]["steps"]:
            entry["lora_grads_sha256"] = None


def _environment(quad):
    quad["B2"][1]["environment"]["soup_commit"] = "def"


def _readback(quad):
    for name in RUNS:
        quad[name][0]["cgroup"]["readback"]["memory.max"] = "max"


def _series_readback(quad):
    quad["B1"][0]["series"]["readback"]["memory.max"] = "max"


def _no_series(quad):
    quad["A2"][0]["series"] = None


def _series_path(quad):
    quad["B2"][0]["series"]["path"] = "/sys/fs/cgroup/other"


def _old_schema(quad):
    quad["A1"][0]["schema"] = "two-read-cgroup-run/1"


def _foreign_run_id(quad):
    quad["B2"][1]["supervisor_run_id"] = quad["B1"][0]["run_id"]


def _no_run_id(quad):
    quad["A1"][1]["supervisor_run_id"] = None


def _not_strict(quad):
    for name in RUNS:  # all four alike: equal across runs, still not strict
        quad[name][1]["args"]["deterministic"] = "off"


def _numerics(quad):
    quad["A2"][1]["numerics"]["deterministic_warn_only"] = True


def _attention(quad):
    quad["B1"][1]["attn_implementation"] = "eager"


def _initial_state(quad):
    quad["B2"][1]["initial_trainable_sha256"] = "c" * 64


@pytest.mark.parametrize("mutate, needle", [
    (_swap_arms, "A1: two_read_experts True"),
    (_seed, "args"),
    (_oom, "A2: supervisor status 'oom-killed'"),
    (_no_memory_verdict, "B1: memory_verdict is null"),
    (_null_digests, "lora_grads_sha256 None"),
    (_environment, "environment differs"),
    (_readback, "A1: cgroup readback memory.max 'max' != --memory-max"),
    (_series_readback, "B1: series readback memory.max 'max' != --memory-max"),
    (_no_series, "A2: series None is not a series parent record"),
    (_series_path, "supervisor series.path differs"),
    (_old_schema, "A1: run record schema 'two-read-cgroup-run/1'"),
    (_foreign_run_id, f"B2: step JSON supervisor_run_id '{2:032x}' != record run_id"),
    (_no_run_id, "A1: step JSON supervisor_run_id None != record run_id"),
    (_not_strict, "A1: args.deterministic 'off', expected 'strict'"),
    (_numerics, "numerics differs across runs"),
    (_attention, "attn_implementation differs across runs"),
    (_initial_state, "initial_trainable_sha256 differs across runs"),
])
def test_precondition_breaks_refuse_without_timing(cmp, tmp_path, mutate, needle):
    quad = _quad()
    mutate(quad)
    code, report = _run(cmp, tmp_path, quad, limits=True)
    _refused_with(code, report, needle)


@pytest.mark.parametrize("roles, needles", [
    (("A1", "B1", "B1", "A1"), ("A1 and A2 share one run record file",
                                "B1 and B2 share one run record file",
                                "A1 and A2 share one run_id")),
    # B1's record copied under a new file name into the B2 role
    (("A1", "B1", "copy", "A2"), ("B1 and B2 share one run_id",
                                  "B1 and B2 share one step JSON file")),
])
def test_one_run_in_two_roles_refuses(cmp, tmp_path, roles, needles):
    paths = _write(tmp_path, _quad())
    paths["copy"] = tmp_path / "out" / "B2-copy.run.json"
    shutil.copyfile(paths["B1"], paths["copy"])
    code, report = _compare(cmp, tmp_path, [paths[role] for role in roles], limits=True)
    for needle in needles:
        _refused_with(code, report, needle)


def test_step_json_rewritten_after_its_record_refuses(cmp, tmp_path):
    quad = _quad()
    paths = _write(tmp_path, quad)
    _mutate_step(quad, "B2", 1, "seconds", 0.5)  # new numbers under the old ok record
    (tmp_path / "out" / "B2.step.json").write_text(json.dumps(quad["B2"][1]),
                                                   encoding="utf-8")
    code, report = _compare(cmp, tmp_path, [paths[n] for n in RUNS], limits=True)
    _refused_with(code, report, "!= record expect_json.sha256")
    assert all(r.startswith("B2:") for r in report["refusals"]), report["refusals"]


@pytest.mark.parametrize("cwd, chdir, needle", [
    # From out/, neither the comparator's cwd nor the record's directory resolves it.
    (None, "out", None),
    # From the parent of out/, the comparator's cwd would resolve it; a relative record
    # cwd must refuse anyway.
    (".", "", "A1: expect_json.path 'out/A1.step.json' is relative and record cwd '.'"),
])
def test_relative_step_path_resolves_against_the_record_cwd(cmp, tmp_path, monkeypatch,
                                                            cwd, chdir, needle):
    # Records sit in out/ and name out/<run>.step.json relative to the parent of out/.
    paths = _write(tmp_path, _quad(), relative=True, cwd=cwd)
    monkeypatch.chdir(tmp_path / chdir)
    code, report = _compare(cmp, tmp_path, [paths[n] for n in RUNS], limits=True)
    if needle is None:
        assert (code, report["verdict"]) == (0, "PASS"), report["refusals"]
    else:
        _refused_with(code, report, needle)


def _moved(tmp_path, quad, host_path):
    """Write the runs, then move every step JSON into ``published/`` and point each record
    at a host path that does not exist here, as in a record copied off its host."""
    paths = _write(tmp_path, quad)
    published = tmp_path / "published"
    published.mkdir()
    for name in RUNS:
        shutil.move(str(tmp_path / "out" / f"{name}.step.json"),
                    str(published / f"{name}.step.json"))
        record = json.loads(paths[name].read_text(encoding="utf-8"))
        record["expect_json"]["path"] = host_path(name)
        paths[name].write_text(json.dumps(record), encoding="utf-8")
    return paths, published


@pytest.mark.parametrize("host_path", [
    lambda name: f"/workspace/out/{name}.step.json",
    lambda name: f"D:\\runs\\out\\{name}.step.json",
], ids=["posix", "windows"])
def test_step_dir_reads_moved_step_files_by_base_name(cmp, tmp_path, host_path):
    paths, published = _moved(tmp_path, _quad(), host_path)
    record_paths = [paths[n] for n in RUNS]
    code, report = _compare(cmp, tmp_path, record_paths, limits=True)
    _refused_with(code, report, "does not exist")
    code = cmp.main(["--runs", *map(str, record_paths), "--out", str(tmp_path / "c.json"),
                     "--memory-max", str(GIB32), "--swap-max", "0",
                     "--step-dir", str(published)])
    report = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert (code, report["verdict"]) == (0, "PASS"), report["refusals"]
    assert report["inputs"]["B1"]["step_json"] == str((published / "B1.step.json").resolve())


def test_step_dir_keeps_the_hash_and_run_id_binding(cmp, tmp_path):
    quad = _quad()
    paths, published = _moved(tmp_path, quad, lambda name: f"/workspace/out/{name}.step.json")
    _mutate_step(quad, "B2", 1, "seconds", 0.5)
    (published / "B2.step.json").write_text(json.dumps(quad["B2"][1]), encoding="utf-8")
    a2 = json.loads((published / "A2.step.json").read_bytes())
    a2["supervisor_run_id"] = "f" * 32
    data = json.dumps(a2).encode("utf-8")
    (published / "A2.step.json").write_bytes(data)
    record = json.loads(paths["A2"].read_text(encoding="utf-8"))
    record["expect_json"]["sha256"] = hashlib.sha256(data).hexdigest()
    paths["A2"].write_text(json.dumps(record), encoding="utf-8")
    code = cmp.main(["--runs", *(str(paths[n]) for n in RUNS), "--out", str(tmp_path / "c.json"),
                     "--step-dir", str(published)])
    report = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    _refused_with(code, report, "B2: step JSON")
    _refused_with(code, report, "!= record expect_json.sha256")
    _refused_with(code, report, "A2: step JSON supervisor_run_id")


@pytest.mark.parametrize("missing_in, label", [
    (None, "after guest page-cache drop; host cache not controlled"),
    ("B2", "first step (no cache drop)"),
])
def test_step0_label_requires_every_run_to_have_dropped_caches(cmp, tmp_path, missing_in,
                                                               label):
    quad = _quad()
    if missing_in:
        del quad[missing_in][1]["drop_caches"]
    _, report = _run(cmp, tmp_path, quad)
    assert report["timing"]["step0"]["label"] == label
    assert report["timing"]["step0"]["index"] == 0
    assert all(row["index"] >= 1 for row in report["timing"]["later"])
