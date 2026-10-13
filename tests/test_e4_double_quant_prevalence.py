"""E4 part 1 - the double-quant prevalence probe (benchmarks/harness/e4_double_quant_prevalence.py).

The record ``benchmarks/probe-e4-double-quant-prevalence.md`` carries the rule that
was committed in ``df3dae1`` before any run. Section 1.1 shows the rule body with
its headings demoted two levels so that they nest under the subsection; nothing
else may differ from the committed bytes. ``TestRuleBodyIsTheCommittedOne`` pins
that.
"""

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD = "benchmarks/probe-e4-double-quant-prevalence.md"
RULE_COMMIT = "df3dae149315dbc5168261d57446cf289075422d"
# SHA-256 of the rule body ("## What B3 established" through the last line of
# "## Deliverable") as committed in RULE_COMMIT. CI checks out one commit, so
# the bytes are pinned here and compared with git itself wherever the commit
# object is present.
RULE_BODY_SHA256 = "6a0053d36568a025112a451ad875db6faba737047e98b6622577596b39ff6aa5"
BODY_START = b"## What B3 established"
BODY_STOP = b"\n### 1.2 Definitions"
BEGIN = b"<!-- e4-rule-body:begin -->\n"
END = b"<!-- e4-rule-body:end -->\n"


def _promoted_body() -> bytes:
    """The rule body from the record, headings promoted back two levels."""
    record = (REPO_ROOT / RECORD).read_bytes()
    assert record.count(BEGIN) == 1 and record.count(END) == 1
    shown = record[record.index(BEGIN) + len(BEGIN):record.index(END)]
    lines = shown.splitlines(keepends=True)
    # Every heading inside the markers is a demoted rule heading; anything else
    # would mean the transform was not "two levels, nothing else".
    assert all(line.startswith(b"#### ") for line in lines if line.startswith(b"#"))
    return b"".join(line[2:] if line.startswith(b"#### ") else line for line in lines)


class TestRuleBodyIsTheCommittedOne:
    def test_promoted_body_matches_the_pinned_hash(self):
        body = _promoted_body()
        assert body.startswith(BODY_START)
        assert hashlib.sha256(body).hexdigest() == RULE_BODY_SHA256

    def test_promoted_body_is_byte_identical_to_the_rule_commit(self):
        shown = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{RULE_COMMIT}^{{commit}}"],
            capture_output=True, check=False,
        )
        if shown.returncode != 0:
            # A depth-1 checkout has no rule commit; the pinned hash above is
            # the check there. The pin itself is verified wherever history is.
            return
        committed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{RULE_COMMIT}:{RECORD}"],
            capture_output=True, check=True,
        ).stdout
        body = committed[committed.index(BODY_START):committed.index(BODY_STOP)]
        assert hashlib.sha256(body).hexdigest() == RULE_BODY_SHA256
        assert _promoted_body() == body

    def test_the_rule_body_is_nested_under_section_1_1(self):
        record = (REPO_ROOT / RECORD).read_bytes()
        section = record[record.index(b"### 1.1 Agreed rule"):record.index(BODY_STOP)]
        assert b"\n## " not in section
        assert section.count(b"\n#### ") == 6


# --------------------------------------------------------------------------
# Rule V4: "a failed fetch, OOM or killed run is kept as void and reported, not
# silently retried". Fault injection against a synthetic three-tensor checkpoint
# served by a mocked ``urlopen``; no network, no bitsandbytes.
# --------------------------------------------------------------------------
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402

HARNESS = REPO_ROOT / "benchmarks" / "harness" / "e4_double_quant_prevalence.py"
RESULTS = REPO_ROOT / "benchmarks" / "results" / "e4-double-quant-prevalence"
KEYS = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(3)]
TENSOR_BYTES = 16


def _load_harness():
    spec = importlib.util.spec_from_file_location("e4_double_quant_prevalence", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._payload


class _FakeHub:
    """A one-file bf16 safetensors checkpoint behind ``urlopen``.

    ``faults`` maps a tensor key to exceptions raised, in order, by successive
    requests for that tensor's data range. ``calls`` counts those requests.
    """

    def __init__(self) -> None:
        entries = {
            key: {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [i * TENSOR_BYTES, (i + 1) * TENSOR_BYTES],
            }
            for i, key in enumerate(KEYS)
        }
        header = json.dumps(entries).encode()
        data = bytes(range(TENSOR_BYTES * len(KEYS)))
        self.blob = len(header).to_bytes(8, "little") + header + data
        self.base = 8 + len(header)
        self.index = json.dumps({"weight_map": {key: "model.safetensors" for key in KEYS}}).encode()
        self.faults: dict = {}
        self.calls = {key: 0 for key in KEYS}

    def _key_for(self, start: int, stop: int):
        for i, key in enumerate(KEYS):
            if (start, stop) == (self.base + i * TENSOR_BYTES, self.base + (i + 1) * TENSOR_BYTES):
                return key
        return None

    def urlopen(self, request, timeout=None):
        if request.full_url.endswith("model.safetensors.index.json"):
            return _Response(self.index)
        first, last = request.get_header("Range").removeprefix("bytes=").split("-")
        start, stop = int(first), int(last) + 1
        key = self._key_for(start, stop)
        if key is not None:
            self.calls[key] += 1
            if self.faults.get(key):
                raise self.faults[key].pop(0)
        return _Response(self.blob[start:stop])


def _fake_row(ckpt, item, table, fetched) -> dict:
    metrics = {
        "frac_over": 0.0, "worst": 0.1, "frac_flipped": 0.0, "frac_absmax_off10": 0.0,
        "rel_fro": 0.09, "finite": True,
    }
    return {
        "key": item["key"], "kind": item["kind"], "dq_off": dict(metrics),
        "dq_on": dict(metrics), "buckets_dq_on": [],
    }


def _lines(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    module = _load_harness()
    hub = _FakeHub()
    monkeypatch.setitem(
        module.MODELS, "tiny", {"repo": "nobody/tiny", "rev": "0" * 40, "fp8": False}
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", hub.urlopen)
    monkeypatch.setattr(module, "measure", _fake_row)
    monkeypatch.setattr(module, "box", lambda: {})
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    selection = {"models": {"tiny": {"keys": [{"key": k, "kind": "dense_mlp"} for k in KEYS]}}}
    (tmp_path / "selected-keys.json").write_text(json.dumps(selection))

    class Rig:
        out = tmp_path
        rows = tmp_path / "tiny.jsonl"
        voids = tmp_path / "tiny.void.jsonl"
        journal = tmp_path / "tiny.attempts.jsonl"

        @staticmethod
        def run(*extra) -> int:
            return module.main(["run", "--model", "tiny", "--out", str(tmp_path), *extra])

        @staticmethod
        def summarize() -> int:
            return module.main(["summarize", "--out", str(tmp_path)])

        @staticmethod
        def attempts_summary() -> dict:
            return json.loads((tmp_path / "attempts-summary.json").read_text())["tiny"]

    Rig.hub = hub
    Rig.module = module
    return Rig


def _unfinished(journal: Path) -> list:
    events = _lines(journal)
    ended = {e["attempt_id"] for e in events if e["event"] in ("ok", "fail")}
    return [e for e in events if e["event"] == "start" and e["attempt_id"] not in ended]


class TestV4FaultInjection:
    def test_a_failed_fetch_is_a_void_and_is_not_retried(self, rig):
        """1. OSError once, then success: one request, one fail event, one void, no row."""
        rig.hub.faults[KEYS[1]] = [OSError("connection reset")]
        assert rig.run() == 0
        assert rig.hub.calls[KEYS[1]] == 1
        fails = [e for e in _lines(rig.journal) if e["event"] == "fail"]
        assert len(fails) == 1
        assert "connection reset" in fails[0]["exc"]
        voids = _lines(rig.voids)
        assert [(v["key"], v["reason"]) for v in voids] == [(KEYS[1], "fetch_failed")]
        assert voids[0]["attempt_id"] == fails[0]["attempt_id"]
        assert sorted(r["key"] for r in _lines(rig.rows)) == [KEYS[0], KEYS[2]]

    def test_resume_does_not_remeasure_a_prior_void(self, rig):
        """2. The voided key is left alone on resume and the summary still reports it."""
        rig.hub.faults[KEYS[1]] = [OSError("connection reset")]
        rig.run()
        assert rig.run() == 0
        assert rig.hub.calls[KEYS[1]] == 1
        assert KEYS[1] not in {r["key"] for r in _lines(rig.rows)}
        assert len(_lines(rig.voids)) == 1
        assert rig.summarize() == 0
        assert json.loads((rig.out / "summary.json").read_text())["tiny"]["voids"] == 1
        attempts = rig.attempts_summary()
        assert attempts["voids_by_reason"]["fetch_failed"] == 1
        assert attempts["remeasured_keys"] == 0

    def test_remeasure_voids_keeps_the_void_and_links_the_new_row(self, rig):
        """3. --remeasure-voids: new row, prior void kept, the two linked by attempt id."""
        rig.hub.faults[KEYS[1]] = [OSError("connection reset")]
        rig.run()
        void = _lines(rig.voids)[0]
        assert rig.run("--remeasure-voids") == 0
        assert rig.hub.calls[KEYS[1]] == 2
        assert _lines(rig.voids) == [void]
        row = next(r for r in _lines(rig.rows) if r["key"] == KEYS[1])
        assert row["supersedes_attempt"] == void["attempt_id"]
        starts = [e for e in _lines(rig.journal) if e["event"] == "start" and e["key"] == KEYS[1]]
        assert starts[-1]["supersedes_attempt"] == void["attempt_id"]
        assert rig.summarize() == 0
        attempts = rig.attempts_summary()
        assert attempts["voids_by_reason"]["fetch_failed"] == 1
        assert attempts["remeasured_keys"] == 1
        assert json.loads((rig.out / "summary.json").read_text())["tiny"]["voids"] == 1

    def test_an_interrupted_fetch_becomes_an_interrupted_void_on_resume(self, rig):
        """4. KeyboardInterrupt mid-fetch leaves a start with no end; resume voids the key."""
        rig.hub.faults[KEYS[1]] = [KeyboardInterrupt()]
        with pytest.raises(KeyboardInterrupt):
            rig.run()
        assert [e["key"] for e in _unfinished(rig.journal)] == [KEYS[1]]
        assert _lines(rig.voids) == []
        assert rig.run() == 0
        assert rig.hub.calls[KEYS[1]] == 1
        voids = {v["key"]: v for v in _lines(rig.voids)}
        assert voids[KEYS[1]]["reason"] == "interrupted"
        assert KEYS[1] not in {r["key"] for r in _lines(rig.rows)}
        # The tensor fetched one ahead was in flight too: its bytes arrived but it
        # was never measured. It is kept as interrupted, not quietly fetched again.
        assert voids[KEYS[2]]["reason"] == "interrupted"
        assert rig.hub.calls[KEYS[2]] == 1
        assert rig.run() == 0
        assert len(_lines(rig.voids)) == 2

    def test_a_clean_run_journals_every_fetch(self, rig):
        """5. Every fetch has a start/ok pair and there is one row per selected key."""
        assert rig.run() == 0
        events = _lines(rig.journal)
        starts = [e for e in events if e["event"] == "start"]
        oks = {e["attempt_id"] for e in events if e["event"] == "ok"}
        assert starts and {e["attempt_id"] for e in starts} == oks
        assert len(starts) == len(events) - len(oks)
        assert not [e for e in events if e["event"] == "fail"]
        assert [e["seq"] for e in starts] == list(range(len(starts)))
        for key in KEYS:
            assert any(e["key"] == key and e["range"] for e in starts)
        assert sorted(r["key"] for r in _lines(rig.rows)) == sorted(KEYS)
        assert not rig.voids.exists()
        assert rig.summarize() == 0
        assert rig.attempts_summary()["provenance"] == "per-attempt log"

    def test_a_void_and_an_unlinked_row_for_one_key_fail_the_summary(self, rig):
        """6. Coexistence guard: the summary refuses a void plus an unlinked success row."""
        assert rig.run() == 0
        rig.voids.write_text(json.dumps({
            "key": KEYS[0], "reason": "fetch_failed", "error": "OSError('x')",
            "attempt_id": "not-the-rows-attempt",
        }) + "\n")
        with pytest.raises(SystemExit) as raised:
            rig.summarize()
        assert KEYS[0] in str(raised.value)


class TestRecordsWithoutAJournal:
    def test_committed_runs_report_no_per_attempt_log_and_the_same_summary(self, tmp_path):
        """The main runs predate the journal: the summary says so instead of reporting
        zero failed attempts, and summary.json is reproduced byte for byte."""
        module = _load_harness()
        for path in RESULTS.iterdir():
            if path.name.endswith((".jsonl", ".json")) and path.name != "selected-keys.json":
                shutil.copy(path, tmp_path / path.name)
        committed = (tmp_path / "summary.json").read_bytes()
        assert module.main(["summarize", "--out", str(tmp_path)]) == 0
        assert (tmp_path / "summary.json").read_bytes() == committed
        attempts = json.loads((tmp_path / "attempts-summary.json").read_text())
        assert set(attempts) == set(module.MODELS)
        for name, entry in attempts.items():
            assert entry["provenance"] == "no per-attempt log", name
            assert entry["attempts"] is None, name
