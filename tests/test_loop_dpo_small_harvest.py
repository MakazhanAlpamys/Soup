"""#1909 - a loop DPO harvest of one or two pairs must reach training.

The generated loop config used the schema default ``val_split`` 0.1, so
``_split_val`` kept ``int(n * 0.9)`` rows: one pair left zero training rows. The
generated config now sets ``data.val_split: 0`` (the schema default and user
configs are unchanged), zero pairs skip training with the named reason
``no_preference_pairs`` and the count, a missing pairs file gets its own
diagnostic, and the training-stage note is saved in the iteration record.
"""

from __future__ import annotations

import json

import pytest
import yaml

import soup_cli.utils.loop_stages as ls
from soup_cli.utils.paths import quote_path


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_TRACE_DIR", raising=False)
    monkeypatch.delenv("SOUP_LOOP_THUMBS_DB", raising=False)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)
    return tmp_path


def _state(**kwargs):
    from soup_cli.utils.loop_state import LoopState

    return LoopState(served_model="registry://dummy", eval_suite="e", baseline="b", **kwargs)


def _seed(source: str, n: int, root) -> dict:
    """Make ``n`` valid preference pairs available through one harvest source."""
    if source == "traces":
        rows = []
        for i in range(n):
            rows.append({"id": f"{i}a", "prompt": f"Q{i}", "output": "good",
                         "signal": "thumbs_up"})
            rows.append({"id": f"{i}b", "prompt": f"Q{i}", "output": "bad",
                         "signal": "thumbs_down"})
        (root / "traces").mkdir()
        (root / "traces" / "t.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        return {"SOUP_LOOP_TRACE_DIR": str(root / "traces")}
    from soup_cli.utils import local_rl

    local_rl.init_local_rl_db("thumbs.db")
    for i in range(n):
        local_rl.record_thumb(db_path="thumbs.db", prompt=f"Q{i}", response="good", thumb="up")
        local_rl.record_thumb(db_path="thumbs.db", prompt=f"Q{i}", response="bad", thumb="down")
    return {"SOUP_LOOP_THUMBS_DB": "thumbs.db"}


class _Runner:
    """Stand-in for ``subprocess.run`` recording every training invocation."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.argvs: list = []

    def __call__(self, argv, **kwargs):
        self.argvs.append(list(argv))
        return self

    def config(self) -> dict:
        path = self.argvs[0][self.argvs[0].index("--config") + 1]
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)


def _run_loop(runner, harvest=ls.harvest_from_traces, train=ls.train_dpo_from_pairs):
    """One real ``watch`` iteration; returns (record, deploy_posts)."""
    from soup_cli.utils.loop_daemon import WatchConfig, watch
    from soup_cli.utils.loop_iteration import list_iterations, read_iteration
    from soup_cli.utils.loop_state import write_state

    write_state(_state(), "state.json")
    posts: list = []
    ls._TRAIN_RUNNER = runner
    ls._DEPLOY_POSTER = lambda endpoint, name: posts.append((endpoint, name)) or True
    try:
        watch(WatchConfig(
            poll_interval_sec=1.0, max_iterations=1, state_path="state.json",
            iteration_dir="iters", harvest_fn=harvest, train_fn=train,
            gate_fn=ls.gate_against_baseline, deploy_fn=ls.deploy_to_canary,
            cost_fn=lambda state: 0.0,
        ))
    finally:
        ls._TRAIN_RUNNER = None
        ls._DEPLOY_POSTER = None
    ids = list_iterations("iters")
    assert len(ids) == 1, ids
    return read_iteration(ids[0], base_dir="iters"), posts


# --- a, b: the generated config and the rows it keeps ----------------------


def test_generated_dpo_config_sets_val_split_zero_and_validates(cwd):
    from soup_cli.config.schema import SoupConfig

    raw = yaml.safe_load(ls._render_dpo_yaml(_state(), "pairs.jsonl", "out"))
    assert raw["data"]["val_split"] == 0
    assert "eval_steps" not in raw["training"]
    cfg = SoupConfig(**raw)
    assert cfg.data.val_split == 0


def test_schema_default_and_user_configs_keep_0_1(cwd):
    from soup_cli.config.schema import DataConfig, SoupConfig

    assert DataConfig(train="x.jsonl").val_split == 0.1
    raw = {"base": "m", "task": "dpo", "data": {"train": "x.jsonl", "format": "dpo"}}
    assert SoupConfig(**raw).data.val_split == 0.1


def _train_rows(cwd, n: int, val_split) -> int:
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    path = cwd / "p.jsonl"
    path.write_text(
        "".join(json.dumps({"prompt": f"q{i}", "chosen": "a", "rejected": "b"}) + "\n"
                for i in range(n)),
        encoding="utf-8",
    )
    kwargs = {} if val_split is None else {"val_split": val_split}
    return len(load_dataset(DataConfig(train=str(path), format="dpo", **kwargs))["train"])


@pytest.mark.parametrize("n", [1, 2])
def test_generated_config_keeps_every_pair_as_a_training_row(cwd, n):
    from soup_cli.config.schema import DataConfig

    raw = yaml.safe_load(ls._render_dpo_yaml(_state(), "p.jsonl", "out"))
    got = _train_rows(cwd, n, DataConfig(**raw["data"]).val_split)
    assert got == n


def test_control_the_old_default_dropped_the_single_pair(cwd):
    assert _train_rows(cwd, 1, None) == 0
    assert _train_rows(cwd, 2, None) == 1


# --- c, d: zero / one / two pairs through both harvest sources -------------


@pytest.mark.parametrize("source", ["traces", "thumbs"])
def test_zero_pairs_skip_training_and_deployment_with_a_saved_reason(cwd, monkeypatch, source):
    for k, v in _seed(source, 0, cwd).items():
        monkeypatch.setenv(k, v)
    runner = _Runner()
    monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")

    record, posts = _run_loop(runner)

    assert runner.argvs == []
    # The subprocess-not-called assertion above is the evidence; these two are
    # consistency checks (deploy returns early on a SKIPPED gate either way).
    assert posts == []
    assert record.run_id is None and record.shipped is False
    assert record.notes == "no_preference_pairs: 0 valid pairs harvested, training skipped"


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("source", ["traces", "thumbs"])
def test_one_or_two_pairs_reach_training_with_val_split_zero(cwd, monkeypatch, source, n):
    for k, v in _seed(source, n, cwd).items():
        monkeypatch.setenv(k, v)
    runner = _Runner()

    record, _ = _run_loop(runner)

    assert len(runner.argvs) == 1
    assert runner.config()["data"]["val_split"] == 0
    assert record.pairs_harvested == n
    assert record.run_id is not None
    assert record.notes == ""


def test_the_training_note_sits_next_to_the_deploy_note(cwd, monkeypatch):
    """A skip note must not be lost when the deploy stage also reports one."""
    harvest = lambda state: {"pairs_harvested": 0, "pairs_path": None}  # noqa: E731
    deploy = lambda state, ctx: {"deployed": False, "notes": "deploy-says-hi"}  # noqa: E731
    from soup_cli.utils.loop_daemon import WatchConfig, watch
    from soup_cli.utils.loop_iteration import list_iterations, read_iteration
    from soup_cli.utils.loop_state import write_state

    write_state(_state(), "state.json")
    watch(WatchConfig(
        poll_interval_sec=1.0, max_iterations=1, state_path="state.json",
        iteration_dir="iters", harvest_fn=harvest, train_fn=ls.train_dpo_from_pairs,
        deploy_fn=deploy, cost_fn=lambda s: 0.0,
    ))
    notes = read_iteration(list_iterations("iters")[0], base_dir="iters").notes
    assert notes == (
        "no_preference_pairs: 0 valid pairs harvested, training skipped | deploy-says-hi"
    )


# --- e: a missing pairs file ------------------------------------------------


_ODD = "gone dir/it's pairs.jsonl"  # quote_path output differs from the raw path


def test_a_missing_pairs_file_has_its_own_diagnostic(cwd):
    runner = _Runner()
    ls._TRAIN_RUNNER = runner
    try:
        out = ls.train_dpo_from_pairs(_state(), {"pairs_harvested": 3, "pairs_path": _ODD})
    finally:
        ls._TRAIN_RUNNER = None
    assert runner.argvs == []
    assert out["skipped"] is True and out["run_id"] is None
    assert out["notes"] == (
        f"pairs_file_missing: {quote_path(_ODD)} not found, training skipped"
    )


def test_a_missing_pairs_file_is_saved_in_the_record(cwd):
    harvest = lambda state: {"pairs_harvested": 3, "pairs_path": _ODD}  # noqa: E731
    runner = _Runner()
    record, posts = _run_loop(runner, harvest=harvest)
    assert runner.argvs == [] and posts == []
    assert record.notes == f"pairs_file_missing: {quote_path(_ODD)} not found, training skipped"


# --- f: existing subprocess failures keep their meaning ---------------------


def test_a_failing_train_subprocess_keeps_its_return_shape(cwd, monkeypatch, caplog):
    for k, v in _seed("traces", 1, cwd).items():
        monkeypatch.setenv(k, v)
    harvested = ls.harvest_from_traces(_state())
    runner = _Runner(returncode=1)
    ls._TRAIN_RUNNER = runner
    try:
        with caplog.at_level("WARNING"):
            out = ls.train_dpo_from_pairs(_state(), harvested)
    finally:
        ls._TRAIN_RUNNER = None
    assert len(runner.argvs) == 1
    assert out["run_id"] is None and out["skipped"] is True and out["adapter_path"] is None
    assert "no_preference_pairs" not in str(out.get("notes", ""))
    assert "soup train exited rc=1" in caplog.text


def test_a_raising_train_subprocess_keeps_its_return_shape(cwd, monkeypatch):
    for k, v in _seed("traces", 1, cwd).items():
        monkeypatch.setenv(k, v)
    harvested = ls.harvest_from_traces(_state())

    def boom(*a, **k):
        raise OSError("nope")

    ls._TRAIN_RUNNER = boom
    try:
        out = ls.train_dpo_from_pairs(_state(), harvested)
    finally:
        ls._TRAIN_RUNNER = None
    assert out["run_id"] is None and out["skipped"] is True and out["adapter_path"] is None
    assert "no_preference_pairs" not in str(out.get("notes", ""))


# --- guard edges ------------------------------------------------------------


def _train_with(ctx):
    runner = _Runner()
    ls._TRAIN_RUNNER = runner
    try:
        return ls.train_dpo_from_pairs(_state(), ctx), runner
    finally:
        ls._TRAIN_RUNNER = None


def test_a_zero_count_skips_even_when_a_pairs_file_exists(cwd):
    (cwd / "present.jsonl").write_text("{}\n", encoding="utf-8")
    out, runner = _train_with({"pairs_harvested": 0, "pairs_path": "present.jsonl"})
    assert runner.argvs == []
    assert out["notes"] == "no_preference_pairs: 0 valid pairs harvested, training skipped"


@pytest.mark.parametrize("path", [None, ""])
def test_pairs_but_no_pairs_path_is_a_missing_file_not_an_empty_harvest(cwd, path):
    out, runner = _train_with({"pairs_harvested": 3, "pairs_path": path})
    assert runner.argvs == []
    assert out["notes"] == "pairs_file_missing: no pairs path recorded, training skipped"


@pytest.mark.parametrize("n", [1, 2])
def test_rows_kept_from_the_pairs_file_the_harvest_wrote(cwd, monkeypatch, n):
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    for k, v in _seed("traces", n, cwd).items():
        monkeypatch.setenv(k, v)
    harvested = ls.harvest_from_traces(_state())
    raw = yaml.safe_load(ls._render_dpo_yaml(_state(), harvested["pairs_path"], "out"))
    rows = load_dataset(DataConfig(**raw["data"]))["train"]
    assert len(rows) == n


def test_a_very_long_missing_path_still_saves_a_valid_record(cwd):
    long_path = "d/" * 3000 + "pairs.jsonl"
    harvest = lambda state: {"pairs_harvested": 2, "pairs_path": long_path}  # noqa: E731
    record, _ = _run_loop(_Runner(), harvest=harvest)
    assert "pairs_file_missing" in record.notes and len(record.notes) <= 4096
